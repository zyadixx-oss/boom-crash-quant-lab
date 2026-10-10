"""Conditional region labels, paired targets, causality and frozen issuance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.research.jump_learning import spike_clock_signals
from app.research.jump_multiframe import PairedMultiframeInputs
from app.research.multiframe_signal import FEATURE_NAMES
from scripts.region_reward_learning import (InsufficientRegionTraining, RegionRewardFit,
    fit_region_reward, issue_region_reward, region_training_labels)


@pytest.fixture(scope="module")
def inputs():
    index = pd.date_range("2025-12-31T23:55Z", periods=7500, freq="5min").as_unit("ns")
    x = np.random.default_rng(20261008).normal(size=(len(index), 44))
    raw = pd.DataFrame(x, index=index, columns=FEATURE_NAMES)
    hybrid = pd.DataFrame(x * np.linspace(2, 4, 44) + 3, index=index, columns=FEATURE_NAMES)
    for frame, atr in ((raw, 2.), (hybrid, .02)):
        frame["feature_valid"] = True;frame["atr"] = atr;frame["close"] = 100.
    available = pd.DataFrame({"raw_feature_valid": True, "transformed_feature_valid": True,
        "common_feature_valid": True, "raw_execution_atr": 2., "transformed_feature_atr": .02,
        "run_id": pd.array(np.zeros(len(index)), dtype="Int64")}, index=index)
    return PairedMultiframeInputs(raw, hybrid, available)


def fixture(inputs):
    start = pd.Timestamp("2026-01-01T00:00Z");end = start + pd.Timedelta(minutes=1004*30 + 31)
    clock = spike_clock_signals(inputs, "BOOM600", start, end).signals
    issues = clock.index
    events = pd.DataFrame({"signal_time": issues, "entry_time": issues + pd.Timedelta(seconds=61),
        "status": "completed", "censored": False,
        "net_R": .2 + .02 * inputs.raw.loc[issues-pd.Timedelta(minutes=5), FEATURE_NAMES[0]].to_numpy()})
    for i, state in enumerate(("expired", "invalidated", "waiting_gap", "entry_gap", "overlap_skipped")):
        events.loc[i, ["status", "entry_time", "net_R", "censored"]] = [state, pd.NaT, np.nan, state.endswith("gap")]
    return clock, events, start, end


def test_known_nonfills_skips_and_unknowns_stay_distinct_without_zero_labels(inputs):
    clock, events, _, _ = fixture(inputs);labels = region_training_labels(clock, events)
    assert labels.completed.sum() == 1000 and labels.known_nonfill.sum() == 2
    assert labels.unknown_path.sum() == 2 and labels.exposure_skipped.sum() == 1
    assert labels.net_R.iloc[:5].isna().all()
    assert (labels.planned_end == labels.index + pd.Timedelta(minutes=30, seconds=1)).all()


def test_exact_completed_region_population_fits_shared_targets_separate_scalers(inputs):
    clock, events, start, end = fixture(inputs);labels = region_training_labels(clock, events)
    fit = fit_region_reward(inputs, labels, "BOOM600", start, end)
    assert fit.metadata["completed"] == 1000 and fit.metadata["known_nonfill"] == 2
    assert fit.metadata["unknown_path"] == 2 and fit.metadata["exposure_skipped"] == 1
    assert fit.metadata["training_is_independent_trade_sample"] is False
    assert fit.models["RAW_REGION"]["means"] != fit.models["HYBRID_REGION"]["means"]
    assert fit.models["RAW_REGION"]["intercept"] == fit.models["HYBRID_REGION"]["intercept"]
    assert fit.models["RAW_REGION"]["fit_rows"] == 1000


def test_fewer_than1000_completed_paths_cannot_use_nonfills_to_fit(inputs):
    clock, events, start, end = fixture(inputs)
    events.loc[5, ["status", "entry_time", "net_R"]] = ["expired", pd.NaT, np.nan]
    with pytest.raises(InsufficientRegionTraining, match="observed 999"):
        fit_region_reward(inputs, region_training_labels(clock, events), "BOOM600", start, end)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown_zero", "nonfill_zero", "status", "censor", "bool_reward"])
def test_invalid_actual_training_dispositions_are_refused(inputs, change):
    clock, events, _, _ = fixture(inputs)
    if change == "missing": events = events.iloc[:-1]
    elif change == "duplicate": events.loc[1, "signal_time"] = events.loc[0, "signal_time"]
    elif change == "unknown_zero": events.loc[2, "net_R"] = 0.
    elif change == "nonfill_zero": events.loc[0, "net_R"] = 0.
    elif change == "status": events.loc[0, "status"] = "purged"
    elif change == "censor": events.loc[0, "censored"] = True
    else: events["net_R"] = True
    with pytest.raises(ValueError): region_training_labels(clock, events)


@pytest.mark.parametrize("change", ["missing", "future", "planned", "overlap_category", "nonbool", "unknown_zero", "bool_reward"])
def test_training_label_population_and_horizon_cannot_change(inputs, change):
    clock, events, start, end = fixture(inputs);labels = region_training_labels(clock, events)
    if change == "missing": labels = labels.iloc[1:]
    elif change == "future": labels.index = labels.index + pd.Timedelta(minutes=30)
    elif change == "planned": labels["planned_end"] += pd.Timedelta(seconds=1)
    elif change == "overlap_category": labels.iloc[0, labels.columns.get_loc("completed")] = True
    elif change == "nonbool": labels["completed"] = labels.completed.astype(int)
    elif change == "unknown_zero": labels.iloc[2, labels.columns.get_loc("net_R")] = 0.
    else: labels["net_R"] = True
    with pytest.raises(ValueError): fit_region_reward(inputs, labels, "BOOM600", start, end)


def test_future_features_cannot_change_training_weights_and_prior_eval_scores(inputs):
    clock, events, start, end = fixture(inputs);labels = region_training_labels(clock, events)
    before = fit_region_reward(inputs, labels, "BOOM600", start, end)
    changed = replace(inputs, raw=inputs.raw.copy(), transformed=inputs.transformed.copy())
    after_train = changed.raw.index + pd.Timedelta(minutes=5) >= end
    changed.raw.loc[after_train, list(FEATURE_NAMES)] *= 100
    changed.transformed.loc[after_train, list(FEATURE_NAMES)] *= -100
    assert fit_region_reward(changed, labels, "BOOM600", start, end) == before
    eval_end = end + pd.Timedelta(hours=8)
    original = issue_region_reward(inputs, before, "BOOM600", end, eval_end)
    future_issue = end + pd.Timedelta(hours=4)
    changed = replace(inputs, raw=inputs.raw.copy(), transformed=inputs.transformed.copy())
    for frame in (changed.raw, changed.transformed):
        frame.loc[frame.index+pd.Timedelta(minutes=5) >= future_issue, list(FEATURE_NAMES)] *= 100
    actual = issue_region_reward(changed, before, "BOOM600", end, eval_end)
    for arm in original:
        pd.testing.assert_frame_equal(original[arm].loc[original[arm].signal_time < future_issue], actual[arm].loc[actual[arm].signal_time < future_issue])
        assert original[arm].atr.eq(2).all()


@pytest.mark.parametrize("change", ["model", "metadata", "side", "early", "missing_arm"])
def test_frozen_region_fit_and_chronology_are_enforced(inputs, change):
    clock, events, start, end = fixture(inputs);labels = region_training_labels(clock, events)
    fit = deepcopy(fit_region_reward(inputs, labels, "BOOM600", start, end))
    if change == "model": fit.models["RAW_REGION"]["coefs"][0] += .1
    if change == "metadata": fit.metadata["completed"] += 1
    if change == "missing_arm": del fit.models["HYBRID_REGION"]
    with pytest.raises(ValueError): issue_region_reward(inputs, fit, "CRASH600" if change == "side" else "BOOM600", start if change == "early" else end, end+pd.Timedelta(hours=8))


def test_live_env_refuses_label_preparation(monkeypatch, inputs):
    clock, events, _, _ = fixture(inputs);monkeypatch.setenv("LIVE_ALLOWED", "true")
    with pytest.raises(RuntimeError): region_training_labels(clock, events)
