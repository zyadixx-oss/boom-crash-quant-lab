"""Synthetic-only paired nonlinear region population, causality and identity tests."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
spec = importlib.util.spec_from_file_location("nonlinear_region_learning_under_test", Path(__file__).with_name("learning.py"))
learning = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = learning
spec.loader.exec_module(learning)

from app.research.jump_learning import spike_clock_signals
from app.research.jump_multiframe import PairedMultiframeInputs
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.nonlinear_signal import DEFAULT_PARAMETERS, predict_histogram_boost
from scripts.region_reward_learning import region_training_labels, fingerprint


@pytest.fixture(scope="module")
def inputs():
    index = pd.date_range("2025-12-31T23:55Z", periods=7500, freq="5min").as_unit("ns")
    row = np.arange(len(index))[:, None]
    col = np.arange(44)[None, :]
    x = ((row // (6 * (1 + col % 3))) % 2).astype(float)
    raw = pd.DataFrame(x, index=index, columns=FEATURE_NAMES)
    hybrid = pd.DataFrame(x * np.linspace(2, 4, 44) + 3, index=index, columns=FEATURE_NAMES)
    for frame, atr, close in ((raw, 2., 100.), (hybrid, .02, .8)):
        frame["feature_valid"] = True
        frame["atr"] = atr
        frame["close"] = close
    available = pd.DataFrame({"raw_feature_valid": True, "transformed_feature_valid": True,
        "common_feature_valid": True, "raw_execution_atr": 2., "transformed_feature_atr": .02,
        "run_id": pd.array(np.zeros(len(index)), dtype="Int64")}, index=index)
    return PairedMultiframeInputs(raw, hybrid, available)


def population(inputs):
    start = pd.Timestamp("2026-01-01T00:00Z")
    end = start + pd.Timedelta(minutes=1004 * 30 + 31)
    clock = spike_clock_signals(inputs, "BOOM600", start, end).signals
    issues = clock.index
    x = inputs.raw.loc[issues - pd.Timedelta(minutes=5), FEATURE_NAMES[0]].to_numpy()
    events = pd.DataFrame({"signal_time": issues, "entry_time": issues + pd.Timedelta(seconds=61),
                          "status": "completed", "censored": False, "net_R": .1 + .3 * x})
    for i, status in enumerate(("expired", "invalidated", "waiting_gap", "entry_gap", "overlap_skipped")):
        events.loc[i, ["status", "entry_time", "net_R", "censored"]] = [status, pd.NaT, np.nan, status.endswith("gap")]
    return clock, events, region_training_labels(clock, events), start, end


@pytest.fixture(scope="module")
def fit(inputs):
    _, _, labels, start, end = population(inputs)
    return learning.fit_region_boost(inputs, labels, "BOOM600", start, end)


def test_completed_population_and_exact_fixed_algorithm(inputs, fit):
    _, _, labels, _, _ = population(inputs)
    metadata = fit.metadata
    assert metadata["completed"] == 1000 and metadata["opportunities"] == 1005
    assert (metadata["known_nonfill"], metadata["unknown_path"], metadata["exposure_skipped"]) == (2, 2, 1)
    assert metadata["parameters"] == DEFAULT_PARAMETERS == learning.FIXED_PARAMETERS
    assert metadata["algorithm"] == learning.ALGORITHM
    assert metadata["standardization"] is False
    assert metadata["training_is_independent_trade_sample"] is False
    assert metadata["label_selection_is_conditional_on_CLOCK_fill"] is True
    h = hashlib.sha256(b"paired-region-filled-target-v1\0")
    h.update(np.asarray(labels.index[labels.completed].asi8, dtype=">i8").tobytes())
    h.update(np.asarray(labels.loc[labels.completed, "net_R"], dtype=">f8").tobytes())
    assert metadata["target_sha256"] == h.hexdigest()
    for arm, frame in zip(learning.ARMS, (inputs.raw, inputs.transformed), strict=True):
        model = fit.models[arm]
        assert model["parameters"] == model["defaults"] == DEFAULT_PARAMETERS
        assert model["fit_rows"] == 1000 and len(model["trees"]) == 100
        selected = frame.loc[labels.index[labels.completed] - pd.Timedelta(minutes=5), list(FEATURE_NAMES)]
        predictions = predict_histogram_boost(selected, model, FEATURE_NAMES)
        assert model["training_score_quantile"] == np.quantile(predictions, .75, method="linear")
        assert model["threshold"] == max(0., model["training_score_quantile"])
        m = hashlib.sha256(metadata["target_sha256"].encode())
        m.update(np.asarray(selected.to_numpy(float), dtype=">f8").tobytes())
        assert metadata["matrix_sha256"][arm] == m.hexdigest()
    assert fit.models[learning.ARMS[0]]["initial_prediction"] == fit.models[learning.ARMS[1]]["initial_prediction"]
    assert fit.models[learning.ARMS[0]]["bin_boundaries"] != fit.models[learning.ARMS[1]]["bin_boundaries"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "future", "planned", "category", "nonbool", "unknown_zero", "nonfill_zero", "infinite", "bool_reward"])
def test_invalid_label_population_rejected(inputs, change):
    _, _, labels, start, end = population(inputs)
    if change == "missing": labels = labels.iloc[1:]
    elif change == "duplicate": labels.index = labels.index.where(np.arange(len(labels)) != 1, labels.index[0])
    elif change == "future": labels.index = labels.index + pd.Timedelta(minutes=30)
    elif change == "planned": labels["planned_end"] += pd.Timedelta(seconds=1)
    elif change == "category": labels.iloc[0, labels.columns.get_loc("completed")] = True
    elif change == "nonbool": labels["completed"] = labels.completed.astype(int)
    elif change == "unknown_zero": labels.iloc[2, labels.columns.get_loc("net_R")] = 0.
    elif change == "nonfill_zero": labels.iloc[0, labels.columns.get_loc("net_R")] = 0.
    elif change == "infinite": labels.iloc[5, labels.columns.get_loc("net_R")] = np.inf
    else: labels["net_R"] = True
    with pytest.raises(ValueError):
        learning.fit_region_boost(inputs, labels, "BOOM600", start, end)


def test_insufficient_completed_cannot_borrow_nonfills(inputs):
    clock, events, _, start, end = population(inputs)
    events.loc[5, ["status", "entry_time", "net_R"]] = ["expired", pd.NaT, np.nan]
    labels = region_training_labels(clock, events)
    with pytest.raises(learning.InsufficientRegionTraining, match="observed 999"):
        learning.fit_region_boost(inputs, labels, "BOOM600", start, end)


def test_future_features_do_not_change_fit_or_prior_issuance(inputs, fit):
    _, _, labels, start, end = population(inputs)
    changed = replace(inputs, raw=inputs.raw.copy(), transformed=inputs.transformed.copy())
    future = changed.raw.index + pd.Timedelta(minutes=5) >= end
    for frame in (changed.raw, changed.transformed):
        frame.loc[future, list(FEATURE_NAMES)] = np.nan
    assert learning.fit_region_boost(changed, labels, "BOOM600", start, end) == fit
    eval_end = end + pd.Timedelta(hours=8)
    original = learning.issue_region_boost(inputs, fit, "BOOM600", end, eval_end)
    changed = replace(inputs, raw=inputs.raw.copy(), transformed=inputs.transformed.copy())
    boundary = end + pd.Timedelta(hours=4)
    for frame in (changed.raw, changed.transformed):
        frame.loc[frame.index + pd.Timedelta(minutes=5) >= boundary, list(FEATURE_NAMES)] += 100
    actual = learning.issue_region_boost(changed, fit, "BOOM600", end, eval_end)
    for arm in learning.ARMS:
        pd.testing.assert_frame_equal(original[arm].loc[original[arm].signal_time < boundary],
                                      actual[arm].loc[actual[arm].signal_time < boundary])
        assert original[arm].atr.eq(2.).all()
        assert original[arm].signal_close.eq(100.).all()
        assert original[arm].side.eq(1).all()
        assert original[arm].signal_time.dt.minute.isin([0, 30]).all()


@pytest.mark.parametrize("change", ["model", "metadata", "side", "early", "missing_arm", "algorithm", "parameters", "scaler"])
def test_frozen_identity_and_chronology(inputs, fit, change):
    _, _, _, start, end = population(inputs)
    changed = deepcopy(fit)
    if change == "model": changed.models[learning.ARMS[0]]["initial_prediction"] += .1
    elif change == "metadata": changed.metadata["completed"] += 1
    elif change == "missing_arm": del changed.models[learning.ARMS[1]]
    elif change == "algorithm": changed.metadata["algorithm"] = "other"
    elif change == "parameters": changed.metadata["parameters"]["n_trees"] = 99
    elif change == "scaler": changed.metadata["standardization"] = True
    if change in ("algorithm", "parameters", "scaler"):
        changed.metadata["metadata_sha256"] = fingerprint({k: v for k, v in changed.metadata.items() if k != "metadata_sha256"})
    with pytest.raises(ValueError):
        learning.issue_region_boost(inputs, changed, "CRASH600" if change == "side" else "BOOM600",
                                    start if change == "early" else end, end + pd.Timedelta(hours=8))


def test_changed_inherited_defaults_refused(inputs, fit, monkeypatch):
    _, _, labels, start, end = population(inputs)
    monkeypatch.setitem(learning.DEFAULT_PARAMETERS, "n_trees", 99)
    with pytest.raises(ValueError, match="defaults changed"):
        learning.fit_region_boost(inputs, labels, "BOOM600", start, end)
    with pytest.raises(ValueError, match="defaults changed"):
        learning.issue_region_boost(inputs, fit, "BOOM600", end, end + pd.Timedelta(hours=8))


def test_inputs_labels_and_model_are_not_mutated(inputs, fit):
    _, _, labels, _, end = population(inputs)
    before = deepcopy(fit)
    raw, hybrid, available = inputs.raw.copy(deep=True), inputs.transformed.copy(deep=True), inputs.availability.copy(deep=True)
    label_copy = labels.copy(deep=True)
    learning.issue_region_boost(inputs, fit, "BOOM600", end, end + pd.Timedelta(hours=8))
    assert before == fit
    pd.testing.assert_frame_equal(inputs.raw, raw)
    pd.testing.assert_frame_equal(inputs.transformed, hybrid)
    pd.testing.assert_frame_equal(inputs.availability, available)
    pd.testing.assert_frame_equal(labels, label_copy)


@pytest.mark.parametrize("flag", ["LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"])
def test_live_guard_all_entry_points(inputs, fit, monkeypatch, flag):
    clock, events, labels, start, end = population(inputs)
    monkeypatch.setenv(flag, "true")
    with pytest.raises(RuntimeError): learning.region_training_labels(clock, events)
    with pytest.raises(RuntimeError): learning.fit_region_boost(inputs, labels, "BOOM600", start, end)
    with pytest.raises(RuntimeError): learning.issue_region_boost(inputs, fit, "BOOM600", end, end + pd.Timedelta(hours=8))
