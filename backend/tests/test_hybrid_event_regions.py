"""Synthetic raw-event semantics, gap resets, causality and isolation."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.research.jump_multiframe import causal_jump_multiframe_inputs
from app.research.jump_representation import jump_representation
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.tick_tail import FixedTailDetector
from scripts.hybrid_event_regions import AGE, named_signals, raw_event_clock_inputs


@pytest.fixture(scope="module")
def ticks():
    n = 6 * 86400
    phase = np.arange(n)
    increments = 1e-6 * (.7 * np.sin(phase / 41) + .4 * np.cos(phase / 83)) - 2e-7
    increments[5000::10000] += .003
    increments[0] = 0
    return pd.DataFrame({"quote": 1000 * np.exp(np.cumsum(increments))},
                        index=pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC"))


def prepare(ticks, direction):
    if direction == "crash":
        ticks = pd.DataFrame({"quote": 1_000_000 / ticks.quote}, index=ticks.index)
    _, bars = jump_representation(ticks, FixedTailDetector(1 if direction == "boom" else -1, 1e-5))
    return causal_jump_multiframe_inputs(bars, direction)


@pytest.fixture(scope="module")
def original(ticks):
    return {d: prepare(ticks, d) for d in ("boom", "crash")}


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_observed_age_enables_distinct_schema_without_changing43_or_raw(original, direction):
    before = original[direction]
    untouched = before.transformed.copy(deep=True)
    actual = raw_event_clock_inputs(before, direction)
    assert not before.transformed.feature_valid.any()
    assert actual.availability.common_feature_valid.any()
    pd.testing.assert_frame_equal(before.transformed, untouched)
    pd.testing.assert_frame_equal(actual.raw, before.raw)
    names = [n for n in FEATURE_NAMES if n != AGE]
    pd.testing.assert_frame_equal(actual.transformed[names], before.transformed[names])
    pd.testing.assert_series_equal(actual.transformed[AGE], before.raw[AGE])
    pd.testing.assert_series_equal(actual.availability.common_feature_valid,
                                   actual.raw.feature_valid & actual.transformed.feature_valid, check_names=False)
    pd.testing.assert_series_equal(actual.availability.raw_execution_atr, before.availability.raw_execution_atr)
    pd.testing.assert_series_equal(actual.availability.transformed_feature_atr, before.availability.transformed_feature_atr)
    assert actual.transformed.loc[actual.transformed.feature_valid, FEATURE_NAMES].notna().all().all()


def test_unknown_original_age_remains_unknown_and_invalid(original):
    x = original["boom"]
    x = type(x)(x.raw.copy(), x.transformed.copy(), x.availability.copy())
    x.raw[AGE] = np.nan
    x.raw["feature_valid"] = False
    x.availability["raw_feature_valid"] = False
    x.availability["common_feature_valid"] = False
    actual = raw_event_clock_inputs(x, "boom")
    assert actual.transformed[AGE].isna().all()
    assert not actual.transformed.feature_valid.any()
    assert not actual.availability.common_feature_valid.any()


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_at_or_after_issue_ticks_cannot_change_any_prior_hybrid_input(ticks, original, direction):
    issue = pd.Timestamp("2026-01-04T12:00Z")
    altered = ticks.copy()
    altered.loc[altered.index >= issue, "quote"] *= 1.4
    changed = raw_event_clock_inputs(prepare(altered, direction), direction)
    expected = raw_event_clock_inputs(original[direction], direction)
    prior = expected.raw.index + pd.Timedelta(minutes=5) <= issue
    for field in ("raw", "transformed", "availability"):
        pd.testing.assert_frame_equal(getattr(changed, field).loc[prior], getattr(expected, field).loc[prior])


def test_gap_does_not_reuse_original_event_age_or_rolling_context(ticks):
    gap = pd.Timestamp("2026-01-04T12:03:30Z")
    actual = raw_event_clock_inputs(prepare(ticks.drop(gap), "boom"), "boom")
    row = gap.floor("5min")
    assert pd.isna(actual.transformed.loc[row, AGE])
    assert pd.isna(actual.raw.loc[row, AGE])
    assert not actual.availability.loc[row, "common_feature_valid"]
    after = (actual.raw.index >= row) & (actual.raw.index < row + pd.Timedelta(hours=48))
    assert not actual.availability.loc[after, "common_feature_valid"].any()


@pytest.mark.parametrize("change", ["index", "run", "direction"])
def test_inconsistent_lineage_is_refused(original, change):
    x = original["boom"]
    x = type(x)(x.raw.copy(), x.transformed.copy(), x.availability.copy())
    if change == "index": x.transformed.index += pd.Timedelta(minutes=5)
    if change == "run": x.availability.loc[x.availability.run_id.notna(), "run_id"] += 1
    with pytest.raises(ValueError): raw_event_clock_inputs(x, "other" if change == "direction" else "boom")


def test_explicit_arm_names_preserve_values_and_leave_original_issuance_unchanged():
    source = {n: pd.DataFrame({"variant": [n + "_SPIKE"], "score": [.2]})
              for n in ("CLOCK", "RAW44", "TRANSFORMED44")}
    result = named_signals(source)
    assert set(result) == {"CLOCK", "RAW44", "HYBRID44"}
    assert result["HYBRID44"].variant.iloc[0] == "HYBRID44_SPIKE"
    assert source["TRANSFORMED44"].variant.iloc[0] == "TRANSFORMED44_SPIKE"
    assert result["HYBRID44"].score.iloc[0] == .2
    with pytest.raises(ValueError): named_signals({})


def test_live_env_refuses_hybrid_preparation(monkeypatch, original):
    monkeypatch.setenv("OPENED_TRADES", "true")
    with pytest.raises(RuntimeError): raw_event_clock_inputs(original["boom"], "boom")
