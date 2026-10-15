"""Synthetic-only completed-frame alignment, gap and formula checks."""

import numpy as np
import pandas as pd
import pytest

from app.research.learned_signal import FEATURE_NAMES as BASE_NAMES, causal_inputs
from app.research.multiframe_signal import (
    EXTRA_FEATURE_FORMULAS,
    FEATURE_FORMULAS,
    FEATURE_NAMES,
    FRAME_MINUTES,
    causal_multiframe_inputs,
)


@pytest.fixture(scope="module")
def minute_history():
    n = 14 * 1440
    index = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    phase = np.arange(n)
    close = 1000 + 3 * np.sin(phase / 67) + 0.2 * np.sin(phase / 11) + phase / 5000
    opening = np.r_[close[0] - 0.05, close[:-1]]
    frame = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 0.15,
        "low": np.minimum(opening, close) - 0.15, "close": close,
    }, index=index)
    # Observed events make the unchanged base age input known, without outcomes.
    for i in range(150, n, 1440):
        frame.loc[index[i], "high"] += 12
    return frame


@pytest.fixture(scope="module")
def original_inputs(minute_history):
    return {direction: causal_multiframe_inputs(minute_history, direction)
            for direction in ("boom", "crash")}


def _bar(frame, closed, minutes):
    interval = pd.Timedelta(minutes=minutes)
    rows = frame.loc[(frame.index >= closed - interval) & (frame.index < closed)]
    assert len(rows) == minutes and rows.notna().all().all()
    return {"open": rows.open.iloc[0], "high": rows.high.max(),
            "low": rows.low.min(), "close": rows.close.iloc[-1]}


def _true_ranges(frame, closed, minutes, count):
    interval = pd.Timedelta(minutes=minutes)
    result = []
    for i in range(count):
        current = _bar(frame, closed - i * interval, minutes)
        prior = _bar(frame, closed - (i + 1) * interval, minutes)
        result.append(max(current["high"] - current["low"],
                          abs(current["high"] - prior["close"]),
                          abs(current["low"] - prior["close"])))
    return result


def _expected(frame, issue, direction):
    side = 1 if direction == "boom" else -1
    result = {}
    bars = {}
    for label, minutes in FRAME_MINUTES.items():
        closed = issue.floor(f"{minutes}min")
        interval = pd.Timedelta(minutes=minutes)
        current = _bar(frame, closed, minutes)
        bars[label] = current
        atr = np.mean(_true_ranges(frame, closed, minutes, 14))
        ranges = current["high"] - current["low"]
        aligned_body = side * (current["close"] - current["open"]) / atr
        prior_3 = _bar(frame, closed - 3 * interval, minutes)
        aligned_return = side * (current["close"] - prior_3["close"]) / atr
        position = ((current["close"] - current["low"]) if side == 1 else
                    (current["high"] - current["close"])) / ranges
        if label in ("h4", "m15"):
            result[f"{label}_aligned_return_3_atr"] = aligned_return
            result[f"{label}_range_atr"] = ranges / atr
            result[f"{label}_favorable_close_position"] = position
        if label in ("h1", "m15", "m1"):
            result[f"{label}_aligned_body_atr"] = aligned_body
            result[f"{label}_range_atr"] = ranges / atr
        if label == "h1":
            result["h1_favorable_close_position"] = position
        if label in ("h4", "m15", "m1"):
            count = 12 if label == "h4" else 5
            preceding = [_bar(frame, closed - j * interval, minutes) for j in range(1, count + 1)]
            extreme = (max(row["high"] for row in preceding) if side == 1 else
                       min(row["low"] for row in preceding))
            result[f"{label}_aligned_distance_prior_favorable_{count}_atr"] = side * (current["close"] - extreme) / atr
        if label in ("m15", "m1"):
            ranges_30 = _true_ranges(frame, closed, minutes, 30)
            result[f"{label}_atr_5_over_30"] = np.mean(ranges_30[:5]) / np.mean(ranges_30)
            prior_2 = _bar(frame, closed - 2 * interval, minutes)
            gap = (current["low"] - prior_2["high"] if side == 1 else
                   prior_2["low"] - current["high"])
            result[f"{label}_favorable_fvg_atr"] = max(0, gap) / atr
        if label == "m1":
            result["m1_aligned_return_3_atr"] = aligned_return
            result["m1_body_over_range"] = abs(current["close"] - current["open"]) / ranges
            preceding = [_bar(frame, closed - j * interval, minutes) for j in range(1, 21)]
            median_body = np.median([abs(row["close"] - row["open"]) for row in preceding])
            result["m1_body_over_prior_median_20"] = abs(current["close"] - current["open"]) / median_body
    current_m5 = _bar(frame, issue, 5)
    atr_m5 = np.mean(_true_ranges(frame, issue, 5, 14))
    h1 = bars["h1"]
    favorable = h1["high"] if side == 1 else h1["low"]
    adverse = h1["low"] if side == 1 else h1["high"]
    result["h1_aligned_distance_favorable_boundary_m5_atr"] = side * (current_m5["close"] - favorable) / atr_m5
    result["h1_aligned_distance_adverse_boundary_m5_atr"] = side * (current_m5["close"] - adverse) / atr_m5
    recent = [_bar(frame, issue - pd.Timedelta(minutes=5 * j), 5) for j in range(3)]
    swept = (min(row["low"] for row in recent) < h1["low"] if side == 1 else
             max(row["high"] for row in recent) > h1["high"])
    result["h1_recent_m5_sweep_reclaim"] = float(swept and h1["low"] < current_m5["close"] < h1["high"])
    return result


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_schema_has_exactly_base19_plus_declared25(minute_history, original_inputs, direction):
    inputs, names = original_inputs[direction]
    base, base_names = causal_inputs(minute_history, direction)
    assert len(names) == 44 and len(EXTRA_FEATURE_FORMULAS) == 25
    assert tuple(names) == FEATURE_NAMES == tuple(FEATURE_FORMULAS)
    assert names[:19] == base_names == list(BASE_NAMES)
    assert names[19:] == list(EXTRA_FEATURE_FORMULAS)
    assert len(set(names)) == 44
    pd.testing.assert_index_equal(inputs.index, base.index)
    pd.testing.assert_frame_equal(inputs[base_names], base[base_names])
    assert inputs.feature_valid.any()
    assert np.isfinite(inputs.loc[inputs.feature_valid, names]).all().all()
    assert not (inputs.feature_valid & ~base.feature_valid).any()


@pytest.mark.parametrize("direction", ["boom", "crash"])
@pytest.mark.parametrize("cutoff", ["2026-01-07 11:55", "2026-01-07 12:00", "2026-01-07 12:05", "2026-01-07 13:00"])
def test_prefix_and_future_mutation_match_across_h4_and_h1_closes(minute_history, original_inputs, direction, cutoff):
    cutoff = pd.Timestamp(cutoff, tz="UTC")
    original, names = original_inputs[direction]
    replaced = minute_history.copy()
    replaced.loc[cutoff:, :] *= 5
    altered, altered_names = causal_multiframe_inputs(replaced, direction)
    prefix, prefix_names = causal_multiframe_inputs(minute_history.loc[minute_history.index < cutoff], direction)
    completed = original.index + pd.Timedelta(minutes=5) <= cutoff
    assert names == altered_names == prefix_names
    pd.testing.assert_frame_equal(original.loc[completed], altered.loc[completed])
    pd.testing.assert_frame_equal(original.loc[completed], prefix)


def test_context_uses_latest_closed_frame_including_exact_boundaries(original_inputs):
    inputs, _ = original_inputs["boom"]
    for text in ("2026-01-07 12:00", "2026-01-07 12:05", "2026-01-07 12:15", "2026-01-07 13:00", "2026-01-07 16:00"):
        issue = pd.Timestamp(text, tz="UTC")
        row = inputs.loc[issue - pd.Timedelta(minutes=5)]
        for label, minutes in FRAME_MINUTES.items():
            assert row[f"{label}_closed_at"] == issue.floor(f"{minutes}min")
            assert row[f"{label}_closed_at"] <= issue
            assert row[f"{label}_row_valid"]


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_partial_h4_stays_unavailable_and_incomplete_final_m5_is_invalid(minute_history, original_inputs, direction):
    cutoff = pd.Timestamp("2026-01-07 13:03", tz="UTC")
    prefix, _ = causal_multiframe_inputs(minute_history.loc[minute_history.index < cutoff], direction)
    assert prefix.index[-1] == pd.Timestamp("2026-01-07 13:00", tz="UTC")
    assert not prefix.feature_valid.iloc[-1]
    assert not prefix.m1_row_valid.iloc[-1]
    assert prefix.m1_closed_at.iloc[-1] == cutoff
    assert prefix.h4_closed_at.iloc[-1] == pd.Timestamp("2026-01-07 12:00", tz="UTC")
    original, _ = original_inputs[direction]
    pd.testing.assert_frame_equal(prefix.iloc[:-1], original.loc[prefix.index[:-1]])


@pytest.mark.parametrize("drop_row", [True, False])
def test_missing_minute_invalidates_latest_rows_and_full_rolling_windows(minute_history, drop_row):
    missing = pd.Timestamp("2026-01-06 09:04", tz="UTC")
    frame = minute_history.copy()
    if drop_row:
        frame = frame.drop(missing)
    else:
        frame.loc[missing, "high"] = np.nan
    inputs, _ = causal_multiframe_inputs(frame, "boom")
    # The missing final M1 cannot be replaced by the preceding valid minute.
    row = inputs.loc[missing.floor("5min")]
    assert row.m1_closed_at == missing + pd.Timedelta(minutes=1)
    assert not row.m1_row_valid and np.isnan(row.m1_aligned_body_atr)
    assert not row.feature_valid
    h1_issue = pd.Timestamp("2026-01-06 10:00", tz="UTC")
    row = inputs.loc[h1_issue - pd.Timedelta(minutes=5)]
    assert row.h1_closed_at == h1_issue and not row.h1_row_valid
    assert np.isnan(row.h1_aligned_body_atr)
    assert np.isnan(row.h1_recent_m5_sweep_reclaim)
    h4_issue = pd.Timestamp("2026-01-06 12:00", tz="UTC")
    row = inputs.loc[h4_issue - pd.Timedelta(minutes=5)]
    assert row.h4_closed_at == h4_issue and not row.h4_row_valid
    assert np.isnan(row.h4_range_atr)
    next_issue = pd.Timestamp("2026-01-06 16:00", tz="UTC")
    row = inputs.loc[next_issue - pd.Timedelta(minutes=5)]
    assert row.h4_row_valid and np.isnan(row.h4_range_atr)
    # TR at the gap and the following H4 row is unknown; fourteen good TRs
    # recover only at the H4 candle closing 2026-01-09 00:00 UTC.
    recovery = pd.Timestamp("2026-01-09 00:00", tz="UTC")
    issues = inputs.index + pd.Timedelta(minutes=5)
    assert not inputs.feature_valid.loc[(issues >= h4_issue) & (issues < recovery)].any()
    assert inputs.feature_valid.loc[recovery - pd.Timedelta(minutes=5)]
    assert (inputs.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_all25_formulas_match_independent_minute_window_calculation(minute_history, original_inputs, direction):
    issue = pd.Timestamp("2026-01-07 12:30", tz="UTC")
    inputs, _ = original_inputs[direction]
    expected = _expected(minute_history, issue, direction)
    assert set(expected) == set(EXTRA_FEATURE_FORMULAS)
    row = inputs.loc[issue - pd.Timedelta(minutes=5)]
    for name in EXTRA_FEATURE_FORMULAS:
        assert row[name] == pytest.approx(expected[name]), name


def _reflected(frame):
    return pd.DataFrame({"open": 4000 - frame.open, "high": 4000 - frame.low,
                         "low": 4000 - frame.high, "close": 4000 - frame.close}, index=frame.index)


def test_all25_extra_inputs_have_boom_crash_reflection_symmetry(minute_history, original_inputs):
    boom, _ = original_inputs["boom"]
    crash, _ = causal_multiframe_inputs(_reflected(minute_history), "crash")
    # The unchanged base BB/price and logATR/price formulas are deliberately
    # excluded: their price denominators are not reflection invariant.
    np.testing.assert_allclose(boom[list(EXTRA_FEATURE_FORMULAS)], crash[list(EXTRA_FEATURE_FORMULAS)],
                               rtol=1e-8, atol=1e-10, equal_nan=True)
    for label in FRAME_MINUTES:
        pd.testing.assert_series_equal(boom[f"{label}_row_valid"], crash[f"{label}_row_valid"])


def test_sweep_reclaim_is_known_true_and_missing_recent_input_remains_unknown(minute_history):
    issue = pd.Timestamp("2026-01-07 12:30", tz="UTC")
    reference = _bar(minute_history, issue.floor("h"), 60)
    frame = minute_history.copy()
    frame.loc[issue - pd.Timedelta(minutes=15), "low"] = reference["low"] - 1
    last = issue - pd.Timedelta(minutes=1)
    frame.loc[last, "close"] = (reference["high"] + reference["low"]) / 2
    frame.loc[last, "high"] = max(frame.loc[last, "open"], frame.loc[last, "close"]) + 0.15
    frame.loc[last, "low"] = min(frame.loc[last, "open"], frame.loc[last, "close"]) - 0.15
    for direction, source in (("boom", frame), ("crash", _reflected(frame))):
        inputs, _ = causal_multiframe_inputs(source, direction)
        assert inputs.loc[issue - pd.Timedelta(minutes=5), "h1_recent_m5_sweep_reclaim"] == 1
        incomplete = source.drop(issue - pd.Timedelta(minutes=8))
        missing_inputs, _ = causal_multiframe_inputs(incomplete, direction)
        assert np.isnan(missing_inputs.loc[issue - pd.Timedelta(minutes=5), "h1_recent_m5_sweep_reclaim"])


def test_positive_fvg_on_m15_and_m1_is_measured_at_its_own_atr(minute_history):
    issue = pd.Timestamp("2026-01-07 12:30", tz="UTC")
    frame = minute_history.copy()
    prior = _bar(frame, issue - pd.Timedelta(minutes=30), 15)
    target_minutes = frame.index[(frame.index >= issue - pd.Timedelta(minutes=15)) & (frame.index < issue)]
    # A high completed M15 with a rising final three-minute sequence creates
    # positive three-candle gaps on both frames, without any later data.
    for i, stamp in enumerate(target_minutes):
        price = prior["high"] + 2 + i * 0.2
        frame.loc[stamp, :] = [price + 0.02, price + 0.12, price, price + 0.08]
    for direction, source in (("boom", frame), ("crash", _reflected(frame))):
        inputs, _ = causal_multiframe_inputs(source, direction)
        expected = _expected(source, issue, direction)
        row = inputs.loc[issue - pd.Timedelta(minutes=5)]
        for label in ("m15", "m1"):
            name = f"{label}_favorable_fvg_atr"
            assert row[name] > 0 and row[name] == pytest.approx(expected[name])


def test_invalid_direction_and_non_minute_timestamps_are_refused(minute_history):
    with pytest.raises(ValueError, match="direction"):
        causal_multiframe_inputs(minute_history, "both")
    malformed = minute_history.copy()
    malformed.index += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="minute openings"):
        causal_multiframe_inputs(malformed, "boom")
