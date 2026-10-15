"""Adversarial checks of timestamps, causal replay, gaps and research safety."""

import numpy as np
import pandas as pd
import pytest

from app.research.spike_hunter import (
    SAFETY_FLAGS,
    aggregate_complete,
    assert_offline,
    eligible,
    features,
    load_m1,
    outcomes,
    partitions,
    signals,
)


def minute_history(n=3000):
    index = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    phase = np.arange(n)
    close = 100 + 0.5 * np.sin(phase / 15) + 0.1 * np.sin(phase / 73)
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) + 0.2,
            "low": np.minimum(opening, close) - 0.2,
            "close": close,
        },
        index=index,
    )


def csv_rows(frame):
    rows = frame.copy()
    rows.insert(0, "epoch", frame.index.as_unit("ns").asi8 // 1_000_000_000)
    return rows.reset_index(drop=True)


def setup_bars(n=48):
    index = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {"open": 105.0, "high": 110.0, "low": 100.0, "close": 105.0},
        index=index,
    )
    for key in (
        "disp", "mss", "fvg", "atr_compression", "bb_squeeze",
        "candle_compression", "candle_structure", "sr_alignment",
    ):
        frame[key] = False
    frame["atr"] = 10.0
    frame["sweep_atr"] = 10.0
    frame["feature_valid"] = True
    return frame


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_features_and_past_signals_ignore_replaced_future(direction):
    frame = minute_history()
    cutoff = frame.index[1800]
    altered = frame.copy()
    # Replace all future OHLC, including enough bars to change future H1 ranges.
    for key in ("open", "high", "low", "close"):
        altered.loc[cutoff:, key] *= 10
    original_m5 = features(frame, direction)
    altered_m5 = features(altered, direction)
    past = original_m5.index + pd.Timedelta(minutes=5) <= cutoff
    pd.testing.assert_frame_equal(original_m5.loc[past], altered_m5.loc[past])
    before, after = signals(original_m5, direction), signals(altered_m5, direction)
    assert sum(len(v) for v in before.values()) > 0
    for name in before:
        np.testing.assert_array_equal(before[name][past[before[name]]],
                                      after[name][past[after[name]]])


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_completed_prefix_produces_identical_features_and_signals(direction):
    frame = minute_history()
    cutoff = frame.index[1800]
    full = features(frame, direction)
    prefix = features(frame.loc[frame.index < cutoff], direction)
    pd.testing.assert_frame_equal(full.iloc[:len(prefix)], prefix)
    full_signals, prefix_signals = signals(full, direction), signals(prefix, direction)
    for name in full_signals:
        np.testing.assert_array_equal(full_signals[name][full_signals[name] < len(prefix)],
                                      prefix_signals[name])


def test_late_reclaim_cannot_suppress_an_earlier_next_hour_signal():
    frame = setup_bars(36)
    frame.iloc[12:23, frame.columns.get_loc("close")] = 111.0
    frame.iloc[12:23, frame.columns.get_loc("high")] = 112.0
    # Hour 1 sweep reclaims at 02:10; hour 2 sweep already reclaims at 02:05.
    frame.loc[frame.index[23], ["open", "high", "low", "close"]] = [105, 109, 98, 99]
    frame.loc[frame.index[24], ["open", "high", "low", "close"]] = [99, 113, 96, 99]
    frame.loc[frame.index[25], ["open", "high", "low", "close"]] = [99, 110, 99, 105]
    prefix = signals(frame.iloc[:25], "boom")["CRT"]
    full = signals(frame, "boom")["CRT"]
    np.testing.assert_array_equal(prefix, [24])
    np.testing.assert_array_equal(full[full < 25], prefix)


@pytest.mark.parametrize("direction", ["boom", "crash"])
@pytest.mark.parametrize("reclaim_close", [99.0, 100.0, 110.0, 111.0])
def test_reclaim_must_close_strictly_between_both_reference_edges(direction, reclaim_close):
    frame = setup_bars()
    for j in range(12, 15):
        opening = min(max(reclaim_close, 100), 110)
        high = max(112 if direction == "crash" and j == 12 else 110, reclaim_close)
        low = min(98 if direction == "boom" and j == 12 else 100, reclaim_close)
        frame.loc[frame.index[j], ["open", "high", "low", "close"]] = [
            opening, high, low, reclaim_close,
        ]
    assert signals(frame, direction)["CRT"].size == 0


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_valid_sweep_and_interior_reclaim_emit_one_setup_signal(direction):
    frame = setup_bars()
    edge, value = ("low", 98.0) if direction == "boom" else ("high", 112.0)
    frame.loc[frame.index[12:15], edge] = value
    np.testing.assert_array_equal(signals(frame, direction)["CRT"], [12])


@pytest.mark.parametrize("direction", ["boom", "crash"])
@pytest.mark.parametrize("index_unit", ["s", "us", "ns"])
def test_confirmation_extreme_is_not_a_future_hit_and_first_hit_is_one_minute(direction, index_unit):
    frame = minute_history(180)
    frame.index = frame.index.as_unit(index_unit)
    frame.loc[:, :] = 100.0
    m5 = setup_bars(1)
    m5.index = pd.DatetimeIndex([frame.index[60]]).as_unit("ns")
    m5.loc[:, ["open", "high", "low", "close"]] = 100.0
    m5["atr"] = 1.0
    key, extreme = ("high", 110.0) if direction == "boom" else ("low", 90.0)
    frame.loc[frame.index[60:65], key] = extreme
    hit, tts, valid = outcomes(frame, m5, direction, multiplier=2, horizon=5)
    assert valid.tolist() == [True]
    assert hit.tolist() == [False]
    assert np.isnan(tts[0])
    frame.loc[frame.index[65], key] = extreme
    hit, tts, valid = outcomes(frame, m5, direction, multiplier=2, horizon=5)
    assert valid.tolist() == [True]
    assert hit.tolist() == [True]
    assert tts.tolist() == [1.0]
    frame.loc[frame.index[65], key] = 100.0
    frame.loc[frame.index[69], key] = extreme
    _, tts, _ = outcomes(frame, m5, direction, multiplier=2, horizon=5)
    assert tts.tolist() == [5.0]
    frame.loc[frame.index[69], key] = 100.0
    frame.loc[frame.index[70], key] = extreme
    hit, _, _ = outcomes(frame, m5, direction, multiplier=2, horizon=5)
    assert hit.tolist() == [False]


@pytest.mark.parametrize("remove_row", [False, True])
def test_missing_future_minute_invalidates_even_a_visible_hit(remove_row):
    frame = minute_history(180)
    frame.loc[:, :] = 100.0
    frame.loc[frame.index[65], "high"] = 110.0
    m5 = setup_bars(1)
    m5.index = pd.DatetimeIndex([frame.index[60]])
    m5["close"] = 100.0
    m5["atr"] = 1.0
    missing = frame.index[67]
    if remove_row:
        frame = frame.drop(missing)
    else:
        frame.loc[missing] = np.nan
    hit, tts, valid = outcomes(frame, m5, "boom", multiplier=2, horizon=5)
    assert valid.tolist() == [False]
    assert hit.tolist() == [False]
    assert np.isnan(tts[0])


def test_partition_windows_purge_thirty_minutes_before_each_boundary():
    frame = minute_history(3000)
    m5 = features(frame, "boom")
    splits = partitions(frame)
    start, end = frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1)
    assert splits["development"] == (start, start + 0.7 * (end - start))
    assert splits["final_test"] == (splits["development"][1], end)
    assert splits["wf1"][1] == splits["wf2"][0]
    assert splits["wf2"][1] == splits["wf3"][0]
    assert splits["wf3"][1] == splits["development"][1]
    close = m5.index + pd.Timedelta(minutes=5)
    for lo, hi in splits.values():
        chosen = eligible(m5, np.ones(len(m5), dtype=bool), lo, hi)
        assert (close[chosen] >= lo).all()
        assert (close[chosen] + pd.Timedelta(minutes=30) <= hi).all()
        unsafe = (close >= lo) & (close < hi) & (close + pd.Timedelta(minutes=30) > hi)
        assert unsafe.any()
        assert not chosen[unsafe].any()


def test_gap_invalidates_reference_and_resets_rolling_lookback(tmp_path):
    frame = minute_history()
    missing_minute = frame.index[1200]
    source = tmp_path / "gapped.csv"
    csv_rows(frame.drop(missing_minute)).to_csv(source, index=False)
    loaded, audit = load_m1(source)
    assert audit["missing_minutes"] == 1
    assert audit["gap_intervals"] == 1
    assert loaded.loc[missing_minute].isna().all()
    h1 = aggregate_complete(loaded, 60, 60)
    assert h1.loc[missing_minute.floor("h")].isna().all()
    m5 = features(loaded, "boom")
    gap_idx = m5.index.get_loc(missing_minute.floor("5min"))
    assert m5.feature_valid.iloc[gap_idx - 1]
    assert not m5.feature_valid.iloc[gap_idx:gap_idx + 120].any()
    assert m5.feature_valid.iloc[gap_idx + 120]
    for indices in signals(m5, "boom").values():
        assert not ((indices >= gap_idx) & (indices < gap_idx + 120)).any()


def test_loader_removes_identical_duplicates_but_rejects_conflicting_ohlc(tmp_path):
    rows = csv_rows(minute_history(1200))
    source = tmp_path / "duplicate.csv"
    duplicate = rows.iloc[[600]].copy()
    pd.concat([rows, duplicate], ignore_index=True).to_csv(source, index=False)
    frame, audit = load_m1(source)
    assert len(frame) == 1200
    assert audit["duplicate_equal_rows_removed"] == 1
    duplicate["high"] += 1
    pd.concat([rows, duplicate], ignore_index=True).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Conflicting prices"):
        load_m1(source)


@pytest.mark.parametrize("unsafe_flag", SAFETY_FLAGS)
def test_every_live_flag_individually_refuses_research(monkeypatch, unsafe_flag):
    for name in SAFETY_FLAGS:
        monkeypatch.setenv(name, "false")
    assert assert_offline() == {name: False for name in SAFETY_FLAGS}
    monkeypatch.setenv(unsafe_flag, "true")
    with pytest.raises(RuntimeError, match="Safety invariant"):
        assert_offline()
