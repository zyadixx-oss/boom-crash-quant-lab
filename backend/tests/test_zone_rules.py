"""Synthetic causal region definitions; no historical prices or outcomes."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.research.zone_dataset import ZoneDataset
from app.research.zone_rules import VARIANTS, zone_candidates


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for flag in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"):
        monkeypatch.setenv(flag, "false")


def _dataset(frame):
    coverage = pd.DataFrame({"minute_valid": True, "observed_seconds": 60,
                             "run_id": pd.array(np.zeros(len(frame)), dtype="Int64")}, index=frame.index)
    return ZoneDataset(frame, coverage, {})


@pytest.fixture(scope="module")
def rising_dataset():
    index = pd.date_range("2026-01-01", periods=8 * 1440, freq="min", tz="UTC")
    opening = 100.0 + np.arange(len(index)) * 0.01
    return _dataset(pd.DataFrame({"open": opening, "high": opening + 0.02,
                                  "low": opening - 0.01, "close": opening + 0.005}, index=index))


def _mirror(dataset):
    source = dataset.m1
    frame = pd.DataFrame({"open": 10000 / source.open, "high": 10000 / source.low,
                           "low": 10000 / source.high, "close": 10000 / source.close}, index=source.index)
    return replace(dataset, m1=frame)


def _row(result, issue, variant):
    rows = result.loc[(result.signal_time == issue) & (result.variant == variant)]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.mark.parametrize("direction", ("boom", "crash"))
def test_all_clock_rows_retained_and_regions_ahead_with_native_side(rising_dataset, direction):
    dataset = rising_dataset if direction == "boom" else _mirror(rising_dataset)
    result = zone_candidates(dataset, direction)
    assert len(result) == 8 * 48 * 4
    assert tuple(result.variant.unique()) == VARIANTS
    assert result.zone_id.is_unique
    assert (result.signal_time.dt.minute % 30 == 0).all()
    assert result.signal_time.equals(result.issue_time)
    assert not result.loc[result.signal_time < pd.Timestamp("2026-01-01 04:00Z"), "eligible"].any()
    for variant in ("FIB_RETRACE", "TREND_RETEST", "CONTEXT_GEOMETRIC"):
        assert result.loc[result.variant == variant, "eligible"].any()
    valid = result.loc[result.eligible]
    assert valid.zone_low.gt(0).all() and (valid.zone_low < valid.zone_high).all()
    assert valid.side.eq(1 if direction == "boom" else -1).all()
    if direction == "boom":
        assert (valid.zone_high < valid.issue_price).all()
        assert (valid.invalidation < valid.zone_low).all()
    else:
        assert (valid.issue_price < valid.zone_low).all()
        assert (valid.zone_high < valid.invalidation).all()
    assert result.loc[~result.eligible, ["zone_low", "zone_high", "invalidation"]].isna().all().all()
    assert result.reason.ne("").all()


def test_exact_fibonacci_band_and_geometry_control(rising_dataset):
    issue = pd.Timestamp("2026-01-05 12:30Z")
    result = zone_candidates(rising_dataset, "boom")
    fib = _row(result, issue, "FIB_RETRACE")
    source = rising_dataset.m1.loc[issue - pd.Timedelta(hours=1):issue - pd.Timedelta(minutes=1)]
    low, high = source.low.min(), source.high.max()
    assert fib.eligible
    assert fib.zone_low == pytest.approx(high - 0.786 * (high - low))
    assert fib.zone_high == pytest.approx(high - 0.618 * (high - low))
    geometric = _row(result, issue, "CONTEXT_GEOMETRIC")
    assert geometric.zone_low == pytest.approx(geometric.issue_price - 0.55 * geometric.atr)
    assert geometric.zone_high == pytest.approx(geometric.issue_price - 0.45 * geometric.atr)


def test_crt_uses_reference_closed_before_m15_setup(rising_dataset):
    frame = rising_dataset.m1.copy(deep=True)
    issue = pd.Timestamp("2026-01-05 12:30Z")
    reference = frame.loc[pd.Timestamp("2026-01-05 11:00Z"):pd.Timestamp("2026-01-05 11:59Z")]
    crl = reference.low.min()
    frame.loc[pd.Timestamp("2026-01-05 12:16Z"), "low"] = crl - 0.03
    for offset in range(5):
        stamp = issue - pd.Timedelta(minutes=5 - offset)
        price = crl + 0.12 + 0.02 * offset
        frame.loc[stamp, ["open", "high", "low", "close"]] = [price, price + 0.02, price - 0.01, price + 0.005]
    row = _row(zone_candidates(replace(rising_dataset, m1=frame), "boom"), issue, "CRT_RETEST")
    assert row.eligible
    assert row.zone_low == crl
    assert row.zone_high == pytest.approx(crl + 0.10 * row.atr)


def test_latest_m1_body_is_not_a_native_direction_gate(rising_dataset):
    issue = pd.Timestamp("2026-01-05 12:30Z")
    frame = rising_dataset.m1.copy()
    stamp = issue - pd.Timedelta(minutes=1)
    frame.loc[stamp, "open"] = frame.loc[stamp, "close"] + 0.001
    result = zone_candidates(replace(rising_dataset, m1=frame), "boom")
    assert _row(result, issue, "CONTEXT_GEOMETRIC").eligible


@pytest.mark.parametrize("direction", ("boom", "crash"))
def test_future_bars_cannot_move_preissued_regions_or_confirm_pivots(rising_dataset, direction):
    dataset = rising_dataset if direction == "boom" else _mirror(rising_dataset)
    issue = pd.Timestamp("2026-01-05 12:30Z")
    original = zone_candidates(dataset, direction)
    changed = dataset.m1.copy()
    changed.loc[changed.index >= issue, :] *= 10.0
    altered = zone_candidates(replace(dataset, m1=changed), direction)
    assert_frame_equal(original.loc[original.signal_time <= issue].reset_index(drop=True),
                       altered.loc[altered.signal_time <= issue].reset_index(drop=True), check_exact=True)


def test_true_gap_invalidates_old_h4_context_and_restarts_ema(rising_dataset):
    issue = pd.Timestamp("2026-01-05 12:30Z")
    missing = pd.Timestamp("2026-01-05 12:03Z")
    frame, coverage = rising_dataset.m1.copy(), rising_dataset.coverage.copy()
    frame.loc[missing, :] = np.nan
    coverage.loc[missing, "minute_valid"] = False
    coverage.loc[missing, "observed_seconds"] = 59
    coverage.loc[missing, "run_id"] = pd.NA
    coverage.loc[coverage.index > missing, "run_id"] = 1
    result = zone_candidates(replace(rising_dataset, m1=frame, coverage=coverage), "boom")
    early = result.loc[(result.signal_time >= issue) & (result.signal_time < pd.Timestamp("2026-01-05 20:00Z"))]
    assert not early.eligible.any()
    assert set(early.reason) <= {"context_unavailable", "context_crosses_native_gap", "raw_m5_atr_unavailable"}
    assert _row(result, pd.Timestamp("2026-01-06 12:30Z"), "TREND_RETEST").eligible


def test_invalid_latest_closed_h4_does_not_fall_back(rising_dataset):
    frame, coverage = rising_dataset.m1.copy(), rising_dataset.coverage.copy()
    missing = pd.Timestamp("2026-01-05 09:00Z")
    frame.loc[missing, :] = np.nan
    coverage.loc[missing, ["minute_valid", "observed_seconds", "run_id"]] = [False, 0, pd.NA]
    coverage.loc[coverage.index > missing, "run_id"] = 1
    issue = pd.Timestamp("2026-01-05 12:30Z")
    result = zone_candidates(replace(rising_dataset, m1=frame, coverage=coverage), "boom")
    assert not result.loc[result.signal_time == issue, "eligible"].any()


def test_no_native_context_and_all_unknown_source_keep_audited_waits(rising_dataset):
    wrong = zone_candidates(rising_dataset, "crash")
    assert not wrong.eligible.any()
    frame, coverage = rising_dataset.m1.copy(), rising_dataset.coverage.copy()
    frame.loc[:, :] = np.nan
    coverage.loc[:, "minute_valid"] = False
    coverage.loc[:, "observed_seconds"] = 0
    coverage.loc[:, "run_id"] = pd.NA
    unknown = zone_candidates(replace(rising_dataset, m1=frame, coverage=coverage), "boom")
    assert len(unknown) == 8 * 48 * 4 and not unknown.eligible.any()


def test_source_dataset_is_not_mutated(rising_dataset):
    before = rising_dataset.m1.copy(deep=True), rising_dataset.coverage.copy(deep=True)
    result = zone_candidates(rising_dataset, "boom")
    result.loc[:, "zone_low"] = 1.0
    assert_frame_equal(rising_dataset.m1, before[0])
    assert_frame_equal(rising_dataset.coverage, before[1])


@pytest.mark.parametrize("kind", ("bad_type", "bad_direction", "misaligned_grid", "counts", "partial_ohlc", "coverage_index", "run_bool"))
def test_invalid_dataset_contract_is_refused(rising_dataset, kind):
    dataset, direction = rising_dataset, "boom"
    if kind == "bad_type":
        dataset = rising_dataset.m1
    elif kind == "bad_direction":
        direction = "both"
    elif kind == "misaligned_grid":
        dataset = replace(dataset, m1=dataset.m1.drop(dataset.m1.index[2]), coverage=dataset.coverage.drop(dataset.coverage.index[2]))
    elif kind == "counts":
        coverage = dataset.coverage.copy()
        coverage.iloc[0, coverage.columns.get_loc("observed_seconds")] = 59
        dataset = replace(dataset, coverage=coverage)
    elif kind == "partial_ohlc":
        frame = dataset.m1.copy()
        frame.iloc[0, 0] = np.nan
        dataset = replace(dataset, m1=frame)
    elif kind == "coverage_index":
        dataset = replace(dataset, coverage=dataset.coverage.iloc[1:])
    elif kind == "run_bool":
        coverage = dataset.coverage.copy()
        coverage["run_id"] = True
        dataset = replace(dataset, coverage=coverage)
    with pytest.raises((TypeError, ValueError)):
        zone_candidates(dataset, direction)


@pytest.mark.parametrize("flag", ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"))
def test_unsafe_flags_refused_before_reading_dataset(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    with pytest.raises(RuntimeError):
        zone_candidates(None, "boom")
