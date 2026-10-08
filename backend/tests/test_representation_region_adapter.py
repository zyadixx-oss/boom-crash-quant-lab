import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.jump_representation import jump_representation
from app.research.tick_tail import FixedTailDetector

P = Path(__file__).resolve().parents[2] / "scripts/representation_regions.py"
S = importlib.util.spec_from_file_location("representation_regions", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def quotes(n=240):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="s")
    price = 100 * np.exp(-np.arange(n) * .00001)
    price[70:] *= 1.02
    return pd.DataFrame({"quote": price}, index=index)


@pytest.mark.parametrize("cuts", [[1, 59, 61, 71, 130], [60, 120, 180], [11, 53, 112, 223]])
def test_chunk_equivalence_and_small_pending(cuts):
    q = quotes()
    detector = FixedTailDetector(1, .00001)
    _, expected = jump_representation(q, detector)
    positions = [0, *cuts, len(q)]
    chunks = [q.iloc[a:b] for a, b in zip(positions, positions[1:])]
    actual, report = M.build_paired_minutes(chunks, detector, start=q.index[0], end=q.index[0] + pd.Timedelta(minutes=4))
    for name in ("raw", "transformed", "coverage"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_names=False)
    assert report["maximum_pending_seconds"] <= 59
    assert report["observed_seconds"] == 240 and report["tail_increments_removed"] == 1


def test_gap_resets_chain_and_preserves_full_empty_minute():
    q = quotes(360).drop(quotes(360).index[120:180])
    detector = FixedTailDetector(1, .00001)
    bars, report = M.build_paired_minutes([q.iloc[:145], q.iloc[145:]], detector,
        start=q.index[0], end=q.index[0] + pd.Timedelta(minutes=6))
    assert len(bars.raw) == 6 and bars.raw.iloc[2].isna().all()
    assert bars.transformed.iloc[3].open == 1
    assert report["missing_seconds"] == 60 and report["runs"] == 2


def test_empty_full_planned_grid_and_outside_value_poison():
    q = quotes(240)
    q.loc[q.index >= q.index[120], "quote"] = np.nan
    detector = FixedTailDetector(1, .00001)
    bars, report = M.build_paired_minutes([q], detector, start=q.index[0], end=q.index[120])
    assert len(bars.raw) == 2 and report["observed_seconds"] == 120
    empty, report = M.build_paired_minutes([], detector, start=q.index[0], end=q.index[120])
    assert empty.raw.isna().all().all() and report["missing_seconds"] == 120
    assert empty.raw.dtypes.eq(float).all()


def test_overlapping_source_chronology_refused():
    q = quotes()
    with pytest.raises(ValueError): M.build_paired_minutes([q, q], FixedTailDetector(1, .00001),
        start=q.index[0], end=q.index[0] + pd.Timedelta(minutes=4))


@pytest.mark.parametrize("symbol,side,lo,hi,inv", [("BOOM600", 1, 98.9, 99.1, 98.4), ("CRASH600", -1, 100.9, 101.1, 101.6)])
def test_fixed_geometric_regions_use_original_quote_and_atr(symbol, side, lo, hi, inv):
    signals = pd.DataFrame({"signal_time": [pd.Timestamp("2026-01-01T00:00Z")],
        "signal_close": [100.], "atr": [2.], "side": [side], "score": [9.]})
    before = signals.copy(deep=True)
    z = M.fixed_regions(signals, symbol=symbol, variant="RAW44")[0]
    assert (z["zone_low"], z["zone_high"], z["invalidation"]) == pytest.approx((lo, hi, inv))
    assert z["atr"] == 2. and z["issue_price"] == 100.
    pd.testing.assert_frame_equal(signals, before)


@pytest.mark.parametrize("field,value", [("side", True), ("side", 1.), ("atr", 0), ("signal_close", np.nan), ("signal_time", "2026-01-01T00:05Z")])
def test_invalid_issue_fields_refused(field, value):
    signals = pd.DataFrame({"signal_time": [pd.Timestamp("2026-01-01T00:00Z")],
        "signal_close": [100.], "atr": [2.], "side": [1]})
    signals[field] = value
    with pytest.raises(ValueError): M.fixed_regions(signals, symbol="BOOM600", variant="RAW44")


def test_missing_entry_and_censored_labels_remain_unknown():
    index = pd.date_range("2026-01-01T00:00Z", periods=3, freq="30min")
    clock = pd.DataFrame({"signal_time": index})
    ledger = pd.DataFrame({"signal_time": index[[0, 2]], "entry_time": index[[0, 2]] + pd.Timedelta(seconds=61),
        "censored": [False, True], "net_R": [.2, np.nan]})
    result = M.training_labels(clock, ledger)
    assert result.completed.tolist() == [True, False, False]
    assert result.net_R.iloc[0] == .2 and result.net_R.iloc[1:].isna().all()
    assert result.planned_end.tolist() == (index + pd.Timedelta(minutes=16)).tolist()


def test_empty_ledger_has_numeric_and_boolean_schema():
    from app.research.payoff_ticks import replay_ticks, TickExitConfig
    from app.research.payoff_metrics import summarize
    from scripts.run_spike_timed_study import weekly_inference
    q = quotes()
    start, end = q.index[0], q.index[0] + pd.Timedelta(days=2)
    raw, _ = replay_ticks(q, [], TickExitConfig(), start, end)
    ledger = M.numeric_ledger(raw)
    assert ledger.net_R.dtype == float and ledger.censored.dtype == bool
    assert summarize(ledger, start, end, stop_atr=2)["completed"] == 0
    assert weekly_inference(ledger, ledger, start, end, repeats=19)["weekly_p"] == 1.
