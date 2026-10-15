"""Behavioral checks for uncapped, delayed, offline stop-or-time exits."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.research.payoff import TRADE_COLUMNS
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.spike_hunter import SAFETY_FLAGS


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for name in SAFETY_FLAGS:
        monkeypatch.setenv(name, "false")


def history(n=90):
    index = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}, index=index)


def signal(frame, minute=0, side=1):
    return {"signal_time": frame.index[minute], "atr": 2.0, "side": side,
            "signal_close": 999.0, "variant": "TEST"}


def cfg(**kwargs):
    return replace(TimedExitConfig(1.0, 5, round_trip_cost_atr=0.0), **kwargs)


def run(frame, issued=None, config=None, start=None, end=None, purge_minutes=31):
    return replay_timed(frame, [signal(frame)] if issued is None else issued,
                        cfg() if config is None else config,
                        frame.index[0] if start is None else start,
                        frame.index[-1] + pd.Timedelta(minutes=1) if end is None else end,
                        purge_minutes=purge_minutes)


def setbar(frame, minute, op, hi, lo, cl):
    frame.loc[frame.index[minute], ["open", "high", "low", "close"]] = [op, hi, lo, cl]


def test_default_protocol_has_cost_delay_and_no_target():
    config = TimedExitConfig(1, 15)
    assert config.entry_delay_minutes == 1
    assert config.round_trip_cost_atr == 0.10
    assert config.fill_mode == "adverse_extreme"
    assert not hasattr(config, "target_atr")
    _, audit = run(history(), config=config)
    assert audit["take_profit"] is None and audit["ambiguous"] == 0


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_entry_is_exact_delayed_open_with_signal_atr_frozen(unit):
    frame = history()
    frame.index = frame.index.as_unit(unit)
    frame.iloc[1:6] = [103, 104, 102, 103]
    trades, audit = run(frame)
    trade = trades.iloc[0]
    assert trade.entry_time == frame.index[1]
    assert trade.entry == 103 and trade.atr == 2
    assert trade.reason == "time" and trade.exit == 103
    assert trade.net_R == 0 and audit["completed"] == 1


@pytest.mark.parametrize("side", [1, -1])
def test_large_spike_profit_is_uncapped_and_only_realized_at_timeout(side):
    frame = history()
    if side == 1:
        setbar(frame, 1, 100, 120, 99, 118)
        frame.iloc[2:6] = [118, 121, 117, 118]
    else:
        setbar(frame, 1, 100, 101, 80, 82)
        frame.iloc[2:6] = [82, 83, 79, 82]
    trades, audit = run(frame, issued=[signal(frame, side=side)])
    trade = trades.iloc[0]
    assert trade.reason == "time" and trade.gross_R == 9
    assert trade.holding_minutes == 5 and trade.exit_time == frame.index[6]
    assert not trade.ambiguous and audit["ambiguous"] == 0


@pytest.mark.parametrize("side", [1, -1])
def test_spike_high_or_low_does_not_create_profit_exit_before_later_stop(side):
    frame = history()
    if side == 1:
        setbar(frame, 1, 100, 130, 99, 100)
        setbar(frame, 2, 100, 101, 97, 99)
    else:
        setbar(frame, 1, 100, 101, 70, 100)
        setbar(frame, 2, 100, 103, 99, 101)
    trades, _ = run(frame, issued=[signal(frame, side=side)])
    assert trades.iloc[0].reason == "sl" and trades.iloc[0].gross_R == -1.5
    assert trades.iloc[0].exit_time == frame.index[3]


@pytest.mark.parametrize("side", [1, -1])
def test_favorable_and_adverse_extremes_in_same_minute_still_stop(side):
    frame = history()
    setbar(frame, 1, 100, 120, 80, 100)
    trades, _ = run(frame, issued=[signal(frame, side=side)])
    assert trades.iloc[0].reason == "sl" and trades.iloc[0].gross_R == -10
    assert not trades.iloc[0].ambiguous  # There is only one barrier.


@pytest.mark.parametrize("side", [1, -1])
def test_opening_gap_stop_proxy_uses_worse_open_and_stress_uses_minute_extreme(side):
    frame = history()
    if side == 1:
        setbar(frame, 2, 97, 110, 95, 105)
    else:
        setbar(frame, 2, 103, 105, 90, 95)
    proxy, _ = run(frame, issued=[signal(frame, side=side)], config=cfg(fill_mode="barrier_proxy"))
    stress, _ = run(frame, issued=[signal(frame, side=side)])
    assert proxy.iloc[0].gross_R == -1.5 and stress.iloc[0].gross_R == -2.5
    assert proxy.iloc[0].exit_time == frame.index[2]
    assert stress.iloc[0].exit_time == frame.index[3]
    assert proxy.iloc[0].reason == stress.iloc[0].reason == "sl"


@pytest.mark.parametrize("side", [1, -1])
def test_intrabar_proxy_stops_at_barrier_and_stress_at_extreme(side):
    frame = history()
    setbar(frame, 1, 100, 101 if side == 1 else 105, 95 if side == 1 else 99, 100)
    proxy, _ = run(frame, issued=[signal(frame, side=side)], config=cfg(fill_mode="barrier_proxy"))
    stress, _ = run(frame, issued=[signal(frame, side=side)])
    assert proxy.iloc[0].gross_R == -1 and stress.iloc[0].gross_R == -2.5
    assert proxy.iloc[0].exit_time == stress.iloc[0].exit_time == frame.index[2]


def test_known_gap_proxy_can_exit_at_open_without_later_intrabar_prices():
    frame = history()
    frame.loc[frame.index[2]] = [97, np.nan, np.nan, np.nan]
    proxy, _ = run(frame, config=cfg(fill_mode="barrier_proxy"))
    stress, _ = run(frame)
    assert not proxy.iloc[0].censored and proxy.iloc[0].exit == 97
    assert proxy.iloc[0].exit_time == frame.index[2]
    assert stress.iloc[0].censored and np.isnan(stress.iloc[0].net_R)


def test_timeout_uses_only_last_included_close_not_next_minute():
    frame = history()
    setbar(frame, 5, 100, 104, 99, 103)
    setbar(frame, 6, 103, 150, 95, 140)
    trades, _ = run(frame)
    trade = trades.iloc[0]
    assert trade.reason == "time" and trade.exit == 103
    assert trade.exit_time == frame.index[6] and trade.gross_R == 1.5


def test_last_minute_stop_precedes_timeout_even_if_close_recovers():
    frame = history()
    setbar(frame, 5, 100, 106, 97, 105)
    trades, _ = run(frame)
    assert trades.iloc[0].reason == "sl" and trades.iloc[0].exit == 97
    assert trades.iloc[0].exit_time == frame.index[6]


def test_confirmation_extremes_and_latency_before_entry_are_not_execution():
    frame = history()
    setbar(frame, 4, 100, 150, 90, 100)
    setbar(frame, 5, 100, 150, 90, 100)
    delayed, _ = run(frame, issued=[signal(frame, 5)])
    immediate, _ = run(frame, issued=[signal(frame, 5)], config=cfg(entry_delay_minutes=0))
    assert delayed.iloc[0].reason == "time" and delayed.iloc[0].gross_R == 0
    assert immediate.iloc[0].reason == "sl" and immediate.iloc[0].gross_R == -5


@pytest.mark.parametrize("drop", [False, True])
def test_missing_exact_entry_skips_without_shifting_to_next_row(drop):
    frame = history()
    issued = [signal(frame)]
    if drop:
        frame = frame.drop(frame.index[1])
    else:
        frame.loc[frame.index[1], "open"] = np.nan
    trades, audit = run(frame, issued=issued)
    assert trades.empty and audit["missing_entry"] == 1 and audit["filled"] == 0


@pytest.mark.parametrize("drop", [False, True])
def test_missing_path_censors_and_holds_occupancy_until_planned_end(drop):
    frame = history()
    issued = [signal(frame, minute) for minute in (0, 2, 5)]
    missing = frame.index[2]
    if drop:
        frame = frame.drop(missing)
    else:
        frame.loc[missing] = np.nan
    trades, audit = run(frame, issued=issued)
    first = trades.iloc[0]
    assert first.censored and first.reason == "censored_missing_path"
    assert np.isnan(first.gross_R) and np.isnan(first.net_R)
    assert first.missing_time == missing
    assert first.exit_time == first.planned_end == frame.index[0] + pd.Timedelta(minutes=6)
    assert len(trades) == 2 and audit["overlap_skipped"] == audit["censored"] == 1
    assert not trades.iloc[1].censored


def test_missing_path_before_later_stop_stays_unknown():
    frame = history()
    frame.loc[frame.index[2], "high"] = np.nan
    setbar(frame, 3, 100, 101, 90, 100)
    trades, _ = run(frame)
    assert trades.iloc[0].censored and trades.iloc[0].missing_time == frame.index[2]
    assert np.isnan(trades.iloc[0].net_R)


def test_missing_after_stop_is_irrelevant_and_trade_is_prefix_invariant():
    frame = history()
    setbar(frame, 1, 100, 101, 95, 100)
    issued = [signal(frame)]
    start, end = frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1)
    full, _ = run(frame, issued=issued, start=start, end=end)
    prefix, _ = run(frame.iloc[:2], issued=issued, start=start, end=end)
    frame.iloc[2:] = np.nan
    changed, _ = run(frame, issued=issued, start=start, end=end)
    pd.testing.assert_frame_equal(full, prefix)
    pd.testing.assert_frame_equal(full, changed)
    assert full.iloc[0].reason == "sl" and not full.iloc[0].censored


def test_one_position_rejects_overlap_but_allows_entry_at_exact_exit():
    frame = history()
    trades, audit = run(frame, issued=[signal(frame, minute) for minute in (0, 2, 5)])
    assert trades.signal_time.tolist() == [frame.index[0], frame.index[5]]
    assert audit["overlap_skipped"] == 1
    assert trades.iloc[0].exit_time == trades.iloc[1].entry_time


def test_execution_adds_no_cooldown_after_early_stop():
    frame = history()
    setbar(frame, 1, 100, 101, 95, 100)
    trades, audit = run(frame, issued=[signal(frame, 0), signal(frame, 2)])
    assert len(trades) == 2 and audit["overlap_skipped"] == 0


def test_unsorted_signals_and_duplicates_replay_by_issue_time_then_occupancy():
    frame = history()
    trades, audit = run(frame, issued=[signal(frame, 10), signal(frame, 0), signal(frame, 0)])
    assert trades.signal_time.tolist() == [frame.index[0], frame.index[10]]
    assert audit["overlap_skipped"] == 1


def test_common_purge_and_full_planned_horizon_are_both_required():
    frame = history(80)
    trades, audit = run(frame, issued=[signal(frame, 49), signal(frame, 50)])
    assert trades.signal_time.tolist() == [frame.index[49]] and audit["purged"] == 1
    longer, audit = run(frame, issued=[signal(frame, 40)], config=cfg(max_hold_minutes=40))
    assert longer.empty and audit["purged"] == 1


def test_elapsed_time_partition_can_have_fractional_minute_boundaries():
    frame = history()
    start, end = frame.index[0] + pd.Timedelta(seconds=18), frame.index[70] + pd.Timedelta(seconds=42)
    trades, audit = run(frame, issued=[signal(frame, i) for i in (0, 1, 40, 71)], start=start, end=end)
    assert trades.signal_time.tolist() == [frame.index[1]]
    assert audit["outside_partition"] == 2 and audit["purged"] == 1


def test_cost_is_once_in_initial_frozen_stop_risk_units():
    frame = history()
    setbar(frame, 5, 100, 105, 99, 104)
    trades, _ = run(frame, config=cfg(stop_atr=2, round_trip_cost_atr=0.10))
    assert trades.iloc[0].gross_R == 1 and trades.iloc[0].net_R == pytest.approx(0.95)


@pytest.mark.parametrize("name", SAFETY_FLAGS)
def test_each_live_flag_refuses_even_empty_replay(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(RuntimeError, match="Safety invariant"):
        run(history(), issued=[])


@pytest.mark.parametrize("change", [
    {"stop_atr": 0}, {"stop_atr": -1}, {"stop_atr": np.inf}, {"stop_atr": np.nan}, {"stop_atr": True},
    {"max_hold_minutes": 0}, {"max_hold_minutes": 1.5}, {"max_hold_minutes": True},
    {"entry_delay_minutes": -1}, {"entry_delay_minutes": True},
    {"round_trip_cost_atr": -0.1}, {"round_trip_cost_atr": np.inf}, {"fill_mode": "optimistic"},
])
def test_invalid_config_refused(change):
    with pytest.raises(ValueError):
        cfg(**change)


@pytest.mark.parametrize("change", [{"atr": 0}, {"atr": np.nan}, {"atr": np.inf}, {"side": 0}, {"side": True}])
def test_invalid_signal_refused(change):
    frame = history()
    issued = signal(frame)
    issued.update(change)
    with pytest.raises(ValueError):
        run(frame, issued=[issued])


def test_naive_or_offgrid_signal_and_naive_grid_are_refused():
    frame = history()
    issued = signal(frame)
    issued["signal_time"] = pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError, match="timezone-aware"):
        run(frame, issued=[issued])
    issued["signal_time"] = frame.index[0] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="minute boundary"):
        run(frame, issued=[issued])
    frame.index = frame.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        run(frame)


def test_duplicate_unsorted_offgrid_data_and_invalid_used_ohlc_are_refused():
    frame = history()
    with pytest.raises(ValueError, match="sorted and unique"):
        run(pd.concat([frame.iloc[:2], frame.iloc[1:]]))
    with pytest.raises(ValueError, match="sorted and unique"):
        run(frame.iloc[::-1], start=frame.index[0], end=frame.index[-1])
    shifted = frame.copy()
    shifted.index += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="UTC minute openings"):
        run(shifted)
    setbar(frame, 1, 100, 99, 98, 100)
    with pytest.raises(ValueError, match="Invalid OHLC bounds"):
        run(frame)


def test_empty_schema_is_compatible_with_existing_metrics_and_safety_audit():
    trades, audit = run(history(), issued=[])
    assert list(trades.columns) == TRADE_COLUMNS and trades.empty
    assert str(trades.signal_time.dtype) == "datetime64[ns, UTC]"
    assert trades.net_R.dtype == float and trades.censored.dtype == bool
    assert audit["issued"] == audit["filled"] == 0
    assert all(value is False for value in audit["safety"].values())
