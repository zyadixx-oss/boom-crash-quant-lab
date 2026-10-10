"""Execution-boundary, gap, censoring and causality checks for offline payoff."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.research.payoff import BracketConfig, replay_brackets
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
    return replace(BracketConfig(1.0, 2.0, 5, round_trip_cost_atr=0.0,
                                 fill_mode="barrier_proxy"), **kwargs)


def run(frame, issued=None, config=None, start=None, end=None, purge_minutes=31):
    return replay_brackets(frame, [signal(frame)] if issued is None else issued,
                           cfg() if config is None else config,
                           frame.index[0] if start is None else start,
                           frame.index[-1] + pd.Timedelta(minutes=1) if end is None else end,
                           purge_minutes=purge_minutes)


def setbar(frame, minute, op, hi, lo, cl):
    frame.loc[frame.index[minute], ["open", "high", "low", "close"]] = [op, hi, lo, cl]


def test_defaults_are_delayed_costed_and_pessimistic():
    config = BracketConfig(1, 2, 15)
    assert config.entry_delay_minutes == 1
    assert config.round_trip_cost_atr == 0.10
    assert config.fill_mode == "adverse_extreme"
    assert config.same_bar_policy == "stop_first"


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_entry_uses_exact_delayed_next_open_and_frozen_signal_atr(unit):
    frame = history()
    frame.index = frame.index.as_unit(unit)
    setbar(frame, 1, 103, 104, 102, 103)
    frame.iloc[2:6] = [103, 104, 102, 103]
    trades, audit = run(frame)
    trade = trades.iloc[0]
    assert trade.entry_time == frame.index[1]
    assert trade.entry == 103 and trade.atr == 2
    assert trade.reason == "time" and trade.exit == 103
    assert trade.net_R == 0  # TP is 107, not 104 from the signal close.
    assert audit["filled"] == audit["completed"] == 1


def test_latency_zero_and_one_use_distinct_precommitted_entry_candles():
    frame = history()
    setbar(frame, 0, 100, 105, 99, 104)
    immediate, _ = run(frame, config=cfg(entry_delay_minutes=0))
    delayed, _ = run(frame, config=cfg(entry_delay_minutes=1))
    assert immediate.iloc[0].reason == "tp" and immediate.iloc[0].gross_R == 2
    assert delayed.iloc[0].reason == "time" and delayed.iloc[0].gross_R == 0


def test_confirmation_candle_extreme_is_never_an_execution_hit():
    frame = history()
    setbar(frame, 4, 100, 150, 99, 100)
    trades, _ = run(frame, issued=[signal(frame, 5)])
    assert trades.iloc[0].entry_time == frame.index[6]
    assert trades.iloc[0].reason == "time"


@pytest.mark.parametrize("side", [1, -1])
def test_tp_and_sl_are_symmetric_for_boom_and_crash(side):
    frame = history()
    if side == 1:
        setbar(frame, 1, 100, 105, 99, 104)
    else:
        setbar(frame, 1, 100, 101, 95, 96)
    trades, _ = run(frame, issued=[signal(frame, side=side)])
    assert trades.iloc[0].reason == "tp"
    assert trades.iloc[0].exit == (104 if side == 1 else 96)
    assert trades.iloc[0].gross_R == 2
    assert trades.iloc[0].exit_time == frame.index[2]


@pytest.mark.parametrize("side", [1, -1])
def test_both_intrabar_barriers_are_ambiguous_stop_first_or_target_sensitivity(side):
    frame = history()
    setbar(frame, 1, 100, 105, 95, 100)
    stops, _ = run(frame, issued=[signal(frame, side=side)])
    targets, _ = run(frame, issued=[signal(frame, side=side)], config=cfg(same_bar_policy="target_first"))
    assert stops.iloc[0].ambiguous and stops.iloc[0].reason == "sl"
    assert stops.iloc[0].gross_R == -1
    assert targets.iloc[0].ambiguous and targets.iloc[0].reason == "tp"
    assert targets.iloc[0].gross_R == 2


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("fill_mode", ["barrier_proxy", "adverse_extreme"])
def test_opening_target_gap_precedes_opposite_intrabar_stop(side, fill_mode):
    frame = history()
    if side == 1:
        setbar(frame, 2, 105, 106, 97, 100)
    else:
        setbar(frame, 2, 95, 103, 94, 100)
    trades, _ = run(frame, issued=[signal(frame, side=side)], config=cfg(fill_mode=fill_mode))
    trade = trades.iloc[0]
    assert trade.reason == "tp" and trade.gross_R == 2
    assert not trade.ambiguous
    assert trade.exit_time == frame.index[2]  # The opening itself determines order.
    assert trade.holding_minutes == 1


@pytest.mark.parametrize("side", [1, -1])
def test_stop_opening_gap_proxy_uses_open_while_stress_uses_adverse_extreme(side):
    frame = history()
    if side == 1:
        setbar(frame, 2, 97, 105, 95, 100)
    else:
        setbar(frame, 2, 103, 105, 95, 100)
    proxy, _ = run(frame, issued=[signal(frame, side=side)])
    stress, _ = run(frame, issued=[signal(frame, side=side)], config=cfg(fill_mode="adverse_extreme"))
    assert proxy.iloc[0].gross_R == -1.5
    assert stress.iloc[0].gross_R == -2.5
    assert proxy.iloc[0].reason == stress.iloc[0].reason == "sl"
    assert not proxy.iloc[0].ambiguous and not stress.iloc[0].ambiguous
    assert proxy.iloc[0].exit_time == frame.index[2]
    assert stress.iloc[0].exit_time == frame.index[3]


def test_intrabar_stop_stress_exceeds_barrier_loss():
    frame = history()
    setbar(frame, 1, 100, 101, 95, 99)
    proxy, _ = run(frame)
    stress, _ = run(frame, config=cfg(fill_mode="adverse_extreme"))
    assert proxy.iloc[0].exit == 98 and proxy.iloc[0].gross_R == -1
    assert stress.iloc[0].exit == 95 and stress.iloc[0].gross_R == -2.5


def test_time_exit_uses_last_included_close_and_excludes_next_minute():
    frame = history()
    setbar(frame, 5, 100, 103, 99, 102)
    setbar(frame, 6, 102, 120, 101, 110)
    trades, _ = run(frame)
    trade = trades.iloc[0]
    assert trade.reason == "time" and trade.exit == 102
    assert trade.exit_time == frame.index[6] and trade.holding_minutes == 5
    assert trade.gross_R == 1


def test_last_included_minute_barrier_precedes_scheduled_time_exit():
    frame = history()
    setbar(frame, 5, 100, 105, 99, 101)
    trades, _ = run(frame)
    assert trades.iloc[0].reason == "tp" and trades.iloc[0].exit == 104
    assert trades.iloc[0].exit_time == frame.index[6]


@pytest.mark.parametrize("drop", [False, True])
def test_missing_entry_is_explicit_and_never_forward_filled(drop):
    frame = history()
    issued = [signal(frame)]
    if drop:
        frame = frame.drop(frame.index[1])
    else:
        frame.loc[frame.index[1], "open"] = np.nan
    trades, audit = run(frame, issued=issued)
    assert trades.empty and audit["missing_entry"] == 1 and audit["filled"] == 0


@pytest.mark.parametrize("drop", [False, True])
def test_missing_path_censors_and_reserves_position_until_planned_end(drop):
    frame = history()
    issued = [signal(frame, minute) for minute in (0, 2, 5)]
    missing = frame.index[2]
    if drop:
        frame = frame.drop(missing)
    else:
        frame.loc[missing] = np.nan
    trades, audit = run(frame, issued=issued)
    censored = trades.iloc[0]
    assert censored.censored and censored.reason == "censored_missing_path"
    assert np.isnan(censored.gross_R) and np.isnan(censored.net_R)
    assert censored.missing_time == missing
    assert censored.exit_time == censored.planned_end == frame.index[0] + pd.Timedelta(minutes=6)
    assert audit["censored"] == 1 and audit["overlap_skipped"] == 1
    assert len(trades) == 2 and not trades.iloc[1].censored


def test_partial_path_ohlc_censors_even_when_target_high_is_visible():
    frame = history()
    setbar(frame, 2, 100, 110, 99, 100)
    frame.loc[frame.index[2], "low"] = np.nan
    trades, _ = run(frame)
    assert trades.iloc[0].censored and np.isnan(trades.iloc[0].net_R)


def test_known_opening_target_exit_does_not_need_later_intrabar_prices():
    frame = history()
    frame.loc[frame.index[2]] = [105, np.nan, np.nan, np.nan]
    trades, _ = run(frame)
    assert not trades.iloc[0].censored
    assert trades.iloc[0].reason == "tp" and trades.iloc[0].exit_time == frame.index[2]


def test_gap_after_completed_early_exit_does_not_retroactively_drop_trade():
    frame = history()
    setbar(frame, 1, 100, 105, 99, 104)
    frame.loc[frame.index[3]] = np.nan
    trades, audit = run(frame)
    assert trades.iloc[0].reason == "tp" and not trades.iloc[0].censored
    assert audit["completed"] == 1 and audit["censored"] == 0


def test_position_overlap_is_rejected_without_queuing_and_exact_exit_allows_entry():
    frame = history()
    trades, audit = run(frame, issued=[signal(frame, minute) for minute in (0, 2, 5)])
    assert audit["overlap_skipped"] == 1
    assert trades.signal_time.tolist() == [frame.index[0], frame.index[5]]
    assert trades.iloc[1].entry_time == trades.iloc[0].exit_time


def test_early_exit_allows_new_issued_signal_without_extra_cooldown():
    frame = history()
    setbar(frame, 1, 100, 105, 99, 104)
    trades, audit = run(frame, issued=[signal(frame, 0), signal(frame, 2)])
    assert len(trades) == 2 and audit["overlap_skipped"] == 0


def test_common_signal_purge_is_independent_of_actual_early_exit():
    frame = history(80)
    setbar(frame, 51, 100, 105, 99, 104)
    trades, audit = run(frame, issued=[signal(frame, 49), signal(frame, 50)])
    assert trades.signal_time.tolist() == [frame.index[49]]
    assert audit["purged"] == 1  # t50+31>end, even though a TP would arrive early.


def test_entry_plus_holding_boundary_is_required_in_addition_to_common_purge():
    frame = history(80)
    trades, audit = run(frame, issued=[signal(frame, 40)], config=cfg(max_hold_minutes=40))
    assert trades.empty and audit["purged"] == 1


def test_partition_start_is_inclusive_end_is_exclusive():
    frame = history(90)
    start, end = frame.index[10], frame.index[70]
    trades, audit = run(frame, issued=[signal(frame, minute) for minute in (9, 10, 70)], start=start, end=end)
    assert trades.signal_time.tolist() == [start]
    assert audit["outside_partition"] == 2


def test_elapsed_time_partition_boundaries_can_fall_between_minutes():
    frame = history(91)
    start = frame.index[0] + pd.Timedelta(seconds=18)
    end = frame.index[70] + pd.Timedelta(seconds=42)
    trades, audit = run(frame, issued=[signal(frame, minute) for minute in (0, 1, 40)], start=start, end=end)
    assert trades.signal_time.tolist() == [frame.index[1]]
    assert audit["outside_partition"] == 1 and audit["purged"] == 1


def test_cost_is_charged_once_in_frozen_stop_risk_units():
    frame = history()
    setbar(frame, 1, 100, 105, 99, 104)
    trades, _ = run(frame, config=cfg(stop_atr=2, round_trip_cost_atr=0.10))
    assert trades.iloc[0].gross_R == 1
    assert trades.iloc[0].net_R == pytest.approx(0.95)


def test_completed_trade_is_prefix_invariant_and_ignores_replaced_future():
    frame = history()
    setbar(frame, 1, 100, 105, 99, 104)
    issued = [signal(frame)]
    start, end = frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1)
    full, _ = run(frame, issued=issued, start=start, end=end)
    prefix, _ = run(frame.iloc[:2], issued=issued, start=start, end=end)
    changed = frame.copy()
    changed.iloc[2:] *= 100
    future_changed, _ = run(changed, issued=issued, start=start, end=end)
    pd.testing.assert_frame_equal(full, prefix)
    pd.testing.assert_frame_equal(full, future_changed)


def test_unsorted_issued_signals_replay_in_causal_issue_order():
    frame = history()
    trades, _ = run(frame, issued=[signal(frame, 10), signal(frame, 0)])
    assert trades.signal_time.tolist() == [frame.index[0], frame.index[10]]


@pytest.mark.parametrize("name", SAFETY_FLAGS)
def test_each_live_flag_refuses_execution_even_with_no_signals(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(RuntimeError, match="Safety invariant"):
        run(history(), issued=[])


@pytest.mark.parametrize("change", [
    {"stop_atr": 0}, {"target_atr": -1}, {"stop_atr": np.nan},
    {"max_hold_minutes": 0}, {"max_hold_minutes": 1.5}, {"entry_delay_minutes": -1},
    {"entry_delay_minutes": True}, {"round_trip_cost_atr": -0.1},
    {"round_trip_cost_atr": np.inf}, {"fill_mode": "optimistic"},
    {"same_bar_policy": "unavailable"},
])
def test_invalid_protocol_parameters_are_rejected(change):
    with pytest.raises(ValueError):
        cfg(**change)


@pytest.mark.parametrize("change", [{"atr": 0}, {"atr": np.nan}, {"side": 0}, {"side": True}])
def test_invalid_issued_signal_is_rejected(change):
    frame = history()
    issued = signal(frame)
    issued.update(change)
    with pytest.raises(ValueError):
        run(frame, issued=[issued])


def test_naive_clock_duplicate_grid_and_invalid_path_bounds_are_rejected():
    frame = history()
    naive = signal(frame)
    naive["signal_time"] = pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError, match="timezone-aware"):
        run(frame, issued=[naive])
    with pytest.raises(ValueError, match="sorted and unique"):
        run(pd.concat([frame.iloc[:2], frame.iloc[1:]]))
    setbar(frame, 1, 100, 99, 98, 100)
    with pytest.raises(ValueError, match="Invalid OHLC bounds"):
        run(frame)


def test_empty_results_retain_stable_schema_and_audit():
    trades, audit = run(history(), issued=[])
    assert trades.empty and audit["issued"] == audit["filled"] == 0
    assert str(trades.signal_time.dtype) == "datetime64[ns, UTC]"
    assert trades.net_R.dtype == float and trades.censored.dtype == bool
    assert all(value is False for value in audit["safety"].values())
