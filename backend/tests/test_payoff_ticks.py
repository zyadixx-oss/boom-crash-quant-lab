"""Hand-calculated synthetic quote-path checks; no historical tick outcomes."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.research.payoff import TRADE_COLUMNS
from app.research.payoff_ticks import TickExitConfig, TICK_TRADE_COLUMNS, replay_ticks
from app.research.spike_hunter import SAFETY_FLAGS


BASE = pd.Timestamp("2026-01-01T00:00Z")


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for name in SAFETY_FLAGS:
        monkeypatch.setenv(name, "false")


def history(n=3600, start=BASE, freq="s"):
    return pd.DataFrame({"quote": 100.}, index=pd.date_range(start, periods=n, freq=freq, tz="UTC"))


def signal(minute=0, side=1, **overrides):
    return {"signal_time": BASE + pd.Timedelta(minutes=minute), "atr": 1., "side": side,
            "signal_close": 999., "variant": "SYNTHETIC", **overrides}


def cfg(**overrides):
    return replace(TickExitConfig(), **overrides)


def run(frame, issued=None, config=None, start=BASE, end=BASE + pd.Timedelta(hours=1), purge_minutes=31):
    return replay_ticks(frame, [signal()] if issued is None else issued,
                        cfg() if config is None else config, start, end, purge_minutes)


def quote(frame, second, value):
    frame.loc[BASE + pd.Timedelta(seconds=second), "quote"] = value


def test_fixed_primary_defaults_and_explicit_proxy_safety_audit():
    config = TickExitConfig()
    assert config.stop_atr == 2 and config.max_hold_minutes == 15
    assert config.entry_delay_minutes == 1 and config.round_trip_cost_atr == .10
    assert config.max_gap_seconds == 1 and config.stop_latency_ticks == 1
    assert not hasattr(config, "target_atr")
    trades, audit = run(history())
    assert list(trades.columns) == TICK_TRADE_COLUMNS
    assert list(trades.columns[:len(TRADE_COLUMNS)]) == TRADE_COLUMNS
    assert audit["fill_interpretation"] == "quoteproxy_notbrokerfills"
    assert audit["take_profit"] is None and audit["ambiguous"] == 0
    assert audit["stop_rule"] == "next_observed_quote_strictly_after_trigger"
    assert all(audit["safety"][name] is False for name in SAFETY_FLAGS)


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_entry_is_first_strictly_next_quote_and_expiry_is_nominal_not_actual(unit):
    frame = history()
    frame.index = frame.index.as_unit(unit)
    quote(frame, 60, 900)  # At the request timestamp: cannot be used for entry.
    quote(frame, 61, 103)
    frame.loc[BASE + pd.Timedelta(seconds=62):, "quote"] = 104.
    quote(frame, 960, 105)
    quote(frame, 961, 106)
    quote(frame, 962, 200)
    trades, audit = run(frame)
    trade = trades.iloc[0]
    assert trade.nominal_entry_time == BASE + pd.Timedelta(seconds=60)
    assert trade.entry_time == BASE + pd.Timedelta(seconds=61) and trade.entry == 103
    assert trade.planned_end == BASE + pd.Timedelta(seconds=960)
    assert trade.exit_time == BASE + pd.Timedelta(seconds=961) and trade.exit == 106
    assert trade.reason == "time" and trade.holding_minutes == 15
    assert trade.gross_R == 1.5 and trade.net_R == pytest.approx(1.45)
    assert pd.isna(trade.trigger_time) and np.isnan(trade.trigger_quote)
    assert audit["completed"] == 1
    for name in ("entry_time", "trigger_time", "planned_end", "nominal_entry_time"):
        assert str(trades[name].dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("side", [1, -1])
def test_stop_crossing_is_trigger_and_next_quote_retains_gap_loss(side):
    frame = history()
    quote(frame, 61, 101)
    frame.loc[BASE + pd.Timedelta(seconds=62):BASE + pd.Timedelta(seconds=69), "quote"] = 101.
    quote(frame, 70, 97 if side == 1 else 105)
    quote(frame, 71, 95 if side == 1 else 107)
    quote(frame, 72, 80 if side == 1 else 140)  # Further adverse move after exit.
    trades, audit = run(frame, issued=[signal(side=side, atr=2.)])
    trade = trades.iloc[0]
    assert trade.entry == 101 and trade.atr == 2
    assert trade.trigger_time == BASE + pd.Timedelta(seconds=70)
    assert trade.trigger_quote == (97 if side == 1 else 105)
    assert trade.exit_time == BASE + pd.Timedelta(seconds=71)
    assert trade.exit == (95 if side == 1 else 107)
    assert trade.gross_R == -1.5 and trade.net_R == pytest.approx(-1.55)
    assert trade.reason == "sl" and not trade.censored and trade.side == side
    assert audit["filled"] == audit["completed"] == 1


@pytest.mark.parametrize("side", [1, -1])
def test_recovery_after_trigger_is_used_and_never_cancels_or_clamps_exit(side):
    frame = history()
    quote(frame, 70, 90 if side == 1 else 110)
    quote(frame, 71, 103 if side == 1 else 97)
    trades, _ = run(frame, issued=[signal(side=side)])
    trade = trades.iloc[0]
    assert trade.reason == "sl"  # Trigger persists despite favorable next quote.
    assert trade.gross_R == 1.5 and trade.net_R == pytest.approx(1.45)
    assert trade.exit_time == BASE + pd.Timedelta(seconds=71)


@pytest.mark.parametrize("side", [1, -1])
def test_zero_stop_latency_is_named_trigger_sensitivity_not_primary(side):
    frame = history()
    quote(frame, 70, 95 if side == 1 else 105)
    quote(frame, 71, 94 if side == 1 else 106)
    primary, _ = run(frame, issued=[signal(side=side)])
    sensitivity, audit = run(frame, issued=[signal(side=side)], config=cfg(stop_latency_ticks=0))
    assert primary.iloc[0].gross_R == -3
    assert sensitivity.iloc[0].gross_R == -2.5
    assert sensitivity.iloc[0].exit_time == sensitivity.iloc[0].trigger_time == BASE + pd.Timedelta(seconds=70)
    assert audit["stop_rule"] == "optimistic_trigger_quote_sensitivity"


@pytest.mark.parametrize("side", [1, -1])
def test_trigger_exactly_at_nominal_expiry_has_stop_priority(side):
    frame = history()
    quote(frame, 120, 98 if side == 1 else 102)
    quote(frame, 121, 99 if side == 1 else 101)
    trades, _ = run(frame, issued=[signal(side=side)], config=cfg(max_hold_minutes=1))
    trade = trades.iloc[0]
    assert trade.planned_end == trade.trigger_time == BASE + pd.Timedelta(seconds=120)
    assert trade.reason == "sl" and trade.exit_time == BASE + pd.Timedelta(seconds=121)
    assert trade.gross_R == -.5


def test_expiry_uses_first_strictly_after_quote_not_terminal_before_or_at_price():
    frame = history()
    quote(frame, 119, 110)
    quote(frame, 120, 112)
    quote(frame, 121, 114)
    quote(frame, 122, 50)
    trades, _ = run(frame, config=cfg(max_hold_minutes=1))
    trade = trades.iloc[0]
    assert trade.reason == "time" and trade.exit == 114 and trade.gross_R == 7
    assert trade.exit_time == BASE + pd.Timedelta(seconds=121)
    assert trade.holding_minutes == 1


@pytest.mark.parametrize("side", [1, -1])
def test_favorable_jump_is_uncapped_and_does_not_exit_before_timeout(side):
    frame = history()
    frame.loc[BASE + pd.Timedelta(seconds=70):, "quote"] = 120 if side == 1 else 80
    trades, _ = run(frame, issued=[signal(side=side)], config=cfg(max_hold_minutes=1))
    assert trades.iloc[0].reason == "time" and trades.iloc[0].gross_R == 10


def test_missing_second_before_later_crossing_remains_unknown():
    frame = history()
    frame = frame.drop(BASE + pd.Timedelta(seconds=70))
    quote(frame, 80, 90)
    trades, audit = run(frame)
    trade = trades.iloc[0]
    assert trade.censored and trade.reason == "censored_missing_path"
    assert trade.missing_time == BASE + pd.Timedelta(seconds=70)
    assert pd.isna(trade.trigger_time) and np.isnan(trade.exit)
    assert np.isnan(trade.gross_R) and np.isnan(trade.net_R)
    assert trade.exit_time == trade.planned_end
    assert audit["censored"] == audit["missing_path"] == 1


def test_missing_stop_successor_censors_and_preserves_known_trigger():
    frame = history()
    quote(frame, 70, 95)
    frame = frame.drop(BASE + pd.Timedelta(seconds=71))
    trades, audit = run(frame)
    trade = trades.iloc[0]
    assert trade.censored and trade.trigger_quote == 95
    assert trade.trigger_time == BASE + pd.Timedelta(seconds=70)
    assert trade.missing_time == BASE + pd.Timedelta(seconds=71)
    assert np.isnan(trade.net_R) and audit["completed"] == 0
    sensitivity, _ = run(frame, config=cfg(stop_latency_ticks=0))
    assert not sensitivity.iloc[0].censored and sensitivity.iloc[0].gross_R == -2.5


@pytest.mark.parametrize("missing", [120, 121])
def test_missing_deadline_or_timeout_quote_censors_without_stale_fallback(missing):
    frame = history().drop(BASE + pd.Timedelta(seconds=missing))
    trades, _ = run(frame, config=cfg(max_hold_minutes=1))
    assert trades.iloc[0].censored and np.isnan(trades.iloc[0].net_R)
    assert trades.iloc[0].missing_time == BASE + pd.Timedelta(seconds=missing)


def test_missing_after_completed_exit_and_future_prices_do_not_change_trade():
    frame = history()
    quote(frame, 70, 98)
    quote(frame, 71, 97)
    full, _ = run(frame)
    prefix, _ = run(frame.loc[:BASE + pd.Timedelta(seconds=71)])
    changed = frame.drop(BASE + pd.Timedelta(seconds=72)).copy()
    changed.loc[BASE + pd.Timedelta(seconds=73):, "quote"] = 10000.
    future_changed, _ = run(changed)
    pd.testing.assert_frame_equal(full, prefix)
    pd.testing.assert_frame_equal(full, future_changed)
    assert not full.iloc[0].censored


def test_no_usable_successor_at_end_of_source_is_censored_not_timeout_marked():
    frame = history(n=121)
    trades, _ = run(frame, config=cfg(max_hold_minutes=1))
    assert trades.iloc[0].censored
    assert trades.iloc[0].missing_time == BASE + pd.Timedelta(seconds=121)
    assert np.isnan(trades.iloc[0].exit)


def test_first_quote_one_second_after_minute_is_valid_without_exact_open_quote():
    frame = history(start=BASE + pd.Timedelta(seconds=61))
    trades, audit = run(frame)
    assert trades.iloc[0].entry_time == BASE + pd.Timedelta(seconds=61)
    assert not trades.iloc[0].censored and audit["missing_entry"] == 0


def test_missing_entry_reserves_planned_occupancy_and_does_not_shift_entry():
    frame = history().drop(BASE + pd.Timedelta(seconds=61))
    issued = [signal(0), signal(1), signal(16)]
    trades, audit = run(frame, issued=issued)
    assert audit["missing_entry"] == audit["overlap_skipped"] == 1
    assert len(trades) == 1 and trades.iloc[0].signal_time == BASE + pd.Timedelta(minutes=16)
    assert trades.iloc[0].entry_time == BASE + pd.Timedelta(minutes=17, seconds=1)


def test_censored_path_reserves_planned_end_and_actual_exit_controls_completed_occupancy():
    frame = history().drop(BASE + pd.Timedelta(seconds=70))
    trades, audit = run(frame, issued=[signal(0), signal(1), signal(16)])
    assert len(trades) == 2 and trades.iloc[0].censored
    assert audit["censored"] == audit["overlap_skipped"] == 1
    completed, audit = run(history(), issued=[signal(0), signal(15), signal(16)])
    assert completed.signal_time.tolist() == [BASE, BASE + pd.Timedelta(minutes=16)]
    assert audit["overlap_skipped"] == 1


def test_stable_issuance_order_and_no_extra_cooldown_after_early_stop():
    frame = history()
    quote(frame, 70, 98)
    quote(frame, 71, 97)
    trades, audit = run(frame, issued=[signal(2, variant="LATER"), signal(0, variant="FIRST"), signal(0, variant="DUPLICATE")])
    assert trades.variant.tolist() == ["FIRST", "LATER"]
    assert audit["overlap_skipped"] == 1
    assert trades.iloc[0].exit_time < trades.iloc[1].nominal_entry_time


def test_actual_position_can_be_reentered_when_new_nominal_order_equals_last_exit():
    frame = history()
    quote(frame, 119, 98)
    quote(frame, 120, 97)
    trades, audit = run(frame, issued=[signal(0), signal(1)])
    assert len(trades) == 2 and audit["overlap_skipped"] == 0
    assert trades.iloc[0].exit_time == trades.iloc[1].nominal_entry_time
    assert trades.iloc[1].entry_time > trades.iloc[0].exit_time


def test_planned31min_purge_is_outcome_independent_and_exit_allowance_is_guarded():
    frame = history()
    quote(frame, 62, 98)
    quote(frame, 63, 97)
    trades, audit = run(frame, end=BASE + pd.Timedelta(minutes=30))
    assert trades.empty and audit["purged"] == 1
    trades, audit = run(frame, config=cfg(max_hold_minutes=30), end=BASE + pd.Timedelta(minutes=31))
    assert trades.empty and audit["purged"] == 1  # Even though an early stop is known.
    trades, audit = run(frame, config=cfg(max_hold_minutes=30), end=BASE + pd.Timedelta(minutes=31, seconds=1))
    assert len(trades) == 1 and audit["purged"] == 0
    assert audit["partition_exit_allowance_seconds"] == 1


def test_fractional_partition_boundaries_filter_signals_before_price_paths():
    trades, audit = run(history(), issued=[signal(0), signal(1), signal(40), signal(59)],
                        start=BASE + pd.Timedelta(seconds=18), end=BASE + pd.Timedelta(minutes=59, seconds=42))
    assert trades.signal_time.tolist() == [BASE + pd.Timedelta(minutes=1)]
    assert audit["outside_partition"] == 1 and audit["purged"] == 2


def test_initial_stop_risk_and_total_cost_are_applied_once():
    frame = history()
    quote(frame, 121, 104)
    trades, _ = run(frame, issued=[signal(atr=2.)], config=cfg(max_hold_minutes=1))
    assert trades.iloc[0].gross_R == 1
    assert trades.iloc[0].net_R == pytest.approx(.95)


def test_generic_declared_two_second_tolerance_does_not_change_primary_default():
    frame = history(n=1800, freq="2s")
    missing, audit = run(frame)
    assert missing.empty and audit["missing_entry"] == 1
    trades, audit = run(frame, config=cfg(max_gap_seconds=2))
    assert not trades.iloc[0].censored
    assert trades.iloc[0].entry_time == BASE + pd.Timedelta(seconds=62)
    assert trades.iloc[0].exit_time == BASE + pd.Timedelta(seconds=962)
    assert audit["config"]["max_gap_seconds"] == 2 and TickExitConfig().max_gap_seconds == 1


def test_empty_quote_source_returns_missing_entry_audit_and_typed_empty_ledger():
    frame = pd.DataFrame({"quote": pd.Series(dtype=float)}, index=pd.DatetimeIndex([], tz="UTC"))
    trades, audit = run(frame)
    assert trades.empty and list(trades.columns) == TICK_TRADE_COLUMNS
    assert audit["issued"] == audit["missing_entry"] == 1
    assert audit["filled"] == audit["completed"] == audit["censored"] == 0


@pytest.mark.parametrize("name", SAFETY_FLAGS)
def test_every_live_flag_blocks_even_empty_replay(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(RuntimeError, match="Safety invariant"):
        run(history(), issued=[])


@pytest.mark.parametrize("change", [
    {"stop_atr": 0}, {"stop_atr": -1}, {"stop_atr": np.inf}, {"stop_atr": np.nan},
    {"stop_atr": True}, {"stop_atr": "2"},
    {"max_hold_minutes": 0}, {"max_hold_minutes": 1.5}, {"max_hold_minutes": True},
    {"entry_delay_minutes": -1}, {"entry_delay_minutes": False}, {"entry_delay_minutes": 1.0},
    {"round_trip_cost_atr": -.1}, {"round_trip_cost_atr": np.inf}, {"round_trip_cost_atr": True},
    {"max_gap_seconds": 0}, {"max_gap_seconds": 1.5}, {"max_gap_seconds": True},
    {"stop_latency_ticks": -1}, {"stop_latency_ticks": 2}, {"stop_latency_ticks": False}, {"stop_latency_ticks": 1.0},
])
def test_invalid_config_values_are_refused(change):
    with pytest.raises(ValueError):
        cfg(**change)


@pytest.mark.parametrize("change", [
    {"atr": 0}, {"atr": np.inf}, {"atr": np.nan}, {"atr": True}, {"atr": "1"},
    {"side": True}, {"side": 0}, {"side": 1.0}, {"side": 2},
    {"signal_time": pd.Timestamp("2026-01-01")},
    {"signal_time": BASE + pd.Timedelta(seconds=1)}, {"signal_time": pd.NaT}, {"variant": 3},
])
def test_invalid_signal_values_are_refused(change):
    with pytest.raises(ValueError):
        run(history(), issued=[signal(**change)])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0., -1.])
def test_nonfinite_or_nonpositive_quotes_are_refused(value):
    frame = history()
    quote(frame, 61, value)
    with pytest.raises(ValueError, match="finite and positive"):
        run(frame)


@pytest.mark.parametrize("kind", ["boolean", "textual", "complex", "duplicate_column", "duplicate", "unsorted", "naive", "nat", "fractional_second"])
def test_quote_types_and_timestamp_order_are_strictly_validated(kind):
    frame = history()
    if kind == "boolean":
        frame["quote"] = True
    elif kind == "textual":
        frame["quote"] = "100"
    elif kind == "complex":
        frame["quote"] = 100. + 5.j
    elif kind == "duplicate_column":
        frame = pd.concat([frame, frame], axis=1)
    elif kind == "duplicate":
        frame = pd.concat([frame.iloc[:5], frame.iloc[4:]])
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "naive":
        frame.index = frame.index.tz_localize(None)
    elif kind == "nat":
        frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]], tz="UTC")
    else:
        frame.index = frame.index + pd.Timedelta(milliseconds=500)
    with pytest.raises(ValueError):
        run(frame)


def test_nonpositive_stop_and_nonfinite_risk_are_refused():
    frame = history()
    frame["quote"] = 1.
    with pytest.raises(ValueError, match="Stop risk"):
        run(frame)
    with pytest.raises(ValueError, match="Stop risk"):
        run(history(), issued=[signal(atr=1e308)], config=cfg(stop_atr=2.))


def test_invalid_container_config_and_partition_arguments_are_refused():
    with pytest.raises(TypeError, match="TickExitConfig"):
        run(history(), config=object())
    with pytest.raises(TypeError, match="list"):
        run(history(), issued=())
    with pytest.raises(ValueError, match="requires"):
        run(history(), issued=[{}])
    with pytest.raises(ValueError, match="DataFrame"):
        run(pd.DataFrame({"wrong": [1.]}))
    for purge in (-1, 1.5, True):
        with pytest.raises(ValueError, match="purge_minutes"):
            run(history(), purge_minutes=purge)
    with pytest.raises(ValueError, match="start must be before"):
        run(history(), start=BASE, end=BASE)
    with pytest.raises(ValueError, match="timezone-aware"):
        run(history(), start=BASE.tz_localize(None))


def test_timezone_conversion_normalizes_tick_and_signal_outputs_to_utc():
    frame = history()
    frame.index = frame.index.tz_convert("Asia/Riyadh")
    trades, _ = run(frame, issued=[signal(signal_time=BASE.tz_convert("Asia/Riyadh"))])
    assert trades.iloc[0].entry_time == BASE + pd.Timedelta(seconds=61)
    assert str(trades.entry_time.dtype) == "datetime64[ns, UTC]"


def test_full_audit_identity_covers_missing_filled_overlap_purged_and_outside():
    frame = history().drop(BASE + pd.Timedelta(seconds=61))
    signals = [signal(-1), signal(0), signal(1), signal(16), signal(50)]
    trades, audit = run(frame, issued=signals)
    assert len(trades) == audit["filled"] == audit["completed"] + audit["censored"]
    assert audit["issued"] == audit["filled"] + audit["missing_entry"] + audit["outside_partition"] + audit["purged"] + audit["overlap_skipped"]
    assert audit["missing_entry"] == audit["outside_partition"] == audit["purged"] == audit["overlap_skipped"] == 1
    assert all(value is False for value in audit["safety"].values())
