"""Synthetic quote-path scenarios; no strategy, market history or performance claims."""
import math

import pytest

from independent_profit_research import ExecutionConfig, Quote, simulate_trade


def q(time, bid=100.0, ask=101.0):
    return Quote(time, bid, ask)


def run(quotes=None, **kwargs):
    return simulate_trade(
        [q(0), q(250), q(1000), q(1250)] if quotes is None else quotes,
        decision_ms=kwargs.pop("decision_ms", 0),
        direction=kwargs.pop("direction", 1),
        horizon_end_ms=kwargs.pop("horizon_end_ms", 1000),
        stop_distance=kwargs.pop("stop_distance", 10.0),
        **kwargs,
    )


def test_flat_buy_pays_spread_and_is_not_cash_profit():
    result = run()
    assert result.status == "COMPLETE"
    assert result.entry_price == 101 and result.exit_price == 100
    assert result.signed_move_after_spread_and_slippage == -1
    assert result.costs_verified is result.execution_verified is False
    assert result.cash_pnl is None and result.move_units == "QUOTE_PRICE_UNITS_ONLY"


def test_flat_sell_also_pays_spread():
    assert run(direction=-1).signed_move_after_spread_and_slippage == -1


def test_extra_adverse_slippage_is_charged_on_both_legs():
    result = run(config=ExecutionConfig(adverse_slippage_per_leg=0.2))
    assert result.entry_price == pytest.approx(101.2)
    assert result.exit_price == pytest.approx(99.8)
    assert result.signed_move_after_spread_and_slippage == pytest.approx(-1.4)
    assert result.fees_model == "NOT_INCLUDED_NOT_VERIFIED"


def test_default_entry_uses_250ms_deadline_not_decision_quote():
    result = run([q(0, 1, 2), q(249, 2, 3), q(300, 100, 101), q(1000), q(1250)])
    assert result.entry_time_msc == 300 and result.entry_quote_price == 101


def test_exit_latency_can_turn_target_trigger_into_adverse_move():
    result = run([q(0), q(250), q(500, 104, 105), q(600, 102, 103), q(850, 99, 100)],
                 target_distance=2)
    assert result.trigger_reason == "TARGET" and result.trigger_time_msc == 600
    assert result.trigger_source_time_msc == 500
    assert result.exit_time_msc == 850 and result.exit_price == 99
    assert result.signed_move_after_spread_and_slippage == -2


def test_stop_jump_is_never_clipped_to_stop_distance():
    result = run([q(0), q(250), q(500, 80, 81), q(750, 70, 71), q(1000, 60, 61)],
                 stop_distance=2)
    assert result.trigger_reason == "STOP" and result.exit_price == 60
    assert result.signed_move_after_spread_and_slippage == -41


def test_both_barriers_in_same_millisecond_bundle_choose_stop():
    result = run([q(0), q(250), q(500, 98, 99), q(500, 104, 105), q(750, 97, 98), q(1000, 96, 97)],
                 stop_distance=2, target_distance=2)
    assert result.both_barriers_in_bundle is True
    assert result.reason == "STOP_PRIORITY_SAME_MILLISECOND"
    assert result.trigger_reason == "STOP"


def test_atomic_entry_uses_worst_ask_and_preserves_duplicates_and_order():
    bundle = [q(250, 100, 101), q(250, 99, 105), q(250, 99, 105)]
    quotes = [q(0), *bundle, q(1000, 105, 106), q(1250, 106, 107)]
    original = quotes.copy()
    result = run(quotes)
    assert result.entry_price == 105 and result.entry_bundle == tuple(bundle)
    assert quotes == original
    reversed_bundle = run([q(0), *reversed(bundle), q(1000, 105, 106), q(1250, 106, 107)])
    assert result.signed_move_after_spread_and_slippage == reversed_bundle.signed_move_after_spread_and_slippage


def test_sell_entry_worst_bid_and_exit_worst_ask():
    result = run([q(0), q(250, 100, 101), q(250, 98, 99), q(1000, 97, 98),
                  q(1250, 97, 98), q(1250, 96, 100)], direction=-1)
    assert result.entry_quote_price == 98 and result.exit_quote_price == 100
    assert result.signed_move_after_spread_and_slippage == -2


def test_missing_entry_is_unknown_without_invented_move():
    result = run([q(0), q(100)])
    assert result.status == "UNKNOWN" and result.reason == "MISSING_ENTRY_QUOTE"
    assert result.entry_price is result.signed_move_after_spread_and_slippage is None


def test_latency_reaching_horizon_skips_exposure():
    result = run(horizon_end_ms=200)
    assert result.status == "SKIP" and result.entry_price is None


def test_first_eligible_entry_after_horizon_is_not_late_fill():
    result = run([q(0), q(1100), q(1400)])
    assert result.status == "SKIP" and result.reason == "NO_ENTRY_BEFORE_HORIZON"


def test_missing_horizon_exit_never_uses_last_available_quote():
    result = run([q(0), q(250), q(1000)])
    assert result.status == "UNKNOWN" and result.reason == "MISSING_HORIZON_EXIT_QUOTE"
    assert result.exit_price is result.signed_move_after_spread_and_slippage is None


def test_missing_quote_after_stop_latency_is_unknown():
    result = run([q(0), q(250), q(500, 80, 81), q(750, 70, 71)], stop_distance=2)
    assert result.reason == "MISSING_EXIT_QUOTE"
    assert result.trigger_reason == "STOP" and result.exit_price is None


def test_gap_before_entry_is_unknown():
    result = run([q(0), q(6000), q(10000), q(10250)], horizon_end_ms=10000)
    assert result.reason == "GAP_BEFORE_ENTRY" and result.status == "UNKNOWN"


def test_gap_during_hold_is_unknown_even_if_later_quote_crosses_stop():
    result = run([q(0), q(250), q(6000, 80, 81), q(6250, 70, 71)],
                 horizon_end_ms=10000, stop_distance=2)
    assert result.reason == "GAP_DURING_HOLD" and result.trigger_reason is None


def test_gap_before_horizon_exit_is_unknown_and_exact_gap_limit_is_allowed():
    late = run([q(0), q(250), q(1000), q(7000)])
    assert late.reason == "GAP_BEFORE_EXIT" and late.exit_price is None
    boundary = run([q(0), q(250), q(5250), q(5500)], horizon_end_ms=5250)
    assert boundary.status == "COMPLETE"


def test_observations_during_long_entry_latency_can_keep_path_known():
    result = run([q(0), q(4000), q(6000), q(6250), q(6500)], horizon_end_ms=6250,
                 config=ExecutionConfig(entry_latency_ms=6000))
    assert result.status == "COMPLETE" and result.entry_time_msc == 6000


def test_zero_latency_spread_stop_still_waits_for_bundle_confirmation():
    result = run([q(0), q(100)], stop_distance=0.5,
                 config=ExecutionConfig(entry_latency_ms=0, exit_latency_ms=0))
    assert result.reason == "STOP" and result.exit_time_msc == 100
    assert result.trigger_source_time_msc == 0 and result.trigger_time_msc == 100
    assert result.signed_move_after_spread_and_slippage == -1


def test_future_suffix_does_not_change_completed_stop_outcome():
    prefix = [q(0), q(250), q(500, 80, 81), q(750, 70, 71), q(1000, 60, 61)]
    assert run(prefix, stop_distance=2) == run(prefix + [q(100000, 1, 2)], stop_distance=2)


def test_old_prefix_cannot_replace_explicit_issue_time_or_horizon():
    result = run([q(0, 500, 501), q(1000, 200, 201), q(1250), q(2000), q(2250)],
                 decision_ms=1000, horizon_end_ms=2000)
    assert result.decision_ms == 1000 and result.entry_time_msc == 1250
    assert result.trigger_time_msc == 2000 and result.exit_time_msc == 2250


def test_barrier_confirmation_is_not_backdated_to_source_bundle():
    result = run([q(0), q(250), q(500, 98, 99), q(600, 97, 98), q(850, 96, 97)],
                 stop_distance=2)
    assert result.trigger_source_time_msc == 500
    assert result.trigger_time_msc == 600
    assert result.exit_time_msc == 850


def test_horizon_wins_when_barrier_bundle_closes_at_or_after_horizon():
    for confirmation_time in (1000, 1100):
        result = run([q(0), q(250), q(500, 80, 81), q(confirmation_time, 75, 76),
                      q(1250, 70, 71)], stop_distance=2)
        assert result.trigger_reason == "HORIZON"
        assert result.trigger_time_msc == 1000 and result.trigger_source_time_msc is None
        assert result.exit_time_msc == 1250


def test_invalid_prices_and_nonfinite_values_are_rejected():
    for bid, ask in [(0, 1), (2, 1), (math.nan, 1), (1, math.inf), (True, 2)]:
        with pytest.raises(ValueError):
            q(0, bid, ask)
    with pytest.raises(ValueError):
        Quote(True, 1, 2)


def test_late_out_of_order_row_is_rejected_without_sorting():
    with pytest.raises(ValueError, match="nondecreasing"):
        run([q(0), q(250), q(1000), q(1250), q(1200)])


def test_invalid_config_direction_and_distances_are_rejected():
    for kwargs in [{"entry_latency_ms": -1}, {"exit_latency_ms": True}, {"max_gap_ms": 0},
                   {"adverse_slippage_per_leg": -0.1}, {"adverse_slippage_per_leg": math.inf}]:
        with pytest.raises(ValueError):
            ExecutionConfig(**kwargs)
    for kwargs in [{"direction": True}, {"direction": 0}, {"stop_distance": 0},
                   {"target_distance": math.nan}, {"horizon_end_ms": 0}]:
        with pytest.raises(ValueError):
            run(**kwargs)
@pytest.mark.parametrize('direction', [1, -1])
def test_adverse_slippage_cannot_produce_nonpositive_effective_prices(direction):
    rows = [Quote(t, 1, 2) for t in range(5)]
    with pytest.raises(ValueError):
        simulate_trade(rows, 0, direction, 3, 10,
                       config=ExecutionConfig(0, 0, 5000, 2))

