"""Artificial tapes and supplied signals; no raw files, fitting or measurements."""
from dataclasses import replace
import json
import math

import pytest

from independent_multiframe_research.runner import evaluate_signals
from independent_profit_research.execution import ExecutionConfig as C, Quote as Q
from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape


S = 10**8
DAY = 86400000
T = 1800000
END = 7200000
POLICY = 'NEW_FIXED_MTF_POLICY'


def tape(points):
    return QuoteTape(Q(t, bid*S, ask*S) for t, bid, ask in points)


def signal(decision=T, horizon=None, stop=10, direction=1, policy=POLICY):
    return Signal(policy, direction, decision, decision+2000 if horizon is None else horizon, stop*S, None)


def run(rows, signals, *, config=None, start=0, end=END, baseline=False, policy=POLICY):
    return evaluate_signals(rows, signals, start, end, S, C(0, 0) if config is None else config,
                            policy_id=policy, baseline=baseline)


def test_new_policy_identifier_and_list_iterable_are_supported():
    result = run(tape([(T, 100, 101), (T+2000, 102, 103)]), [signal()])
    assert result['policy_id'] == POLICY
    assert result['summary']['completed'] == 1
    assert result['ledger'][0]['R'] == pytest.approx(.1)


def test_entry_exit_delay_worst_bundle_and_assumed_slippage_are_frozen_kernel():
    rows = tape([(T, 50, 51), (T+250, 100, 101), (T+250, 101, 102),
                 (T+1000, 110, 111), (T+1250, 109, 110), (T+1250, 108, 109)])
    result = run(rows, [signal(T, T+1000, stop=20)], config=C(250, 250, 5000, S))
    row = result['ledger'][0]
    assert row['entry_time_msc'] == T+250 and row['entry_price'] == 103
    assert row['exit_time_msc'] == T+1250 and row['exit_price'] == 107
    assert row['R'] == pytest.approx(.2)
    assert len(row['entry_bundle']) == len(row['exit_bundle']) == 2


def test_reservation_covers_pending_entry_hold_pending_exit_and_inclusive_exit():
    signals = [signal(T, T+1000), signal(T+100), signal(T+700), signal(T+1100),
               signal(T+1250), signal(T+1500, T+2500)]
    rows = tape((t, 100, 101) for t in (T, T+100, T+250, T+700, T+1000,
                                       T+1100, T+1250, T+1500, T+1750, T+2500, T+2750))
    result = run(rows, signals, config=C())
    assert [r['decision_ms'] for r in result['ledger']] == [T, T+1500]
    assert result['summary']['blocked_overlap'] == 4


def test_skip_has_no_exposure_and_allows_equal_decision_next_signal():
    signals = [signal(T, T+1000), signal(T, T+3000)]
    rows = tape([(T, 100, 101), (T+1000, 100, 101), (T+3000, 102, 103)])
    result = run(rows, signals, config=C(1000, 0))
    assert [r['status'] for r in result['ledger']] == ['SKIP', 'COMPLETE']
    assert result['summary']['blocked_overlap'] == 0


def test_unknown_closes_generator_without_consuming_remaining_stage():
    closed = []
    def issued():
        try:
            yield signal()
            raise AssertionError('Censored suffix was consumed')
        finally:
            closed.append(True)
    result = run(tape([(T, 100, 101), (T+500, 100, 101)]), issued())
    assert closed == [True]
    assert result['summary']['unknown_and_remaining_censored'] is True
    assert result['summary']['remaining_opportunities_count'] is None
    for field in ('sum_R', 'sum_quote', 'pf', 'marked_drawdown_pct', 'ending_equity_index'):
        assert result['summary'][field] is None
    assert result['daily'][0]['sum_R'] is None
    assert result['ledger'][0]['R'] is None


def test_unknown_preserves_only_descriptive_completed_prefix():
    rows = tape([(T, 100, 101), (T+1000, 102, 103), (T+2000, 100, 101)])
    result = run(rows, [signal(T, T+1000), signal(T+2000, T+4000)])
    assert result['summary']['completed'] == 1 and result['summary']['unknown'] == 1
    assert result['summary']['observed_sum_R'] == pytest.approx(.1)
    assert result['summary']['sum_R'] is None
    assert result['daily'][0]['observed_sum_R'] == pytest.approx(.1)
    assert all(day['sum_R'] is None and day['sum_quote'] is None for day in result['daily'])


def test_idle_days_have_explicit_zero_and_no_cash_or_profit_claim():
    result = run(tape([]), [], end=2*DAY)
    assert len(result['daily']) == 2
    assert all(day['sum_R'] == 0 and day['cash_net'] is None for day in result['daily'])
    assert result['summary']['completed'] == 0
    assert result['summary']['pf'] is None
    assert result['summary']['ending_equity_index'] == 10000
    assert result['cash_profit_tested'] is False


def test_flat_zero_spread_path_has_zero_r_undefined_pf_and_zero_drawdown():
    result = run(tape([(T, 100, 100), (T+2000, 100, 100)]), [signal()])
    assert result['summary']['sum_R'] == 0
    assert result['summary']['pf'] is None
    assert result['summary']['marked_drawdown_pct'] == 0
    assert result['summary']['pf_reason'] == 'NO_NONZERO_COMPLETED_MOVES'


def test_exit_utc_day_not_decision_day_receives_return():
    rows = tape([(DAY-1000, 100, 101), (DAY+1000, 102, 103)])
    result = run(rows, [signal(DAY-1000, DAY+1000)], start=DAY-2400000, end=DAY+2400000)
    assert [d['day_start_ms'] for d in result['daily']] == [0, DAY]
    assert [d['completed'] for d in result['daily']] == [0, 1]
    assert [d['sum_R'] for d in result['daily']] == pytest.approx([0, .1])


def test_daily_and_stage_are_exact_fsum_of_their_completed_ledger_rows():
    decisions = [T, T+4000, DAY+T]
    rows, signals = [], []
    for decision, last_bid in zip(decisions, (102, 100, 104)):
        rows.extend([(decision, 100, 101), (decision+1000, last_bid, last_bid+1)])
        signals.append(signal(decision, decision+1000))
    result = run(tape(rows), signals, end=2*DAY)
    ledger = result['ledger']
    assert result['summary']['observed_sum_R'] == math.fsum(r['R'] for r in ledger)
    for day in result['daily']:
        own = [r for r in ledger if day['day_start_ms'] <= r['exit_time_msc'] < day['day_start_ms']+DAY]
        assert day['observed_sum_R'] == math.fsum(r['R'] for r in own)
        assert day['observed_sum_quote'] == math.fsum(r['signed_move_after_spread_and_slippage'] for r in own)
    assert result['accounting_reconciliation']['summary_to_ledger_fsum_exact'] is True
    assert result['accounting_reconciliation']['each_day_to_exit_day_ledger_fsum_exact'] is True


def test_daily_rounding_residual_is_disclosed_without_changing_day_returns():
    signals = [Signal(POLICY, 1, T, T+1000, 1e-6, None),
               Signal(POLICY, 1, T+2000, T+3000, 1., None),
               Signal(POLICY, 1, DAY+T, DAY+T+1000, 1e-6, None)]
    rows = tape([(T, 100, 100), (T+1000, 200, 200),
                 (T+2000, 100, 100), (T+3000, 100+1/S, 100+1/S),
                 (DAY+T, 200, 200), (DAY+T+1000, 100, 100)])
    result = run(rows, signals, end=2*DAY)
    assert [r['R'] for r in result['ledger']] == [1e16, 1., -1e16]
    assert result['summary']['sum_R'] == 1
    assert [d['sum_R'] for d in result['daily']] == [1e16, -1e16]
    assert result['accounting_reconciliation']['daily_to_stage_fsum_residual']['observed_sum_R'] == -1


def test_embargo_includes_left_edge_and_excludes_right_edge_without_suffix_read():
    upper = END-T
    def issued():
        yield signal(T-1)
        yield signal(T)
        yield signal(upper, upper+1000)
        raise AssertionError('Excluded suffix was consumed')
    result = run(tape([(T, 100, 101), (T+2000, 102, 103)]), issued())
    assert [r['decision_ms'] for r in result['ledger']] == [T]
    assert result['summary']['excluded_before_embargo'] == 1
    assert result['summary']['excluded_after_embargo'] == 1
    assert result['unconsumed_signal_suffix_not_validated'] is True


def test_invalid_signal_is_rejected_even_when_its_decision_would_be_blocked():
    rows = tape([(T, 100, 101), (T+2000, 102, 103)])
    invalid = replace(signal(T+1), stop_distance=float('nan'))
    with pytest.raises(ValueError, match='stop_distance'):
        run(rows, [signal(), invalid])


def test_nondecreasing_order_is_checked_across_excluded_prefix():
    with pytest.raises(ValueError, match='nondecreasing'):
        run(tape([]), [signal(T-1), signal(T-2)])


@pytest.mark.parametrize('bad', [
    replace(signal(), direction=True), replace(signal(), direction=0),
    replace(signal(), decision_ms=True), replace(signal(), horizon_end_ms=T),
    replace(signal(), horizon_end_ms=END-250), replace(signal(), stop_distance=0),
    replace(signal(), target_distance=float('inf')), replace(signal(), policy_id='OTHER'),
])
def test_invalid_signal_contract_raises_without_silent_filtering(bad):
    with pytest.raises(ValueError):
        run(tape([]), [bad], config=C())


def test_baseline_is_a_label_for_supplied_clock_stream_and_not_regenerated():
    clock = signal(T+123, T+1123, policy='CLOCK_00_30')
    rows = tape([(T+123, 100, 101), (T+1123, 102, 103)])
    result = run(rows, [clock], policy='CLOCK_00_30', baseline=True)
    assert result['ledger'][0]['decision_ms'] == T+123
    assert result['ledger'][0]['baseline'] is True


def test_marked_drawdown_uses_intratrade_opposite_quotes_and_exit_slippage():
    result = run(tape([(T, 100, 101), (T+500, 110, 111),
                       (T+1000, 100, 101), (T+2000, 100, 101)]), [signal()],
                 config=C(0, 0, 5000, .5*S))
    assert result['summary']['marked_drawdown_pct'] > .2
    assert result['summary']['marked_quote_rows'] == 4
    assert result['ledger'][0]['entry_price'] == 101.5
    assert result['ledger'][0]['exit_price'] == 99.5
    assert result['equity_model'].endswith('NOT_CASH')


def test_future_quotes_after_complete_path_cannot_change_saved_outcome():
    points = [(T, 100, 101), (T+2000, 102, 103)]
    before = run(tape(points), [signal()])
    after = run(tape(points+[(T+5000, 50, 51)]), [signal()])
    assert before['ledger'] == after['ledger']
    assert before['summary'] == after['summary']


def test_stage_slice_never_passes_outside_quotes_to_execution(monkeypatch):
    seen = []
    reference = QuoteTape.simulate
    def inspect(self, *args, **kwargs):
        seen.extend(q.time_msc for q in self)
        return reference(self, *args, **kwargs)
    monkeypatch.setattr(QuoteTape, 'simulate', inspect)
    rows = tape([(0, 100, 101), (T+1000, 100, 101), (T+3000, 102, 103), (END, 300, 301)])
    run(rows, [signal(T+1000, T+3000)], start=1000)
    assert seen == [T+1000, T+3000]


def test_invalid_configuration_and_unsafe_value_types_are_rejected():
    args = [QuoteTape([]), [], 0, END, S, C()]
    for index, value in ((0, []), (1, None), (2, True), (3, 0), (4, float('nan')), (5, object())):
        bad = list(args); bad[index] = value
        with pytest.raises(ValueError):
            evaluate_signals(*bad, policy_id=POLICY)
    with pytest.raises(ValueError):
        evaluate_signals(*args, policy_id=POLICY, baseline=1)
    config = C(); object.__setattr__(config, 'max_gap_ms', 0)
    with pytest.raises(ValueError):
        run(tape([]), [], config=config)


def test_public_result_is_json_safe_and_keeps_all_eight_safety_flags_false():
    result = run(tape([(T, 100, 101), (T+2000, 102, 103)]), [signal()])
    assert len(result['safety']) == 8 and not any(result['safety'].values())
    assert all(len(day['safety']) == 8 and not any(day['safety'].values()) for day in result['daily'])
    assert result['summary']['qualified_for_profit_claim'] is False
    assert result['cost_verified'] is result['execution_verified'] is False
    assert result['ledger'][0]['cash_pnl'] is result['cash_net'] is None
    assert result['ledger'][0]['entry_quote_price'] == 101
    json.dumps(result, allow_nan=False)
