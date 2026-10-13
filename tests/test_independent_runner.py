"""Artificial quotes and issued decisions only; no evidence files or platform APIs."""
import json
import math

import pytest

from independent_profit_research import runner
from independent_profit_research.execution import ExecutionConfig as C, Quote as Q
from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape


S = 10**8
T = 1800000
END = 7200000


def tape(points):
    return QuoteTape(Q(t, bid*S, ask*S) for t, bid, ask in points)


def signal(decision=T, horizon=None, stop=10, direction=1):
    return Signal('SIDE_LEAD', direction, decision,
                  decision+2000 if horizon is None else horizon, stop*S, None)


def issue(monkeypatch, signals):
    def source(*args, **kwargs):
        yield from signals
    monkeypatch.setattr(runner, 'generate_signals', source)


def evaluate(rows, *, policy='SIDE_LEAD', baseline=False, config=None,
             start=0, end=END, unit=1, native=1):
    return runner.evaluate_tape(rows, policy, native, unit*S, start, end,
                                C(0, 0) if config is None else config, baseline)


def flat(start=T, end=T+40000, step=1000):
    return tape((t, 100, 101) for t in range(start, end+1, step))


def test_baseline_is_half_hour_utc_schedule_not_half_minute():
    rows = tape((t, 100, 101) for t in range(0, END, 1000))
    result = evaluate(rows, baseline=True)
    assert [x['decision_ms'] for x in result['ledger']] == [1800000, 3600000]
    assert result['baseline_clock']['period_ms'] == 1800000
    assert result['baseline_clock']['clock'] == 'UTC_MINUTE_00_AND_30'


def test_baseline_full_day_has_46_eligible_of_48_schedules():
    rows = tape((t, 100, 101) for t in range(0, runner.DAY_MS, 1000))
    result = evaluate(rows, baseline=True, end=runner.DAY_MS)
    assert result['summary']['completed'] == 46
    assert result['ledger'][0]['decision_ms'] == runner.EMBARGO_MS
    assert result['ledger'][-1]['decision_ms'] == runner.DAY_MS-2*runner.EMBARGO_MS
    summary, clock = result['summary'], result['baseline_clock']
    assert summary['excluded_before_embargo'] == summary['excluded_after_embargo'] == 1
    assert clock['scheduled'] == (summary['completed']+summary['skipped']+summary['unknown']+
                                   summary['blocked_overlap']+summary['excluded_before_embargo']+
                                   summary['excluded_after_embargo']+clock['missing_observation']+
                                   clock['stale_first_observation']) == 48


@pytest.mark.parametrize('policy,horizon,stop,target,direction', [
    ('SIDE_LEAD', 5000, 2., 2., 1),
    ('QUOTE_CLOCK_ACCELERATION', 30000, 2., 2., 1),
    ('POST_JUMP_CARRY_30S', 28000, 4., None, -1),
])
def test_baseline_matching_frozen_horizon_direction_and_barriers(policy, horizon, stop, target, direction):
    result = evaluate(flat(), policy=policy, baseline=True)
    row = result['ledger'][0]
    assert row['direction'] == direction
    assert row['horizon_end_ms']-row['decision_ms'] == horizon
    assert row['stop_distance'] == stop and row['target_distance'] == target
    assert row['decision_ms'] == T+(2000 if policy == 'POST_JUMP_CARRY_30S' else 0)


def test_baseline_horizon_uses_observed_decision_for_a_b():
    rows = flat(T+1000)
    row = evaluate(rows, baseline=True)['ledger'][0]
    assert row['decision_ms'] == T+1000
    assert row['horizon_end_ms'] == T+6000


def test_carry_baseline_horizon_stays_clock_based_after_delayed_due_observation():
    rows = flat(T+3500, T+40000, 500)
    row = evaluate(rows, policy='POST_JUMP_CARRY_30S', baseline=True)['ledger'][0]
    assert row['decision_ms'] == T+3500
    assert row['horizon_end_ms'] == T+30000


def test_stale_first_clock_observation_is_skipped_without_searching_later():
    result = evaluate(flat(T+2001), baseline=True)
    assert result['ledger'] == []
    # The stage's clock0 and the eligible clockT both see a stale first quote.
    assert result['baseline_clock']['stale_first_observation'] == 2
    assert result['summary']['sum_R'] == 0
    assert evaluate(flat(T+2000), baseline=True)['ledger'][0]['decision_ms'] == T+2000


def test_reservation_includes_pending_entry_pending_exit_and_exit_equality(monkeypatch):
    issue(monkeypatch, [signal(T, T+1000), signal(T+100), signal(T+1100),
                        signal(T+1250), signal(T+1500, T+2500)])
    rows = tape((t, 100, 101) for t in (T, T+100, T+250, T+500, T+1000,
                                       T+1100, T+1250, T+1500, T+1750, T+2500, T+2750))
    result = evaluate(rows, config=C())
    assert [x['decision_ms'] for x in result['ledger']] == [T, T+1500]
    assert result['summary']['completed'] == 2
    assert result['summary']['blocked_overlap'] == 3


def test_skip_without_exposure_does_not_reserve_next_decision(monkeypatch):
    issue(monkeypatch, [signal(T, T+500), signal(T+1000, T+2000)])
    result = evaluate(tape([(T+1000, 100, 101), (T+2000, 102, 103)]))
    assert [x['status'] for x in result['ledger']] == ['SKIP', 'COMPLETE']
    assert result['summary']['skipped'] == 1
    assert result['summary']['blocked_overlap'] == 0


def test_unknown_stops_consuming_future_signals_and_never_zero_fills_them(monkeypatch):
    def source(*args, **kwargs):
        yield signal()
        raise AssertionError('Future censored signal must not be requested')
    monkeypatch.setattr(runner, 'generate_signals', source)
    result = evaluate(tape([(T, 100, 101), (T+500, 100, 101)]))
    summary = result['summary']
    assert summary['unknown'] == 1 and summary['completed'] == 0
    assert summary['sum_R'] is None and summary['marked_drawdown_pct'] is None
    assert summary['remaining_censored'] is True
    assert summary['remaining_opportunities_count'] is None
    assert summary['censored_from_ms'] == T
    assert result['daily'][0]['sum_R'] is None
    assert result['ledger'][0]['R'] is None


def test_completed_prefix_retained_but_primary_null_after_unknown(monkeypatch):
    issue(monkeypatch, [signal(T, T+1000), signal(T+2000, T+4000)])
    result = evaluate(tape([(T, 100, 101), (T+1000, 102, 103),
                            (T+2000, 100, 101), (T+2500, 100, 101)]))
    assert result['summary']['completed'] == 1
    assert result['summary']['observed_sum_R'] == pytest.approx(.1)
    assert result['summary']['sum_R'] is None and result['summary']['pf'] is None
    assert result['daily'][0]['observed_sum_R'] == pytest.approx(.1)
    assert result['daily'][0]['sum_quote'] is None


def test_split_embargo_is_inclusive_left_exclusive_right(monkeypatch):
    upper = END-runner.EMBARGO_MS
    issue(monkeypatch, [signal(T-1), signal(T), signal(upper-1), signal(upper)])
    rows = tape([(T-1, 100, 101), (T, 100, 101), (T+2000, 100, 101),
                 (upper-1, 100, 101), (upper+1999, 100, 101)])
    result = evaluate(rows)
    assert [x['decision_ms'] for x in result['ledger']] == [T, upper-1]
    assert result['summary']['excluded_before_embargo'] == 1
    assert result['summary']['excluded_after_embargo'] == 1


def test_daily_compensated_totals_reconcile_with_ledger_under_cancellation(monkeypatch):
    # Legitimate synthetic execution paths isolate accumulation from strategy
    # logic: large gain, tiny gain, large loss, all exiting on the same UTC day.
    issue(monkeypatch, [signal(T, T+1000, stop=1/S),
                        signal(T+2000, T+3000, stop=1/S),
                        signal(T+4000, T+5000, stop=1/S)])
    entry, large = 100*S, 1e16
    rows = QuoteTape([Q(T, entry, entry), Q(T+1000, entry+large, entry+large),
                      Q(T+2000, entry, entry), Q(T+3000, entry+1, entry+1),
                      Q(T+4000, entry+large, entry+large), Q(T+5000, entry, entry)])
    result = evaluate(rows)
    ledger, day, summary = result['ledger'], result['daily'][0], result['summary']
    values = [row['R'] for row in ledger]
    assert values == [large, 1., -large]
    expected = {'observed_sum_R': math.fsum(values),
                'observed_sum_quote': math.fsum(row['signed_move_after_spread_and_slippage'] for row in ledger),
                'observed_gain_R': math.fsum(max(0., value) for value in values),
                'observed_loss_R': math.fsum(max(0., -value) for value in values)}
    assert expected['observed_sum_R'] == 1 and expected['observed_sum_quote'] == 1e-8
    for field, value in expected.items():
        assert day[field] == summary[field] == value
    assert day['sum_R'] == summary['sum_R'] == 1
    assert day['sum_quote'] == summary['sum_quote'] == 1e-8
    assert summary['median_trade_R_observed'] == 1
    assert summary['median_holding_ms_observed'] == 1000
    json.dumps(result, allow_nan=False)


def test_no_cross_stage_warmup_is_used(monkeypatch):
    seen = []
    def source(quotes, **kwargs):
        seen.extend(q.time_msc for q in quotes)
        yield from ()
    monkeypatch.setattr(runner, 'generate_signals', source)
    evaluate(tape([(0, 100, 101), (1000, 100, 101), (END, 100, 101)]), start=1000)
    assert seen == [1000]


def test_quotes_after_stage_cannot_complete_unknown_exit(monkeypatch):
    issue(monkeypatch, [signal(T, END+1000)])
    rows = tape([(T, 100, 101), (T+1000, 100, 101), (END+1000, 102, 103)])
    result = evaluate(rows)
    assert result['summary']['unknown'] == 1
    assert result['ledger'][0]['exit_time_msc'] is None


def test_future_suffix_does_not_change_already_complete_ledger(monkeypatch):
    issue(monkeypatch, [signal(T, T+1000)])
    prefix = [(T, 100, 101), (T+1000, 102, 103)]
    before = evaluate(tape(prefix))
    after = evaluate(tape(prefix+[(T+5000, 50, 51), (T+10000, 500, 501)]))
    assert before['ledger'] == after['ledger']
    assert before['summary'] == after['summary']


def test_daily_groups_by_utc_exit_day_and_includes_idle_day(monkeypatch):
    day = runner.DAY_MS
    issue(monkeypatch, [signal(day-1000, day+1000)])
    result = evaluate(tape([(day-1000, 100, 101), (day+1000, 102, 103)]),
                      start=day-2400000, end=day+2400000)
    assert [x['day_start_ms'] for x in result['daily']] == [0, day]
    assert [x['completed'] for x in result['daily']] == [0, 1]
    assert [x['sum_R'] for x in result['daily']] == pytest.approx([0, .1])
    assert result['daily'][1]['observed_sum_quote'] == 1


def test_idle_days_are_explicit_zero_without_cash_claim(monkeypatch):
    issue(monkeypatch, [])
    result = evaluate(tape([]), end=2*runner.DAY_MS)
    assert len(result['daily']) == 2
    assert all(day['sum_R'] == 0 and day['cash_net'] is None for day in result['daily'])


def test_drawdown_marks_intratrade_opposite_side_not_only_realized_exit(monkeypatch):
    issue(monkeypatch, [signal()])
    result = evaluate(tape([(T, 100, 101), (T+500, 110, 111),
                            (T+1000, 100, 101), (T+2000, 100, 101)]))
    assert result['summary']['marked_drawdown_pct'] == pytest.approx(100*25/10022.5)
    assert result['summary']['ending_equity_index'] == pytest.approx(9997.5)
    assert result['summary']['marked_quote_rows'] == 4


def test_same_ms_marked_envelope_preserves_extremes_without_order_claim(monkeypatch):
    issue(monkeypatch, [signal(stop=50)])
    head, tail = [(T, 100, 101)], [(T+2000, 100, 101)]
    bundle = [(T+500, 110, 111), (T+500, 90, 91), (T+500, 90, 91)]
    a = evaluate(tape(head+bundle+tail))
    b = evaluate(tape(head+list(reversed(bundle))+tail))
    assert a['summary']['marked_drawdown_pct'] == b['summary']['marked_drawdown_pct']
    assert a['summary']['marked_drawdown_pct'] == pytest.approx(100*10/10004.5)
    assert a['summary']['marked_quote_rows'] == 5
    assert a['same_millisecond_order_known'] is False


def test_drawdown_includes_assumed_exit_slippage_and_both_spread_legs(monkeypatch):
    issue(monkeypatch, [signal()])
    result = evaluate(tape([(T, 100, 101), (T+2000, 110, 111)]),
                      config=C(0, 0, 5000, S))
    assert result['ledger'][0]['entry_price'] == 102
    assert result['ledger'][0]['exit_price'] == 109
    assert result['ledger'][0]['R'] == pytest.approx(.7)
    assert result['summary']['marked_drawdown_pct'] == pytest.approx(.075)


def test_sell_marks_ask_and_normalizes_signed_move(monkeypatch):
    issue(monkeypatch, [signal(direction=-1)])
    result = evaluate(tape([(T, 100, 101), (T+1000, 90, 91), (T+2000, 99, 100)]))
    assert result['ledger'][0]['entry_price'] == 100
    assert result['ledger'][0]['exit_price'] == 100
    assert result['ledger'][0]['R'] == 0
    assert result['summary']['marked_drawdown_pct'] > 0


def test_public_units_safety_and_json_never_claim_costs_or_cash(monkeypatch):
    issue(monkeypatch, [signal()])
    result = evaluate(tape([(T, 100, 101), (T+2000, 102, 103)]))
    row = result['ledger'][0]
    assert row['entry_quote_price'] == 101
    assert row['signed_move_after_spread_and_slippage'] == 1
    assert row['entry_bundle'][0]['bid'] == 100
    assert result['summary']['sum_R'] == pytest.approx(.1)
    assert result['cost_verified'] is result['execution_verified'] is False
    assert result['summary']['qualified_for_profit_claim'] is False
    assert len(result['safety']) == 8 and not any(result['safety'].values())
    assert row['cash_pnl'] is row['cash_net'] is result['cash_net'] is None
    assert 'QuoteTape' not in json.dumps(result, allow_nan=False)


def test_conditional_pf_and_zero_loss_are_json_safe(monkeypatch):
    issue(monkeypatch, [signal(T, T+1000), signal(T+2000, T+3000)])
    result = evaluate(tape([(T, 100, 101), (T+1000, 103, 104),
                            (T+2000, 100, 101), (T+3000, 100, 101)]))
    assert result['summary']['pf'] == pytest.approx(2)
    assert result['summary']['observed_gain_R'] == pytest.approx(.2)
    assert result['summary']['observed_loss_R'] == pytest.approx(.1)
    issue(monkeypatch, [signal(T, T+1000)])
    result = evaluate(tape([(T, 100, 101), (T+1000, 103, 104)]))
    assert result['summary']['pf'] is None
    assert result['summary']['pf_reason'] == 'NO_OBSERVED_LOSSES'
    json.dumps(result, allow_nan=False)


def test_real_side_policy_decisions_remain_causal_after_future_append():
    bids = [100, 100.125, 100.25, 100.375, 100.25, 100.375, 100.25, 100.375, 100.25]
    asks = [100.75, 100.75, 100.75, 100.75, 100.625, 100.75, 100.625, 100.75, 100.625]
    points = [(T+i*1000, bid, ask) for i, (bid, ask) in enumerate(zip(bids, asks))]
    points += [(T+i*1000, 100.25, 100.625) for i in range(9, 15)]
    before = evaluate(tape(points))
    after = evaluate(tape(points+[(T+15000, 150, 151)]))
    assert len(before['ledger']) == 1
    assert before['ledger'][0] == after['ledger'][0]
    assert before['ledger'][0]['decision_ms'] == T+9000


def test_invalid_contract_inputs_rejected_before_measurement():
    args = [QuoteTape([]), 'SIDE_LEAD', 1, S, 0, END, C()]
    for index, value in ((0, []), (1, 'OTHER'), (2, True), (3, float('nan')),
                         (3, 0), (4, -1), (4, True), (5, 0), (6, object())):
        invalid = list(args)
        invalid[index] = value
        with pytest.raises(ValueError):
            runner.evaluate_tape(*invalid)
    with pytest.raises(ValueError):
        runner.evaluate_tape(*args, baseline=1)
    config = C()
    object.__setattr__(config, 'entry_latency_ms', -1)
    args[6] = config
    with pytest.raises(ValueError):
        runner.evaluate_tape(*args)


def test_short_stage_has_no_eligible_window(monkeypatch):
    issue(monkeypatch, [signal(1000, 2000)])
    result = evaluate(tape([(1000, 100, 101), (2000, 102, 103)]), end=3000000)
    assert result['status'] == 'NO_ELIGIBLE_DECISION_WINDOW'
    assert result['ledger'] == []
