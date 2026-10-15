"""Artificial supplied signals only: compare every frozen output field."""
from bisect import bisect_left
from dataclasses import replace
import random

import pytest

from independent_multiframe_research.runner import evaluate_signals
from independent_multiframe_research.sharded import DAY_MS, MAX_HOLD_MS, ShardedQuoteStore
from independent_multiframe_research.streaming import IssuanceQuoteIndex
from independent_multiframe_research.streaming_runner import evaluate_streamed_signals
from independent_profit_research.execution import ExecutionConfig as C, Quote as Q
from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape


POLICY = 'SYNTHETIC_STREAMED_RUNNER_REVIEW'
T = 1_800_000


def signal(decision=T, horizon=None, stop=10., side=1, target=None, policy=POLICY):
    return Signal(policy, side, decision, decision + 2000 if horizon is None else horizon, stop, target)


def access(rows, start=0, end=2 * DAY_MS, cache_days=2):
    full = QuoteTape(rows)
    calls = []

    def loader(day):
        assert start <= day < end and day % DAY_MS == 0
        assert len(store.cached_day_starts) < cache_days
        calls.append(day)
        lo, hi = bisect_left(full.timestamps, day), bisect_left(full.timestamps, day + DAY_MS)
        return QuoteTape(full.quotes[lo:hi])

    store = ShardedQuoteStore(start, end, loader, cache_days)
    return full, store, calls


def pair(rows, signals, *, start=0, end=2 * DAY_MS, config=None, unit=1.,
         baseline=False, cache_days=2, policy=POLICY):
    config = C(0, 0) if config is None else config
    full, store, calls = access(rows, start, end, cache_days)
    left = evaluate_signals(full, list(signals), start, end, unit, config,
                            policy_id=policy, baseline=baseline)
    right = evaluate_streamed_signals(store, list(signals), start, end, unit, config,
                                     policy_id=policy, baseline=baseline)
    assert right == left
    assert len(store.cached_day_starts) <= cache_days <= 2
    assert all(start <= day < end for day in calls)
    return right, calls


@pytest.mark.parametrize('days', [2, 3])
@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('baseline', [False, True])
def test_seeded_complete_output_equivalence_across_days_gaps_and_overlap(days, seed, side, baseline):
    rng = random.Random(21000 + seed)
    end = days * DAY_MS
    centers = [T + 10000, DAY_MS - 3000, DAY_MS + 10000]
    if days == 3:
        centers.append(2 * DAY_MS - 2000)
    stamps = set()
    for center in centers:
        for offset in range(0, 6001, 250):
            if seed % 5 == 0 and center == centers[0] and 750 < offset < 6000:
                continue
            stamps.add(center + offset)
    rows = [Q(0, 100, 101)]
    for stamp in sorted(stamps):
        bid = 100 + rng.randrange(-300, 301) / 100
        quote = Q(stamp, bid, bid + rng.choice([0., .25, 1.]))
        rows.append(quote)
        if rng.random() < .2:
            rows.append(quote)
        if rng.random() < .1:
            rows.append(Q(stamp, bid - 3, quote.ask + 3))
    rows.append(Q(end, 50, 200))  # Outside the declared source stage.
    issued = [signal(T - 1, side=side)]
    for center in centers:
        issued.extend([signal(center, center + 2500, 2., side, 4.),
                       signal(center + 500, center + 2800, 2., side, None)])
    issued.append(signal(end - T, end - T + 2000, side=side))
    for config in (C(0, 0), C(250, 250, 5000, .25), C(2000, 2000, 5000, .5)):
        result, _ = pair(rows, issued, end=end, config=config, baseline=baseline,
                         cache_days=1 + seed % 2)
        if seed % 5:
            assert result['summary']['completed'] > 0


@pytest.mark.parametrize('side', [-1, 1])
def test_inclusive_reservation_covers_pending_entry_holding_and_pending_exit(side):
    decisions = [T, T + 100, T + 700, T + 1100, T + 1250, T + 1500]
    issued = [signal(d, T + 1000 if d == T else d + 2000, side=side) for d in decisions]
    stamps = (T, T + 100, T + 250, T + 700, T + 1000, T + 1100, T + 1250,
              T + 1500, T + 1750, T + 3500, T + 3750)
    result, _ = pair([Q(t, 100, 101) for t in stamps], issued, config=C())
    assert [r['decision_ms'] for r in result['ledger']] == [T, T + 1500]
    assert result['summary']['blocked_overlap'] == 4


@pytest.mark.parametrize('side', [-1, 1])
def test_skip_reserves_nothing_and_equal_decision_can_enter(side):
    result, _ = pair([Q(T, 100, 101), Q(T + 1000, 100, 101), Q(T + 3000, 102, 103)],
                     [signal(T, T + 1000, side=side), signal(T, T + 3000, side=side)],
                     config=C(1000, 0))
    assert [r['status'] for r in result['ledger']] == ['SKIP', 'COMPLETE']
    assert result['summary']['blocked_overlap'] == 0


@pytest.mark.parametrize('side', [-1, 1])
def test_exit_day_owns_return_and_same_ms_drawdown_marks_match(side):
    d = DAY_MS - 1000
    rows = [Q(d, 100, 101), Q(d + 500, 110, 111), Q(d + 500, 90, 91),
            Q(DAY_MS, 99, 100), Q(DAY_MS + 1000, 102, 103)]
    result, _ = pair(rows, [signal(d, DAY_MS + 1000, 50, side)], config=C(0, 0, 5000, .5))
    assert [day['completed'] for day in result['daily']] == [0, 1]
    assert result['summary']['marked_drawdown_pct'] > 0
    assert result['summary']['marked_quote_rows'] == len(rows)


@pytest.mark.parametrize('side', [-1, 1])
def test_same_ms_both_barriers_keep_stop_priority(side):
    rows = [Q(T, 100, 100), Q(T + 500, 90, 90), Q(T + 500, 110, 110),
            Q(T + 1000, 99, 101), Q(T + 1250, 98, 102), Q(T + 3000, 100, 100)]
    result, _ = pair(rows, [signal(T, T + 2000, 5, side, 5)], config=C(0, 250))
    assert result['ledger'][0]['reason'] == 'STOP_PRIORITY_SAME_MILLISECOND'
    assert result['ledger'][0]['both_barriers_in_bundle'] is True


@pytest.mark.parametrize('rows,reason', [([], 'MISSING_ENTRY_QUOTE'),
                                      ([Q(T, 100, 101), Q(T + 500, 100, 101)],
                                       'MISSING_HORIZON_EXIT_QUOTE'),
                                      ([Q(T, 100, 101), Q(T + 10000, 100, 101)],
                                       'GAP_BEFORE_EXIT')])
def test_unknown_closes_prefix_and_censors_all_primary_days(rows, reason):
    full, store, _ = access(rows, end=3 * DAY_MS)
    closures = []

    def issued():
        try:
            yield signal()
            raise AssertionError('The UNKNOWN suffix must not be consumed')
        finally:
            closures.append(True)

    expected = evaluate_signals(full, issued(), 0, 3 * DAY_MS, 1., C(0, 0), policy_id=POLICY)
    actual = evaluate_streamed_signals(store, issued(), 0, 3 * DAY_MS, 1., C(0, 0), policy_id=POLICY)
    assert actual == expected and closures == [True, True]
    assert actual['ledger'][0]['reason'] == reason
    assert all(d['sum_R'] is None and d['sum_quote'] is None for d in actual['daily'])
    assert actual['summary']['remaining_opportunities_count'] is None


def test_empty_middle_shard_retains_late_bundle_for_skip_not_missing_entry():
    result, calls = pair([Q(2 * DAY_MS + 10, 100, 101)], [signal()], end=3 * DAY_MS)
    assert result['ledger'][0]['reason'] == 'NO_ENTRY_BEFORE_HORIZON'
    assert result['ledger'][0]['status'] == 'SKIP'
    assert calls == [0, DAY_MS, 2 * DAY_MS]


def test_unknown_after_complete_keeps_prefix_description_but_nulls_whole_calendar():
    rows = [Q(T, 100, 101), Q(T + 1000, 102, 103), Q(DAY_MS + T, 100, 101)]
    issued = [signal(T, T + 1000), signal(DAY_MS + T, DAY_MS + T + 2000)]
    result, _ = pair(rows, issued, end=3 * DAY_MS)
    assert result['summary']['completed'] == result['summary']['unknown'] == 1
    assert result['summary']['observed_sum_R'] == .1
    assert result['summary']['sum_R'] is None
    assert [d['observed_sum_R'] for d in result['daily']] == [.1, 0., 0.]


def test_large_R_daily_fsum_rounding_residual_matches_reference_exactly():
    rows = [Q(T, 100e8, 100e8), Q(T + 1000, 200e8, 200e8),
            Q(T + 2000, 100e8, 100e8), Q(T + 3000, 100e8 + 1, 100e8 + 1),
            Q(DAY_MS + T, 200e8, 200e8), Q(DAY_MS + T + 1000, 100e8, 100e8)]
    issued = [signal(T, T + 1000, 1e-6), signal(T + 2000, T + 3000, 1.),
              signal(DAY_MS + T, DAY_MS + T + 1000, 1e-6)]
    result, _ = pair(rows, issued, unit=1e8)
    assert [r['R'] for r in result['ledger']] == [1e16, 1., -1e16]
    assert result['summary']['sum_R'] == 1.
    assert result['accounting_reconciliation']['daily_to_stage_fsum_residual']['observed_sum_R'] == -1.


def test_embargo_edges_match_and_unconsumed_invalid_suffix_is_not_validated():
    upper = 2 * DAY_MS - T
    full, store, _ = access([Q(T, 100, 101), Q(T + 2000, 102, 103)])
    closed = []

    def issued():
        try:
            yield signal(T - 1)
            yield signal(T)
            yield signal(upper, upper + 1000)
            yield object()
            raise AssertionError('Excluded suffix must not be consumed')
        finally:
            closed.append(True)

    expected = evaluate_signals(full, issued(), 0, 2 * DAY_MS, 1, C(0, 0), policy_id=POLICY)
    actual = evaluate_streamed_signals(store, issued(), 0, 2 * DAY_MS, 1, C(0, 0), policy_id=POLICY)
    assert actual == expected and closed == [True, True]
    assert actual['summary']['excluded_before_embargo'] == actual['summary']['excluded_after_embargo'] == 1


@pytest.mark.parametrize('bad', [replace(signal(), direction=True), replace(signal(), decision_ms=True),
                               replace(signal(), target_distance=float('inf')),
                               replace(signal(), horizon_end_ms=2 * DAY_MS),
                               replace(signal(), stop_distance=0), replace(signal(), policy_id='OTHER')])
def test_invalid_consumed_signal_errors_match_frozen_reference(bad):
    full, store, calls = access([])
    with pytest.raises(ValueError) as expected:
        evaluate_signals(full, [bad], 0, 2 * DAY_MS, 1, C(), policy_id=POLICY)
    with pytest.raises(ValueError) as actual:
        evaluate_streamed_signals(store, [bad], 0, 2 * DAY_MS, 1, C(), policy_id=POLICY)
    assert actual.value.args == expected.value.args
    assert calls == []


@pytest.mark.parametrize('where', ['blocked', 'before_embargo', 'decreasing_excluded_prefix'])
def test_snapshot_validation_precedes_overlap_and_embargo_filtering(where):
    rows = [Q(T, 100, 101), Q(T + 2000, 102, 103)]
    if where == 'blocked':
        issued = [signal(), replace(signal(T + 1), stop_distance=float('nan'))]
    elif where == 'before_embargo':
        issued = [replace(signal(T - 1), stop_distance=float('nan'))]
    else:
        issued = [signal(T - 1), signal(T - 2)]
    full, store, _ = access(rows)
    with pytest.raises(ValueError) as expected:
        evaluate_signals(full, issued, 0, 2 * DAY_MS, 1, C(0, 0), policy_id=POLICY)
    with pytest.raises(ValueError) as actual:
        evaluate_streamed_signals(store, issued, 0, 2 * DAY_MS, 1, C(0, 0), policy_id=POLICY)
    assert actual.value.args == expected.value.args


def test_no_signals_loads_no_days_and_keeps_idle_daily_zeroes():
    result, calls = pair([], [], start=5 * DAY_MS, end=8 * DAY_MS)
    assert calls == []
    assert all(day['sum_R'] == day['sum_quote'] == 0 for day in result['daily'])


def test_exact_nonzero_stage_excludes_outer_quotes_and_never_requests_outer_days():
    start, end = 5 * DAY_MS, 7 * DAY_MS
    d = start + T
    result, calls = pair([Q(0, 50, 100), Q(d, 100, 101), Q(d + 2000, 102, 103),
                          Q(end, 1, 200)], [signal(d)], start=start, end=end)
    assert result['summary']['completed'] == 1
    assert calls == [start]


@pytest.mark.parametrize('start,end', [(1, 2 * DAY_MS), (0, DAY_MS), (DAY_MS, 2 * DAY_MS)])
def test_calendar_mismatch_rejected_before_iterator_or_loader_access(start, end):
    _, store, calls = access([])
    with pytest.raises(ValueError, match='exactly match aligned'):
        evaluate_streamed_signals(store, object(), start, end, 1, C(), policy_id=POLICY)
    assert calls == []


@pytest.mark.parametrize('wrong_store', [None, QuoteTape([]), IssuanceQuoteIndex([])])
def test_fake_full_tape_or_sparse_index_is_not_a_store(wrong_store):
    with pytest.raises(ValueError, match='exact ShardedQuoteStore'):
        evaluate_streamed_signals(wrong_store, [], 0, DAY_MS, 1, C(), policy_id=POLICY)


def test_unsupported_execution_window_fails_without_silent_truncation():
    _, store, calls = access([])
    with pytest.raises(ValueError, match='one hour'):
        evaluate_streamed_signals(store, [signal(T, T + MAX_HOLD_MS + 1)],
                                  0, 2 * DAY_MS, 1, C(), policy_id=POLICY)
    assert calls == []
