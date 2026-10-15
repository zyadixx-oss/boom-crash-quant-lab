"""Artificial quotes only: no market files, fit, runner, orders or network."""
from dataclasses import dataclass

import pytest

from independent_profit_research.policies import Signal, generate_signals


@dataclass(frozen=True)
class Q:
    time_msc: int
    bid: float
    ask: float


def signals(rows, policy='SIDE_LEAD', native=1, unit=1):
    return list(generate_signals(iter(rows), policy_id=policy,
                                 native_direction=native, spread_unit=unit))


def side_rows():
    # Alternating both-side updates create events without one-sided opposition.
    bids = [100, 100.125, 100.25, 100.375, 100.25, 100.375, 100.25, 100.375, 100.25]
    asks = [100.75, 100.75, 100.75, 100.75, 100.625, 100.75, 100.625, 100.75, 100.625]
    return [Q(i*1000, bid, ask) for i, (bid, ask) in enumerate(zip(bids, asks))]


def clock_rows(*, recent=15, baseline_step=2000, jump=False):
    # Closed source time315000 has exactly315s coverage, with recent count15.
    times = list(range(0, 300001, baseline_step)) + list(range(301000, 316000, 1000))[-recent:]
    rows = [Q(t, 100+i*.125, 100.5+i*.125) for i, t in enumerate(times)]
    if jump:
        rows[-1] = Q(rows[-1].time_msc, rows[-2].bid+4, rows[-2].ask+4)
    rows.append(Q(316000, rows[-1].bid, rows[-1].ask))
    return rows


def carry_rows():
    return [Q(0, 100, 101), Q(1000, 110, 111), Q(2000, 110, 111),
            Q(3000, 110, 111), Q(4000, 110, 111), Q(5000, 110, 111)]


def mirror(rows):
    return [Q(q.time_msc, 1000-q.ask, 1000-q.bid) for q in rows]


def test_side_requires_eight_completed_changed_events():
    rows = side_rows()
    assert signals(rows) == []  # Last changed bundle is unfinished.
    result = signals(rows+[Q(9000, rows[-1].bid, rows[-1].ask)])
    assert result == [Signal('SIDE_LEAD', 1, 9000, 14000, 2., 2.)]


def test_identical_same_ms_duplicates_create_no_extra_event():
    rows = side_rows()
    expanded = [q for row in rows for q in (row, row, row)]
    current = Q(9000, rows[-1].bid, rows[-1].ask)
    assert signals(expanded+[current]) == signals(rows+[current])


def test_unfinished_current_bundle_never_changes_issued_signal():
    rows = side_rows()+[Q(9000, 100.25, 100.625)]
    issued = signals(rows)
    assert len(issued) == 1
    assert signals(rows+[Q(9000, 95, 97)]) == issued
    assert signals(rows+[Q(9000, 95, 97), Q(10000, 100, 100.5)]) == issued


def test_ambiguous_completed_bundle_resets_warmup():
    rows = side_rows()
    rows.insert(5, Q(4000, 99, 100))
    assert signals(rows+[Q(9000, 100.25, 100.625)]) == []


def test_side_xor_and_last_three_opposition():
    rows = side_rows()
    rows[-1] = Q(8000, rows[-2].bid-.125, rows[-2].ask)  # Opposite bid-only.
    assert signals(rows+[Q(9000, rows[-1].bid, rows[-1].ask)]) == []
    both = [Q(i*1000, 100+i*.125, 100.5+i*.125) for i in range(10)]
    assert signals(both) == []


def test_side_uses_current_first_spread_and_range_filter():
    rows = side_rows()
    assert signals(rows+[Q(9000, 100, 102)]) == []
    rows = [Q(q.time_msc, q.bid+i*.25, q.ask+i*.25) for i, q in enumerate(rows)]
    assert signals(rows+[Q(9000, 103, 103.5)]) == []


@pytest.mark.parametrize('policy,rows', [('SIDE_LEAD', side_rows()+[Q(9000, 100.25, 100.625)]),
                                      ('QUOTE_CLOCK_ACCELERATION', clock_rows()),
                                      ('POST_JUMP_CARRY_30S', carry_rows())])
def test_every_prefix_is_unchanged_by_future_append(policy, rows):
    full = signals(rows, policy)
    for length in range(len(rows)+1):
        prefix = rows[:length]
        issued = signals(prefix, policy)
        if prefix:
            assert issued == [s for s in full if s.decision_ms <= prefix[-1].time_msc]
        else:
            assert issued == []


@pytest.mark.parametrize('policy,rows', [('SIDE_LEAD', side_rows()+[Q(9000, 100.25, 100.625)]),
                                      ('QUOTE_CLOCK_ACCELERATION', clock_rows()),
                                      ('POST_JUMP_CARRY_30S', carry_rows())])
def test_direction_mirror(policy, rows):
    original = signals(rows, policy, native=1)
    reflected = signals(mirror(rows), policy, native=-1)
    assert original and len(original) == len(reflected)
    assert [(s.decision_ms, s.horizon_end_ms, s.direction) for s in original] == [
        (s.decision_ms, s.horizon_end_ms, -s.direction) for s in reflected]


def test_stale_decision_skips_and_gap_resets_side_warmup():
    rows = side_rows()
    assert signals(rows+[Q(10001, 100.25, 100.625)]) == []
    assert signals(rows+[Q(13001, 100.25, 100.625), Q(14001, 100.25, 100.625)]) == []
    assert signals(rows+[Q(10000, 100.25, 100.625)])[0].decision_ms == 10000


def test_clock_coverage_rate_and_native_direction():
    result = signals(clock_rows(), 'QUOTE_CLOCK_ACCELERATION', native=-1)
    assert result == [Signal('QUOTE_CLOCK_ACCELERATION', -1, 316000, 346000, 2., 2.)]
    assert signals(clock_rows()[:-2], 'QUOTE_CLOCK_ACCELERATION') == []
    assert signals(clock_rows(recent=11), 'QUOTE_CLOCK_ACCELERATION') == []
    assert signals(clock_rows(baseline_step=1000), 'QUOTE_CLOCK_ACCELERATION') == []
    rows = clock_rows()
    rows[-1] = Q(316000, rows[-1].bid, rows[-1].bid+10)
    assert signals(rows, 'QUOTE_CLOCK_ACCELERATION')  # No extra spread gate.


def test_clock_large_recent_increment_is_excluded_inclusively():
    assert signals(clock_rows(jump=True), 'QUOTE_CLOCK_ACCELERATION') == []
    rows = clock_rows()
    rows[-2] = Q(315000, rows[-3].bid+3.875, rows[-3].ask+3.875)
    assert signals(rows, 'QUOTE_CLOCK_ACCELERATION')


def test_clock_window_boundaries_are_open_left_closed_right():
    changes = set(range(86000, 246000, 1000)) | set(range(386000, 398000, 1000)) | {85000}
    def path(events):
        bid, rows = 100., []
        for t in range(0, 402000, 1000):
            if t in events: bid += .125
            rows.append(Q(t, bid, bid+.5))
        return rows
    # At source400000: the event at85000 is excluded, baseline=160 and
    # recent=12, so the1.5 rate boundary holds at equality. Earlier windows
    # retain85000 and fail the rate test. The current first tick is401000.
    result = signals(path(changes), 'QUOTE_CLOCK_ACCELERATION')
    assert result == [Signal('QUOTE_CLOCK_ACCELERATION', 1, 401000, 431000, 2., 2.)]
    # An event at385000 is included in baseline, not recent; then161 vs12 fails.
    result = signals(path(changes | {385000}), 'QUOTE_CLOCK_ACCELERATION')
    assert not any(s.decision_ms == 401000 for s in result)
    rows = clock_rows()
    boundary = next(i for i, row in enumerate(rows) if row.time_msc == 300000)
    shifted = [Q(q.time_msc, q.bid+(4 if i>=boundary else 0), q.ask+(4 if i>=boundary else 0))
               for i, q in enumerate(rows)]
    assert signals(shifted, 'QUOTE_CLOCK_ACCELERATION') == signals(rows, 'QUOTE_CLOCK_ACCELERATION')


def test_clock_requires_positive_baseline_and_resets_after_large_gap():
    rows = [Q(t, 100, 101) for t in range(0, 301000, 1000)]
    rows += [Q(t, 100+(t-300000)/8000, 101+(t-300000)/8000) for t in range(301000, 317000, 1000)]
    assert signals(rows, 'QUOTE_CLOCK_ACCELERATION') == []
    rows = clock_rows()
    rows[-1] = Q(321000, rows[-1].bid, rows[-1].ask)
    rows.append(Q(322000, rows[-1].bid, rows[-1].ask))
    assert signals(rows, 'QUOTE_CLOCK_ACCELERATION') == []


def test_carry_schedule_uses_detection_and_has_no_target():
    rows = carry_rows()
    assert signals(rows[:4], 'POST_JUMP_CARRY_30S') == []
    result = signals(rows, 'POST_JUMP_CARRY_30S')
    assert result == [Signal('POST_JUMP_CARRY_30S', -1, 4000, 32000, 4., None)]


def test_carry_first_due_spread_failure_cancels_without_retry():
    rows = carry_rows()
    rows[4] = Q(4000, 110, 113)
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []


def test_carry_cancellation_preserves_sixty_second_cooldown():
    rows = carry_rows()
    rows[4] = Q(4000, 110, 113)
    rows += [Q(6000, 120, 121), Q(7000, 120, 121), Q(8000, 120, 121), Q(9000, 120, 121)]
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []
    rows += [Q(t, 120, 121) for t in range(10000, 61001, 1000)]
    rows += [Q(62000, 130, 131), Q(63000, 130, 131), Q(64000, 130, 131), Q(65000, 130, 131)]
    result = signals(rows, 'POST_JUMP_CARRY_30S')
    assert result == [Signal('POST_JUMP_CARRY_30S', -1, 65000, 93000, 4., None)]


def test_carry_requires_both_sides_jump_and_short_source_gap():
    rows = carry_rows()
    rows[1] = Q(1000, 100, 121)
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []
    rows = [Q(0, 100, 101), Q(2001, 110, 111), Q(3000, 110, 111), Q(4000, 110, 111), Q(5000, 110, 111)]
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []


def test_carry_ambiguous_or_stale_due_observation_cancels():
    rows = carry_rows()
    rows.insert(4, Q(3000, 111, 112))
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []
    rows = carry_rows()[:3]+[Q(4500, 110, 111), Q(5000, 110, 111)]
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []
    rows = carry_rows()[:3]+[Q(10000, 110, 111), Q(11000, 120, 121),
                            Q(12000, 120, 121), Q(13000, 120, 121), Q(14000, 120, 121)]
    assert signals(rows, 'POST_JUMP_CARRY_30S') == []  # Gap cancels, cooldown survives.


def test_invalid_configuration_and_original_order_rejected():
    for value in (0, -1, True, float('nan'), float('inf'), 1e308):
        with pytest.raises(ValueError): signals([], unit=value)
    with pytest.raises(ValueError): signals([], policy='UNKNOWN')
    with pytest.raises(ValueError): signals([], native=True)
    with pytest.raises(ValueError): signals([Q(1000, 100, 101), Q(999, 100, 101)])
    with pytest.raises(ValueError): signals([Q(1000, 102, 101)])


def test_streaming_yield_does_not_consume_future_rows():
    rows = side_rows()+[Q(9000, 100.25, 100.625)]
    def source():
        yield from rows
        raise AssertionError('Future suffix was consumed before issued signal')
    generated = generate_signals(source(), policy_id='SIDE_LEAD', native_direction=1, spread_unit=1)
    assert next(generated).decision_ms == 9000


def test_side_issues_only_native_direction_at_new_change_event():
    rows = side_rows()+[Q(9000, 100.25, 100.625)]
    assert signals(rows, native=-1) == []
    assert signals(rows+[Q(10000, 100.25, 100.625)]) == signals(rows)
