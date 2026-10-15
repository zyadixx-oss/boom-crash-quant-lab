"""Exact reference equivalence on artificial quotes, never market data."""
from dataclasses import asdict
import random

import pytest

from independent_profit_research import execution
from independent_profit_research.execution import ExecutionConfig as C, Quote as Q
from independent_profit_research.tape import QuoteTape


CASES = [
    ('empty', [], 0, 1000, 1, None, C()),
    ('only_past', [Q(0, 100, 101)], 100, 1000, 1, None, C()),
    ('late_entry_ceiling', [Q(10000, 100, 101)], 0, 1000, 1, None, C()),
    ('latency_reaches_horizon', [Q(10000, 100, 101)], 0, 250, 1, None, C()),
    ('zero_latency', [Q(0, 100, 100.5), Q(1000, 101, 101.5)], 0, 1000, 3, None, C(0, 0)),
    ('entry_duplicates', [Q(250, 100, 101), Q(250, 100.5, 102), Q(500, 101, 102),
                          Q(1000, 101, 102), Q(1250, 102, 103)], 0, 1000, 4, None, C()),
    ('ceiling_duplicate_tail', [Q(250, 100, 101), Q(1000, 100, 101), Q(1250, 100, 101),
                               Q(1250, 90, 110), Q(1500, 100, 101)], 0, 1000, 20, None, C()),
    ('entry_spread_stop', [Q(250, 100, 102), Q(500, 100, 102), Q(750, 99, 101),
                           Q(1250, 99, 101)], 0, 1000, 1, None, C()),
    ('both_barriers', [Q(250, 100, 101), Q(500, 98, 99), Q(500, 105, 106),
                       Q(750, 100, 101), Q(1000, 99, 100), Q(1250, 99, 100)], 0, 1000, 2, 2, C()),
    ('confirmation_equals_horizon', [Q(250, 100, 101), Q(500, 90, 91),
                                      Q(1000, 102, 103), Q(1250, 103, 104)], 0, 1000, 2, 2, C()),
    ('confirmation_after_horizon', [Q(250, 100, 101), Q(500, 90, 91), Q(1250, 103, 104)],
     0, 1000, 2, None, C()),
    ('horizon_bundle_barrier', [Q(250, 100, 101), Q(1000, 90, 91), Q(1250, 103, 104)],
     0, 1000, 2, None, C()),
    ('gap_during_hold', [Q(250, 100, 101), Q(6000, 100, 101), Q(9000, 100, 101)],
     0, 8000, 2, None, C()),
    ('gap_before_entry', [Q(6000, 100, 101), Q(7000, 100, 101), Q(8250, 100, 101)],
     0, 8000, 2, None, C()),
    ('gap_before_exit', [Q(250, 100, 101), Q(1000, 100, 101), Q(10000, 100, 101)],
     0, 1000, 2, None, C()),
    ('missing_horizon_exit', [Q(250, 100, 101), Q(500, 100, 101)], 0, 1000, 2, None, C()),
    ('missing_barrier_exit', [Q(250, 100, 101), Q(500, 90, 91), Q(750, 90, 91)],
     0, 1000, 2, None, C()),
    ('long_latency', [Q(t, 100+t/1000, 101+t/1000) for t in range(0, 10001, 1000)],
     0, 8000, 4, 3, C(2000, 2000)),
    ('exact_gap_boundary', [Q(5000, 100, 101), Q(10000, 100, 101), Q(15000, 100, 101)],
     0, 10000, 4, None, C(2000, 2000, 5000, .125)),
    ('decision_at_bundle', [Q(0, 100, 101), Q(1000, 101, 102), Q(1000, 102, 103),
                            Q(1500, 101, 102), Q(2250, 101, 102)], 1000, 2000, 3, 2, C(0, 250)),
]


@pytest.mark.parametrize('name,rows,decision,horizon,stop,target,config', CASES, ids=[x[0] for x in CASES])
def test_complete_outcome_equivalence(name, rows, decision, horizon, stop, target, config):
    tape = QuoteTape(iter(rows))
    for direction in (-1, 1):
        actual = tape.simulate(decision, direction, horizon, stop, target, config)
        expected = execution.simulate_trade(tape.quotes, decision, direction, horizon, stop, target, config)
        assert actual == expected, name
        assert asdict(actual) == asdict(expected), name


def test_seeded_random_full_field_equivalence():
    rng = random.Random(20261013)
    checked = 0
    for _ in range(64):
        stamp, bid, rows = 0, 100., []
        for _ in range(70):
            stamp += rng.choice((1, 100, 250, 500, 1000, 2000, 5000, 5001, 10000))
            bid = max(10., bid+rng.choice((-3., -1., -.125, 0., .125, 1., 3.)))
            for _ in range(rng.randint(1, 3)):
                local_bid = max(1., bid+rng.choice((-1., 0., 0., 1.)))
                rows.append(Q(stamp, local_bid, local_bid+rng.choice((0., .125, .5, 2.))))
        tape = QuoteTape(rows)
        for direction in (-1, 1):
            for entry_latency in (0, 250, 2000):
                for exit_latency in (0, 250, 2000):
                    config = C(entry_latency, exit_latency, rng.choice((1000, 5000)), rng.choice((0., .125)))
                    for _ in range(3):
                        decision = rng.randint(0, stamp+5000)
                        horizon = decision+rng.choice((1, 250, 1000, 5000, 30000))
                        stop, target = rng.choice((.125, 1., 4.)), rng.choice((None, .125, 2., 10.))
                        actual = tape.simulate(decision, direction, horizon, stop, target, config)
                        expected = execution.simulate_trade(tape.quotes, decision, direction, horizon, stop, target, config)
                        assert asdict(actual) == asdict(expected)
                        checked += 1
    assert checked == 3456


def test_constructor_validates_entire_input_including_unused_suffix():
    valid = [Q(0, 100, 101), Q(250, 100, 101), Q(1250, 100, 101)]
    bad = Q(999999, 100, 101)
    object.__setattr__(bad, 'ask', float('nan'))
    for rows in (valid+[bad], valid+[object()], valid+[Q(1000, 100, 101)]):
        with pytest.raises(ValueError):
            QuoteTape(rows)
        with pytest.raises(ValueError):
            execution.simulate_trade(rows, 0, 1, 1000, 2)


def test_snapshots_preserve_order_duplicates_and_one_pass_input():
    rows = [Q(0, 100, 101), Q(0, 100, 101), Q(250, 100, 102), Q(1250, 100, 101)]
    visits = []
    def source():
        for i, quote in enumerate(rows):
            visits.append(i)
            yield quote
    tape = QuoteTape(source())
    assert visits == [0, 1, 2, 3]
    assert tape.quotes == tuple(rows) and tape.timestamps == (0, 0, 250, 1250)
    assert len(tape) == 4 and tuple(tape) == tuple(rows)
    object.__setattr__(rows[-1], 'ask', float('nan'))
    actual = tape.simulate(0, 1, 1000, 5)
    assert asdict(actual) == asdict(execution.simulate_trade(tape.quotes, 0, 1, 1000, 5))
    assert visits == [0, 1, 2, 3]


def test_bisect_slice_is_bounded_and_keeps_entire_ceiling_bundle(monkeypatch):
    rows = [Q(t, 100, 101) for t in range(0, 1000000, 100)]
    rows.insert(1014, Q(101300, 90, 110))
    tape = QuoteTape(iter(rows))
    reference = execution.simulate_trade
    observed = []
    def spy(quotes, *args):
        retained = tuple(quotes)
        observed.append(retained)
        return reference(retained, *args)
    monkeypatch.setattr(execution, 'simulate_trade', spy)
    result = tape.simulate(100000, 1, 101000, 20)
    assert len(observed) == 1 and len(observed[0]) == 15
    assert observed[0][0].time_msc == 100000
    assert [q.time_msc for q in observed[0][-2:]] == [101300, 101300]
    assert asdict(result) == asdict(reference(tape.quotes, 100000, 1, 101000, 20))


def test_request_validation_matches_reference():
    tape = QuoteTape([Q(0, 100, 101)])
    invalid = [(True, 1, 1000, 1, None, None), (0, True, 1000, 1, None, None),
               (0, 1, 0, 1, None, None), (0, 1, 1000, 0, None, None),
               (0, 1, 1000, 1, float('nan'), None), (0, 1, 1000, 1, None, object())]
    config = C()
    object.__setattr__(config, 'exit_latency_ms', -1)
    invalid.append((0, 1, 1000, 1, None, config))
    for args in invalid:
        with pytest.raises(ValueError) as reference:
            execution.simulate_trade(tape.quotes, *args)
        with pytest.raises(ValueError) as actual:
            tape.simulate(*args)
        assert str(actual.value) == str(reference.value)
