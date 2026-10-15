"""Pre-outcome synthetic sparse-clock checks, never market/strategy evidence."""
from dataclasses import asdict
import random

import pytest

from independent_multiframe_research.bars import Bar
from independent_multiframe_research.replication_control import (
    DAY_MS, SLOT_MS, POLICY_ID, _uniform_slot, sparse_schedule, generate_sparse_control,
)
from independent_multiframe_research.streaming import IssuanceQuoteIndex
from independent_multiframe_research.streaming_policies import clock_signals
from independent_profit_research.execution import Quote

SEED = '00' * 32
M = 60000


def fixture(days=5):
    frames = {size: [Bar(t-size*M, t, t+123, 100, 102, 98, 100, 0)
                     for t in range(size*M, days*DAY_MS+1, size*M)]
              for size in (1, 5, 15, 60, 240)}
    rows = [Quote(t+123, 100, 101) for t in range(M, days*DAY_MS+1, M)]
    return frames, IssuanceQuoteIndex(rows)


def test_independent_golden_wire_schedule_and_last_singleton():
    actual = sparse_schedule(0, 5*DAY_MS, SEED)
    assert [(x.possible_slots, (x.scheduled_ms-x.block_start_ms)//SLOT_MS,
             x.rejection_counter, x.digest_hex) for x in actual] == [
        (96, 3, 0, 'f671d01fcab74883536f3f61bc881291b9957cb49b15e657350fd6f8e75659a3'),
        (96, 3, 0, 'c389d22fe49cad217a240f36b7e0e1f48035b63b2beed037499007842e72baa3'),
        (48, 33, 0, '66397b5182e4e716159e5dbf5aebc7e47860115e8b24c1f268451626a16dbee1'),
    ]


def test_exact219_calendar_slots_generate110_ex_ante_opportunities():
    start = 19000*DAY_MS
    schedule = sparse_schedule(start, start+219*DAY_MS, 'ab'*32)
    assert len(schedule) == 110
    assert [x.possible_slots for x in schedule] == [96]*109+[48]
    assert all(x.block_start_ms <= x.scheduled_ms < x.block_end_exclusive_ms
               and x.scheduled_ms % SLOT_MS == 0 for x in schedule)
    assert len({x.scheduled_ms for x in schedule}) == 110


def test_rejection_branch_does_not_modulo_bias_or_redraw_by_price():
    calls = []
    def digest(wire):
        calls.append(wire)
        return b'\xff'*32 if len(calls) == 1 else (95).to_bytes(32, 'big')
    slot, counter, raw = _uniform_slot(bytes(32), 0, 96, digest)
    assert (slot, counter, raw) == (95, 1, (95).to_bytes(32, 'big').hex())
    assert calls[0][:-4] == calls[1][:-4]
    assert calls[0][-4:] == bytes(4) and calls[1][-4:] == (1).to_bytes(4, 'big')


def test_global_rng_state_and_future_complete_blocks_do_not_change_prior_schedule():
    original = sparse_schedule(0, 4*DAY_MS, SEED)
    random.seed(999); [random.random() for _ in range(100)]
    assert sparse_schedule(0, 4*DAY_MS, SEED) == original
    assert sparse_schedule(0, 8*DAY_MS, SEED)[:2] == original


@pytest.mark.parametrize('direction', [-1, 1])
def test_nonempty_sparse_issuance_exactly_filters_dense_and_preserves_each_signal_field(direction):
    frames, index = fixture()
    due = {x.scheduled_ms for x in sparse_schedule(0, 5*DAY_MS, SEED)}
    dense_audit, audit = {}, {}
    dense = list(clock_signals(frames, index, 1, direction, 0, 5*DAY_MS, audit=dense_audit))
    expected = [x for x in dense if dense_audit[('CLOCK_00_30', x.decision_ms)]['scheduled_ms'] in due]
    actual = list(generate_sparse_control(frames, index, 1, direction, 0, 5*DAY_MS, SEED, audit=audit))
    assert len(expected) == len(actual) == 3
    for left, right in zip(expected, actual, strict=True):
        assert asdict(right) == {**asdict(left), 'policy_id': POLICY_ID}
        assert audit[(POLICY_ID, right.decision_ms)]['scheduled_ms'] in due
        assert audit[(POLICY_ID, right.decision_ms)]['premeasurement_declaration_sha256'] == SEED


@pytest.mark.parametrize('cause', ['spread', 'freshness', 'missing'])
def test_selected_slot_failure_cancels_without_later_clean_substitute(cause):
    frames, index = fixture(2)
    due = sparse_schedule(0, 2*DAY_MS, SEED)[0].scheduled_ms
    assert len(list(generate_sparse_control(frames, index, 1, -1, 0, 2*DAY_MS, SEED))) == 1
    rows = list(index.quotes)
    if cause == 'spread':
        rows = [Quote(q.time_msc, q.bid, 1000 if q.time_msc == due+123 else q.ask) for q in rows]
    elif cause == 'freshness':
        rows = [Quote(due+2500 if q.time_msc == due+123 else q.time_msc, q.bid, q.ask) for q in rows]
    else:
        rows = [q for q in rows if q.time_msc != due+123]
    changed = IssuanceQuoteIndex(rows)
    assert list(clock_signals(frames, changed, 1, -1, 0, 2*DAY_MS))
    assert list(generate_sparse_control(frames, changed, 1, -1, 0, 2*DAY_MS, SEED)) == []
    assert sparse_schedule(0, 2*DAY_MS, SEED)[0].scheduled_ms == due


def test_downstream_close_explicitly_closes_dense_iterator(monkeypatch):
    from independent_multiframe_research import replication_control as module
    from independent_profit_research.policies import Signal
    closed = []
    due = sparse_schedule(0, 2*DAY_MS, SEED)[0].scheduled_ms
    def dense(*args, **kwargs):
        try:
            kwargs['audit'][('CLOCK_00_30', due+123)] = {'scheduled_ms': due}
            yield Signal('CLOCK_00_30', 1, due+123, due+15*M, 1, 2)
            raise AssertionError('Unconsumed suffix must not be visited')
        finally:
            closed.append(True)
    monkeypatch.setattr(module, 'dense_clock_signals', dense)
    iterator = generate_sparse_control(None, None, 1, 1, 0, 2*DAY_MS, SEED)
    assert next(iterator).policy_id == POLICY_ID
    iterator.close()
    assert closed == [True]


@pytest.mark.parametrize('start,end,seed', [
    (True, DAY_MS, SEED), (-DAY_MS, DAY_MS, SEED), (1, DAY_MS, SEED),
    (0, DAY_MS+1, SEED), (0, 0, SEED), (0, 2**63, SEED),
    (0, DAY_MS, 'ab'), (0, DAY_MS, 'AB'*32), (0, DAY_MS, None),
])
def test_invalid_calendar_or_hash_rejected_before_any_issuance(start, end, seed):
    with pytest.raises(ValueError):
        sparse_schedule(start, end, seed)
    with pytest.raises(ValueError):
        list(generate_sparse_control(None, None, 1, 1, start, end, seed))
