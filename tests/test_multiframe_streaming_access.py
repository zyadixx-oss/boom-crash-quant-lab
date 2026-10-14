"""Synthetic equivalence to frozen bars/policies; no market data or returns."""
from bisect import bisect_left
import random

import pytest

from independent_multiframe_research.bars import build_frames, MINUTE_MS as M
from independent_multiframe_research.policies import POLICY_IDS, generate_signals, clock_signals
from independent_multiframe_research.streaming import build_streamed_frames, IssuanceQuoteIndex
from independent_multiframe_research.streaming_policies import (
    generate_signals as indexed_signals, clock_signals as indexed_clock,
)
from independent_multiframe_research.runner import evaluate_signals
from independent_profit_research.runner import evaluate_tape
from independent_profit_research.execution import Quote, ExecutionConfig, simulate_trade
from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape
from test_multiframe_quote_policies import fixture


@pytest.mark.parametrize('seed', range(8))
def test_all_frame_fields_and_first_issuance_quotes_match_full_reference(seed):
    rng = random.Random(14000 + seed)
    start, end = 16 * 60 * M, 34 * 60 * M
    rows, price = [], 1_000_000
    for stamp in range(start, end + 8000, 4000):
        if seed % 2 and start + 5 * 60 * M <= stamp < start + 5 * 60 * M + 12000:
            continue
        price += rng.randrange(-50, 51)
        quote = Quote(stamp + 31, price, price + rng.choice([0, 2, 4]))
        rows.append(quote)
        if rng.random() < .003:
            rows.append(quote)
        if seed % 3 and rng.random() < .0001:
            rows.append(Quote(quote.time_msc, price - 1, quote.ask + 1))
    full = QuoteTape(rows)
    streamed = build_streamed_frames(iter(rows), start, end)
    assert streamed.frames == build_frames(full, start, end)
    assert streamed.source_rows_within_window == sum(start <= q.time_msc < end for q in rows)
    assert len(streamed.issuance_index) == len(streamed.frames[1])
    for bar, actual in zip(streamed.frames[1], streamed.issuance_index.quotes, strict=True):
        expected = full.quotes[bisect_left(full.timestamps, bar.end_ms)]
        assert actual == expected and actual.time_msc == bar.confirmed_ms
    for direction in (-1, 1):
        for policy in sorted(POLICY_IDS):
            old_audit, new_audit = {}, {}
            old = list(generate_signals(streamed.frames, policy_id=policy,
                                       native_direction=direction, spread_unit=4,
                                       tape=full, audit=old_audit))
            new = list(indexed_signals(streamed.frames, policy_id=policy,
                                       native_direction=direction, spread_unit=4,
                                       issuance_index=streamed.issuance_index, audit=new_audit))
            assert new == old and new_audit == old_audit
        old_clock = list(clock_signals(streamed.frames, full, 4, direction, start, end))
        old_clock_audit, new_clock_audit = {}, {}
        old_clock = list(clock_signals(streamed.frames, full, 4, direction, start, end,
                                      audit=old_clock_audit))
        new_clock = list(indexed_clock(streamed.frames, streamed.issuance_index, 4, direction,
                                      start, end, audit=new_clock_audit))
        assert new_clock == old_clock
        assert new_clock_audit == old_clock_audit
        assert old_clock  # A nonempty control comparison, including the midnight boundary.


@pytest.mark.parametrize('policy', sorted(POLICY_IDS))
@pytest.mark.parametrize('direction', [-1, 1])
def test_nonempty_fixed_policy_issuance_and_audit_match_sparse_index(policy, direction):
    frames, rows = fixture(policy)
    if direction == -1:
        from dataclasses import replace
        frames = {size: [replace(b, open=1000-b.open, high=1000-b.low,
                                 low=1000-b.high, close=1000-b.close) for b in bars]
                  for size, bars in frames.items()}
        rows = [Quote(q.time_msc, 1000-q.ask, 1000-q.bid) for q in rows]
    full = QuoteTape(rows)
    sparse = IssuanceQuoteIndex(full.quotes[bisect_left(full.timestamps, b.end_ms)]
                               for b in frames[1])
    left, right = {}, {}
    expected = list(generate_signals(frames, policy_id=policy, native_direction=direction,
                                     spread_unit=1, tape=full, audit=left))
    actual = list(indexed_signals(frames, policy_id=policy, native_direction=direction,
                                 spread_unit=1, issuance_index=sparse, audit=right))
    assert len(expected) == 1
    assert actual == expected and right == left


def test_sparse_issuance_index_refuses_execution():
    index = IssuanceQuoteIndex([Quote(0, 100, 102)])
    assert not isinstance(index, QuoteTape)
    with pytest.raises(ValueError, match='cannot be used for execution'):
        index.simulate(0, 1, 1000, 1)
    with pytest.raises(TypeError):
        iter(index)
    with pytest.raises(TypeError):
        QuoteTape(index)
    with pytest.raises(TypeError):
        simulate_trade(index, 0, 1, 1000, 1)


@pytest.mark.parametrize('cropped_calendar', [False, True])
def test_frozen_runners_reject_index_before_cropping_or_consuming_signals(cropped_calendar):
    index = IssuanceQuoteIndex([Quote(0, 100000, 100002), Quote(1920000, 100000, 100002)])
    start = 60001 if cropped_calendar else 0
    consumed = []

    def signals():
        consumed.append(True)
        yield Signal('SYNTHETIC_REVIEW_ONLY', 1, 1920000, 2820000, 1000, 2000)

    with pytest.raises(ValueError, match='validated QuoteTape'):
        evaluate_signals(index, signals(), start, 4100000, 2, ExecutionConfig(),
                         policy_id='SYNTHETIC_REVIEW_ONLY')
    assert consumed == []
    with pytest.raises(ValueError, match='validated QuoteTape'):
        evaluate_tape(index, 'SIDE_LEAD', 1, 2, start, 4100000, ExecutionConfig())


def test_frozen_public_issuance_functions_also_reject_index_type():
    frames, rows = fixture('CRT_SWEEP_RECLAIM')
    index = IssuanceQuoteIndex(rows)
    with pytest.raises(TypeError, match='QuoteTape'):
        list(generate_signals(frames, policy_id='CRT_SWEEP_RECLAIM', native_direction=1,
                              spread_unit=1, tape=index))
    with pytest.raises(TypeError, match='QuoteTape'):
        list(clock_signals(frames, index, 1, 1, 0, 100000000))


def test_same_ms_ambiguity_in_confirmation_preserves_previous_bar_and_first_quote():
    rows = [Quote(t, 100, 102) for t in range(0, M+1, 1000)]
    rows += [Quote(M, 99, 103)] + [Quote(t, 100, 102) for t in range(M+1000, 3*M, 1000)]
    result = build_streamed_frames(rows, 0, 3*M)
    assert result.frames == build_frames(QuoteTape(rows), 0, 3*M)
    assert [b.open_ms for b in result.frames[1]] == [0]
    assert result.issuance_index.quotes == (Quote(M, 100, 102),)


def test_file_chunk_boundaries_do_not_reset_runs_or_fabricate_tail_confirmation():
    start, end = 24*60*M - 5*M, 24*60*M + 6*M
    first = [Quote(t, 100, 102) for t in range(start, 24*60*M, 1000)]
    second = [Quote(t, 101, 103) for t in range(24*60*M, end+1, 1000)]
    def source():
        yield from first
        yield from second
    actual = build_streamed_frames(source(), start, end)
    assert actual.frames == build_frames(QuoteTape(first + second), start, end)
    assert {b.run_id for b in actual.frames[1]} == {0}
    assert actual.frames[1][-1].end_ms == end-M


def test_source_mutation_after_consumption_does_not_change_confirmation_snapshot():
    rows = [Quote(t, 100, 102) for t in range(0, 2*M, 1000)]
    def source():
        for quote in rows:
            yield quote
            object.__setattr__(quote, 'bid', 1)
    result = build_streamed_frames(source(), 0, 2*M)
    assert result.frames[1][0].close == 101
    assert result.issuance_index.quotes == (Quote(M, 100, 102),)


@pytest.mark.parametrize('bad', [object(), Quote(1, 100.5, 102), Quote(1, 100, 2**45)])
def test_invalid_in_window_rows_close_iterator(bad):
    closed = []
    def source():
        try:
            yield Quote(0, 100, 102)
            yield bad
        finally:
            closed.append(True)
    with pytest.raises(ValueError):
        build_streamed_frames(source(), 0, 2*M)
    assert closed == [True]


@pytest.mark.parametrize('start,end', [(True, M), (-1, M), (0, 0), (M, 0), (0, 2**63)])
def test_invalid_calendar_rejected(start, end):
    with pytest.raises(ValueError):
        build_streamed_frames([], start, end)


def test_decreasing_timestamps_rejected_without_sorting():
    with pytest.raises(ValueError, match='nondecreasing'):
        build_streamed_frames([Quote(1000, 100, 102), Quote(0, 100, 102)], 0, M)


def test_outside_successor_cannot_confirm_tail_and_iterator_is_closed():
    closed = []
    def source():
        try:
            for t in range(0, M+1, 1000):
                yield Quote(t, 100, 102)
            raise AssertionError('Outside suffix must not be consumed')
        finally:
            closed.append(True)
    result = build_streamed_frames(source(), 0, M)
    assert result.frames[1] == [] and len(result.issuance_index) == 0
    assert closed == [True]


def test_index_copies_original_quotes_and_preserves_duplicates():
    quote = Quote(100, 100, 102)
    index = IssuanceQuoteIndex([quote, quote, Quote(100, 99, 103)])
    object.__setattr__(quote, 'bid', 1)
    assert index.quotes == (Quote(100, 100, 102), Quote(100, 100, 102), Quote(100, 99, 103))
    assert index.timestamps == (100, 100, 100)


@pytest.mark.parametrize('rows', [[object()], [Quote(2, 100, 102), Quote(1, 100, 102)],
                                  [Quote(2**63, 100, 102)]])
def test_index_rejects_wrong_type_order_or_unbounded_stamp(rows):
    with pytest.raises(ValueError):
        IssuanceQuoteIndex(rows)


@pytest.mark.parametrize('side,unit,index', [(True, 1, IssuanceQuoteIndex([])),
                                          (0, 1, IssuanceQuoteIndex([])),
                                          (1, 0, IssuanceQuoteIndex([])),
                                          (1, 1, QuoteTape([]))])
def test_new_wrappers_validate_their_explicit_index_contract(side, unit, index):
    frames, _ = fixture('CRT_SWEEP_RECLAIM')
    with pytest.raises((ValueError, TypeError)):
        list(indexed_signals(frames, policy_id='CRT_SWEEP_RECLAIM', native_direction=side,
                             spread_unit=unit, issuance_index=index))
    with pytest.raises((ValueError, TypeError)):
        list(indexed_clock(frames, index, unit, side, 0, 100000000))
