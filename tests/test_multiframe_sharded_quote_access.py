"""Synthetic access-equivalence checks; no source, alpha or strategy measurement."""
from bisect import bisect_left, bisect_right
from dataclasses import asdict
import random

import pytest

from independent_multiframe_research.sharded import (
    DAY_MS, MAX_HOLD_MS, ShardedQuoteStore,
)
from independent_profit_research.execution import ExecutionConfig, Quote
from independent_profit_research.runner import _mark_trade
from independent_profit_research.tape import QuoteTape


def store_for(quotes, days=3, cache_days=2):
    full = QuoteTape(quotes)
    calls = []

    def loader(day):
        assert 0 <= day < days * DAY_MS and day % DAY_MS == 0
        calls.append(day)
        lo = bisect_left(full.timestamps, day)
        hi = bisect_left(full.timestamps, day + DAY_MS)
        return QuoteTape(full.quotes[lo:hi])

    return full, ShardedQuoteStore(0, days * DAY_MS, loader, cache_days), calls


def expected_slice(full, decision, horizon, config):
    lo = bisect_left(full.timestamps, decision)
    at = bisect_left(full.timestamps, horizon + config.exit_latency_ms, lo=lo)
    hi = len(full) if at == len(full) else bisect_right(full.timestamps, full.timestamps[at], lo=at)
    return full.quotes[lo:hi]


class Marks:
    def __init__(self):
        self.marked_rows = 0
        self.values = []

    def mark(self, value, stamp):
        self.values.append((value, stamp))


def assert_equivalent(full, store, decision, horizon, direction, stop, target, config):
    local = store.trade_tape(decision, horizon, config)
    assert type(local) is QuoteTape
    assert local.quotes == expected_slice(full, decision, horizon, config)
    actual = store.simulate(decision, direction, horizon, stop, target, config)
    reference = full.simulate(decision, direction, horizon, stop, target, config)
    assert asdict(actual) == asdict(reference)
    if actual.status == 'COMPLETE':
        left, right = Marks(), Marks()
        assert _mark_trade(local, actual, 2.5, left) == _mark_trade(full, reference, 2.5, right)
        assert vars(left) == vars(right)
    assert len(store.cached_day_starts) <= store.cache_days <= 2
    return actual


@pytest.mark.parametrize('days', [2, 3])
@pytest.mark.parametrize('seed', range(12))
def test_seeded_two_and_three_day_all_field_reference_equivalence(days, seed):
    rng = random.Random(9700 + seed)
    times = set()
    for day in range(days):
        # Include empty shards in a deterministic subset of three-day fixtures.
        if days == 3 and day == 1 and seed % 3 == 0:
            continue
        for center in (day * DAY_MS + 1000, (day + 1) * DAY_MS - 4000):
            for _ in range(35):
                t = center + rng.randrange(-900, 2900)
                if day * DAY_MS <= t < (day + 1) * DAY_MS:
                    times.add(t)
    quotes = []
    for stamp in sorted(times):
        for _ in range(1 + (rng.random() < 0.3)):
            bid = 100 + rng.randrange(-800, 801) / 100
            quotes.append(Quote(stamp, bid, bid + rng.choice([0.0, 0.1, 0.7])))
    full, store, _ = store_for(quotes, days)
    configs = [ExecutionConfig(0, 0, 5000, 0), ExecutionConfig(250, 250, 5000, 0.25),
               ExecutionConfig(2000, 2000, 1000, 0.5),
               ExecutionConfig(0, 250, days * DAY_MS, 0.1)]
    for config in configs:
        for decision in (0, 1000, DAY_MS - 3000, DAY_MS - 100, DAY_MS + 500):
            horizon = min(decision + 2800, days * DAY_MS)
            for direction in (-1, 1):
                for target in (None, 4.0):
                    assert_equivalent(full, store, decision, horizon, direction, 3.0, target, config)


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('latency', [0, 250, 2000])
def test_stop_priority_entire_same_ms_bundles_cross_midnight(direction, latency):
    d = DAY_MS - 4000
    quotes = [Quote(d, 100, 100), Quote(d + 500, 100, 100), Quote(d + 2000, 100, 100),
              Quote(d + 2100, 90, 90), Quote(d + 2100, 110, 110),
              Quote(d + 2100, 90, 90), Quote(d + 2500, 99, 101),
              Quote(DAY_MS, 98, 102), Quote(DAY_MS, 97, 103),
              Quote(DAY_MS + 2000, 96, 104), Quote(DAY_MS + 2000, 95, 105),
              Quote(DAY_MS + 3000, 100, 100)]
    full, store, calls = store_for(quotes, 2)
    config = ExecutionConfig(latency, latency, 5000, 0)
    result = assert_equivalent(full, store, d, DAY_MS + 1000, direction, 5, 5, config)
    assert result.status == 'COMPLETE'
    assert result.both_barriers_in_bundle
    assert result.reason == 'STOP_PRIORITY_SAME_MILLISECOND'
    assert set(calls) <= {0, DAY_MS}


@pytest.mark.parametrize('decision,quotes,expected_status,expected_reason', [
    (100, [], 'UNKNOWN', 'MISSING_ENTRY_QUOTE'),
    (100, [Quote(2 * DAY_MS + 10, 100, 101)], 'SKIP', 'NO_ENTRY_BEFORE_HORIZON'),
    (100, [Quote(100, 100, 101), Quote(2 * DAY_MS + 10, 100, 101)], 'UNKNOWN', 'GAP_BEFORE_EXIT'),
    (100, [Quote(100, 100, 101), Quote(6000, 100, 101)], 'UNKNOWN', 'GAP_DURING_HOLD'),
])
def test_empty_shards_and_first_arbitrarily_late_bundle_preserve_reason(
        decision, quotes, expected_status, expected_reason):
    full, store, calls = store_for(quotes)
    horizon = 7000 if expected_reason == 'GAP_DURING_HOLD' else 1000
    result = assert_equivalent(full, store, decision, horizon, 1, 50, None, ExecutionConfig(0, 0))
    assert (result.status, result.reason) == (expected_status, expected_reason)
    assert all(0 <= day < 3 * DAY_MS for day in calls)


def test_half_open_calendar_end_does_not_fetch_exit_outside_calendar():
    full, store, calls = store_for([Quote(DAY_MS - 1000, 100, 101), Quote(DAY_MS - 1, 100, 101)], 1)
    result = assert_equivalent(full, store, DAY_MS - 1000, DAY_MS, 1, 50, None, ExecutionConfig(0, 250))
    assert result.status == 'UNKNOWN'
    assert result.reason == 'MISSING_HORIZON_EXIT_QUOTE'
    assert calls == [0]


def test_cache_uses_lru_and_never_exceeds_two_days():
    calls = []

    def loader(day):
        calls.append(day)
        assert len(store.cached_day_starts) <= 2
        return QuoteTape([Quote(day + 10, 100, 101), Quote(day + 1000, 100, 101)])

    store = ShardedQuoteStore(0, 5 * DAY_MS, loader)
    for day in (0, DAY_MS, 0, 2 * DAY_MS):
        store.trade_tape(day + 10, day + 500, ExecutionConfig(0, 0))
    assert calls == [0, DAY_MS, 2 * DAY_MS]
    assert store.cached_day_starts == (0, 2 * DAY_MS)
    store.trade_tape(DAY_MS + 10, DAY_MS + 500, ExecutionConfig(0, 0))
    assert calls[-1] == DAY_MS
    assert store.cached_day_starts == (2 * DAY_MS, DAY_MS)


def test_empty_days_keep_lru_bound_while_searching_late_bundle():
    calls = []

    def loader(day):
        assert len(store.cached_day_starts) <= 2
        calls.append(day)
        return QuoteTape([Quote(day + 1, 100, 101)] if day == 5 * DAY_MS else [])

    store = ShardedQuoteStore(0, 6 * DAY_MS, loader)
    tape = store.trade_tape(1, 1000)
    assert tape.quotes == (Quote(5 * DAY_MS + 1, 100, 101),)
    assert calls == list(range(0, 6 * DAY_MS, DAY_MS))
    assert store.cached_day_starts == (4 * DAY_MS, 5 * DAY_MS)


def test_first_late_bundle_keeps_every_row_and_no_later_rows():
    rows = [Quote(2 * DAY_MS + 1, 100, 101), Quote(2 * DAY_MS + 1, 99, 102),
            Quote(2 * DAY_MS + 1, 99, 102), Quote(2 * DAY_MS + 1000, 103, 104)]
    full, store, calls = store_for(rows)
    assert store.trade_tape(1, 1000).quotes == tuple(rows[:3])
    assert calls == [0, DAY_MS, 2 * DAY_MS]
    assert_equivalent(full, store, 1, 1000, 1, 10, None, ExecutionConfig())


def test_ceiling_bundle_on_decision_day_does_not_discover_later_shards():
    full, store, calls = store_for([Quote(10, 100, 101), Quote(1000, 100, 101),
                                  Quote(1000, 99, 102), Quote(1000, 99, 102),
                                  Quote(DAY_MS, 101, 102)])
    assert_equivalent(full, store, 10, 750, 1, 10, None, ExecutionConfig(0, 250))
    assert calls == [0]


@pytest.mark.parametrize('direction', [-1, 1])
def test_entry_deadline_at_horizon_preserves_skip(direction):
    full, store, _ = store_for([Quote(0, 100, 101), Quote(1000, 100, 101)], 1)
    result = assert_equivalent(full, store, 0, 1000, direction, 5, None, ExecutionConfig(1000, 0))
    assert (result.status, result.reason) == ('SKIP', 'ENTRY_LATENCY_REACHES_HORIZON')


@pytest.mark.parametrize('direction', [-1, 1])
def test_gap_before_entry_preserves_unknown(direction):
    full, store, _ = store_for([Quote(6000, 100, 101), Quote(10000, 100, 101)], 1)
    result = assert_equivalent(full, store, 0, 10000, direction, 50, None, ExecutionConfig(0, 0))
    assert (result.status, result.reason) == ('UNKNOWN', 'GAP_BEFORE_ENTRY')


@pytest.mark.parametrize('cache_days', [1, 2])
def test_loader_and_returned_tape_are_copy_isolated(cache_days):
    supplied = QuoteTape([Quote(10, 100, 101), Quote(1000, 102, 103)])
    store = ShardedQuoteStore(0, DAY_MS, lambda _: supplied, cache_days)
    original = store.trade_tape(10, 500, ExecutionConfig(0, 0))
    object.__setattr__(supplied.quotes[0], 'bid', 1.0)
    object.__setattr__(supplied, '_quotes', ())
    object.__setattr__(original.quotes[0], 'ask', 999.0)
    fresh = store.trade_tape(10, 500, ExecutionConfig(0, 0))
    assert fresh.quotes == (Quote(10, 100, 101), Quote(1000, 102, 103))
    assert fresh is not original


@pytest.mark.parametrize('start,end,loader,cache_days', [
    (True, DAY_MS, lambda _: QuoteTape([]), 2), (-DAY_MS, DAY_MS, lambda _: QuoteTape([]), 2),
    (1, DAY_MS, lambda _: QuoteTape([]), 2), (0, DAY_MS + 1, lambda _: QuoteTape([]), 2),
    (DAY_MS, DAY_MS, lambda _: QuoteTape([]), 2), (0, 0, lambda _: QuoteTape([]), 2),
    (0, DAY_MS, None, 2), (0, DAY_MS, lambda _: QuoteTape([]), 0),
    (0, DAY_MS, lambda _: QuoteTape([]), 3), (0, DAY_MS, lambda _: QuoteTape([]), True),
    (0, DAY_MS, lambda _: QuoteTape([]), 1.0),
])
def test_invalid_constructor_rejected(start, end, loader, cache_days):
    with pytest.raises(ValueError):
        ShardedQuoteStore(start, end, loader, cache_days)


@pytest.mark.parametrize('start,end', [(0.0, DAY_MS), (0, 2**63), (2**63, 2**63 + DAY_MS)])
def test_constructor_timestamp_type_and_upper_bound(start, end):
    with pytest.raises(ValueError):
        ShardedQuoteStore(start, end, lambda _: QuoteTape([]))


@pytest.mark.parametrize('decision,horizon,config', [
    (True, 1000, None), (-1, 1000, None), (1000, 1000, None),
    (DAY_MS, DAY_MS + 1000, None), (DAY_MS - 10, DAY_MS + 1, None),
    (0, MAX_HOLD_MS + 1, None), (0, 1000, {}),
    (0, 1000, ExecutionConfig(3_600_001, 0)),
    (0, 1000, ExecutionConfig(0, 3_600_001)),
])
def test_invalid_window_or_config_fails_without_loader_calls(decision, horizon, config):
    calls = []
    store = ShardedQuoteStore(0, DAY_MS, lambda day: calls.append(day))
    with pytest.raises(ValueError):
        store.trade_tape(decision, horizon, config)
    assert calls == []


@pytest.mark.parametrize('supplied', [[], None, QuoteTape([Quote(DAY_MS, 100, 101)])])
def test_loader_type_and_half_open_day_validation(supplied):
    store = ShardedQuoteStore(0, DAY_MS, lambda _: supplied)
    with pytest.raises(ValueError):
        store.trade_tape(0, 1000)
    assert store.cached_day_starts == ()


def test_corrupted_loader_tape_order_and_quote_types_revalidated():
    for rows in [(Quote(1000, 100, 101), Quote(10, 100, 101)), (object(),)]:
        supplied = QuoteTape([])
        object.__setattr__(supplied, '_quotes', rows)
        store = ShardedQuoteStore(0, DAY_MS, lambda _: supplied)
        with pytest.raises(ValueError):
            store.trade_tape(0, 1000)
        assert store.cached_day_starts == ()


def test_corrupted_loader_timestamp_and_prices_revalidated():
    for field, value in [('time_msc', True), ('time_msc', -1), ('bid', float('nan')),
                         ('ask', float('inf')), ('bid', 200), ('ask', 0)]:
        supplied = QuoteTape([Quote(10, 100, 101)])
        object.__setattr__(supplied.quotes[0], field, value)
        store = ShardedQuoteStore(0, DAY_MS, lambda _: supplied)
        with pytest.raises(ValueError):
            store.trade_tape(0, 1000)
        assert store.cached_day_starts == ()


def test_loader_timestamps_index_is_not_trusted_over_original_rows():
    supplied = QuoteTape([Quote(10, 100, 101), Quote(1000, 102, 103)])
    object.__setattr__(supplied, '_timestamps', (5000, 6000))
    store = ShardedQuoteStore(0, DAY_MS, lambda _: supplied)
    assert store.trade_tape(0, 500, ExecutionConfig(0, 0)).timestamps == (10, 1000)


def test_quote_tape_and_config_subclass_shadowing_rejected():
    class ShadowTape(QuoteTape):
        @property
        def quotes(self):
            return ()

    class ShadowConfig(ExecutionConfig):
        pass

    store = ShardedQuoteStore(0, DAY_MS, lambda _: ShadowTape([Quote(10, 100, 101)]))
    with pytest.raises(ValueError, match='exact validated QuoteTape'):
        store.trade_tape(0, 1000)
    with pytest.raises(ValueError, match='exact ExecutionConfig'):
        store.trade_tape(0, 1000, ShadowConfig())


def test_mutated_config_revalidated_and_valid_config_snapshot_isolated():
    config = ExecutionConfig(0, 0)
    full, store, _ = store_for([Quote(10, 100, 101), Quote(1000, 100, 101)], 1)
    actual = assert_equivalent(full, store, 10, 500, 1, 50, None, config)
    object.__setattr__(config, 'max_gap_ms', 0)
    assert actual.config.max_gap_ms == 5000
    with pytest.raises(ValueError):
        store.trade_tape(10, 500, config)


@pytest.mark.parametrize('field,value', [('entry_latency_ms', -1), ('exit_latency_ms', True),
                                       ('max_gap_ms', 0), ('adverse_slippage_per_leg', float('inf'))])
def test_each_mutated_config_field_validated_before_loading(field, value):
    config = ExecutionConfig()
    object.__setattr__(config, field, value)
    calls = []
    store = ShardedQuoteStore(0, DAY_MS, lambda day: calls.append(day))
    with pytest.raises(ValueError):
        store.trade_tape(0, 1000, config)
    assert calls == []


@pytest.mark.parametrize('direction,stop,target', [(True, 1, None), (0, 1, None), (1, 0, None),
                                               (1, 1, float('nan')), (1, 1, True)])
def test_invalid_simulation_arguments_fail_without_loader_calls(direction, stop, target):
    calls = []
    store = ShardedQuoteStore(0, DAY_MS, lambda day: calls.append(day))
    with pytest.raises(ValueError):
        store.simulate(0, direction, 1000, stop, target)
    assert calls == []


def test_no_loader_call_before_nonzero_permitted_calendar_start():
    calls = []

    def loader(day):
        calls.append(day)
        assert day == 10 * DAY_MS
        return QuoteTape([Quote(day + 100, 100, 101), Quote(day + 2000, 100, 101)])

    store = ShardedQuoteStore(10 * DAY_MS, 11 * DAY_MS, loader)
    assert len(store.trade_tape(10 * DAY_MS + 100, 10 * DAY_MS + 1000)) == 2
    assert calls == [10 * DAY_MS]
