"""Bounded quote access for a caller-declared calendar, not a study evaluator.

Only synthetic equivalence tests may be run before a future protocol binds this
adapter. It generates no signals, fits no parameters, establishes no coverage,
and makes no alpha, fee, fill, cash-profit or streaming-evaluator claim. Existing
frozen runners must not be handed an object pretending to be a full QuoteTape.

The loader owns validated daily source acquisition. This adapter keeps isolated
snapshots of at most two days and returns an actual local QuoteTape containing
the exact execution window. That tape can separately be supplied to the frozen
``_mark_trade`` for a COMPLETE outcome. No rows are sorted, deduplicated, filled,
or taken from outside the explicit half-open permitted calendar.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from typing import Callable

from independent_profit_research import execution
from independent_profit_research.execution import ExecutionConfig, TradeOutcome
from independent_profit_research.tape import QuoteTape


DAY_MS = 86_400_000
MAX_HOLD_MS = 3_600_000
MAX_LEG_LATENCY_MS = 3_600_000
MAX_TIMESTAMP_MS = 2**63 - 1


def _timestamp(value: int, name: str) -> int:
    value = execution._integer(value, name)
    if value > MAX_TIMESTAMP_MS:
        raise ValueError(name + ' exceeds the permitted timestamp range')
    return value


def _config_snapshot(config: ExecutionConfig | None) -> ExecutionConfig:
    if config is None:
        return ExecutionConfig()
    # Explicit types avoid property/subclass shadowing of validated values.
    if type(config) is not ExecutionConfig:
        raise ValueError('config must be an exact ExecutionConfig instance')
    config = ExecutionConfig(
        entry_latency_ms=config.entry_latency_ms,
        exit_latency_ms=config.exit_latency_ms,
        max_gap_ms=config.max_gap_ms,
        adverse_slippage_per_leg=config.adverse_slippage_per_leg,
    )
    if max(config.entry_latency_ms, config.exit_latency_ms) > MAX_LEG_LATENCY_MS:
        raise ValueError('each leg latency must not exceed one hour')
    return config


class ShardedQuoteStore:
    """Single-threaded, explicit-day access with a one- or two-day LRU cache.

    ``loader(day_start_ms)`` must return an exact QuoteTape containing only that
    UTC day. Empty shards are valid access sequences, never coverage proof. Every
    loaded tape is copied and revalidated, so later caller mutations cannot alter
    cached observations. Returned trade tapes are independent snapshots too.

    Horizons must follow an in-calendar decision by at most one hour and must not
    exceed the permitted calendar end. Each leg latency is also at most one hour;
    the ordinary quoted window therefore spans at most two UTC days. An empty or
    gapped archive may require looking through more *explicitly permitted* days
    to retain the first late bundle: it must not be silently truncated. The LRU
    limit remains two while those days are visited. No shard before the decision
    day or at/after the calendar end is requested.
    """

    __slots__ = ('_day_start_ms', '_day_end_ms', '_loader', '_cache_days', '_cache')

    def __init__(self, day_start_ms: int, day_end_ms: int,
                 loader: Callable[[int], QuoteTape], cache_days: int = 2):
        start = _timestamp(day_start_ms, 'day_start_ms')
        end = _timestamp(day_end_ms, 'day_end_ms')
        if start % DAY_MS or end % DAY_MS or end <= start:
            raise ValueError('source calendar must be aligned, nonempty and half-open')
        if not callable(loader):
            raise ValueError('loader must be callable')
        if type(cache_days) is not int or cache_days not in (1, 2):
            raise ValueError('cache_days must be integer 1 or 2')
        self._day_start_ms = start
        self._day_end_ms = end
        self._loader = loader
        self._cache_days = cache_days
        self._cache: OrderedDict[int, QuoteTape] = OrderedDict()

    @property
    def day_start_ms(self) -> int:
        return self._day_start_ms

    @property
    def day_end_ms(self) -> int:
        return self._day_end_ms

    @property
    def cache_days(self) -> int:
        return self._cache_days

    @property
    def cached_day_starts(self) -> tuple[int, ...]:
        """Access metadata in least-to-most-recent order; no quotes are exposed."""
        return tuple(self._cache)

    def _day_tape(self, day: int) -> QuoteTape:
        if day % DAY_MS or not self._day_start_ms <= day < self._day_end_ms:
            raise ValueError('attempted access outside the permitted source calendar')
        if day in self._cache:
            self._cache.move_to_end(day)
            return self._cache[day]
        # Release the oldest retained day before materializing another shard.
        # A failing loader can evict a cache entry, but cannot commit bad quotes.
        if len(self._cache) == self._cache_days:
            self._cache.popitem(last=False)
        supplied = self._loader(day)
        if type(supplied) is not QuoteTape:
            raise ValueError('loader must return an exact validated QuoteTape')
        snapshot = QuoteTape(supplied.quotes)
        if any(not day <= stamp < day + DAY_MS for stamp in snapshot.timestamps):
            raise ValueError('loaded quote timestamps must belong to the requested UTC day')
        self._cache[day] = snapshot
        self._cache.move_to_end(day)
        return snapshot

    def _window(self, decision_ms: int, horizon_end_ms: int,
                config: ExecutionConfig | None) -> tuple[int, int, ExecutionConfig]:
        decision = _timestamp(decision_ms, 'decision_ms')
        horizon = _timestamp(horizon_end_ms, 'horizon_end_ms')
        if horizon <= decision:
            raise ValueError('horizon must follow the issued decision')
        if not self._day_start_ms <= decision < self._day_end_ms:
            raise ValueError('decision must be inside the permitted source calendar')
        if horizon > self._day_end_ms:
            raise ValueError('horizon must not exceed the permitted source calendar end')
        if horizon - decision > MAX_HOLD_MS:
            raise ValueError('holding horizon must not exceed one hour')
        return decision, horizon, _config_snapshot(config)

    def trade_tape(self, decision_ms: int, horizon_end_ms: int,
                   config: ExecutionConfig | None = None) -> QuoteTape:
        """Copy first quote >= D through the entire first bundle >= H+latency.

        If the ceiling has no quote, include all remaining permitted quotes.
        Retaining the first arbitrarily late bundle preserves reference SKIP and
        UNKNOWN reasons; it does not assert native activity during empty shards.
        """
        decision, horizon, config = self._window(decision_ms, horizon_end_ms, config)
        ceiling = horizon + config.exit_latency_ms
        day = decision // DAY_MS * DAY_MS
        rows = []
        while day < self._day_end_ms:
            tape = self._day_tape(day)
            first = bisect_left(tape.timestamps, decision)
            last = bisect_left(tape.timestamps, ceiling, lo=first)
            if last < len(tape):
                end = bisect_right(tape.timestamps, tape.timestamps[last], lo=last)
                rows.extend(tape.quotes[first:end])
                break
            rows.extend(tape.quotes[first:])
            day += DAY_MS
        return QuoteTape(rows)

    def simulate(self, decision_ms: int, direction: int, horizon_end_ms: int,
                 stop_distance: float, target_distance: float | None = None,
                 config: ExecutionConfig | None = None) -> TradeOutcome:
        """Run only the unchanged conditional execution kernel on the local tape."""
        decision = _timestamp(decision_ms, 'decision_ms')
        horizon = _timestamp(horizon_end_ms, 'horizon_end_ms')
        if horizon <= decision:
            raise ValueError('horizon must follow the issued decision')
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError('direction must be +1 or -1')
        stop = execution._real(stop_distance, 'stop_distance', positive=True)
        target = (None if target_distance is None else
                  execution._real(target_distance, 'target_distance', positive=True))
        decision, horizon, config = self._window(decision, horizon, config)
        local = self.trade_tape(decision, horizon, config)
        return local.simulate(decision, direction, horizon, stop, target, config)
