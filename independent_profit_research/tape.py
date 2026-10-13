"""Once-validated quote index for exact slices of the unchanged execution model.

This helper changes input access only. It does not calculate strategy returns,
fit parameters, sort, deduplicate, interpolate or infer missing observations.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Iterable, Iterator

from . import execution
from .execution import ExecutionConfig, Quote, TradeOutcome


class QuoteTape:
    """Own validated ordered Quote snapshots, including every duplicate row.

    Supply an iterator to avoid holding a second complete input collection.
    Plain frozen snapshots isolate the index from later mutation/subclasses of
    the caller's objects. ``quotes`` is also the full-reference comparison input.
    """
    __slots__ = ('_quotes', '_timestamps')

    def __init__(self, quotes: Iterable[Quote]):
        rows, timestamps = [], []
        previous = None
        for quote in quotes:
            if not isinstance(quote, Quote):
                raise ValueError('quotes must contain Quote instances')
            stamp = execution._integer(quote.time_msc, 'time_msc')
            bid = execution._real(quote.bid, 'bid', positive=True)
            ask = execution._real(quote.ask, 'ask', positive=True)
            if bid > ask:
                raise ValueError('bid must not exceed ask')
            if previous is not None and stamp < previous:
                raise ValueError('quote timestamps must be nondecreasing; no sorting is performed')
            rows.append(Quote(stamp, bid, ask))
            timestamps.append(stamp)
            previous = stamp
        self._quotes = tuple(rows)
        self._timestamps = tuple(timestamps)

    @property
    def quotes(self) -> tuple[Quote, ...]:
        return self._quotes

    @property
    def timestamps(self) -> tuple[int, ...]:
        return self._timestamps

    def __len__(self) -> int:
        return len(self._quotes)

    def __iter__(self) -> Iterator[Quote]:
        return iter(self._quotes)

    def simulate(self, decision_ms: int, direction: int, horizon_end_ms: int,
                 stop_distance: float, target_distance: float | None = None,
                 config: ExecutionConfig | None = None) -> TradeOutcome:
        """Return every field of the full-reference result without an O(N) scan.

        Include all rows from the first timestamp >= decision through the
        entire first bundle >= horizon+exit latency. A confirmed barrier must
        precede horizon, so its exit deadline cannot exceed this ceiling.
        Keeping the first later bundle also preserves SKIP versus missing-entry
        results and missing-path/gap evidence when the next quote is very late.
        """
        # Match reference validation order before arithmetic/bisection. Trading
        # calculations and outcome construction remain exclusively in execution.
        decision_ms = execution._integer(decision_ms, 'decision_ms')
        horizon_end_ms = execution._integer(horizon_end_ms, 'horizon_end_ms')
        if horizon_end_ms <= decision_ms:
            raise ValueError('horizon must follow the issued decision')
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError('direction must be +1 or -1')
        stop_distance = execution._real(stop_distance, 'stop_distance', positive=True)
        if target_distance is not None:
            target_distance = execution._real(target_distance, 'target_distance', positive=True)
        config = ExecutionConfig() if config is None else config
        if not isinstance(config, ExecutionConfig):
            raise ValueError('config must be ExecutionConfig')
        execution._integer(config.entry_latency_ms, 'entry_latency_ms')
        execution._integer(config.exit_latency_ms, 'exit_latency_ms')
        execution._integer(config.max_gap_ms, 'max_gap_ms', positive=True)
        execution._real(config.adverse_slippage_per_leg, 'adverse_slippage_per_leg')
        start = bisect_left(self._timestamps, decision_ms)
        ceiling = bisect_left(self._timestamps, horizon_end_ms+config.exit_latency_ms, lo=start)
        end = (len(self._quotes) if ceiling == len(self._quotes)
               else bisect_right(self._timestamps, self._timestamps[ceiling], lo=ceiling))
        return execution.simulate_trade(self._quotes[start:end], decision_ms, direction,
                                        horizon_end_ms, stop_distance, target_distance, config)
