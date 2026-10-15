"""Stream the frozen candle definition; keep only bars and issuance quotes.

This is input access for a future declared study. It does not run a strategy,
fit spreads or infer cash profit. Frozen v2 files are unchanged. A sparse quote
index supports signal issuance lookups only. It is neither a QuoteTape nor a
quote iterable, so the frozen runners reject it before any cropping or copying.
Execution requires the separately validated local tape from ShardedQuoteStore.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable

from independent_profit_research import execution
from independent_profit_research.data import MAX_SCALED_PRICE
from independent_profit_research.execution import Quote
from .bars import Bar, FRAME_MINUTES, MAX_GAP_MS, MINUTE_MS, _Minute, _aggregate


class IssuanceQuoteIndex:
    """Copied first confirmation quotes, deliberately incomplete for execution.

    This is not a QuoteTape subclass or an iterable execution input. Only the
    new issuance wrappers may consume its explicit scalar lookup metadata. The
    public immutable tuples remain inspectable evidence; deliberately extracting
    them into a fresh execution input would discard this access contract.
    """

    __slots__ = ('_quotes', '_timestamps')

    def __init__(self, quotes: Iterable[Quote]):
        rows, timestamps, previous = [], [], None
        for quote in quotes:
            if not isinstance(quote, Quote):
                raise ValueError('Issuance index requires Quote instances')
            stamp = execution._integer(quote.time_msc, 'time_msc')
            if stamp > 2**63 - 1:
                raise ValueError('Issuance timestamps must be bounded')
            bid = execution._real(quote.bid, 'bid', positive=True)
            ask = execution._real(quote.ask, 'ask', positive=True)
            if bid > ask:
                raise ValueError('bid must not exceed ask')
            if previous is not None and stamp < previous:
                raise ValueError('Issuance timestamps must be nondecreasing; no sorting')
            rows.append(Quote(stamp, bid, ask))
            timestamps.append(stamp)
            previous = stamp
        self._quotes, self._timestamps = tuple(rows), tuple(timestamps)

    @property
    def quotes(self) -> tuple[Quote, ...]:
        return self._quotes

    @property
    def timestamps(self) -> tuple[int, ...]:
        return self._timestamps

    def __len__(self) -> int:
        return len(self._quotes)

    def simulate(self, *args, **kwargs):
        raise ValueError('Sparse issuance quotes cannot be used for execution')


@dataclass(frozen=True)
class StreamedFrames:
    frames: dict[int, list[Bar]]
    issuance_index: IssuanceQuoteIndex
    source_rows_within_window: int


def build_streamed_frames(quotes: Iterable[Quote], start_ms: int, end_ms: int) -> StreamedFrames:
    """Reproduce all frozen candle fields without retaining every source row.

The caller declares the half-open source calendar and its ordered iterator.
Every consumed in-window row is validated and snapshotted. Equal timestamps
remain bundled, including a conflicting final row. No sorting, interpolation
or successor outside end_ms is allowed. Iterator close is propagated.

Memory grows with candle count, not quote count. The issuance index contains
the actual first quote that confirmed each valid M1 bar. The new wrappers reuse
the frozen policy helpers for end/confirmation lookups, matching a complete
tape. Frozen public policies/runners intentionally reject this separate type;
the index is never a substitute for the execution quote path.
"""
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms < 0
            or end_ms <= start_ms or end_ms > 2**63 - 1):
        raise ValueError('Expected bounded nonnegative half-open millisecond window')
    iterator = iter(quotes)
    minutes, confirmations = [], []
    active = None
    run_id, poisoned_through_minute = 0, -1

    def consume(pending: Quote, ambiguous: bool, successor: Quote):
        nonlocal active, run_id, poisoned_through_minute
        stamp, next_stamp = pending.time_msc, successor.time_msc
        opening = stamp // MINUTE_MS * MINUTE_MS
        if active is None:
            active = _Minute(opening, stamp, stamp, run_id,
                             opening >= start_ms and stamp - opening <= MAX_GAP_MS
                             and opening > poisoned_through_minute)
        elif active.open_ms != opening:
            raise ValueError('Unexpected unconfirmed minute transition')
        if ambiguous:
            active.valid = False
            poisoned_through_minute = max(poisoned_through_minute, opening)
            run_id += 1
        else:
            active.observe(stamp, (pending.bid + pending.ask) / 2)
        if next_stamp - stamp > MAX_GAP_MS:
            active.valid = False
            poisoned_through_minute = max(poisoned_through_minute,
                                         next_stamp // MINUTE_MS * MINUTE_MS)
            run_id += 1
        if next_stamp >= opening + MINUTE_MS:
            if (active.valid and active.closing is not None
                    and opening + MINUTE_MS - active.last_ms <= MAX_GAP_MS):
                minutes.append(Bar(opening, opening + MINUTE_MS, next_stamp,
                                   active.opening, active.high, active.low,
                                   active.closing, active.run_id))
                confirmations.append(successor)
            active = None

    previous_stamp = pending = None
    ambiguous, rows = False, 0
    try:
        for supplied in iterator:
            if (not isinstance(supplied, Quote) or type(supplied.time_msc) is not int
                    or supplied.time_msc < 0 or supplied.time_msc > 2**63 - 1):
                raise ValueError('Expected Quote snapshots with bounded integer timestamps')
            stamp = supplied.time_msc
            if previous_stamp is not None and stamp < previous_stamp:
                raise ValueError('Quote timestamps must remain nondecreasing; no sorting')
            previous_stamp = stamp
            if stamp < start_ms:
                continue
            if stamp >= end_ms:
                break
            for price in (supplied.bid, supplied.ask):
                if (isinstance(price, bool) or not isinstance(price, Real)
                        or not math.isfinite(price) or not float(price).is_integer()
                        or not 0 < price < MAX_SCALED_PRICE):
                    raise ValueError('Expected finite positive integer-scaled Bid/Ask below 2**45')
            if supplied.bid > supplied.ask:
                raise ValueError('Bid must not exceed Ask')
            quote = Quote(stamp, float(supplied.bid), float(supplied.ask))
            rows += 1
            if pending is None:
                pending = quote
            elif stamp == pending.time_msc:
                ambiguous |= (quote.bid != pending.bid or quote.ask != pending.ask)
            else:
                consume(pending, ambiguous, quote)
                pending, ambiguous = quote, False
    finally:
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()
    # The final timestamp has no distinct observed in-window successor.
    frames = {size: minutes if size == 1 else _aggregate(minutes, size, start_ms, end_ms)
              for size in FRAME_MINUTES}
    return StreamedFrames(frames, IssuanceQuoteIndex(confirmations), rows)
