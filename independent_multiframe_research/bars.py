"""Causal UTC midpoint bars from an explicitly bounded validated quote tape.

``build_frames(tape, start_ms, end_ms)`` returns only valid completed bars for
1/5/15/60/240 minutes. Prices retain the input 1e-8 scale; midpoints are exact
integer/half-integer Float64 values. ``confirmed_ms`` is an observed first quote
in a later minute, never a synthetic boundary. The half-open source window does
not borrow a quote at/after end_ms to complete its last unfinished minute.

Missing/ambiguous minutes are absent. Consumers must require the latest expected
UTC bar and consecutive same-run history, rather than substitute an older valid
bar. This module does not fit features, calculate outcomes, or mutate raw quotes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from independent_profit_research.data import MAX_SCALED_PRICE
from independent_profit_research.execution import Quote
from independent_profit_research.tape import QuoteTape

MINUTE_MS = 60000
MAX_GAP_MS = 5000
FRAME_MINUTES = (1, 5, 15, 60, 240)


@dataclass(frozen=True)
class Bar:
    open_ms: int
    end_ms: int
    confirmed_ms: int
    open: float
    high: float
    low: float
    close: float
    run_id: int

    def __post_init__(self):
        for name in ('open_ms', 'end_ms', 'confirmed_ms', 'run_id'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name+' must be a nonnegative integer')
        if self.end_ms <= self.open_ms or self.confirmed_ms < self.end_ms:
            raise ValueError('Bar confirmation must follow its positive interval')
        for name in ('open', 'high', 'low', 'close'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError('Bar prices must be positive finite numbers')
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Bar prices must be positive finite numbers')
            object.__setattr__(self, name, value)
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError('Inconsistent bar OHLC bounds')


@dataclass
class _Minute:
    open_ms: int
    first_ms: int
    last_ms: int
    run_id: int
    valid: bool
    opening: float | None = None
    high: float | None = None
    low: float | None = None
    closing: float | None = None

    def observe(self, stamp: int, mid: float) -> None:
        if self.opening is None:
            self.opening = self.high = self.low = mid
        self.high = max(self.high, mid)
        self.low = min(self.low, mid)
        self.closing = mid
        self.last_ms = stamp


def _aggregate(minutes: list[Bar], size: int, start_ms: int, end_ms: int) -> list[Bar]:
    width = size*MINUTE_MS
    groups: dict[int, list[Bar]] = {}
    for minute in minutes:
        opening = minute.open_ms//width*width
        if opening >= start_ms and opening+width <= end_ms:
            groups.setdefault(opening, []).append(minute)
    output = []
    for opening, group in groups.items():
        if len(group) != size:
            continue
        if any(bar.open_ms != opening+i*MINUTE_MS or bar.end_ms != opening+(i+1)*MINUTE_MS
               for i,bar in enumerate(group)):
            continue
        if any(bar.run_id != group[0].run_id for bar in group):
            continue
        output.append(Bar(opening, opening+width, max(bar.confirmed_ms for bar in group),
                          group[0].open, max(bar.high for bar in group),
                          min(bar.low for bar in group), group[-1].close, group[0].run_id))
    return output


def build_frames(tape: QuoteTape, start_ms: int, end_ms: int) -> dict[int, list[Bar]]:
    """Build completed midpoint candles, preserving gaps and same-ms ambiguity.

    A valid minute has observed quotes within five seconds of each edge, no
    >5000ms internal/neighbor interval, and unambiguous Bid/Ask bundles. Source
    gaps restart run ids and invalidate both endpoint minutes and all crossed
    minutes. Equal-ms identical pairs are harmless duplicates, kept in ``tape``.
    First/last partial or unconfirmed minutes never produce a fabricated bar.
    """
    if not isinstance(tape, QuoteTape):
        raise ValueError('tape must be a validated QuoteTape')
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms < 0
            or end_ms <= start_ms or end_ms > 2**63-1):
        raise ValueError('Expected bounded nonnegative half-open millisecond window')

    minutes: list[Bar] = []
    active: _Minute | None = None
    run_id = 0
    poisoned_through_minute = -1

    def consume(stamp: int, bid: float, ask: float, ambiguous: bool, next_stamp: int) -> None:
        nonlocal active, run_id, poisoned_through_minute
        opening = stamp//MINUTE_MS*MINUTE_MS
        if active is None:
            active = _Minute(opening, stamp, stamp, run_id,
                opening >= start_ms and stamp-opening <= MAX_GAP_MS
                and opening > poisoned_through_minute)
        elif active.open_ms != opening:
            raise ValueError('Unexpected unconfirmed minute transition')
        if ambiguous:
            active.valid = False
            poisoned_through_minute = max(poisoned_through_minute, opening)
            run_id += 1
        else:
            active.observe(stamp, (bid+ask)/2)

        # next_stamp is known exactly when its first row arrives. Its prices
        # and any later equal-ms rows are excluded from the closing bar.
        if next_stamp-stamp > MAX_GAP_MS:
            active.valid = False
            poisoned_through_minute = max(poisoned_through_minute,
                                         next_stamp//MINUTE_MS*MINUTE_MS)
            run_id += 1
        if next_stamp >= opening+MINUTE_MS:
            if (active.valid and active.closing is not None
                    and opening+MINUTE_MS-active.last_ms <= MAX_GAP_MS):
                minutes.append(Bar(opening, opening+MINUTE_MS, next_stamp,
                    active.opening, active.high, active.low, active.closing, active.run_id))
            active = None

    pending_stamp = None
    pending_bid = pending_ask = None
    pending_ambiguous = False
    previous_stamp = None
    for quote in tape:
        if not isinstance(quote, Quote) or type(quote.time_msc) is not int or quote.time_msc < 0:
            raise ValueError('Expected Quote snapshots with nonnegative integer timestamps')
        stamp = quote.time_msc
        if previous_stamp is not None and stamp < previous_stamp:
            raise ValueError('Quote timestamps must remain nondecreasing; no sorting')
        previous_stamp = stamp
        if stamp < start_ms:
            continue
        if stamp >= end_ms:
            break
        for price in (quote.bid, quote.ask):
            if (isinstance(price, bool) or not isinstance(price, Real)
                    or not math.isfinite(price) or not float(price).is_integer()
                    or not 0 < price < MAX_SCALED_PRICE):
                raise ValueError('Expected finite positive integer-scaled Bid/Ask below 2**45')
        if quote.bid > quote.ask:
            raise ValueError('Bid must not exceed Ask')
        if pending_stamp is None:
            pending_stamp, pending_bid, pending_ask = stamp, quote.bid, quote.ask
        elif stamp == pending_stamp:
            pending_ambiguous |= (quote.bid != pending_bid or quote.ask != pending_ask)
        else:
            consume(pending_stamp, pending_bid, pending_ask, pending_ambiguous, stamp)
            pending_stamp, pending_bid, pending_ask = stamp, quote.bid, quote.ask
            pending_ambiguous = False
    # The pending last timestamp has no observed distinct successor. In
    # particular, end_ms itself is neither a quote nor a confirmation event.
    return {size: minutes if size == 1 else _aggregate(minutes,size,start_ms,end_ms)
            for size in FRAME_MINUTES}
