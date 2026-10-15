"""Pure streaming quote hypotheses; no fitting, execution, account or data access.

A distinct timestamp closes the preceding bundle. Its first quote is the actual
decision clock and may supply a spread filter, but never historical features.
Different prices within a completed same-ms bundle reset feature state rather
than inventing a within-bundle order. The final input bundle is never flushed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Iterator, Protocol


POLICY_IDS = frozenset({'SIDE_LEAD', 'QUOTE_CLOCK_ACCELERATION', 'POST_JUMP_CARRY_30S'})


class QuoteLike(Protocol):
    time_msc: int
    bid: float
    ask: float


@dataclass(frozen=True)
class Signal:
    policy_id: str
    direction: int
    decision_ms: int
    horizon_end_ms: int
    stop_distance: float
    target_distance: float | None


@dataclass(frozen=True)
class _Point:
    time_msc: int
    bid: float
    ask: float

    @property
    def mid(self):
        return self.bid + (self.ask - self.bid) / 2

    @property
    def spread(self):
        return self.ask - self.bid


@dataclass(frozen=True)
class _Event:
    time_msc: int
    prior_mid: float
    mid: float
    bid_delta: float
    ask_delta: float
    source_gap_ms: int

    @property
    def one_side_direction(self):
        if (self.bid_delta != 0) == (self.ask_delta != 0):
            return 0
        delta = self.bid_delta if self.bid_delta else self.ask_delta
        return 1 if delta > 0 else -1


class _State:
    def __init__(self, policy_id):
        self.policy_id = policy_id
        self.cooldown_until = 0
        self.reset()

    def reset(self):
        # A loss of observation cancels a pending signal, never its cooldown.
        self.last = None
        self.coverage_start = None
        self.events = deque(maxlen=8 if self.policy_id == 'SIDE_LEAD' else None)
        self.pending = None

    def close(self, point):
        event = None
        if self.last is None:
            self.coverage_start = point.time_msc
        elif point.bid != self.last.bid or point.ask != self.last.ask:
            event = _Event(point.time_msc, self.last.mid, point.mid,
                           point.bid-self.last.bid, point.ask-self.last.ask,
                           point.time_msc-self.last.time_msc)
            if self.policy_id != 'POST_JUMP_CARRY_30S':
                self.events.append(event)
        self.last = point
        if self.policy_id == 'QUOTE_CLOCK_ACCELERATION':
            lower = point.time_msc - 315000
            while self.events and self.events[0].time_msc <= lower:
                self.events.popleft()
        return event


def _point(quote):
    try:
        stamp, bid, ask = quote.time_msc, quote.bid, quote.ask
    except AttributeError as error:
        raise ValueError('Quotes require time_msc, bid and ask') from error
    if type(stamp) is not int or stamp < 0:
        raise ValueError('Quote timestamp must be a nonnegative integer')
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0
           for x in (bid, ask)) or bid > ask:
        raise ValueError('Quotes require finite positive noncrossed bid and ask')
    return _Point(stamp, float(bid), float(ask))


def _side_signal(state, current, native_direction, unit):
    if len(state.events) != 8 or current.spread > unit:
        return None
    if abs(state.last.mid-state.events[0].prior_mid) > unit:
        return None
    directions = [event.one_side_direction for event in state.events]
    if directions.count(native_direction) >= 3 and -native_direction not in directions[-3:]:
        return Signal('SIDE_LEAD', native_direction, current.time_msc,
                      current.time_msc+5000, 2*unit, 2*unit)
    return None


def _clock_signal(state, current, native_direction, unit):
    last = state.last.time_msc
    if last-state.coverage_start < 315000:
        return None
    boundary = last-15000
    recent = [event for event in state.events if event.time_msc > boundary]
    baseline = len(state.events)-len(recent)
    if (baseline <= 0 or len(recent) < 12 or len(recent)*40 < baseline*3
            or any(abs(event.mid-event.prior_mid) >= 4*unit for event in recent)):
        return None
    return Signal('QUOTE_CLOCK_ACCELERATION', native_direction, current.time_msc,
                  current.time_msc+30000, 2*unit, 2*unit)


def _carry_signal(state, event, current, native_direction, unit, fresh):
    result = None
    if state.pending is not None and current.time_msc >= state.pending[0]:
        _, horizon = state.pending
        state.pending = None  # The first due observation decides; never retry.
        if fresh and current.spread <= 2*unit and current.time_msc < horizon:
            result = Signal('POST_JUMP_CARRY_30S', -native_direction,
                            current.time_msc, horizon, 4*unit, None)
    if (fresh and event is not None and current.time_msc >= state.cooldown_until
            and event.source_gap_ms <= 2000
            and native_direction*(event.mid-event.prior_mid) >= 10*unit
            and native_direction*event.bid_delta > 0 and native_direction*event.ask_delta > 0):
        detection = current.time_msc
        state.pending = (detection+2000, detection+30000)
        state.cooldown_until = detection+60000
    return result


def generate_signals(quotes: Iterable[QuoteLike], *, policy_id: str,
                     native_direction: int, spread_unit: float) -> Iterator[Signal]:
    """Yield candidates at observed first-tick clocks, without opening positions.

    U is supplied from a separately frozen training fit. No input suffix is
    inspected, including later rows in the current timestamp bundle. A >5s gap
    or ambiguous completed bundle resets warmup and pending carry, preserving
    the 60s carry cooldown. A >2s decision/source gap skips issuance; a due carry
    is cancelled at that stale first observation rather than retried later.
    """
    if policy_id not in POLICY_IDS:
        raise ValueError('Unknown frozen policy_id')
    if type(native_direction) is not int or native_direction not in (-1, 1):
        raise ValueError('native_direction must be integer -1 or 1')
    if (isinstance(spread_unit, bool) or not isinstance(spread_unit, Real)
            or not math.isfinite(spread_unit) or spread_unit <= 0
            or not math.isfinite(10*spread_unit)):
        raise ValueError('spread_unit must be explicit finite positive training U')
    unit = float(spread_unit)
    state = _State(policy_id)
    bundle = None
    ambiguous = False
    for raw in quotes:
        current = _point(raw)
        if bundle is not None and current.time_msc < bundle.time_msc:
            raise ValueError('Quotes must remain in original nondecreasing timestamp order')
        if bundle is not None and current.time_msc == bundle.time_msc:
            ambiguous |= current.bid != bundle.bid or current.ask != bundle.ask
            continue
        signal = None
        if bundle is not None:
            gap = current.time_msc-bundle.time_msc
            if ambiguous or gap > 5000:
                state.reset()
            else:
                event = state.close(bundle)
                fresh = gap <= 2000
                if policy_id == 'POST_JUMP_CARRY_30S':
                    signal = _carry_signal(state, event, current, native_direction, unit, fresh)
                elif fresh:
                    if policy_id == 'SIDE_LEAD':
                        if event is not None:
                            signal = _side_signal(state, current, native_direction, unit)
                    else:
                        signal = _clock_signal(state, current, native_direction, unit)
        bundle, ambiguous = current, False
        if signal is not None:
            yield signal
