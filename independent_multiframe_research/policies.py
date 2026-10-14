"""Fixed closed-bar setups with actual first-quote issuance; no fitting or returns.

All prices retain the caller's common units (including the integer-scaled v1
tape convention). A pending setup owns its frozen context and ATR. Future bars
cannot replace that context, and the issue quote is the first row of its
timestamp, without inspecting the rest of that same-ms bundle.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, MutableMapping

from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape


MINUTE_MS = 60_000
FRAME_MINUTES = (1, 5, 15, 60, 240)
POLICY_IDS = frozenset({"CRT_SWEEP_RECLAIM", "SQUEEZE_BREAKOUT", "PULLBACK_CONTINUATION"})
CLOCK_POLICY_ID = "CLOCK_00_30"


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer in the permitted range")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be positive and finite")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return value


def _inputs(native_direction, spread_unit, tape, quotes):
    if type(native_direction) is not int or native_direction not in (-1, 1):
        raise ValueError("native_direction must be +1 or -1")
    unit = _positive(spread_unit, "spread_unit")
    if (tape is None) == (quotes is None):
        raise ValueError("Supply exactly one of tape or quotes")
    if tape is None:
        tape = QuoteTape(quotes)
    if not isinstance(tape, QuoteTape):
        raise TypeError("tape must be QuoteTape")
    return unit, tape


class _Book:
    """Validate without sorting, and index only already-closed source bars."""

    def __init__(self, frames):
        if not isinstance(frames, Mapping) or any(size not in frames for size in FRAME_MINUTES):
            raise ValueError("All five declared frames are required")
        self.rows, self.by_end = {}, {}
        for size in FRAME_MINUTES:
            period = size * MINUTE_MS
            rows, lookup, prior = [], {}, None
            for bar in frames[size]:
                for field in ("open_ms", "end_ms", "confirmed_ms", "run_id"):
                    _integer(getattr(bar, field), field)
                if bar.open_ms % period or bar.end_ms - bar.open_ms != period:
                    raise ValueError("Bars must use exact complete UTC intervals")
                if bar.confirmed_ms < bar.end_ms:
                    raise ValueError("A bar cannot be confirmed before its end")
                prices = [_positive(getattr(bar, field), field) for field in ("open", "high", "low", "close")]
                opening, high, low, close = prices
                if high < max(opening, low, close) or low > min(opening, high, close):
                    raise ValueError("Invalid bar OHLC")
                if prior is not None and (bar.open_ms <= prior.open_ms or
                        bar.confirmed_ms < prior.confirmed_ms or bar.run_id < prior.run_id):
                    raise ValueError("Bars must be ordered, unique and have chronological runs")
                rows.append(bar)
                lookup[bar.end_ms] = bar
                prior = bar
            self.rows[size], self.by_end[size] = tuple(rows), lookup
        self.ema15 = self._ema20()

    def at(self, size, end, now, run=None):
        bar = self.by_end[size].get(end)
        if bar is None or bar.confirmed_ms > now or (run is not None and bar.run_id != run):
            return None
        return bar

    def expected(self, size, time, now, run):
        period = size * MINUTE_MS
        return self.at(size, time // period * period, now, run)

    def history(self, size, end, count, now, run):
        period = size * MINUTE_MS
        bars = [self.at(size, end - offset * period, now, run)
                for offset in range(count - 1, -1, -1)]
        return bars if all(bar is not None for bar in bars) else None

    def atr(self, end, now, run):
        bars = self.history(5, end, 15, now, run)
        if bars is None:
            return None
        ranges = [max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
                  for p, b in zip(bars, bars[1:])]
        value = math.fsum(ranges) / 14
        return value if math.isfinite(value) and value > 0 else None

    def compression(self, bar, now):
        bars = self.history(15, bar.end_ms, 31, now, bar.run_id)
        if bars is None:
            return None
        ranges = [max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
                  for p, b in zip(bars, bars[1:])]
        denominator = math.fsum(ranges) / 30
        return (math.fsum(ranges[-5:]) / 5 / denominator) if denominator > 0 else None

    def _ema20(self):
        values, window, previous, ema = {}, deque(maxlen=20), None, None
        for bar in self.rows[15]:
            if previous is None or bar.open_ms != previous.end_ms or bar.run_id != previous.run_id:
                window.clear()
                ema = None
            window.append(float(bar.close))
            if ema is None:
                if len(window) == 20:
                    ema = math.fsum(window) / 20
            else:
                ema = (2 / 21) * float(bar.close) + (19 / 21) * ema
            if ema is not None:
                values[bar.end_ms] = ema
            previous = bar
        return values


@dataclass
class _Setup:
    source: object
    atr: float
    deadline: int
    level: float | None = None
    reference: object | None = None
    depth_atr: float | None = None
    break_end: int | None = None


def _native(bar, side):
    return side * (bar.close - bar.open) > 0


def _setup(book, bar, policy, side, now, used_refs):
    run, time = bar.run_id, bar.end_ms
    h4 = book.expected(240, time, now, run)
    h1 = book.expected(60, time, now, run)
    last5 = book.at(5, time, now, run)
    atr = book.atr(time, now, run)
    if h4 is None or not _native(h4, side) or last5 is None or atr is None:
        return None
    if policy == "CRT_SWEEP_RECLAIM":
        reference_end = bar.open_ms // (60 * MINUTE_MS) * (60 * MINUTE_MS)
        reference = book.at(60, reference_end, now, run)
        depth_atr = book.atr(bar.open_ms, now, run)
        if reference is None or reference.end_ms in used_refs or depth_atr is None or not _native(last5, side):
            return None
        depth = reference.low - bar.low if side == 1 else bar.high - reference.high
        if not (0.10 * depth_atr <= depth <= 0.65 * depth_atr and
                reference.low < bar.close < reference.high):
            return None
        return _Setup(bar, atr, time + 5 * MINUTE_MS,
                      reference=reference, depth_atr=depth_atr)
    if h1 is None or not _native(h1, side):
        return None
    if policy == "SQUEEZE_BREAKOUT":
        ratio = book.compression(bar, now)
        if ratio is None or not math.isfinite(ratio) or ratio > 0.80:
            return None
        return _Setup(bar, atr, time + 15 * MINUTE_MS,
                      level=float(bar.high if side == 1 else bar.low))
    ema = book.ema15.get(time)
    if ema is None or not (bar.low <= ema <= bar.high and side * (bar.close - ema) > 0):
        return None
    return _Setup(bar, atr, time + 15 * MINUTE_MS, level=ema)


def _break(book, setup, bar, policy, side, now):
    if not (setup.source.end_ms < bar.end_ms <= setup.source.end_ms + 15 * MINUTE_MS):
        return False
    if policy == "SQUEEZE_BREAKOUT":
        return side * (bar.close - setup.level) > 0
    previous = book.at(5, bar.open_ms, now, bar.run_id)
    if previous is None or not _native(bar, side):
        return False
    boundary = previous.high if side == 1 else previous.low
    return side * (bar.close - boundary) > 0


def _first_quote(tape, time):
    index = bisect_left(tape.timestamps, time)
    return None if index == len(tape) else tape.quotes[index]


def _issuable(tape, bar, unit):
    quote = _first_quote(tape, bar.end_ms)
    return (quote is not None and quote.time_msc == bar.confirmed_ms and
            bar.confirmed_ms - bar.end_ms <= 2000 and quote.ask - quote.bid <= 2 * unit)


def _audit_record(policy, setup, bar, tape, unit):
    quote = _first_quote(tape, bar.confirmed_ms)
    record = {"policy_id": policy, "setup_open_ms": setup.source.open_ms,
              "setup_end_ms": setup.source.end_ms,
              "setup_confirmed_ms": setup.source.confirmed_ms,
              "run_id": setup.source.run_id, "frozen_atr": setup.atr,
              "trigger_m5_end_ms": setup.break_end,
              "issue_m1_end_ms": bar.end_ms, "decision_ms": bar.confirmed_ms,
              "issue_spread": quote.ask - quote.bid, "spread_unit": unit,
              "frozen_level": setup.level}
    if setup.reference is not None:
        record.update(reference_h1_open_ms=setup.reference.open_ms,
                      reference_h1_end_ms=setup.reference.end_ms,
                      reference_h1_low=setup.reference.low,
                      reference_h1_high=setup.reference.high,
                      frozen_depth_atr=setup.depth_atr)
    return record


def generate_signals(frames, *, policy_id, native_direction, spread_unit,
                     tape=None, quotes=None, audit: MutableMapping | None = None):
    """Yield candidate signals only; busy positions and stage censoring are external.

    One pending setup per policy ignores replacement setups. Missing M1 or a
    run change cancels pending state. The first qualifying minute consumes the
    opportunity even if freshness/spread fails. No future outcome affects any
    signal. ATR is frozen at setup; the prior setup-opening ATR is CRT depth's
    separate denominator. Targets/stops are 2A/A from eventual effective entry.
    """
    if policy_id not in POLICY_IDS:
        raise ValueError("Unknown fixed policy_id")
    unit, tape = _inputs(native_direction, spread_unit, tape, quotes)
    book = _Book(frames)
    setup, previous, used_refs = None, None, set()
    for minute in book.rows[1]:
        now = minute.confirmed_ms
        if previous is not None and (minute.open_ms != previous.end_ms or minute.run_id != previous.run_id):
            setup = None
        previous = minute
        if setup is not None and (minute.run_id != setup.source.run_id or now > setup.deadline):
            setup = None
        if setup is not None:
            after = setup.source.end_ms if policy_id == "CRT_SWEEP_RECLAIM" else setup.break_end
            if after is not None and minute.end_ms > after:
                stays = (setup.reference.low < minute.close < setup.reference.high
                         if policy_id == "CRT_SWEEP_RECLAIM" else
                         native_direction * (minute.close - setup.level) > 0)
                if not stays:
                    setup = None
                elif _native(minute, native_direction):
                    selected, setup = setup, None
                    if _issuable(tape, minute, unit):
                        if policy_id == "CRT_SWEEP_RECLAIM":
                            used_refs.add(selected.reference.end_ms)
                        signal = Signal(policy_id, native_direction, now, now + 15 * MINUTE_MS,
                                        selected.atr, 2 * selected.atr)
                        if audit is not None:
                            audit[(policy_id, now)] = _audit_record(policy_id, selected, minute, tape, unit)
                        yield signal
        if setup is not None and policy_id != "CRT_SWEEP_RECLAIM" and setup.break_end is None:
            m5 = book.at(5, minute.end_ms, now, minute.run_id)
            if m5 is not None and _break(book, setup, m5, policy_id, native_direction, now):
                setup.break_end = m5.end_ms
                setup.deadline = min(setup.source.end_ms + 20 * MINUTE_MS,
                                     m5.end_ms + 5 * MINUTE_MS)
        if setup is None:
            m15 = book.at(15, minute.end_ms, now, minute.run_id)
            if m15 is not None:
                setup = _setup(book, m15, policy_id, native_direction, now, used_refs)


def clock_signals(frames, tape, spread_unit, native_direction, start_ms, end_ms, *, audit=None):
    """Fixed UTC00/30 control, with the same frozen1A/2A/15min exposure rules.

    A scheduled clock requires its exact completed M1/M5 history, never an
    older available candle. Its first quote at/after the clock must be the bar
    confirmation and pass freshness/spread; a later better quote is not used.
    """
    start_ms = _integer(start_ms, "start_ms")
    end_ms = _integer(end_ms, "end_ms", minimum=1)
    if end_ms <= start_ms:
        raise ValueError("end_ms must follow start_ms")
    unit, tape = _inputs(native_direction, spread_unit, tape, None)
    book = _Book(frames)
    for minute in book.rows[1]:
        time, now = minute.end_ms, minute.confirmed_ms
        if time % (30 * MINUTE_MS) or not (start_ms <= time < end_ms and start_ms <= now < end_ms):
            continue
        quote = _first_quote(tape, time)
        if quote is None or quote.time_msc != now or not _issuable(tape, minute, unit):
            continue
        atr = book.atr(time, now, minute.run_id)
        if atr is None:
            continue
        signal = Signal(CLOCK_POLICY_ID, native_direction, now, now + 15 * MINUTE_MS, atr, 2 * atr)
        if audit is not None:
            audit[(CLOCK_POLICY_ID, now)] = {"scheduled_ms": time, "decision_ms": now,
                                           "frozen_atr": atr, "run_id": minute.run_id,
                                           "issue_spread": quote.ask - quote.bid, "spread_unit": unit}
        yield signal
