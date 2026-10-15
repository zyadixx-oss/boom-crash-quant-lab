"""Issuance-only access adapters for the unchanged frozen v2 policy logic.

These two loops preserve the original scheduling and cancellation semantics.
All feature/setup/break/native/freshness/audit calculations reuse frozen private
helpers directly; no globals are replaced. They return Signals, never execution
outcomes. Their non-QuoteTape index must never enter a frozen execution runner.
A future declared streaming evaluator must use ShardedQuoteStore's full local
trade path and separately validate stage, reservation and censoring semantics.
"""
from __future__ import annotations

from typing import MutableMapping

from independent_profit_research.policies import Signal
from . import policies as frozen
from .streaming import IssuanceQuoteIndex


def _inputs(native_direction, spread_unit, issuance_index):
    if type(native_direction) is not int or native_direction not in (-1, 1):
        raise ValueError('native_direction must be +1 or -1')
    unit = frozen._positive(spread_unit, 'spread_unit')
    if type(issuance_index) is not IssuanceQuoteIndex:
        raise TypeError('issuance_index must be an exact IssuanceQuoteIndex')
    return unit, issuance_index


def generate_signals(frames, *, policy_id, native_direction, spread_unit,
                     issuance_index: IssuanceQuoteIndex,
                     audit: MutableMapping | None = None):
    """Mirror frozen issuance scheduling using only real first-quote lookups."""
    if policy_id not in frozen.POLICY_IDS:
        raise ValueError('Unknown fixed policy_id')
    unit, index = _inputs(native_direction, spread_unit, issuance_index)
    book = frozen._Book(frames)
    setup, previous, used_refs = None, None, set()
    for minute in book.rows[1]:
        now = minute.confirmed_ms
        if previous is not None and (minute.open_ms != previous.end_ms or minute.run_id != previous.run_id):
            setup = None
        previous = minute
        if setup is not None and (minute.run_id != setup.source.run_id or now > setup.deadline):
            setup = None
        if setup is not None:
            after = setup.source.end_ms if policy_id == 'CRT_SWEEP_RECLAIM' else setup.break_end
            if after is not None and minute.end_ms > after:
                stays = (setup.reference.low < minute.close < setup.reference.high
                         if policy_id == 'CRT_SWEEP_RECLAIM' else
                         native_direction * (minute.close - setup.level) > 0)
                if not stays:
                    setup = None
                elif frozen._native(minute, native_direction):
                    selected, setup = setup, None
                    if frozen._issuable(index, minute, unit):
                        if policy_id == 'CRT_SWEEP_RECLAIM':
                            used_refs.add(selected.reference.end_ms)
                        signal = Signal(policy_id, native_direction, now, now + 15 * frozen.MINUTE_MS,
                                        selected.atr, 2 * selected.atr)
                        if audit is not None:
                            audit[(policy_id, now)] = frozen._audit_record(policy_id, selected, minute, index, unit)
                        yield signal
        if setup is not None and policy_id != 'CRT_SWEEP_RECLAIM' and setup.break_end is None:
            m5 = book.at(5, minute.end_ms, now, minute.run_id)
            if m5 is not None and frozen._break(book, setup, m5, policy_id, native_direction, now):
                setup.break_end = m5.end_ms
                setup.deadline = min(setup.source.end_ms + 20 * frozen.MINUTE_MS,
                                     m5.end_ms + 5 * frozen.MINUTE_MS)
        if setup is None:
            m15 = book.at(15, minute.end_ms, now, minute.run_id)
            if m15 is not None:
                setup = frozen._setup(book, m15, policy_id, native_direction, now, used_refs)


def clock_signals(frames, issuance_index: IssuanceQuoteIndex, spread_unit,
                  native_direction, start_ms, end_ms, *, audit=None):
    """Mirror frozen UTC00/30 control issuance without providing an execution tape."""
    start_ms = frozen._integer(start_ms, 'start_ms')
    end_ms = frozen._integer(end_ms, 'end_ms', minimum=1)
    if end_ms <= start_ms:
        raise ValueError('end_ms must follow start_ms')
    unit, index = _inputs(native_direction, spread_unit, issuance_index)
    book = frozen._Book(frames)
    for minute in book.rows[1]:
        time, now = minute.end_ms, minute.confirmed_ms
        if time % (30 * frozen.MINUTE_MS) or not (start_ms <= time < end_ms and start_ms <= now < end_ms):
            continue
        quote = frozen._first_quote(index, time)
        if quote is None or quote.time_msc != now or not frozen._issuable(index, minute, unit):
            continue
        atr = book.atr(time, now, minute.run_id)
        if atr is None:
            continue
        signal = Signal(frozen.CLOCK_POLICY_ID, native_direction, now, now + 15 * frozen.MINUTE_MS,
                        atr, 2 * atr)
        if audit is not None:
            audit[(frozen.CLOCK_POLICY_ID, now)] = {
                'scheduled_ms': time, 'decision_ms': now, 'frozen_atr': atr,
                'run_id': minute.run_id, 'issue_spread': quote.ask - quote.bid, 'spread_unit': unit,
            }
        yield signal
