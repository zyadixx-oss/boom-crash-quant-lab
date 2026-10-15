"""Ex-ante sparse-clock schedule for a future declared replication only.

The schedule is a function of the full premeasurement declaration hash and UTC
calendar, never source prices, coverage or payoff. One 00/30 slot is selected
per two-day block; a final singleton uses 48 slots. Cancellations never redraw.
The dense frozen clock still owns causal issuance, ATR, freshness and spread.
This changes the new comparator's activity, not any frozen strategy or gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, MutableMapping

from independent_profit_research.policies import Signal
from .policies import CLOCK_POLICY_ID
from .sharded import DAY_MS
from .streaming_policies import clock_signals as dense_clock_signals

SLOT_MS = 30 * 60_000
POLICY_ID = 'SPARSE_CLOCK_2D'
SALT = b'CRT_REPLICATION_SPARSE_CLOCK_TWO_DAY_V1\x00'


@dataclass(frozen=True)
class ScheduledSlot:
    block_start_ms: int
    block_end_exclusive_ms: int
    scheduled_ms: int
    possible_slots: int
    rejection_counter: int
    digest_hex: str


def _calendar(start_ms, end_ms):
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms < 0
            or not start_ms < end_ms <= 2**63 - 1
            or start_ms % DAY_MS or end_ms % DAY_MS):
        raise ValueError('Expected nonempty aligned half-open UTC-day calendar')


def _uniform_slot(seed: bytes, block_start: int, slots: int,
                  digest: Callable[[bytes], bytes] = lambda value: hashlib.sha256(value).digest()):
    """Rejection sampling avoids modulo bias under the SHA pseudorandom assumption."""
    ceiling = 2**256 - (2**256 % slots)
    for counter in range(2**32):
        wire = SALT + seed + block_start.to_bytes(8, 'big') + counter.to_bytes(4, 'big')
        raw = digest(wire)
        if type(raw) is not bytes or len(raw) != 32:
            raise ValueError('Expected a SHA256-sized digest')
        value = int.from_bytes(raw, 'big')
        if value < ceiling:
            return value % slots, counter, raw.hex()
    raise ValueError('Sparse-clock rejection counter exhausted')


def sparse_schedule(start_ms: int, end_ms: int, declaration_sha256: str) -> tuple[ScheduledSlot, ...]:
    """Generate once from the PREmeasurement hash, before reading test quotes.

    The returned schedule hash belongs in a subsequent receipt referring back
    to this declaration. Including it in its own seed declaration is circular.
    The same stage schedule is shared across all policies and BASE/STRESS.
    """
    _calendar(start_ms, end_ms)
    if not isinstance(declaration_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', declaration_sha256):
        raise ValueError('Expected full canonical premeasurement declaration SHA256')
    seed = bytes.fromhex(declaration_sha256)
    output = []
    for block_start in range(start_ms, end_ms, 2 * DAY_MS):
        block_end = min(block_start + 2 * DAY_MS, end_ms)
        slots = (block_end - block_start) // SLOT_MS
        selected, counter, digest = _uniform_slot(seed, block_start, slots)
        output.append(ScheduledSlot(block_start, block_end,
                                    block_start + selected * SLOT_MS, slots, counter, digest))
    return tuple(output)


def generate_sparse_control(frames, issuance_index, spread_unit, native_direction,
                            start_ms: int, end_ms: int, declaration_sha256: str,
                            *, audit: MutableMapping | None = None):
    """Filter unchanged dense-clock issuance by the fixed ex-ante schedule.

    A selected slot lacking a valid causal dense-clock signal remains cancelled;
    later clean slots are never substituted. Embargo, reservation and UNKNOWN
    remain the evaluator's responsibility and never modify this schedule.
    """
    selected = {item.scheduled_ms: item for item in
                sparse_schedule(start_ms, end_ms, declaration_sha256)}
    scratch = {}
    dense = dense_clock_signals(frames, issuance_index, spread_unit, native_direction,
                               start_ms, end_ms, audit=scratch)
    try:
        for signal in dense:
            record = scratch.pop((CLOCK_POLICY_ID, signal.decision_ms))
            slot = selected.get(record['scheduled_ms'])
            if slot is None:
                continue
            result = Signal(POLICY_ID, signal.direction, signal.decision_ms,
                            signal.horizon_end_ms, signal.stop_distance, signal.target_distance)
            if audit is not None:
                audit[(POLICY_ID, result.decision_ms)] = {
                    **record, 'premeasurement_declaration_sha256': declaration_sha256,
                    'schedule_block_start_ms': slot.block_start_ms,
                    'schedule_block_end_exclusive_ms': slot.block_end_exclusive_ms,
                    'schedule_digest_hex': slot.digest_hex,
                    'scope': 'PREDECLARED_SPARSE_CLOCK_NOT_DENSE_OUTPERFORMANCE_OR_CASH',
                }
            yield result
    finally:
        close = getattr(dense, 'close', None)
        if callable(close):
            close()
