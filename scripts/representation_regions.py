"""Bounded streaming preparation and fixed price bands for paired research.

The transformed chain supplies features only. A learned score never changes
the original quote, raw ATR, fixed band, execution, or data-completeness rules.
There is no order API, fitted fallback or replacement for unknown features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research.jump_representation import JumpRepresentationState, PairedMinuteBars, complete_m1_bars
from app.research.zone_dataset import _bound, _source_index
from app.research.spike_hunter import assert_offline


def build_paired_minutes(chunks, detector, *, start, end):
    """Retain at most59 pending seconds, with a full planned UTC minute grid."""
    assert_offline()
    start, end = _bound(start, "start"), _bound(end, "end")
    if start >= end: raise ValueError("start must precede end")
    state, pending, source_last = JumpRepresentationState(detector), None, None
    raw, transformed, coverage = [], [], []
    observed, removed, known, maximum_pending = 0, 0, 0, 0

    def save(description):
        if description.empty: return
        bars = complete_m1_bars(description)
        raw.append(bars.raw); transformed.append(bars.transformed); coverage.append(bars.coverage)

    for chunk in chunks:
        index = _source_index(chunk, source_last)
        if len(index): source_last = int(index.asi8[-1])
        selected = (index >= start) & (index < end)
        if not selected.any(): continue
        bounded = chunk.loc[selected, ["quote"]].copy()
        bounded.index = index[selected]
        description = state.consume(bounded)
        observed += len(description)
        known += int(description.increment_known.sum())
        removed += int(description.tail_removed.fillna(False).sum())
        if pending is not None: description = pd.concat([pending, description])
        last_minute = description.index[-1].floor("min")
        # A complete final minute needs no pending state; otherwise carry all
        # its actual seconds across arbitrary chunks, including a real gap.
        tail = description.loc[description.index >= last_minute]
        if len(tail) == 60:
            save(description); pending = None
        else:
            save(description.loc[description.index < last_minute])
            pending = tail
            maximum_pending = max(maximum_pending, len(tail))
    if pending is not None: save(pending)
    grid = pd.date_range(start, end, freq="min", inclusive="left").as_unit("ns")
    def merge(parts, columns):
        frame = pd.concat(parts) if parts else pd.DataFrame(columns=columns, index=grid[:0])
        if frame.index.duplicated().any(): raise ValueError("Minute assembly duplicated a population")
        return frame.reindex(grid)
    raw = merge(raw, ["open", "high", "low", "close"]).astype(float)
    transformed = merge(transformed, ["open", "high", "low", "close"]).astype(float)
    coverage = merge(coverage, ["observed_seconds", "minute_valid", "run_id"])
    coverage["observed_seconds"] = coverage.observed_seconds.fillna(0).astype(np.int64)
    coverage["minute_valid"] = coverage.minute_valid.eq(True)
    coverage["run_id"] = pd.array(coverage.run_id, dtype="Int64")
    if int(coverage.observed_seconds.sum()) != observed: raise ValueError("Observed population was erased")
    summary = {"planned_minutes": len(grid), "observed_seconds": observed,
        "missing_seconds": len(grid) * 60 - observed,
        "complete_minutes": int(coverage.minute_valid.sum()), "tail_increments_removed": removed,
        "known_increments": known, "maximum_pending_seconds": maximum_pending,
        "runs": state.snapshot.run_id + 1, "safety": assert_offline()}
    return PairedMinuteBars(raw, transformed, coverage), summary


def fixed_regions(signals, *, symbol, variant):
    """Unchanged geometric control band, gated only by the supplied past score.

    Boom band is p−[.55,.45]rawATR; Crash mirrors it. Invalidation is another
    .25rawATR outside the adverse edge. No same-row native body gate is added.
    """
    assert_offline()
    if symbol not in ("BOOM600", "CRASH600"): raise ValueError("Declared symbols only")
    native = 1 if symbol == "BOOM600" else -1
    required = {"signal_time", "signal_close", "atr", "side"}
    if not required.issubset(signals): raise ValueError("Saved original issuance fields required")
    if signals.signal_time.duplicated().any(): raise ValueError("Duplicate opportunities")
    zones = []
    for row in signals.to_dict("records"):
        p, atr = row["signal_close"], row["atr"]
        if isinstance(row["side"], (bool, np.bool_)) or not isinstance(row["side"], (int, np.integer)) or row["side"] != native:
            raise ValueError("Original native direction must be retained")
        if not np.isfinite([p, atr]).all() or min(p, atr) <= 0: raise ValueError("Positive original price/ATR required")
        t = pd.Timestamp(row["signal_time"])
        if t.tz is None or t.value % (30 * 60 * 10**9): raise ValueError("Fixed UTC00/30 clock required")
        boundaries = sorted([p - native * .45 * atr, p - native * .55 * atr])
        inv = (boundaries[0] - .25 * atr if native == 1 else boundaries[1] + .25 * atr)
        if inv <= 0 or boundaries[0] <= 0: raise ValueError("Nonpositive region")
        zones.append({"signal_time": t, "variant": variant,
            "zone_id": f"{symbol}:{variant}:{t.isoformat()}", "side": native,
            "zone_low": boundaries[0], "zone_high": boundaries[1], "invalidation": inv,
            "issue_price": p, "atr": atr})
    return zones


def training_labels(clock, ledger):
    """Preserve the exact opportunity universe, including no-entry unknowns."""
    index = pd.DatetimeIndex(clock.signal_time, tz="UTC").as_unit("ns")
    if index.duplicated().any() or ledger.signal_time.duplicated().any(): raise ValueError("Duplicate training labels")
    known = ledger.set_index("signal_time").reindex(index)
    completed = known.entry_time.notna() & ~known.censored.eq(True) & known.net_R.notna()
    return pd.DataFrame({"net_R": pd.to_numeric(known.net_R, errors="raise").where(completed),
        "completed": completed.astype(bool), "planned_end": index + pd.Timedelta(minutes=16)}, index=index)


def numeric_ledger(ledger):
    """Explicit empty schemas for inherited inference, without changing values."""
    result = ledger.copy(deep=True)
    for name in ("entry", "exit", "atr", "gross_R", "net_R", "holding_minutes"):
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
    for name in ("signal_time", "entry_time", "exit_time"):
        result[name] = pd.to_datetime(result[name], utc=True)
    for name in ("censored", "ambiguous"):
        if not result[name].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError("Ledger states must be actual booleans")
        result[name] = result[name].astype(bool)
    return result
