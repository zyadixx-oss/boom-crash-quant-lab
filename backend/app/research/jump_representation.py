"""Causal native-tail removal for a feature-only price representation.

The caller supplies a frozen detector; no scale is estimated here. In each
maximal exact-one-second run, log(Q) starts at zero. Observed native-tail
increments become zero and every other original signed log increment is kept.
This is a specified transformation, not an identified price without jumps.
Original quotes remain available separately for labels and execution.

Neumaier compensated summation follows observation order, independent of chunk
boundaries. Every run has the fixed numerical anchor Q=1. Missing seconds reset
the chain; they are never interpolated or bridged. Only complete UTC minutes
containing seconds00..59 from one run produce paired OHLC bars. Consumers must
restart feature histories at each run and preserve invalid UTC rows.

No features or trading rules are implemented. In particular, the existing
44-feature large-bar age must stay unknown until a qualifying transformed bar
actually occurs, and reset after gaps. Tail removal may make that age, and thus
the complete44 representation, unavailable for long periods or entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
import pandas as pd

from app.research.tick_tail import (
    SECOND_NS,
    FixedTailDetector,
    _log_return,
    _positive_real,
    _validated_ticks,
)


DESCRIPTION_COLUMNS = (
    "quote", "transformed_quote", "run_id", "increment_known", "tail_removed",
    "log_return", "retained_log_return", "log_transformed_quote",
)
OHLC_COLUMNS = ("open", "high", "low", "close")


class RepresentationArithmeticError(ValueError):
    """The declared transformation cannot produce finite positive prices."""


@dataclass(frozen=True, slots=True)
class RepresentationSnapshot:
    """Immutable state inspection; log_sum and correction are not prices."""

    last_epoch: int | None
    last_quote: float | None
    run_id: int
    log_sum: float
    correction: float


@dataclass(frozen=True, slots=True)
class PairedMinuteBars:
    """Separate OHLC tables and their shared full-minute validity/run identity.

    ``coverage.run_id`` is known when the observed part of a minute belongs to
    one run, even if that minute is incomplete. It is unknown for empty or
    mixed-run minutes. ``minute_valid`` alone permits using either OHLC row.
    """

    raw: pd.DataFrame
    transformed: pd.DataFrame
    coverage: pd.DataFrame


def _compensated_add(total: float, correction: float, value: float) -> tuple[float, float]:
    """Fixed Neumaier update; a zero increment leaves the state unchanged."""
    if value == 0.0:
        return total, correction
    updated = total + value
    error = ((total - updated) + value if abs(total) >= abs(value)
             else (value - updated) + total)
    corrected = correction + error
    if not all(math.isfinite(item) for item in (updated, corrected, updated + corrected)):
        raise RepresentationArithmeticError("Compensated log-price sum is not finite")
    return updated, corrected


def _price(log_price: float) -> float:
    try:
        result = math.exp(log_price)
    except OverflowError as error:
        raise RepresentationArithmeticError("Transformed price exponentiation overflow") from error
    if not math.isfinite(result) or result <= 0.0:
        raise RepresentationArithmeticError("Transformed price must remain finite and positive")
    return result


class JumpRepresentationState:
    """Consume chronological nonempty quote chunks under one immutable detector.

    ``consume`` validates a complete chunk before committing its new state.
    Validation or arithmetic failure leaves the prior state untouched. A new
    chunk may continue the previous exact-second run or begin after a gap; an
    overlapping, duplicate or earlier timestamp is refused. No chunk boundary
    itself resets the chain. Descriptions can be concatenated before building
    minute bars, so a minute split across chunks remains complete.
    """

    __slots__ = ("_detector", "_state")

    def __init__(self, detector: FixedTailDetector) -> None:
        if not isinstance(detector, FixedTailDetector):
            raise TypeError("detector must be a FixedTailDetector")
        self._detector = detector
        self._state = RepresentationSnapshot(None, None, -1, 0.0, 0.0)

    @property
    def detector(self) -> FixedTailDetector:
        return self._detector

    @property
    def snapshot(self) -> RepresentationSnapshot:
        return self._state

    def consume(self, ticks: pd.DataFrame) -> pd.DataFrame:
        index, quotes = _validated_ticks(ticks)
        state = self._state
        epochs = index.asi8 // SECOND_NS
        if state.last_epoch is not None and int(epochs[0]) <= state.last_epoch:
            raise ValueError("Chunks must be strictly later than previously consumed quotes")
        last_epoch, last_quote, run = state.last_epoch, state.last_quote, state.run_id
        total, correction = state.log_sum, state.correction
        transformed = np.empty(len(quotes), dtype=float)
        runs = np.empty(len(quotes), dtype=np.int64)
        known = np.zeros(len(quotes), dtype=bool)
        removed: list[bool | None] = []
        returns = np.full(len(quotes), np.nan)
        retained = np.full(len(quotes), np.nan)
        log_prices = np.empty(len(quotes), dtype=float)
        for i, (epoch, quote) in enumerate(zip(epochs, quotes, strict=True)):
            epoch, quote = int(epoch), float(quote)
            if last_epoch is None or epoch - last_epoch != 1:
                run += 1
                total, correction = 0.0, 0.0
                removed.append(None)
            else:
                value = _log_return(last_quote, quote)
                if not math.isfinite(value):
                    raise RepresentationArithmeticError("Original log increment is not finite")
                event = self._detector.side * value > self._detector.threshold
                keep = 0.0 if event else value
                total, correction = _compensated_add(total, correction, keep)
                known[i], returns[i], retained[i] = True, value, keep
                removed.append(event)
            log_prices[i] = total + correction
            transformed[i] = _price(float(log_prices[i]))
            runs[i] = run
            last_epoch, last_quote = epoch, quote
        result = pd.DataFrame({
            "quote": quotes.copy(), "transformed_quote": transformed, "run_id": runs,
            "increment_known": known, "tail_removed": pd.array(removed, dtype="boolean"),
            "log_return": returns, "retained_log_return": retained,
            "log_transformed_quote": log_prices,
        }, index=index.copy())
        self._state = RepresentationSnapshot(last_epoch, last_quote, run, total, correction)
        return result.loc[:, DESCRIPTION_COLUMNS]


def complete_m1_bars(description: pd.DataFrame) -> PairedMinuteBars:
    """Aggregate aligned raw/transformed quotes, preserving unknown minutes.

    Run identity is mandatory and checked against exact observed gaps. No
    incomplete minute receives even a partially observed OHLC value. This
    function aggregates a supplied description; it does not remove more tails
    or interpret an arbitrary restart as an observed market increment.
    """
    index, raw = _validated_ticks(description)
    if any(name not in description for name in ("transformed_quote", "run_id")):
        raise ValueError("Both transformed_quote and run_id are required")
    transformed = np.fromiter(
        (_positive_real(value, "transformed_quote") for value in description.transformed_quote.array),
        dtype=float, count=len(description),
    )
    values = description.run_id.array
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0
           for value in values):
        raise ValueError("Run ids must be nonnegative integers")
    try:
        runs = np.asarray(values, dtype=np.int64)
    except (OverflowError, ValueError) as error:
        raise ValueError("Run ids must fit signed64 integers") from error
    expected_change = (np.diff(index.asi8) != SECOND_NS).astype(np.int64)
    if np.any(np.diff(runs) != expected_change):
        raise ValueError("Run identity must change exactly once at each observed gap")
    frame = pd.DataFrame({"raw": raw, "transformed": transformed, "run_id": runs}, index=index)
    opening = index.floor("min")
    groups = frame.groupby(opening, sort=True)
    aggregate = groups.agg(
        raw_open=("raw", "first"), raw_high=("raw", "max"),
        raw_low=("raw", "min"), raw_close=("raw", "last"),
        transformed_open=("transformed", "first"), transformed_high=("transformed", "max"),
        transformed_low=("transformed", "min"), transformed_close=("transformed", "last"),
        observed_seconds=("raw", "size"), run_count=("run_id", "nunique"), run_id=("run_id", "first"),
    )
    # Preserve integer identity when reindexing inserts wholly absent minutes;
    # a float intermediate could round a large valid run id.
    aggregate["run_id"] = pd.array(aggregate.run_id, dtype="Int64")
    grid = pd.date_range(opening[0], opening[-1], freq="min").as_unit("ns")
    grid.name = description.index.name
    aggregate = aggregate.reindex(grid)
    count = aggregate.observed_seconds.fillna(0).astype(np.int64)
    one_run = aggregate.run_count.eq(1)
    # Unique whole-second observations within a UTC minute have exactly the
    # required00..59 population iff there are60 of them.
    valid = count.eq(60) & one_run
    run_identity = pd.array(aggregate.run_id.where(one_run), dtype="Int64")
    raw_bars = aggregate.loc[:, ["raw_" + name for name in OHLC_COLUMNS]].copy()
    transformed_bars = aggregate.loc[:, ["transformed_" + name for name in OHLC_COLUMNS]].copy()
    raw_bars.columns = transformed_bars.columns = list(OHLC_COLUMNS)
    raw_bars = raw_bars.where(valid, axis=0)
    transformed_bars = transformed_bars.where(valid, axis=0)
    coverage = pd.DataFrame({"observed_seconds": count, "minute_valid": valid,
                             "run_id": run_identity}, index=grid)
    return PairedMinuteBars(raw_bars, transformed_bars, coverage)


def jump_representation(ticks: pd.DataFrame, detector: FixedTailDetector) -> tuple[pd.DataFrame, PairedMinuteBars]:
    """Batch convenience API; streaming callers retain JumpRepresentationState."""
    description = JumpRepresentationState(detector).consume(ticks)
    return description, complete_m1_bars(description)
