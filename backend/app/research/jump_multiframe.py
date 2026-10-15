"""Pair unchanged closed-frame inputs on raw and tail-removed minute bars.

This module has no outcomes, fitting, signal selection or execution. Raw and
transformed candles must describe the same complete native-second populations.
Every maximal consecutive-second run receives its own rolling history, so a
missing second cannot turn a representation reset into a measured price move.
The full UTC M5 grid survives, including unknown rows and unknown large-bar age.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.research.jump_representation import OHLC_COLUMNS, PairedMinuteBars
from app.research.learned_signal import _validated_m1
from app.research.multiframe_signal import FEATURE_NAMES, causal_multiframe_inputs


@dataclass(frozen=True, slots=True)
class PairedMultiframeInputs:
    """Two 44-feature tables and their past-only common availability.

    ``availability.raw_execution_atr`` belongs to original observed prices;
    ``transformed_feature_atr`` is solely a feature normalization denominator.
    No execution or target calculation may substitute the latter for the former.
    An unknown common row remains present instead of being dropped.
    """

    raw: pd.DataFrame
    transformed: pd.DataFrame
    availability: pd.DataFrame
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _validated_pair(bars: PairedMinuteBars) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(bars, PairedMinuteBars):
        raise TypeError("bars must be PairedMinuteBars")
    raw, transformed = _validated_m1(bars.raw), _validated_m1(bars.transformed)
    coverage = bars.coverage.copy(deep=True)
    if not isinstance(coverage.index, pd.DatetimeIndex) or coverage.index.tz is None:
        raise ValueError("Coverage requires a timezone-aware minute index")
    coverage.index = coverage.index.tz_convert("UTC").as_unit("ns")
    if not raw.index.equals(transformed.index) or not raw.index.equals(coverage.index):
        raise ValueError("Raw, transformed and coverage populations must share one minute grid")
    expected = pd.date_range(raw.index[0], raw.index[-1], freq="min").as_unit("ns")
    if not raw.index.equals(expected):
        raise ValueError("Paired bars must retain the full UTC minute grid")
    required = {"observed_seconds", "minute_valid", "run_id"}
    if not required.issubset(coverage.columns):
        raise ValueError("Coverage requires observed_seconds, minute_valid and run_id")
    valid = coverage.minute_valid
    if valid.isna().any() or not pd.api.types.is_bool_dtype(valid.dtype):
        raise ValueError("minute_valid must contain known boolean values")
    counts = coverage.observed_seconds
    if (counts.isna().any() or pd.api.types.is_bool_dtype(counts.dtype)
            or not pd.api.types.is_integer_dtype(counts.dtype)
            or not counts.between(0, 60).all()):
        raise ValueError("observed_seconds must contain integers from 0 through 60")
    run = coverage.run_id
    if pd.api.types.is_bool_dtype(run.dtype) or not pd.api.types.is_integer_dtype(run.dtype):
        raise ValueError("run_id must contain nullable nonnegative integers")
    known = run.dropna()
    if (known < 0).any() or not known.is_monotonic_increasing:
        raise ValueError("Known run ids must be nonnegative and chronological")
    if not valid.equals(counts.eq(60) & run.notna()):
        raise ValueError("Complete minutes require 60 observed seconds from one known run")
    if (counts.eq(0) & run.notna()).any():
        raise ValueError("Empty minutes cannot have a known run id")
    for original, normalized in ((bars.raw, raw), (bars.transformed, transformed)):
        values = original.loc[:, OHLC_COLUMNS].to_numpy(dtype=float)
        complete = np.isfinite(values).all(axis=1)
        entirely_unknown = np.isnan(values).all(axis=1)
        if not np.array_equal(complete, valid.to_numpy(bool)) or not (complete | entirely_unknown).all():
            raise ValueError("Both OHLC populations must be complete exactly at minute_valid rows")
        if not normalized.loc[~valid, :].isna().all().all():
            raise ValueError("Incomplete paired minutes must stay unknown")
    return raw, transformed, coverage


def _stable_bb_width(close: pd.Series) -> pd.Series:
    """Compute the unchanged20-close population width independently per window.

    Scaling a complete window by its own positive maximum preserves the exact
    formula4*std(ddof=0)/mean. ``fsum`` accumulates centered variance without
    subtracting large historical terms as prices drift toward a different gauge.
    Missing windows stay unknown and constant windows have width zero.
    """
    values = close.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for end in range(19, len(values)):
        window = values[end - 19:end + 1]
        if not np.isfinite(window).all() or (window <= 0).any():
            continue
        normalized = window / float(window.max())
        mean = math.fsum(normalized) / 20
        variance = math.fsum((float(value) - mean) ** 2 for value in normalized) / 20
        result[end] = 4 * math.sqrt(variance) / mean
    return pd.Series(result, index=close.index)


def _stable_bb_inputs(frame: pd.DataFrame, direction: str) -> None:
    """Replace only equivalent BB arithmetic and recompute full44 validity.

    The original helper's feature_valid includes its rolling BB calculation;
    retaining that flag could conceal rows made finite by stable arithmetic.
    Every original non-BB prerequisite is therefore expressed again below.
    """
    width = _stable_bb_width(frame.close)
    prior = width.shift(1).rolling(100, min_periods=100)
    q35 = prior.quantile(0.35)
    frame["bb_width_over_prior_median_100"] = width / prior.median()
    frame["bb_squeeze"] = width <= q35
    complete = frame.loc[:, OHLC_COLUMNS].notna().all(axis=1)
    previous = frame.close.shift(1)
    ranges = frame.high - frame.low
    body = (frame.close - frame.open).abs()
    tr = pd.concat([ranges, (frame.high - previous).abs(),
                    (frame.low - previous).abs()], axis=1).max(axis=1)
    tr = tr.where(complete & previous.notna())
    med_body = body.shift(1).rolling(20, min_periods=20).median()
    sr_level = (frame.low.shift(1).rolling(12, min_periods=12).min()
                if direction == "boom" else
                frame.high.shift(1).rolling(12, min_periods=12).max())
    base_valid = (q35.notna() & med_body.notna() & sr_level.notna()
                  & tr.rolling(30, min_periods=30).count().eq(30) & frame.atr.gt(0))
    context_valid = frame.loc[:, [f"{label}_row_valid" for label in ("h4", "h1", "m15", "m1")]].eq(True).all(axis=1)
    exact_m1 = frame.m1_closed_at.eq(frame.index + pd.Timedelta(minutes=5))
    finite = np.isfinite(frame.loc[:, FEATURE_NAMES].to_numpy()).all(axis=1)
    frame["feature_valid"] = base_valid & complete & context_valid & exact_m1 & finite


def _run_frames(minute: pd.DataFrame, coverage: pd.DataFrame, direction: str,
                grid: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    for run in coverage.run_id.dropna().unique():
        positions = coverage.run_id.eq(run).fillna(False).to_numpy(bool)
        isolated = minute.loc[positions]
        frame, names = causal_multiframe_inputs(isolated, direction)
        if tuple(names) != FEATURE_NAMES:
            raise ValueError("The unchanged causal44 feature schema differs")
        _stable_bb_inputs(frame, direction)
        frame["representation_run_id"] = int(run)
        frames.append(frame)
    if not frames:
        # A minute with multiple runs has no usable population at all. Obtain
        # the unchanged output schema from an explicitly unknown input row.
        schema, _ = causal_multiframe_inputs(minute.iloc[:1] * np.nan, direction)
        frame = schema.iloc[:0].reindex(grid)
        frame["representation_run_id"] = pd.array([pd.NA] * len(grid), dtype="Int64")
    else:
        frame = pd.concat(frames)
        # A gap inside an M5 interval can leave two partial rows. Both are
        # invalid; retain the newer run's state without importing older context.
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index().reindex(grid)
        frame["representation_run_id"] = pd.array(frame.representation_run_id, dtype="Int64")
    for name in frame.columns:
        if name == "feature_valid" or name.endswith("_row_valid"):
            frame[name] = frame[name].eq(True)
    return frame, frame.representation_run_id.copy()


def causal_jump_multiframe_inputs(bars: PairedMinuteBars, direction: str) -> PairedMultiframeInputs:
    """Calculate raw44/transformed44 on identical past-only minute populations.

    The returned index remains the M5 opening time; every decision must wait
    until index + 5 minutes. No argument supplies future outcomes, and common
    availability depends only on the two unchanged feature_valid states.
    Consumers must apply ``availability.common_feature_valid`` to both models;
    the individual frame flags remain unchanged for availability diagnostics.
    """
    if direction not in ("boom", "crash"):
        raise ValueError("direction must be boom or crash")
    raw, transformed, coverage = _validated_pair(bars)
    grid = pd.date_range(raw.index[0].floor("5min"), raw.index[-1].floor("5min"),
                         freq="5min").as_unit("ns")
    grid.name = raw.index.name
    raw_inputs, raw_run = _run_frames(raw, coverage, direction, grid)
    transformed_inputs, transformed_run = _run_frames(transformed, coverage, direction, grid)
    if not raw_run.equals(transformed_run):
        raise ValueError("Raw and transformed run populations differ")
    raw_valid = raw_inputs.feature_valid.copy()
    transformed_valid = transformed_inputs.feature_valid.copy()
    availability = pd.DataFrame({
        "raw_feature_valid": raw_valid,
        "transformed_feature_valid": transformed_valid,
        "common_feature_valid": raw_valid & transformed_valid,
        "raw_execution_atr": raw_inputs.atr.copy(),
        "transformed_feature_atr": transformed_inputs.atr.copy(),
        "run_id": raw_run,
    }, index=grid)
    return PairedMultiframeInputs(raw_inputs, transformed_inputs, availability)
