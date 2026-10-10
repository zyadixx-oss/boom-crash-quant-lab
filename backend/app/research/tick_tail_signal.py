"""Pure fixed-M5 join of closed-frame inputs and strictly prior tick-tail state.

The original table is indexed by M5 opening time. A signal belongs to opening
time plus five minutes, and its tail state must exist at that exact timestamp.
No state, feature, invalid context row or missing timestamp is backfilled. The
caller supplies descriptions from the frozen training-only detector; this
module neither reads prices nor fits a detector, estimator or trading rule.
"""

from __future__ import annotations

from datetime import timedelta
import math
from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd

from app.research.multiframe_signal import FEATURE_NAMES


CADENCE_MINUTES = 5
AGE_DENOMINATOR_SECONDS = 600.0
TAIL_FEATURE_NAMES = ("tail_age_log1p", "prior_tail_mark")
_SECOND_NS = 1_000_000_000
_RESERVED = ("signal_time", "multiframe_feature_valid", "tail_feature_valid",
             "common_available", *TAIL_FEATURE_NAMES)


def _index(frame: pd.DataFrame, name: str, step_ns: int) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware UTC")
    if frame.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be nonempty, strictly sorted and unique")
    if any(stamp.utcoffset() != timedelta(0) for stamp in index):
        raise ValueError(f"{name} timestamps must be UTC")
    index = index.tz_convert("UTC").as_unit("ns")
    if np.any(index.asi8 % step_ns):
        raise ValueError(f"{name} timestamps must align to the fixed UTC grid")
    return index


def _real_or_unknown(value: object, name: str) -> float:
    if value is None or value is pd.NA:
        return math.nan
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real, not bool/complex/text")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite or unknown") from exc
    if math.isinf(result):
        raise ValueError(f"{name} must be finite or unknown")
    return result


def _values(series: pd.Series, name: str) -> np.ndarray:
    return np.fromiter((_real_or_unknown(value, name) for value in series.array),
                       dtype=float, count=len(series))


def _boolean(series: pd.Series, name: str) -> np.ndarray:
    if not all(isinstance(value, (bool, np.bool_)) for value in series.array):
        raise TypeError(f"{name} must contain explicit bool values")
    return series.to_numpy(dtype=bool)


def join_tick_tail_inputs(
    multiframe_inputs: pd.DataFrame,
    feature_names: Sequence[str],
    tail_description: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return one common availability table and ordered44/46 feature lists.

    ``multiframe_inputs`` must be the unfiltered M5-opening table returned by
    ``causal_multiframe_inputs``. Exactly its original ordered44 names are
    accepted. ``feature_valid`` is the caller's closed-frame validity flag.
    ``tail_description`` must expose the helper's strictly prior state:
    ``increment_known``, ``pre_event_age_seconds`` and ``prior_tail_mark``.
    Current ``tail_event``, quote and return never enter either feature list.

    NaN/None state is unknown and unavailable. Known age must be a positive
    whole second and known prior mark a positive directed log return. Numeric
    bool/text/complex/infinity is refused. Unknown original features keep their
    rows unavailable. The shared ``feature_valid``/``common_available`` mask
    applies to RIDGE44, RIDGE46 and the clock reference; no rows are dropped.
    """
    opening = _index(multiframe_inputs, "multiframe_inputs",
                     CADENCE_MINUTES * 60 * _SECOND_NS)
    tail_index = _index(tail_description, "tail_description", _SECOND_NS)
    if isinstance(feature_names, (str, bytes)) or tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("Exactly the original ordered44 feature names are required")
    required = (*FEATURE_NAMES, "feature_valid")
    if any(name not in multiframe_inputs for name in required):
        raise ValueError("Original44 inputs and feature_valid are required")
    if any(name in multiframe_inputs for name in _RESERVED):
        raise ValueError("Output column names must not already exist in multiframe_inputs")
    tail_columns = ("increment_known", "pre_event_age_seconds", "prior_tail_mark")
    if any(name not in tail_description for name in tail_columns):
        raise ValueError("Exact tail increment/prior-state columns are required")

    frame_flag = _boolean(multiframe_inputs.feature_valid, "feature_valid")
    original_finite = np.ones(len(multiframe_inputs), dtype=bool)
    for name in FEATURE_NAMES:
        original_finite &= np.isfinite(_values(multiframe_inputs[name], name))
    increment_known = _boolean(tail_description.increment_known, "increment_known")
    ages = _values(tail_description.pre_event_age_seconds, "pre_event_age_seconds")
    marks = _values(tail_description.prior_tail_mark, "prior_tail_mark")
    known_age, known_mark = np.isfinite(ages), np.isfinite(marks)
    if np.any(known_age & ((ages < 1) | (ages != np.floor(ages)))):
        raise ValueError("Known prior age must be a positive whole second")
    if np.any(known_mark & (marks <= 0)):
        raise ValueError("Known prior mark must be a positive directed log return")

    # Reindex preserves unknown/missing exact rows; it never chooses an older
    # description. In particular, current-event flags are not used as state.
    prior = pd.DataFrame({"increment_known": increment_known,
                          "pre_event_age_seconds": ages,
                          "prior_tail_mark": marks}, index=tail_index)
    issues = opening + pd.Timedelta(minutes=CADENCE_MINUTES)
    joined = prior.reindex(issues)
    tail_valid = (joined.increment_known.eq(True).to_numpy()
                  & np.isfinite(joined.pre_event_age_seconds.to_numpy())
                  & np.isfinite(joined.prior_tail_mark.to_numpy()))
    frame_valid = frame_flag & original_finite
    common = frame_valid & tail_valid
    output = multiframe_inputs.copy(deep=True)
    output["signal_time"] = issues
    output["tail_age_log1p"] = np.log1p(joined.pre_event_age_seconds.to_numpy()
                                          / AGE_DENOMINATOR_SECONDS)
    output["prior_tail_mark"] = joined.prior_tail_mark.to_numpy()
    output["multiframe_feature_valid"] = frame_valid
    output["tail_feature_valid"] = tail_valid
    output["common_available"] = common
    output["feature_valid"] = common
    return output, list(FEATURE_NAMES), [*FEATURE_NAMES, *TAIL_FEATURE_NAMES]
