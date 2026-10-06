"""One strictly prior native-adverse tick-path feature; no fitted parameters.

At a decision timestamp t, native_adverse_semivariance600 is

    mean(min(s * log(P_i / P_(i-1)), 0)**2 for i=t-600,...,t-1) / sigma40**2

where s is the IMMUTABLE symbol direction (+1 Boom, -1 Crash) and sigma40 is
that symbol's already frozen first40 median absolute log-return scale. It is
native-adverse for BOTH SPIKE and DRIFT strategies, not strategy-adverse.

Exactly601 positive finite observed quotes t-601,...,t-1 are required, at one
second intervals. Quote t and all later quotes are excluded. A missing second
invalidates the feature until a new complete601-quote history exists; missing
quotes are never filled and windows never bridge gaps. The numerical form is
mean((min(s*r_i,0)/sigma40)**2), avoiding an underflowing squared denominator.

Descriptions are indexed by each observed quote timestamp+1second. Thus the
feature at t does not require quote t itself to exist. An exact absent decision
row remains unknown on reindex. The M5 join preserves every original input row
and imposes identical availability on the original44 and augmented45 inputs.
This module does not choose clocks, sides, splits or scales, fit models, read
files, detect events, or calculate trades, cash returns or profit factors.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from app.research.multiframe_signal import FEATURE_NAMES as MULTIFRAME_NAMES
from app.research.tick_tail import _validated_ticks, _side, _positive_real, _log_return
from app.research.tick_tail_signal import _index, _values, _boolean


WINDOW_INCREMENTS = 600
REQUIRED_QUOTES = WINDOW_INCREMENTS + 1
PATH_FEATURE_NAME = "native_adverse_semivariance600"
PATH_FEATURE_NAMES = (PATH_FEATURE_NAME,)
FEATURE_FORMULAS = {
    PATH_FEATURE_NAME: "mean(min(native_side*log(P_i/P_(i-1)),0)^2; i=t-600,...,t-1)/frozen_first40_scale^2"
}
FEATURE_NAMES = (*MULTIFRAME_NAMES, *PATH_FEATURE_NAMES)
_SECOND_NS = 1_000_000_000
_RESERVED = ("signal_time", "multiframe_feature_valid", "tick_path_feature_valid", "common_available", PATH_FEATURE_NAME)


def describe_tick_path_risk(ticks: pd.DataFrame, native_side: int,
                            frozen_scale: float) -> pd.DataFrame:
    """Describe one feature from an immutable supplied quote stream and scale.

    Native symbol side must be supplied independently of strategy mode. Input
    quotes are strictly sorted unique whole-second UTC values. The output row
    at timestamp t uses only the preceding601 consecutive quotes and is unknown
    before sufficient history or after a gap. Unknown values stay NaN with an
    explicit False validity flag. No source values are modified.
    """
    side = _side(native_side)
    scale = _positive_real(frozen_scale, "frozen_first40_scale")
    index, quotes = _validated_ticks(ticks)
    consecutive = np.r_[False, np.diff(index.asi8) == _SECOND_NS]
    squared = np.full(len(quotes), np.nan)
    for i in np.flatnonzero(consecutive):
        adverse = min(side * _log_return(float(quotes[i-1]), float(quotes[i])), 0.)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized = np.float64(adverse) / scale
                squared[i] = normalized * normalized
        except FloatingPointError as error:
            raise ValueError("Native-adverse variation arithmetic is not finite") from error
    values = pd.Series(squared).rolling(WINDOW_INCREMENTS, min_periods=WINDOW_INCREMENTS).mean().to_numpy(copy=True)
    # The rolling NaN at the first quote and at each gap enforces601 observed
    # quotes. The timestamp-span check independently excludes a bridged window.
    enough = np.zeros(len(quotes), dtype=bool)
    if len(quotes) >= REQUIRED_QUOTES:
        enough[WINDOW_INCREMENTS:] = (index.asi8[WINDOW_INCREMENTS:] - index.asi8[:-WINDOW_INCREMENTS]
                                       == WINDOW_INCREMENTS * _SECOND_NS)
    values[~enough] = np.nan
    valid = enough & np.isfinite(values)
    if np.any(enough & (~np.isfinite(values) | (values < 0))):
        raise ValueError("A complete native-adverse window must have finite nonnegative variation")
    decisions = (index + pd.Timedelta(seconds=1)).as_unit("ns")
    return pd.DataFrame({PATH_FEATURE_NAME: values, "tick_path_feature_valid": valid}, index=decisions)


def join_tick_path_risk_inputs(multiframe_inputs: pd.DataFrame,
                               feature_names: Sequence[str],
                               path_description: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Exact closed-M5 join returning the original grid and paired44/45 schemas.

    The original M5 opening index is retained. Its decision is opening+5min;
    only a descriptor row at that exact timestamp is considered. No nearest
    join, forward value or carry through a missing/invalid context is allowed.
    Existing frame validity and finite44 inputs are both required, as is known
    prior path risk. Invalid interior original rows are never removed.
    """
    opening = _index(multiframe_inputs, "multiframe_inputs", 5 * 60 * _SECOND_NS)
    descriptor_index = _index(path_description, "path_description", _SECOND_NS)
    if isinstance(feature_names, (str, bytes)) or tuple(feature_names) != MULTIFRAME_NAMES:
        raise ValueError("Exactly the original ordered44 feature names are required")
    if any(name not in multiframe_inputs for name in (*MULTIFRAME_NAMES, "feature_valid")):
        raise ValueError("All original44 inputs and explicit feature_valid are required")
    if any(name in multiframe_inputs for name in _RESERVED):
        raise ValueError("Output column names must not already exist in multiframe_inputs")
    if any(name not in path_description for name in (PATH_FEATURE_NAME, "tick_path_feature_valid")):
        raise ValueError("Exact risk feature and validity columns are required")
    original_flag = _boolean(multiframe_inputs.feature_valid, "feature_valid")
    all_finite = np.ones(len(multiframe_inputs), dtype=bool)
    for name in MULTIFRAME_NAMES:
        all_finite &= np.isfinite(_values(multiframe_inputs[name], name))
    risk = _values(path_description[PATH_FEATURE_NAME], PATH_FEATURE_NAME)
    risk_flag = _boolean(path_description.tick_path_feature_valid, "tick_path_feature_valid")
    if np.any(np.isfinite(risk) & (risk < 0)) or np.any(risk_flag & ~np.isfinite(risk)):
        raise ValueError("Known risk must be finite and nonnegative; unknown risk cannot be valid")
    prior = pd.DataFrame({PATH_FEATURE_NAME: risk, "tick_path_feature_valid": risk_flag}, index=descriptor_index)
    decisions = opening + pd.Timedelta(minutes=5)
    joined = prior.reindex(decisions)
    path_valid = joined.tick_path_feature_valid.eq(True).to_numpy() & np.isfinite(joined[PATH_FEATURE_NAME].to_numpy())
    frame_valid = original_flag & all_finite
    common = frame_valid & path_valid
    output = multiframe_inputs.copy(deep=True)
    output["signal_time"] = decisions
    output[PATH_FEATURE_NAME] = joined[PATH_FEATURE_NAME].to_numpy()
    output["multiframe_feature_valid"] = frame_valid
    output["tick_path_feature_valid"] = path_valid
    output["common_available"] = common
    output["feature_valid"] = common
    return output, list(MULTIFRAME_NAMES), list(FEATURE_NAMES)
