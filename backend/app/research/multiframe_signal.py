"""Forty-four causal inputs from closed H4/H1/M15/M5/M1 candles.

The returned M5 index is the candle opening time. A decision may be issued only
at index + five minutes. Context joins include every row of each UTC frame,
including invalid rows: a missing latest candle never falls back to an older
valid candle. These inputs contain no fitted parameters or trading operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.research.learned_signal import (
    FEATURE_FORMULAS as BASE_FEATURE_FORMULAS,
    _validated_m1,
    causal_inputs,
)
from app.research.spike_hunter import aggregate_complete


EXTRA_FEATURE_FORMULAS = {
    "h4_aligned_return_3_atr": "s*(H4.close-H4.close.shift(3))/H4.A",
    "h4_range_atr": "H4.R/H4.A",
    "h4_favorable_close_position": "(H4.close-H4.low for Boom; H4.high-H4.close for Crash)/H4.R",
    "h4_aligned_distance_prior_favorable_12_atr": "s*(H4.close-H4.previous12favorableextreme)/H4.A",
    "h1_aligned_body_atr": "s*(H1.close-H1.open)/H1.A",
    "h1_range_atr": "H1.R/H1.A",
    "h1_favorable_close_position": "(H1.close-H1.low for Boom; H1.high-H1.close for Crash)/H1.R",
    "h1_aligned_distance_favorable_boundary_m5_atr": "s*(M5.close-H1.latest_favorable_boundary)/M5.A",
    "h1_aligned_distance_adverse_boundary_m5_atr": "s*(M5.close-H1.latest_adverse_boundary)/M5.A",
    "h1_recent_m5_sweep_reclaim": "float(min(M5.low,3)<H1.low for Boom; max(M5.high,3)>H1.high for Crash; AND H1.low<M5.close<H1.high); unknown if any input missing",
    "m15_aligned_return_3_atr": "s*(M15.close-M15.close.shift(3))/M15.A",
    "m15_atr_5_over_30": "M15.TR.rolling(5).mean()/M15.TR.rolling(30).mean()",
    "m15_range_atr": "M15.R/M15.A",
    "m15_aligned_body_atr": "s*(M15.close-M15.open)/M15.A",
    "m15_favorable_close_position": "(M15.close-M15.low for Boom; M15.high-M15.close for Crash)/M15.R",
    "m15_aligned_distance_prior_favorable_5_atr": "s*(M15.close-M15.previous5favorableextreme)/M15.A",
    "m15_favorable_fvg_atr": "max(0,M15.low-M15.high.shift(2) for Boom; M15.low.shift(2)-M15.high for Crash)/M15.A",
    "m1_aligned_body_atr": "s*(M1.close-M1.open)/M1.A",
    "m1_range_atr": "M1.R/M1.A",
    "m1_aligned_return_3_atr": "s*(M1.close-M1.close.shift(3))/M1.A",
    "m1_atr_5_over_30": "M1.TR.rolling(5).mean()/M1.TR.rolling(30).mean()",
    "m1_body_over_range": "abs(M1.close-M1.open)/M1.R",
    "m1_aligned_distance_prior_favorable_5_atr": "s*(M1.close-M1.previous5favorableextreme)/M1.A",
    "m1_favorable_fvg_atr": "max(0,M1.low-M1.high.shift(2) for Boom; M1.low.shift(2)-M1.high for Crash)/M1.A",
    "m1_body_over_prior_median_20": "abs(M1.close-M1.open)/abs(M1.close-M1.open).shift(1).rolling(20).median()",
}
FEATURE_FORMULAS = {**BASE_FEATURE_FORMULAS, **EXTRA_FEATURE_FORMULAS}
FEATURE_NAMES = tuple(FEATURE_FORMULAS)
FRAME_MINUTES = {"h4": 240, "h1": 60, "m15": 15, "m1": 1}


def _frame_values(frame: pd.DataFrame, side: int) -> pd.DataFrame:
    """Calculate on a full frame grid before aligning completed frame rows."""
    out = frame.copy()
    complete = frame[["open", "high", "low", "close"]].notna().all(axis=1)
    previous = frame.close.shift(1)
    ranges = frame.high - frame.low
    body = (frame.close - frame.open).abs()
    tr = pd.concat([ranges, (frame.high - previous).abs(),
                    (frame.low - previous).abs()], axis=1).max(axis=1)
    tr = tr.where(complete & previous.notna())
    atr = tr.rolling(14, min_periods=14).mean().where(lambda value: value > 0)
    positive_range = ranges.where(ranges > 0)
    out["row_valid"] = complete
    out["aligned_return_3_atr"] = side * (frame.close - frame.close.shift(3)) / atr
    out["range_atr"] = ranges / atr
    out["aligned_body_atr"] = side * (frame.close - frame.open) / atr
    out["favorable_close_position"] = (
        frame.close - frame.low if side == 1 else frame.high - frame.close
    ) / positive_range
    out["atr_5_over_30"] = (
        tr.rolling(5, min_periods=5).mean() / tr.rolling(30, min_periods=30).mean()
    )
    out["body_over_range"] = body / positive_range
    for periods in (5, 12):
        preceding = (frame.high.shift(1).rolling(periods, min_periods=periods).max()
                     if side == 1 else
                     frame.low.shift(1).rolling(periods, min_periods=periods).min())
        out[f"aligned_distance_prior_favorable_{periods}_atr"] = side * (frame.close - preceding) / atr
    gap = (frame.low - frame.high.shift(2) if side == 1 else
           frame.low.shift(2) - frame.high)
    out["favorable_fvg_atr"] = gap.clip(lower=0) / atr
    median_body = body.shift(1).rolling(20, min_periods=20).median()
    out["body_over_prior_median_20"] = body / median_body.where(median_body > 0)
    return out


def _closed_asof(frame: pd.DataFrame, issues: pd.DatetimeIndex,
                 minutes: int) -> pd.DataFrame:
    """Index-only backward as-of; invalid latest rows stay invalid."""
    source = frame.copy()
    source.insert(0, "closed_at", frame.index + pd.Timedelta(minutes=minutes))
    source = source.reset_index(drop=True)
    return pd.merge_asof(
        pd.DataFrame({"issue": issues}), source,
        left_on="issue", right_on="closed_at", direction="backward",
        allow_exact_matches=True,
    )


def causal_multiframe_inputs(m1: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, list[str]]:
    """Return the original M5 opening grid and exactly 44 ordered inputs.

    Validity requires the original nineteen inputs, every added finite input,
    and the latest closed OHLC row from all four added frames. M1 must be the
    exact minute ending at the decision time. Missing minute rows are inserted
    as NaN before rolling; no prices, events or derived inputs are interpolated.
    """
    if direction not in ("boom", "crash"):
        raise ValueError("direction must be boom or crash")
    minute = _validated_m1(m1)
    minute = minute.reindex(pd.date_range(minute.index[0], minute.index[-1], freq="min").as_unit("ns"))
    m5, base_names = causal_inputs(minute, direction)
    if base_names != list(BASE_FEATURE_FORMULAS):
        raise ValueError("Base input schema changed")
    issues = m5.index + pd.Timedelta(minutes=5)
    side = 1 if direction == "boom" else -1
    context = {}
    context_valid = pd.Series(True, index=m5.index)
    for label, minutes in FRAME_MINUTES.items():
        frame = minute if minutes == 1 else aggregate_complete(minute, minutes, minutes)
        joined = _closed_asof(_frame_values(frame, side), issues, minutes)
        m5[f"{label}_closed_at"] = joined.closed_at.array
        valid = joined.row_valid.eq(True).to_numpy()
        if minutes == 1:
            valid = valid & joined.closed_at.eq(issues).to_numpy()
        m5[f"{label}_row_valid"] = valid
        context_valid &= valid
        context[label] = joined
        for name in EXTRA_FEATURE_FORMULAS:
            prefix = label + "_"
            if name.startswith(prefix) and name[len(prefix):] in joined.columns:
                m5[name] = joined[name[len(prefix):]].to_numpy()

    h1 = context["h1"]
    favorable = h1.high if side == 1 else h1.low
    adverse = h1.low if side == 1 else h1.high
    atr = m5.atr.where(m5.atr > 0)
    m5["h1_aligned_distance_favorable_boundary_m5_atr"] = side * (m5.close.to_numpy() - favorable.to_numpy()) / atr
    m5["h1_aligned_distance_adverse_boundary_m5_atr"] = side * (m5.close.to_numpy() - adverse.to_numpy()) / atr
    recent_extreme = (m5.low.rolling(3, min_periods=3).min() if side == 1 else
                      m5.high.rolling(3, min_periods=3).max())
    ref_low = pd.Series(h1.low.to_numpy(), index=m5.index)
    ref_high = pd.Series(h1.high.to_numpy(), index=m5.index)
    swept = recent_extreme < ref_low if side == 1 else recent_extreme > ref_high
    reclaimed = (m5.close > ref_low) & (m5.close < ref_high)
    known = recent_extreme.notna() & m5.close.notna() & ref_low.notna() & ref_high.notna()
    m5["h1_recent_m5_sweep_reclaim"] = (swept & reclaimed).astype(float).where(known)
    finite = np.isfinite(m5.loc[:, FEATURE_NAMES].to_numpy()).all(axis=1)
    m5["feature_valid"] = m5.feature_valid & context_valid & finite
    return m5, list(FEATURE_NAMES)
