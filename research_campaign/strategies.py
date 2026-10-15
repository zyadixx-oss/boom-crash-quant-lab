"""Finite, declared, single-frame research hypotheses; no broker operations.

All array positions describe a *closed* candle.  Entry belongs to the following
candle and is implemented by engine.py.  Synthetic arrays are for tests only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd

TIMEFRAMES = {"M1": 60, "M2": 120, "M3": 180, "M5": 300, "M10": 600,
              "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
FAMILIES = (
    "ict_liquidity_mss", "crt", "support_resistance", "false_break",
    "bollinger_squeeze", "atr_compression_expansion", "rsi_momentum",
    "pre_spike", "post_spike", "mean_reversion", "trend",
    "statistical_volatility", "empirical_conditional_return",
)
LABELS = {
    "ict_liquidity_mss": "ICT liquidity sweep and structure shift",
    "crt": "Candle range reclaim", "support_resistance": "Support/resistance bounce",
    "false_break": "False range break", "bollinger_squeeze": "Bollinger squeeze breakout",
    "atr_compression_expansion": "ATR compression and expansion",
    "rsi_momentum": "RSI momentum cross", "pre_spike": "Pre-spike compression hypothesis",
    "post_spike": "Post-spike continuation", "mean_reversion": "Bollinger/RSI mean reversion",
    "trend": "EMA trend resumption", "statistical_volatility": "Volatility-normalized momentum",
    "empirical_conditional_return": "Frozen training-only conditional return states",
}


@dataclass(frozen=True)
class StrategySpec:
    family: str
    round_index: int
    lookback: int
    threshold: float
    stop_atr: float
    target_atr: float
    max_hold_bars: int
    atr_period: int = 14
    trailing_stop: bool = False

    def __post_init__(self) -> None:
        if self.family not in FAMILIES or isinstance(self.round_index,bool) or not isinstance(self.round_index,int) or not 0 <= self.round_index < 32:
            raise ValueError("Unknown family or undeclared round")
        if not all(np.isfinite(x) for x in (self.threshold,self.stop_atr,self.target_atr)):
            raise ValueError("Non-finite strategy parameters")
        if self.trailing_stop is not False or min(self.stop_atr, self.target_atr,self.threshold) <= 0:
            raise ValueError("Only positive fixed stops and targets are permitted")
        if any(isinstance(x,bool) or not isinstance(x,int) for x in (self.lookback,self.max_hold_bars,self.atr_period)):
            raise ValueError("Candle counts must be integers")
        if self.lookback < 3 or self.max_hold_bars < 1 or self.atr_period != 14:
            raise ValueError("Invalid predeclared parameters")

    @property
    def id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"{self.family}.r{self.round_index:02d}.{sha256(payload.encode()).hexdigest()[:12]}"

    @property
    def label(self) -> str:
        return LABELS[self.family]

    @property
    def name(self) -> str:
        return self.label

    @property
    def scientific_id(self) -> str:
        payload = asdict(self)
        payload.pop("round_index")
        if self.family in {"ict_liquidity_mss", "trend"}:
            payload.pop("threshold")  # These signals never consume this value.
        return sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "id": self.id, "scientific_id":self.scientific_id,"label": self.label}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StrategySpec":
        payload = {k: v for k, v in values.items() if k not in {"id", "scientific_id", "label", "name"}}
        result = cls(**payload)
        if values.get("id", result.id) != result.id:
            raise ValueError("Strategy identity mismatch")
        if values.get("scientific_id",result.scientific_id)!=result.scientific_id:
            raise ValueError("Scientific cache identity mismatch")
        return result


def candidates_for_round(round_index: int) -> list[StrategySpec]:
    """Exactly 13 candidates per round, 416 declared candidates overall.

    Five binary choices are fixed before measurement; no optimizer generates
    another candidate from final-test results.  Each symbol/frame ranks its own
    validation results.  These parameters are hypotheses, not calibrated values.
    """
    if isinstance(round_index, bool) or not isinstance(round_index, int) or not 0 <= round_index < 32:
        raise ValueError("round_index must be an integer in [0,31]")
    lookback = (12, 24)[round_index & 1]
    threshold = (0.8, 1.2)[(round_index >> 1) & 1]
    stop = (1.5, 2.5)[(round_index >> 2) & 1]
    target = (2.0, 3.0)[(round_index >> 3) & 1]
    hold = (4, 8)[(round_index >> 4) & 1]
    return [StrategySpec(f, round_index, lookback, threshold, stop, target, hold) for f in FAMILIES]


def direction_for_symbol(symbol: str) -> int:
    name = symbol.upper()
    if name.startswith("BOOM"):
        return 1
    if name.startswith("CRASH"):
        return -1
    raise ValueError("Only independently researched Boom/Crash symbols are supported")


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Accept seconds under epoch or timestamp; do not silently sort/repair."""
    source = bars.copy(deep=False)
    if "epoch" not in source and "timestamp" in source:
        source = source.rename(columns={"timestamp": "epoch"})
    elif "epoch" in source and "timestamp" in source:
        if not np.array_equal(source["epoch"].to_numpy(),source["timestamp"].to_numpy()):
            raise ValueError("Conflicting epoch/timestamp aliases")
    required = {"epoch", "open", "high", "low", "close"}
    if not required.issubset(source):
        raise ValueError(f"Missing candle fields: {sorted(required - set(source))}")
    values = source[list(sorted(required))].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite candle data")
    epochs = source["epoch"].to_numpy(dtype=float)
    if np.any(epochs != np.floor(epochs)) or np.any(np.diff(epochs) <= 0):
        raise ValueError("Timestamps must be unique increasing integer UTC seconds")
    o, h, l, c = (source[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    if np.any(l <= 0) or np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)):
        raise ValueError("Invalid OHLC candle envelope")
    return source


@dataclass
class StrategyContext:
    bars: pd.DataFrame
    arrays: dict[str, np.ndarray]
    timeframe_seconds: int | None = None


def _lag(values: np.ndarray, n: int = 1) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if n < len(values):
        out[n:] = values[:-n]
    return out


def _prepare_contiguous(bars: pd.DataFrame) -> StrategyContext:
    frame = normalize_bars(bars)
    a = {k: frame[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    c, h, l = a["close"], a["high"], a["low"]
    prev = _lag(c)
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev), np.abs(l-prev)))
    if len(tr):
        tr[0] = h[0]-l[0]
    a["atr"] = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    a["body"] = c-a["open"]
    a["return"] = c-prev
    a["prior_high3"] = pd.Series(h).rolling(3, min_periods=3).max().shift().to_numpy()
    a["prior_low3"] = pd.Series(l).rolling(3, min_periods=3).min().shift().to_numpy()
    for n in (12, 24):
        cs = pd.Series(c)
        a[f"high{n}"] = pd.Series(h).rolling(n, min_periods=n).max().shift().to_numpy()
        a[f"low{n}"] = pd.Series(l).rolling(n, min_periods=n).min().shift().to_numpy()
        a[f"mean{n}"] = cs.rolling(n, min_periods=n).mean().to_numpy()
        a[f"sd{n}"] = cs.rolling(n, min_periods=n).std(ddof=0).to_numpy()
        a[f"ema{n}"] = cs.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()
        a[f"slow_ema{n}"] = cs.ewm(span=n*3, adjust=False, min_periods=n*3).mean().to_numpy()
        width = 4*a[f"sd{n}"]/np.maximum(np.abs(a[f"mean{n}"]), 1e-12)
        a[f"width{n}"] = width
        a[f"width_floor{n}"] = pd.Series(width).rolling(n*3, min_periods=n*3).quantile(.25).shift().to_numpy()
        a[f"slow_atr{n}"] = pd.Series(tr).rolling(n*3, min_periods=n*3).mean().to_numpy()
    change = pd.Series(a["return"])
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ratio = np.divide(up.to_numpy(), down.to_numpy(), out=np.full(len(c), np.inf), where=down.to_numpy()>0)
    a["rsi"] = 100-100/(1+ratio)
    a["rsi"][(up.to_numpy()==0) & (down.to_numpy()==0)] = 50
    return StrategyContext(frame, a)


def prepare_context(bars: pd.DataFrame, timeframe: str | None = None) -> StrategyContext:
    """Reuse once per cell; every missing candle restarts indicator warmup.

    Supply the known timeframe whenever possible.  Inference of the smallest
    positive timestamp difference is only a compatibility fallback and cannot
    certify a source's true granularity.
    """
    frame=normalize_bars(bars)
    epochs=frame["epoch"].to_numpy(dtype=np.int64)
    if timeframe is not None and timeframe not in TIMEFRAMES:
        raise ValueError("Unsupported timeframe")
    step=TIMEFRAMES[timeframe] if timeframe else (int(np.min(np.diff(epochs))) if len(epochs)>1 else None)
    breaks=np.flatnonzero(np.diff(epochs)!=step)+1 if step else np.array([],dtype=int)
    if not len(breaks):
        result=_prepare_contiguous(frame); result.timeframe_seconds=step; return result
    boundaries=np.r_[0,breaks,len(frame)]
    chunks=[_prepare_contiguous(frame.iloc[int(a):int(b)]) for a,b in zip(boundaries[:-1],boundaries[1:])]
    return StrategyContext(frame,{key:np.concatenate([chunk.arrays[key] for chunk in chunks]) for key in chunks[0].arrays},step)


@dataclass(frozen=True)
class StrategyFit:
    strategy_id: str
    symbol: str
    trained_until_epoch: int
    state_counts: tuple[int, ...]
    state_means: tuple[float, ...]
    training_rows: int

    def __post_init__(self) -> None:
        if len(self.state_counts)!=6 or len(self.state_means)!=6:
            raise ValueError("Six frozen states required")
        if any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in self.state_counts):
            raise ValueError("Invalid fitted state counts")
        if not all(np.isfinite(v) for v in self.state_means) or sum(self.state_counts)!=self.training_rows:
            raise ValueError("Invalid fitted state means or training count")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StrategyFit":
        return cls(**{**values, "state_counts": tuple(values["state_counts"]),
                      "state_means": tuple(values["state_means"])})


def _states(context: StrategyContext, spec: StrategySpec, direction: int) -> np.ndarray:
    a = context.arrays
    ratio = a["atr"]/np.maximum(a[f"slow_atr{spec.lookback}"], 1e-12)
    vol = np.where(ratio < .8, 0, np.where(ratio > 1.2, 2, 1))
    momentum = (direction*a["return"] > 0).astype(int)
    state = vol*2+momentum
    state[~np.isfinite(ratio) | ~np.isfinite(a["return"])] = -1
    return state


def fit_strategy(train_bars: pd.DataFrame | StrategyContext, spec: StrategySpec,
                 symbol: str) -> StrategyFit | None:
    """Fit this extra hypothesis only on the caller's isolated training slice.

    Response for state at i is close[i+1]-close[i], observed within training.
    The last training state has no label and is excluded.  No inference-time
    refit is performed.  Its 6 fixed states and minimum 50 observations are
    declared independently of validation or final outcomes.
    """
    if spec.family != "empirical_conditional_return":
        return None
    ctx = train_bars if isinstance(train_bars, StrategyContext) else prepare_context(train_bars)
    d = direction_for_symbol(symbol)
    states = _states(ctx, spec, d)
    a = ctx.arrays
    response = np.full(len(states), np.nan)
    if len(states)>1:
        response[:-1] = d*np.diff(a["close"])/np.maximum(a["atr"][:-1], 1e-12)
        step=ctx.timeframe_seconds
        if step:
            response[:-1][np.diff(ctx.bars["epoch"].to_numpy(dtype=np.int64))!=step]=np.nan
    mask = (states>=0) & np.isfinite(response) & (a["atr"]>0)
    counts = np.bincount(states[mask], minlength=6)
    sums = np.bincount(states[mask], weights=response[mask], minlength=6)
    means = np.divide(sums, counts, out=np.zeros(6), where=counts>0)
    return StrategyFit(spec.id, symbol, int(ctx.bars["epoch"].iloc[-1]) if len(states) else -1,
                       tuple(int(x) for x in counts), tuple(float(x) for x in means), int(mask.sum()))


def generate_signals(bars: pd.DataFrame | StrategyContext, spec: StrategySpec,
                     symbol: str, fit: StrategyFit | None = None) -> np.ndarray:
    """Vectorized, native-direction signals from a single frame's closed bars."""
    ctx = bars if isinstance(bars, StrategyContext) else prepare_context(bars)
    a, n, d, threshold = ctx.arrays, spec.lookback, direction_for_symbol(symbol), spec.threshold
    o,h,l,c,atr = (a[k] for k in ("open","high","low","close","atr"))
    low, high = a[f"low{n}"], a[f"high{n}"]
    native = d*a["body"] > 0
    previous_c = _lag(c)
    if spec.family == "ict_liquidity_mss":
        m = ((l < low-.05*atr) & (c > a["prior_high3"])) if d==1 else ((h > high+.05*atr) & (c < a["prior_low3"]))
    elif spec.family == "crt":
        ph, pl = _lag(h), _lag(l)
        m = ((l < pl-.05*atr) & (c > pl) & (c < ph)) if d==1 else ((h > ph+.05*atr) & (c < ph) & (c > pl))
        m &= native & (np.abs(a["body"]) >= threshold*.15*atr)
    elif spec.family == "support_resistance":
        m = ((l <= low+threshold*.2*atr) & (c > low+.2*atr)) if d==1 else ((h >= high-threshold*.2*atr) & (c < high-.2*atr))
        m &= native
    elif spec.family == "false_break":
        m = ((l < low-threshold*.1*atr) & (c > low)) if d==1 else ((h > high+threshold*.1*atr) & (c < high))
        m &= native
    elif spec.family == "bollinger_squeeze":
        upper, lower = a[f"mean{n}"]+2*a[f"sd{n}"], a[f"mean{n}"]-2*a[f"sd{n}"]
        squeezed = _lag(a[f"width{n}"]) <= a[f"width_floor{n}"]
        m = squeezed & ((c > _lag(upper)) if d==1 else (c < _lag(lower))) & native
    elif spec.family == "atr_compression_expansion":
        compressed = _lag(atr) <= (.7+.1*threshold)*_lag(a[f"slow_atr{n}"])
        m = compressed & (np.abs(a["body"]) > threshold*_lag(atr)) & native
    elif spec.family == "rsi_momentum":
        level = 50+d*threshold*5
        rsi = a["rsi"]
        m = ((rsi>=level)&(_lag(rsi)<level)) if d==1 else ((rsi<=level)&(_lag(rsi)>level))
    elif spec.family == "pre_spike":
        small = pd.Series(np.abs(a["body"])).rolling(3,min_periods=3).mean().to_numpy()<threshold*.3*atr
        near = (c-low < .6*atr) if d==1 else (high-c < .6*atr)
        m = small & near & (atr<.9*a[f"slow_atr{n}"]) & (d*(c-_lag(c,3))<0)
    elif spec.family == "post_spike":
        m = (d*a["body"] > threshold*1.5*_lag(atr)) & native
    elif spec.family == "mean_reversion":
        m = ((c<a[f"mean{n}"]-threshold*a[f"sd{n}"]) & (a["rsi"]<35)) if d==1 else ((c>a[f"mean{n}"]+threshold*a[f"sd{n}"]) & (a["rsi"]>65))
    elif spec.family == "trend":
        fast, slow = a[f"ema{n}"], a[f"slow_ema{n}"]
        aligned = d*(fast-slow)>0
        m = aligned & (d*(c-fast)>0) & (d*(previous_c-_lag(fast))<=0)
    elif spec.family == "statistical_volatility":
        m = (d*a["return"] > threshold*atr) & (atr>a[f"slow_atr{n}"])
    else:
        if fit is None or fit.strategy_id != spec.id or fit.symbol != symbol:
            raise ValueError("A matching training-only frozen fit is required")
        if len(fit.state_counts)!=6 or len(fit.state_means)!=6:
            raise ValueError("Invalid fixed-state fit")
        states = _states(ctx, spec, d)
        safe = np.maximum(states, 0)
        m = (states>=0) & (np.asarray(fit.state_counts)[safe]>=50) & (np.asarray(fit.state_means)[safe] > .02*threshold)
        # The frozen model becomes available only at the close of its last
        # training candle.  Earlier training rows must never use that fit.
        m &= ctx.bars["epoch"].to_numpy(dtype=np.int64)>=fit.trained_until_epoch
    # The first closed candle must have full declared indicator context.
    m &= np.isfinite(atr) & (atr>0) & np.isfinite(a[f"slow_atr{n}"])
    return np.where(m, d, 0).astype(np.int8)
