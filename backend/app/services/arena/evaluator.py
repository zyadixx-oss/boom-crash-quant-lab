from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..indicators import add_features
from ..market_structure import add_market_structure
from ..metrics import classification_metrics, spike_detection_metrics, trading_metrics
from ..spike import label_spikes


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _attach_completed_tf_context(x: pd.DataFrame, seconds: int, prefix: str) -> pd.DataFrame:
    """Attach only fully completed higher-timeframe context to each M1 row."""
    base = x[["epoch", "close"]].copy()
    base["bucket"] = (base["epoch"].astype(int) // seconds) * seconds
    agg = base.groupby("bucket", as_index=False).agg(close=("close", "last"))
    agg["context_epoch"] = agg["bucket"] + seconds
    agg[f"{prefix}_close"] = agg["close"]
    agg[f"{prefix}_trend"] = np.sign(agg["close"] - agg["close"].shift(3))
    context = agg[["context_epoch", f"{prefix}_close", f"{prefix}_trend"]].sort_values("context_epoch")

    left = pd.DataFrame({"row_index": x.index, "eval_epoch": x["epoch"].astype(int) + 60}).sort_values("eval_epoch")
    merged = pd.merge_asof(left, context, left_on="eval_epoch", right_on="context_epoch", direction="backward")
    merged = merged.sort_values("row_index")
    x[f"{prefix}_close"] = merged[f"{prefix}_close"].to_numpy()
    x[f"{prefix}_trend"] = merged[f"{prefix}_trend"].to_numpy()
    return x


def prepare_arena_frame(candles: pd.DataFrame, profile, with_spikes: bool = True) -> pd.DataFrame:
    required = {"epoch", "open", "high", "low", "close"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    if len(candles) < 80:
        raise ValueError("At least 80 chronological candles are required")

    x = add_features(candles, profile)
    x = add_market_structure(x, lookback=profile.support_resistance_lookback)
    x["body_atr"] = x["body"] / x["atr"].replace(0, np.nan)
    x["range_atr"] = x["range"] / x["atr"].replace(0, np.nan)
    x["upper_wick_ratio"] = x["upper_wick"] / x["body"].replace(0, np.nan)
    x["lower_wick_ratio"] = x["lower_wick"] / x["body"].replace(0, np.nan)
    x["small_cluster_3"] = (x["range_atr"] < 0.9).rolling(3, min_periods=3).sum() >= 3
    x["small_cluster_4"] = (x["range_atr"] < 0.9).rolling(4, min_periods=4).sum() >= 4
    x = _attach_completed_tf_context(x, 300, "m5")
    x = _attach_completed_tf_context(x, 900, "m15")
    x["range_atr_mean"] = x["range_atr"].rolling(20, min_periods=10).mean().shift(1)
    if with_spikes:
        direction = "UP" if profile.symbol.startswith("BOOM") else "DOWN"
        x = label_spikes(x, profile, direction)
    return x


def _generic_scores(row, profile, direction: str) -> dict[str, float]:
    sweep = bool(row.get("liquidity_sweep_down", False)) if direction == "UP" else bool(row.get("liquidity_sweep_up", False))
    fvg = bool(row.get("fvg_up", False)) if direction == "UP" else bool(row.get("fvg_down", False))
    ob = bool(row.get("order_block_bull", False)) if direction == "UP" else bool(row.get("order_block_bear", False))
    pd_zone = bool(row.get("discount", False)) if direction == "UP" else bool(row.get("premium", False))
    inducement = bool(row.get("equal_lows", False)) if direction == "UP" else bool(row.get("equal_highs", False))
    rejection = _finite(row.get("lower_wick_ratio")) and row["lower_wick_ratio"] >= 1.5 if direction == "UP" else _finite(row.get("upper_wick_ratio")) and row["upper_wick_ratio"] >= 1.5

    ict = 20 * sweep + 15 * bool(row.get("choch", False)) + 15 * fvg + 15 * ob + 15 * pd_zone + 10 * inducement
    if _finite(row.get("body_atr")) and row["body_atr"] >= 0.65:
        ict += 10

    volatility = 0.0
    if _finite(row.get("bb_ratio")) and row["bb_ratio"] < profile.bb_squeeze_threshold:
        volatility += 35
    if _finite(row.get("atr_ratio")) and row["atr_ratio"] < profile.atr_compression_threshold:
        volatility += 35
    if _finite(row.get("tick_acceleration")) and abs(float(row["tick_acceleration"])) > profile.tick_velocity_threshold:
        volatility += 15
    if _finite(row.get("range_atr")) and _finite(row.get("range_atr_mean")) and row["range_atr"] > row["range_atr_mean"] * 1.4:
        volatility += 15

    candle = 0.0
    if bool(row.get("small_cluster_3", False)):
        candle += 25
    if _finite(row.get("range_atr")) and row["range_atr"] < 0.8:
        candle += 20
    if rejection:
        candle += 20
    if sweep:
        candle += 20
    if _finite(row.get("range_atr")) and _finite(row.get("range_atr_mean")) and row["range_atr"] > row["range_atr_mean"] * 1.5:
        candle += 15

    near_sr = False
    if _finite(row.get("atr")) and float(row["atr"]) > 0:
        if direction == "UP" and _finite(row.get("support")):
            near_sr = abs(float(row["close"]) - float(row["support"])) <= float(row["atr"]) * 0.4
        if direction == "DOWN" and _finite(row.get("resistance")):
            near_sr = abs(float(row["resistance"]) - float(row["close"])) <= float(row["atr"]) * 0.4
    rejections = float(row.get("rejection_low_count", 0) or 0) if direction == "UP" else float(row.get("rejection_high_count", 0) or 0)
    equal_pool = bool(row.get("equal_lows", False)) if direction == "UP" else bool(row.get("equal_highs", False))
    sr = 35 * near_sr + 20 * sweep + 20 * (rejections >= 2) + 15 * pd_zone + 10 * equal_pool

    if direction == "UP":
        m5 = _finite(row.get("m5_trend")) and float(row["m5_trend"]) > 0
        m15 = _finite(row.get("m15_trend")) and float(row["m15_trend"]) > 0
    else:
        m5 = _finite(row.get("m5_trend")) and float(row["m5_trend"]) < 0
        m15 = _finite(row.get("m15_trend")) and float(row["m15_trend"]) < 0
    m1 = (_finite(row.get("bb_ratio")) and row["bb_ratio"] < profile.bb_squeeze_threshold) or sweep
    tfw = profile.timeframe_weights
    mtf = 100.0 * (tfw.get("M1", 0.25) * bool(m1) + tfw.get("M5", 0.45) * bool(m5) + tfw.get("M15", 0.30) * bool(m15))

    return {
        "ict": min(100.0, float(ict)),
        "volatility": min(100.0, float(volatility)),
        "candle": min(100.0, float(candle)),
        "sr": min(100.0, float(sr)),
        "mtf": min(100.0, float(mtf)),
    }


def score_candidate_row(row, candidate, profile, direction: str) -> tuple[float, list[str]]:
    p = dict(candidate.parameters or {})
    agent = candidate.agent_name
    base = _generic_scores(row, profile, direction)
    reasons: list[str] = []

    if agent == "ICT / Liquidity Agent":
        score = base["ict"]
        displacement = float(p.get("displacement_atr", 0.65))
        if _finite(row.get("body_atr")) and float(row["body_atr"]) < displacement:
            score = max(0.0, score - 15.0)
        if p.get("inducement_required"):
            inducement = bool(row.get("equal_lows", False)) if direction == "UP" else bool(row.get("equal_highs", False))
            if not inducement:
                score = max(0.0, score - 20.0)
        reasons = ["ICT", "LIQUIDITY", "STRUCTURE"]

    elif agent == "Volatility Compression Agent":
        score = 0.0
        bb = float(p.get("bb_squeeze_threshold", profile.bb_squeeze_threshold))
        atr = float(p.get("atr_compression_threshold", profile.atr_compression_threshold))
        if _finite(row.get("bb_ratio")) and row["bb_ratio"] < bb:
            score += 40
            reasons.append("BB_SQUEEZE")
        if _finite(row.get("atr_ratio")) and row["atr_ratio"] < atr:
            score += 40
            reasons.append("ATR_COMPRESSION")
        if _finite(row.get("tick_acceleration")) and abs(float(row["tick_acceleration"])) > profile.tick_velocity_threshold:
            score += 20
            reasons.append("VOL_EXPANSION")

    elif agent == "Candle Structure Agent":
        cluster_bars = int(p.get("cluster_bars", 3))
        tight = float(p.get("tight_range_atr", 0.8))
        cluster = bool(row.get("small_cluster_4" if cluster_bars >= 4 else "small_cluster_3", False))
        score = 30.0 * cluster
        if _finite(row.get("range_atr")) and row["range_atr"] < tight:
            score += 25
            reasons.append("TIGHT_RANGE")
        rejection_ratio = row.get("lower_wick_ratio") if direction == "UP" else row.get("upper_wick_ratio")
        if _finite(rejection_ratio) and float(rejection_ratio) >= 1.5:
            score += 20
            reasons.append("REJECTION_WICK")
        sweep = bool(row.get("liquidity_sweep_down", False)) if direction == "UP" else bool(row.get("liquidity_sweep_up", False))
        if sweep:
            score += 15
            reasons.append("FAKE_BREAK_RETURN")
        if _finite(row.get("range_atr")) and _finite(row.get("range_atr_mean")) and row["range_atr"] > row["range_atr_mean"] * 1.5:
            score += 10
            reasons.append("RANGE_EXPANSION")

    elif agent == "Support / Resistance Agent":
        proximity = float(p.get("proximity_atr", 0.4))
        near = False
        if _finite(row.get("atr")) and float(row["atr"]) > 0:
            ref = row.get("support") if direction == "UP" else row.get("resistance")
            near = _finite(ref) and abs(float(row["close"]) - float(ref)) <= float(row["atr"]) * proximity
        score = 40.0 * near + 0.6 * base["sr"]
        reasons = ["SUPPORT_RESISTANCE", "LIQUIDITY_POOL"] if score else []

    elif agent == "Multi-Timeframe Agent":
        if direction == "UP":
            m5 = _finite(row.get("m5_trend")) and float(row["m5_trend"]) > 0
            m15 = _finite(row.get("m15_trend")) and float(row["m15_trend"]) > 0
        else:
            m5 = _finite(row.get("m5_trend")) and float(row["m5_trend"]) < 0
            m15 = _finite(row.get("m15_trend")) and float(row["m15_trend"]) < 0
        m1 = (_finite(row.get("bb_ratio")) and row["bb_ratio"] < profile.bb_squeeze_threshold) or base["ict"] >= 40
        score = 100.0 * (
            float(p.get("m1_weight", profile.timeframe_weights["M1"])) * bool(m1)
            + float(p.get("m5_weight", profile.timeframe_weights["M5"])) * bool(m5)
            + float(p.get("m15_weight", profile.timeframe_weights["M15"])) * bool(m15)
        )
        reasons = [name for name, ok in [("M1_EARLY", m1), ("M5_CONFIRM", m5), ("M15_CONTEXT", m15)] if ok]

    else:
        weights = {
            "ict": float(p.get("ict_weight", 0.25)),
            "volatility": float(p.get("volatility_weight", 0.20)),
            "candle": float(p.get("candle_weight", 0.15)),
            "sr": float(p.get("sr_weight", 0.20)),
            "mtf": float(p.get("mtf_weight", 0.20)),
        }
        total = sum(weights.values()) or 1.0
        score = sum(base[k] * w for k, w in weights.items()) / total
        reasons = [f"{k.upper()}_VOTE" for k, v in base.items() if v >= 50]

    return round(min(100.0, max(0.0, float(score))), 4), reasons


def _curves(trades: list[dict]) -> tuple[list[dict], list[dict]]:
    equity = 0.0
    peak = 0.0
    equity_curve = []
    drawdown_curve = []
    for trade in trades:
        equity += float(trade["pnl"])
        peak = max(peak, equity)
        dd = peak - equity
        equity_curve.append({"epoch": int(trade["epoch"]), "equity": round(equity, 8)})
        drawdown_curve.append({"epoch": int(trade["epoch"]), "drawdown": round(dd, 8)})
    return equity_curve, drawdown_curve


def evaluate_candidate(candles: pd.DataFrame, candidate, profile) -> dict:
    x = prepare_arena_frame(candles, profile, with_spikes=True)
    direction = "UP" if profile.symbol.startswith("BOOM") else "DOWN"
    threshold = float((candidate.parameters or {}).get("min_signal_score", profile.min_signal_score))
    warmup = max(60, profile.bb_period + 40, profile.atr_period + 40)
    horizon = profile.test_window_bars

    truths: list[bool] = []
    predictions: list[bool] = []
    signals: list[int] = []
    trades: list[dict] = []
    last_signal = -10**9

    for i in range(warmup, len(x) - horizon):
        row = x.iloc[i]
        score, reasons = score_candidate_row(row, candidate, profile, direction)
        pred = score >= threshold
        predictions.append(pred)
        truths.append(bool(row.get("pre_spike", False)))
        if not pred or i - last_signal < profile.cooldown_bars:
            continue
        entry = float(row["close"])
        exit_price = float(x.iloc[i + horizon]["close"])
        pnl = exit_price - entry if direction == "UP" else entry - exit_price
        signals.append(i)
        trades.append(
            {
                "epoch": int(row["epoch"]),
                "direction": direction,
                "entry": entry,
                "exit": exit_price,
                "pnl": float(pnl),
                "score": score,
                "reasons": reasons,
            }
        )
        last_signal = i

    metrics = trading_metrics([t["pnl"] for t in trades])
    metrics.update(classification_metrics(truths, predictions))
    metrics["total_signals"] = len(signals)
    metrics["signal_frequency"] = len(signals) / max(1, len(predictions))

    spike_metrics = spike_detection_metrics(x, signals, profile, direction)
    equity_curve, drawdown_curve = _curves(trades)
    return {
        "metrics": metrics,
        "spike_metrics": spike_metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "rows_evaluated": len(predictions),
    }


def clone_candidate(candidate, parameters: dict):
    return SimpleNamespace(
        id=candidate.id,
        name=candidate.name,
        agent_name=candidate.agent_name,
        symbol=candidate.symbol,
        version=candidate.version,
        parameters=parameters,
        rules=candidate.rules,
        indicators_used=candidate.indicators_used,
        timeframes=candidate.timeframes,
    )
