from dataclasses import dataclass, asdict
import hashlib
import json

from ...symbol_profiles import get_profile


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    capabilities: list[str]
    indicators_used: list[str]
    timeframes: list[str]
    rules: list[str]


AGENT_SPECS: dict[str, AgentSpec] = {
    "ICT / Liquidity Agent": AgentSpec(
        name="ICT / Liquidity Agent",
        description="Liquidity/structure strategy family using only completed candle information.",
        capabilities=["liquidity_sweep", "displacement", "market_structure_shift", "fair_value_gap", "order_block", "premium_discount", "inducement"],
        indicators_used=["ATR", "market_structure", "FVG", "order_blocks", "liquidity"],
        timeframes=["M1", "M5", "M15"],
        rules=[
            "Score direction-specific liquidity sweep and structure shift.",
            "Confirm displacement/FVG/order-block context without future candles.",
            "Prefer discount for Boom and premium for Crash.",
        ],
    ),
    "Volatility Compression Agent": AgentSpec(
        name="Volatility Compression Agent",
        description="Compression-to-expansion strategy family using Bollinger and ATR state.",
        capabilities=["bollinger_squeeze", "atr_compression", "volatility_expansion"],
        indicators_used=["Bollinger Bands", "ATR", "tick acceleration"],
        timeframes=["M1", "M5"],
        rules=[
            "Require volatility compression before threshold crossing.",
            "Use acceleration/expansion only from the current or past candle.",
        ],
    ),
    "Candle Structure Agent": AgentSpec(
        name="Candle Structure Agent",
        description="Price-action strategy family for tight clusters, wicks, fake breaks and range expansion.",
        capabilities=["small_candle_clusters", "tight_ranges", "rejection_wicks", "fake_breakout_return", "abnormal_range_expansion"],
        indicators_used=["ATR-normalized candle range", "wick ratios", "rolling range"],
        timeframes=["M1", "M5"],
        rules=[
            "Measure candle body/range relative to ATR.",
            "Reward direction-specific rejection and fake-break return.",
            "Detect abnormal range state without looking ahead.",
        ],
    ),
    "Support / Resistance Agent": AgentSpec(
        name="Support / Resistance Agent",
        description="Structure strategy family around local S/R, swings, pools and repeated rejection.",
        capabilities=["local_support_resistance", "swing_highs_lows", "liquidity_pools", "repeated_rejection_zones"],
        indicators_used=["rolling support/resistance", "swings", "equal highs/lows"],
        timeframes=["M1", "M5", "M15"],
        rules=[
            "Use symbol-specific S/R lookback.",
            "Score proximity, sweep and repeated rejection independently.",
        ],
    ),
    "Multi-Timeframe Agent": AgentSpec(
        name="Multi-Timeframe Agent",
        description="Context/confirmation strategy family with M15 context, M5 confirmation and M1 early signal.",
        capabilities=["m15_context", "m5_confirmation", "m1_early_signal"],
        indicators_used=["rolling trend", "EMA", "ATR", "Bollinger"],
        timeframes=["M1", "M5", "M15"],
        rules=[
            "M15 contributes context only.",
            "M5 contributes confirmation.",
            "M1 contributes the earliest signal component.",
        ],
    ),
    "Ensemble Agent": AgentSpec(
        name="Ensemble Agent",
        description="Weighted ensemble over the five specialist agent scores.",
        capabilities=["agent_weighting", "consensus", "diversification"],
        indicators_used=["specialist agent scores"],
        timeframes=["M1", "M5", "M15"],
        rules=[
            "Combine specialist scores; do not treat any single indicator as sufficient.",
            "Weights are explicit and stored in candidate parameters.",
        ],
    ),
}


def _variants(agent_name: str, profile) -> list[dict]:
    base_threshold = float(profile.min_signal_score)
    if agent_name == "ICT / Liquidity Agent":
        return [
            {"min_signal_score": base_threshold, "displacement_atr": 0.65, "inducement_required": False},
            {"min_signal_score": base_threshold + 5, "displacement_atr": 0.85, "inducement_required": True},
            {"min_signal_score": base_threshold - 3, "displacement_atr": 0.55, "inducement_required": False},
        ]
    if agent_name == "Volatility Compression Agent":
        return [
            {"min_signal_score": base_threshold, "bb_squeeze_threshold": profile.bb_squeeze_threshold, "atr_compression_threshold": profile.atr_compression_threshold},
            {"min_signal_score": base_threshold + 5, "bb_squeeze_threshold": profile.bb_squeeze_threshold * 0.92, "atr_compression_threshold": profile.atr_compression_threshold * 0.94},
            {"min_signal_score": base_threshold - 3, "bb_squeeze_threshold": profile.bb_squeeze_threshold * 1.05, "atr_compression_threshold": profile.atr_compression_threshold * 1.04},
        ]
    if agent_name == "Candle Structure Agent":
        return [
            {"min_signal_score": base_threshold, "cluster_bars": 3, "tight_range_atr": 0.80},
            {"min_signal_score": base_threshold + 5, "cluster_bars": 4, "tight_range_atr": 0.70},
            {"min_signal_score": base_threshold - 3, "cluster_bars": 2, "tight_range_atr": 0.90},
        ]
    if agent_name == "Support / Resistance Agent":
        return [
            {"min_signal_score": base_threshold, "sr_lookback": profile.support_resistance_lookback, "proximity_atr": 0.40},
            {"min_signal_score": base_threshold + 5, "sr_lookback": profile.support_resistance_lookback + 4, "proximity_atr": 0.30},
            {"min_signal_score": base_threshold - 3, "sr_lookback": max(10, profile.support_resistance_lookback - 4), "proximity_atr": 0.50},
        ]
    if agent_name == "Multi-Timeframe Agent":
        return [
            {"min_signal_score": base_threshold, "m1_weight": profile.timeframe_weights["M1"], "m5_weight": profile.timeframe_weights["M5"], "m15_weight": profile.timeframe_weights["M15"]},
            {"min_signal_score": base_threshold + 5, "m1_weight": 0.20, "m5_weight": 0.50, "m15_weight": 0.30},
            {"min_signal_score": base_threshold - 3, "m1_weight": 0.35, "m5_weight": 0.40, "m15_weight": 0.25},
        ]
    return [
        {"min_signal_score": base_threshold, "ict_weight": 0.25, "volatility_weight": 0.20, "candle_weight": 0.15, "sr_weight": 0.20, "mtf_weight": 0.20},
        {"min_signal_score": base_threshold + 5, "ict_weight": 0.30, "volatility_weight": 0.20, "candle_weight": 0.10, "sr_weight": 0.20, "mtf_weight": 0.20},
        {"min_signal_score": base_threshold - 3, "ict_weight": 0.20, "volatility_weight": 0.25, "candle_weight": 0.20, "sr_weight": 0.15, "mtf_weight": 0.20},
    ]


def generate_candidate_payloads(symbol: str, agent_name: str | None = None, variants_per_agent: int = 2) -> list[dict]:
    profile = get_profile(symbol)
    selected = [agent_name] if agent_name else list(AGENT_SPECS)
    unknown = [name for name in selected if name not in AGENT_SPECS]
    if unknown:
        raise KeyError(f"Unknown agent: {unknown[0]}")

    payloads = []
    for name in selected:
        spec = AGENT_SPECS[name]
        for variant_index, params in enumerate(_variants(name, profile)[:variants_per_agent], start=1):
            identity = json.dumps({"symbol": profile.symbol, "agent": name, "params": params}, sort_keys=True)
            digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
            payloads.append(
                {
                    "id": f"{profile.symbol.lower()}-{digest}",
                    "name": f"{profile.display_name} · {name.replace(' Agent', '')} v{variant_index}",
                    "agent_name": name,
                    "symbol": profile.symbol,
                    "version": f"1.0.{variant_index}",
                    "parameters": params,
                    "rules": spec.rules,
                    "indicators_used": spec.indicators_used,
                    "timeframes": spec.timeframes,
                    "training_period": None,
                    "validation_period": None,
                    "status": "created",
                }
            )
    return payloads


def serialized_agent_specs() -> list[dict]:
    return [asdict(spec) for spec in AGENT_SPECS.values()]
