from copy import deepcopy
from dataclasses import dataclass, asdict, field


@dataclass
class SymbolProfile:
    symbol: str
    display_name: str
    api_symbol: str
    min_signal_score: float = 60.0
    atr_period: int = 14
    atr_compression_threshold: float = 0.82
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_squeeze_threshold: float = 0.80
    stochastic_period: int = 14
    support_resistance_lookback: int = 20
    false_signal_threshold: float = 0.75
    tick_velocity_threshold: float = 1.5
    spike_atr_multiple: float = 2.5
    spike_window: int = 5
    pre_spike_window: int = 8
    cooldown_bars: int = 10
    test_window_bars: int = 5
    timeframe_weights: dict = field(default_factory=lambda: {"M1": 0.25, "M5": 0.45, "M15": 0.30})
    feature_flags: dict = field(default_factory=lambda: {
        "ict": True,
        "rsi": True,
        "bollinger": True,
        "atr": True,
        "tick_velocity": True,
        "m15": True,
    })
    weights: dict = field(default_factory=lambda: {
        "bollinger_squeeze": 15,
        "atr_compression": 15,
        "candle_structure": 10,
        "liquidity_sweep": 15,
        "support_resistance": 10,
        "rsi_stoch": 10,
        "tick_acceleration": 10,
        "m5_confirmation": 10,
        "m15_context": 5,
    })


# api_symbol stays configurable because Deriv active_symbols is the source of truth.
BASE = {
    "BOOM300": ("Boom 300", "BOOM300"),
    "BOOM500": ("Boom 500", "BOOM500"),
    "BOOM600": ("Boom 600", "BOOM600"),
    "BOOM900": ("Boom 900", "BOOM900"),
    "BOOM1000": ("Boom 1000", "BOOM1000"),
    "CRASH300": ("Crash 300", "CRASH300"),
    "CRASH500": ("Crash 500", "CRASH500"),
    "CRASH600": ("Crash 600", "CRASH600"),
    "CRASH900": ("Crash 900", "CRASH900"),
    "CRASH1000": ("Crash 1000", "CRASH1000"),
}


def _profile(symbol: str, name: str, api: str) -> SymbolProfile:
    p = SymbolProfile(symbol=symbol, display_name=f"{name} Index", api_symbol=api)
    n = int("".join(filter(str.isdigit, symbol)))
    p.spike_atr_multiple = {300: 2.2, 500: 2.4, 600: 2.5, 900: 2.7, 1000: 2.8}[n]
    p.tick_velocity_threshold = {300: 1.30, 500: 1.40, 600: 1.45, 900: 1.55, 1000: 1.60}[n]
    p.min_signal_score = {300: 58, 500: 60, 600: 61, 900: 63, 1000: 64}[n]
    p.support_resistance_lookback = {300: 16, 500: 20, 600: 22, 900: 26, 1000: 28}[n]
    p.false_signal_threshold = {300: 0.80, 500: 0.75, 600: 0.73, 900: 0.70, 1000: 0.68}[n]
    p.spike_window = {300: 4, 500: 5, 600: 5, 900: 6, 1000: 6}[n]
    p.pre_spike_window = {300: 6, 500: 8, 600: 8, 900: 10, 1000: 10}[n]
    # Independent seeds, intentionally conservative and not profitability claims.
    if symbol.startswith("BOOM"):
        p.timeframe_weights = {"M1": 0.30, "M5": 0.45, "M15": 0.25}
    else:
        p.timeframe_weights = {"M1": 0.25, "M5": 0.50, "M15": 0.25}
    return p


PROFILES = {s: _profile(s, *v) for s, v in BASE.items()}


def get_profile(symbol: str) -> SymbolProfile:
    key = symbol.upper().replace(" ", "").replace("INDEX", "")
    if key not in PROFILES:
        raise KeyError(f"Unsupported symbol: {symbol}")
    return deepcopy(PROFILES[key])


def list_profiles():
    return [asdict(p) for p in PROFILES.values()]
