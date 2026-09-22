import numpy as np
import pandas as pd


def add_market_structure(
    df: pd.DataFrame,
    swing: int = 3,
    tolerance: float = 0.0015,
    lookback: int = 20,
) -> pd.DataFrame:
    x = df.copy()
    win = 2 * swing + 1

    # Causal pivot confirmation: at row i we may confirm only the pivot at i-swing.
    # This removes the old center=True lookahead while retaining swing structure.
    candidate_high = x["high"].shift(swing)
    candidate_low = x["low"].shift(swing)
    x["swing_high"] = candidate_high.where(candidate_high.eq(x["high"].rolling(win).max()))
    x["swing_low"] = candidate_low.where(candidate_low.eq(x["low"].rolling(win).min()))

    prev_high = x["swing_high"].ffill().shift(1)
    prev_low = x["swing_low"].ffill().shift(1)
    x["bos_up"] = x["close"] > prev_high
    x["bos_down"] = x["close"] < prev_low
    trend = np.where(x["bos_up"], 1, np.where(x["bos_down"], -1, np.nan))
    x["trend"] = pd.Series(trend, index=x.index).ffill().fillna(0)
    x["choch"] = ((x["trend"].shift(1) == 1) & x["bos_down"]) | ((x["trend"].shift(1) == -1) & x["bos_up"])

    rolling_high = x["high"].rolling(lookback, min_periods=max(5, lookback // 4)).max().shift(1)
    rolling_low = x["low"].rolling(lookback, min_periods=max(5, lookback // 4)).min().shift(1)
    x["liquidity_sweep_up"] = (x["high"] > rolling_high) & (x["close"] < rolling_high)
    x["liquidity_sweep_down"] = (x["low"] < rolling_low) & (x["close"] > rolling_low)
    x["support"] = rolling_low
    x["resistance"] = rolling_high

    x["fvg_up"] = x["low"] > x["high"].shift(2)
    x["fvg_down"] = x["high"] < x["low"].shift(2)
    x["equal_highs"] = (x["high"] - x["high"].shift(1)).abs() / x["close"].replace(0, np.nan) < tolerance
    x["equal_lows"] = (x["low"] - x["low"].shift(1)).abs() / x["close"].replace(0, np.nan) < tolerance
    x["order_block_bull"] = (x["close"].shift(1) < x["open"].shift(1)) & x["bos_up"]
    x["order_block_bear"] = (x["close"].shift(1) > x["open"].shift(1)) & x["bos_down"]

    midpoint = (rolling_high + rolling_low) / 2.0
    x["range_mid"] = midpoint
    x["discount"] = x["close"] <= midpoint
    x["premium"] = x["close"] >= midpoint
    x["rejection_high_count"] = x["liquidity_sweep_up"].astype(int).rolling(lookback, min_periods=1).sum()
    x["rejection_low_count"] = x["liquidity_sweep_down"].astype(int).rolling(lookback, min_periods=1).sum()
    return x
