import pandas as pd


def label_spikes(df: pd.DataFrame, profile, direction: str) -> pd.DataFrame:
    """Label spike arrival at the bar where a large directional move has actually materialized."""
    x = df.copy()
    h = int(profile.spike_window)
    prior_close = x["close"].shift(h)
    signed_move = (x["close"] - prior_close) if direction == "UP" else (prior_close - x["close"])
    start_atr = x["atr"].shift(h).replace(0, pd.NA)
    x["spike_move_atr"] = signed_move / start_atr

    raw_spike = (x["spike_move_atr"] >= profile.spike_atr_multiple).fillna(False)
    # Count a sustained move once, at its first threshold crossing.
    x["is_spike"] = raw_spike & ~raw_spike.shift(1, fill_value=False)

    x["pre_spike"] = False
    spike_idx = x.index[x["is_spike"]].tolist()
    for i in spike_idx:
        if i > 0:
            x.loc[max(0, i - profile.pre_spike_window) : i - 1, "pre_spike"] = True

    x["post_spike"] = False
    for i in spike_idx:
        x.loc[i + 1 : min(len(x) - 1, i + h), "post_spike"] = True

    # Backward-compatible field used by persistence/metrics.
    x["future_move_atr"] = x["spike_move_atr"]
    return x
