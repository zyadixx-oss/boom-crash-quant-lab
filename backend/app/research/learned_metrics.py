"""Profit-factor uncertainty from paired UTC-day gains and losses.

This module measures hypothetical completed net-R ledgers. It has no data,
account, order, or model-fitting access. Ratios without losses stay unknown.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.research.payoff_metrics import _window


def _positive_integer(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _ledger_days(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, dict]:
    gains, losses = np.zeros(len(days)), np.zeros(len(days))
    counts = np.zeros(len(days), dtype=int)
    if trades.empty:
        return gains, losses, {"trades": 0, "completed": 0, "censored": 0,
                              "invalid_uncensored": 0, "active_days": 0}
    required = {"signal_time", "entry_time", "net_R", "censored"}
    if not required.issubset(trades):
        raise ValueError(f"Missing profit-factor fields: {sorted(required - set(trades))}")
    cohort = trades.copy()
    for name in ("signal_time", "entry_time"):
        cohort[name] = pd.to_datetime(cohort[name], utc=True, errors="coerce")
    if cohort.signal_time.isna().any():
        raise ValueError("Trade signal timestamps must be valid")
    cohort = cohort.loc[(cohort.signal_time >= start) & (cohort.signal_time < end)].copy()
    if cohort.censored.isna().any():
        raise ValueError("Trade censor flags must be known")
    if not cohort.censored.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Trade censor flags must be boolean")
    censored = cohort.censored.astype(bool).to_numpy()
    values = pd.to_numeric(cohort.net_R, errors="coerce").to_numpy(float)
    finite = np.isfinite(values)
    completed = cohort.loc[~censored & finite].copy()
    net = values[~censored & finite]
    if completed.entry_time.isna().any():
        raise ValueError("Completed trades require valid entry timestamps")
    if ((completed.entry_time < start) | (completed.entry_time >= end)).any():
        raise ValueError("Completed entry falls outside the analysis interval")
    if (completed.entry_time < completed.signal_time).any():
        raise ValueError("Completed entry cannot precede its signal")
    if len(completed):
        ids = ((completed.entry_time.dt.floor("D") - days[0]) // pd.Timedelta(days=1)).to_numpy(int)
        counts = np.bincount(ids, minlength=len(days))
        gains = np.bincount(ids, weights=np.maximum(net, 0), minlength=len(days))
        losses = np.bincount(ids, weights=np.maximum(-net, 0), minlength=len(days))
    if not np.isfinite(gains).all() or not np.isfinite(losses).all():
        raise ValueError("Non-finite daily gain or loss sums")
    return gains, losses, {"trades": len(cohort), "completed": len(completed),
                          "censored": int(censored.sum()),
                          "invalid_uncensored": int((~censored & ~finite).sum()),
                          "active_days": int((counts > 0).sum())}


def _ratio(gains: float, losses: float) -> float | None:
    if losses <= 0:
        return None
    ratio = gains / losses
    return float(ratio) if math.isfinite(ratio) else None


def _resample(gains: np.ndarray, losses: np.ndarray, repeats: int,
              seed: int, requested_block_days: int) -> dict:
    n = len(gains)
    block = min(requested_block_days, n)
    full_blocks, remainder = divmod(n, block)
    offsets = np.arange(block)
    windows = (np.arange(n)[:, None] + offsets) % n
    full_gain, full_loss = gains[windows].sum(axis=1), losses[windows].sum(axis=1)
    partial_gain = gains[windows[:, :remainder]].sum(axis=1) if remainder else None
    partial_loss = losses[windows[:, :remainder]].sum(axis=1) if remainder else None
    rng = np.random.default_rng(seed)
    samples = np.full(repeats, np.nan)
    no_loss, empty, nonfinite = 0, 0, 0
    # The same block starts resample gains and losses; truncate the last block
    # to preserve exactly n calendar-day observations in every replicate.
    for offset in range(0, repeats, 500):
        size = min(500, repeats - offset)
        starts = rng.integers(0, n, size=(size, full_blocks + bool(remainder)))
        gain_sum = full_gain[starts[:, :full_blocks]].sum(axis=1)
        loss_sum = full_loss[starts[:, :full_blocks]].sum(axis=1)
        if remainder:
            gain_sum += partial_gain[starts[:, -1]]
            loss_sum += partial_loss[starts[:, -1]]
        no_denominator = loss_sum <= 0
        no_loss += int(no_denominator.sum())
        empty += int((no_denominator & (gain_sum == 0)).sum())
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratios = np.divide(gain_sum, loss_sum, out=np.full(size, np.nan), where=~no_denominator)
        bad_numeric = ~no_denominator & ~np.isfinite(ratios)
        nonfinite += int(bad_numeric.sum())
        ratios[bad_numeric] = np.nan
        samples[offset:offset + size] = ratios
    finite = samples[np.isfinite(samples)]
    return {
        "profit_factor_ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None],
        "valid_replicates": len(finite), "undefined_replicates": repeats - len(finite),
        "no_loss_replicates": no_loss, "empty_replicates": empty,
        "nonfinite_ratio_replicates": nonfinite,
        "requested_block_days": requested_block_days, "effective_block_days": block,
        "resampled_calendar_days": n,
        "resampling": "iid_utc_days" if block == 1 else "circular_moving_utc_day_blocks",
        "inference_status": "NO_FINITE_RATIOS" if not len(finite) else
            ("INSUFFICIENT_CALENDAR_SPAN" if n < requested_block_days else "AVAILABLE"),
    }


def profit_factor_inference(trades: pd.DataFrame, start, end, repeats: int = 9999,
                            seed: int = 20261005, block_days: int = 1) -> dict:
    """Return empirical PF, daily CI, circular seven-day CI, and selected CI.

    Cohorts use signal timestamps in [start,end); clusters use completed entry
    UTC dates. Empty and partial calendar days remain in the resampling universe.
    Naive interval timestamps follow the existing metrics convention of UTC.
    CI quantiles use finite ratios only; every omitted no-loss replicate is
    counted explicitly. Neither zero losses nor an empty ledger implies success.
    """
    repeats = _positive_integer(repeats, "repeats")
    seed = _positive_integer(seed, "seed", minimum=0)
    block_days = _positive_integer(block_days, "block_days")
    start, end, days = _window(start, end)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Analysis boundaries must be valid timestamps")
    gains, losses, audit = _ledger_days(trades, start, end, days)
    gain_sum, loss_sum = float(gains.sum()), float(losses.sum())
    if not math.isfinite(gain_sum) or not math.isfinite(loss_sum):
        raise ValueError("Non-finite total gain or loss sums")
    day = _resample(gains, losses, repeats, seed, 1)
    weekly = _resample(gains, losses, repeats, seed, 7)
    selected = day if block_days == 1 else weekly if block_days == 7 else _resample(
        gains, losses, repeats, seed, block_days)
    return {
        **audit, "calendar_days": len(days), "gain_sum_net_R": gain_sum,
        "loss_sum_net_R": loss_sum, "profit_factor": _ratio(gain_sum, loss_sum),
        "day_profit_factor_ci95": day["profit_factor_ci95"],
        "weekly_profit_factor_ci95": weekly["profit_factor_ci95"],
        "day_valid_replicates": day["valid_replicates"],
        "day_undefined_replicates": day["undefined_replicates"],
        "weekly_valid_replicates": weekly["valid_replicates"],
        "weekly_undefined_replicates": weekly["undefined_replicates"],
        "profit_factor_ci95": selected["profit_factor_ci95"],
        "valid_replicates": selected["valid_replicates"],
        "undefined_replicates": selected["undefined_replicates"],
        "block_days": block_days, "day": day, "weekly": weekly, "selected": selected,
        "bootstrap_repeats": repeats, "bootstrap_seed": seed,
        "paired_daily_gain_loss_resampling": True,
        "undefined_ratio_policy": "unknown; counted explicitly and omitted from finite-ratio percentile CI",
    }
