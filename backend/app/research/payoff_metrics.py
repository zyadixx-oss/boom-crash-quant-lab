"""UTC-day inference for offline quote-path R; no account or execution access."""

from __future__ import annotations

import numpy as np
import pandas as pd

_FIELDS = ("signal_time", "entry_time", "exit_time", "gross_R", "net_R",
           "reason", "ambiguous", "holding_minutes", "censored")


def _utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def _window(start, end) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("Analysis end must be after start")
    days = pd.date_range(start.floor("D"), (end - pd.Timedelta(nanoseconds=1)).floor("D"), freq="D")
    return start, end, days


def _cohort(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cohort = trades.copy()
    if cohort.empty:
        for field in _FIELDS:
            if field not in cohort:
                cohort[field] = pd.Series(dtype=object)
        return cohort
    missing = set(_FIELDS) - set(cohort)
    if missing:
        raise ValueError(f"Missing payoff fields: {sorted(missing)}")
    for field in ("signal_time", "entry_time", "exit_time"):
        cohort[field] = pd.to_datetime(cohort[field], utc=True, errors="coerce")
    if cohort.signal_time.isna().any():
        raise ValueError("Trade signal timestamps must be valid")
    return cohort.loc[(cohort.signal_time >= start) & (cohort.signal_time < end)].copy()


def _completed(cohort: pd.DataFrame) -> pd.DataFrame:
    if cohort.empty:
        return cohort.copy()
    net = pd.to_numeric(cohort.net_R, errors="coerce")
    completed = cohort.loc[~cohort.censored.astype(bool) & np.isfinite(net)].copy()
    completed["net_R"] = pd.to_numeric(completed.net_R, errors="coerce").astype(float)
    completed["gross_R"] = pd.to_numeric(completed.gross_R, errors="coerce")
    if completed[["entry_time", "exit_time"]].isna().any().any():
        raise ValueError("Completed trades require valid entry and exit timestamps")
    return completed


def _daily(completed: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    counts, sums = np.zeros(len(days)), np.zeros(len(days))
    if not completed.empty:
        ids = ((completed.entry_time.dt.floor("D") - days[0]) // pd.Timedelta(days=1)).to_numpy(int)
        if ((ids < 0) | (ids >= len(days))).any():
            raise ValueError("Completed entry falls outside analysis calendar days")
        counts = np.bincount(ids, minlength=len(days)).astype(float)
        sums = np.bincount(ids, weights=completed.net_R.to_numpy(float), minlength=len(days))
    return counts, sums


def _interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]


def summarize(trades: pd.DataFrame, start, end, stop_atr: float,
              risk_fraction: float = 0.0025) -> dict:
    if not np.isfinite(stop_atr) or stop_atr <= 0:
        raise ValueError("stop_atr must be finite and positive")
    if not np.isfinite(risk_fraction) or not 0 <= risk_fraction < 1:
        raise ValueError("risk_fraction must lie in [0,1)")
    start, end, days = _window(start, end)
    cohort = _cohort(trades, start, end)
    completed = _completed(cohort)
    values = completed.net_R.to_numpy(float)
    n = len(values)
    counts, sums = _daily(completed, days)
    mean = float(values.mean()) if n else None
    positive, negative = values[values > 0], values[values < 0]
    loss_sum = float(-negative.sum())
    gross = completed.gross_R.to_numpy(float)
    gross = gross[np.isfinite(gross)]
    mean_gross = float(gross.mean()) if len(gross) == n and n else None
    se = None
    if n and len(days) > 1:
        # Per-day influence is (sumDayR - mean * nDay)/N; divide by N once.
        residual = sums - mean * counts
        se = float(np.sqrt(len(days) / (len(days) - 1) * np.sum(residual ** 2)) / n)
    equity, peak, drawdown, ruined = 1.0, 1.0, 0.0, False
    for value in completed.sort_values(["exit_time", "entry_time"], kind="stable").net_R:
        factor = 1 + risk_fraction * float(value)
        if factor <= 0:
            ruined = True
        equity = 0.0 if ruined else equity * factor
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    holding = pd.to_numeric(completed.holding_minutes, errors="coerce").to_numpy(float)
    holding = holding[np.isfinite(holding)]
    censored = int(cohort.censored.astype(bool).sum()) if len(cohort) else 0
    invalid = len(cohort) - censored - n
    return {
        "trades": len(cohort), "completed": n, "censored": censored,
        "invalid_uncensored": invalid, "wins": len(positive), "losses": len(negative),
        "flat_trades": int((values == 0).sum()), "win_rate": len(positive) / n if n else None,
        "mean_net_R": mean, "median_net_R": float(np.median(values)) if n else None,
        "sum_net_R": float(values.sum()), "mean_gross_R": mean_gross,
        "profit_factor": float(positive.sum()) / loss_sum if loss_sum > 0 else None,
        "average_win_R": float(positive.mean()) if len(positive) else None,
        "average_loss_R": float(negative.mean()) if len(negative) else None,
        "active_days": int((counts > 0).sum()), "calendar_days": len(days),
        "median_holding_minutes": float(np.median(holding)) if len(holding) else None,
        "ambiguous": int(cohort.ambiguous.astype(bool).sum()) if len(cohort) else 0,
        "break_even_cost_atr": mean_gross * stop_atr if mean_gross is not None else None,
        "cluster_se": se, "selection_score": mean - 1.96 * se if se is not None else None,
        "closed_trade_return": equity - 1, "closed_trade_max_drawdown": drawdown,
        "equity_ruin": ruined, "illustrative_risk_fraction": risk_fraction,
        "drawdown_basis": "closed trades only; fixed fractional risk illustration",
    }


def _centered_one_sided(samples: np.ndarray, estimate: float | None) -> float:
    if estimate is None or estimate <= 0:
        return 1.0
    finite = np.isfinite(samples)
    # Undefined empty-exposure resamples count conservatively against rejection.
    exceed = ~finite | (samples - estimate >= estimate)
    return (int(exceed.sum()) + 1) / (len(samples) + 1)


def paired_inference(trades: pd.DataFrame, baseline: pd.DataFrame, start, end,
                     repeats: int = 9999, seed: int = 20261004) -> dict:
    if not isinstance(repeats, (int, np.integer)) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    start, end, days = _window(start, end)
    model = _completed(_cohort(trades, start, end))
    clock = _completed(_cohort(baseline, start, end))
    model_n, model_sum = _daily(model, days)
    base_n, base_sum = _daily(clock, days)
    weights = np.random.default_rng(seed).multinomial(len(days),
        np.full(len(days), 1 / len(days)), size=repeats)
    bn, bm = weights @ model_n, weights @ model_sum
    cn, cm = weights @ base_n, weights @ base_sum
    mean_samples = np.divide(bm, bn, out=np.full(repeats, np.nan), where=bn > 0)
    base_samples = np.divide(cm, cn, out=np.full(repeats, np.nan), where=cn > 0)
    differences = mean_samples - base_samples
    mean = float(model.net_R.mean()) if len(model) else None
    base_mean = float(clock.net_R.mean()) if len(clock) else None
    difference = mean - base_mean if mean is not None and base_mean is not None else None
    mean_p = _centered_one_sided(mean_samples, mean)
    difference_p = _centered_one_sided(differences, difference)
    return {
        "mean_net_R_ci95": _interval(mean_samples),
        "baseline_mean_net_R": base_mean, "baseline_mean_net_R_ci95": _interval(base_samples),
        "baseline_difference": difference, "baseline_difference_ci95": _interval(differences),
        "mean_p": mean_p, "difference_p": difference_p, "p": max(mean_p, difference_p),
        "bootstrap_repeats": repeats, "bootstrap_seed": seed,
        "bootstrap_mean_valid_replicates": int(np.isfinite(mean_samples).sum()),
        "bootstrap_difference_valid_replicates": int(np.isfinite(differences).sum()),
        "completed": len(model), "baseline_completed": len(clock),
        "active_days": int((model_n > 0).sum()), "baseline_active_days": int((base_n > 0).sum()),
        "calendar_days": len(days), "paired_utc_day_resampling": True,
        "empty_resamples_pvalue_policy": "conservative exceedance; omitted from CI",
        "inference_status": "AVAILABLE" if len(model) and len(clock) and len(days) > 1 else "INSUFFICIENT_EXPOSURE",
    }
