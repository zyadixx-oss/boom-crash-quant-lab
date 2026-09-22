from itertools import product
from copy import deepcopy

from ..config import settings
from .backtest import BacktestEngine


def grid_optimize(validation_df, profile, search_space=None):
    """Small validation-only search. Deliberately rejects large brute-force spaces."""
    space = search_space or {
        "min_signal_score": [profile.min_signal_score - 5, profile.min_signal_score, profile.min_signal_score + 5],
        "spike_atr_multiple": [profile.spike_atr_multiple * 0.9, profile.spike_atr_multiple, profile.spike_atr_multiple * 1.1],
    }
    keys = list(space)
    if len(keys) > settings.arena_max_optimized_parameters:
        raise ValueError(
            f"Anti-overfitting guard: at most {settings.arena_max_optimized_parameters} parameters may be optimized per experiment"
        )
    combinations = 1
    for key in keys:
        if not space[key]:
            raise ValueError(f"Empty optimization values for {key}")
        combinations *= len(space[key])
    if combinations > 64:
        raise ValueError("Anti-overfitting guard: optimization grid is too large (>64 combinations)")

    best = None
    results = []
    for vals in product(*[space[k] for k in keys]):
        candidate_profile = deepcopy(profile)
        for key, value in zip(keys, vals):
            if not hasattr(candidate_profile, key):
                raise ValueError(f"Unknown profile parameter: {key}")
            setattr(candidate_profile, key, value)

        metrics = BacktestEngine(candidate_profile).run(validation_df)["metrics"]
        score = (
            metrics.get("expectancy", 0)
            + 0.15 * min(metrics.get("profit_factor", 0), 3.0)
            + 0.25 * metrics.get("stability_score", 0)
            - 0.10 * metrics.get("max_drawdown", 0)
        )
        row = {"params": dict(zip(keys, vals)), "score": float(score), "metrics": metrics}
        results.append(row)
        if best is None or score > best["score"]:
            best = row

    return {"best": best, "results": sorted(results, key=lambda item: item["score"], reverse=True)}
