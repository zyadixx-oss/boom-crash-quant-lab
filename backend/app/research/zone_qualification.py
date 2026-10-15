"""Premeasurement evidence gates for each single-symbol zone hypothesis."""
from __future__ import annotations

import math


def _above(value, threshold):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > threshold


def _ci_above(interval, threshold):
    return isinstance(interval, (list, tuple)) and len(interval) == 2 and _above(interval[0], threshold)


def historical_zone_gate(*, metrics: dict, pf: dict, weekly: dict,
                         folds: list[dict], validation: dict,
                         thirds: list[dict], doubled_cost: dict,
                         unknown_regions: int, baseline_unknown_regions: int,
                         holm_p: float) -> dict:
    """No pooling and no promotion from a historical point estimate alone.

    ``metrics`` includes day-paired inference; ``folds`` are the three fixed
    chronological validation ledgers' summaries. This gate can only describe
    historical criteria: these reused prices are never fresh OOS evidence.
    """
    reasons = []
    development_ok = (
        len(folds) == 3 and validation.get("completed", 0) >= 500
        and validation.get("active_days", 0) >= 30
        and validation.get("censored") == 0 and validation.get("invalid_uncensored") == 0
        and validation.get("unknown_regions") == 0 and validation.get("control_unknown_regions") == 0
        and _above(validation.get("selection_score"), 0)
        and all(f.get("completed", 0) >= 100 and f.get("censored") == 0
                and f.get("invalid_uncensored") == 0 and _above(f.get("mean_net_R"), 0)
                and f.get("unknown_regions") == 0 and f.get("control_unknown_regions") == 0
                and _above(f.get("profit_factor"), 1) for f in folds)
    )
    checks = (
        (development_ok, "development_rejected_or_insufficient"),
        (metrics.get("completed", 0) >= 1000, "fewer_than_1000_completed"),
        (metrics.get("active_days", 0) >= 60, "fewer_than_60_active_days"),
        (_above(metrics.get("profit_factor"), 0) and metrics["profit_factor"] >= 1.5, "PF_below_1.5_or_unknown"),
        (metrics.get("censored") == 0 and metrics.get("invalid_uncensored") == 0, "incomplete_payoff"),
        (unknown_regions == 0 and baseline_unknown_regions == 0, "unknown_region_outcomes"),
        (_ci_above(pf.get("day_profit_factor_ci95"), 1), "day_PF_CI_not_above_1"),
        (_ci_above(pf.get("weekly_profit_factor_ci95"), 1), "week_PF_CI_not_above_1"),
        (_ci_above(metrics.get("mean_net_R_ci95"), 0), "day_mean_CI_not_positive"),
        (_ci_above(weekly.get("weekly_mean_net_R_ci95"), 0), "week_mean_CI_not_positive"),
        (_ci_above(metrics.get("baseline_difference_ci95"), 0), "day_control_advantage_not_positive"),
        (_ci_above(weekly.get("weekly_difference_ci95"), 0), "week_control_advantage_not_positive"),
        (_above(0.05 - holm_p, 0), "Holm_not_significant"),
        (len(thirds) == 3 and all(t.get("completed", 0) >= 200 and _above(t.get("mean_net_R"), 0) for t in thirds), "chronological_thirds_not_stable"),
        (metrics.get("closed_trade_max_drawdown") is not None and metrics["closed_trade_max_drawdown"] <= .10
         and metrics.get("equity_ruin") is False, "drawdown_or_ruin"),
        (_above(doubled_cost.get("mean_net_R"), 0), "doubled_cost_not_positive"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    lower_supports_target = (pf.get("day_profit_factor_ci95", [None])[0] is not None
                            and pf.get("weekly_profit_factor_ci95", [None])[0] is not None
                            and pf["day_profit_factor_ci95"][0] >= 1.5
                            and pf["weekly_profit_factor_ci95"][0] >= 1.5)
    return {"historical_criteria_passed": not reasons, "development_eligible": development_ok,
            "historical_rejection_reasons": reasons, "PF_1_5_lower_CI_supported": lower_supports_target,
            "qualified": False, "fresh_out_of_sample": False,
            "promotion_status": "NOT_QUALIFIED_REUSED_HISTORY",
            "economic_execution": "NOT TESTED"}


def holm_adjust(pvalues: list[float]) -> list[float]:
    if any(not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("p values must be finite probabilities")
    result, running = [1.0] * len(pvalues), 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[i]))
        result[i] = running
    return result
