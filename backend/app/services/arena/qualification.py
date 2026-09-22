from __future__ import annotations

from ...config import settings
from .evaluator import clone_candidate, evaluate_candidate


def qualification_gate(metrics: dict, spike_metrics: dict, stage: str = "validation") -> dict:
    checks = {
        "minimum_trades": metrics.get("total_trades", 0) >= settings.arena_minimum_trades,
        "minimum_profit_factor": metrics.get("profit_factor", 0) >= settings.arena_minimum_profit_factor,
        "maximum_drawdown": metrics.get("max_drawdown", float("inf")) <= settings.arena_maximum_drawdown,
        "minimum_expectancy": metrics.get("expectancy", float("-inf")) >= settings.arena_minimum_expectancy,
        "minimum_spike_precision": spike_metrics.get("spike_precision", 0) >= settings.arena_minimum_spike_precision,
        "minimum_spike_recall": spike_metrics.get("spike_recall", 0) >= settings.arena_minimum_spike_recall,
        "maximum_false_alert_rate": spike_metrics.get("false_alert_rate", 1) <= settings.arena_maximum_false_alert_rate,
    }
    if stage == "oos":
        checks["minimum_oos_stability"] = metrics.get("stability_score", 0) >= settings.arena_minimum_oos_stability
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed": failed, "stage": stage}


def quality_score(metrics: dict, spike_metrics: dict) -> float:
    pf = min(max(float(metrics.get("profit_factor", 0)), 0.0) / 2.0, 1.0)
    avg_loss = abs(float(metrics.get("average_loss", 0) or 0))
    expectancy = float(metrics.get("expectancy", 0) or 0)
    expectancy_quality = min(max(expectancy / (avg_loss + 1e-9), 0.0), 1.0) if expectancy > 0 else 0.0
    stability = min(max(float(metrics.get("stability_score", 0)), 0.0), 1.0)
    precision = min(max(float(spike_metrics.get("spike_precision", 0)), 0.0), 1.0)
    recall = min(max(float(spike_metrics.get("spike_recall", 0)), 0.0), 1.0)
    false_quality = 1.0 - min(max(float(spike_metrics.get("false_alert_rate", 1)), 0.0), 1.0)
    return round(100.0 * (0.20 * pf + 0.20 * expectancy_quality + 0.20 * stability + 0.15 * precision + 0.15 * recall + 0.10 * false_quality), 4)


def parameter_sensitivity_test(candles, candidate, profile) -> dict:
    base = evaluate_candidate(candles, candidate, profile)
    numeric = [
        (k, v)
        for k, v in (candidate.parameters or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ][: settings.arena_max_optimized_parameters]

    runs = []
    pct = settings.arena_parameter_sensitivity_pct
    base_pf = float(base["metrics"].get("profit_factor", 0))
    base_expectancy = float(base["metrics"].get("expectancy", 0))
    base_precision = float(base["spike_metrics"].get("spike_precision", 0))

    for key, value in numeric:
        for direction, factor in (("minus", 1.0 - pct), ("plus", 1.0 + pct)):
            params = dict(candidate.parameters or {})
            new_value = float(value) * factor
            if isinstance(value, int):
                new_value = max(1, int(round(new_value)))
            params[key] = new_value
            perturbed = clone_candidate(candidate, params)
            result = evaluate_candidate(candles, perturbed, profile)
            pf = float(result["metrics"].get("profit_factor", 0))
            expectancy = float(result["metrics"].get("expectancy", 0))
            precision = float(result["spike_metrics"].get("spike_precision", 0))

            no_collapse = True
            if base_expectancy > 0 and expectancy < 0:
                no_collapse = False
            if base_pf > 0 and pf < base_pf * 0.60:
                no_collapse = False
            if base_precision > 0 and precision < base_precision * 0.60:
                no_collapse = False

            runs.append(
                {
                    "parameter": key,
                    "direction": direction,
                    "value": new_value,
                    "profit_factor": pf,
                    "expectancy": expectancy,
                    "spike_precision": precision,
                    "passed": no_collapse,
                }
            )

    passed_count = sum(1 for r in runs if r["passed"])
    stability = passed_count / len(runs) if runs else 1.0
    return {
        "passed": stability >= 0.80,
        "stability_score": round(stability, 6),
        "sensitivity_pct": pct,
        "parameters_tested": [k for k, _ in numeric],
        "runs": runs,
        "base_quality_score": quality_score(base["metrics"], base["spike_metrics"]),
    }
