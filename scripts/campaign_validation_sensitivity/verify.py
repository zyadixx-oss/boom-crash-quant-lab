"""Verify the saved sensitivity matrices independently of analyse.py.

No import of the producer, campaign engine or final result files. Validation
columns alone supply all expected values; file hashes preserve the evidence.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
METRICS = ("completed_trades", "profit_factor", "net_profit", "max_drawdown")
AXES = ("lookback", "threshold", "stop_atr", "target_atr", "max_hold_bars")
VALUES = ((12, 24), (.8, 1.2), (1.5, 2.5), (2., 3.), (4, 8))
FLAGS = {"LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES", "ready_for_live", "live_allowed", "opened_trades"}


def digest(path):
    with Path(path).open("rb") as stream:
        return __import__("hashlib").file_digest(stream, "sha256").hexdigest()


def number(value, integer=False):
    if value == "":
        return None
    result = int(value) if integer else float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite output")
    return result


def stats(values):
    finite = sorted(x for x in values if x is not None)
    n = len(finite)
    mid = None if not n else finite[n // 2] if n % 2 else (finite[n // 2 - 1] + finite[n // 2]) / 2
    return {"points": len(values), "finite": n, "null": len(values) - n,
            "min": finite[0] if n else None, "median": mid, "max": finite[-1] if n else None,
            "positive": len([x for x in finite if x > 0]), "negative": len([x for x in finite if x < 0]),
            "zero": len([x for x in finite if x == 0])}


def science_id(settings):
    value = {k: v for k, v in settings.items() if k not in {"round_index", "id", "scientific_id", "label"}}
    if value["family"] in {"trend", "ict_liquidity_mss"}:
        value.pop("threshold")
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def edge_expected(axis, low, high, seen):
    if low["science_id"] == high["science_id"]:
        return "INACTIVE_AXIS_ALIAS"
    key = (axis, low["science_id"], high["science_id"])
    if key in seen:
        return "DUPLICATE_EFFECTIVE_EDGE"
    seen.add(key)
    return "EFFECTIVE_NEIGHBOR"


def sign(value):
    return "NULL" if value is None else "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "ZERO"


def run():
    checks, errors, pins = Counter(), [], {}

    def check(ok, name, item=None):
        checks[name] += 1
        if not ok:
            errors.append({"check": name, "item": item})

    def pin(path, expected=None):
        actual = digest(path)
        pins[str(path)] = actual
        if expected is not None:
            check(actual == expected, "file_hash", str(path))
        return actual

    receipt_path = HERE / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    pin(receipt_path)
    pin(HERE / "analyse.py", receipt["script_sha256"])
    for value in receipt["outputs"].values():
        pin(Path(value["path"]), value["sha256"])
        check(Path(value["path"]).stat().st_size == value["bytes"], "output_bytes", value["path"])
    source_path = Path(receipt["input"]["path"])
    pin(source_path, receipt["input"]["sha256"])
    report = json.loads((HERE / "validation_sensitivity.json").read_text())
    check(receipt["status"] == "COMPLETED_DESCRIPTIVE_VALIDATION_ONLY", "descriptive_receipt_scope")
    check(set(report["safety"]) == FLAGS and all(x is False for x in report["safety"].values()), "strict_false_safety")
    check(report["new_backtests"] == report["new_signals_generated"] == report["final_artifacts_read"] == 0 and report["final_selection_read"] is False and report["re_ranking_performed"] is False and report["pooled_trade_statistics"] is False, "no_research_or_final_use")
    source, originals, groups = {}, {}, defaultdict(list)
    with source_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            # Do not access any final disposition/reasons/ranking columns.
            group = "/".join(row[key] for key in ("symbol", "timeframe", "family"))
            round_index = int(row["cycle"]) - 1
            key = (group, round_index)
            settings = json.loads(row["settings_json"])
            science = science_id(settings)
            metrics = {name: number(row[name], name == "completed_trades") for name in METRICS}
            entry = {"group": group, "round": round_index, "science_id": science, "metrics": metrics,
                     "cache": row["cache_hit"] == "True", "settings": settings,
                     "artifact": row["artifact"], "strategy_id": row["strategy_id"]}
            check(key not in source and row["metrics_partition"] == "validation", "source_point_unique_validation", str(key))
            check(science == settings["scientific_id"], "source_independent_scientific_id", str(key))
            source[key] = entry
            if not entry["cache"]:
                scientific_key = (group, science)
                check(scientific_key not in originals, "new_scientific_measurement_once", str(key))
                originals[scientific_key] = entry
                groups[group].append(entry)
    check(len(source) == 41600 and len(originals) == 38400 and len(groups) == 1300, "source_expected_population")
    for entry in source.values():
        original = originals[entry["group"], entry["science_id"]]
        check(entry["metrics"] == original["metrics"] and entry["artifact"] == original["artifact"] and entry["settings"] == original["settings"], "source_alias_consistency", str((entry["group"], entry["round"])))
    matrix_points = set()
    with (HERE / "declared_grid_points.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["group"], int(row["attempted_round"])
            expected = source[key]
            check(key not in matrix_points, "declared_matrix_unique_point", str(key))
            matrix_points.add(key)
            check(row["scientific_id"] == expected["science_id"] and row["canonical_strategy_id"] == expected["strategy_id"] and row["metrics_partition"] == "validation", "declared_matrix_identity", str(key))
            check(row["coverage_status"] == ("CACHE_ALIAS" if expected["cache"] else "MEASURED") and row["measurement_status"] == "TESTED", "declared_matrix_coverage", str(key))
            for bit, (axis, values) in enumerate(zip(AXES, VALUES)):
                check(float(row[axis]) == values[(key[1] >> bit) & 1], "each_actual_declared_grid_coordinate", str((key, axis)))
            for metric in METRICS:
                check(number(row[metric], metric == "completed_trades") == expected["metrics"][metric], "declared_matrix_validation_metric", str((key, metric)))
    check(matrix_points == set(source), "all_declared_points_exactly_once")
    scientific_points = set()
    with (HERE / "scientific_points.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["group"], row["scientific_id"]
            check(key in originals and key not in scientific_points, "scientific_matrix_exact_unique", str(key))
            scientific_points.add(key)
            check(int(row["measured_attempts"]) == 1 and int(row["alias_attempts"]) == (1 if row["family"] in {"trend", "ict_liquidity_mss"} else 0), "scientific_matrix_alias_count", str(key))
            for metric in METRICS:
                check(number(row[metric], metric == "completed_trades") == originals[key]["metrics"][metric], "scientific_matrix_metric", str((key, metric)))
    check(scientific_points == set(originals), "all_scientific_points_exactly_once")
    all_edge_keys, effective_seen, edge_counts = set(), defaultdict(set), Counter()
    deltas = defaultdict(lambda: {metric: [] for metric in METRICS})
    axis_counts, flips = defaultdict(Counter), Counter()
    with (HERE / "adjacent_comparisons.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            group, axis = row["group"], row["axis"]
            bit = AXES.index(axis)
            low_round, high_round = int(row["low_round"]), int(row["high_round"])
            key = group, axis, low_round
            low, high = source[group, low_round], source[group, high_round]
            check(key not in all_edge_keys and not ((low_round >> bit) & 1) and high_round == low_round | 1 << bit, "one_axis_adjacent_identity", str(key))
            all_edge_keys.add(key)
            expected_status = edge_expected(axis, low, high, effective_seen[group])
            check(row["comparison_status"] == expected_status, "adjacent_status_independent", str(key))
            check(row["low_scientific_id"] == low["science_id"] and row["high_scientific_id"] == high["science_id"], "adjacent_scientific_identity", str(key))
            edge_counts[expected_status] += 1
            axis_counts[group, axis][expected_status] += 1
            pair_signs = sign(low["metrics"]["net_profit"]), sign(high["metrics"]["net_profit"])
            reversal = expected_status == "EFFECTIVE_NEIGHBOR" and set(pair_signs) == {"POSITIVE", "NEGATIVE"}
            check(row["strict_net_sign_reversal"] == str(reversal) and (row["low_net_sign"], row["high_net_sign"]) == pair_signs, "strict_sign_reversal", str(key))
            flips[group, axis] += reversal
            for metric in METRICS:
                a, b = low["metrics"][metric], high["metrics"][metric]
                expected_delta = b - a if expected_status == "EFFECTIVE_NEIGHBOR" and a is not None and b is not None else None
                check(number(row[f"{metric}_delta"]) == expected_delta, "neighbor_delta_from_source", str((key, metric)))
                if expected_status == "EFFECTIVE_NEIGHBOR":
                    deltas[group, axis][metric].append(expected_delta)
    check(len(all_edge_keys) == 104000 and edge_counts == Counter({"EFFECTIVE_NEIGHBOR": 94400, "DUPLICATE_EFFECTIVE_EDGE": 6400, "INACTIVE_AXIS_ALIAS": 3200}), "all_edges_exact_population")
    summaries = {value["group"]: value for value in report["groups"]}
    check(set(summaries) == set(groups), "all_group_summaries")
    for group, entries in groups.items():
        for metric in METRICS:
            expected = stats([entry["metrics"][metric] for entry in entries])
            check(summaries[group]["metric_distribution_scientific_points"][metric] == expected, "group_distribution_from_originals", str((group, metric)))
        for axis_summary in summaries[group]["axes"]:
            axis = axis_summary["axis"]
            check(axis_summary["comparison_status_counts"] == dict(axis_counts[group, axis]) and axis_summary["strict_net_sign_reversals"] == flips[group, axis], "axis_counts_and_flips", str((group, axis)))
            for metric in METRICS:
                values = deltas[group, axis][metric]
                expected = stats(values)
                absolute = stats([abs(value) if value is not None else None for value in values])
                check(axis_summary["metric_high_minus_low"][metric] == expected and axis_summary["metric_absolute_delta"][metric] == absolute, "axis_signed_absolute_distribution", str((group, axis, metric)))
    for metric in METRICS:
        expected = stats([entry["metrics"][metric] for entry in originals.values()])
        check(report["metric_distributions_across_scientific_points"][metric] == expected, "global_parameter_point_distribution", metric)
    for family, summary in report["family_distributions"].items():
        entries = [entry for entry in originals.values() if entry["settings"]["family"] == family]
        for metric in METRICS:
            check(summary["metrics"][metric] == stats([entry["metrics"][metric] for entry in entries]), "family_parameter_distribution", f"{family}/{metric}")
    check(digest(source_path) == receipt["input"]["sha256"], "source_unchanged_after_verification")
    axis_distributions = {}
    for axis in AXES:
        net = [value for (group, name), values in deltas.items() if name == axis for value in values["net_profit"]]
        axis_distributions[axis] = {"net_delta": stats(net), "absolute_net_delta": stats([abs(x) if x is not None else None for x in net]),
                                    "strict_net_sign_reversals": sum(value for (group, name), value in flips.items() if name == axis)}
    result = {"schema_version": 1, "status": "PASS" if not errors else "FAIL", "created_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent validation-source-to-matrix verification; no producer imports, final reads, ranking, trades pooling or backtests",
              "checks": dict(checks), "check_count": sum(checks.values()), "errors": errors, "file_pins": pins,
              "script_sha256": digest(Path(__file__)), "producer_receipt_sha256": digest(receipt_path),
              "counts": {"attempts": len(source), "scientific_points": len(originals), "groups": len(groups), "edges": len(all_edge_keys), "edge_status_counts": dict(edge_counts)},
              "descriptive_axis_distributions": axis_distributions, "new_backtests": 0, "final_artifacts_read": 0,
              "safety": {key: False for key in sorted(FLAGS)}}
    with (HERE / "independent_verification.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "errors": errors[:10], "counts": result["counts"]}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
