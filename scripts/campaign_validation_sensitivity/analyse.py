"""Descriptive sensitivity of saved validation metrics; no strategy imports.

Only the explicitly allowed validation columns are projected from the mixed
export. Final decisions/reasons, final artifacts, ledgers and frozen selection
are never accessed. A parameter point is not an independent trade sample.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_INPUT_SHA = "9d0ba585e55cc6e9694b956b974e65f6e8da7a7210c402dd1a0de76d40d15242"
EXPECTED_STRATEGIES_SHA = "973d71f1f3ed520c52a8d69c48c0c4e0a92a181135aecc50dcc0778ab5d240b5"
SYMBOLS = tuple(f"{kind}{size}" for kind in ("BOOM", "CRASH") for size in (300, 500, 600, 900, 1000))
FRAMES = ("M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4")
LABELS = {
    "ict_liquidity_mss": "ICT liquidity sweep and structure shift", "crt": "Candle range reclaim",
    "support_resistance": "Support/resistance bounce", "false_break": "False range break",
    "bollinger_squeeze": "Bollinger squeeze breakout", "atr_compression_expansion": "ATR compression and expansion",
    "rsi_momentum": "RSI momentum cross", "pre_spike": "Pre-spike compression hypothesis",
    "post_spike": "Post-spike continuation", "mean_reversion": "Bollinger/RSI mean reversion",
    "trend": "EMA trend resumption", "statistical_volatility": "Volatility-normalized momentum",
    "empirical_conditional_return": "Frozen training-only conditional return states",
}
FAMILIES = tuple(LABELS)
AXES = {"lookback": (12, 24), "threshold": (0.8, 1.2), "stop_atr": (1.5, 2.5),
        "target_atr": (2.0, 3.0), "max_hold_bars": (4, 8)}
INACTIVE_THRESHOLD = {"ict_liquidity_mss", "trend"}
METRICS = ("completed_trades", "profit_factor", "net_profit", "max_drawdown")
ALLOWED_COLUMNS = ("cycle", "symbol", "timeframe", "family", "strategy_id", "attempted_strategy_id", "status",
                   "cache_hit", "settings_json", *METRICS, "artifact", "metrics_partition")
FORBIDDEN_COLUMNS = ("selection_disposition", "reasons", "disposition_partition", "validation_score")
CORE_FIELDS = ("family", "round_index", *AXES, "atr_period", "trailing_stop")


def digest(path):
    with Path(path).open("rb") as stream:
        return __import__("hashlib").file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def spec_for(family, round_index):
    if family not in LABELS or type(round_index) is not int or not 0 <= round_index < 32:
        raise ValueError("Unknown family or noncanonical round")
    core = {"family": family, "round_index": round_index, "atr_period": 14, "trailing_stop": False}
    core.update({axis: values[(round_index >> bit) & 1] for bit, (axis, values) in enumerate(AXES.items())})
    identity = f"{family}.r{round_index:02d}.{sha256(canonical(core).encode()).hexdigest()[:12]}"
    science = {k: v for k, v in core.items() if k != "round_index" and not (family in INACTIVE_THRESHOLD and k == "threshold")}
    return {**core, "id": identity, "scientific_id": sha256(canonical(science).encode()).hexdigest(), "label": LABELS[family]}


def strict_int(value, field, minimum=0):
    if not isinstance(value, str) or str(int(value)) != value or int(value) < minimum:
        raise ValueError(f"Noncanonical integer in {field}")
    return int(value)


def metric_value(value, name):
    if value == "":
        return None
    if name == "completed_trades":
        return strict_int(value, name)
    answer = float(value)
    if not math.isfinite(answer) or name in {"profit_factor", "max_drawdown"} and answer < 0:
        raise ValueError(f"Invalid finite metric {name}")
    return answer


def project_row(row):
    """Deliberately never access final dispositions, reasons or ranking score."""
    return {key: row[key] for key in ALLOWED_COLUMNS}


def parse_row(unprojected):
    row = project_row(unprojected)
    cycle = strict_int(row["cycle"], "cycle", 1)
    if cycle > 32 or row["metrics_partition"] != "validation":
        raise ValueError("Only original validation rounds 1..32 allowed")
    symbol, frame, family = (row[k] for k in ("symbol", "timeframe", "family"))
    if symbol not in SYMBOLS or frame not in FRAMES or family not in FAMILIES:
        raise ValueError("Foreign symbol, frame or family")
    settings = json.loads(row["settings_json"], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    expected_fields = set(CORE_FIELDS) | {"id", "scientific_id", "label"}
    if set(settings) != expected_fields:
        raise ValueError("Unknown or missing settings field")
    actual = spec_for(family, settings["round_index"])
    attempted = spec_for(family, cycle - 1)
    if canonical(settings) != canonical(actual) or row["strategy_id"] != actual["id"] or row["attempted_strategy_id"] != attempted["id"]:
        raise ValueError("Settings/strategy identity differs from actual frozen grid")
    if actual["scientific_id"] != attempted["scientific_id"]:
        raise ValueError("Alias changes an effective scientific parameter")
    if row["cache_hit"] not in {"True", "False"}:
        raise ValueError("Cache hit must be exact boolean spelling")
    cache_hit = row["cache_hit"] == "True"
    if (not cache_hit and actual != attempted) or (cache_hit and actual["round_index"] >= cycle - 1):
        raise ValueError("Inconsistent cache source round")
    if row["status"] not in {"TESTED", "INSUFFICIENT_DATA"}:
        raise ValueError("Only development measurement status allowed")
    metrics = {name: metric_value(row[name], name) for name in METRICS}
    if row["status"] == "INSUFFICIENT_DATA":
        if any(value is not None for value in metrics.values()):
            raise ValueError("Insufficient data has invented numeric metrics")
    elif any(metrics[name] is None for name in ("completed_trades", "net_profit", "max_drawdown")):
        raise ValueError("TESTED metric missing")
    artifact = row["artifact"]
    if artifact and (not artifact.startswith("artifacts/") or "/../" in artifact or not artifact.endswith(".json.gz")):
        raise ValueError("Noncanonical saved artifact reference")
    return {"group": f"{symbol}/{frame}/{family}", "symbol": symbol, "timeframe": frame, "family": family,
            "cycle": cycle, "attempted_round": cycle - 1, "attempted": attempted, "actual": actual,
            "cache_hit": cache_hit, "measurement_status": row["status"], "metrics": metrics, "artifact_reference_only": artifact}


def distribution(values):
    values = list(values)
    finite = [x for x in values if x is not None]
    return {"points": len(values), "finite": len(finite), "null": len(values) - len(finite),
            "min": min(finite) if finite else None, "median": median(finite) if finite else None,
            "max": max(finite) if finite else None,
            "positive": sum(x > 0 for x in finite), "negative": sum(x < 0 for x in finite), "zero": sum(x == 0 for x in finite)}


def sign(value):
    return "NULL" if value is None else "POSITIVE" if value > 0 else "NEGATIVE" if value < 0 else "ZERO"


def analyse_group(group, rows):
    symbol, frame, family = group.split("/")
    by_round, by_science = {}, defaultdict(list)
    for row in rows:
        if row["group"] != group or row["attempted_round"] in by_round:
            raise ValueError("Duplicate or foreign attempted grid point")
        by_round[row["attempted_round"]] = row
        by_science[row["actual"]["scientific_id"]].append(row)
    points, scientific = [], []
    for round_index in range(32):
        spec = spec_for(family, round_index)
        row = by_round.get(round_index)
        points.append({"group": group, "symbol": symbol, "timeframe": frame, "family": family,
                       "attempted_round": round_index, "cycle": round_index + 1, "attempted_strategy_id": spec["id"],
                       "scientific_id": spec["scientific_id"], **{axis: spec[axis] for axis in AXES},
                       "coverage_status": "MISSING" if row is None else "CACHE_ALIAS" if row["cache_hit"] else "MEASURED",
                       "measurement_status": row["measurement_status"] if row else "MISSING",
                       "canonical_strategy_id": row["actual"]["id"] if row else None,
                       "canonical_round": row["actual"]["round_index"] if row else None,
                       "canonical_threshold": row["actual"]["threshold"] if row else None,
                       "metrics_partition": "validation", **(row["metrics"] if row else dict.fromkeys(METRICS)),
                       "artifact_reference_only": row["artifact_reference_only"] if row else None})
    expected_science = {}
    for round_index in range(32):
        spec = spec_for(family, round_index)
        expected_science.setdefault(spec["scientific_id"], spec)
    for science_id, representative in expected_science.items():
        records = by_science.get(science_id, [])
        originals = [row for row in records if not row["cache_hit"]]
        if len(originals) > 1 or records and not originals:
            raise ValueError("Scientific point lacks exactly one measured cache origin")
        original = originals[0] if originals else None
        for record in records:
            if (record["actual"] != original["actual"] or record["metrics"] != original["metrics"]
                    or record["measurement_status"] != original["measurement_status"]
                    or record["artifact_reference_only"] != original["artifact_reference_only"]):
                raise ValueError("Alias conflicts with its measured validation origin")
        scientific.append({"group": group, "symbol": symbol, "timeframe": frame, "family": family,
                           "scientific_id": science_id, "representative_round": representative["round_index"],
                           **{axis: representative[axis] if axis != "threshold" or family not in INACTIVE_THRESHOLD else "INACTIVE" for axis in AXES},
                           "coverage_status": "MISSING" if not original else "MEASURED",
                           "measurement_status": original["measurement_status"] if original else "MISSING",
                           "measured_attempts": len(originals), "alias_attempts": len(records) - len(originals),
                           "attempted_rounds": ";".join(str(row["attempted_round"]) for row in records),
                           "metrics_partition": "validation", **(original["metrics"] if original else dict.fromkeys(METRICS))})
    edges, seen = [], set()
    for bit, (axis, values) in enumerate(AXES.items()):
        for low_round in range(32):
            if (low_round >> bit) & 1:
                continue
            high_round = low_round | 1 << bit
            low, high = points[low_round], points[high_round]
            pair = (axis, low["scientific_id"], high["scientific_id"])
            if low["scientific_id"] == high["scientific_id"]:
                status = "INACTIVE_AXIS_ALIAS"
            elif pair in seen:
                status = "DUPLICATE_EFFECTIVE_EDGE"
            else:
                seen.add(pair)
                status = "MISSING_ENDPOINT" if "MISSING" in {low["measurement_status"], high["measurement_status"]} else "INSUFFICIENT_ENDPOINT" if "INSUFFICIENT_DATA" in {low["measurement_status"], high["measurement_status"]} else "EFFECTIVE_NEIGHBOR"
            edge = {"group": group, "symbol": symbol, "timeframe": frame, "family": family, "axis": axis,
                    "low_value": values[0], "high_value": values[1], "low_round": low_round, "high_round": high_round,
                    "low_scientific_id": low["scientific_id"], "high_scientific_id": high["scientific_id"],
                    "comparison_status": status, "metrics_partition": "validation",
                    "low_net_sign": sign(low["net_profit"]), "high_net_sign": sign(high["net_profit"])}
            edge["strict_net_sign_reversal"] = status == "EFFECTIVE_NEIGHBOR" and {edge["low_net_sign"], edge["high_net_sign"]} == {"POSITIVE", "NEGATIVE"}
            for metric in METRICS:
                a, b = low[metric], high[metric]
                edge[f"{metric}_delta"] = b - a if status == "EFFECTIVE_NEIGHBOR" and a is not None and b is not None else None
            edges.append(edge)
    axes = []
    for axis in AXES:
        relevant = [edge for edge in edges if edge["axis"] == axis]
        effective = [edge for edge in relevant if edge["comparison_status"] == "EFFECTIVE_NEIGHBOR"]
        axes.append({"group": group, "axis": axis, "axis_active": axis != "threshold" or family not in INACTIVE_THRESHOLD,
                     "declared_edges": len(relevant), "comparison_status_counts": dict(Counter(edge["comparison_status"] for edge in relevant)),
                     "strict_net_sign_reversals": sum(edge["strict_net_sign_reversal"] for edge in effective),
                     "metric_high_minus_low": {metric: distribution(edge[f"{metric}_delta"] for edge in effective) for metric in METRICS},
                     "metric_absolute_delta": {metric: distribution(abs(edge[f"{metric}_delta"]) if edge[f"{metric}_delta"] is not None else None for edge in effective) for metric in METRICS}})
    measured = [point for point in scientific if point["coverage_status"] == "MEASURED"]
    summary = {"group": group, "symbol": symbol, "timeframe": frame, "family": family, "declared_dimensions": 5,
               "effective_dimensions": 4 if family in INACTIVE_THRESHOLD else 5, "fixed_parameters": {"atr_period": 14, "trailing_stop": False},
               "declared_points": 32, "observed_attempts": len(rows), "measured_attempts": sum(not row["cache_hit"] for row in rows),
               "cache_alias_attempts": sum(row["cache_hit"] for row in rows), "missing_attempts": 32 - len(rows),
               "expected_scientific_points": len(scientific), "observed_scientific_points": len(measured),
               "missing_scientific_points": len(scientific) - len(measured),
               "insufficient_scientific_points": sum(point["measurement_status"] == "INSUFFICIENT_DATA" for point in measured),
               "metric_distribution_scientific_points": {metric: distribution(point[metric] for point in scientific) for metric in METRICS},
               "edge_status_counts": dict(Counter(edge["comparison_status"] for edge in edges)),
               "axes": axes}
    return summary, points, scientific, edges


def csv_writer(path, fieldnames):
    stream = path.open("x", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    return stream, writer


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output != HERE:
        raise ValueError("Outputs must remain in this operational analysis directory")
    input_sha = digest(source)
    if input_sha != EXPECTED_INPUT_SHA:
        raise ValueError("Source export bytes differ from the pinned original")
    strategy_source = ROOT / "research_campaign/strategies.py"
    if digest(strategy_source) != EXPECTED_STRATEGIES_SHA:
        raise ValueError("Frozen scientific identity/parameter contract changed")
    started = datetime.now(timezone.utc).isoformat()
    groups = {f"{symbol}/{frame}/{family}" for symbol, frame, family in product(SYMBOLS, FRAMES, FAMILIES)}
    grouped = defaultdict(list)
    input_columns = None
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        input_columns = reader.fieldnames
        if not set(ALLOWED_COLUMNS).issubset(input_columns or []):
            raise ValueError("Required explicitly validation-only columns missing")
        for row in reader:
            record = parse_row(row)
            grouped[record["group"]].append(record)
    if set(grouped) - groups:
        raise ValueError("Undeclared group")
    aggregate_points, aggregate_edges, summaries = [], Counter(), []
    handles, writers, output_paths = {}, {}, []
    names = {"points": "declared_grid_points.csv", "scientific": "scientific_points.csv", "edges": "adjacent_comparisons.csv"}
    try:
        for number, group in enumerate(sorted(groups), 1):
            summary, points, scientific, edges = analyse_group(group, grouped.get(group, []))
            summaries.append(summary)
            aggregate_points.extend(scientific)
            aggregate_edges.update(summary["edge_status_counts"])
            for kind, records in [("points", points), ("scientific", scientific), ("edges", edges)]:
                if kind not in writers:
                    path = output / names[kind]
                    handles[kind], writers[kind] = csv_writer(path, list(records[0]))
                    output_paths.append(path)
                writers[kind].writerows(records)
            if number % 250 == 0:
                print(json.dumps({"validation_groups_analyzed": number}), flush=True)
    finally:
        for stream in handles.values():
            stream.close()
    # Histograms summarize parameter points, never pooled trades or aggregate PnL.
    family_distributions = {}
    for family in FAMILIES:
        selected = [point for point in aggregate_points if point["family"] == family]
        family_distributions[family] = {"scientific_points": len(selected),
                                        "metrics": {metric: distribution(point[metric] for point in selected) for metric in METRICS}}
    report = {
        "schema_version": 1, "scope": "DESCRIPTIVE_VALIDATION_PARAMETER_SENSITIVITY_ONLY", "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(), "input": {"path": str(source), "sha256": input_sha, "bytes": source.stat().st_size},
        "frozen_grid_identity_contract": {"source_path": str(strategy_source), "sha256": EXPECTED_STRATEGIES_SHA,
                                           "implementation_imported": False, "independently_reconstructed_ids_match_all_rows": True},
        "columns_used": list(ALLOWED_COLUMNS), "forbidden_columns_not_accessed": list(FORBIDDEN_COLUMNS),
        "axes": AXES, "fixed_parameters": {"atr_period": 14, "trailing_stop": False},
        "declared_shape": [2, 2, 2, 2, 2], "inactive_axis_families": {family: ["threshold"] for family in sorted(INACTIVE_THRESHOLD)},
        "counts": {"groups": len(groups), "symbol_frame_cells": len(SYMBOLS) * len(FRAMES),
                   "observed_attempts": sum(summary["observed_attempts"] for summary in summaries),
                   "new_scientific_measurements": sum(summary["measured_attempts"] for summary in summaries),
                   "cache_alias_attempts": sum(summary["cache_alias_attempts"] for summary in summaries),
                   "declared_points": 32 * len(groups), "scientific_grid_points": len(aggregate_points),
                   "missing_attempts": sum(summary["missing_attempts"] for summary in summaries),
                   "missing_scientific_points": sum(summary["missing_scientific_points"] for summary in summaries),
                   "insufficient_scientific_points": sum(summary["insufficient_scientific_points"] for summary in summaries),
                   "declared_neighbor_edges": sum(aggregate_edges.values()), "neighbor_edge_status_counts": dict(aggregate_edges)},
        "metric_distributions_across_scientific_points": {metric: distribution(point[metric] for point in aggregate_points) for metric in METRICS},
        "family_distributions": family_distributions, "groups": summaries,
        "source_unchanged_after_read": digest(source) == input_sha,
        "strategy_source_unchanged_after_read": digest(strategy_source) == EXPECTED_STRATEGIES_SHA,
        "new_signals_generated": 0, "new_backtests": 0, "final_artifacts_read": 0, "final_selection_read": False,
        "re_ranking_performed": False, "acceptance_criteria_added_or_changed": False, "pooled_trade_statistics": False,
        "safety": {key: False for key in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES", "ready_for_live", "live_allowed", "opened_trades")},
        "limitations": ["Validation was used for development; this analysis is descriptive and selection-dependent, not new OOS evidence.",
                        "Metric min/median/max and signs count parameter points, not independent trades. No pooled PnL or pooled PF is calculated.",
                        "Only the four existing CSV metrics are analyzed. No win rate, execution prices, fills or signal timing is invented.",
                        "Neighbor deltas are finite-grid high-minus-low contrasts with other effective axes fixed, not causal estimates or continuous derivatives.",
                        "Inactive threshold edges and duplicate effective edges are explicitly excluded from contrast summaries.",
                        "PF NULL is preserved even when net PnL is zero or positive; NULL is not zero or infinity.",
                        "No selected/final performance, acceptance, dispositions or reasons are used; no winner or new candidate is proposed.",
                        "All families and binary combinations were predeclared; this report does not prove new strategy invention each cycle.",
                        "PnL is in assumed account units under the original hypothetical cost model; CFD costs and execution remain unverified."]}
    if not report["source_unchanged_after_read"] or not report["strategy_source_unchanged_after_read"]:
        raise ValueError("Read-only pinned inputs changed during analysis")
    report_path = output / "validation_sensitivity.json"
    with report_path.open("xb") as stream:
        stream.write(encoded(report))
    output_paths.append(report_path)
    matrix_rows = []
    for summary in summaries:
        row = {key: value for key, value in summary.items() if key not in {"axes", "fixed_parameters", "metric_distribution_scientific_points", "edge_status_counts"}}
        for metric, dist in summary["metric_distribution_scientific_points"].items():
            row.update({f"{metric}_{key}": value for key, value in dist.items()})
        matrix_rows.append(row)
    matrix_path = output / "group_metric_summary.csv"
    handle, writer = csv_writer(matrix_path, list(matrix_rows[0]))
    with handle:
        writer.writerows(matrix_rows)
    output_paths.append(matrix_path)
    lines = ["# حساسية الإعدادات من validation فقط", "",
             "هذا تحليل وصفي جديد للأرقام المحفوظة؛ لم يولد إشارات أو يشغّل backtest أو يقرأ نتائج final أو قبوله، ولم يعد اختيار السياسات أو ترتيبها.", "",
             f"التغطية: {report['counts']['groups']:,} مجموعة مستقلة؛ {report['counts']['observed_attempts']:,} محاولة، منها {report['counts']['new_scientific_measurements']:,} قياس علمي مختلف و{report['counts']['cache_alias_attempts']:,} aliases. نقاط مفقودة: {report['counts']['missing_attempts']:,}؛ نقص بيانات صريح: {report['counts']['insufficient_scientific_points']:,}.", "",
             "الشبكة الأصلية خمسة أبعاد ثنائية، وليست سطحًا ثنائي الأبعاد: lookback={12,24}، threshold={0.8,1.2}، stop_atr={1.5,2.5}، target_atr={2.0,3.0}، max_hold_bars={4,8}. ATR=14 وtrailing_stop=false ثابتان. لكل مجموعة 32 محاولة؛ لعائلتي ICT/trend أربع محاور فعالة و16 نقطة علمية، وللعائلات الإحدى عشرة الباقية خمسة محاور فعالة و32 نقطة.", "",
             "المقارنات المتجاورة تغير محورًا فعالًا واحدًا من قيمته الدنيا إلى العليا مع تثبيت البقية. استُبعدت حواف threshold الخامل والتكرارات العلمية من ملخصات الفروق. هذه فروق وصفية للشبكة المحدودة؛ لا تقديرات سببية أو حدود قبول جديدة.", "",
             f"حواف الشبكة: {report['counts']['declared_neighbor_edges']:,}؛ التصنيف: {json.dumps(dict(aggregate_edges), ensure_ascii=False)}.", "",
             "إحصاءات المعلمات العلمية المختلفة مجتمعة أدناه تصف توزيع النقاط فقط؛ لا تجمع الصفقات أو العوائد في استراتيجية واحدة:", "",
             "| المقياس | القيم المتاحة | NULL | الأدنى | الوسيط | الأعلى |", "|---|---:|---:|---:|---:|---:|"]
    for metric, dist in report["metric_distributions_across_scientific_points"].items():
        lines.append(f"| {metric} | {dist['finite']:,} | {dist['null']:,} | {dist['min']} | {dist['median']} | {dist['max']} |")
    net = report["metric_distributions_across_scientific_points"]["net_profit"]
    lines.extend(["", f"نقاط صافي الربح المفترض: موجب={net['positive']:,}، سالب={net['negative']:,}، صفر={net['zero']:,}، NULL={net['null']:,}. لا تمثل هذه نسب نجاح للصفقات أو نجاحًا خارج العينة.", "",
                  "مساهمة empirical_conditional_return معروضة مستقلًا ضمن المجموعة والعائلة مثل بقية الفرضيات؛ لا تصبح بذلك عائلة مبتكرة كل دورة أو استراتيجية مؤهلة.", "",
                  "تحتوي declared_grid_points.csv كل نقطة معلنة مع MEASURED/CACHE_ALIAS/MISSING، وscientific_points.csv كل هوية علمية مرة واحدة، وadjacent_comparisons.csv جميع الحواف مع تصنيف الاستبعاد وفروق المقاييس. group_metric_summary.csv وvalidation_sensitivity.json يحفظان min/median/max والتعدادات لكل مجموعة مستقلة.", "",
                  "PnL بوحدات حساب مفترضة، والتكلفة والتنفيذ الفعليان غير موثقين. لا حكم تأهيل أو فائز؛ ولا تحويل تلقائي لحالة أي بند تدقيق إلى VERIFIED.", "",
                  "جميع أعلام التداول الحقيقي ونظائرها الصغيرة false. لم تتغير المدخلات المثبتة أو مصدر الحملة أو حالتها أو اللوحة."])
    arabic_path = output / "REPORT.ar.md"
    with arabic_path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    output_paths.append(arabic_path)
    receipt = {"schema_version": 1, "status": "COMPLETED_DESCRIPTIVE_VALIDATION_ONLY", "created_at": datetime.now(timezone.utc).isoformat(),
               "script_sha256": digest(Path(__file__)), "input": report["input"], "counts": report["counts"],
               "outputs": {path.name: {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for path in output_paths},
               "source_unchanged_after_read": True, "strategy_source_unchanged_after_read": True,
               "new_backtests": 0, "final_artifacts_read": 0, "re_ranking_performed": False, "safety": report["safety"]}
    with (output / "receipt.json").open("xb") as stream:
        stream.write(encoded(receipt))
    print(json.dumps({"status": receipt["status"], "counts": report["counts"], "receipt": str(output / "receipt.json")}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "work/search_phase_labelled_20261012.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.input, args.output_dir)
