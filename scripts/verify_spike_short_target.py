#!/usr/bin/env python3
"""Independent stdlib audit of the declared SHORT1/LONG15 target ablation.

Saved closed44 inputs are audited, not independently regenerated. This module
does not import feature, fitting, replay or research metrics implementations.
Historical execution requires completed results and separate authorization.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_spike_nonlinear as v6
from scripts import verify_spike_native300 as v3
from scripts import verify_spike_tick_tail_signal as v9

FLAGS = v6.FLAGS
SYMBOLS, MODES = ("BOOM600", "CRASH600"), ("SPIKE", "DRIFT")
FAMILIES = ("SHORT44", "LONG44")
NAMES = v9.NAMES44
CONFIG = {"stop_atr": 2., "max_hold_minutes": 1, "entry_delay_minutes": 1,
          "round_trip_cost_atr": .10, "fill_mode": "adverse_extreme"}
TARGET_CONFIGS = {name: {**CONFIG, "max_hold_minutes": hold}
                  for name, hold in (("SHORT44", 1), ("LONG44", 15))}
HISTORY = {"known_prices": True, "known_fifteen_minute_payoffs": True,
           "price_oos": False, "new_label_holdout": False, "prospective": False,
           "adaptive_history": "round6_and_later_results_seen_before_short_target_declaration"}
DEPENDENCIES = {
    "scripts/verify_spike_nonlinear.py": "1a6d25a60093d36d7c8a66e66ef966d144d81d16eb84695f3ceb400aad307aee",
    "scripts/verify_spike_native300.py": "c7835e5b91f6fb376d01bcd25ed7ca3f8d8315c193377724242674d943bf6663",
    "scripts/verify_spike_payoff.py": "ac0b34c978b549711e383e9cec86410e205535d5cf04b9d2ba234ef9b5161cdf",
    "scripts/verify_spike_learned.py": "3846861910f67d4b8e9a50d73962cbd45b70069033a4c93eea0ae159b7ac3f51",
    "scripts/verify_spike_timed.py": "fbecf3b85dfd0b1e4731b2c8d6984c954a100c36d0f447e79059aee187f58658",
    "scripts/verify_spike_multiframe.py": "ccc3703a89a5af2a0094b28e2b0e5eb4e7b5fc0d451198ace7ce3963d89f5b90",
    "scripts/verify_spike_tick_tail_signal.py": "114d6a6e62752aa3ee9378f4775b4b40e1d7c9a2746e6e6becbd3549e86bfc3e",
    "scripts/verify_spike_tick_tail.py": "a3b94835b857e3bcb839bb8100f1c5ca037e56456bf954b777c743b069c54a5a",
    "scripts/verify_spike_tick_execution.py": "f2360d5556fbc43ef3bb140df4e6dfd5d80767f859fb37f38934bd11872b476b",
}
COUNTERS = ("issued", "filled", "completed", "censored", "missing_entry",
            "outside_partition", "purged", "overlap_skipped", "ambiguous")
Audit = v9.Audit
stamp, finite = v9.stamp, v9.finite
digest, read_json = v6.sha256, v6.read_json


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(sep=" ")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected canonical repository-relative POSIX path")
    lexical = PurePosixPath(value)
    if lexical.is_absolute() or ".." in lexical.parts or str(lexical) != value or value == ".":
        raise ValueError("Path aliases, traversal and absolute paths are refused")
    path = ROOT / value
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Artifact symlink escapes the repository")
    return path


def compare(audit, label, actual, expected):
    if isinstance(expected, dict):
        audit.require(label + "/mapping", isinstance(actual, dict))
        for key, value in expected.items():
            audit.require(label + "/field/" + key, key in actual)
            compare(audit, label + "/" + key, actual[key], value)
    elif isinstance(expected, (list, tuple)):
        audit.require(label + "/sequence", isinstance(actual, (list, tuple)))
        audit.require(label + "/length", len(actual) == len(expected))
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare(audit, f"{label}/{i}", a, e)
    else:
        if type(expected) in (bool, int):
            audit.require(label + "/exact_type", type(actual) is type(expected))
        audit.equal(label, actual, expected)


def artifact(value, audit):
    compare(audit, "artifact_descriptor", value, {"local_ignored_artifact": True})
    audit.require("artifact_rows_integer", type(value["rows"]) is int and value["rows"] >= 0)
    path = relative_path(value["path"])
    audit.equal("artifact_bytes/" + value["path"], digest(path), value["sha256"])
    return path


def csv_rows(value, audit, *, indexed=False):
    with artifact(value, audit).open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = (["signal_time"] if indexed else []) + value["columns"]
        audit.require("artifact_exact_csv_schema", reader.fieldnames == expected)
        rows = list(reader)
    audit.equal("artifact_exact_row_count", len(rows), value["rows"])
    return rows


def hash_doubles(values):
    return hashlib.sha256(b"".join(struct.pack("<d", finite(v)) for v in values)).hexdigest()


def hash_times(values):
    return hashlib.sha256(b"".join(struct.pack("<q", t * 1_000_000_000) for t in values)).hexdigest()


def read_inputs(value, minutes, start, end, audit, label):
    rows = csv_rows(value, audit, indexed=True)
    audit.require(label + "/exact44_columns", all(name in value["columns"] for name in NAMES))
    result, previous = {}, None
    for i, raw in enumerate(rows):
        t = stamp(raw["signal_time"]); item = f"{label}/{i}"
        audit.require(item + "/closedUTC00_30", t % 1800 == 0)
        audit.require(item + "/source_bounds", start <= t <= end)
        audit.require(item + "/sorted_unique", previous is None or t > previous)
        row = {name: finite(raw[name]) for name in NAMES}
        row.update(atr=finite(raw["atr"]), close=finite(raw["close"]))
        audit.require(item + "/feature_valid", raw["feature_valid"] == "True")
        audit.equal(item + "/causal_atr", row["atr"], v6.causal_atr(minutes, t))
        audit.equal(item + "/closed_price", row["close"], minutes[t - 60][3])
        for frame, width in (("h4", 14400), ("h1", 3600), ("m15", 900), ("m1", 60)):
            audit.equal(item + "/latest_closed_" + frame, stamp(raw[frame + "_closed_at"]), t // width * width)
            audit.require(item + "/latest_valid_" + frame, raw[frame + "_row_valid"] == "True")
            frame_end = t // width * width
            audit.require(item + "/complete_context_" + frame,
                          all(at in minutes for at in range(frame_end - width, frame_end, 60)))
        result[t] = row; previous = t
    return result


def score(row, estimator):
    return v9.prediction(row, estimator)


def expected_signals(inputs, symbol, mode, model=None, start=None, end=None):
    if symbol not in SYMBOLS or mode not in MODES:
        raise ValueError("Unknown symbol/direction")
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    family = model["family"] if model else "CLOCK"
    expected = []
    for t, row in inputs.items():
        if start is not None and not start <= t < end:
            continue
        prediction = None
        if model:
            if model["status"] != "TESTED":
                continue
            prediction = score(row, model["estimator"])
            if prediction <= 0 or prediction < model["estimator"]["threshold"]:
                continue
        record = {"signal_time": t, "atr": row["atr"], "side": side,
                  "signal_close": row["close"], "variant": f"{family}_{mode}"}
        if prediction is not None:
            record["score"] = prediction
        expected.append(record)
    return expected


def replay(minutes, signals, start, end, config=CONFIG):
    """Scalar M1 reconstruction; all invalid/missing paths retain unknowns."""
    counts = dict.fromkeys(COUNTERS, 0); counts["issued"] = len(signals)
    rows, busy_until = [], start
    previous = None
    for signal in signals:
        t, atr, side = signal["signal_time"], signal["atr"], signal["side"]
        if previous is not None and t <= previous:
            raise ValueError("Signals must be sorted and unique")
        previous = t
        entry = t + 60; planned = entry + config["max_hold_minutes"] * 60
        if not start <= t < end:
            counts["outside_partition"] += 1; continue
        if t + 31 * 60 > end or planned > end:
            counts["purged"] += 1; continue
        if entry < busy_until:
            counts["overlap_skipped"] += 1; continue
        if entry not in minutes:
            counts["missing_entry"] += 1; continue
        row = v3.path_or_censor(minutes, t, atr, side, config)
        row["variant"] = signal["variant"]
        row["missing_time"] = None if row["missing_time"] == "" else row["missing_time"]
        counts["filled"] += 1
        counts["censored" if row["censored"] else "completed"] += 1
        busy_until = row["planned_end"] if row["censored"] else row["exit_time"]
        rows.append(row)
    return rows, counts


def audit_counts(saved, expected, audit, label, config=CONFIG, *, targets=False):
    compare(audit, label, saved, expected)
    compare(audit, label + "/configuration", saved,
            {"config": config, "purge_minutes": 31, "take_profit": None,
             "safety": dict.fromkeys(FLAGS, False),
             "fill_interpretation": "hypothetical_ohlc_proxy_or_extreme_stress_not_measured_execution"})
    if targets:
        compare(audit, label + "/target_caveats", saved,
                {"training_targets_not_strategy_evidence": True,
                 "target_horizon_minutes": config["max_hold_minutes"], "common_purge_minutes": 31})


def audit_ledger(value, expected, audit, label):
    saved = csv_rows(value, audit)
    audit.equal(label + "/ledger_records", len(saved), len(expected))
    for i, (raw, reconstructed) in enumerate(zip(saved, expected, strict=True)):
        item = f"{label}/{i}"
        for key, wanted in reconstructed.items():
            observed = v3.observed_value(raw, key, wanted, item)
            audit.equal(item + "/" + key, observed, wanted)
    return expected


def audit_signals(value, expected, audit, label):
    rows = csv_rows(value, audit)
    audit.equal(label + "/signal_records", len(rows), len(expected))
    for i, (raw, signal) in enumerate(zip(rows, expected, strict=True)):
        compare(audit, f"{label}/{i}",
                {key: stamp(raw[key]) if key == "signal_time" else
                 raw[key] if key == "variant" else int(raw[key]) if key == "side" else finite(raw[key]) for key in signal}, signal)


def audit_metrics(saved, rows, start, end, audit, label):
    computed = v3.mixed_summary(rows, start, end, 2., .0025)
    compare(audit, label, saved, computed)
    return computed


def audit_policy(report, inputs, minutes, symbol, mode, start, end, audit, label, model=None):
    signals = expected_signals(inputs, symbol, mode, model, start, end)
    audit_signals(report["artifacts"]["signals"], signals, audit, label + "/signals")
    rows, counts = replay(minutes, signals, start, end)
    audit_ledger(report["artifacts"]["ledger"], rows, audit, label + "/ledger")
    audit_counts(report["audit"], counts, audit, label + "/accounting")
    metrics = audit_metrics(report["metrics"], rows, start, end, audit, label + "/metrics")
    return rows, counts, metrics


def common_targets(labels):
    known = [{r["signal_time"]: r for r in labels[name] if not r["censored"] and r["net_R"] is not None
              and math.isfinite(r["net_R"])}
             for name in FAMILIES]
    return {t: (known[0][t], known[1][t]) for t in sorted(known[0].keys() & known[1].keys())}


def audit_training(context, inputs, minutes, symbol, mode, start, end, audit, label):
    signals = expected_signals(inputs, symbol, mode)
    labels = {}
    for family in FAMILIES:
        rows, counts = replay(minutes, signals, start, end, TARGET_CONFIGS[family])
        audit.require(label + "/targets_do_not_overlap", counts["overlap_skipped"] == 0)
        audit_ledger(context["target_artifacts"][family], rows, audit, label + "/labels/" + family)
        audit_counts(context["target_audits"][family], counts, audit, label + "/labels/" + family,
                     TARGET_CONFIGS[family], targets=True)
        labels[family] = rows
    common = common_targets(labels)
    n = len(common); times = list(common)
    matrix_value = context["models"]["SHORT44"]["artifacts"]["matrix_and_both_targets"]
    matrix = csv_rows(matrix_value, audit, indexed=True)
    audit.require(label + "/matrix_columns", matrix_value["columns"] == [*NAMES, "short_net_R", "long_net_R", "short_planned_end", "long_planned_end"])
    audit.require(label + "/matrix_common_membership", len(matrix) == n)
    native_x, short_y, long_y, observed_times = [], [], [], []
    for i, (raw, t) in enumerate(zip(matrix, times, strict=True)):
        item = f"{label}/matrix/{i}"; observed_times.append(stamp(raw["signal_time"]))
        audit.equal(item + "/timestamp", observed_times[-1], t)
        for name in NAMES:
            value = finite(raw[name]); native_x.append(value)
            audit.equal(item + "/" + name, value, inputs[t][name])
        for j, target in enumerate(("short", "long")):
            value = finite(raw[target + "_net_R"])
            (short_y if j == 0 else long_y).append(value)
            audit.equal(item + "/" + target + "_target", value, common[t][j]["net_R"])
            audit.equal(item + "/" + target + "_planned_end", stamp(raw[target + "_planned_end"]), common[t][j]["planned_end"])
    identity = {"common_training_rows": n, "feature_names": list(NAMES),
        "byte_hash_schema": {"matrix_and_targets": "C_row_major_little_endian_float64",
                             "issuance_times": "little_endian_int64_UTC_nanoseconds"},
        "matrix_native_sha256": hash_doubles(native_x), "issue_timestamps_native_sha256": hash_times(observed_times),
        "short_target_native_sha256": hash_doubles(short_y), "long_target_native_sha256": hash_doubles(long_y),
        "training_latest_issue": iso(times[-1]) if n else None,
        "training_latest_long_planned_end": iso(max(r[1]["planned_end"] for r in common.values())) if n else None}
    compare(audit, label + "/common_identity", context["identity"], identity)
    for family in FAMILIES:
        model = context["models"][family]
        audit.equal(label + "/enclosing_target_family/" + family, model["family"], family)
        compare(audit, label + "/identity/" + family, model["training_identity"], identity)
        compare(audit, label + "/shared_matrix_artifact/" + family, model["artifacts"]["matrix_and_both_targets"], matrix_value)
        validate_model(model, inputs, common, start, end, audit, label + "/model/" + family)
    audit.equal(label + "/same_training_scaler", context["models"]["SHORT44"]["shared_scaler_sha256"], context["models"]["LONG44"]["shared_scaler_sha256"])
    return sum(len(r) for r in labels.values()), n


def validate_model(model, inputs, common, start, end, audit, label):
    family, n = model["family"], len(common)
    audit.require(label + "/family", family in FAMILIES)
    compare(audit, label, model, {"feature_names": list(NAMES), "training_completed_labels": n,
        "target_horizon_minutes": TARGET_CONFIGS[family]["max_hold_minutes"], "execution_horizon_minutes": 1,
        "target_name": "SHORT1" if family == "SHORT44" else "LONG15",
        "label_purge_minutes": 31, "training_start": iso(start), "training_end": iso(end)})
    if n < 1000:
        compare(audit, label + "/fit_floor", model, {"status": "NOT TESTED", "estimator": None,
                "serialized_estimator_sha256": None, "shared_scaler_sha256": None})
        return
    audit.equal(label + "/fit_status", model["status"], "TESTED")
    estimator = model["estimator"]
    compare(audit, label + "/semantics", estimator, {"version": 1, "feature_names": list(NAMES),
        "fit_rows": n, "penalty": .1, "quantile": .75, "score_kind": "continuous_uncalibrated_score",
        "objective": "mean_squared_error_plus_penalty_times_squared_coefficients",
        "scaler_fitted_on": "training_rows_only", "target_clipped": False, "features_clipped": False})
    audit.equal(label + "/estimator_bytes", model["serialized_estimator_sha256"], canonical_hash(estimator))
    audit.equal(label + "/scaler_bytes", model["shared_scaler_sha256"], canonical_hash({key: estimator[key] for key in ("means", "std")}))
    for vector in ("means", "std", "coefs"):
        audit.require(label + "/dimensions/" + vector, len(estimator[vector]) == 44)
        for value in estimator[vector]:
            finite(value)
    target = [pair[0 if family == "SHORT44" else 1]["net_R"] for pair in common.values()]
    audit.equal(label + "/intercept", estimator["intercept"], statistics.fmean(target))
    for i, name in enumerate(NAMES):
        values = [inputs[t][name] for t in common]
        mean = statistics.fmean(values)
        audit.equal(label + "/training_mean/" + name, estimator["means"][i], mean)
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / n) or 1.
        audit.equal(label + "/training_population_sd/" + name, estimator["std"][i], sd)
    scores = [score(inputs[t], estimator) for t in common]
    audit.equal(label + "/train_q75_positive_floor", estimator["threshold"], max(0., v9.v8.quantile(scores, .75)))
    audit.require(label + "/31minute_purge", all(t + 31 * 60 <= end for t in common))


def known(metrics, counts):
    return metrics["censored"] == metrics["invalid_uncensored"] == counts["missing_entry"] == 0


def eligible(candidate):
    folds, metric = candidate["walk_forward"], candidate["validation"]
    return (candidate["family"] == "SHORT44" and
        [(f["training"], f["test"]) for f in folds] == [("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")] and
        metric["completed"] >= 500 and metric["active_days"] >= 30 and
        metric["censored"] == metric["invalid_uncensored"] == 0 and
        metric["selection_score"] is not None and metric["selection_score"] > 0 and
        all(f["metrics"]["completed"] >= 100 and known(f["metrics"], f["audit"]) and
            (f["metrics"]["mean_net_R"] or -math.inf) > 0 and (f["metrics"]["profit_factor"] or 0) > 1 for f in folds))


def audit_inference(test, model, reference, start, end, audit, label):
    day, week = test["day"], test["weekly"]
    m, r = model["metrics"], reference["metrics"]
    difference = m["mean_net_R"] - r["mean_net_R"] if m["mean_net_R"] is not None and r["mean_net_R"] is not None else None
    compare(audit, label + "/point_comparison", day, {"baseline_mean_net_R": r["mean_net_R"],
        "baseline_difference": difference, "completed": m["completed"], "baseline_completed": r["completed"],
        "active_days": m["active_days"], "baseline_active_days": r["active_days"], "calendar_days": m["calendar_days"],
        "bootstrap_repeats": 9999, "bootstrap_seed": 20261005, "paired_utc_day_resampling": True})
    compare(audit, label + "/weekly_config", week,
        {"weekly_block_days": 7, "weekly_bootstrap_repeats": 9999, "weekly_bootstrap_seed": 20261005})
    for container, names in ((day, ("bootstrap_mean_valid_replicates", "bootstrap_difference_valid_replicates")),
                             (week, ("weekly_valid_mean_replicates", "weekly_valid_difference_replicates"))):
        for key in names:
            audit.require(label + "/exact_draw_count/" + key, type(container[key]) is int and 0 <= container[key] <= 9999)
    for key in ("mean_p", "difference_p", "p"):
        audit.require(label + "/p_range/" + key, 0 <= finite(day[key]) <= 1)
    audit.equal(label + "/day_absolute_increment_conjunction", day["p"], max(day["mean_p"], day["difference_p"]))
    audit.require(label + "/weekly_p_range", 0 <= finite(week["weekly_p"]) <= 1)
    available = (known(m, model["audit"]) and known(r, reference["audit"]) and
        day["bootstrap_mean_valid_replicates"] == day["bootstrap_difference_valid_replicates"] == 9999 and
        week["weekly_valid_mean_replicates"] == week["weekly_valid_difference_replicates"] == 9999 and day["calendar_days"] >= 7)
    compare(audit, label + "/reference_unknown_policy", test,
        {"policy_inference_available": available, "p": max(day["p"], week["weekly_p"]) if available else 1.,
         "day_mean_undefined_replicates": 9999-day["bootstrap_mean_valid_replicates"],
         "day_difference_undefined_replicates": 9999-day["bootstrap_difference_valid_replicates"],
         "weekly_mean_undefined_replicates": 9999-week["weekly_valid_mean_replicates"],
         "weekly_difference_undefined_replicates": 9999-week["weekly_valid_difference_replicates"],
         "policy_unknowns_forbid_promotion": not (known(m, model["audit"]) and known(r, reference["audit"]))})


def lower(interval, floor=0):
    return interval[0] is not None and finite(interval[0]) > floor


def audit_gate(row, audit, label):
    if row["family"] != "SHORT44":
        compare(audit, label + "/fixed_reference", row, {"historical_candidate": False,
            "supports_expected_pf_1_5": False, "reference_only": True,
            "rejection_reasons": ["fixed_diagnostic_reference_only"]})
        return
    m = row["metrics"]
    checks = [(m["completed"] >= 1000 and (m["profit_factor"] or 0) >= 1.5, "user_pf1_5_n1000_not_met"),
        (m["active_days"] >= 60, "under60_active_days"), (row["development_eligible"], "development_rejected"),
        (known(m, row["audit"]), "missing_or_censored_outcomes"),
        (m["day_undefined_replicates"] == m["weekly_undefined_replicates"] == 0, "undefined_pf_draws"),
        (lower(m["day_profit_factor_ci95"], 1) and lower(m["weekly_profit_factor_ci95"], 1), "pf_ci_not_above1"),
        (m.get("holm_p", 1) < .05, "four_conjunction_holm_not_significant"),
        (all(third["metrics"]["completed"] >= 200 and (third["metrics"]["mean_net_R"] or -math.inf) > 0 for third in row["thirds"]), "chronological_thirds_unstable"),
        (not m["equity_ruin"] and m["closed_trade_max_drawdown"] <= .1, "drawdown_over10pct_or_ruin"),
        ((row["doubled_cost"]["mean_net_R"] or -math.inf) > 0, "doubled_cost_not_positive")]
    # The runner's frozen name for this explicit user research requirement.
    checks.insert(1, ((m["mean_net_R"] if m["mean_net_R"] is not None else -math.inf) >= .10, "mean_net_R_under0_10"))
    for reference in ("LONG44", "CLOCK"):
        test = row["comparisons"][reference]
        checks.extend([(test["policy_inference_available"], f"unknown_or_undefined_{reference}_inference"),
            (lower(test["day"]["mean_net_R_ci95"]) and lower(test["weekly"]["weekly_mean_net_R_ci95"]), "mean_ci_not_positive"),
            (lower(test["day"]["baseline_difference_ci95"]) and lower(test["weekly"]["weekly_difference_ci95"]), f"no_day_week_advantage_over_{reference}")])
    reasons = list(dict.fromkeys(name for passed, name in checks if not passed))
    reasons += ["adaptive_known_history_requires_new_evidence", "broker_costs_and_fills_not_tested"]
    compare(audit, label + "/economic_gate", row, {"user_target_observed": checks[0][0],
        "historical_candidate": False, "conditional_economic_gates_pass": all(passed for passed, _ in checks),
        "conditional_lower_pf_at_least1_5": (m["day_undefined_replicates"] == m["weekly_undefined_replicates"] == 0 and
             known(m, row["audit"]) and all(interval[0] is not None and finite(interval[0]) >= 1.5 for interval in
             (m["day_profit_factor_ci95"], m["weekly_profit_factor_ci95"]))),
        "supports_expected_pf_1_5": False, "reference_only": False, "rejection_reasons": reasons,
        "actual_money_profit": "NOT TESTED", "prospective_validation": "NOT TESTED"})


def metadata(study, audit):
    declaration, selection, result = [read_json(study / name) for name in ("declaration.json", "selection.json", "results.json")]
    for name, value in (("declaration", declaration), ("selection", selection), ("results", result)):
        audit.safety(value, name)
        compare(audit, name + "/safety", value, {"safety": dict.fromkeys(FLAGS, False), "adaptive_round": 11,
                "goal_achieved": False, "live_candidate": False, "history": HISTORY})
        audit.equal(name + "/sha_sidecar", digest(study / f"{name}.json"), (study / f"{name}.sha256").read_text().strip())
    compare(audit, "declaration_stage", declaration, {"stage": "short_target_premeasurement_declaration",
        "prices_decoded": False, "new_features_labels_models_or_ledgers_computed": False,
        "feature_names": list(NAMES), "symbols": list(SYMBOLS), "modes": list(MODES), "families": list(FAMILIES),
        "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000, "active_heldout_days": 60}})
    compare(audit, "selection_stage", selection, {"stage": "frozen_short_target_development", "declaration_sha256": digest(study / "declaration.json")})
    compare(audit, "selection_copied_premeasurement_snapshot", selection,
        {"prices_decoded": declaration["prices_decoded"],
         "new_features_labels_models_or_ledgers_computed": declaration["new_features_labels_models_or_ledgers_computed"]})
    compare(audit, "result_stage", result, {"stage": "short_target_known_history_evaluation",
        "declaration_sha256": digest(study / "declaration.json"), "selection_sha256": digest(study / "selection.json"),
        "primary_holm_family": 4, "historical_strategy_candidate": False, "actual_money_profit": "NOT TESTED"})
    compare(audit, "declared_configuration", declaration["config"], {"execution": CONFIG, "target_configs": TARGET_CONFIGS,
        "target_names": {"SHORT44": "SHORT1", "LONG44": "LONG15"},
        "purge_minutes": 31, "clock_minutes": [0, 30], "ridge_penalty": .1, "training_quantile": .75,
        "minimum_common_training_labels": 1000, "bootstrap_repeats": 9999, "bootstrap_seed": 20261005,
        "primary_holm_family": 4, "only_short_is_candidate": True, "unknown_outcomes_are_zero": False,
        "undefined_draws_block_bounded_inference": True})
    for value, name in ((selection, "selection"), (result, "results")):
        for key in ("config", "history", "lineage", "science_code_hashes"):
            compare(audit, name + "/frozen/" + key, value[key], declaration[key])
    for kind, hashes in (("fixed_independent_helpers", DEPENDENCIES), ("frozen_science", declaration["science_code_hashes"]),
                         ("frozen_lineage", declaration["lineage"]["files"])):
        for name, wanted in hashes.items():
            audit.equal("captured_source_hash/" + kind + "/" + name, digest(relative_path(name)), wanted)
    old_audit = read_json(relative_path("docs/spike_nonlinear_20261005/independent_audit.json"))
    compare(audit, "ancestor_pass", old_audit, {"status": "PASS", "pass": True, "error_count": 0, "errors": []})
    chronology = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in
                  (declaration["run_utc"], selection["frozen_utc"], result["run_utc"])]
    audit.require("ordered_declare_develop_evaluate_utc", all(t.tzinfo is not None and
                  t.utcoffset().total_seconds() == 0 for t in chronology) and chronology[0] <= chronology[1] <= chronology[2])
    return declaration, selection, result


def audit_thirds(row, paths, start, end, audit, label):
    thirds = row["thirds"]
    audit.require(label + "/three_fixed_thirds", len(thirds) == 3)
    bounds = [start + int((end-start) * i / 3) for i in range(4)]
    for i, third in enumerate(thirds):
        # Legacy helper uses Timestamp duration arithmetic; these180/54day
        # intervals divide exactly into whole UTC minutes and seconds.
        compare(audit, f"{label}/{i}/bounds", third, {"start": iso(bounds[i]), "end": iso(bounds[i+1])})
        audit_metrics(third["metrics"], [r for r in paths if bounds[i] <= r["signal_time"] and r["signal_time"] + 31*60 <= bounds[i+1]],
                      bounds[i], bounds[i+1], audit, f"{label}/{i}")


def verify(study, audit):
    declaration, selection, result = metadata(study, audit)
    summaries, sources = [], []
    label_records = common_records = policy_records = development_clock_records = 0
    hypothesis_rows = []
    for symbol in SYMBOLS:
        minutes = {}
        for kind in ("old", "fresh"):
            source = declaration["lineage"]["sources"][symbol][kind]
            minutes[kind], source_report = v6.audit_source(symbol, kind, source, audit)
            sources.append(source_report)
        src = declaration["lineage"]["sources"][symbol]["old"]
        start, end = src["first_epoch"], src["last_epoch"] + 60
        split = {"development": (start, start+(end-start)*7//10), "final_test": (start+(end-start)*7//10, end)}
        for i, fraction in enumerate((4, 5, 6), 1):
            split[f"train{fraction*10}"] = (start, start+(end-start)*fraction//10)
            split[f"wf{i}"] = (start+(end-start)*fraction//10, start+(end-start)*(fraction+1)//10)
        saved = selection["symbols"][symbol]
        audit.require(symbol + "/fixed_candidate_set", set(saved["models"]) == {f"{f}_{m}" for f in FAMILIES for m in MODES})
        audit.require(symbol + "/fixed_cohorts", set(result["symbols"][symbol]["cohorts"]) == {"old_final30_secondary", "later180_known_history_primary"})
        for name, pair in split.items():
            audit.equal(symbol + "/partition/" + name, [stamp(t) for t in saved["partitions"][name]], list(pair))
        context_inputs = {}
        input_descriptors = {}
        for mode in MODES:
            contexts = saved["models"][f"SHORT44_{mode}"]["training_contexts"]
            audit.require(symbol + "/fixed_training_contexts/" + mode, set(contexts) == {"train40", "train50", "train60", "development"})
            for training, context in contexts.items():
                label = f"{symbol}/{mode}/{training}"; first, cutoff = split[training]
                if training not in context_inputs:
                    context_inputs[training] = read_inputs(context["input_artifact"], minutes["old"], first, cutoff, audit, label + "/inputs")
                    input_descriptors[training] = context["input_artifact"]
                else:
                    compare(audit, label + "/same_input_artifact_across_directions", context["input_artifact"], input_descriptors[training])
                compare(audit, label + "/prefix", context["prefix_audit"],
                    {"prefix_end": iso(cutoff), "last_completed_m1_close": iso(cutoff),
                     "source_prefix_rows": (cutoff-start)//60, "feature_count": 44,
                     "all_features_computed_after_prefix_clip": True,
                     "eligible_clock_rows": len(context_inputs[training])})
                nlabels, ncommon = audit_training(context, context_inputs[training], minutes["old"], symbol, mode, first, cutoff, audit, label)
                label_records += nlabels; common_records += ncommon
            for family in FAMILIES:
                candidate = saved["models"][f"{family}_{mode}"]
                compare(audit, symbol + "/enclosing_candidate/" + family + mode, candidate,
                        {"family": family, "mode": mode, "config": CONFIG})
                audit.require(symbol + "/exact_three_folds/" + family + mode,
                    [(f["training"], f["test"]) for f in candidate["walk_forward"]] == [("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")])
                combined = []
                for fold in candidate["walk_forward"]:
                    training, test = fold["training"], fold["test"]
                    context = saved["models"][f"SHORT44_{mode}"]["training_contexts"][training]
                    compare(audit, symbol + "/frozen_fold_fit/" + family + training, fold["model"], context["models"][family])
                    inputs = {t:r for t,r in context_inputs["development"].items() if split[test][0] <= t < split[test][1]}
                    paths, counts, points = audit_policy(fold, inputs, minutes["old"], symbol, mode, *split[test], audit, f"{symbol}/{family}/{mode}/{test}", fold["model"])
                    clock, clock_counts = replay(minutes["old"], expected_signals(inputs, symbol, mode), *split[test])
                    development_clock_records += len(clock)
                    audit_signals(fold["baseline_artifacts"]["signals"], expected_signals(inputs, symbol, mode), audit, symbol + "/development_clock_signals")
                    audit_ledger(fold["baseline_artifacts"]["ledger"], clock, audit, symbol + "/development_clock_ledger")
                    audit_counts(fold["baseline_audit"], clock_counts, audit, symbol + "/development_clock")
                    audit_metrics(fold["baseline"], clock, *split[test], audit, symbol + "/development_clock")
                    combined.extend(paths); policy_records += len(paths)
                audit_ledger(candidate["validation_artifact"], combined, audit, f"{symbol}/{family}/{mode}/pooled_validation")
                audit_metrics(candidate["validation"], combined, split["wf1"][0], split["wf3"][1], audit, f"{symbol}/{family}/{mode}/pooled_metrics")
                audit.equal(symbol + "/development_eligibility/" + family + mode, candidate["development_eligible"], eligible(candidate))
                audit.equal(symbol + "/reference_only/" + family + mode, candidate["reference_only"], family == "LONG44")
                compare(audit, symbol + "/final_fit/" + family + mode, candidate["final_model"], saved["models"][f"SHORT44_{mode}"]["training_contexts"]["development"]["models"][family])
        pool = [(key, row) for key, row in saved["models"].items() if row["family"] == "SHORT44" and eligible(row)]
        chosen = sorted(pool, key=lambda pair: (-pair[1]["validation"]["selection_score"], pair[0]))[0][0] if pool else None
        audit.equal(symbol + "/development_only_SHORT_selection", saved["selected_model"], chosen)
        audit.equal(symbol + "/result_copied_selection", result["symbols"][symbol]["selected_model"], chosen)
        for cohort_name, kind in (("old_final30_secondary", "old"), ("later180_known_history_primary", "fresh")):
            cohort = result["symbols"][symbol]["cohorts"][cohort_name]
            audit.require(symbol + "/fixed_cohort_models/" + cohort_name,
                set(cohort["models"]) == {f"{f}_{m}" for f in (*FAMILIES, "CLOCK") for m in MODES})
            cohort_start, cohort_end = [stamp(cohort[key]) for key in ("start", "end")]
            wanted = split["final_test"] if kind == "old" else (min(minutes[kind]), max(minutes[kind])+60)
            audit.equal(symbol + "/cohort_bounds/" + cohort_name, (cohort_start, cohort_end), wanted)
            inputs = read_inputs(cohort["input_artifact"], minutes[kind], cohort_start, cohort_end, audit, symbol + "/" + cohort_name + "/inputs")
            audit.equal(symbol + "/common_clock_rows/" + cohort_name, cohort["common_clock_rows"], len(inputs))
            for mode in MODES:
                for family in (*FAMILIES, "CLOCK"):
                    row = cohort["models"][f"{family}_{mode}"]
                    fit = saved["models"][f"{family}_{mode}"]["final_model"] if family != "CLOCK" else None
                    label = f"{symbol}/{cohort_name}/{family}/{mode}"
                    compare(audit, label + "/policy", row, {"family": family, "mode": mode, "config": CONFIG,
                        "actual_money_profit": "NOT TESTED", "prospective_validation": "NOT TESTED",
                        "selected_model": family == "SHORT44" and f"{family}_{mode}" == chosen,
                        "model_sha256": canonical_hash(fit) if fit else None,
                        "development_eligible": saved["models"][f"{family}_{mode}"]["development_eligible"] if fit else False})
                    paths, counts, points = audit_policy(row, inputs, minutes[kind], symbol, mode, cohort_start, cohort_end, audit, label, fit)
                    policy_records += len(paths)
                    audit_thirds(row, paths, cohort_start, cohort_end, audit, label + "/thirds")
                    doubled = [{**r, "net_R": r["gross_R"]-.1 if r["gross_R"] is not None else None} for r in paths]
                    audit_metrics(row["doubled_cost"], doubled, cohort_start, cohort_end, audit, label + "/double_cost")
                    summaries.append({"symbol": symbol, "cohort": cohort_name, "family": family, "mode": mode,
                        "path_records": len(paths), "completed": points["completed"], "profit_factor": points["profit_factor"],
                        "mean_net_R": points["mean_net_R"]})
                short = cohort["models"][f"SHORT44_{mode}"]
                for reference in ("LONG44", "CLOCK"):
                    audit_inference(short["comparisons"][reference], short, cohort["models"][f"{reference}_{mode}"], cohort_start, cohort_end, audit, f"{symbol}/{cohort_name}/{mode}/vs{reference}")
                audit.equal(symbol + "/SHORT_conjunction/" + cohort_name + mode, short["metrics"]["p"], max(test["p"] for test in short["comparisons"].values()))
                if kind == "fresh":
                    hypothesis_rows.append(short)
                else:
                    for family in (*FAMILIES, "CLOCK"):
                        compare(audit, symbol + "/secondary_only/" + family + mode, cohort["models"][f"{family}_{mode}"],
                            {"historical_candidate": False, "supports_expected_pf_1_5": False,
                             "rejection_reasons": ["secondary_known_older_history"], "reference_only": family != "SHORT44"})
    audit.require("exact_four_SHORT_primary_hypotheses", len(hypothesis_rows) == 4)
    adjusted = v6.holm_values([r["metrics"]["p"] for r in hypothesis_rows])
    for i, (row, corrected) in enumerate(zip(hypothesis_rows, adjusted, strict=True)):
        audit.equal(f"joint_holm/{i}", row["metrics"]["holm_p"], corrected)
    for symbol in SYMBOLS:
        for key, row in result["symbols"][symbol]["cohorts"]["later180_known_history_primary"]["models"].items():
            audit_gate(row, audit, f"{symbol}/{key}/primary_gate")
    return {"sources": sources, "ledgers": summaries, "strategy_path_records": policy_records,
            "development_repeated_clock_reference_path_records": development_clock_records,
            "saved_strategy_and_repeated_reference_path_records": policy_records + development_clock_records,
            "record_counts_are_not_pooled_strategy_sample_evidence": True,
            "training_target_label_records": label_records, "common_training_matrix_records": common_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_short_target_20261006")
    args = parser.parse_args(); study = args.study.resolve()
    if not study.is_relative_to(ROOT.resolve()) or not (study / "results.json").is_file():
        parser.error("Completed in-repository results.json required")
    output = study / "independent_audit.json"
    if output.exists():
        parser.error("Preserve prior audit before any explicitly authorized retry")
    audit, details = Audit(), {}
    try:
        v6.safety_inputs(audit)
        if audit.failures:
            raise ValueError("Requires all four runtime safety flags=false")
        details = verify(study, audit)
    except Exception as exc:
        audit.failures.append({"check": "audit_exception", "error": f"{type(exc).__name__}: {exc}"})
    report = {"status": "FAIL" if audit.failures else "PASS", "pass": not audit.failures,
        "run_utc": datetime.now(timezone.utc).isoformat(), "safety": dict.fromkeys(FLAGS, False),
        "script_sha256": digest(Path(__file__)), "helper_sha256": {name: digest(relative_path(name)) for name in DEPENDENCIES},
        "declaration_sha256": digest(study / "declaration.json"), "selection_sha256": digest(study / "selection.json"),
        "results_sha256": digest(study / "results.json"), "checks": audit.checks,
        "saved_false_safety_values": audit.safety_values_checked,
        "implementation": "Independent standard-library M1 paths/scalers/scalar predictions and arithmetic; no research engines imported",
        "selection_premeasurement_flags_interpretation": "Copied declaration snapshot; NOT a current-state claim that development decoded no prices or computed no features/labels/models/ledgers",
        "scope": ["pinned source normalization/page lineage, frozen science/declaration/selection/results",
            "saved closed44 inputs: exact time/ATR/context availability and matrix bytes",
            "SHORT1/LONG15 label paths, shared complete-target membership and native timestamp/target/matrix hashes",
            "saved train-only scalers/intercepts, scalar prediction/q75 and all saved issuance",
            "all saved model/CLOCK short-policy ledgers and development/secondary/primary point metrics",
            "chronology/purge/unknowns/development selection/onlySHORT gates and Holm-four saved-p arithmetic"],
        "not_verified": ["all44 feature values or complete feature-valid opportunity universe independently regenerated",
            "ridge coefficients independently refitted", "bootstrap draws, CIs or p-values independently regenerated",
            "trade/day tail diagnostics and their export copies",
            "inferential validity, conditional-return process or statistical independence",
            "tick execution, executable bid/ask, measured costs, actual cash profit or prospective performance"],
        **details, "error_count": len(audit.failures), "errors": audit.failures,
        "max_numeric_differences": audit.max_numeric_error}
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "pass", "checks", "error_count")}))
    raise SystemExit(bool(audit.failures))


if __name__ == "__main__":
    main()
