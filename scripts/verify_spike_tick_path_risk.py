#!/usr/bin/env python3
"""Independent stdlib audit of the adaptive44/45 prior-path-risk pilot.

No research/ML/metrics-engine imports or mutations of inherited verifiers.
Historical execution is separate from synthetic/source review authorization.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_spike_tick_tail_signal as v9

v7, v8 = v9.v7, v9.v8
Audit, FLAGS, SYMBOLS, DATES = v9.Audit, v9.FLAGS, v9.SYMBOLS, v9.DATES
MODES, COUNTERS, NAMES44 = v9.MODES, v9.COUNTERS, v9.NAMES44
PATH_FEATURE = "native_adverse_semivariance600"
NAMES45 = (*NAMES44, PATH_FEATURE)
FAMILIES = ("RIDGE44", "RIDGE45")
CONFIG = {"exit": v7.CONFIG, "purge_minutes": 31, "cadence_minutes": 5,
          "ridge_penalty": .1, "training_quantile": .75, "minimum_training_labels": 1000,
          "bootstrap_repeats": 9999, "bootstrap_seed": 20261005,
          "risk_increments": 600, "required_prior_quotes": 601, "scale_refitted": False,
          "native_direction_independent_of_strategy_mode": True, "training_labels_may_overlap": True,
          "unknown_payoffs_are_zero": False, "undefined_draws_block_bounded_inference": True}
HISTORY = {"known_prices": True, "known_payoffs": True, "price_oos": False, "new_label_holdout": False,
           "prospective": False, "adaptive_history": "round9_final_payoffs_already_exposed_before_round10"}
DEPENDENCIES = {**v9.DEPENDENCIES,
    "scripts/verify_spike_tick_tail_signal.py": "114d6a6e62752aa3ee9378f4775b4b40e1d7c9a2746e6e6becbd3549e86bfc3e"}
digest, read_json, source_path = v9.digest, v9.read_json, v9.source_path
stamp, finite, iso, date_of = v9.stamp, v9.finite, v9.iso, v9.date_of
canonical_hash, compare_points = v9.canonical_hash, v9.compare_points
replay_partition, planned_eligible, observed_days = v9.replay_partition, v9.planned_eligible, v9.observed_days
prediction, matrix_hash, point_metrics = v9.prediction, v9.matrix_hash, v9.point_metrics
inference_policy, paired_policy, audit_accounting = v9.inference_policy, v9.paired_policy, v9.audit_accounting


def audited_path(value):
    """Inherited audit pins may be absolute, but must remain inside this repo."""
    path = Path(value)
    if path.is_absolute():
        path = path.resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Inherited audit path escaped repository")
        return path
    return source_path(value)


def audit_inherited_inputs(files, old_input_pins, audit, label):
    """Translate original absolute audit keys to new portable lineage keys."""
    anchor = "docs/spike_tick_age_payoff_20261005/declaration.json"
    matching = [name for name in old_input_pins if name == anchor or name.endswith("/" + anchor)]
    prefix = None
    if matching:
        audit.require(label + "/unique_declaration_anchor", len(matching) == 1)
        key = matching[0]
        if key != anchor:
            prefix = key[:-(len(anchor) + 1)]
            audit.require(label + "/canonical_legacy_repository",
                          PurePosixPath(prefix).is_absolute() and str(PurePosixPath(prefix)) == prefix
                          and ".." not in PurePosixPath(prefix).parts)
    seen = set()
    for name, sha in old_input_pins.items():
        if PurePosixPath(name).is_absolute() and prefix is not None:
            audit.require(label + "/same_legacy_repository", name.startswith(prefix + "/"))
            relative = name[len(prefix) + 1:]
            audit.require(label + "/canonical_relative_pin", str(PurePosixPath(relative)) == relative
                          and ".." not in PurePosixPath(relative).parts)
            source_path(relative)
        else:
            relative = audited_path(name).relative_to(ROOT.resolve()).as_posix()
        audit.require(label + "/unique_relative_anchor", relative not in seen)
        seen.add(relative)
        audit.equal(label + "/" + relative, files.get(relative), sha)


def prior_adverse_window(times, quotes, scale, side, window=600, *, clock_only=False):
    """Each output t uses601 contiguous quotes ending at t-1; no quote t needed."""
    if type(window) is not int or window != 600 or type(clock_only) is not bool:
        raise ValueError("Exactly600 prior increments and explicit clock flag required")
    if type(side) is not int or side not in (-1, 1):
        raise ValueError("Native symbol direction must be integer +/-1")
    scale = v8.positive(scale)
    if len(times) != len(quotes) or not times or any(type(t) is not int for t in times):
        raise ValueError("Nonempty matching integer timestamps and quotes required")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Strictly sorted unique seconds required")
    values, squared, previous_t, previous_q = {}, deque(), None, None
    for t, raw in zip(times, quotes, strict=True):
        q = v8.positive(raw)
        if previous_t is None or t - previous_t != 1:
            squared.clear()
        else:
            adverse = min(side * v8.log_increment(previous_q, q), 0.) / scale
            term = adverse * adverse
            if not math.isfinite(term):
                raise ValueError("Native-adverse square must be finite")
            squared.append(term)
            if len(squared) > window:
                squared.popleft()
        decision = t + 1
        if not clock_only or decision % 300 == 0:
            try:
                values[decision] = math.fsum(squared) / window if len(squared) == window else None
            except OverflowError as exc:
                raise ValueError("Native-adverse rolling arithmetic is not finite") from exc
        previous_t, previous_q = t, q
    return values


def read_signals(path):
    records = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["signal_time", "atr", "side", "variant", "signal_close", PATH_FEATURE, "score"]:
            raise ValueError("Exact44/45 M5 issuance schema required")
        for raw in reader:
            row = {"signal_time": stamp(raw["signal_time"]), "atr": finite(raw["atr"]), "side": int(raw["side"]),
                   "variant": raw["variant"], "signal_close": finite(raw["signal_close"]), PATH_FEATURE: finite(raw[PATH_FEATURE])}
            if raw["score"] != "":
                row["score"] = finite(raw["score"])
            if row["signal_time"] % 300 or row["atr"] <= 0 or row["side"] not in (-1, 1) or row["signal_close"] <= 0 or row[PATH_FEATURE] < 0:
                raise ValueError("Invalid fixed UTC M5 path-risk issuance")
            if records and row["signal_time"] <= records[-1]["signal_time"]:
                raise ValueError("Sorted unique issuance required")
            records.append(row)
    return records


def issuance_hash(records):
    return canonical_hash([{**r, "signal_time": iso(r["signal_time"])} for r in records])


def artifact(value, audit):
    if value.get("local_ignored_artifact") is not True:
        raise ValueError("Saved local artifact required")
    path = source_path(value["path"])
    audit.pin(path, value["sha256"])
    return path


def pin_artifacts(value, audit):
    if isinstance(value, dict):
        if value.get("local_ignored_artifact") is True:
            artifact(value, audit)
        for child in value.values():
            pin_artifacts(child, audit)
    elif isinstance(value, list):
        for child in value:
            pin_artifacts(child, audit)


def read_inputs(value, end, minute, risk, audit, label, fingerprints=None):
    path = artifact(value["inputs"], audit)
    count, previous, eligible, frame_cache = 0, None, {}, {}
    start = v7.date_bounds(DATES[0])[0]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"m5_open_time", "signal_time", *NAMES45, "atr", "close", "original_feature_valid",
                    "original44_all_finite", "multiframe_feature_valid", "tick_path_feature_valid", "common_available", "feature_valid"}
        required |= {name + suffix for name in ("h4", "h1", "m15", "m1") for suffix in ("_closed_at", "_row_valid")}
        audit.require(label + "/exact45_schema_no_age_or_mark", set(reader.fieldnames) == required)
        for raw in reader:
            t = stamp(raw["signal_time"])
            audit.require(label + "/original_M5_grid", t == stamp(raw["m5_open_time"]) + 300 and t % 300 == 0
                          and t == (start if previous is None else previous + 300) and t < end)
            previous, count = t, count + 1
            if fingerprints is not None:
                fingerprint = canonical_hash(raw)
                if t in fingerprints:
                    audit.equal(label + "/full_grid_prefix_consistency", fingerprint, fingerprints[t])
                else:
                    fingerprints[t] = fingerprint
            numbers = {n: finite(raw[n], True) for n in NAMES45}
            flags = {n: v7.boolean(raw[n]) for n in ("original_feature_valid", "original44_all_finite",
                     "multiframe_feature_valid", "tick_path_feature_valid", "common_available", "feature_valid")}
            original_finite = all(numbers[n] is not None for n in NAMES44)
            frame_valid = flags["original_feature_valid"] and original_finite
            expected_risk = risk.get(t)
            path_valid = expected_risk is not None
            common = frame_valid and path_valid
            compare_points(audit, label + "/common44_45_mask", flags,
                           {"original44_all_finite": original_finite, "multiframe_feature_valid": frame_valid,
                            "tick_path_feature_valid": path_valid, "common_available": common, "feature_valid": common})
            audit.equal(label + "/strict_prior_native_adverse600", numbers[PATH_FEATURE], expected_risk)
            context_valid = True
            for name, seconds in (("h4", 14400), ("h1", 3600), ("m15", 900), ("m1", 60)):
                closed = stamp(raw[name + "_closed_at"]) if raw[name + "_closed_at"] not in ("", "NaT") else None
                last = t // seconds * seconds
                audit.equal(label + "/closed_context/" + name, closed, last)
                if (name, last) not in frame_cache:
                    frame_cache[name, last] = all(s in minute for s in range(last - seconds, last, 60))
                valid = frame_cache[name, last]
                audit.equal(label + "/unfilled_context/" + name, v7.boolean(raw[name + "_row_valid"]), valid)
                context_valid &= valid
            if flags["original_feature_valid"]:
                audit.require(label + "/valid_context_required", context_valid)
            if common:
                atr, close = finite(raw["atr"]), finite(raw["close"])
                audit.equal(label + "/causal_M5_ATR", atr, v7.causal_atr(minute, t))
                audit.equal(label + "/closed_M5_price", close, minute[t - 60][3])
                eligible[t] = {**numbers, "atr": atr, "close": close}
    compare_points(audit, label + "/prefix_policy", value,
                   {"feature_end_exclusive": iso(end), "suffix_calculations": False, "scale_refit": False,
                    "age_or_mark_availability_used": False, "clock": "every closed UTC M5",
                    "eligible_common_clock_rows": len(eligible), "canonical_boundary_day_quotes_decoded": True})
    audit.equal(label + "/saved_rows", value["inputs"]["rows"], count)
    audit.equal(label + "/complete_grid", count, len(range(start, end, 300)))
    inventory = v9.m1_prefix_inventory(minute, end)
    compare_points(audit, label + "/closed_grid_vs_available", value,
                   {"m1_closed_rows_used": inventory["closed_grid_slots"],
                    "m1_available_closed_rows_used": inventory["available_closed_candles"],
                    "m1_unknown_closed_rows_retained": inventory["unavailable_closed_slots"],
                    "last_m1_close": iso(inventory["last_grid_close"])})
    return eligible


def expected_signals(inputs, symbol, mode, family="CLOCK", model=None, start=None, end=None):
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    issued = []
    for t, row in sorted(inputs.items()):
        if start is not None and not start <= t < end:
            continue
        score = None
        if family != "CLOCK":
            if model["status"] == "NOT TESTED":
                continue
            score = prediction(row, model["estimator"])
            if score <= 0 or score < model["estimator"]["threshold"]:
                continue
        record = {"signal_time": t, "atr": row["atr"], "side": side, "variant": family + "_" + mode,
                  "signal_close": row["close"], PATH_FEATURE: row[PATH_FEATURE]}
        if score is not None:
            record["score"] = score
        issued.append(record)
    return issued


def validate_model(model, inputs, labels, start, end, audit, label):
    audit.require(label + "/exact_family", model["family"] in FAMILIES)
    names = NAMES44 if model["family"] == "RIDGE44" else NAMES45
    compare_points(audit, label, model, {"feature_names": list(names), "training_start": iso(start), "training_end": iso(end),
                   "training_labels_may_overlap": True, "statistically_independent_labels": False,
                   "counts_toward_profit_sample_target": False, "label_purge_minutes": 31})
    planned = {r["signal_time"] for r in planned_eligible(labels, start, end)}
    audit.require(label + "/original31minute_partition_purge", all(r["signal_time"] in planned for r in labels))
    matrix, target, known = matrix_hash(names, inputs, labels)
    n = len(known)
    compare_points(audit, label + "/exact_rows_targets_matrix", model,
                   {"matrix_and_target_sha256": matrix, "common_timestamp_target_sha256": target, "training_completed_labels": n,
                    "training_censored_labels": sum(r["censored"] for r in labels),
                    "training_invalid_labels": sum(not r["censored"] and r["net_R"] is None for r in labels),
                    "training_latest_issue": iso(known[-1]["signal_time"]) if n else None})
    audit_training_matrix(model["training_matrix"], inputs, known, audit, label)
    if n < 1000:
        compare_points(audit, label + "/training_floor", model, {"status": "NOT TESTED", "estimator": None, "serialized_estimator_sha256": None})
        return
    audit.equal(label + "/fit_status", model["status"], "TESTED")
    estimator = model["estimator"]
    compare_points(audit, label + "/saved_estimator", estimator,
                   {"version": 1, "feature_names": list(names), "fit_rows": n, "penalty": .1, "quantile": .75,
                    "score_kind": "continuous_uncalibrated_score", "scaler_fitted_on": "training_rows_only",
                    "features_clipped": False, "target_clipped": False})
    audit.equal(label + "/serialized_bytes", model["serialized_estimator_sha256"], canonical_hash(estimator))
    audit.equal(label + "/target_intercept", estimator["intercept"], statistics.fmean(r["net_R"] for r in known))
    for i, name in enumerate(names):
        values = [inputs[r["signal_time"]][name] for r in known]
        audit.equal(label + "/training_mean/" + name, estimator["means"][i], statistics.fmean(values))
        saved_mean = finite(estimator["means"][i])
        scale = math.sqrt(math.fsum((v - saved_mean) ** 2 for v in values) / n) or 1.
        audit.equal(label + "/training_std/" + name, estimator["std"][i], scale)
    scores = [prediction(inputs[r["signal_time"]], estimator) for r in known]
    audit.equal(label + "/training_q75_positive_threshold", estimator["threshold"], max(0., v8.quantile(scores, .75)))


def audit_training_matrix(value, inputs, known, audit, label):
    with artifact(value, audit).open(newline="") as stream:
        reader = csv.DictReader(stream)
        audit.require(label + "/45_training_matrix_schema", reader.fieldnames == ["signal_time", *NAMES45, "net_R"])
        rows = list(reader)
    audit.require(label + "/matrix_row_count", len(rows) == len(known) == value["rows"])
    for raw, target in zip(rows, known, strict=True):
        t = stamp(raw["signal_time"])
        audit.equal(label + "/matrix_timestamp", t, target["signal_time"])
        audit.equal(label + "/matrix_target", finite(raw["net_R"]), target["net_R"])
        for name in NAMES45:
            audit.equal(label + "/matrix_input/" + name, finite(raw[name]), inputs[t][name])


def audit_signals(value, expected, audit, label):
    saved = read_signals(artifact(value, audit))
    audit.equal(label + "/rows", value["rows"], len(saved))
    compare_points(audit, label + "/issuance", saved, expected)
    audit.equal(label + "/canonical_issuance_hash", value["canonical_issuance_sha256"], issuance_hash(saved))
    return saved


def audit_paths(value, expected, audit, label):
    saved = v7.read_ledger(artifact(value, audit))
    audit.equal(label + "/ledger_rows", value["rows"], len(saved))
    compare_points(audit, label + "/scalar_one_second_paths", saved, expected)
    return saved


def audit_metrics(saved, rows, counts, days, audit, label, *, overlapping=False):
    points = point_metrics(rows, counts, days)
    if overlapping:
        for field in ("profit_factor", "sum_gains_R", "sum_losses_R"):
            points.pop(field)
        points["day_sums_return_count"] = [r[:2] for r in points.pop("day_sums_return_count_gain_loss")]
        points["profit_factor_evidence"] = "NOT TESTED_overlapping_training_targets"
    compare_points(audit, label, saved, points)
    sums = v9.day_sums(rows, days)
    mean_defined = inference_policy(saved["mean_inference"], sums, 1, audit, label + "/mean_inference")
    audit.equal(label + "/mean_interval_identity", saved["day_mean_ci95"], saved["mean_inference"]["ci95"])
    score = points["mean_net_R"] - 1.96 * points["day_ratio_cluster_se"] if points["mean_net_R"] is not None and points["day_ratio_cluster_se"] is not None and mean_defined else None
    audit.equal(label + "/selection_score_arithmetic", saved["selection_score"], score)
    if not overlapping:
        pf_defined = inference_policy(saved["pf_inference"], sums, 3, audit, label + "/pf_inference")
        audit.equal(label + "/bounded_inference_policy", saved["bounded_inference_allowed"], mean_defined and pf_defined)


def audit_report(report, expected, ticks, start, end, audit, label):
    issued = audit_signals(report["signals"], expected, audit, label)
    audit.equal(label + "/issued_hash", report["issuance_sha256"], issuance_hash(issued))
    rows, counts = replay_partition(ticks, issued, start, end)
    saved = audit_paths(report["ledger"], rows, audit, label)
    audit_accounting(report["audit"], counts, audit, label + "/counts", False)
    audit_metrics(report["metrics"], saved, counts, observed_days(start, end), audit, label + "/metrics")
    compare_points(audit, label + "/quote_proxy_policy", report, {"strategy_returns": True, "quoteproxy_not_actual_fills": True})
    return saved, counts


def audit_labels(model, inputs, ticks, symbol, mode, start, end, audit, label):
    issued = audit_signals(model["training_signals"],
              expected_signals(inputs, symbol, mode, start=start, end=end), audit, label + "/training")
    audit.equal(label + "/training_issuance_hash", model["training_clock_issuance_sha256"], issuance_hash(issued))
    rows, counts = replay_partition(ticks, issued, start, end, overlapping=True)
    saved = audit_paths(model["training_labels"], rows, audit, label + "/labels")
    audit_accounting(model["training_label_audit"], counts, audit, label + "/counts", True)
    validate_model(model, inputs, saved, start, end, audit, label + "/model")
    return saved


def audit_comparisons(saved, ledgers, reports, days, audit, label):
    keys = ("RIDGE45_minus_RIDGE44", "RIDGE44_minus_CLOCK", "RIDGE45_minus_CLOCK")
    audit.require(label + "/exact_comparisons", set(saved) == set(keys))
    for key in keys:
        left, right = key.split("_minus_")
        tested = all(reports[family]["status"] == "TESTED" for family in (left, right))
        audit.equal(label + "/" + key + "/fit_status", saved[key]["status"], "TESTED" if tested else "NOT TESTED")
        audit.equal(label + "/" + key + "/partial_wrapper", "partial_descriptive_only" in saved[key], not tested)
        paired_policy(saved[key], ledgers[left], ledgers[right], days, audit, label + "/" + key)


def audit_opportunities(value, inputs, ticks, symbol, mode, start, end, audit, label):
    issued = audit_signals(value["signals"], expected_signals(inputs, symbol, mode, start=start, end=end), audit, label)
    rows, counts = replay_partition(ticks, issued, start, end, overlapping=True)
    saved = audit_paths(value["labels"], rows, audit, label)
    audit_accounting(value["audit"], counts, audit, label + "/counts", True)
    audit_metrics(value["metrics"], saved, counts, observed_days(start, end), audit, label + "/metrics", overlapping=True)
    planned = planned_eligible(issued, start, end)
    missing = len({s["signal_time"] for s in planned} - {r["signal_time"] for r in saved})
    compare_points(audit, label + "/training_target_policy", value,
                   {"planned_eligible_signals": len(planned), "missing_entry_labels": missing,
                    "known_subset_only": bool(missing or counts["censored"] or value["metrics"]["invalid_uncensored"]),
                    "labels_may_overlap": True, "strategy_pf_evidence": False, "discovery_claim": False})
    return len(saved)


def audit_expansion(value, reports, comparisons, audit, label):
    risk = reports["RIDGE45"]
    metrics = risk["metrics"]
    mean_ci = metrics["day_mean_ci95"]
    increments = [comparisons[key] for key in ("RIDGE45_minus_RIDGE44", "RIDGE45_minus_CLOCK")]
    positive = (risk["status"] == "TESTED" and metrics["mean_inference"]["all_draws_defined"] and mean_ci is not None and mean_ci[0] > 0
                and all(reports[family]["metrics"][key] == 0 for family in ("CLOCK", *FAMILIES)
                        for key in ("censored", "missing_entry", "invalid_uncensored"))
                and all(row["status"] == "TESTED" and row["all_draws_defined"] and row["ci95"] is not None and row["ci95"][0] > 0 for row in increments))
    compare_points(audit, label, value, {"positive_absolute_and_incremental_day_bounds": bool(positive),
                   "expansion_authorized": False, "historical_gate": False, "live_candidate": False,
                   "scope": "known_history_economic_prerequisite_only_not_stability_or_profit_discovery"})


def metadata(study, audit):
    declaration, selected, result = (read_json(study / n) for n in ("declaration.json", "selection.json", "results.json"))
    for name, document in (("declaration", declaration), ("selection", selected), ("results", result)):
        audit.pin(study / (name + ".json"), (study / (name + ".sha256")).read_text().strip())
        audit.safety(document, name)
        compare_points(audit, name, document, {"config": CONFIG, "history": HISTORY, "safety": dict.fromkeys(FLAGS, False),
                       "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False})
    compare_points(audit, "declaration", declaration, {"stage": "tick_path_risk_premeasurement_declaration",
                   "dates": list(DATES), "symbols": list(SYMBOLS), "families": list(FAMILIES), "modes": list(MODES),
                   "feature_names45": list(NAMES45), "quotes_parsed": False, "new_risk_features_models_or_ledgers_computed": False})
    compare_points(audit, "selection", selected, {"stage": "frozen_tick_path_risk_development",
                   "declaration_sha256": digest(study / "declaration.json"), "final45_features_scores_or_ledgers_computed": False})
    compare_points(audit, "results", result, {"stage": "known_history_tick_path_risk_final30_diagnostics",
                   "declaration_sha256": digest(study / "declaration.json"), "selection_sha256": digest(study / "selection.json"),
                   "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED"})
    audit.require("chronology", v8.timestamp(declaration["run_utc"]) <= v8.timestamp(selected["run_utc"]) <= v8.timestamp(result["run_utc"]))
    for document in (selected, result):
        compare_points(audit, "frozen10_science_lineage", document, {"science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"]})
    for name, sha in {**declaration["science_code_hashes"], **declaration["lineage"]["files"], **DEPENDENCIES}.items():
        audit.pin(audited_path(name), sha)
    old = ROOT / "docs/spike_tick_age_payoff_20261005"
    old_declaration, old_selection, old_result, old8 = v9.metadata(old, audit)
    old_audit = read_json(old / "independent_audit.json")
    audit.pin(old / "independent_audit.json", declaration["lineage"]["round9_audit_sha256"])
    audit.require("passed9_independent_audit", old_audit["stage"] == "independent_tick_age_payoff_audit" and old_audit["passed"] is True and old_audit["errors"] == [])
    audit.equal("passing9_verifier_identity", old_audit["verifier_sha256"], DEPENDENCIES["scripts/verify_spike_tick_tail_signal.py"])
    for file, field in (("selection.json", "round9_selection_sha256"), ("results.json", "round9_results_sha256")):
        audit.pin(old / file, declaration["lineage"][field])
    audit_inherited_inputs(declaration["lineage"]["files"], old_audit["input_sha256"], audit, "all9_audited_inputs_anchored")
    for key in ("boundaries", "tick_sources", "m1_sources", "detectors"):
        audit.equal("known9_lineage/" + key, declaration["lineage"][key], old_selection["lineage"][key])
    audit.require("passing9_before10_declaration", v8.timestamp(old_audit["run_utc"]) <= v8.timestamp(declaration["run_utc"]))
    for symbol in SYMBOLS:
        bounds = declaration["lineage"]["boundaries"][symbol]
        audit.equal("boundary0/" + symbol, stamp(bounds["0"]), v7.date_bounds(DATES[0])[0])
        audit.equal("boundary100/" + symbol, stamp(bounds["100"]), v7.date_bounds(DATES[-1])[1])
    pin_artifacts(selected, audit); pin_artifacts(result, audit)
    return declaration, selected, result, old8


def verify_symbol(declaration, selected, result, old8, symbol, audit):
    times, quotes, sources = v8.load_symbol(old8, symbol, audit)
    detector = declaration["lineage"]["detectors"][symbol]
    if detector["adequate"] is not True:
        for value in (selected, result):
            audit.equal(symbol + "/untested", value["status"], "NOT TESTED")
            audit.equal(symbol + "/untested_reason", value["reason"], "inadequate_independently_audited_scale")
        audit.equal(symbol + "/no_untested_models", selected["final_models"], {})
        audit.equal(symbol + "/no_untested_candidates", selected["candidates"], {})
        audit.equal(symbol + "/no_untested_final_models", result["models"], {})
        return {"symbol": symbol, "status": "NOT TESTED", "observed_rows": len(times), "strategy_paths_checked": 0,
                "overlapping_label_paths_checked_including_repeated_family_artifacts": 0}, sources
    cuts = v8.cutoffs(len(times))
    scale, _ = v8.fit_scale(times, quotes, cuts["40"])
    audit.equal(symbol + "/fixed40_scale", detector["median_abs_log_return"], scale)
    native_side = 1 if symbol.startswith("BOOM") else -1
    risk = prior_adverse_window(times, quotes, scale, native_side, clock_only=True)
    ticks = dict(zip(times, quotes, strict=True))
    bounds = {k: stamp(v) for k, v in declaration["lineage"]["boundaries"][symbol].items()}
    for key in ("40", "50", "60", "70"):
        audit.equal(symbol + "/observed_row_cut/" + key, bounds[key], times[cuts[key]])
    source = declaration["lineage"]["m1_sources"][symbol]
    audit.pin(source_path(source["path"]), source["sha256"])
    minutes = v7.read_m1(source_path(source["path"]))
    full_m1 = v9.m1_prefix_inventory(minutes, max(minutes) + 60)
    contexts, fingerprints, coverage = {}, {}, {}
    strategy_count = label_count = 0
    for key in ("40", "50", "60", "70", "100"):
        prefix = selected["audits"][key] if key != "100" else result["audit"]
        contexts[key] = read_inputs(prefix, bounds[key], minutes, risk, audit, symbol + "/prefix" + key, fingerprints)
        coverage[key] = v9.m1_prefix_inventory(minutes, bounds[key])
        compare_points(audit, symbol + "/prefix_source_policy", prefix,
                       {"native_side": native_side, "frozen_first40_scale": scale, "source_rows": len(minutes),
                        "grid_minutes": full_m1["closed_grid_slots"], "missing_minutes": full_m1["unavailable_closed_slots"],
                        "tick_prefix_rows": sum(t < bounds[key] for t in times),
                        "tick_last_used": iso(max(t for t in times if t < bounds[key]))})
    for mode in MODES:
        pooled_rows, pooled_counts = {}, {}
        for family in FAMILIES:
            candidate = selected["candidates"][mode + "_" + family]
            audit.equal(symbol + "/no_eligible_candidate", candidate["development_eligible"], False)
            audit.require(symbol + "/three_folds", len(candidate["walk_forward"]) == 3)
            pooled_rows[family], pooled_counts[family] = [], dict.fromkeys(COUNTERS, 0)
            for i, (train, test) in enumerate((("40", "50"), ("50", "60"), ("60", "70"))):
                fold, label = candidate["walk_forward"][i], f"{symbol}/{mode}/{family}/wf{i+1}"
                model = fold["model"]
                audit.equal(label + "/training_input_pin", model["training_inputs"], selected["audits"][train]["inputs"])
                label_count += len(audit_labels(model, contexts[train], ticks, symbol, mode, bounds["0"], bounds[train], audit, label))
                expected = expected_signals(contexts[test], symbol, mode, family, model, bounds[train], bounds[test])
                rows, counts = audit_report(fold, expected, ticks, bounds[train], bounds[test], audit, label)
                audit.equal(label + "/common_clock", fold["common_available_rows"], sum(bounds[train] <= t < bounds[test] for t in contexts[test]))
                audit.equal(label + "/fit_status", fold["status"], model["status"])
                strategy_count += len(rows); pooled_rows[family].extend(rows)
                for key in COUNTERS: pooled_counts[family][key] += counts[key]
        pooled_rows["CLOCK"], pooled_counts["CLOCK"] = [], dict.fromkeys(COUNTERS, 0)
        for train, test in (("40", "50"), ("50", "60"), ("60", "70")):
            clock = expected_signals(contexts[test], symbol, mode, start=bounds[train], end=bounds[test])
            rows, counts = replay_partition(ticks, clock, bounds[train], bounds[test])
            pooled_rows["CLOCK"].extend(rows)
            for key in COUNTERS: pooled_counts["CLOCK"][key] += counts[key]
        days = observed_days(bounds["40"], bounds["70"])
        validation = selected["validation"][mode]
        for family in ("CLOCK", *FAMILIES):
            pooled = validation[family]
            rows = audit_paths(pooled["ledger"], pooled_rows[family], audit, f"{symbol}/{mode}/{family}/pooled")
            if family == "CLOCK": strategy_count += len(rows)
            compare_points(audit, symbol + "/pooled_counts/" + family, pooled["audit"], pooled_counts[family])
            statuses = [f["model"]["status"] for f in selected["candidates"][mode + "_" + family]["walk_forward"]] if family != "CLOCK" else ["TESTED"] * 3
            status = "TESTED" if all(s == "TESTED" for s in statuses) else "PARTIALLY TESTED" if any(s == "TESTED" for s in statuses) else "NOT TESTED"
            compare_points(audit, symbol + "/pooled_fit_status", pooled, {"status": status, "fold_fit_statuses": statuses})
            audit_metrics(pooled["metrics"], rows, pooled_counts[family], days, audit, symbol + "/pooled_metrics/" + family)
        audit_comparisons(validation["comparisons"], pooled_rows, validation, days, audit, symbol + "/pooled_comparisons/" + mode)
        pair = {family: selected["final_models"][mode + "_" + family] for family in FAMILIES}
        for field in ("common_timestamp_target_sha256", "training_clock_issuance_sha256", "training_completed_labels", "training_matrix"):
            audit.equal(symbol + "/matched44_45_training/" + field, pair["RIDGE44"][field], pair["RIDGE45"][field])
        for family, model in pair.items():
            audit.equal(symbol + "/final_training_input", model["training_inputs"], selected["audits"]["70"]["inputs"])
            label_count += len(audit_labels(model, contexts["70"], ticks, symbol, mode, bounds["0"], bounds["70"], audit, f"{symbol}/{mode}/{family}/final_fit"))
        final_rows, final = {}, result["models"][mode]
        for family in ("CLOCK", *FAMILIES):
            model = None if family == "CLOCK" else pair[family]
            expected = expected_signals(contexts["100"], symbol, mode, family, model, bounds["70"], bounds["100"])
            report = final["reports"][family]
            rows, _ = audit_report(report, expected, ticks, bounds["70"], bounds["100"], audit, f"{symbol}/{mode}/{family}/final30")
            audit.equal(symbol + "/final_common_clock", report["common_available_rows"], sum(bounds["70"] <= t < bounds["100"] for t in contexts["100"]))
            audit.equal(symbol + "/final_fit_status", report["status"], model["status"] if model else "TESTED")
            final_rows[family] = rows; strategy_count += len(rows)
            if model:
                compare_points(audit, symbol + "/no_final_promotion", report, {"development_eligible": False,
                               "historical_gate": False, "live_candidate": False, "model_sha256": model["serialized_estimator_sha256"]})
        audit_comparisons(final["comparisons"], final_rows, final["reports"], observed_days(bounds["70"], bounds["100"]), audit, symbol + "/final_comparisons/" + mode)
        audit_expansion(final["expansion_diagnostic"], final["reports"], final["comparisons"], audit, symbol + "/expansion/" + mode)
        for value, lo, hi, name in ((selected["opportunity_diagnostics"][mode], "40", "70", "development"),
                                    (result["opportunity_diagnostics"][mode], "70", "100", "final30")):
            label_count += audit_opportunities(value, contexts[hi], ticks, symbol, mode, bounds[lo], bounds[hi], audit, f"{symbol}/{mode}/targets/{name}")
    audit.equal(symbol + "/selected_candidate", selected["selected_candidate"], None)
    audit.equal(symbol + "/final_candidate", result["selected_candidate"], None)
    audit.equal(symbol + "/eligible_candidates", selected["eligible_candidates"], 0)
    return {"symbol": symbol, "observed_rows": len(times), "available_prefix_rows": {k: len(v) for k, v in contexts.items()},
            "prefix_m1_coverage": coverage, "strategy_paths_checked": strategy_count,
            "overlapping_label_paths_checked_including_repeated_family_artifacts": label_count}, sources


def verify_metrics_csv(path, document, development, audit):
    audit.pin(path)
    expected = []
    for symbol in SYMBOLS:
        value = document["symbols"][symbol]
        if value["status"] == "NOT TESTED":
            expected.append({"symbol": symbol, "status": "NOT TESTED"})
            continue
        for mode in MODES:
            reports = value["validation"][mode] if development else value["models"][mode]["reports"]
            for family in ("CLOCK", *FAMILIES):
                report = reports[family]
                expected.append({"symbol": symbol, "mode": mode, "family": family, "status": report.get("status", "TESTED"),
                    **{k: report["metrics"][k] for k in ("completed", "censored", "missing_entry", "profit_factor", "mean_net_R", "active_days", "observed_day_clusters")},
                    "historical_gate": False, "live_candidate": False})
    with path.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    audit.require("metric_table/row_count", len(saved) == len(expected))
    for raw, row in zip(saved, expected, strict=True):
        parsed = {}
        for key, value in row.items():
            parsed[key] = v7.boolean(raw[key]) if type(value) is bool else v9.csv_integer(raw[key]) if type(value) is int else finite(raw[key], True) if type(value) is float or value is None else raw[key]
        if len(row) == 2:
            audit.require("metric_table/no_untested_payoff", all(value == "" for key, value in raw.items() if key not in row))
        compare_points(audit, "metric_table/" + path.name, parsed, row)


def verify(study, audit):
    declaration, selection, result, old8 = metadata(study, audit)
    symbols, sources = [], []
    for symbol in SYMBOLS:
        report, source = verify_symbol(declaration, selection["symbols"][symbol], result["symbols"][symbol], old8, symbol, audit)
        symbols.append(report); sources.extend(source)
    verify_metrics_csv(study / "development_metrics.csv", selection, True, audit)
    verify_metrics_csv(study / "metrics.csv", result, False, audit)
    for name, expected in tuple(audit.inputs.items()):
        audit.require("unchanged_during_audit/" + name, digest(name) == expected)
    return symbols, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_tick_path_risk_20261006")
    args = parser.parse_args(); audit = Audit()
    for flag in FLAGS:
        audit.require("environment/" + flag, os.environ.get(flag, "false").lower() == "false")
    target = args.output / "independent_audit.json"
    if target.exists():
        raise ValueError("Refusing audit overwrite; preserve any previous attempt before repair")
    symbols, sources = [], []
    try:
        symbols, sources = verify(args.output.resolve(), audit)
    except Exception as exc:
        audit.failures.append({"check": "fatal_exception", "type": type(exc).__name__, "message": str(exc)})
    value = {"stage": "independent_tick_path_risk_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
             "passed": not audit.failures, "checks": audit.checks, "errors": audit.failures, "symbols": symbols, "sources": sources,
             "safety": dict.fromkeys(FLAGS, False), "saved_false_values_checked": audit.safety_values_checked,
             "input_sha256": audit.inputs, "max_numeric_error": audit.max_numeric_error,
             "verifier_file": "scripts/verify_spike_tick_path_risk.py", "verifier_sha256": digest(Path(__file__)),
             "scope": {"included": ["all24 immutable source/gap grids and passing9/8/7 raw provenance lineage pins",
                        "native-adverse600 strictly prior increments/601 contiguous quotes, fixed40 scale, gap resets and independent common44/45 masks",
                        "dense closed M1 slots versus available/unknown prices, causal ATR/context stamps and exact observed-row cutoffs",
                        "saved45 training CSV and native44/45 matrix/target/model hashes; scalers/intercept/scalar scores/training q75 and issuance",
                        "all individually replayed overlapping target paths and jointly replayed one-open model/clock paths with planned31minute purge",
                        "observed-day point means/PF/paired comparisons and saved undefined-draw/status/nonpromotion policy arithmetic"],
                       "excluded": ["full44 indicator regeneration or original rolling-feature-valid proof; ridge coefficient refitting",
                                    "bootstrap regeneration, inference validity, or scale-estimation uncertainty",
                                    "raw normalization already anchored by passing immutable7/8/9 audits",
                                    "fresh price/payoff OOS, independence, broker fills, measured costs, prospective or actual profit"]}}
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"passed": value["passed"], "checks": audit.checks, "sources": len(sources),
                     "symbols": len(symbols), "errors": len(audit.failures), "audit": str(target)}), flush=True)
    raise SystemExit(0 if value["passed"] else 1)


if __name__ == "__main__":
    main()
