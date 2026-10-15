#!/usr/bin/env python3
"""Retrospective necessary-condition screen of immutable saved statistics only.

No price input, model fitting, replay, bootstrap, account or network operation.
This does not implement the full qualification gate or promote rejected models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SOURCE_PINS = {
    "docs/spike_multiframe_20261005/results.json": "46a10899922959c142e7aaf5359e40c06c9a6130042a267063eced2992cef9d9",
    "docs/spike_nonlinear_20261005/results.json": "4d00095faaa6b839452bcf61ad3cdd984a9bf1f632c7ece768f83ae52355e180",
    "docs/spike_tick_execution_20261005/results.json": "72d2d4ccb60fe75616480bacf33ca95c6a648313963f8f470418ca7202fe05b0",
    "docs/spike_tick_age_payoff_20261005/results.json": "820631643edc437ee5ef23e05a45cddf2e571791443ddfec87b27f7ac58ea3bb",
    "docs/spike_tick_path_risk_20261006/results.json": "b7ce626811d3fe51ff16315bf482632a256c1f698c171ff70ef4bb757324eb2d",
    "docs/spike_short_target_20261006/results.json": "abf9b54c03dcaf1db3db9107ac89cf0aeb124305d0e6ad2d715c79114d8b4d6f",
    "docs/spike_short_tick_20261006/results.json": "80276ad3eb67f155d753e7c7651a7a556aa91288f142192b3b01178e79a629d1",
    "docs/tail_successor_20261007/results.json": "a822c65561ffc63fb09e4732fb9860f748df441163d7ff7cfef31146d5ccdf6e",
    "docs/zone_study_20261008/results.json": "80d11be442287cd1491b9bc139bc002d3364ee29da55cbef6a02c9fc77fe824c",
    "docs/representation_regions_20261008/results.json": "212b2ec1c9330d7b3788fb0ad9cf4f860b8d29a8ff469423fea9ae39c4ad6589",
    "docs/hybrid_event_regions_20261008/results.json": "fb345c068a899157e15ec6393a72a7eeb5d78f932b7f7b7f0fac0640823b4abd",
    "docs/region_reward_20261008/results.json": "6e470b505d4f32ffb28600cd136ae886001f35e34631663553d8807b114d8115",
}
EXPECTED_ROWS = {
    "spike_multiframe_20261005": 4, "spike_nonlinear_20261005": 8,
    "spike_tick_execution_20261005": 6, "spike_tick_age_payoff_20261005": 6,
    "spike_tick_path_risk_20261006": 6, "spike_short_target_20261006": 12,
    "spike_short_tick_20261006": 6, "tail_successor_20261007": 0,
    "zone_study_20261008": 16, "representation_regions_20261008": 24,
    "hybrid_event_regions_20261008": 24, "region_reward_20261008": 20,
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def safety():
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() not in ("false", "0", "no", "off", ""):
            raise ValueError("Live gate must be false: " + flag)
    return dict.fromkeys(FLAGS, False)


def pointer(*tokens):
    return "/" + "/".join(str(t).replace("~", "~0").replace("/", "~1") for t in tokens)


def resolve(document, path):
    current = document
    for token in path.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def necessary_conditions(metrics):
    for name in ("completed", "active_days"):
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Saved nonnegative integer required: " + name)
    pf, mean = metrics["profit_factor"], metrics["mean_net_R"]
    for name, value in (("profit_factor", pf), ("mean_net_R", mean)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Finite numeric or unknown saved statistic required: " + name)
    return {"strict_net_PF_gt_1": positive(pf) and pf > 1, "strict_mean_net_R_gt_0": positive(mean),
            "completed_ge_1000": metrics["completed"] >= 1000,
            "active_days_ge_60": metrics["active_days"] >= 60}


def extract_descriptors(document, name):
    """Explicit scope; never recurse into training labels or alternate fills."""
    result = []
    def add(symbol, policy, part, tokens, endpoint="TIMED", role="model_or_rule", metrics_suffix=("metrics",)):
        result.append({"symbol": symbol, "policy": policy, "partition": part, "endpoint": endpoint,
            "role": role, "row_pointer": pointer(*tokens), "metrics_pointer": pointer(*tokens, *metrics_suffix)})
    if name == "spike_multiframe_20261005":
        for symbol, data in document["symbols"].items():
            for part in ("reused_final30", "fresh_temporal180"):
                add(symbol, "RIDGE44_SPIKE", part, ("symbols", symbol, "cohorts", part, "models", "SPIKE"))
    elif name == "spike_nonlinear_20261005":
        for symbol in document["symbols"]:
            for part in ("old_final30", "later_temporal180"):
                for policy in ("BOOST44_SPIKE", "RIDGE44_SPIKE"):
                    add(symbol, policy, part, ("symbols", symbol, "cohorts", part, "models", policy))
    elif name == "spike_tick_execution_20261005":
        for symbol in document["symbols"]:
            for policy in ("CLOCK_SPIKE", "BOOST44_SPIKE", "RIDGE44_SPIKE"):
                add(symbol, policy, "sampled_12_known_days", ("symbols", symbol, "models", policy),
                    role="clock_control" if policy == "CLOCK_SPIKE" else "model_or_rule")
    elif name in ("spike_tick_age_payoff_20261005", "spike_tick_path_risk_20261006"):
        extended = "RIDGE46" if name == "spike_tick_age_payoff_20261005" else "RIDGE45"
        for symbol in document["symbols"]:
            for policy in ("CLOCK", "RIDGE44", extended):
                add(symbol, policy + "_SPIKE", "sampled_final30_4_known_days",
                    ("symbols", symbol, "models", "SPIKE", "reports", policy),
                    role="clock_control" if policy == "CLOCK" else "model_or_rule")
    elif name == "spike_short_target_20261006":
        for symbol in document["symbols"]:
            for part in ("old_final30_secondary", "later180_known_history_primary"):
                for policy in ("SHORT44_SPIKE", "LONG44_SPIKE", "CLOCK_SPIKE"):
                    add(symbol, policy, part, ("symbols", symbol, "cohorts", part, "models", policy),
                        role="clock_control" if policy == "CLOCK_SPIKE" else "model_or_rule")
    elif name == "spike_short_tick_20261006":
        for symbol in document["symbols"]:
            for policy in ("SHORT44_SPIKE", "LONG44_SPIKE", "CLOCK_SPIKE"):
                add(symbol, policy, "sampled_12_known_days", ("symbols", symbol, "models", policy, "tick"),
                    role="clock_control" if policy == "CLOCK_SPIKE" else "model_or_rule")
    elif name == "tail_successor_20261007":
        if document["profit_factor"] != "NOT TESTED":
            raise ValueError("Tail prerequisite unexpectedly claims policy PF")
    else:
        for i, row in enumerate(document["rows"]):
            if row["partition"] in ("final_test", "later180"):
                add(row["symbol"], row["variant"], row["partition"], ("rows", i),
                    row.get("endpoint", "REGION"),
                    "clock_control" if row["variant"] in ("CLOCK", "CONTEXT_GEOMETRIC") else "model_or_rule")
    if len(result) != EXPECTED_ROWS[name]:
        raise ValueError("Exact scoped stored row count changed: " + name)
    return result


def uncertainty(row, metrics):
    pf = row.get("profit_factor_inference", {})
    daily = row.get("day_inference", {})
    weekly = row.get("weekly_inference", {})
    # A missing weekly field remains unknown. Sparse observed-day pilots are not
    # silently relabeled weekly inference, and a point mean is never a CI.
    return {"weekly_profit_factor_ci95": pf.get("weekly_profit_factor_ci95", metrics.get("weekly_profit_factor_ci95")),
        "weekly_mean_net_R_ci95": weekly.get("weekly_mean_net_R_ci95", metrics.get("weekly_mean_net_R_ci95")),
        "day_mean_net_R_ci95": daily.get("mean_net_R_ci95", metrics.get("mean_net_R_ci95", metrics.get("day_mean_ci95"))),
        "day_profit_factor_ci95": pf.get("day_profit_factor_ci95", metrics.get("day_profit_factor_ci95", metrics.get("profit_factor_ci95"))),
        "saved_selection_score": metrics.get("selection_score"),
        "saved_day_inference": daily or None, "saved_weekly_inference": weekly or None,
        "saved_profit_factor_inference": pf or None,
        "not_recomputed_or_newly_validated": True}


def development(document, descriptor, row):
    eligible = row.get("development_eligible")
    # The SHORT1 tick result stores development on its model parent.
    if descriptor["row_pointer"].endswith("/tick"):
        row = resolve(document, descriptor["row_pointer"][:-5])
        eligible = row.get("development_eligible")
    if eligible is not None and not isinstance(eligible, bool):
        raise ValueError("Known saved development flag or unknown required")
    folds = []
    if descriptor["row_pointer"].startswith("/rows/"):
        for i, other in enumerate(document["rows"]):
            if (other["symbol"] == descriptor["symbol"] and other["variant"] == descriptor["policy"]
                    and other.get("endpoint", "REGION") == descriptor["endpoint"]
                    and other["partition"] in ("wf1", "wf2", "wf3", "walk_forward_combined")):
                m = other["metrics"]
                folds.append({"partition": other["partition"], "metrics_pointer": pointer("rows", i, "metrics"),
                    **{key: m.get(key) for key in ("completed", "active_days", "profit_factor", "mean_net_R", "selection_score")},
                    "unknown_regions": other.get("replay_audit", {}).get("unknown")})
    return {"saved_development_eligible": eligible, "saved_historical_gate": row.get("historical_gate"),
        "saved_original_qualification": row.get("qualification"), "saved_rejection_reasons": row.get("rejection_reasons"),
        "saved_fold_evidence": folds, "gate_reimplemented": False, "original_target_and_gate_results_unchanged": True}


def make_row(document, source, descriptor):
    row = resolve(document, descriptor["row_pointer"])
    metrics = resolve(document, descriptor["metrics_pointer"])
    condition = necessary_conditions(metrics)
    config = row.get("config") or row.get("replay_audit", {}).get("config") or document.get("config")
    if source.endswith("spike_short_tick_20261006/results.json"):
        config = document["config"]["tick"]
    # Exact stored source and pointer are retained even for known empty cells.
    return {"source_path": source, "source_sha256": SOURCE_PINS[source], **descriptor,
        "native_side": 1 if descriptor["symbol"].startswith("BOOM") else -1,
        "timeframes": ["H4", "H1", "M15", "M5", "M1"],
        "point": {key: metrics.get(key) for key in ("completed", "active_days", "profit_factor", "mean_net_R", "censored", "mean_gross_R")},
        "modeled_configuration": config, "necessary_conditions": condition,
        "necessary_conjunction_passed": all(condition.values()),
        "uncertainty": uncertainty(row, metrics), "development": development(document, descriptor, row),
        "source_safety_verified_false": True, "fresh_out_of_sample": False,
        "QUALIFIED": False, "prior_gate_results_unchanged": True}


def diagnostics(documents):
    source = "docs/spike_nonlinear_20261005/results.json"
    document = documents[source]
    base = ("symbols", "CRASH600", "cohorts", "later_temporal180", "models", "BOOST44_SPIKE")
    model = resolve(document, pointer(*base)); result = []
    if model["development_eligible"] is not False:
        raise ValueError("Original rejected model development identity changed")
    for index in (1, 7, 8, 13):
        location = pointer(*base, "sensitivities", index); metric = resolve(document, location)
        if not (positive(metric["round_trip_cost_atr"]) and positive(metric["profit_factor"])
                and metric["profit_factor"] > 1 and metric["completed"] >= 1000
                and metric["selection_score"] < 0):
            raise ValueError("Exact four diagnostic sensitivity identities changed")
        result.append({"source_path": source, "source_sha256": SOURCE_PINS[source],
            "json_pointer": location, "symbol": "CRASH600", "policy": "BOOST44_SPIKE",
            "partition": "later_temporal180", "excluded_from_primary_screen": True,
            **{k: metric[k] for k in ("completed", "active_days", "profit_factor", "mean_net_R", "round_trip_cost_atr",
                                      "fill_mode", "entry_delay_minutes", "selection_score")},
            "development_eligible_pointer": pointer(*base, "development_eligible"),
            "saved_development_eligible": False, "positive_weekly_uncertainty_evidence_saved": False,
            "full_saved_sensitivity": metric,
            "reason": "diagnostic_cost_fill_delay_sensitivity_of_development_rejected_model_not_new_candidate",
            "QUALIFIED": False, "prior_gate_results_unchanged": True})
    return result


def assessment():
    flags = safety(); documents, rows, sources = {}, [], []
    for source, expected in SOURCE_PINS.items():
        payload = (ROOT / source).read_bytes()
        if digest(payload) != expected: raise ValueError("Immutable saved result bytes changed: " + source)
        document = json.loads(payload)
        if document.get("safety") != flags: raise ValueError("All four saved live gates must be false: " + source)
        documents[source] = document; name = Path(source).parent.name
        scoped = extract_descriptors(document, name)
        rows += [make_row(document, source, descriptor) for descriptor in scoped]
        sources.append({"path": source, "sha256": expected, "included_stored_rows": len(scoped),
            "scope": "native_completed_feature_grid_primary_fill_summaries_only" if scoped else
                     "non_economic_tail_prerequisite_profit_factor_NOT_TESTED_no_policy_rows"})
    if len(rows) != 132 or len({(r["source_path"], r["row_pointer"]) for r in rows}) != 132:
        raise ValueError("Exact132 unique stored identities required")
    roles = dict(Counter(r["role"] for r in rows))
    if roles != {"model_or_rule": 96, "clock_control": 36}:
        raise ValueError("Exact model/rule and control scope changed")
    for source, expected in SOURCE_PINS.items():
        if digest((ROOT / source).read_bytes()) != expected: raise ValueError("Saved source changed during screening")
    return {"stage": "retrospective_saved_statistics_necessary_condition_screen",
        "screened_utc": datetime.now(timezone.utc).isoformat(),
        "screen_source_sha256": digest(Path(__file__).read_bytes()),
        "objective": "stable_positive_net_profit_no_minimum_PF_above_strict_1",
        "point_criterion": "strict_finite_PF_gt_1_AND_strict_finite_mean_net_R_gt_0",
        "necessary_sample_criteria": {"completed_ge": 1000, "active_days_ge": 60},
        "scope": {"saved_result_files": 12, "stored_rows": 132, "roles": roles,
            "row_counts_by_source": {s["path"]: s["included_stored_rows"] for s in sources},
            "chronology": "older_last30_and_later180_plus_declared_sparse_tick_followups_are_reused_known_history",
            "not_independent_rows": "references_repeat_across_studies_and_empty_representation_cells_are_retained; no_pooling",
            "excluded": ["M5_only_RIDGE19_and_prior19input_studies", "opposite_DRIFT", "training_labels",
                "walk_forward_unions_and_development_as_heldout", "classification_lift", "zero_cost_diagnostics",
                "alternate_fill_cost_delay_sensitivities_from_primary_rows", "duplicate_coarse_M1_SHORT1_replays"],
            "all_possible_policies_or_symbols_reviewed": False},
        "sources": sources, "rows": rows, "nonzero_cost_positive_diagnostics": diagnostics(documents),
        "summary": {"primary_stored_rows": len(rows), "model_or_rule_rows": roles["model_or_rule"],
            "clock_control_rows": roles["clock_control"],
            "undefined_PF_rows": sum(r["point"]["profit_factor"] is None for r in rows),
            "strict_positive_point_rows": sum(r["necessary_conditions"]["strict_net_PF_gt_1"] and r["necessary_conditions"]["strict_mean_net_R_gt_0"] for r in rows),
            "rows_with_ge1000_completed": sum(r["necessary_conditions"]["completed_ge_1000"] for r in rows),
            "rows_with_ge1000_completed_and_ge60_days": sum(r["necessary_conditions"]["completed_ge_1000"] and r["necessary_conditions"]["active_days_ge_60"] for r in rows),
            "necessary_conjunction_passed": sum(r["necessary_conjunction_passed"] for r in rows)},
        "assessment_kind": "necessary_condition_screen_not_full_qualification_gate_reimplementation",
        "original_paths_features_models_bootstraps_or_gates_recomputed": False,
        "uncertainty_and_development_requirements_waived": False,
        "broker_execution_and_measured_costs": "NOT TESTED", "prospective_paper": "NOT TESTED",
        "fresh_out_of_sample": False, "QUALIFIED": False,
        "prior_gate_results_unchanged": True, "safety": flags}


def self_test():
    checks = 0
    def check(condition):
        nonlocal checks
        if not condition: raise AssertionError("Saved-statistics screen self-test failed")
        checks += 1
    metric = {"completed": 1000, "active_days": 60, "profit_factor": 1.001, "mean_net_R": .0001}
    check(all(necessary_conditions(metric).values()))
    for key, value in (("profit_factor", 1.), ("mean_net_R", 0.), ("completed", 999), ("active_days", 59), ("profit_factor", None)):
        changed = {**metric, key: value}; check(not all(necessary_conditions(changed).values()))
    for key, value in (("completed", True), ("active_days", 60.), ("profit_factor", float("nan")), ("mean_net_R", True)):
        try: necessary_conditions({**metric, key: value})
        except ValueError: check(True)
        else: check(False)
    check(pointer("a/b", "x~y", 0) == "/a~1b/x~0y/0")
    check(resolve({"a/b": {"x~y": [7]}}, pointer("a/b", "x~y", 0)) == 7)
    missing = uncertainty({}, metric)
    check(missing["weekly_profit_factor_ci95"] is None and missing["weekly_mean_net_R_ci95"] is None)
    for flag in FLAGS:
        before = os.environ.get(flag); os.environ[flag] = "true"
        try:
            try: safety()
            except ValueError: check(True)
            else: check(False)
        finally:
            if before is None: os.environ.pop(flag)
            else: os.environ[flag] = before
    check(sum(EXPECTED_ROWS.values()) == 132 and len(SOURCE_PINS) == 12)
    print(json.dumps({"self_tests_passed": checks, "historical_price_rows_read": 0, "files_written": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FOLDER / "assessment.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(); safety()
    if args.self_test:
        self_test(); return
    target = (ROOT / args.output).resolve(); target.relative_to(FOLDER)
    if target.exists(): raise FileExistsError("Exclusive screen output already exists: " + str(target))
    report = assessment()
    with target.open("x") as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target.relative_to(ROOT)), "assessment_sha256": digest(target.read_bytes()),
        "screen_source_sha256": report["screen_source_sha256"], "summary": report["summary"],
        "QUALIFIED": False, "fresh_out_of_sample": False, "prior_gate_results_unchanged": True}), flush=True)


if __name__ == "__main__": main()
