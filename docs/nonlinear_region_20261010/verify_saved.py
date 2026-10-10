#!/usr/bin/env python3
"""Offline saved-evidence CI checks; never decode raw ticks or cached features.

This checks published bytes, model/label identities and gates from saved numbers.
It relies on the separately pinned independent audit for training arithmetic,
price paths and inference; it does not reproduce that audit in CI.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FOLDER = "docs/nonlinear_region_20261010"
PARENT = "docs/region_reward_20261008"
DECLARATION_SHA256 = "35756c799689ef6a03f76e3aeda86a4568994a328f79af13685a929f91bee09c"
RESULT_SHA256 = "4664f8d2ad7282acf1336e449efa6376f467452d86df8a3b2e70a83ab04cf4d7"
FINAL_AUDIT = FOLDER + "/independent_audit_final.json"
FINAL_AUDIT_SHA256 = "dcb6ef38f3ee3635f56cd467bf3179fe5f68604a77246d1d7615e0cbf223df4c"
PARENT_PINS = {
    "declaration.json": "5a5cd147eefc7d12ac83c8ef839c931292c5fb338e07a1f083001c893c5196e5",
    "results.json": "6e470b505d4f32ffb28600cd136ae886001f35e34631663553d8807b114d8115",
    "independent_audit.json": "a6194dd9178d2a14a7dee4d317429a1aaaad28d3d3ce4c195d9378f8e588f25b",
}
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SYMBOLS = ("BOOM600", "CRASH600")
FAMILIES = ("CLOCK", "RAW_REGION", "HYBRID_REGION", "RAW_BOOST_REGION", "HYBRID_BOOST_REGION")
NEW = FAMILIES[-2:]
PARTS = ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180")
REFERENCES = {NEW[0]: ("CLOCK", "RAW_REGION"), NEW[1]: ("CLOCK", "HYBRID_REGION", NEW[0])}
PARAMETERS = {"n_trees": 100, "learning_rate": .05, "max_depth": 3, "min_leaf": 200,
              "n_bins": 16, "leaf_regularization": 20., "quantile": .75}
POINT = "finite PF > 1 AND finite mean_net_R > 0"
AUDIT_SCOPE = "exact_parent_cache_labels_references_training_matrix_target_model_identity_training_bins_routed_tree_residual_arithmetic_cutoff_independent_prediction_issuance_native_tick_paths_netR_point_metrics_day_week_CIs_drawdown_thirds_Holm_development_and_stable_positive_gates"
AUDIT_OMISSIONS = ["full_optimal_split_search_and_tie_optimality", "original44_feature_formulas_and_intrinsic_validity",
                   "parent_training_quote_paths_this_run", "reference_quote_paths_this_run"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def utc(value):
    require(isinstance(value, str), "Explicit UTC timestamp string required")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0) and stamp.microsecond == 0,
            "UTC whole-second timestamp required")
    return stamp


def instant(value):
    require(isinstance(value, str), "Explicit audit/execution UTC timestamp required")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0), "UTC audit/execution timestamp required")
    return stamp


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def above(value, limit=0):
    return finite(value) and value > limit


def ci_positive(value, limit=0):
    return isinstance(value, list) and len(value) == 2 and all(finite(v) for v in value) and limit < value[0] <= value[1]


def safety(value):
    require(isinstance(value, dict) and set(value) == set(FLAGS) and all(v is False for v in value.values()),
            "Exactly four false live gates required")


def recursive_safety(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "safety":
                safety(item)
            if key in ("QUALIFIED", "qualified", "fresh_out_of_sample"):
                require(item is False, "Saved evidence cannot enable qualification or fresh OOS")
            recursive_safety(item)
    elif isinstance(value, list):
        for item in value:
            recursive_safety(item)


def reference_copy(row):
    output = {k: deepcopy(v) for k, v in row.items()
              if k not in ("qualification", "holm_p", "required_reference_checks", "target_comparisons")}
    output["reference_origin"] = {"result_sha256": PARENT_PINS["results.json"],
                                  **{k: row[k] for k in ("symbol", "variant", "partition")}}
    return output


def cell_index(rows):
    require(isinstance(rows, list) and len(rows) == 60, "Exactly60 result cells required")
    index = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    require(set(index) == {(s, f, p) for s in SYMBOLS for f in FAMILIES for p in PARTS}, "Exact unique60 cell identities required")
    return index


def holm(values):
    require(all(finite(v) and 0 <= v <= 1 for v in values), "Finite p-values required")
    output, running = [1.] * len(values), 0.
    for rank, i in enumerate(sorted(range(len(values)), key=lambda k: values[k])):
        running = max(running, min(1., (len(values) - rank) * values[i]))
        output[i] = running
    return output


def gate_expected(row, index):
    """Independent conjunction/reasons using saved numbers, not raw paths."""
    symbol, family, part = row["symbol"], row["variant"], row["partition"]
    refs = REFERENCES[family]

    def clean(p):
        r = index[(symbol, family, p)]
        return (r["metrics"]["censored"] == r["metrics"]["invalid_uncensored"] == 0
                and r["replay_audit"]["unknown"] == 0
                and all(index[(symbol, ref, p)]["replay_audit"]["unknown"] == 0 for ref in refs))

    validation = index[(symbol, family, "walk_forward_combined")]["metrics"]
    development = (validation["completed"] >= 500 and validation["active_days"] >= 30
                   and above(validation["selection_score"]) and clean("walk_forward_combined"))
    for p in ("wf1", "wf2", "wf3"):
        m = index[(symbol, family, p)]["metrics"]
        development = development and m["completed"] >= 100 and above(m["mean_net_R"]) and above(m["profit_factor"], 1) and clean(p)
    m, day, week, pf = row["metrics"], row["day_inference"], row["weekly_inference"], row["profit_factor_inference"]
    checks = [
        (development, "development_rejected_or_insufficient"),
        (m["completed"] >= 1000, "fewer_than_1000_completed"),
        (m["active_days"] >= 60, "fewer_than_60_active_days"),
        (m["censored"] == m["invalid_uncensored"] == 0, "incomplete_payoff"),
        (row["replay_audit"]["unknown"] == 0 and all(index[(symbol, ref, part)]["replay_audit"]["unknown"] == 0 for ref in refs), "unknown_region_outcomes"),
        (ci_positive(pf["day_profit_factor_ci95"], 1), "day_PF_CI_not_above_1"),
        (ci_positive(pf["weekly_profit_factor_ci95"], 1), "week_PF_CI_not_above_1"),
        (ci_positive(day["mean_net_R_ci95"]), "day_mean_CI_not_positive"),
        (ci_positive(week["weekly_mean_net_R_ci95"]), "week_mean_CI_not_positive"),
        (ci_positive(day["baseline_difference_ci95"]), "day_control_advantage_not_positive"),
        (ci_positive(week["weekly_difference_ci95"]), "week_control_advantage_not_positive"),
        (finite(row["holm_p"]) and 0 <= row["holm_p"] < .05, "Holm_not_significant"),
        (len(row["thirds"]) == 3 and all(t["metrics"]["completed"] >= 200 and above(t["metrics"]["mean_net_R"]) for t in row["thirds"]), "chronological_thirds_not_stable"),
        (finite(m["closed_trade_max_drawdown"]) and m["closed_trade_max_drawdown"] <= .10 and m["equity_ruin"] is False, "drawdown_or_ruin"),
        (above(row["double_cost_metrics"]["mean_net_R"]), "doubled_cost_not_positive"),
        (above(m["profit_factor"], 1) and above(m["mean_net_R"]), "PF_not_above_1_or_mean_not_positive_or_unknown"),
    ]
    reasons = [reason for passed, reason in checks if not passed]
    extra = {}
    for ref in refs[1:]:
        pair = row["target_comparisons"][ref]
        extra[ref] = {
            "no_unknown_reference_outcomes": index[(symbol, ref, part)]["replay_audit"]["unknown"] == 0,
            "day_mean_CI_positive": ci_positive(pair["day"]["mean_net_R_ci95"]),
            "day_advantage_CI_positive": ci_positive(pair["day"]["baseline_difference_ci95"]),
            "week_mean_CI_positive": ci_positive(pair["week"]["weekly_mean_net_R_ci95"]),
            "week_advantage_CI_positive": ci_positive(pair["week"]["weekly_difference_ci95"]),
        }
        reasons += [f"required_reference_{ref}_{name}" for name, passed in extra[ref].items() if not passed]
    return bool(development), reasons, extra


class SavedVerifier:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pins, self.tables, self.checks = {}, {}, 0

    def check(self, condition, message):
        self.checks += 1
        require(condition, message)

    def pin(self, name, expected):
        self.check(valid_digest(expected), "Valid SHA256 required: " + str(name))
        target = (self.root / name).resolve()
        target.relative_to(self.root)
        relative = target.relative_to(self.root).as_posix()
        self.check(relative not in self.pins or self.pins[relative] == expected, "Conflicting artifact pins: " + relative)
        if relative not in self.pins:
            self.check(target.is_file() and digest(target) == expected, "Missing/changed saved artifact: " + relative)
            self.pins[relative] = expected
        return target

    def document(self, name, expected):
        return json.loads(self.pin(name, expected).read_text())

    def artifact(self, artifact):
        target = self.pin(artifact["path"], artifact["sha256"])
        if "rows" in artifact:
            self.check(type(artifact["rows"]) is int and artifact["rows"] >= 0, "Known nonnegative row count required")
            if artifact["path"] not in self.tables:
                opener = gzip.open if target.suffix == ".gz" else open
                with opener(target, "rt", newline="", encoding="utf-8") as stream:
                    self.tables[artifact["path"]] = list(csv.DictReader(stream))
            self.check(len(self.tables[artifact["path"]]) == artifact["rows"], "Saved table row count changed: " + artifact["path"])
        return target

    def artifacts(self, value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= set(value):
                self.artifact(value)
            else:
                for item in value.values():
                    self.artifacts(item)
        elif isinstance(value, list):
            for item in value:
                self.artifacts(item)


def training_labels(records, start, end):
    """Check saved categories/target bytes; source prices are not read."""
    count, dispositions, h = Counter(), Counter(), hashlib.sha256(b"paired-region-filled-target-v1\0")
    completed = []
    previous = None
    for row in records:
        issue, planned = utc(row["signal_time"]), utc(row["planned_end"])
        require(start <= issue < end and issue.second == 0 and issue.minute in (0, 30)
                and issue + timedelta(minutes=31) <= end and planned == issue + timedelta(minutes=30, seconds=1)
                and planned < end and (previous is None or previous < issue), "Saved training chronology/purge differs")
        previous = issue
        status = row["status"]
        categories = {"completed": ("completed",), "known_nonfill": ("expired", "invalidated"),
                      "unknown_path": ("waiting_gap", "entry_gap", "path_gap"), "exposure_skipped": ("overlap_skipped",)}
        require(status in {s for values in categories.values() for s in values}, "Known training disposition required")
        for key, values in categories.items():
            require(row[key] in ("True", "False") and (row[key] == "True") == (status in values), "Saved label category differs")
            count[key] += row[key] == "True"
        dispositions[status] += 1
        if status == "completed":
            value = float(row["net_R"])
            require(math.isfinite(value), "Finite original completed reward required")
            completed.append((issue, value))
        else:
            require(row["net_R"] == "", "Unknown/nonfill/skipped reward must remain empty, never zero")
    for stamp, _ in completed:
        seconds = int((stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())
        h.update(struct.pack(">q", seconds * 1_000_000_000))
    for _, value in completed:
        h.update(struct.pack(">d", value))
    return dict(count), dict(dispositions), h.hexdigest()


def verify(root, audit_name, audit_sha256):
    v = SavedVerifier(root)
    v.check(audit_name == FINAL_AUDIT and audit_sha256 == FINAL_AUDIT_SHA256, "Exact final audit identity required; earlier failed audit remains separate")
    d = v.document(FOLDER + "/declaration.json", DECLARATION_SHA256)
    r = v.document(FOLDER + "/results.json", RESULT_SHA256)
    for name, expected in (("declaration", DECLARATION_SHA256), ("results", RESULT_SHA256)):
        v.check((v.root / FOLDER / (name + ".sha256")).read_text().strip() == expected, "Exact result/declaration sidecar required")
    v.check(d["stage"] == "frozen_nonlinear_conditional_region_reward" and r["stage"] == "executed_nonlinear_conditional_region_reward"
            and r["declaration_sha256"] == DECLARATION_SHA256, "Frozen study/result identity changed")
    recursive_safety(d); recursive_safety(r)
    v.check(r["known_history_adaptive"] is True and d["new_historical_targets_or_scores_computed"] is False, "Adaptive premeasurement declaration required")
    parent = {name: v.document(PARENT + "/" + name, pin) for name, pin in PARENT_PINS.items()}
    pdcl, prior, paudit = (parent[n] for n in ("declaration.json", "results.json", "independent_audit.json"))
    v.check(paudit["passed"] is True and paudit["errors"] == [] and paudit["models_checked"] == 16, "Prior independently audited evidence required")
    expected_refs = [x for x in prior["rows"] if x["variant"] in FAMILIES[:3]]
    expected_inputs = {"parent_declaration_sha256": PARENT_PINS["declaration.json"], "parent_result_sha256": PARENT_PINS["results.json"],
        "parent_audit_sha256": PARENT_PINS["independent_audit.json"], "fold_inputs": pdcl["inputs"]["fold_inputs"],
        "training_artifacts": {s: prior["symbols"][s]["folds"] for s in SYMBOLS}, "reference_rows": expected_refs}
    v.check(same(d["inputs"], expected_inputs) and same(d["sources"], pdcl["sources"]), "Exact inherited manifest required")
    spec = d["specification"]
    for key, value in {"symbols": list(SYMBOLS), "families": list(FAMILIES), "new_families": list(NEW),
        "folds": pdcl["specification"]["folds"], "parameters": PARAMETERS, "point_criterion": POINT,
        "references": {k: list(a) for k, a in REFERENCES.items()}, "joint_heldout_Holm_family": 8,
        "completed_target_per_symbol_model_period": 1000, "active_days_target": 60,
        "bootstrap_repeats": 9999, "bootstrap_seed": 20261008, "known_unknown_references_preclude_historical_qualification": True}.items():
        v.check(same(spec[key], value), "Fixed saved specification changed: " + key)
    for name, pin in d["code_sha256"].items():
        v.pin(name, pin)
    source_audit = d["sources"]
    v.pin(source_audit["audit_path"], source_audit["audit_sha256"])
    omitted = {}
    quotes = gaps = 0
    for symbol in SYMBOLS:
        sources = source_audit["symbols"][symbol]
        v.check(len(sources) == 361 and len({a["date"] for a in sources}) == 361, "Complete daily source identity required")
        for a in sources:
            v.check(valid_digest(a["sha256"]) and type(a["rows"]) is int and type(a["missing_seconds"]) is int
                    and a["rows"] + a["missing_seconds"] == 86400, "Preserved source count/hash declaration required")
            omitted[a["path"]] = a["sha256"]
            quotes += a["rows"]; gaps += a["missing_seconds"]
        for manifest in d["inputs"]["fold_inputs"][symbol].values():
            for family in ("RAW44", "HYBRID44"):
                a = manifest["features"][family]; omitted[a["path"]] = a["sha256"]
            v.artifact(manifest["features"]["availability"])
            v.artifact(manifest["timed_fit"])
    v.check(quotes == 62380130 and gaps == 670 and len(omitted) == 738, "Exact native source/cached-feature manifest counts required")
    index = cell_index(r["rows"])
    for row in expected_refs:
        v.check(same(index[(row["symbol"], row["variant"], row["partition"])], reference_copy(row)), "Exact36 parent reference clones required")
    bounds = {part: values for fold in spec["folds"].values() for part, values in fold["evaluate"].items()}
    bounds["walk_forward_combined"] = [bounds["wf1"][0], bounds["wf3"][1]]
    held, probabilities, issued, dispositions = [], [], 0, 0
    for row in r["rows"]:
        v.check([row["start"], row["end_exclusive"]] == bounds[row["partition"]], "Exact partition dates required")
        v.artifacts(row["artifacts"])
        if row["variant"] not in NEW:
            continue
        v.check(row["endpoint"] == "REGION" and row["baseline_variant"] == "CLOCK"
                and set(row["target_comparisons"]) == set(REFERENCES[row["variant"]]) - {"CLOCK"}, "Required conditional references changed")
        if row["partition"] != "walk_forward_combined":
            issued += row["artifacts"]["signals"]["rows"]
            dispositions += row["artifacts"]["events"]["rows"]
        if row["partition"] in ("final_test", "later180"):
            held.append(row)
            values = [row["day_inference"]["p"], row["weekly_inference"]["weekly_p"]]
            for pair in row["target_comparisons"].values():
                values += [pair["day"]["p"], pair["week"]["weekly_p"]]
            v.check(all(finite(x) and 0 <= x <= 1 for x in values), "Saved conjunction p-values invalid")
            probabilities.append(max(values))
    v.check(len(held) == 8, "Exactly8 new held-out gates required")
    for row, adjusted in zip(held, holm(probabilities), strict=True):
        q = row["qualification"]
        development, reasons, checks = gate_expected(row, index)
        v.check(row["holm_p"] == adjusted, "Saved-number joint8 Holm changed")
        v.check(q["point_criterion"] == POINT and "PF_1_5_lower_CI_supported" not in q
                and q["development_eligible"] is development and q["historical_criteria_passed"] is (not reasons)
                and q["historical_rejection_reasons"] == reasons and same(row["required_reference_checks"], checks), "Saved-number stable-positive gates changed")
    model_files = model_count = labels_total = completed_total = 0
    v.check(set(r["symbols"]) == set(SYMBOLS), "Exact paired symbol states required")
    for symbol, s in r["symbols"].items():
        v.check(set(s["folds"]) == set(spec["folds"]), "Exactly4 fitting prefixes required")
        for fold, artifacts in s["folds"].items():
            old = d["inputs"]["training_artifacts"][symbol][fold]
            v.check(same(artifacts["labels"], old["labels"]) and same(artifacts["events"], old["events"])
                    and same(artifacts["parent_models"], old["models"]), "Exact saved parent label/event/model lineage required")
            v.artifacts(artifacts)
            state = json.loads((v.root / artifacts["models"]["path"]).read_text())
            previous = json.loads((v.root / old["models"]["path"]).read_text())
            recursive_safety(state)
            records = v.tables[artifacts["labels"]["path"]]
            start, end = map(utc, spec["folds"][fold]["train"])
            counts, statuses, target = training_labels(records, start, end)
            v.check(state["status"] == artifacts["status"] == "FITTED" and state["error"] is None
                    and set(state["models"]) == set(NEW) and same(state["training_audit"], previous["training_audit"])
                    and same(state["training_dispositions"], statuses), "Eight paired frozen fitting states required")
            meta, oldmeta = state["model_metadata"], previous["model_metadata"]
            for key, count in counts.items():
                v.check(state[key] == meta[key] == count, "Saved training category count differs: " + key)
            v.check(counts["completed"] >= 1000 and meta["opportunities"] == len(records)
                    and meta["target_sha256"] == oldmeta["target_sha256"] == target
                    and meta["training_start"] == start.isoformat() and meta["training_end"] == end.isoformat()
                    and same(meta["clock"], oldmeta["clock"]), "Training target/bounds/clock identity changed")
            v.check(meta["metadata_sha256"] == fingerprint({k: a for k, a in meta.items() if k != "metadata_sha256"})
                    and meta["symbol"] == symbol and meta["native_side"] == (1 if symbol == "BOOM600" else -1)
                    and same(meta["parameters"], PARAMETERS) and meta["standardization"] is False
                    and meta["algorithm"] == "app.research.nonlinear_signal.fit_histogram_boost"
                    and same(meta["feature_names"], pdcl["specification"]["feature_names"])
                    and meta["training_is_independent_trade_sample"] is False
                    and meta["label_selection_is_conditional_on_CLOCK_fill"] is True, "Paired booster metadata identity changed")
            for arm, oldarm in zip(NEW, ("RAW_REGION", "HYBRID_REGION"), strict=True):
                model = state["models"][arm]
                v.check(fingerprint(model) == meta["model_sha256"][arm]
                        and model["integrity_sha256"] == fingerprint({k: a for k, a in model.items() if k != "integrity_sha256"})
                        and same(model["parameters"], PARAMETERS) and same(model["defaults"], PARAMETERS)
                        and model["fit_rows"] == counts["completed"] and len(model["trees"]) == 100
                        and len(model["bin_boundaries"]) == 44 and same(model["feature_names"], meta["feature_names"])
                        and finite(model["training_score_quantile"]) and model["threshold"] == max(0., model["training_score_quantile"])
                        and meta["matrix_sha256"][arm] == oldmeta["matrix_sha256"][oldarm], "Frozen booster/model matrix digest changed")
                model_count += 1
            labels_total += len(records); completed_total += counts["completed"]; model_files += 1
    v.check(model_files == 8 and model_count == 16, "Exactly8 model artifacts/16 paired boosters required")
    audit = v.document(audit_name, audit_sha256)
    v.check((v.root / audit_name).with_suffix(".sha256").read_text().strip() == audit_sha256, "Exact independent audit sidecar required")
    recursive_safety(audit)
    v.check(audit["stage"] == "independent_nonlinear_conditional_region_audit" and audit["passed"] is True
            and audit["errors"] == [] and type(audit["checks"]) is int and audit["checks"] > 0
            and audit["declaration_sha256"] == DECLARATION_SHA256 and audit["result_sha256"] == RESULT_SHA256
            and audit["scope"] == AUDIT_SCOPE and audit["not_independently_recomputed"] == AUDIT_OMISSIONS
            and audit["parent_training_and_reference_paths_basis"] == "exact byte identity to pinned prior independently audited artifacts"
            and audit["known_history_adaptive"] is True, "Complete pinned independent audit with explicit scope required")
    for key, count in {"models_checked": model_count, "training_clock_labels_checked_including_overlapping_prefixes": labels_total,
        "completed_training_labels_checked_including_overlapping_prefixes": completed_total, "issued_scores_checked": issued,
        "evaluation_region_dispositions_checked": dispositions, "native_quote_rows_checked": quotes,
        "missing_seconds_preserved": gaps, "cells_checked": 60, "frozen_reference_cells_checked": 36,
        "paired_day_week_comparisons_checked": 120, "heldout_gates_checked": 8}.items():
        v.check(audit[key] == count, "Independent audit identity/count differs: " + key)
    audit_pins = audit["input_sha256"]
    v.check(isinstance(audit_pins, dict), "Independent source pins required")
    for name, pin in tuple(v.pins.items()):
        if name == audit_name:
            continue
        v.check(audit_pins.get(name) == pin, "Independent audit must pin saved input: " + name)
    for name, pin in omitted.items():
        v.check(audit_pins.get(name) == pin, "Independent audit must pin omitted raw/features: " + name)
    review = FOLDER + "/review_study.py"
    v.check(audit["verifier_sha256"] == audit_pins.get(review), "Independent review source identity changed")
    v.check(FOLDER + "/test_review_study.py" in audit_pins, "Independent review tests must be pinned")
    for name, pin in audit_pins.items():
        if name not in omitted:
            v.pin(name, pin)
    execution = json.loads((v.root / FOLDER / "execution_started.json").read_text())
    safety(execution["safety"])
    v.check(execution["declaration_sha256"] == DECLARATION_SHA256
            and instant(d["declared_utc"]) <= instant(execution["started_utc"])
            <= instant(r["completed_utc"]) <= instant(audit["run_utc"]), "Declaration/execution/audit chronology differs")
    v.check(not (v.root / FOLDER / "execution_failure.json").exists(), "A failed execution cannot silently pass CI")
    for name, pin in v.pins.items():
        v.check(digest(v.root / name) == pin, "Saved evidence changed during validation")
    return {"status": "PASS", "checks": v.checks, "errors": [], "declaration_sha256": DECLARATION_SHA256,
        "result_sha256": RESULT_SHA256, "independent_audit_path": audit_name, "independent_audit_sha256": audit_sha256,
        "saved_files_hashed": len(v.pins), "saved_csv_tables_checked": len(v.tables),
        "result_cells": 60, "parent_reference_clones": 36, "new_fit_state_artifacts": model_files,
        "paired_boosters": model_count, "heldout_saved_number_gates": len(held),
        "training_labels_including_overlapping_prefixes": labels_total,
        "raw_tick_and_feature_files_not_read": len(omitted),
        "scope": "pinned_saved_bytes_CSV_counts_training_label_categories_and_target_identity_model_digests_parent_reference_clones_dates_references_saved_number_gates_Holm_and_completed_independent_audit_lineage_only",
        "original_quote_paths_or_feature_formulas_recomputed": False,
        "training_arithmetic_or_bootstrap_CIs_recomputed": False,
        "saved_statistics_gates_and_Holm_recomputed": True,
        "independent_audit_scope": AUDIT_SCOPE, "independent_audit_omissions": AUDIT_OMISSIONS,
        "fresh_out_of_sample": False, "QUALIFIED": False, "safety": dict.fromkeys(FLAGS, False)}


class BoundaryTests(unittest.TestCase):
    @staticmethod
    def clean_gate():
        rows = []
        for symbol in SYMBOLS:
            for family in FAMILIES:
                for part in PARTS:
                    rows.append({"symbol": symbol, "variant": family, "partition": part,
                        "metrics": {"completed": 1200, "active_days": 60, "selection_score": .01,
                                    "censored": 0, "invalid_uncensored": 0, "mean_net_R": .02,
                                    "profit_factor": 1.05, "closed_trade_max_drawdown": .05, "equity_ruin": False},
                        "replay_audit": {"unknown": 0},
                        "day_inference": {"mean_net_R_ci95": [.001, .03], "baseline_difference_ci95": [.001, .03]},
                        "weekly_inference": {"weekly_mean_net_R_ci95": [.001, .03], "weekly_difference_ci95": [.001, .03]},
                        "profit_factor_inference": {"day_profit_factor_ci95": [1.001, 1.06], "weekly_profit_factor_ci95": [1.001, 1.06]},
                        "holm_p": .01, "thirds": [{"metrics": {"completed": 400, "mean_net_R": .01}} for _ in range(3)],
                        "double_cost_metrics": {"mean_net_R": .001}})
        index = cell_index(rows)
        for row in rows:
            if row["variant"] in NEW:
                row["target_comparisons"] = {ref: {"day": deepcopy(row["day_inference"]), "week": deepcopy(row["weekly_inference"])}
                                             for ref in REFERENCES[row["variant"]][1:]}
        return index[("CRASH600", NEW[1], "later180")], index

    def test_strict_point_boundary_and_no_one_point_one_floor(self):
        self.assertFalse(above(1., 1)); self.assertTrue(above(1.000001, 1))
        self.assertTrue(above(1.05, 1)); self.assertFalse(above(True, 0))
        self.assertFalse(above(float("nan"), 0)); self.assertFalse(above(float("inf"), 0))

    def test_full_saved_gate_accepts_below_one_point_one_but_not_one_or_zero_mean(self):
        row, index = self.clean_gate()
        self.assertEqual(gate_expected(row, index)[:2], (True, []))
        row["metrics"]["profit_factor"] = 1.
        self.assertIn("PF_not_above_1_or_mean_not_positive_or_unknown", gate_expected(row, index)[1])
        row["metrics"]["profit_factor"] = 1.05; row["metrics"]["mean_net_R"] = 0.
        self.assertIn("PF_not_above_1_or_mean_not_positive_or_unknown", gate_expected(row, index)[1])

    def test_required_reference_unknown_cannot_be_dropped(self):
        row, index = self.clean_gate()
        index[("CRASH600", "RAW_BOOST_REGION", "later180")]["replay_audit"]["unknown"] = 1
        _, reasons, checks = gate_expected(row, index)
        self.assertIn("unknown_region_outcomes", reasons)
        self.assertFalse(checks["RAW_BOOST_REGION"]["no_unknown_reference_outcomes"])
        index[("CRASH600", "CLOCK", "wf2")]["replay_audit"]["unknown"] = 1
        self.assertFalse(gate_expected(row, index)[0])

    def test_ci_boundary_unknown_and_reversed(self):
        self.assertTrue(ci_positive([1.0001, 1.05], 1))
        for interval in ([1., 2.], [None, 2.], [2., 1.5], [1.1, float("inf")]):
            self.assertFalse(ci_positive(interval, 1))

    def test_all_false_and_missing_safety(self):
        safety(dict.fromkeys(FLAGS, False))
        for flag in FLAGS:
            changed = dict.fromkeys(FLAGS, False); changed[flag] = True
            with self.assertRaises(ValueError): safety(changed)
        with self.assertRaises(ValueError): safety({})
        with self.assertRaises(ValueError): recursive_safety({"qualified": True})

    def test_hash_mutation_missing_file_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"; path.write_text("{}\n")
            pin = digest(path); SavedVerifier(directory).pin("saved.json", pin)
            path.write_text("{\"changed\":true}\n")
            with self.assertRaises(ValueError): SavedVerifier(directory).pin("saved.json", pin)
            with self.assertRaises(ValueError): SavedVerifier(directory).pin("missing", pin)
            with self.assertRaises(ValueError): SavedVerifier(directory).pin("../outside", pin)

    def test_csv_rows_handle_embedded_newline_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.csv.gz"
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["value"]); writer.writerow(["a\nb"])
            verifier = SavedVerifier(directory)
            a = {"path": path.name, "sha256": digest(path), "rows": 1}; verifier.artifact(a)
            a["rows"] = 2
            with self.assertRaises(ValueError): verifier.artifact(a)

    def test_exact_cells_duplicate_or_unexpected_rejected(self):
        rows = [{"symbol": s, "variant": f, "partition": p} for s in SYMBOLS for f in FAMILIES for p in PARTS]
        self.assertEqual(len(cell_index(rows)), 60)
        rows[-1] = deepcopy(rows[0])
        with self.assertRaises(ValueError): cell_index(rows)

    def test_holm_exact_boundaries(self):
        self.assertEqual(holm([.01, .02, 1.]), [.03, .04, 1.])
        for bad in ([True], [-.1], [float("nan")], [1.01]):
            with self.assertRaises(ValueError): holm(bad)

    def test_unknown_and_no_fill_never_zero_and_purge(self):
        start, end = utc("2026-01-01T00:00:00Z"), utc("2026-01-01T01:00:00Z")
        row = {"signal_time": start.isoformat(), "planned_end": (start + timedelta(minutes=30, seconds=1)).isoformat(),
               "status": "waiting_gap", "net_R": "", "completed": "False", "known_nonfill": "False",
               "unknown_path": "True", "exposure_skipped": "False"}
        counts, _, _ = training_labels([row], start, end); self.assertEqual(counts["unknown_path"], 1)
        row["net_R"] = "0"
        with self.assertRaises(ValueError): training_labels([row], start, end)
        row["net_R"] = ""; row["signal_time"] = (start + timedelta(minutes=30)).isoformat()
        row["planned_end"] = (start + timedelta(minutes=60, seconds=1)).isoformat()
        with self.assertRaises(ValueError): training_labels([row], start, end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=FINAL_AUDIT, help="Explicit final independent audit path; the earlier failed report is never selected")
    parser.add_argument("--audit-sha256", default=FINAL_AUDIT_SHA256, help="Exact final audit SHA256; never inferred from an untrusted sidecar")
    parser.add_argument("--output", help="Optional exclusive JSON report path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        require(valid_digest(args.audit_sha256), "Supply the exact completed independent audit SHA256 with --audit-sha256")
        report = verify(ROOT, args.audit, args.audit_sha256)
    except Exception as error:
        report = {"status": "FAIL", "errors": [f"{type(error).__name__}: {error}"],
                  "original_quote_paths_or_feature_formulas_recomputed": False,
                  "QUALIFIED": False, "safety": dict.fromkeys(FLAGS, False)}
    report["validator_sha256"] = digest(Path(__file__))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
