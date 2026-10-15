#!/usr/bin/env python3
"""Independent saved BOOST-region arithmetic, quote paths and evidence audit.

Imports only independently authored audit helpers, never the learner/predictor,
production replay, metric or qualification functions. Training split optimality
is outside this audit: saved trees are checked against their routed residuals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import verify_region_reward_study as R
from scripts.verify_zone_study import Audit, FLAGS, epoch, metric_checks, read_events, sha
from scripts.verify_hybrid_event_study import array, clock_rows, frame, inference_oracle

FOLDER = Path(__file__).resolve().parent
PARENT = ROOT / "docs/region_reward_20261008"
PARENT_PINS = {
    "declaration.json": "5a5cd147eefc7d12ac83c8ef839c931292c5fb338e07a1f083001c893c5196e5",
    "results.json": "6e470b505d4f32ffb28600cd136ae886001f35e34631663553d8807b114d8115",
    "independent_audit.json": "a6194dd9178d2a14a7dee4d317429a1aaaad28d3d3ce4c195d9378f8e588f25b",
}
SYMBOLS = ("BOOM600", "CRASH600")
FAMILIES = ("CLOCK", "RAW_REGION", "HYBRID_REGION", "RAW_BOOST_REGION", "HYBRID_BOOST_REGION")
NEW = FAMILIES[-2:]
PARTS = ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180")
REFERENCES = {NEW[0]: ("CLOCK", "RAW_REGION"), NEW[1]: ("CLOCK", "HYBRID_REGION", NEW[0])}
PARAMETERS = {"n_trees": 100, "learning_rate": .05, "max_depth": 3,
              "min_leaf": 200, "n_bins": 16, "leaf_regularization": 20., "quantile": .75}
PROVENANCE = {
    "score_kind": "continuous_uncalibrated_score",
    "objective": "sum_squared_residual_error_plus_lambda_times_squared_leaf_values",
    "boundaries_fitted_on": "training_rows_only", "cutoff_fitted_on": "training_predictions_only",
    "quantile_method": "linear",
    "boundary_method": "training_low_cardinality_adjacent_midpoints_else_unique_internal_linear_quantiles",
    "bin_side": "right", "split_tie_break": "ascending_feature_position_then_split_bin",
    "allow_zero_gain_split": True, "zero_gain_split_scope": "root_only_no_epsilon_tolerance",
    "target_clipped": False, "features_clipped": False, "completed_labels_required": True,
}


def exact(actual, expected):
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def above(value, limit=0.):
    return finite(value) and value > limit


def ci_positive(value, limit=0.):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(finite(v) for v in value) and limit < value[0] <= value[1]


def model_digest(model):
    return R.fingerprint({k: v for k, v in model.items() if k != "integrity_sha256"})


def tree_values(values, tree, boundaries):
    """Route numeric rows by strict x<cut; no production binning or predictor."""
    result = np.empty(len(values), float)

    def visit(node, indices, depth):
        require(depth <= 3 and isinstance(node, dict), "Malformed/deep tree")
        require(finite(node.get("value")), "Finite tree value required")
        if "split_feature" not in node:
            result[indices] = node["value"]
            return
        feature, split = node.get("split_feature"), node.get("split_bin")
        require(type(feature) is int and 0 <= feature < len(boundaries), "Tree feature index")
        require(type(split) is int and 0 <= split < len(boundaries[feature]), "Tree bin index")
        left = values[indices, feature] < boundaries[feature][split]
        visit(node["left"], indices[left], depth + 1)
        visit(node["right"], indices[~left], depth + 1)

    visit(tree, np.arange(len(values)), 0)
    return result


def predict(values, model, names):
    require(exact(model["feature_names"], list(names)), "Frozen feature order differs")
    x = np.asarray(values, float)
    require(x.ndim == 2 and x.shape[1] == len(names) and np.isfinite(x).all(), "Finite ordered prediction matrix required")
    require(model_digest(model) == model["integrity_sha256"], "Boost integrity hash changed")
    require(exact(model["parameters"], PARAMETERS), "Frozen boost parameters changed")
    require(finite(model["initial_prediction"]) and len(model["trees"]) == 100, "Boost tree count/initial prediction")
    score = np.full(len(x), model["initial_prediction"], dtype=float)
    for tree in model["trees"]:
        score += .05 * tree_values(x, tree, model["bin_boundaries"])
    require(np.isfinite(score).all(), "Nonfinite independent score")
    return score


def expected_boundaries(x):
    boundaries = []
    for column in np.asarray(x, float).T:
        distinct = np.unique(column)
        if len(distinct) <= 16:
            cuts = np.array([b if a / 2 + b / 2 <= a else a / 2 + b / 2
                             for a, b in zip(distinct[:-1], distinct[1:])])
        else:
            cuts = np.unique(np.quantile(column, np.arange(1, 16) / 16, method="linear"))
            cuts = cuts[(cuts > column.min()) & (cuts < column.max())]
        boundaries.append(cuts.tolist())
    return boundaries


def check_boost(audit, x, y, model, names):
    """Check the saved route's sufficient statistics; do not search/refit splits."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == 2 and x.shape == (len(y), len(names)) and len(y) >= 1000, "Complete training matrix shape/count")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Finite completed training only")
    require(set(model) == {"version", "feature_names", "parameters", "defaults", "fit_rows",
                          "initial_prediction", "bin_boundaries", "trees", "training_score_quantile",
                          "threshold", "integrity_sha256", *PROVENANCE}, "Exact frozen boost schema")
    require(exact(model["parameters"], PARAMETERS) and exact(model["defaults"], PARAMETERS), "Exact fixed boosting algorithm")
    require(type(model["version"]) is int and model["version"] == 1, "Boost version")
    for key, value in PROVENANCE.items():
        require(exact(model.get(key), value), "Changed boost provenance: " + key)
    require(model_digest(model) == model["integrity_sha256"], "Boost internal digest")
    require(exact(model["feature_names"], list(names)), "Training feature order")
    audit.equal(model["fit_rows"], len(y), "completed-only boost fit rows")
    cuts = expected_boundaries(x)
    require(len(model["bin_boundaries"]) == len(cuts), "Training bin dimensions")
    for saved, expected in zip(model["bin_boundaries"], cuts, strict=True):
        array(audit, saved, expected, "independent training-only bin cuts")
    audit.equal(model["initial_prediction"], float(y.mean()), "initial completed target mean")
    require(len(model["trees"]) == 100, "Exactly100 frozen trees")
    scores = np.full(len(y), float(y.mean()))
    leaf_keys = {"count", "residual_sum", "value"}
    split_keys = leaf_keys | {"split_feature", "split_bin", "gain", "left", "right"}
    for ordinal, tree in enumerate(model["trees"]):
        residual = y - scores

        def node_check(node, indices, depth):
            require(isinstance(node, dict) and set(node) in (leaf_keys, split_keys), "Exact tree node schema")
            require(type(node["count"]) is int and len(indices) >= (1 if depth == 0 else 200), "Tree node sample count")
            require(finite(node["residual_sum"]) and finite(node["value"]), "Finite numeric routed residual fields")
            require(depth <= 3, "Tree depth limit")
            audit.equal(node["count"], len(indices), "routed node count")
            total = float(residual[indices].sum())
            audit.equal(node["residual_sum"], total, "routed training residual sum")
            audit.equal(node["value"], total / (len(indices) + 20.), "regularized routed residual value")
            if set(node) == leaf_keys:
                return
            feature, split = node["split_feature"], node["split_bin"]
            require(depth < 3 and type(feature) is int and 0 <= feature < x.shape[1], "Valid split feature/depth")
            require(type(split) is int and 0 <= split < len(cuts[feature]), "Valid split bin")
            mask = x[indices, feature] < cuts[feature][split]
            left, right = indices[mask], indices[~mask]
            require(len(left) >= 200 and len(right) >= 200, "Both children meet min_leaf")
            sums = [float(residual[v].sum()) for v in (left, right)]
            gain = sum(s * (s / (len(v) + 20.)) for s, v in zip(sums, (left, right))) - total * (total / (len(indices) + 20.))
            require(finite(node["gain"]) and (node["gain"] >= 0 if depth == 0 else node["gain"] > 0), "Root-only zero gain rule")
            # Histogram accumulation may differ slightly from direct row summation.
            array(audit, node["gain"], gain, "routed split gain", close=True)
            node_check(node["left"], left, depth + 1)
            node_check(node["right"], right, depth + 1)

        node_check(tree, np.arange(len(y)), 0)
        scores += .05 * tree_values(x, tree, cuts)
        require(np.isfinite(scores).all(), "Finite stage-wise training scores")
    quantile = float(np.quantile(scores, .75, method="linear"))
    audit.equal(model["training_score_quantile"], quantile, "independent completed-training q75")
    audit.equal(model["threshold"], max(0., quantile), "positive training cutoff")
    array(audit, predict(x, model, names), scores, "independent model training prediction", close=True)
    return scores


def check_model(audit, state, labels, raw, hybrid, positions, names, symbol, bounds):
    completed = R.booleans(labels.completed, "completed")
    count = int(completed.sum())
    counts = {k: int(R.booleans(labels[k], k).sum()) for k in ("completed", "known_nonfill", "unknown_path", "exposure_skipped")}
    for key, value in counts.items(): audit.equal(state[key], value, "fit state " + key)
    audit.equal(state["training_dispositions"], dict(Counter(labels.status)), "all conditional training dispositions")
    require(exact(state["safety"], R.safety()) and state["QUALIFIED"] is False, "Safe unqualified fit state")
    if count < 1000:
        require(state["status"].startswith("NOT_FIT") and state["models"] is None and state["model_metadata"] is None,
                "Insufficient completed-label fallback forbidden")
        return {}
    require(state["status"] == "FITTED" and state["error"] is None and set(state["models"]) == set(NEW), "Both paired boosting fits required")
    meta = state["model_metadata"]
    require(meta["metadata_sha256"] == R.fingerprint({k: v for k, v in meta.items() if k != "metadata_sha256"}), "Metadata hash changed")
    for key, value in {"version": 1, "symbol": symbol, "native_side": 1 if symbol == "BOOM600" else -1,
                       "feature_names": list(names), "opportunities": len(labels), "quantile": .75,
                       "standardization": False, "algorithm": "app.research.nonlinear_signal.fit_histogram_boost",
                       "target": "conditional_completed_original_quote_CLOCK_region_net_R",
                       "training_is_independent_trade_sample": False,
                       "label_selection_is_conditional_on_CLOCK_fill": True, "safety": R.safety(), **counts}.items():
        require(exact(meta[key], value), "Model metadata differs: " + key)
    require(exact(meta["parameters"], PARAMETERS), "Fixed estimator parameters")
    for key, value in zip(("training_start", "training_end"), bounds): audit.equal(epoch(meta[key]), epoch(value), key)
    y = labels.net_R.to_numpy(float)[completed]
    digest = R.target_hash(R.timestamps(labels.signal_time)[completed], y)
    audit.equal(meta["target_sha256"], digest, "exact paired completed target")
    for arm, features in zip(NEW, (raw, hybrid), strict=True):
        x = features.loc[positions[completed], names].to_numpy(float)
        h = hashlib.sha256(digest.encode()); h.update(np.asarray(x, dtype=">f8").tobytes())
        audit.equal(meta["matrix_sha256"][arm], h.hexdigest(), "paired matrix identity")
        audit.equal(meta["model_sha256"][arm], R.fingerprint(state["models"][arm]), "model byte identity")
        check_boost(audit, x, y, state["models"][arm], names)
    return state["models"]


def secondary_metrics(ledger, start, end):
    """Independent gate-relevant point statistics, including equity and thirds."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    issues = pd.to_datetime(ledger.signal_time, utc=True)
    cohort = ledger.loc[(issues >= start) & (issues < end)]
    done = cohort.loc[~cohort.censored & np.isfinite(cohort.net_R.to_numpy(float))]
    values = done.net_R.to_numpy(float)
    mean = float(values.mean()) if len(values) else None
    days = pd.date_range(start.floor("D"), (end - pd.Timedelta(nanoseconds=1)).floor("D"), freq="D")
    counts, totals = np.zeros(len(days)), np.zeros(len(days))
    for stamp, value in zip(done.entry_time, values, strict=True):
        i = int((pd.Timestamp(stamp).floor("D") - days[0]) / pd.Timedelta(days=1))
        require(0 <= i < len(days), "Completed entry outside analysis days")
        counts[i] += 1; totals[i] += value
    se = None
    if len(values) and len(days) > 1:
        se = float(math.sqrt(len(days) / (len(days) - 1) * float(np.sum((totals - mean * counts) ** 2))) / len(values))
    equity = peak = 1.; drawdown = 0.; ruined = False
    for row in sorted(done.to_dict("records"), key=lambda r: (epoch(r["exit_time"]), epoch(r["entry_time"]))):
        factor = 1 + .0025 * row["net_R"]
        ruined = ruined or factor <= 0
        equity = 0. if ruined else equity * factor
        peak = max(peak, equity); drawdown = max(drawdown, 1 - equity / peak)
    positive = values[values > 0].sum(); negative = -values[values < 0].sum()
    hold = done.holding_minutes.to_numpy(float)
    return {"completed": len(values), "trades": len(cohort), "censored": int(cohort.censored.sum()),
            "invalid_uncensored": len(cohort) - int(cohort.censored.sum()) - len(values),
            "mean_net_R": mean, "profit_factor": float(positive / negative) if negative > 0 else None,
            "active_days": int(np.count_nonzero(counts)), "calendar_days": len(days),
            "cluster_se": se, "selection_score": mean - 1.96 * se if se is not None else None,
            "closed_trade_return": equity - 1, "closed_trade_max_drawdown": drawdown,
            "equity_ruin": ruined, "illustrative_risk_fraction": .0025,
            "median_holding_minutes": float(np.median(hold)) if len(hold) else None}


def check_secondary(audit, row, ledger):
    for key, value in secondary_metrics(ledger, row["start"], row["end_exclusive"]).items():
        audit.equal(row["metrics"][key], value, "independent supplementary metric " + key)
    start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end_exclusive"])
    require(len(row["thirds"]) == 3, "Three chronological thirds required")
    for i, saved in enumerate(row["thirds"]):
        left, right = start + (end - start) * i / 3, start + (end - start) * (i + 1) / 3
        audit.equal(pd.Timestamp(saved["start"]).value, left.value, "third start")
        audit.equal(pd.Timestamp(saved["end"]).value, right.value, "third end")
        for key, value in secondary_metrics(ledger, left, right).items():
            audit.equal(saved["metrics"][key], value, "independent third " + str(i) + key)


def gate_outcome(row, index):
    """Independent full conjunction at revised PF>1 and positive-mean target."""
    symbol, family = row["symbol"], row["variant"]
    refs = REFERENCES[family]

    def clean(part):
        candidate = index[(symbol, family, part)]
        m = candidate["metrics"]
        return (m["censored"] == m["invalid_uncensored"] == 0 and candidate["replay_audit"]["unknown"] == 0
                and all(index[(symbol, ref, part)]["replay_audit"]["unknown"] == 0 for ref in refs))

    validation = index[(symbol, family, "walk_forward_combined")]["metrics"]
    development = (validation["completed"] >= 500 and validation["active_days"] >= 30
                   and above(validation["selection_score"]) and clean("walk_forward_combined"))
    for part in ("wf1", "wf2", "wf3"):
        m = index[(symbol, family, part)]["metrics"]
        development = development and m["completed"] >= 100 and above(m["mean_net_R"]) and above(m["profit_factor"], 1) and clean(part)
    m, pf, week = row["metrics"], row["profit_factor_inference"], row["weekly_inference"]
    day = row["day_inference"]
    extra = {}
    for ref in refs[1:]:
        pair = row["target_comparisons"][ref]
        extra[ref] = {
            "no_unknown_reference_outcomes": index[(symbol, ref, row["partition"])]["replay_audit"]["unknown"] == 0,
            "day_mean_CI_positive": ci_positive(pair["day"].get("mean_net_R_ci95")),
            "day_advantage_CI_positive": ci_positive(pair["day"].get("baseline_difference_ci95")),
            "week_mean_CI_positive": ci_positive(pair["week"].get("weekly_mean_net_R_ci95")),
            "week_advantage_CI_positive": ci_positive(pair["week"].get("weekly_difference_ci95")),
        }
    accepted = all((development, m["completed"] >= 1000, m["active_days"] >= 60,
                    above(m["profit_factor"], 1), above(m["mean_net_R"]), clean(row["partition"]),
                    ci_positive(pf["day_profit_factor_ci95"], 1), ci_positive(pf["weekly_profit_factor_ci95"], 1),
                    ci_positive(day["mean_net_R_ci95"]), ci_positive(week["weekly_mean_net_R_ci95"]),
                    ci_positive(day["baseline_difference_ci95"]), ci_positive(week["weekly_difference_ci95"]),
                    finite(row["holm_p"]) and 0 <= row["holm_p"] < .05,
                    all(t["metrics"]["completed"] >= 200 and above(t["metrics"]["mean_net_R"]) for t in row["thirds"]),
                    finite(m["closed_trade_max_drawdown"]) and m["closed_trade_max_drawdown"] <= .10,
                    m["equity_ruin"] is False, above(row["double_cost_metrics"]["mean_net_R"]),
                    all(all(checks.values()) for checks in extra.values())))
    return bool(development), bool(accepted), extra


def pinned_labels(audit, artifacts, parent_artifacts, issues, bounds):
    require(exact(artifacts["labels"], parent_artifacts["labels"]) and exact(artifacts["events"], parent_artifacts["events"]),
            "Training labels/events must be exact audited parent artifacts")
    actual = frame(audit, artifacts["labels"], features=True)
    events = read_events(audit.pin(artifacts["events"]["path"], artifacts["events"]["sha256"]))
    expected_records = []
    for e in events.to_dict("records"):
        expected_records.append({"signal_time": epoch(e["signal_time"]), "status": e["status"],
                                 "censored": bool(e["censored"]), "entry_time": epoch(e["entry_time"]),
                                 "net_R": None if pd.isna(e["net_R"]) else float(e["net_R"])})
    expected = R.conditional_labels(issues, expected_records)
    R.check_labels(audit, actual, expected, bounds[1])
    return actual


def reference_copy(row):
    copy = {k: v for k, v in row.items() if k not in ("qualification", "holm_p", "required_reference_checks", "target_comparisons")}
    copy["reference_origin"] = {"result_sha256": PARENT_PINS["results.json"],
                                **{k: row[k] for k in ("symbol", "variant", "partition")}}
    return copy


def verify(audit):
    R.safety()
    declaration_path = audit.pin(FOLDER / "declaration.json")
    result_path = audit.pin(FOLDER / "results.json")
    d, result = json.loads(declaration_path.read_text()), json.loads(result_path.read_text())
    for name, target in (("declaration", declaration_path), ("results", result_path)):
        require((FOLDER / (name + ".sha256")).read_text().strip() == sha(target), "Study sidecar changed")
    require(result["declaration_sha256"] == sha(declaration_path), "Result/declaration mismatch")
    require(d["stage"] == "frozen_nonlinear_conditional_region_reward" and result["stage"] == "executed_nonlinear_conditional_region_reward", "Exact study stages")
    for name, digest in d["code_sha256"].items(): audit.pin(name, digest)
    parent_data = {name: json.loads(audit.pin(PARENT / name, digest).read_text()) for name, digest in PARENT_PINS.items()}
    pdcl, parent, parent_audit = (parent_data[n] for n in ("declaration.json", "results.json", "independent_audit.json"))
    require(parent_audit["passed"] is True and not parent_audit["errors"] and parent_audit["models_checked"] == 16,
            "Prior independent conditional training/path audit required")
    for name in ("scripts/verify_region_reward_study.py", "scripts/verify_hybrid_event_study.py", "scripts/verify_zone_study.py"):
        audit.pin(name, parent_audit["input_sha256"][name])
    audit.pin(__file__); audit.pin(FOLDER / "test_review_study.py")
    parent_refs = [r for r in parent["rows"] if r["variant"] in FAMILIES[:3]]
    expected_inputs = {
        "parent_declaration_sha256": PARENT_PINS["declaration.json"],
        "parent_result_sha256": PARENT_PINS["results.json"], "parent_audit_sha256": PARENT_PINS["independent_audit.json"],
        "fold_inputs": pdcl["inputs"]["fold_inputs"],
        "training_artifacts": {s: parent["symbols"][s]["folds"] for s in SYMBOLS}, "reference_rows": parent_refs}
    require(exact(d["inputs"], expected_inputs), "Exact parent training/cache/reference manifest")
    require(exact(d["sources"], pdcl["sources"]), "Original source population changed")
    audit.pin(d["sources"]["audit_path"], d["sources"]["audit_sha256"])
    for obj in (d, result):
        require(exact(obj["safety"], R.safety()) and obj["QUALIFIED"] is False, "All live and qualification flags must remain false")
    require(result["fresh_out_of_sample"] is False and result["known_history_adaptive"] is True, "Exposed history cannot become fresh")
    spec, old = d["specification"], pdcl["specification"]
    fixed = {"symbols": list(SYMBOLS), "families": list(FAMILIES), "new_families": list(NEW),
             "folds": old["folds"], "parameters": PARAMETERS, "region_config": R.FIXED_CONFIG,
             "target": "conditional_completed_original_quote_CLOCK_region_net_R",
             "learner": "fixed_existing_histogram_boost", "labels_recomputed": False, "features_recomputed": False,
             "reference_models_refitted": False, "reference_statistics_redrawn": False,
             "minimum_completed_training_labels": 1000, "positive_score_required": True,
             "training_cutoff_population": "completed_training_CLOCK_region_paths_only",
             "region_offsets_raw_atr": [.45, .55], "region_invalidation_extra_raw_atr": .25,
             "references": {k: list(v) for k, v in REFERENCES.items()}, "reference_comparison_is_causal": False,
             "bootstrap_repeats": 9999, "bootstrap_seed": 20261008, "joint_heldout_Holm_family": 8,
             "point_criterion": "finite PF > 1 AND finite mean_net_R > 0",
             "completed_target_per_symbol_model_period": 1000, "active_days_target": 60,
             "known_history_adaptive": True, "fresh_out_of_sample": False, "QUALIFIED": False}
    for key, value in fixed.items(): require(exact(spec[key], value), "Fixed study specification: " + key)
    names, cfg = old["feature_names"], R.FIXED_CONFIG
    require(len(names) == len(set(names)) == 44, "Exact ordered44 features")
    rows = result["rows"]
    index = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    cells = {(s, f, p) for s in SYMBOLS for f in FAMILIES for p in PARTS}
    require(len(rows) == 60 and set(index) == cells, "All60 unique result cells")
    for row in parent_refs:
        key = (row["symbol"], row["variant"], row["partition"])
        require(exact(index[key], reference_copy(row)), "Frozen reference row differs: " + str(key))
    ledgers, models_checked, issued_count, paths_count = {}, 0, 0, 0
    label_rows, labels_completed, source_rows, source_gaps = 0, 0, 0, 0
    for symbol in SYMBOLS:
        print(symbol + ": independently decoding pinned original quotes", flush=True)
        times, prices = R.decode_sources(audit, d["sources"]["symbols"][symbol])
        source_rows += len(times); source_gaps += int(np.maximum(np.diff(times) - 1, 0).sum())
        for part in ("wf1", "wf2", "wf3", "final_test", "later180"):
            for family in FAMILIES[:3]:
                row = index[(symbol, family, part)]
                for kind in ("signals", "events", "ledger"):
                    a = row["artifacts"][kind]; audit.pin(a["path"], a["sha256"])
                a = row["artifacts"]["ledger"]
                ledgers[(symbol, family, part)] = read_events(audit.pin(a["path"], a["sha256"]))
        for fold, contract in spec["folds"].items():
            manifest = d["inputs"]["fold_inputs"][symbol][fold]
            raw, hybrid, _ = R.load_features(audit, manifest, {"features": manifest["features"], "fit": manifest["timed_fit"]}, names)
            positions, issues = clock_rows(raw, hybrid, *contract["train"])
            artifacts = result["symbols"][symbol]["folds"][fold]
            prior = d["inputs"]["training_artifacts"][symbol][fold]
            require(exact(artifacts["parent_models"], prior["models"]), "Exact parent training model identity")
            prior_model = json.loads(audit.pin(prior["models"]["path"], prior["models"]["sha256"]).read_text())
            labels = pinned_labels(audit, artifacts, prior, issues, contract["train"])
            label_rows += len(labels); labels_completed += int(labels.completed.sum())
            state = json.loads(audit.pin(artifacts["models"]["path"], artifacts["models"]["sha256"]).read_text())
            require(exact(state["training_audit"], prior_model["training_audit"]), "Unchanged parent training replay audit")
            if state["model_metadata"] is not None:
                require(exact(state["model_metadata"]["clock"], prior_model["model_metadata"]["clock"]), "Unchanged training clock accounting")
            models = check_model(audit, state, labels, raw, hybrid, positions, names, symbol, contract["train"])
            models_checked += len(models)
            require(artifacts["status"] == state["status"], "Fold status differs from model state")
            for part, bounds in contract["evaluate"].items():
                positions, issues = clock_rows(raw, hybrid, *bounds)
                for family, features in zip(NEW, (raw, hybrid), strict=True):
                    row = index[(symbol, family, part)]
                    require(exact([row["start"], row["end_exclusive"]], bounds) and row["endpoint"] == "REGION", "Exact evaluation region interval")
                    require(row["fit_status"] == state["status"] and row["baseline_variant"] == "CLOCK", "Evaluation fit/reference identity")
                    score = predict(features.loc[positions, names].to_numpy(float), models[family], names) if family in models else np.zeros(len(issues))
                    chosen = (score >= models[family]["threshold"]) & (score > 0) if family in models else np.zeros(len(issues), bool)
                    signals = frame(audit, row["artifacts"]["signals"])
                    array(audit, R.timestamps(signals.signal_time).asi8, issues[chosen].asi8, "independent boost selected clock")
                    array(audit, signals.atr, raw.atr.iloc[positions[chosen]], "raw execution ATR")
                    array(audit, signals.signal_close, raw.close.iloc[positions[chosen]], "original signal price")
                    array(audit, signals.side, np.full(int(chosen.sum()), 1 if symbol == "BOOM600" else -1), "native signal side")
                    array(audit, signals.variant, np.full(len(signals), family + "_SPIKE"), "new signal family")
                    array(audit, pd.to_numeric(signals.score, errors="raise"), score[chosen], "independent boost score", close=True)
                    issued_count += len(signals)
                    expected = R.oracle_events(times, prices, signals, symbol, family, bounds, cfg)
                    a = row["artifacts"]["events"]; events = read_events(audit.pin(a["path"], a["sha256"]))
                    R.check_events(audit, events, expected, row["replay_audit"], cfg, symbol + part + family)
                    a = row["artifacts"]["ledger"]; ledger = read_events(audit.pin(a["path"], a["sha256"]))
                    R.check_ledger(audit, ledger, events, expected, symbol + part + family)
                    paths_count += len(expected); ledgers[(symbol, family, part)] = ledger
                print(symbol + "/" + part + ": independent scores and actual quote paths checked", flush=True)
            del raw, hybrid
        del times, prices
    audit.equal(source_rows, 62380130, "native source quote count")
    audit.equal(source_gaps, 670, "native missing seconds preserved")
    for symbol in SYMBOLS:
        for family in FAMILIES:
            row = index[(symbol, family, "walk_forward_combined")]
            union = pd.concat([ledgers[(symbol, family, p)] for p in ("wf1", "wf2", "wf3")], ignore_index=True)
            if family in NEW:
                a = row["artifacts"]["ledger"]; saved = read_events(audit.pin(a["path"], a["sha256"]))
                R.check_reference(audit, saved, union, "separately replayed fold union")
                for part in ("wf1", "wf2", "wf3"):
                    require(exact(row["artifacts"]["folds"][part], index[(symbol, family, part)]["artifacts"]), "Union artifact lineage")
                    require(exact(row["replay_audit"]["component_folds"][part], index[(symbol, family, part)]["replay_audit"]), "Union occupancy audit lineage")
                require(row["replay_audit"]["independent_replays_at_frozen_partition_boundaries"] is True, "Separate fold occupancies required")
            audit.equal(row["replay_audit"]["unknown"], sum(index[(symbol, family, p)]["replay_audit"]["unknown"] for p in ("wf1", "wf2", "wf3")), "Combined unknown count")
            ledgers[(symbol, family, "walk_forward_combined")] = union
    held, pvalues, comparisons = [], [], 0
    for row in rows:
        symbol, family, part = row["symbol"], row["variant"], row["partition"]
        ledger = ledgers[(symbol, family, part)]
        metric_checks(audit, row, ledger, cfg, 9999, 20261008)
        check_secondary(audit, row, ledger)
        if family not in NEW:
            continue
        require(set(row["target_comparisons"]) == set(REFERENCES[family]) - {"CLOCK"}, "Every required reference including fold union")
        joint = []
        for reference in REFERENCES[family]:
            baseline = ledgers[(symbol, reference, part)]
            for week in (False, True):
                saved = row["weekly_inference" if week else "day_inference"] if reference == "CLOCK" else row["target_comparisons"][reference]["week" if week else "day"]
                values = inference_oracle(ledger, baseline, row["start"], row["end_exclusive"], 9999, 20261008, week)
                joint.append(R.compare_inference(audit, saved, values, week, symbol + family + part + reference))
                comparisons += 1
        if part in ("final_test", "later180"):
            held.append(row); pvalues.append(max(joint))
    require(len(held) == 8, "Exact8 new held-out comparisons")
    for row, value in zip(held, R.holm(pvalues), strict=True):
        audit.equal(row["holm_p"], float(value), "Independent joint8 Holm")
        development, passed, refs = gate_outcome(row, index)
        q = row["qualification"]
        audit.equal(q["development_eligible"], development, "Independent development conjunction")
        audit.equal(q["historical_criteria_passed"], passed, "Independent stable-positive conjunction")
        require(exact(row["required_reference_checks"], refs), "Independent required-reference checks")
        require(q["qualified"] is False and q["fresh_out_of_sample"] is False, "Never qualify reused history")
        audit.equal(not q["historical_rejection_reasons"], passed, "Rejection reasons agree with gate")
    for name, digest in tuple(audit.pins.items()): audit.equal(sha(ROOT / name), digest, "Final stable bytes " + name)
    return {
        "models_checked": models_checked, "training_clock_labels_checked_including_overlapping_prefixes": label_rows,
        "completed_training_labels_checked_including_overlapping_prefixes": labels_completed,
        "issued_scores_checked": issued_count, "evaluation_region_dispositions_checked": paths_count,
        "native_quote_rows_checked": source_rows, "missing_seconds_preserved": source_gaps,
        "cells_checked": len(rows), "frozen_reference_cells_checked": 36,
        "paired_day_week_comparisons_checked": comparisons, "heldout_gates_checked": len(held),
        "declaration_sha256": sha(declaration_path), "result_sha256": sha(result_path),
        "scope": "exact_parent_cache_labels_references_training_matrix_target_model_identity_training_bins_routed_tree_residual_arithmetic_cutoff_independent_prediction_issuance_native_tick_paths_netR_point_metrics_day_week_CIs_drawdown_thirds_Holm_development_and_stable_positive_gates",
        "not_independently_recomputed": ["full_optimal_split_search_and_tie_optimality", "original44_feature_formulas_and_intrinsic_validity", "parent_training_quote_paths_this_run", "reference_quote_paths_this_run"],
        "parent_training_and_reference_paths_basis": "exact byte identity to pinned prior independently audited artifacts",
        "fresh_out_of_sample": False, "known_history_adaptive": True,
        "broker_execution": "NOT TESTED", "measured_historical_costs": "NOT TESTED", "prospective_paper": "NOT TESTED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FOLDER / "independent_audit.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(FOLDER), "-p", "test_review_study.py", "-v"], cwd=ROOT).returncode
    output = Path(args.output).resolve(); output.relative_to(ROOT)
    require(not output.exists() and not output.with_suffix(".sha256").exists(), "Exclusive audit output")
    audit, detail = Audit(), {}
    try:
        detail = verify(audit)
    except Exception as error:
        audit.errors.append(f"{type(error).__name__}: {error}")
    report = {"stage": "independent_nonlinear_conditional_region_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
              "passed": not audit.errors, "checks": audit.checks, "errors": audit.errors,
              "input_sha256": audit.pins, "verifier_sha256": sha(__file__),
              "safety": dict.fromkeys(FLAGS, False), "QUALIFIED": False, **detail}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with output.with_suffix(".sha256").open("x") as stream: stream.write(sha(output) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
