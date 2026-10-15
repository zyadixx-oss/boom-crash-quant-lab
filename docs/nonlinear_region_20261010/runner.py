#!/usr/bin/env python3
"""One frozen nonlinear learner comparison on existing conditional region labels."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import io
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(Path(__file__).resolve().parent)]

from app.research.jump_learning import spike_clock_signals
from app.research.nonlinear_signal import DEFAULT_PARAMETERS
from app.research.payoff_metrics import paired_inference
from app.research.spike_hunter import assert_offline
from app.research.zone_replay import ZoneReplayConfig
from app.research.zone_qualification import holm_adjust
from scripts.region_reward_learning import InsufficientRegionTraining, region_training_labels
from scripts.representation_regions import numeric_ledger
from scripts.run_hybrid_event_regions import folds
from scripts.run_representation_regions import csv, path
from scripts.run_region_reward_regions import load_inputs, load_quotes, positive_interval
from scripts.run_spike_timed_study import weekly_inference
from scripts.run_zone_study import SYMBOLS, audited_sources, code_hashes, digest, json_value, metric_row, partitions
from learning import fit_region_boost, issue_region_boost
from stable_gate import stable_positive_gate

FOLDER = ROOT / "docs/nonlinear_region_20261010"
PARENT = ROOT / "docs/region_reward_20261008"
PARENT_DECLARATION = "5a5cd147eefc7d12ac83c8ef839c931292c5fb338e07a1f083001c893c5196e5"
PARENT_RESULT = "6e470b505d4f32ffb28600cd136ae886001f35e34631663553d8807b114d8115"
PARENT_AUDIT = "a6194dd9178d2a14a7dee4d317429a1aaaad28d3d3ce4c195d9378f8e588f25b"
FAMILIES = ("CLOCK", "RAW_REGION", "HYBRID_REGION", "RAW_BOOST_REGION", "HYBRID_BOOST_REGION")
NEW = FAMILIES[-2:]
PARTS = ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180")
EXTRA = (
    "scripts/region_reward_learning.py", "scripts/run_region_reward_regions.py",
    "scripts/representation_regions.py", "scripts/run_representation_regions.py", "scripts/run_hybrid_event_regions.py",
    "docs/nonlinear_region_20261010/runner.py", "docs/nonlinear_region_20261010/learning.py",
    "docs/nonlinear_region_20261010/stable_gate.py", "docs/nonlinear_region_20261010/PROTOCOL.md",
    "docs/nonlinear_region_20261010/test_learning.py", "docs/nonlinear_region_20261010/test_runner.py",
    "docs/nonlinear_region_20261010/premeasurement_tests.log",
)


def save(target, value):
    relative = Path(target).resolve().relative_to(ROOT.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    return {"path": str(relative), "sha256": digest(target)}


def read_pinned(artifact):
    target = (ROOT / artifact["path"]).resolve()
    target.relative_to(ROOT.resolve())
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise ValueError("Cached artifact changed")
    return target, payload


def references(family):
    if family == "RAW_BOOST_REGION":
        return ("CLOCK", "RAW_REGION")
    if family == "HYBRID_BOOST_REGION":
        return ("CLOCK", "HYBRID_REGION", "RAW_BOOST_REGION")
    raise ValueError("Only the two fixed new hypotheses have reference sets")


def inputs_manifest():
    for name, expected in (("declaration.json", PARENT_DECLARATION), ("results.json", PARENT_RESULT),
                           ("independent_audit.json", PARENT_AUDIT)):
        if digest(PARENT / name) != expected:
            raise ValueError("Exact parent declaration/result/audit required")
    parent_d = json.loads((PARENT / "declaration.json").read_text())
    parent_r = json.loads((PARENT / "results.json").read_text())
    audit = json.loads((PARENT / "independent_audit.json").read_text())
    if audit["passed"] is not True or audit["errors"] or audit["models_checked"] != 16:
        raise ValueError("Parent economic audit must pass")
    rows = [r for r in parent_r["rows"] if r["variant"] in FAMILIES[:3]]
    if len(rows) != 36 or len({(r["symbol"], r["variant"], r["partition"]) for r in rows}) != 36:
        raise ValueError("All 36 exact frozen reference cells required")
    return {
        "parent_declaration_sha256": PARENT_DECLARATION, "parent_result_sha256": PARENT_RESULT,
        "parent_audit_sha256": PARENT_AUDIT, "fold_inputs": parent_d["inputs"]["fold_inputs"],
        "training_artifacts": {s: {f: v for f, v in parent_r["symbols"][s]["folds"].items()} for s in SYMBOLS},
        "reference_rows": rows,
    }


def spec():
    return json_value({
        "symbols": SYMBOLS, "families": FAMILIES, "new_families": NEW, "folds": folds(),
        "learner": "fixed_existing_histogram_boost", "parameters": DEFAULT_PARAMETERS,
        "target": "conditional_completed_original_quote_CLOCK_region_net_R",
        "labels_recomputed": False, "features_recomputed": False,
        "reference_models_refitted": False, "reference_statistics_redrawn": False,
        "minimum_completed_training_labels": 1000, "positive_score_required": True,
        "training_cutoff_population": "completed_training_CLOCK_region_paths_only",
        "region_config": asdict(ZoneReplayConfig()), "region_offsets_raw_atr": [.45, .55],
        "region_invalidation_extra_raw_atr": .25, "references": {v: references(v) for v in NEW},
        "reference_comparison_is_causal": False, "bootstrap_repeats": 9999, "bootstrap_seed": 20261008,
        "joint_heldout_Holm_family": 8, "point_criterion": "finite PF > 1 AND finite mean_net_R > 0",
        "completed_target_per_symbol_model_period": 1000, "active_days_target": 60,
        "known_parent_CLOCK_unknowns_each_symbol": {"wf2": 3, "wf3": 2, "walk_forward_combined": 5,
            "final_test": 3, "later180": 8},
        "known_unknown_references_preclude_historical_qualification": True,
        "all_other_prior_development_uncertainty_stability_risk_gates_retained": True,
        "known_history_adaptive": True, "fresh_out_of_sample": False, "QUALIFIED": False,
        "CFD_execution": "NOT TESTED", "measured_historical_costs": "NOT TESTED", "cash_profit": "NOT TESTED",
        "no_post_result_rescue": True,
    })


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}


def freeze():
    assert_offline()
    if any((FOLDER / n).exists() for n in ("declaration.json", "execution_started.json", "results.json", "execution_failure.json")):
        raise ValueError("Existing study cannot be replaced")
    d = {"stage": "frozen_nonlinear_conditional_region_reward", "declared_utc": datetime.now(timezone.utc).isoformat(),
         "specification": spec(), "inputs": inputs_manifest(), "sources": audited_sources(),
         "code_sha256": code_hashes(EXTRA), "runtime": runtime(),
         "new_historical_targets_or_scores_computed": False, "safety": assert_offline(), "QUALIFIED": False}
    artifact = save(FOLDER / "declaration.json", d)
    with (FOLDER / "declaration.sha256").open("x") as target:
        target.write(artifact["sha256"] + "\n")
    return artifact


def frozen(expected):
    assert_offline()
    if digest(FOLDER / "declaration.json") != expected or (FOLDER / "declaration.sha256").read_text().strip() != expected:
        raise ValueError("Exact declaration and sidecar required")
    d = json.loads((FOLDER / "declaration.json").read_text())
    if (d["specification"] != spec() or d["inputs"] != inputs_manifest() or d["sources"] != audited_sources()
            or d["code_sha256"] != code_hashes(EXTRA) or d["runtime"] != runtime()
            or d["safety"] != assert_offline() or d["QUALIFIED"] is not False):
        raise ValueError("Frozen lineage/code/runtime/specification/safety changed")
    return d


def read_table(artifact):
    _, payload = read_pinned(artifact)
    result = pd.read_csv(io.BytesIO(payload), compression="gzip", float_precision="round_trip")
    if "rows" in artifact and len(result) != artifact["rows"]:
        raise ValueError("Pinned table row count differs")
    return result


def cached_labels(inputs, symbol, bounds, artifacts):
    clock = spike_clock_signals(inputs, symbol, *bounds).signals
    events = read_table(artifacts["events"])
    expected = region_training_labels(clock, events)
    saved = read_table(artifacts["labels"])
    saved.index = pd.DatetimeIndex(pd.to_datetime(saved.pop("signal_time"), utc=True)).as_unit("ns")
    saved["planned_end"] = pd.DatetimeIndex(pd.to_datetime(saved.planned_end, utc=True)).as_unit("ns")
    try:
        pd.testing.assert_frame_equal(saved, expected, check_exact=True, check_names=False)
    except AssertionError as error:
        raise ValueError("Saved complete conditional labels differ from exact shared clock/events") from error
    return saved


def reference_row(parent_row):
    row = deepcopy(parent_row)
    # Legacy PF1.5 judgments belong to the immutable parent, not this comparison.
    for key in ("qualification", "holm_p", "required_reference_checks", "target_comparisons"):
        row.pop(key, None)
    row["reference_origin"] = {"result_sha256": PARENT_RESULT,
        **{k: row[k] for k in ("symbol", "variant", "partition")}}
    return row


def new_rows(symbol, part, bounds, ledgers, audits, artifacts, statuses):
    rows = []
    for family in NEW:
        r = metric_row(symbol, family, part, bounds, ledgers[family], ledgers["CLOCK"], audits[family], artifacts[family])
        r.update(endpoint="REGION", baseline_variant="CLOCK", fit_status=statuses[family])
        r["target_comparisons"] = {
            ref: {"day": paired_inference(ledgers[family], ledgers[ref], *bounds, repeats=9999, seed=20261008),
                  "week": weekly_inference(ledgers[family], ledgers[ref], *bounds, 9999, seed=20261008)}
            for ref in references(family) if ref != "CLOCK"}
        rows.append(r)
    return rows


def attach_gates(rows):
    index = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    expected = {(s, v, p) for s in SYMBOLS for v in FAMILIES for p in PARTS}
    if len(rows) != 60 or set(index) != expected:
        raise ValueError("Exact 60 unique cells required")
    held = [r for r in rows if r["variant"] in NEW and r["partition"] in ("final_test", "later180")]
    if len(held) != 8:
        raise ValueError("Exact eight new held-out conjunctions required")
    for r in rows:
        if r["variant"] in NEW and set(r["target_comparisons"]) != set(references(r["variant"])) - {"CLOCK"}:
            raise ValueError("Every new cell including fold union requires all fixed reference comparisons")
    def probability(r):
        values = [r["day_inference"]["p"], r["weekly_inference"]["weekly_p"]]
        for value in r["target_comparisons"].values():
            values += [value["day"]["p"], value["week"]["weekly_p"]]
        return max(values)
    for r, corrected in zip(held, holm_adjust([probability(r) for r in held]), strict=True):
        symbol, family, part = r["symbol"], r["variant"], r["partition"]
        def development(p):
            x = index[(symbol, family, p)]
            unknown = max(index[(symbol, ref, p)]["replay_audit"]["unknown"] for ref in references(family))
            return {**x["metrics"], "unknown_regions": x["replay_audit"]["unknown"], "control_unknown_regions": unknown}
        r["holm_p"] = corrected
        q = stable_positive_gate(metrics={**r["metrics"], **r["day_inference"]}, pf=r["profit_factor_inference"],
            weekly=r["weekly_inference"], folds=[development(f) for f in ("wf1", "wf2", "wf3")],
            validation=development("walk_forward_combined"), thirds=[t["metrics"] for t in r["thirds"]],
            doubled_cost=r["double_cost_metrics"], unknown_regions=r["replay_audit"]["unknown"],
            baseline_unknown_regions=max(index[(symbol, ref, part)]["replay_audit"]["unknown"] for ref in references(family)), holm_p=corrected)
        checks = {}
        for ref, inference in r["target_comparisons"].items():
            day, week = inference["day"], inference["week"]
            checks[ref] = {
                "no_unknown_reference_outcomes": index[(symbol, ref, part)]["replay_audit"]["unknown"] == 0,
                "day_mean_CI_positive": positive_interval(day.get("mean_net_R_ci95")),
                "day_advantage_CI_positive": positive_interval(day.get("baseline_difference_ci95")),
                "week_mean_CI_positive": positive_interval(week.get("weekly_mean_net_R_ci95")),
                "week_advantage_CI_positive": positive_interval(week.get("weekly_difference_ci95")),
            }
        r["required_reference_checks"] = checks
        q["historical_rejection_reasons"] += [f"required_reference_{ref}_{name}" for ref, values in checks.items() for name, passed in values.items() if not passed]
        q["historical_criteria_passed"] = not q["historical_rejection_reasons"]
        r["qualification"] = q


def execute(expected):
    d = frozen(expected)
    save(FOLDER / "execution_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
         "declaration_sha256": expected, "safety": assert_offline()})
    try:
        result = {"stage": "executed_nonlinear_conditional_region_reward", "declaration_sha256": expected,
            "rows": [reference_row(r) for r in d["inputs"]["reference_rows"]], "symbols": {},
            "safety": assert_offline(), "QUALIFIED": False, "fresh_out_of_sample": False, "known_history_adaptive": True}
        reference_index = {(r["symbol"], r["variant"], r["partition"]): r for r in result["rows"]}
        for symbol in SYMBOLS:
            print(symbol + ": decoding pinned native quotes; no new acquisition", flush=True)
            ticks = load_quotes(d["sources"]["symbols"][symbol])
            cache = {}; result["symbols"][symbol] = {"folds": {}}
            for fold, contract in folds().items():
                inputs, _ = load_inputs(d["inputs"]["fold_inputs"][symbol][fold])
                prior = d["inputs"]["training_artifacts"][symbol][fold]
                labels = cached_labels(inputs, symbol, contract["train"], prior)
                _, payload = read_pinned(prior["models"])
                prior_model = json.loads(payload)
                fit, error = None, None
                try:
                    fit = fit_region_boost(inputs, labels, symbol, *contract["train"])
                except InsufficientRegionTraining as exc:
                    error = str(exc)
                state = {"status": "FITTED" if fit else "NOT_FIT_INSUFFICIENT_REGION_LABELS", "error": error,
                    "models": fit.models if fit else None, "model_metadata": fit.metadata if fit else None,
                    "training_dispositions": dict(Counter(labels.status)), "completed": int(labels.completed.sum()),
                    "known_nonfill": int(labels.known_nonfill.sum()), "unknown_path": int(labels.unknown_path.sum()),
                    "exposure_skipped": int(labels.exposure_skipped.sum()), "training_audit": prior_model["training_audit"],
                    "safety": assert_offline(), "QUALIFIED": False}
                models = save(FOLDER / f"{symbol}_{fold}_models.json", state)
                result["symbols"][symbol]["folds"][fold] = {"models": models, "labels": prior["labels"],
                    "events": prior["events"], "parent_models": prior["models"], "status": state["status"]}
                print(f"{symbol}/{fold}: fixed paired boost fit {state['status']}; labels={state['completed']}", flush=True)
                for part, bounds in contract["evaluate"].items():
                    clock = spike_clock_signals(inputs, symbol, *bounds).signals
                    signals = issue_region_boost(inputs, fit, symbol, *bounds) if fit else {v: clock.iloc[:0].copy() for v in NEW}
                    ledgers, audits, artifacts = {}, {}, {}
                    for family in FAMILIES[:3]:
                        row = reference_index[(symbol, family, part)]
                        ledgers[family] = numeric_ledger(read_table(row["artifacts"]["ledger"]))
                    for family in NEW:
                        events, ledger, audit = path(ticks, signals[family], symbol, family, "REGION", bounds)
                        name = f"{symbol}_{part}_{family}"
                        artifacts[family] = {"signals": csv(FOLDER / f"{name}_signals.csv.gz", signals[family]),
                            "events": csv(FOLDER / f"{name}_events.csv.gz", events),
                            "ledger": csv(FOLDER / f"{name}_ledger.csv.gz", ledger)}
                        ledgers[family], audits[family] = ledger, audit
                        cache[(part, family)] = ledger, audit, artifacts[family]
                    result["rows"] += new_rows(symbol, part, bounds, ledgers, audits, artifacts, {v: state["status"] for v in NEW})
                    for family in NEW:
                        m = result["rows"][-2 if family == NEW[0] else -1]["metrics"]
                        print(f"{symbol}/{part}/{family}: n={m['completed']} PF={m['profit_factor']}", flush=True)
                del inputs, fit, labels
            bounds = partitions()["wf1"][0], partitions()["wf3"][1]
            ledgers = {v: numeric_ledger(pd.concat([cache[(p, v)][0] for p in ("wf1", "wf2", "wf3")], ignore_index=True)) for v in NEW}
            audits = {v: {"unknown": sum(cache[(p, v)][1]["unknown"] for p in ("wf1", "wf2", "wf3")),
                "component_folds": {p: cache[(p, v)][1] for p in ("wf1", "wf2", "wf3")},
                "independent_replays_at_frozen_partition_boundaries": True} for v in NEW}
            artifacts = {v: {"ledger": csv(FOLDER / f"{symbol}_walk_forward_combined_{v}_ledger.csv.gz", ledgers[v]),
                "folds": {p: cache[(p, v)][2] for p in ("wf1", "wf2", "wf3")}} for v in NEW}
            for family in FAMILIES[:3]:
                ledgers[family] = numeric_ledger(pd.concat([
                    read_table(reference_index[(symbol, family, p)]["artifacts"]["ledger"])
                    for p in ("wf1", "wf2", "wf3")], ignore_index=True))
            result["rows"] += new_rows(symbol, "walk_forward_combined", bounds, ledgers, audits, artifacts, {v: "SEE_FOLDS" for v in NEW})
            del ticks, cache, ledgers
        attach_gates(result["rows"])
        frozen(expected)
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        artifact = save(FOLDER / "results.json", result)
        with (FOLDER / "results.sha256").open("x") as target:
            target.write(artifact["sha256"] + "\n")
        return artifact
    except Exception as error:
        save(FOLDER / "execution_failure.json", {"error_type": type(error).__name__, "error": str(error),
            "declaration_sha256": expected, "safety": assert_offline(), "QUALIFIED": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--execute")
    args = p.parse_args()
    print(json.dumps(freeze() if args.freeze else execute(args.execute)), flush=True)


if __name__ == "__main__":
    main()
