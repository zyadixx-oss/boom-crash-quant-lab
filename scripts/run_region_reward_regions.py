#!/usr/bin/env python3
"""Freeze/run paired region-filled reward targets against frozen timed targets."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.jump_learning import PairedRidgeFit, spike_clock_signals, issue_paired_spike
from app.research.jump_multiframe import PairedMultiframeInputs
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.payoff_metrics import paired_inference
from app.research.zone_replay import ZoneReplayConfig
from app.research.zone_qualification import historical_zone_gate, holm_adjust
from app.research.spike_hunter import assert_offline
from scripts.region_reward_learning import (fit_region_reward, issue_region_reward, region_training_labels, InsufficientRegionTraining)
from scripts.run_representation_regions import csv, path
from scripts.run_hybrid_event_regions import folds
from scripts.run_spike_timed_study import weekly_inference
from scripts.run_zone_study import audited_sources, code_hashes, digest, json_value, metric_row, partitions, SYMBOLS
from scripts.representation_regions import numeric_ledger

FOLDER = ROOT / "docs/region_reward_20261008"
PARENT = ROOT / "docs/hybrid_event_regions_20261008"
PARENT_DECLARATION = "6ade87b7207754cc21468d68ab315c3f66f5b84bedc2e274907b453759e03af5"
PARENT_RESULT = "fb345c068a899157e15ec6393a72a7eeb5d78f932b7f7b7f0fac0640823b4abd"
FAMILIES = ("CLOCK", "RAW_TIMED", "HYBRID_TIMED", "RAW_REGION", "HYBRID_REGION")
NEW = ("RAW_REGION", "HYBRID_REGION")
EXTRA = ("scripts/region_reward_learning.py", "scripts/run_region_reward_regions.py",
    "scripts/representation_regions.py", "scripts/run_representation_regions.py", "scripts/run_hybrid_event_regions.py",
    "backend/tests/test_region_reward_learning.py", "backend/tests/test_region_reward_runner.py",
    "docs/region_reward_20261008/PROTOCOL.md", "docs/region_reward_20261008/premeasurement_tests.log",
    "docs/hybrid_event_regions_20261008/declaration.json", "docs/hybrid_event_regions_20261008/results.json",
    "docs/hybrid_event_regions_20261008/independent_audit.json", "scripts/verify_hybrid_event_study.py")


def save(path, value):
    relative = Path(path).resolve().relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:f.write(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    return {"path": str(relative), "sha256": digest(path)}


def inputs_manifest():
    if digest(PARENT / "declaration.json") != PARENT_DECLARATION or digest(PARENT / "results.json") != PARENT_RESULT:
        raise ValueError("Exact previous frozen experiment required")
    audit_path = PARENT / "independent_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["errors"] or audit["models_checked"] != 16:
        raise ValueError("Previous independent model/source/path audit must pass")
    parent = json.loads((PARENT / "results.json").read_text())
    return {"parent_declaration_sha256": PARENT_DECLARATION, "parent_result_sha256": PARENT_RESULT,
        "parent_audit_sha256": digest(audit_path),
        "fold_inputs": {s: {f: {"features": v["features"], "timed_fit": v["fit"]}
            for f, v in parent["symbols"][s]["folds"].items()} for s in SYMBOLS}}


def spec():
    return json_value({"symbols": SYMBOLS, "families": FAMILIES, "folds": folds(),
        "feature_names": FEATURE_NAMES, "features_recomputed": False,
        "new_training_target": "conditional_completed_CLOCK_region_net_R",
        "training_nonfills": "known_nan_reward_not_zero_trade", "unknown_paths": "unknown_nan",
        "training_behavior": "same_one_pending_open_CLOCK_region_stream; overlap_skips_not_labels",
        "minimum_completed_training_labels": 1000, "ridge_penalty": .1, "training_quantile": .75,
        "cutoff_population": "completed_training_CLOCK_region_paths_only", "positive_score_required": True,
        "region_config": asdict(ZoneReplayConfig()), "region_offsets_raw_atr": [.45, .55],
        "region_invalidation_extra_raw_atr": .25, "endpoints": ["REGION"],
        "references": {"RAW_REGION": ["CLOCK", "RAW_TIMED"],
                       "HYBRID_REGION": ["CLOCK", "HYBRID_TIMED", "RAW_REGION"]},
        "all_required_references": "positive_day_week_mean_and_advantage_CIs; no_unknown_heldout_or_development_outcomes",
        "bootstrap_repeats": 9999, "bootstrap_seed": 20261008, "joint_heldout_Holm_family": 8,
        "profit_factor_target": 1.5, "completed_target": 1000, "active_days_target": 60,
        "known_history_adaptive": True, "fresh_out_of_sample": False, "QUALIFIED": False,
        "conditional_fill_selection_is_not_a_causal_policy_effect": True, "no_post_result_rescue": True})


def freeze():
    assert_offline()
    if any((FOLDER / n).exists() for n in ("declaration.json", "execution_started.json", "results.json", "execution_failure.json")):
        raise ValueError("Existing study cannot be replaced")
    d = {"stage": "frozen_region_conditional_reward", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "specification": spec(), "inputs": inputs_manifest(), "sources": audited_sources(), "code_sha256": code_hashes(EXTRA),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "new_historical_targets_or_scores_computed": False, "safety": assert_offline(), "QUALIFIED": False}
    a = save(FOLDER / "declaration.json", d)
    with (FOLDER / "declaration.sha256").open("x") as f: f.write(a["sha256"] + "\n")
    return a


def frozen(expected):
    assert_offline()
    if digest(FOLDER / "declaration.json") != expected or (FOLDER / "declaration.sha256").read_text().strip() != expected:
        raise ValueError("Exact frozen declaration required")
    d = json.loads((FOLDER / "declaration.json").read_text())
    if (d["specification"] != spec() or d["inputs"] != inputs_manifest() or d["sources"] != audited_sources()
            or d["code_sha256"] != code_hashes(EXTRA)
            or d["runtime"] != {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}):
        raise ValueError("Frozen study lineage/code/runtime changed")
    return d


def read_pinned(artifact):
    p = ROOT / artifact["path"]
    raw = p.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != artifact["sha256"]: raise ValueError("Cached artifact changed")
    return p, raw


def load_inputs(manifest):
    frames = []
    for arm in ("RAW44", "HYBRID44"):
        p, raw = read_pinned(manifest["features"][arm])
        f = pd.read_csv(io.BytesIO(raw), compression="gzip", float_precision="round_trip")
        f.index = pd.DatetimeIndex(pd.to_datetime(f.pop("m5_open"), utc=True)).as_unit("ns")
        if list(f.columns) != [*FEATURE_NAMES, "feature_valid", "atr", "close"]:
            raise ValueError("Exact cached44 schema required")
        frames.append(f)
    raw, hybrid = frames
    if not raw.index.equals(hybrid.index): raise ValueError("Cached feature populations differ")
    _, payload = read_pinned(manifest["features"]["availability"])
    clock = pd.read_csv(io.BytesIO(payload), compression="gzip", float_precision="round_trip")
    clock.index = pd.DatetimeIndex(pd.to_datetime(clock.m5_open, utc=True)).as_unit("ns")
    available = pd.DataFrame({"raw_feature_valid": raw.feature_valid,
        "transformed_feature_valid": hybrid.feature_valid,
        "common_feature_valid": raw.feature_valid & hybrid.feature_valid,
        "raw_execution_atr": raw.atr, "transformed_feature_atr": hybrid.atr}, index=raw.index)
    # Only scheduled00/30 opportunities require a known run id. Nonclock rows
    # retain their exact feature validity but get no invented run identity.
    available["run_id"] = pd.array(clock.run_id.reindex(raw.index), dtype="Int64")
    for name in available.columns:
        try:
            pd.testing.assert_series_equal(available.loc[clock.index, name], clock[name],
                check_dtype=False, check_names=False, check_exact=True)
        except AssertionError as exc:
            raise ValueError("Saved scheduled availability differs: " + name) from exc
    p, payload = read_pinned(manifest["timed_fit"]); fit = json.loads(payload)
    if fit["status"] != "FITTED": raise ValueError("Complete frozen timed-reference pair required")
    return PairedMultiframeInputs(raw, hybrid, available), PairedRidgeFit(fit["models"], fit["model_metadata"])


def load_quotes(sources):
    expected = sum(s["rows"] for s in sources); times = np.empty(expected, np.int64); prices = np.empty(expected, float)
    cursor, last = 0, None
    for source in sources:
        _, payload = read_pinned(source)
        f = pd.read_csv(io.BytesIO(payload), dtype={"epoch": np.int64, "quote": float})
        if list(f.columns) != ["epoch", "quote"] or len(f) != source["rows"]: raise ValueError("Source schema/count changed")
        t, p = f.epoch.to_numpy(), f.quote.to_numpy(); start = pd.Timestamp(source["date"], tz="UTC").value // 10**9
        if (np.any(np.diff(t) <= 0) or (len(t) and last is not None and t[0] <= last)
                or np.any(t < start) or np.any(t >= start + 86400) or not np.isfinite(p).all() or np.any(p <= 0)):
            raise ValueError("Native source chronology/price changed")
        n = len(f);times[cursor:cursor+n] = t * 10**9;prices[cursor:cursor+n] = p;cursor += n
        if len(t): last = int(t[-1])
    return pd.DataFrame({"quote": prices}, index=pd.DatetimeIndex(times, tz="UTC"), copy=False)


def references(family):
    return ("CLOCK", "RAW_TIMED") if family == "RAW_REGION" else ("CLOCK", "HYBRID_TIMED", "RAW_REGION")


def positive_interval(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
        and 0 < value[0] <= value[1])


def rows_for(symbol, part, bounds, ledgers, audits, artifacts, statuses):
    rows = []
    for family in FAMILIES:
        row = metric_row(symbol, family, part, bounds, ledgers[family], ledgers["CLOCK"], audits[family], artifacts.get(family, {}))
        row.update(endpoint="REGION", baseline_variant="CLOCK", fit_status=statuses[family])
        if family in NEW:
            row["target_comparisons"] = {r: {"day": paired_inference(ledgers[family], ledgers[r], *bounds, repeats=9999, seed=20261008),
                "week": weekly_inference(ledgers[family], ledgers[r], *bounds, 9999, seed=20261008)} for r in references(family) if r != "CLOCK"}
        rows.append(row)
    return rows


def attach_gates(rows):
    index = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    held = [r for r in rows if r["variant"] in NEW and r["partition"] in ("final_test", "later180")]
    if len(held) != 8: raise ValueError("Exact eight held-out comparisons required")
    for r in held:
        if set(r["target_comparisons"]) != set(references(r["variant"])) - {"CLOCK"}:
            raise ValueError("All required target references must be present")
    def prob(r):
        values = [r["day_inference"]["p"], r["weekly_inference"]["weekly_p"]]
        for v in r["target_comparisons"].values(): values += [v["day"]["p"], v["week"]["weekly_p"]]
        return max(values)
    for r, corrected in zip(held, holm_adjust([prob(r) for r in held]), strict=True):
        symbol, family, part = r["symbol"], r["variant"], r["partition"]
        def development(p):
            x = index[(symbol, family, p)]
            unknown = max(index[(symbol, ref, p)]["replay_audit"]["unknown"] for ref in references(family))
            return {**x["metrics"], "unknown_regions": x["replay_audit"]["unknown"], "control_unknown_regions": unknown}
        r["holm_p"] = corrected
        r["qualification"] = historical_zone_gate(metrics={**r["metrics"], **r["day_inference"]}, pf=r["profit_factor_inference"],
            weekly=r["weekly_inference"], folds=[development(f) for f in ("wf1", "wf2", "wf3")], validation=development("walk_forward_combined"),
            thirds=[t["metrics"] for t in r["thirds"]], doubled_cost=r["double_cost_metrics"], unknown_regions=r["replay_audit"]["unknown"],
            baseline_unknown_regions=max(index[(symbol, ref, part)]["replay_audit"]["unknown"] for ref in references(family)), holm_p=corrected)
        checks = {}
        for ref, inference in r["target_comparisons"].items():
            day, week = inference["day"], inference["week"]
            checks[ref] = {
                "no_unknown_reference_outcomes": index[(symbol, ref, part)]["replay_audit"]["unknown"] == 0,
                "day_mean_CI_positive": positive_interval(day.get("mean_net_R_ci95")),
                "day_advantage_CI_positive": positive_interval(day.get("baseline_difference_ci95")),
                "week_mean_CI_positive": positive_interval(week.get("weekly_mean_net_R_ci95")),
                "week_advantage_CI_positive": positive_interval(week.get("weekly_difference_ci95"))}
        r["required_reference_checks"] = checks
        q = r["qualification"]
        q["historical_rejection_reasons"] += [f"required_reference_{ref}_{name}" for ref, values in checks.items() for name, passed in values.items() if not passed]
        q["historical_criteria_passed"] = not q["historical_rejection_reasons"]


def execute(expected):
    d = frozen(expected)
    save(FOLDER / "execution_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "declaration_sha256": expected, "safety": assert_offline()})
    try:
        result = {"stage": "executed_region_conditional_reward", "declaration_sha256": expected, "rows": [], "symbols": {},
                  "safety": assert_offline(), "QUALIFIED": False, "fresh_out_of_sample": False, "known_history_adaptive": True}
        for symbol in SYMBOLS:
            print(symbol + ": decoding immutable original sources", flush=True)
            ticks = load_quotes(d["sources"]["symbols"][symbol]); cache = {}; result["symbols"][symbol] = {"folds": {}}
            for fold, contract in folds().items():
                inputs, timed_fit = load_inputs(d["inputs"]["fold_inputs"][symbol][fold])
                start, end = contract["train"];clock = spike_clock_signals(inputs, symbol, start, end).signals
                events, _, audit = path(ticks, clock, symbol, "CLOCK", "REGION", (start, end))
                labels = region_training_labels(clock, events)
                label_artifact = csv(FOLDER / f"{symbol}_{fold}_training_labels.csv.gz", labels.reset_index(names="signal_time"))
                event_artifact = csv(FOLDER / f"{symbol}_{fold}_training_events.csv.gz", events)
                fit, error = None, None
                try: fit = fit_region_reward(inputs, labels, symbol, start, end)
                except InsufficientRegionTraining as exc: error = str(exc)
                state = {"status": "FITTED" if fit else "NOT_FIT_INSUFFICIENT_REGION_LABELS", "error": error,
                    "models": fit.models if fit else None, "model_metadata": fit.metadata if fit else None,
                    "training_dispositions": dict(Counter(labels.status)), "completed": int(labels.completed.sum()),
                    "known_nonfill": int(labels.known_nonfill.sum()), "unknown_path": int(labels.unknown_path.sum()),
                    "exposure_skipped": int(labels.exposure_skipped.sum()), "training_audit": audit, "safety": assert_offline(), "QUALIFIED": False}
                model_artifact = save(FOLDER / f"{symbol}_{fold}_models.json", state)
                result["symbols"][symbol]["folds"][fold] = {"models": model_artifact, "labels": label_artifact, "events": event_artifact, "status": state["status"]}
                statuses = {"CLOCK": "UNFITTED_REFERENCE", "RAW_TIMED": "FROZEN_PRIOR_TIMED_TARGET", "HYBRID_TIMED": "FROZEN_PRIOR_TIMED_TARGET",
                    **{v: state["status"] for v in NEW}}
                print(f"{symbol}/{fold}: conditional region fit {state['status']}; completed={state['completed']}", flush=True)
                for part, bounds in contract["evaluate"].items():
                    previous = issue_paired_spike(inputs, timed_fit, symbol, *bounds).signals
                    signals = {"CLOCK": previous["CLOCK"], "RAW_TIMED": previous["RAW44"].copy(), "HYBRID_TIMED": previous["TRANSFORMED44"].copy()}
                    for v in ("RAW_TIMED", "HYBRID_TIMED"): signals[v]["variant"] = v + "_SPIKE"
                    signals.update(issue_region_reward(inputs, fit, symbol, *bounds) if fit else {v: signals["CLOCK"].iloc[:0].copy() for v in NEW})
                    ledgers, audits, artifacts = {}, {}, {}
                    for family in FAMILIES:
                        events, ledger, audit = path(ticks, signals[family], symbol, family, "REGION", bounds)
                        name = f"{symbol}_{part}_{family}"
                        artifacts[family] = {"signals": csv(FOLDER / f"{name}_signals.csv.gz", signals[family]),
                            "events": csv(FOLDER / f"{name}_events.csv.gz", events), "ledger": csv(FOLDER / f"{name}_ledger.csv.gz", ledger)}
                        ledgers[family], audits[family] = ledger, audit;cache[(part, family)] = ledger, audit
                    result["rows"] += rows_for(symbol, part, bounds, ledgers, audits, artifacts, statuses)
                    print(f"{symbol}/{part}: all five original-price region streams evaluated", flush=True)
            bounds = partitions()["wf1"][0], partitions()["wf3"][1]
            ledgers = {v: numeric_ledger(pd.concat([cache[(p, v)][0] for p in ("wf1", "wf2", "wf3")], ignore_index=True)) for v in FAMILIES}
            audits = {v: {"unknown": sum(cache[(p, v)][1]["unknown"] for p in ("wf1", "wf2", "wf3"))} for v in FAMILIES}
            result["rows"] += rows_for(symbol, "walk_forward_combined", bounds, ledgers, audits, {},
                {v: "UNFITTED_REFERENCE" if v == "CLOCK" else "SEE_FOLDS" for v in FAMILIES})
            del ticks, cache, inputs
        attach_gates(result["rows"])
        if len(result["rows"]) != 60: raise ValueError("Exact60 result cells required")
        frozen(expected);result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        a = save(FOLDER / "results.json", result)
        with (FOLDER / "results.sha256").open("x") as f:f.write(a["sha256"] + "\n")
        return a
    except Exception as error:
        save(FOLDER / "execution_failure.json", {"error_type": type(error).__name__, "error": str(error), "declaration_sha256": expected, "safety": assert_offline(), "QUALIFIED": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__);g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true");g.add_argument("--execute");args = p.parse_args()
    print(json.dumps(freeze() if args.freeze else execute(args.execute)), flush=True)


if __name__ == "__main__": main()
