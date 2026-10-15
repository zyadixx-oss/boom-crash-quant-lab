#!/usr/bin/env python3
"""Freeze/run paired native-tail representation models and pre-issued regions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.research.jump_learning import (calibrate_training_detector, spike_clock_signals,
    fit_paired_spike_ridge, issue_paired_spike, InsufficientTrainingEvidenceError)
from app.research.jump_multiframe import causal_jump_multiframe_inputs
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.payoff_ticks import TickExitConfig, replay_ticks
from app.research.payoff_metrics import paired_inference
from app.research.zone_replay import ZoneReplayConfig, replay_zones, zone_trades
from app.research.zone_qualification import historical_zone_gate, holm_adjust
from app.research.spike_hunter import assert_offline
from scripts.representation_regions import build_paired_minutes, fixed_regions, training_labels, numeric_ledger
from scripts.run_spike_timed_study import weekly_inference
from scripts.run_zone_study import (DATA_START, OLD_START, OLD_END, LATER_END, SYMBOLS,
    audited_sources, code_hashes, digest, json_value, load_symbol, metric_row, partitions)

FOLDER = ROOT / "docs/representation_regions_20261008"
FEATURE_FOLDER = ROOT / "data/representation_regions_20261008"
FAMILIES = ("CLOCK", "RAW44", "TRANSFORMED44")
ENDPOINTS = ("REGION", "TIMED")
EXTRA = ("scripts/representation_regions.py", "scripts/run_representation_regions.py",
    "backend/tests/test_representation_region_adapter.py", "backend/tests/test_representation_region_runner.py",
    "backend/tests/test_jump_representation.py", "backend/tests/test_jump_multiframe.py", "backend/tests/test_jump_learning.py",
    "docs/representation_regions_20261008/PROTOCOL.md", "docs/representation_regions_20261008/premeasurement_tests.log")


def folds():
    p = partitions()
    return {**{f"fit{n}": {"train": [OLD_START, p[f"wf{i}"][0]], "evaluate": {f"wf{i}": p[f"wf{i}"]}}
        for n, i in ((40, 1), (50, 2), (60, 3))},
        "fit70": {"train": [OLD_START, p["final_test"][0]],
                  "evaluate": {k: p[k] for k in ("final_test", "later180")}}}


def spec():
    return json_value({"symbols": SYMBOLS, "families": FAMILIES, "endpoints": ENDPOINTS,
        "folds": folds(), "data_start": DATA_START, "feature_names": FEATURE_NAMES,
        "training_labels": "original_quote_TIMED_CLOCK_net_R_with_unknowns",
        "minimum_training_labels": 1000, "penalty": .1, "training_quantile": .75,
        "clock": "UTC00/30 closedM5", "purge_minutes": 31,
        "native_tail_detector": "native_log_increment>10*training_only_median_abs_consecutive_log_increment",
        "gap_policy": "reset both histories; no interpolation; unknown large-bar age preserved",
        "region_offsets_raw_atr": [.45, .55], "region_invalidation_extra_raw_atr": .25,
        "timed_config": asdict(TickExitConfig()), "region_config": asdict(ZoneReplayConfig()),
        "bootstrap_repeats": 9999, "bootstrap_seed": 20261008,
        "holm_heldout_family": 16, "transformed_conjunction_references": ["CLOCK", "RAW44"],
        "profit_factor_target": 1.5, "completed_target_per_symbol_model": 1000, "active_days_target": 60,
        "known_history_adaptive": True, "fresh_out_of_sample": False, "QUALIFIED": False,
        "no_post_result_rescue": True})


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f: f.write(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(frame)}


def freeze():
    assert_offline()
    if any((FOLDER / n).exists() for n in ("declaration.json", "execution_started.json", "results.json", "execution_failure.json")):
        raise ValueError("Existing frozen study cannot be overwritten")
    d = {"stage": "frozen_paired_representation_and_regions", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "specification": spec(), "sources": audited_sources(), "code_sha256": code_hashes(EXTRA),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "new_historical_features_or_labels_computed": False, "safety": assert_offline(), "QUALIFIED": False}
    artifact = save(FOLDER / "declaration.json", d)
    with (FOLDER / "declaration.sha256").open("x") as f: f.write(artifact["sha256"] + "\n")
    return artifact


def frozen(expected):
    assert_offline()
    p = FOLDER / "declaration.json"
    if digest(p) != expected or (FOLDER / "declaration.sha256").read_text().strip() != expected:
        raise ValueError("Exact frozen declaration SHA required")
    d = json.loads(p.read_text())
    if d["specification"] != spec() or d["code_sha256"] != code_hashes(EXTRA) or d["sources"] != audited_sources():
        raise ValueError("Frozen code/specification/source lineage changed")
    if d["runtime"] != {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}:
        raise ValueError("Frozen runtime changed")
    return d


def prepare(ticks, detector, end, symbol, fold):
    right = int(ticks.index.searchsorted(end))
    def chunks():
        for ordinal, left in enumerate(range(0, right, 86400)):
            yield ticks.iloc[left:min(left + 86400, right)]
            if ordinal % 30 == 0: print(f"{symbol}/{fold}: transformed native chunk{ordinal+1}", flush=True)
    bars, summary = build_paired_minutes(chunks(), detector, start=DATA_START, end=end)
    inputs = causal_jump_multiframe_inputs(bars, "boom" if symbol == "BOOM600" else "crash")
    artifacts = {}
    for family, frame in (("RAW44", inputs.raw), ("TRANSFORMED44", inputs.transformed)):
        chosen = frame.loc[:, [*FEATURE_NAMES, "feature_valid", "atr", "close"]].reset_index(names="m5_open")
        artifacts[family] = csv(FEATURE_FOLDER / f"{symbol}_{fold}_{family}_features.csv.gz", chosen)
    available = inputs.availability.copy()
    available["m5_open"] = available.index
    available["issue_time"] = available.index + pd.Timedelta(minutes=5)
    available["raw_large_bar_age_known"] = np.isfinite(inputs.raw.log1p_large_bar_age)
    available["transformed_large_bar_age_known"] = np.isfinite(inputs.transformed.log1p_large_bar_age)
    clock = available.issue_time.dt.minute.isin((0, 30))
    artifacts["availability"] = csv(FOLDER / f"{symbol}_{fold}_availability.csv.gz", available.loc[clock])
    summary.update(common_feature_rows=int(inputs.availability.common_feature_valid.sum()),
        raw_feature_rows=int(inputs.availability.raw_feature_valid.sum()),
        transformed_feature_rows=int(inputs.availability.transformed_feature_valid.sum()),
        raw_large_bar_age_known_rows=int(np.isfinite(inputs.raw.log1p_large_bar_age).sum()),
        transformed_large_bar_age_known_rows=int(np.isfinite(inputs.transformed.log1p_large_bar_age).sum()))
    return inputs, summary, artifacts


def path(ticks, signals, symbol, family, endpoint, bounds):
    if endpoint == "TIMED":
        ledger, audit = replay_ticks(ticks, signals.to_dict("records"), TickExitConfig(), *bounds, purge_minutes=31)
        ledger = numeric_ledger(ledger)
        unknown = audit["missing_entry"] + audit["censored"]
        return ledger, ledger, {**audit, "unknown": unknown}
    zones = fixed_regions(signals, symbol=symbol, variant=family)
    events, audit = replay_zones(ticks, zones, ZoneReplayConfig(), *bounds)
    return events, numeric_ledger(zone_trades(events)), audit


def evaluate(ticks, signals, symbol, fold, part, bounds, statuses, output, cache):
    for endpoint in ENDPOINTS:
        ledgers, audits, artifacts = {}, {}, {}
        for family in FAMILIES:
            offers = csv(FOLDER / f"{symbol}_{part}_{family}_signals.csv.gz", signals[family]) if endpoint == ENDPOINTS[0] else None
            events, ledger, audit = path(ticks, signals[family], symbol, family, endpoint, bounds)
            name = f"{symbol}_{part}_{endpoint}_{family}"
            artifacts[family] = {"events": csv(FOLDER / f"{name}_events.csv.gz", events),
                "ledger": csv(FOLDER / f"{name}_ledger.csv.gz", ledger)}
            if offers: artifacts[family]["signals"] = offers
            ledgers[family], audits[family] = ledger, audit
            cache[(endpoint, part, family)] = (ledger, audit)
        for family in FAMILIES:
            row = metric_row(symbol, family, part, bounds, ledgers[family], ledgers["CLOCK"], audits[family], artifacts[family])
            row.update(endpoint=endpoint, fold=fold, baseline_variant="CLOCK", fit_status=statuses[family],
                       common_availability_comparison=True)
            if family == "TRANSFORMED44":
                row["representation_comparison"] = {
                    "baseline_variant": "RAW44", "day": paired_inference(ledgers[family], ledgers["RAW44"], *bounds, repeats=9999, seed=20261008),
                    "week": weekly_inference(ledgers[family], ledgers["RAW44"], *bounds, 9999, seed=20261008)}
            output["rows"].append(row)
        print(f"{symbol}/{part}/{endpoint}: fixed original-quote paths evaluated", flush=True)


def attach_gates(rows):
    index = {(r["symbol"], r["endpoint"], r["variant"], r["partition"]): r for r in rows}
    heldout = [r for r in rows if r["variant"] != "CLOCK" and r["partition"] in ("final_test", "later180")]
    if len(heldout) != 16: raise ValueError("Incomplete declared16-comparison family")
    def p(row):
        values = [row["day_inference"]["p"], row["weekly_inference"]["weekly_p"]]
        if "representation_comparison" in row:
            values += [row["representation_comparison"]["day"]["p"], row["representation_comparison"]["week"]["weekly_p"]]
        return max(values)
    for row, corrected in zip(heldout, holm_adjust([p(r) for r in heldout]), strict=True):
        sym, ep, family, part = (row[k] for k in ("symbol", "endpoint", "variant", "partition"))
        def development(partition):
            r, b = index[(sym, ep, family, partition)], index[(sym, ep, "CLOCK", partition)]
            return {**r["metrics"], "unknown_regions": r["replay_audit"]["unknown"],
                    "control_unknown_regions": b["replay_audit"]["unknown"]}
        baseline = index[(sym, ep, "CLOCK", part)]
        row["qualification"] = historical_zone_gate(metrics={**row["metrics"], **row["day_inference"]},
            pf=row["profit_factor_inference"], weekly=row["weekly_inference"],
            folds=[development(f) for f in ("wf1", "wf2", "wf3")], validation=development("walk_forward_combined"),
            thirds=[t["metrics"] for t in row["thirds"]], doubled_cost=row["double_cost_metrics"],
            unknown_regions=row["replay_audit"]["unknown"], baseline_unknown_regions=baseline["replay_audit"]["unknown"], holm_p=corrected)
        row["holm_p"] = corrected


def execute(expected):
    declaration = frozen(expected)
    save(FOLDER / "execution_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": expected, "safety": assert_offline()})
    try:
        output = {"stage": "executed_paired_representation_and_regions", "declaration_sha256": expected,
            "safety": assert_offline(), "QUALIFIED": False, "known_history_adaptive": True,
            "fresh_out_of_sample": False, "rows": [], "symbols": {}}
        for symbol in SYMBOLS:
            dataset, ticks = load_symbol(declaration["sources"]["symbols"][symbol])
            output["symbols"][symbol] = {"source_summary": dataset.summary, "folds": {}}
            del dataset
            cache = {}
            for fold, contract in folds().items():
                train_start, train_end = contract["train"]
                eval_end = max(b[1] for b in contract["evaluate"].values())
                save(FOLDER / f"{symbol}_{fold}_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                    "train_start": train_start, "train_end": train_end, "feature_end_exclusive": eval_end,
                    "declaration_sha256": expected, "safety": assert_offline()})
                print(f"{symbol}/{fold}: calibrating on training quotes only", flush=True)
                calibration = calibrate_training_detector(ticks, 1 if symbol == "BOOM600" else -1, train_start, train_end)
                calibration_artifact = save(FOLDER / f"{symbol}_{fold}_detector.json", calibration.metadata)
                print(f"{symbol}/{fold}: training-only detector frozen", flush=True)
                inputs, summary, feature_artifacts = prepare(ticks, calibration.detector, eval_end, symbol, fold)
                clock = spike_clock_signals(inputs, symbol, train_start, train_end)
                train_ledger, train_audit = replay_ticks(ticks, clock.signals.to_dict("records"), TickExitConfig(), train_start, train_end, purge_minutes=31)
                labels = training_labels(clock.signals, numeric_ledger(train_ledger))
                label_artifact = csv(FOLDER / f"{symbol}_{fold}_training_labels.csv.gz", labels.reset_index(names="signal_time"))
                fit, fit_error = None, None
                try: fit = fit_paired_spike_ridge(inputs, labels, symbol, train_start, train_end)
                except InsufficientTrainingEvidenceError as error: fit_error = str(error)
                fit_state = {"status": "FITTED" if fit else "NOT_FIT_INSUFFICIENT_COMMON_LABELS",
                    "error": fit_error, "training_clock": clock.metadata, "training_path_audit": train_audit,
                    "training_labels": len(labels), "training_completed": int(labels.completed.sum()),
                    "training_unknown": int((~labels.completed).sum()), "models": fit.models if fit else None,
                    "model_metadata": fit.metadata if fit else None, "safety": assert_offline(), "QUALIFIED": False}
                fit_artifact = save(FOLDER / f"{symbol}_{fold}_model.json", fit_state)
                output["symbols"][symbol]["folds"][fold] = {"preparation": summary,
                    "fit": fit_artifact, "status": fit_state["status"], "detector": calibration_artifact,
                    "features": feature_artifacts, "labels": label_artifact}
                statuses = {"CLOCK": "UNFITTED_REFERENCE", **{v: fit_state["status"] for v in FAMILIES[1:]}}
                for part, bounds in contract["evaluate"].items():
                    if fit: signals = issue_paired_spike(inputs, fit, symbol, *bounds).signals
                    else:
                        base = spike_clock_signals(inputs, symbol, *bounds).signals
                        signals = {"CLOCK": base, "RAW44": base.iloc[:0].copy(), "TRANSFORMED44": base.iloc[:0].copy()}
                    evaluate(ticks, signals, symbol, fold, part, bounds, statuses, output, cache)
                del inputs, fit, labels, train_ledger
            p = partitions()
            bounds = p["wf1"][0], p["wf3"][1]
            for ep in ENDPOINTS:
                joined = {v: numeric_ledger(pd.concat([cache[(ep, f, v)][0] for f in ("wf1", "wf2", "wf3")], ignore_index=True)) for v in FAMILIES}
                for family in FAMILIES:
                    audit = {"unknown": sum(cache[(ep, f, family)][1]["unknown"] for f in ("wf1", "wf2", "wf3")),
                             "folds": {f: cache[(ep, f, family)][1] for f in ("wf1", "wf2", "wf3")}}
                    row = metric_row(symbol, family, "walk_forward_combined", bounds, joined[family], joined["CLOCK"], audit, {})
                    row.update(endpoint=ep, baseline_variant="CLOCK", fold="three_independent_refits", fit_status="SEE_FOLDS")
                    output["rows"].append(row)
            del ticks, cache
        attach_gates(output["rows"])
        if len(output["rows"]) != 72: raise ValueError("Incomplete72cell study")
        frozen(expected)
        output["completed_utc"] = datetime.now(timezone.utc).isoformat()
        return save(FOLDER / "results.json", output)
    except Exception as error:
        save(FOLDER / "execution_failure.json", {"error_type": type(error).__name__, "error": str(error),
            "declaration_sha256": expected, "safety": assert_offline(), "QUALIFIED": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true"); g.add_argument("--execute")
    args = p.parse_args()
    print(json.dumps(freeze() if args.freeze else execute(args.execute)), flush=True)


if __name__ == "__main__": main()
