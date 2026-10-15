#!/usr/bin/env python3
"""Freeze/run all twelve excursion diagnostics on unchanged saved regions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.spike_hunter import assert_offline
from app.research.zone_qualification import holm_adjust
from scripts.run_zone_study import (
    SOURCE_AUDIT_SHA256, SYMBOLS, VARIANTS, CONTROL, digest, json_value, load_symbol,
    partitions, frozen as original_frozen,
)
from scripts.zone_excursions import (
    DEFINITIONS, label_anchors, bootstrap_weights, compare_labels,
)

FOLDER = ROOT / "docs/zone_excursions_20261008"
ORIGINAL = ROOT / "docs/zone_study_20261008"
ORIGINAL_DECLARATION = "97b80f2ab852427232190f645343a96b24a295849a54ab4df8343b2f7edd4d18"
ORIGINAL_RESULT = "80d11be442287cd1491b9bc139bc002d3364ee29da55cbef6a02c9fc77fe824c"
REPEATS, SEED = 9999, 20261008
SOURCES = ("scripts/zone_excursions.py", "scripts/run_zone_excursions.py",
           "backend/tests/test_region_excursions.py", "backend/tests/test_region_excursion_runner.py",
           "docs/zone_excursions_20261008/PROTOCOL.md", "docs/zone_excursions_20261008/premeasurement_tests.log")


def save_json(path, value):
    with path.open("x") as f:
        f.write(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def archive_bytes(path, payload):
    compressed = gzip.compress(payload, mtime=0)
    with path.open("xb") as f: f.write(compressed)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "uncompressed_sha256": hashlib.sha256(payload).hexdigest(), "uncompressed_bytes": len(payload)}


def archive_frame(path, frame):
    return archive_bytes(path, frame.to_csv(index=False).encode())


def original_inputs():
    declaration, _ = original_frozen(ORIGINAL / "declaration.json", ORIGINAL_DECLARATION)
    if digest(ORIGINAL / "results.json") != ORIGINAL_RESULT:
        raise ValueError("Original economic result changed")
    result = json.loads((ORIGINAL / "results.json").read_text())
    audit_path = ORIGINAL / "independent_economic_audit_attempt2.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("passed") is not True or audit.get("errors") != []:
        raise ValueError("Original path audit is not passing")
    pins = {str((ORIGINAL / name).relative_to(ROOT)): digest(ORIGINAL / name) for name in
            ("declaration.json", "results.json", "independent_economic_audit_attempt2.json")}
    for symbol in SYMBOLS:
        artifact = result["symbols"][symbol]["candidate_artifact"]
        pins[artifact["path"]] = artifact["sha256"]
    for row in result["rows"]:
        if row["partition"] != "walk_forward_combined":
            artifact = row["artifacts"]["ledger"]
            pins[artifact["path"]] = artifact["sha256"]
    return declaration, result, pins


def spec():
    return json_value({"definitions": DEFINITIONS, "endpoints": ["ISSUE_DELAYED", "ENTRY_CONDITIONAL"],
        "partitions": partitions(), "bootstrap_repeats": REPEATS, "bootstrap_seed": SEED,
        "primary_baselines": ["AVAILABILITY_CLOCK", CONTROL], "secondary_baseline": CONTROL,
        "primary_purge_minutes": 32, "secondary_purge_minutes": 46,
        "heldout_multiplicity_per_endpoint": 144, "known_history_adaptive": True,
        "region_parameters_and_profit_ledgers_changed": False, "QUALIFIED": False})


def freeze():
    assert_offline()
    if any((FOLDER / n).exists() for n in ("declaration.json", "execution_started.json", "results.json.gz", "execution_failure.json")):
        raise ValueError("Diagnostic declaration/output already exists")
    _, _, pins = original_inputs()
    declaration = {"stage": "frozen_unchanged_zone_excursion_diagnostic", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "safety": assert_offline(), "specification": spec(), "original_inputs": pins,
        "code_sha256": {n: digest(ROOT / n) for n in SOURCES},
        "original_source_audit_sha256": SOURCE_AUDIT_SHA256, "new_labels_computed": False,
        "profit_goal_met": False, "QUALIFIED": False}
    artifact = save_json(FOLDER / "declaration.json", declaration)
    with (FOLDER / "declaration.sha256").open("x") as f: f.write(artifact["sha256"] + "\n")
    return artifact


def verify_frozen(expected):
    assert_offline()
    path = FOLDER / "declaration.json"
    if digest(path) != expected or (FOLDER / "declaration.sha256").read_text().strip() != expected:
        raise ValueError("Exact frozen diagnostic SHA required")
    d = json.loads(path.read_text())
    if d["specification"] != spec() or d["code_sha256"] != {n: digest(ROOT / n) for n in SOURCES}:
        raise ValueError("Diagnostic code or specification changed")
    for name, pin in d["original_inputs"].items():
        if digest(ROOT / name) != pin:
            raise ValueError(f"Saved region/ledger changed: {name}")
    original, result, pins = original_inputs()
    if pins != d["original_inputs"]:
        raise ValueError("Original input population changed")
    return d, original, result


def available_anchors(candidates):
    """Only past-known complete context creates clock membership."""
    control = candidates.loc[candidates.variant.eq(CONTROL)].copy()
    mask = control.reason.isin(("h4_h1_context_disagrees", "m5_trigger_absent", "issued_fixed_region"))
    selected = control.loc[mask, ["signal_time", "atr", "side"]].rename(columns={"signal_time": "issue_time"})
    selected["issue_time"] = pd.to_datetime(selected.issue_time, utc=True)
    selected["anchor_time"] = selected.issue_time + pd.Timedelta(seconds=61)
    for variant in VARIANTS:
        subset = candidates.loc[candidates.variant.eq(variant) & candidates.eligible, "signal_time"]
        selected["eligible_" + variant] = selected.issue_time.isin(pd.to_datetime(subset, utc=True))
    if selected.issue_time.duplicated().any() or not selected["atr"].gt(0).all():
        raise ValueError("Clock membership is duplicated or lacks raw ATR")
    if any((selected["eligible_" + v] & ~selected["eligible_" + CONTROL]).any() for v in VARIANTS):
        raise ValueError("Declared geometric context is not a superset of region offers")
    return selected


def native_entries(ledger):
    return ledger.loc[:, ["signal_time", "entry_time", "atr", "side"]].rename(
        columns={"signal_time": "issue_time", "entry_time": "anchor_time"}).assign(
        issue_time=lambda f: pd.to_datetime(f.issue_time, utc=True),
        anchor_time=lambda f: pd.to_datetime(f.anchor_time, utc=True))


def describe(symbol, endpoint, part, frames, bounds):
    days, weights = bootstrap_weights(*bounds, repeats=REPEATS, seed=SEED)
    output = []
    for variant in VARIANTS:
        baselines = ("AVAILABILITY_CLOCK", CONTROL) if endpoint == "ISSUE_DELAYED" else (CONTROL,)
        for baseline in baselines:
            for a, h in DEFINITIONS:
                metrics = compare_labels(frames[variant], frames[baseline], a, h,
                    days=days, weights=weights, opportunity_recall=endpoint == "ISSUE_DELAYED")
                output.append({"symbol": symbol, "endpoint": endpoint, "partition": part,
                    "variant": variant, "baseline_variant": baseline, "start": bounds[0], "end_exclusive": bounds[1],
                    **metrics})
    return output


def attach_holm(rows):
    for endpoint in ("ISSUE_DELAYED", "ENTRY_CONDITIONAL"):
        grouped = {}
        for row in rows:
            if row["endpoint"] == endpoint and row["variant"] != CONTROL and row["partition"] in ("final_test", "later180"):
                key = tuple(row[k] for k in ("symbol", "partition", "variant", "atr_multiplier", "horizon_minutes"))
                grouped.setdefault(key, []).append(row)
        if len(grouped) != 144:
            raise ValueError("The frozen144comparison family is incomplete")
        p = [max(r["p"] for r in group) for group in grouped.values()]
        for group, adjusted in zip(grouped.values(), holm_adjust(p), strict=True):
            for row in group: row["holm_p"] = adjusted


def execute(expected):
    declaration, original, result = verify_frozen(expected)
    save_json(FOLDER / "execution_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": expected, "safety": assert_offline()})
    try:
        output = {"stage": "unchanged_zone_directional_excursion_diagnostic", "declaration_sha256": expected,
            "safety": assert_offline(), "known_history_adaptive": True, "QUALIFIED": False,
            "economic_results_unchanged": True, "economic_result_sha256": ORIGINAL_RESULT,
            "rows": [], "artifacts": [], "symbols": {}}
        original_rows = {(r["symbol"], r["variant"], r["partition"]): r for r in result["rows"]}
        for symbol in SYMBOLS:
            dataset, ticks = load_symbol(original["sources"]["symbols"][symbol])
            times = ticks.index.asi8 // 10**9
            prices = ticks.quote.to_numpy(float)
            artifact = result["symbols"][symbol]["candidate_artifact"]
            candidates = pd.read_csv(ROOT / artifact["path"], low_memory=False)
            anchors = available_anchors(candidates)
            frame_by_part = {}
            for part, bounds in partitions().items():
                selected = anchors.loc[anchors.issue_time.ge(bounds[0]) & anchors.issue_time.lt(bounds[1])]
                primary = label_anchors(times, prices, selected, start=bounds[0], end=bounds[1], endpoint="ISSUE_DELAYED")
                frames = {"AVAILABILITY_CLOCK": primary, **{v: primary.loc[primary["eligible_" + v]].copy() for v in VARIANTS}}
                frame_by_part[("ISSUE_DELAYED", part)] = frames
                output["artifacts"].append({"symbol": symbol, "endpoint": "ISSUE_DELAYED", "partition": part,
                    **archive_frame(FOLDER / f"{symbol}_{part}_issue_labels.csv.gz", primary)})
                output["rows"].extend(describe(symbol, "ISSUE_DELAYED", part, frames, bounds))
                entries = {}
                for variant in VARIANTS:
                    ledger_artifact = original_rows[(symbol, variant, part)]["artifacts"]["ledger"]
                    ledger = pd.read_csv(ROOT / ledger_artifact["path"])
                    declared = native_entries(ledger)
                    labels = label_anchors(times, prices, declared, start=bounds[0], end=bounds[1], endpoint="ENTRY_CONDITIONAL")
                    entries[variant] = labels
                    output["artifacts"].append({"symbol": symbol, "endpoint": "ENTRY_CONDITIONAL", "partition": part,
                        "variant": variant, **archive_frame(FOLDER / f"{symbol}_{variant}_{part}_entry_labels.csv.gz", labels)})
                frame_by_part[("ENTRY_CONDITIONAL", part)] = entries
                output["rows"].extend(describe(symbol, "ENTRY_CONDITIONAL", part, entries, bounds))
                print(f"{symbol}/{part}: fixed12definitions/2endpoints measured", flush=True)
            combined_bounds = (partitions()["wf1"][0], partitions()["wf3"][1])
            for endpoint in ("ISSUE_DELAYED", "ENTRY_CONDITIONAL"):
                variants = ("AVAILABILITY_CLOCK", *VARIANTS) if endpoint == "ISSUE_DELAYED" else VARIANTS
                frames = {v: pd.concat([frame_by_part[(endpoint, f)][v] for f in ("wf1", "wf2", "wf3")], ignore_index=True) for v in variants}
                output["rows"].extend(describe(symbol, endpoint, "walk_forward_combined", frames, combined_bounds))
            output["symbols"][symbol] = {"dataset_summary": dataset.summary,
                "scheduled_rows": int(candidates.variant.eq(CONTROL).sum()), "available_clock_rows": len(anchors)}
            del ticks, dataset, times, prices
        attach_holm(output["rows"])
        verify_frozen(expected)
        output["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(json_value(output), indent=2, allow_nan=False).encode() + b"\n"
        result_artifact = archive_bytes(FOLDER / "results.json.gz", payload)
        save_json(FOLDER / "result_index.json", {**result_artifact, "rows": len(output["rows"]),
            "declaration_sha256": expected, "QUALIFIED": False, "safety": assert_offline()})
        return result_artifact
    except Exception as error:
        save_json(FOLDER / "execution_failure.json", {"error_type": type(error).__name__, "error": str(error),
            "declaration_sha256": expected, "safety": assert_offline(), "QUALIFIED": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--execute", metavar="SHA256")
    args = parser.parse_args()
    print(json.dumps(freeze() if args.freeze else execute(args.execute)), flush=True)


if __name__ == "__main__":
    main()
