#!/usr/bin/env python3
"""Independent stdlib audit of combined-timeframe saved quote-path ledgers.

No model, feature, replay engine or research metrics imports. Holm arithmetic
uses saved p-values; this does not independently verify those p-values or CIs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_spike_payoff import Audit, FLAGS, causal_atr, epoch, sha256
from scripts.verify_spike_learned import number, read_json, safety_inputs, source_path
from scripts.verify_spike_native300 import (
    COHORT, MODES, PRIMARY, SYMBOLS, mixed_summary, observed_value,
    path_or_censor, read_ohlc,
)


def timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Chronology requires timezone-aware timestamps")
    return stamp


def holm_values(values):
    values = [number(value, "Holm input") for value in values]
    if not all(0 <= value <= 1 for value in values):
        raise ValueError("Saved p-values must lie in [0,1]")
    adjusted = [None] * len(values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def ledger(study, symbol, mode, role, saved_model, fit, minutes, start, end, audit):
    label = f"{symbol}/{mode}/{role}"
    before = len(audit.failures)
    prefix = f"{symbol}_{COHORT}_{mode}"
    suffix = "" if role == "combined" else "_m5_reference"
    signals_path = study / f"{prefix}{suffix}_signals.csv"
    trades_path = study / f"{prefix}{suffix}_trades.csv"
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    threshold = number(fit["threshold"], label)
    audit.equal(f"{label}/training_before_fresh", epoch(fit["training_end"]) < start, True)
    audit.equal(f"{label}/training_purge", fit["label_purge_minutes"], 31)
    audit.equal(f"{label}/training_latest_issue",
                epoch(fit["training_latest_issue"]) + 31 * 60 <= epoch(fit["training_end"]), True)
    audit.equal(f"{label}/training_latest_planned_end",
                epoch(fit["training_latest_planned_end"]) <= epoch(fit["training_end"]), True)
    execution = saved_model["audit"]
    audit.equal(f"{label}/config", execution["config"], PRIMARY)
    audit.equal(f"{label}/purge", execution["purge_minutes"], 31)
    audit.equal(f"{label}/take_profit", execution["take_profit"], None)
    signals, atr_cache = {}, {}
    previous = None
    with signals_path.open(newline="") as stream:
        for ordinal, saved in enumerate(csv.DictReader(stream)):
            signal = epoch(saved["signal_time"])
            item = f"{label}/signals/{ordinal}"
            audit.equal(f"{item}/UTC00_30", signal % 1800, 0)
            audit.equal(f"{item}/partition", start <= signal < end, True)
            audit.equal(f"{item}/unique", signal not in signals, True)
            audit.equal(f"{item}/side", number(saved["side"], item), side)
            audit.equal(f"{item}/variant", saved["variant"], f"LEARNED_{mode}")
            score = number(saved["score"], item)
            audit.equal(f"{item}/saved_cutoff", score > 0 and score >= threshold, True)
            if previous is not None:
                audit.equal(f"{item}/spacing", signal - previous >= 1800, True)
            atr_cache[signal] = causal_atr(minutes, signal)
            audit.equal(f"{item}/causal_ATR", number(saved["atr"], item), atr_cache[signal])
            audit.equal(f"{item}/closed_candle", number(saved["signal_close"], item), minutes[signal - 60][3])
            signals[signal] = saved
            previous = signal
    admitted = {signal for signal in signals if signal + 31 * 60 <= end}
    missing_entries = {signal for signal in admitted if signal + 60 not in minutes}
    expected_signals = admitted - missing_entries
    reconstructed, observed = [], []
    previous_signal = previous_exit = None
    with trades_path.open(newline="") as stream:
        for ordinal, saved in enumerate(csv.DictReader(stream)):
            signal = epoch(saved["signal_time"])
            item = f"{label}/trades/{ordinal}"
            audit.equal(f"{item}/saved_signal", signal in signals, True)
            audit.equal(f"{item}/UTC00_30", signal % 1800, 0)
            audit.equal(f"{item}/planned_partition", start <= signal and signal + 31 * 60 <= end, True)
            audit.equal(f"{item}/variant", saved["variant"], f"LEARNED_{mode}")
            atr = atr_cache.get(signal)
            if atr is None:
                atr = causal_atr(minutes, signal)
            calculated = path_or_censor(minutes, signal, atr, side, PRIMARY)
            if previous_signal is not None:
                audit.equal(f"{item}/spacing", signal - previous_signal >= 1800, True)
                audit.equal(f"{item}/occupancy", calculated["entry_time"] >= previous_exit, True)
            for key, expected in calculated.items():
                audit.equal(f"{item}/{key}", observed_value(saved, key, expected, item), expected)
            reconstructed.append(calculated)
            observed.append(signal)
            previous_signal, previous_exit = signal, calculated["exit_time"]
    audit.equal(f"{label}/signal_members", sorted(observed), sorted(expected_signals))
    audit.equal(f"{label}/signal_unique", len(observed), len(set(observed)))
    censored = sum(row["censored"] for row in reconstructed)
    for field, expected in {"issued": len(signals), "purged": len(signals) - len(admitted),
                            "missing_entry": len(missing_entries), "filled": len(reconstructed),
                            "completed": len(reconstructed) - censored, "censored": censored,
                            "outside_partition": 0, "overlap_skipped": 0, "ambiguous": 0}.items():
        audit.equal(f"{label}/execution/{field}", execution[field], expected)
    independent = mixed_summary(reconstructed, start, end, PRIMARY["stop_atr"],
                                saved_model["metrics"]["illustrative_risk_fraction"])
    for field, expected in independent.items():
        audit.equal(f"{label}/metrics/{field}", saved_model["metrics"][field], expected)
    if role == "combined":
        audit.equal(f"{label}/observed_target", saved_model["user_target_observed"],
                    independent["completed"] >= 1000 and (independent["profit_factor"] or 0) >= 1.5)
    return {"symbol": symbol, "mode": mode, "role": role, "trades": len(reconstructed),
            "completed": len(reconstructed) - censored, "censored": censored,
            "issued_signals": len(signals), "ledger_sha256": sha256(trades_path),
            "signals_sha256": sha256(signals_path), "mean_net_R": independent["mean_net_R"],
            "profit_factor": independent["profit_factor"], "mismatches": len(audit.failures) - before}


def verify(study, audit):
    result, selection, declaration = [read_json(study / name) for name in
                                      ("results.json", "selection.json", "declaration.json")]
    for label, payload in (("results", result), ("selection", selection), ("declaration", declaration)):
        audit.safety(payload, label)
        audit.equal(f"{label}/safety", payload["safety"], dict.fromkeys(FLAGS, False))
    audit.equal("result/stage", result["stage"], "combined_multiframe_chronological_evaluation")
    audit.equal("selection/stage", selection["stage"], "frozen_combined_multiframe_development")
    audit.equal("declaration/stage", declaration["stage"], "combined_multiframe_predevelopment_declaration")
    audit.equal("feature_count", selection["feature_count"], 44)
    audit.equal("joint_family", result["joint_fresh_hypotheses"], 8)
    audit.equal("config", selection["config"], PRIMARY)
    audit.equal("user_target", selection["user_target"],
                {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000})
    selection_hash = sha256(study / "selection.json")
    declaration_hash = sha256(study / "declaration.json")
    for name, observed in (("selection", selection_hash), ("declaration", declaration_hash)):
        audit.equal(f"hash/{name}", observed, result[f"{name}_sha256"])
        audit.equal(f"hash/{name}_file", observed, (study / f"{name}.sha256").read_text().strip())
    audit.equal("hash/selection_declaration", declaration_hash, selection["declaration_sha256"])
    audit.equal("hash/result_code", result["code_hashes"], selection["code_hashes"])
    for field in ("code_hashes", "sources", "intervals", "native_reference", "config", "feature_formulas",
                  "bootstrap_repeats", "feature_count", "joint_fresh_hypotheses", "user_target"):
        audit.equal(f"declaration/{field}", selection[field], declaration[field])
    for field in ("fresh_features_evaluated", "fresh_payoffs_evaluated"):
        audit.equal(f"declaration/{field}", declaration[field], False)
        audit.equal(f"selection/{field}", selection[field], False)
    for name, expected in selection["code_hashes"].items():
        audit.equal(f"hash/code/{name}", sha256(source_path(name)), expected)
    lineage = selection["native_reference"]
    native_dir = source_path(lineage["study_dir"])
    native_selection = read_json(native_dir / "selection.json")
    native_result = read_json(native_dir / "results.json")
    audit.safety(native_result, "native_result")
    audit.equal("native/result_hash", sha256(native_dir / "results.json"), result["native_results_sha256"])
    audit.equal("native/selection_hash", sha256(native_dir / "selection.json"), lineage["selection_sha256"])
    audit.equal("native/declaration_hash", sha256(native_dir / "declaration.json"), lineage["declaration_sha256"])
    audit.equal("native/code", native_selection["code_hashes"], lineage["code_hashes"])
    audit.equal("native/result_selection", native_result["selection_sha256"], lineage["selection_sha256"])
    audit.equal("native/identical_sources", native_selection["sources"], selection["sources"])
    audit.equal("native/portable_path", Path(lineage["study_dir"]).is_absolute(), False)
    audit.equal("result/native_lineage", result["native_reference"], lineage)
    audit.equal("chronology/native_after_both_frozen",
                timestamp(native_result["run_utc"]) >= timestamp(selection["frozen_utc"]), True)
    audit.equal("chronology/combined_after_native",
                timestamp(result["run_utc"]) >= timestamp(native_result["run_utc"]), True)
    audit.equal("chronology/declaration_before_frozen",
                timestamp(declaration["run_utc"]) <= timestamp(selection["frozen_utc"]), True)
    audit.equal("chronological_old_to_new", result["chronological_old_to_new"], True)
    audit.equal("prospective", result["prospective_paper"], False)
    audit.equal("actual_cash", result["actual_money_profit"], "NOT TESTED")
    ledgers, family = [], []
    for symbol in SYMBOLS:
        pair = selection["sources"][symbol]
        old, fresh = pair["old"], pair["fresh"]
        audit.equal(f"{symbol}/adjacent", old["last_epoch"] + 60, fresh["first_epoch"])
        for kind, source in pair.items():
            audit.equal(f"{symbol}/{kind}/clean_hash", sha256(source_path(source["path"])), source["sha256"])
            manifest_path = source_path(source["manifest"])
            manifest = read_json(manifest_path)
            audit.safety(manifest, f"{symbol}/{kind}/manifest")
            audit.equal(f"{symbol}/{kind}/manifest_hash", sha256(manifest_path), source["manifest_sha256"])
            audit.equal(f"{symbol}/{kind}/raw_hash", sha256(source_path(manifest["raw_file"])), manifest["raw_sha256"])
            audit.equal(f"{symbol}/{kind}/manifest_clean", manifest["normalized_sha256"], source["sha256"])
            audit.equal(f"{symbol}/{kind}/no_interpolation", manifest["fills_or_interpolations"], False)
            audit.equal(f"{symbol}/{kind}/no_auth", manifest["authentication_used"], False)
        cohort = result["symbols"][symbol]["cohorts"][COHORT]
        start, end = epoch(cohort["start"]), epoch(cohort["end"])
        manifest = read_json(source_path(fresh["manifest"]))
        minutes, excluded, duplicates = read_ohlc(source_path(fresh["path"]))
        raw, raw_excluded, raw_duplicates = read_ohlc(source_path(manifest["raw_file"]), allow_off_grid=True)
        audit.equal(f"{symbol}/raw_normalization", minutes == raw, True)
        audit.equal(f"{symbol}/raw_excluded", raw_excluded, [row["epoch"] for row in manifest["excluded_rows"]])
        audit.equal(f"{symbol}/raw_duplicates", raw_duplicates, manifest["equal_duplicates_removed"])
        del raw
        audit.equal(f"{symbol}/clean_excluded", excluded, [])
        audit.equal(f"{symbol}/clean_duplicates", duplicates, 0)
        missing = sorted(set(range(start, end, 60)) - set(minutes))
        audit.equal(f"{symbol}/missing", missing,
                    sorted(t for gap in fresh["declared_gaps"] for t in gap["missing_epochs"]))
        audit.equal(f"{symbol}/missing_count", len(missing), fresh["declared_missing_minutes"])
        audit.equal(f"{symbol}/cohort_source", cohort["source"], fresh)
        audit.equal(f"{symbol}/cohort_hash", cohort["audit"]["sha256"], fresh["sha256"])
        audit.equal(f"{symbol}/start", start, fresh["first_epoch"])
        audit.equal(f"{symbol}/end", end, fresh["last_epoch"] + 60)
        audit.equal(f"{symbol}/rows", len(minutes), fresh["rows"])
        audit.equal(f"{symbol}/full180", len(minutes) + len(missing), 180 * 1440)
        clock = cohort["comparison_clock"]
        audit.equal(f"{symbol}/clock_common", clock["common_clock_rows"], clock["mtf_eligible_clock_rows"])
        audit.equal(f"{symbol}/clock_exclusion", clock["native_opportunities_excluded_by_mtf_features"],
                    clock["full_native_eligible_clock_rows"] - clock["common_clock_rows"])
        audit.equal(f"{symbol}/clock_no_future", clock["intersection_uses_future_outcomes"], False)
        for mode in MODES:
            row = cohort["models"][mode]
            candidate = selection["symbols"][symbol]["models"][mode]
            audit.equal(f"{symbol}/{mode}/frozen_eligibility", row["development_eligible"], candidate["development_eligible"])
            audit.equal(f"{symbol}/{mode}/feature_order", len(candidate["final_model"]["feature_names"]), 44)
            ledgers.append(ledger(study, symbol, mode, "combined", row, candidate["final_model"], minutes, start, end, audit))
            native_fit = native_selection["symbols"][symbol]["models"][mode]["final_model"]
            ledgers.append(ledger(study, symbol, mode, "common_m5", row["m5_reference"], native_fit,
                                  minutes, start, end, audit))
            metric = row["metrics"]
            audit.equal(f"{symbol}/{mode}/conjunction", metric["p"],
                        max(metric["clock_conjunction_p"], metric["m5_reference_day_p"], metric["m5_reference_weekly_p"]))
            original = native_result["symbols"][symbol]["cohorts"][COHORT]["models"][mode]
            copied = result["native_reference_joint_inference"]["symbols"][symbol]["models"][mode]
            audit.equal(f"{symbol}/{mode}/native_original_holm", copied["original_native_holm_p"], original["metrics"]["holm_p"])
            for key in original["metrics"]:
                if key != "holm_p":
                    audit.equal(f"{symbol}/{mode}/native_copy/{key}", copied["metrics"][key], original["metrics"][key])
            for key in ("audit", "config", "baseline", "baseline_audit", "thirds", "tail", "sensitivities"):
                audit.equal(f"{symbol}/{mode}/native_copy/{key}", copied[key], original[key])
            family.extend((copied, row))
    audit.equal("joint_holm/family_size", len(family), 8)
    for index, (row, adjusted) in enumerate(zip(family, holm_values([r["metrics"]["p"] for r in family]))):
        audit.equal(f"joint_holm/{index}", row["metrics"]["holm_p"], adjusted)
    return selection_hash, ledgers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_multiframe_20261005")
    args = parser.parse_args()
    if not (args.study / "results.json").is_file():
        parser.error("Completed combined results.json required")
    audit, selection_hash, ledgers = Audit(), None, []
    try:
        safety_inputs(audit)
        if audit.failures:
            raise ValueError("Requires all four runtime safety flags=false")
        selection_hash, ledgers = verify(args.study, audit)
    except Exception as exc:
        audit.failures.append({"check": "audit_exception", "error": f"{type(exc).__name__}: {exc}"})
    helpers = ("scripts/verify_spike_payoff.py", "scripts/verify_spike_timed.py",
               "scripts/verify_spike_learned.py", "scripts/verify_spike_native300.py")
    report = {"status": "FAIL" if audit.failures else "PASS", "pass": not audit.failures,
              "run_utc": datetime.now(timezone.utc).isoformat(), "safety": dict.fromkeys(FLAGS, False),
              "implementation": "Independent standard-library quote-path and summary arithmetic; no ML, engine or research metrics imports",
              "script_sha256": sha256(Path(__file__)), "helper_sha256": {name: sha256(ROOT / name) for name in helpers},
              "selection_sha256": selection_hash, "results_sha256": sha256(args.study / "results.json"),
              "scope": ["four combined and four common-M5 saved later-period signal/trade ledgers",
                        "frozen code/source/declaration/selection/native-result hashes and recorded chronology",
                        "raw normalization exclusions, missing-minute preservation and no interpolation",
                        "closed M5 ATR14, saved score/cutoff, side, close, UTC00/30 and 31-minute purge",
                        "saved issuance-to-ledger completeness, scalar quote-path R and censored unknowns",
                        "counts/means/PF/day-cluster SE/illustrative closed equity arithmetic",
                        "unchanged copied native metrics and joint Holm-eight arithmetic from saved p-values"],
              "not_verified": ["44 or 19 ML features independently regenerated", "scalers/model fitting/scores independently regenerated",
                               "all possible ML opportunities/common feature eligibility", "clock-baseline replay",
                               "bootstrap CIs or p-values", "sensitivity ledgers", "actual broker fills/costs/cash profits"],
              "checks": audit.checks, "saved_false_safety_values": audit.safety_values_checked,
              "total_trades": sum(row["trades"] for row in ledgers), "ledgers": ledgers,
              "error_count": len(audit.failures), "errors": audit.failures,
              "max_numeric_differences": audit.max_numeric_error}
    (args.study / "independent_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "pass", "checks", "total_trades", "error_count")}))
    raise SystemExit(bool(audit.failures))


if __name__ == "__main__":
    main()
