#!/usr/bin/env python3
"""Independent scalar audit of saved learned-transfer signals and trades.

Standard-library helpers only: no model, study engine, or research metrics
imports. Reconstructs quote paths and summary arithmetic, not ML features,
fitting, prediction scores, missing eligible signals, or bootstrap inference.
Reads outcomes only after the completed study results.json exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_spike_payoff import Audit, FLAGS, causal_atr, epoch, read_minutes, sha256, summary
from scripts.verify_spike_timed import scalar

COHORT = "cross_symbol_transfer"
TRANSFER = {"BOOM500": "BOOM300N", "CRASH500": "CRASH300N"}
MODES = ("SPIKE", "DRIFT")
PRIMARY = {"stop_atr": 2.0, "max_hold_minutes": 15, "entry_delay_minutes": 1,
           "round_trip_cost_atr": .10, "fill_mode": "adverse_extreme"}


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value in {path.name}: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def source_path(value):
    path = Path(value)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(ROOT):
        raise ValueError("Source paths must remain relative to the repository")
    return resolved


def number(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite saved number: {label}")
    return result


def saved_value(saved, key, expected, label):
    if key in ("signal_time", "entry_time", "exit_time", "planned_end"):
        return epoch(saved[key])
    if isinstance(expected, bool):
        if saved[key] not in ("True", "False"):
            raise ValueError(f"Invalid saved boolean: {label}/{key}")
        return saved[key] == "True"
    if isinstance(expected, (int, float)):
        return number(saved[key], f"{label}/{key}")
    return saved[key]


def safety_inputs(audit):
    for flag in FLAGS:
        audit.equal(f"safety/environment/{flag}",
                    os.environ.get(flag, "false").strip().lower() == "false", True)
    # Inspect only these four public configuration keys; never print other .env values.
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if "=" not in stripped or stripped.startswith("#"):
                continue
            key, value = stripped.split("=", 1)
            key = key.strip().upper()
            if key in FLAGS:
                value = value.split("#", 1)[0].strip().strip("\"'").lower()
                audit.equal(f"safety/dotenv/{key}", value == "false", True)


def verify(study, audit):
    result = read_json(study / "results.json")
    selection = read_json(study / "selection.json")
    audit.safety(result, "results")
    audit.safety(selection, "selection")
    audit.equal("safety/results", result["safety"], dict.fromkeys(FLAGS, False))
    audit.equal("safety/selection", selection["safety"], dict.fromkeys(FLAGS, False))
    audit.equal("study/result_stage", result["stage"], "learned_evaluation")
    audit.equal("study/selection_stage", selection["stage"], "frozen_learned_development")
    audit.equal("study/training_symbols", sorted(result["symbols"]), sorted(TRANSFER))
    selection_hash = sha256(study / "selection.json")
    audit.equal("hash/selection", selection_hash, result["selection_sha256"])
    audit.equal("hash/saved_selection", selection_hash, (study / "selection.sha256").read_text().strip())
    audit.equal("hash/code_metadata", result["code_hashes"], selection["code_hashes"])
    for name, expected in selection["code_hashes"].items():
        audit.equal(f"hash/code/{name}", sha256(source_path(name)), expected)
    for training_symbol, sources in selection["sources"].items():
        for kind, source in sources.items():
            audit.equal(f"hash/data/{training_symbol}/{kind}",
                        sha256(source_path(source["path"])), source["sha256"])
    audit.equal("config/primary", selection["config"], PRIMARY)
    ledgers = []
    for training_symbol, expected_symbol in TRANSFER.items():
        cohort = result["symbols"][training_symbol]["cohorts"][COHORT]
        symbol = cohort["symbol"]
        audit.equal(f"{training_symbol}/transfer_symbol", symbol, expected_symbol)
        audit.equal(f"{symbol}/modes", sorted(cohort["models"]), sorted(MODES))
        frozen_source = selection["sources"][training_symbol]["transfer"]
        audit.equal(f"{symbol}/frozen_source", cohort["source"], frozen_source)
        clean = source_path(frozen_source["path"])
        raw = clean.with_name(clean.name.replace("_clean", ""))
        actual_hash = sha256(clean)
        audit.equal(f"{symbol}/cohort_source_hash", cohort["audit"]["sha256"], actual_hash)
        audit.equal(f"{symbol}/raw_clean_hash", sha256(raw), actual_hash)
        manifest = read_json(source_path(frozen_source["manifest"]))
        audit.safety(manifest, f"{symbol}/manifest")
        audit.equal(f"{symbol}/manifest_hash", manifest["normalized_sha256"], actual_hash)
        minutes = read_minutes(raw)
        start, end = epoch(cohort["start"]), epoch(cohort["end"])
        if not minutes:
            raise ValueError(f"Empty source: {symbol}")
        audit.equal(f"{symbol}/source_file", cohort["audit"]["source_file"], clean.name)
        audit.equal(f"{symbol}/rows", len(minutes), cohort["audit"]["source_rows"])
        audit.equal(f"{symbol}/frozen_rows", len(minutes), frozen_source["rows"])
        audit.equal(f"{symbol}/manifest_rows", len(minutes), manifest["normalized_rows"])
        audit.equal(f"{symbol}/start", min(minutes), start)
        audit.equal(f"{symbol}/end", max(minutes) + 60, end)
        audit.equal(f"{symbol}/frozen_start", min(minutes), frozen_source["first_epoch"])
        audit.equal(f"{symbol}/frozen_last", max(minutes), frozen_source["last_epoch"])
        audit.equal(f"{symbol}/full180days", end - start, 180 * 86400)
        audit.equal(f"{symbol}/full180days_rows", len(minutes), 180 * 1440)
        atr_cache = {}
        for mode in MODES:
            model = cohort["models"][mode]
            frozen_model = selection["symbols"][training_symbol]["models"][mode]
            config = model["config"]
            label = f"{symbol}/{mode}"
            failures_before = len(audit.failures)
            audit.equal(f"{label}/mode", model["mode"], mode)
            audit.equal(f"{label}/frozen_config", config, frozen_model["config"])
            audit.equal(f"{label}/primary_config", config, PRIMARY)
            audit.equal(f"{label}/audit_config", model["audit"]["config"], config)
            audit.equal(f"{label}/purge_minutes", model["audit"]["purge_minutes"], 31)
            side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
            threshold = number(frozen_model["final_model"]["threshold"], f"{label}/threshold")
            signals_path = study / f"{symbol}_{COHORT}_{mode}_signals.csv"
            trades_path = study / f"{symbol}_{COHORT}_{mode}_trades.csv"
            signals = {}
            previous_signal = None
            with signals_path.open(newline="") as stream:
                for ordinal, saved in enumerate(csv.DictReader(stream)):
                    signal = epoch(saved["signal_time"])
                    signal_label = f"{label}/signals/{ordinal}"
                    audit.equal(f"{signal_label}/clock_UTC00_30", signal % 1800, 0)
                    audit.equal(f"{signal_label}/partition", start <= signal < end, True)
                    audit.equal(f"{signal_label}/unique", signal not in signals, True)
                    audit.equal(f"{signal_label}/side", number(saved["side"], signal_label), side)
                    audit.equal(f"{signal_label}/variant", saved["variant"], f"LEARNED_{mode}")
                    score = number(saved["score"], f"{signal_label}/score")
                    audit.equal(f"{signal_label}/frozen_score_threshold", score >= threshold and score > 0, True)
                    if previous_signal is not None:
                        audit.equal(f"{signal_label}/spacing", signal - previous_signal >= 1800, True)
                    if signal not in atr_cache:
                        atr_cache[signal] = causal_atr(minutes, signal)
                    audit.equal(f"{signal_label}/causal_ATR", number(saved["atr"], signal_label), atr_cache[signal])
                    audit.equal(f"{signal_label}/completed_candle_close",
                                number(saved["signal_close"], signal_label), minutes[signal - 60][3])
                    signals[signal] = saved
                    previous_signal = signal
            admitted = {signal for signal in signals if signal + 31 * 60 <= end}
            purged = len(signals) - len(admitted)
            reconstructed = []
            observed_signals = []
            previous_signal = previous_exit = None
            with trades_path.open(newline="") as stream:
                for ordinal, saved in enumerate(csv.DictReader(stream)):
                    signal = epoch(saved["signal_time"])
                    trade_label = f"{label}/trades/{ordinal}"
                    audit.equal(f"{trade_label}/saved_signal", signal in signals, True)
                    audit.equal(f"{trade_label}/clock_UTC00_30", signal % 1800, 0)
                    audit.equal(f"{trade_label}/partition_purge", start <= signal < end and signal + 31 * 60 <= end, True)
                    audit.equal(f"{trade_label}/variant", saved["variant"], f"LEARNED_{mode}")
                    if signal not in atr_cache:
                        atr_cache[signal] = causal_atr(minutes, signal)
                    calculated = scalar(minutes, signal, atr_cache[signal], side, config)
                    if previous_signal is not None:
                        audit.equal(f"{trade_label}/spacing", signal - previous_signal >= 1800, True)
                        audit.equal(f"{trade_label}/no_overlap", calculated["entry_time"] >= previous_exit, True)
                    for key, expected in calculated.items():
                        audit.equal(f"{trade_label}/{key}", saved_value(saved, key, expected, trade_label), expected)
                    reconstructed.append(calculated)
                    observed_signals.append(signal)
                    previous_signal, previous_exit = signal, calculated["exit_time"]
            audit.equal(f"{label}/signal_to_ledger_count", len(observed_signals), len(admitted))
            audit.equal(f"{label}/signal_to_ledger_unique", len(set(observed_signals)), len(observed_signals))
            audit.equal(f"{label}/signal_to_ledger_members", sorted(observed_signals), sorted(admitted))
            execution = model["audit"]
            audit.equal(f"{label}/issued", execution["issued"], len(signals))
            audit.equal(f"{label}/purged", execution["purged"], purged)
            for field in ("outside_partition", "missing_entry", "overlap_skipped", "censored", "ambiguous"):
                audit.equal(f"{label}/audit/{field}", execution[field], 0)
            for field in ("filled", "completed"):
                audit.equal(f"{label}/audit/{field}", execution[field], len(reconstructed))
            independent = summary(reconstructed, start, end, config["stop_atr"],
                                  model["metrics"]["illustrative_risk_fraction"])
            for field, expected in independent.items():
                audit.equal(f"{label}/metrics/{field}", model["metrics"][field], expected)
            ledgers.append({"training_symbol": training_symbol, "evaluation_symbol": symbol, "mode": mode,
                            "issued_signals": len(signals), "partition_purged_signals": purged,
                            "trades": len(reconstructed), "ledger_sha256": sha256(trades_path),
                            "signals_sha256": sha256(signals_path), "raw_data_sha256": sha256(raw),
                            "mean_net_R": independent["mean_net_R"], "profit_factor": independent["profit_factor"],
                            "mismatches": len(audit.failures) - failures_before})
    return result, selection_hash, ledgers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_learned_20261005")
    args = parser.parse_args()
    # No selection, price or outcome reads until the completed evaluator output exists.
    if not (args.study / "results.json").is_file():
        parser.error("Completed results.json required before reading learned outcomes")
    audit = Audit()
    result = None
    selection_hash = None
    ledgers = []
    try:
        safety_inputs(audit)
        if audit.failures:
            raise ValueError("Requires all four runtime safety flags=false")
        result, selection_hash, ledgers = verify(args.study, audit)
    except Exception as exc:
        audit.failures.append({"check": "audit_exception", "error": f"{type(exc).__name__}: {exc}"})
    passed = not audit.failures
    report = {"status": "PASS" if passed else "FAIL", "pass": passed,
              "run_utc": datetime.now(timezone.utc).isoformat(), "safety": dict.fromkeys(FLAGS, False),
              "implementation": "Independent scalar standard-library quote-path and arithmetic audit; no ML, engine or metrics imports",
              "script_sha256": sha256(Path(__file__)),
              "helper_sha256": {name: sha256(ROOT / name) for name in
                                 ("scripts/verify_spike_payoff.py", "scripts/verify_spike_timed.py")},
              "selection_sha256": selection_hash, "results_sha256": sha256(args.study / "results.json"),
              "scope": ["all four saved learned transfer signal and trade ledgers, including empty ledgers",
                        "frozen selection/code/data hashes and saved/runtime safety flags",
                        "raw/clean equality, exact 180-day M1 grid and independently aggregated causal M5 ATR14",
                        "saved signal UTC minute00/30, mode side, positive score above frozen cutoff and candle close",
                        "saved issued-signal to trade-ledger completeness after the fixed 31-minute boundary purge",
                        "scalar delayed entry, adverse-extreme stop/time chronology, gross/net R and holding times",
                        "30-minute spacing, no overlapping positions and all summary arithmetic"],
              "not_verified": ["nineteen ML features independently regenerated",
                               "training labels, model fitting, coefficients, scaling and predicted scores independently regenerated",
                               "completeness against all possible eligible ML opportunities",
                               "clock-baseline replay", "bootstrap intervals, p-values or Holm inference",
                               "sensitivity ledgers", "actual broker fills, measured costs or monetary profit"],
              "checks": audit.checks, "saved_false_safety_values": audit.safety_values_checked,
              "total_trades": sum(row["trades"] for row in ledgers), "ledgers": ledgers,
              "error_count": len(audit.failures), "errors": audit.failures,
              "max_numeric_differences": audit.max_numeric_error}
    output = args.study / "independent_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "pass", "checks", "total_trades", "error_count")}))
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
