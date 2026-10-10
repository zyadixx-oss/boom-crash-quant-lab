#!/usr/bin/env python3
"""Independent standard-library audit of native300 later-period ledgers.

No ML, study engine or research metrics imports. Declared missing minutes stay
missing; unknown trade paths retain censoring rather than receiving an R value.
Model features/fitting/scores and bootstrap inference are outside this audit.
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
from scripts.verify_spike_payoff import Audit, FLAGS, causal_atr, epoch, read_minutes, sha256, summary
from scripts.verify_spike_timed import scalar
from scripts.verify_spike_learned import number, read_json, safety_inputs, source_path

SYMBOLS = ("BOOM300N", "CRASH300N")
MODES = ("SPIKE", "DRIFT")
COHORT = "fresh_temporal180"
PRIMARY = {"stop_atr": 2.0, "max_hold_minutes": 15, "entry_delay_minutes": 1,
           "round_trip_cost_atr": .10, "fill_mode": "adverse_extreme"}


def read_ohlc(path, allow_off_grid=False):
    minutes, excluded, duplicates = {}, [], 0
    previous = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp = int(row["epoch"])
            values = tuple(number(row[key], f"{path.name}/{timestamp}/{key}")
                           for key in ("open", "high", "low", "close"))
            opening, high, low, close = values
            if not all(value > 0 for value in values) or not low <= min(opening, close) <= max(opening, close) <= high:
                raise ValueError(f"Invalid source OHLC at {timestamp}")
            if previous is not None and timestamp < previous:
                raise ValueError("Source timestamps must remain sorted")
            previous = timestamp
            if timestamp % 60:
                if not allow_off_grid:
                    raise ValueError("Normalized source contains an off-grid row")
                excluded.append(timestamp)
                continue
            if timestamp in minutes:
                if minutes[timestamp] != values:
                    raise ValueError("Conflicting source duplicate")
                duplicates += 1
                continue
            minutes[timestamp] = values
    if not minutes:
        raise ValueError("Empty source")
    return minutes, excluded, duplicates


def path_or_censor(minutes, signal, atr, side, config):
    """Walk via the independent scalar helper; preserve its first missing quote."""
    try:
        return scalar(minutes, signal, atr, side, config)
    except KeyError as exc:
        missing = exc.args[0]
        entry_time = signal + 60 * config["entry_delay_minutes"]
        if missing == entry_time:
            raise ValueError("Missing-entry signals must not have a trade ledger row") from exc
        planned_end = entry_time + 60 * config["max_hold_minutes"]
        if not entry_time < missing < planned_end:
            raise ValueError("Unexpected missing timestamp in scalar path") from exc
        return {"signal_time": signal, "entry_time": entry_time, "exit_time": planned_end,
                "entry": minutes[entry_time][0], "exit": None, "atr": atr,
                "gross_R": None, "net_R": None, "reason": "censored_missing_path",
                "ambiguous": False, "censored": True,
                "holding_minutes": config["max_hold_minutes"],
                "planned_end": planned_end, "missing_time": missing}


def observed_value(saved, key, expected, label):
    if expected is None:
        return None if saved[key] == "" else saved[key]
    if key in ("signal_time", "entry_time", "exit_time", "planned_end", "missing_time") and expected != "":
        return epoch(saved[key])
    if isinstance(expected, bool):
        if saved[key] not in ("True", "False"):
            raise ValueError(f"Invalid boolean: {label}/{key}")
        return saved[key] == "True"
    if isinstance(expected, (int, float)):
        return number(saved[key], f"{label}/{key}")
    return saved[key]


def mixed_summary(rows, start, end, stop_atr, risk_fraction):
    completed = [row for row in rows if not row["censored"]]
    result = summary(completed, start, end, stop_atr, risk_fraction)
    result.update(trades=len(rows), censored=sum(row["censored"] for row in rows),
                  ambiguous=sum(row["ambiguous"] for row in rows))
    return result


def verify(study, audit):
    result = read_json(study / "results.json")
    selection = read_json(study / "selection.json")
    declaration = read_json(study / "declaration.json")
    for label, value in (("results", result), ("selection", selection), ("declaration", declaration)):
        audit.safety(value, label)
        audit.equal(f"safety/{label}", value["safety"], dict.fromkeys(FLAGS, False))
    audit.equal("study/result_stage", result["stage"], "native300_chronological_evaluation")
    audit.equal("study/selection_stage", selection["stage"], "frozen_native300_development")
    audit.equal("study/declaration_stage", declaration["stage"], "native300_predevelopment_declaration")
    audit.equal("study/symbols", sorted(result["symbols"]), sorted(SYMBOLS))
    audit.equal("study/selection_symbols", sorted(selection["symbols"]), sorted(SYMBOLS))
    audit.equal("study/chronology", result["chronological_old_to_new"], True)
    audit.equal("study/prospective", result["prospective_paper"], False)
    audit.equal("study/bootstrap_repeats", result["bootstrap_repeats"], 9999)
    selection_hash = sha256(study / "selection.json")
    declaration_hash = sha256(study / "declaration.json")
    audit.equal("hash/selection", selection_hash, result["selection_sha256"])
    audit.equal("hash/selection_file", selection_hash, (study / "selection.sha256").read_text().strip())
    audit.equal("hash/declaration", declaration_hash, selection["declaration_sha256"])
    audit.equal("hash/result_declaration", declaration_hash, result["declaration_sha256"])
    audit.equal("hash/declaration_file", declaration_hash, (study / "declaration.sha256").read_text().strip())
    audit.equal("hash/result_code_metadata", result["code_hashes"], selection["code_hashes"])
    for field in ("code_hashes", "sources", "intervals", "config", "feature_formulas", "bootstrap_repeats"):
        audit.equal(f"declaration/{field}", selection[field], declaration[field])
    for field in ("fresh_features_evaluated", "fresh_payoffs_evaluated"):
        audit.equal(f"declaration/{field}", declaration[field], False)
        audit.equal(f"selection/{field}", selection[field], False)
    audit.equal("config/primary", selection["config"], PRIMARY)
    audit.equal("config/user_target", selection["user_target"],
                {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000})
    for name, expected in selection["code_hashes"].items():
        audit.equal(f"hash/code/{name}", sha256(source_path(name)), expected)
    manifests = {}
    for symbol, sources in selection["sources"].items():
        manifests[symbol] = {}
        for kind, source in sources.items():
            clean = source_path(source["path"])
            manifest_path = source_path(source["manifest"])
            manifest = read_json(manifest_path)
            manifests[symbol][kind] = manifest
            audit.safety(manifest, f"{symbol}/{kind}/manifest")
            audit.equal(f"hash/{symbol}/{kind}/clean", sha256(clean), source["sha256"])
            audit.equal(f"hash/{symbol}/{kind}/manifest", sha256(manifest_path), source["manifest_sha256"])
            audit.equal(f"hash/{symbol}/{kind}/manifest_clean", manifest["normalized_sha256"], source["sha256"])
            audit.equal(f"hash/{symbol}/{kind}/raw", sha256(source_path(manifest["raw_file"])), manifest["raw_sha256"])
            audit.equal(f"{symbol}/{kind}/manifest_symbol", manifest["symbol"], symbol)
            audit.equal(f"{symbol}/{kind}/manifest_rows", manifest["normalized_rows"], source["rows"])
            audit.equal(f"{symbol}/{kind}/manifest_start", manifest["first_epoch"], source["first_epoch"])
            audit.equal(f"{symbol}/{kind}/manifest_last", manifest["last_epoch"], source["last_epoch"])
            audit.equal(f"{symbol}/{kind}/gap_metadata", manifest["gaps"], source["declared_gaps"])
            audit.equal(f"{symbol}/{kind}/exclusion_metadata", manifest["excluded_rows"], source["excluded_raw_rows"])
            audit.equal(f"{symbol}/{kind}/no_interpolation", manifest["fills_or_interpolations"], False)
            audit.equal(f"{symbol}/{kind}/no_authentication", manifest["authentication_used"], False)
    ledgers = []
    for symbol in SYMBOLS:
        sources = selection["sources"][symbol]
        old, fresh = sources["old"], sources["fresh"]
        old_minutes = read_minutes(source_path(old["path"]))
        audit.equal(f"{symbol}/old_start", min(old_minutes), old["first_epoch"])
        audit.equal(f"{symbol}/old_last", max(old_minutes), old["last_epoch"])
        audit.equal(f"{symbol}/old_rows", len(old_minutes), old["rows"])
        audit.equal(f"{symbol}/old_complete180", len(old_minutes), 180 * 1440)
        del old_minutes
        audit.equal(f"{symbol}/adjacency", old["last_epoch"] + 60, fresh["first_epoch"])
        audit.equal(f"{symbol}/chronological_disjoint", old["last_epoch"] < fresh["first_epoch"], True)
        cohort = result["symbols"][symbol]["cohorts"][COHORT]
        audit.equal(f"{symbol}/evaluation_symbol", cohort["symbol"], symbol)
        audit.equal(f"{symbol}/models", sorted(cohort["models"]), sorted(MODES))
        audit.equal(f"{symbol}/frozen_source", cohort["source"], fresh)
        start, end = epoch(cohort["start"]), epoch(cohort["end"])
        clean = source_path(fresh["path"])
        manifest = manifests[symbol]["fresh"]
        raw = source_path(manifest["raw_file"])
        minutes, clean_excluded, clean_duplicates = read_ohlc(clean)
        raw_minutes, raw_excluded, raw_duplicates = read_ohlc(raw, allow_off_grid=True)
        audit.equal(f"{symbol}/raw_to_clean_normalization", raw_minutes == minutes, True)
        audit.equal(f"{symbol}/raw_duplicates", raw_duplicates, manifest["equal_duplicates_removed"])
        audit.equal(f"{symbol}/raw_excluded_epochs", raw_excluded, [row["epoch"] for row in manifest["excluded_rows"]])
        del raw_minutes
        audit.equal(f"{symbol}/clean_excluded", clean_excluded, [])
        audit.equal(f"{symbol}/clean_duplicates", clean_duplicates, 0)
        missing = sorted(set(range(start, end, 60)) - set(minutes))
        declared_missing = sorted(t for gap in fresh["declared_gaps"] for t in gap["missing_epochs"])
        audit.equal(f"{symbol}/missing_epochs", missing, declared_missing)
        audit.equal(f"{symbol}/missing_count", len(missing), fresh["declared_missing_minutes"])
        audit.equal(f"{symbol}/cohort_missing_count", len(missing), cohort["audit"]["missing_minutes"])
        audit.equal(f"{symbol}/grid_count", len(minutes) + len(missing), fresh["expected_grid_rows"])
        audit.equal(f"{symbol}/full180_grid", len(minutes) + len(missing), 180 * 1440)
        audit.equal(f"{symbol}/rows", len(minutes), cohort["audit"]["source_rows"])
        audit.equal(f"{symbol}/frozen_rows", len(minutes), fresh["rows"])
        audit.equal(f"{symbol}/start", min(minutes), start)
        audit.equal(f"{symbol}/end", max(minutes) + 60, end)
        audit.equal(f"{symbol}/frozen_start", start, fresh["first_epoch"])
        audit.equal(f"{symbol}/frozen_end", end, fresh["last_epoch"] + 60)
        audit.equal(f"{symbol}/full180_duration", end - start, 180 * 86400)
        audit.equal(f"{symbol}/cohort_hash", cohort["audit"]["sha256"], fresh["sha256"])
        atr_cache = {}
        for mode in MODES:
            label = f"{symbol}/{mode}"
            failures_before = len(audit.failures)
            model = cohort["models"][mode]
            frozen_model = selection["symbols"][symbol]["models"][mode]
            config = model["config"]
            audit.equal(f"{label}/mode", model["mode"], mode)
            audit.equal(f"{label}/frozen_config", config, frozen_model["config"])
            audit.equal(f"{label}/primary_config", config, PRIMARY)
            audit.equal(f"{label}/audit_config", model["audit"]["config"], config)
            audit.equal(f"{label}/purge_minutes", model["audit"]["purge_minutes"], 31)
            audit.equal(f"{label}/development_eligibility", model["development_eligible"],
                        frozen_model["development_eligible"])
            audit.equal(f"{label}/selected_direction", model["selected_direction"],
                        mode == selection["symbols"][symbol]["selected_direction"])
            fit = frozen_model["final_model"]
            training_end = epoch(fit["training_end"])
            audit.equal(f"{label}/training_before_fresh", training_end < start, True)
            audit.equal(f"{label}/training_end_in_old", training_end <= old["last_epoch"] + 60, True)
            audit.equal(f"{label}/training_purge", fit["label_purge_minutes"], 31)
            audit.equal(f"{label}/latest_training_issue_purge",
                        epoch(fit["training_latest_issue"]) + 31 * 60 <= training_end, True)
            audit.equal(f"{label}/latest_training_planned_end",
                        epoch(fit["training_latest_planned_end"]) <= training_end, True)
            side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
            threshold = number(fit["threshold"], f"{label}/threshold")
            signals_path = study / f"{symbol}_{COHORT}_{mode}_signals.csv"
            trades_path = study / f"{symbol}_{COHORT}_{mode}_trades.csv"
            signals = {}
            previous_signal = None
            with signals_path.open(newline="") as stream:
                for ordinal, saved in enumerate(csv.DictReader(stream)):
                    signal = epoch(saved["signal_time"])
                    item = f"{label}/signals/{ordinal}"
                    audit.equal(f"{item}/clock_UTC00_30", signal % 1800, 0)
                    audit.equal(f"{item}/inside_partition", start <= signal < end, True)
                    audit.equal(f"{item}/unique", signal not in signals, True)
                    audit.equal(f"{item}/side", number(saved["side"], item), side)
                    audit.equal(f"{item}/variant", saved["variant"], f"LEARNED_{mode}")
                    score = number(saved["score"], f"{item}/score")
                    audit.equal(f"{item}/frozen_cutoff", score >= threshold and score > 0, True)
                    if previous_signal is not None:
                        audit.equal(f"{item}/spacing", signal - previous_signal >= 1800, True)
                    if signal not in atr_cache:
                        atr_cache[signal] = causal_atr(minutes, signal)
                    audit.equal(f"{item}/causal_ATR", number(saved["atr"], item), atr_cache[signal])
                    audit.equal(f"{item}/candle_close", number(saved["signal_close"], item), minutes[signal - 60][3])
                    signals[signal] = saved
                    previous_signal = signal
            admitted = {signal for signal in signals if signal + 31 * 60 <= end}
            missing_entries = {signal for signal in admitted if signal + 60 not in minutes}
            expected_signals = admitted - missing_entries
            purged = len(signals) - len(admitted)
            reconstructed, observed_signals = [], []
            previous_signal = previous_exit = None
            with trades_path.open(newline="") as stream:
                for ordinal, saved in enumerate(csv.DictReader(stream)):
                    signal = epoch(saved["signal_time"])
                    item = f"{label}/trades/{ordinal}"
                    audit.equal(f"{item}/saved_signal", signal in signals, True)
                    audit.equal(f"{item}/clock_UTC00_30", signal % 1800, 0)
                    audit.equal(f"{item}/partition_purge", start <= signal < end and signal + 31 * 60 <= end, True)
                    audit.equal(f"{item}/variant", saved["variant"], f"LEARNED_{mode}")
                    if signal not in atr_cache:
                        atr_cache[signal] = causal_atr(minutes, signal)
                    calculated = path_or_censor(minutes, signal, atr_cache[signal], side, config)
                    if previous_signal is not None:
                        audit.equal(f"{item}/spacing", signal - previous_signal >= 1800, True)
                        audit.equal(f"{item}/no_overlap", calculated["entry_time"] >= previous_exit, True)
                    for key, expected in calculated.items():
                        audit.equal(f"{item}/{key}", observed_value(saved, key, expected, item), expected)
                    reconstructed.append(calculated)
                    observed_signals.append(signal)
                    previous_signal, previous_exit = signal, calculated["exit_time"]
            audit.equal(f"{label}/signal_to_ledger_count", len(observed_signals), len(expected_signals))
            audit.equal(f"{label}/signal_to_ledger_unique", len(set(observed_signals)), len(observed_signals))
            audit.equal(f"{label}/signal_to_ledger_members", sorted(observed_signals), sorted(expected_signals))
            execution = model["audit"]
            censored = sum(row["censored"] for row in reconstructed)
            audit.equal(f"{label}/issued", execution["issued"], len(signals))
            audit.equal(f"{label}/purged", execution["purged"], purged)
            audit.equal(f"{label}/missing_entry", execution["missing_entry"], len(missing_entries))
            audit.equal(f"{label}/filled", execution["filled"], len(reconstructed))
            audit.equal(f"{label}/completed", execution["completed"], len(reconstructed) - censored)
            audit.equal(f"{label}/censored", execution["censored"], censored)
            for field in ("outside_partition", "overlap_skipped", "ambiguous"):
                audit.equal(f"{label}/audit/{field}", execution[field], 0)
            independent = mixed_summary(reconstructed, start, end, config["stop_atr"],
                                        model["metrics"]["illustrative_risk_fraction"])
            for field, expected in independent.items():
                audit.equal(f"{label}/metrics/{field}", model["metrics"][field], expected)
            audit.equal(f"{label}/user_target_observed", model["user_target_observed"],
                        independent["completed"] >= 1000 and (independent["profit_factor"] or 0) >= 1.5)
            ledgers.append({"symbol": symbol, "mode": mode, "issued_signals": len(signals),
                            "partition_purged_signals": purged, "missing_entries": len(missing_entries),
                            "trades": len(reconstructed), "completed": len(reconstructed) - censored,
                            "censored": censored, "ledger_sha256": sha256(trades_path), "signals_sha256": sha256(signals_path),
                            "source_missing_minutes": len(missing), "mean_net_R": independent["mean_net_R"],
                            "profit_factor": independent["profit_factor"], "mismatches": len(audit.failures) - failures_before})
    return selection_hash, ledgers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_native300_20261005")
    args = parser.parse_args()
    if not (args.study / "results.json").is_file():
        parser.error("Completed results.json required before reading native300 outcomes")
    audit = Audit()
    selection_hash = None
    ledgers = []
    try:
        safety_inputs(audit)
        if audit.failures:
            raise ValueError("Requires all four runtime safety flags=false")
        selection_hash, ledgers = verify(args.study, audit)
    except Exception as exc:
        audit.failures.append({"check": "audit_exception", "error": f"{type(exc).__name__}: {exc}"})
    passed = not audit.failures
    helpers = ("scripts/verify_spike_payoff.py", "scripts/verify_spike_timed.py", "scripts/verify_spike_learned.py")
    report = {"status": "PASS" if passed else "FAIL", "pass": passed,
              "run_utc": datetime.now(timezone.utc).isoformat(), "safety": dict.fromkeys(FLAGS, False),
              "implementation": "Independent standard-library scalar quote-path and arithmetic audit; no ML, engine or metrics imports",
              "script_sha256": sha256(Path(__file__)), "helper_sha256": {name: sha256(ROOT / name) for name in helpers},
              "selection_sha256": selection_hash, "results_sha256": sha256(args.study / "results.json"),
              "scope": ["all four later-period native300 signal/trade ledgers, including empty ledgers",
                        "frozen declaration/selection/code/data/manifest hashes and chronological source adjacency",
                        "independent raw normalization exclusions and missing-minute preservation",
                        "closed M5 ATR14, signal UTC00/30, mode side, saved cutoff and candle close",
                        "saved issuance to ledger counts after fixed31-minute purge and missing entry",
                        "scalar entry/stop/time path, censored unknown paths, R arithmetic, spacing and occupancy",
                        "counts/means/PF/day cluster SE/illustrative closed equity summary arithmetic"],
              "not_verified": ["nineteen ML features independently regenerated", "model fitting/scalers/coefs/scores independently regenerated",
                               "completeness against all possible ML opportunities", "clock-baseline replay",
                               "bootstrap intervals/p-values/Holm inference", "sensitivity ledgers", "actual fills/costs/cash profits"],
              "checks": audit.checks, "saved_false_safety_values": audit.safety_values_checked,
              "total_trades": sum(row["trades"] for row in ledgers), "ledgers": ledgers,
              "error_count": len(audit.failures), "errors": audit.failures,
              "max_numeric_differences": audit.max_numeric_error}
    (args.study / "independent_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "pass", "checks", "total_trades", "error_count")}))
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
