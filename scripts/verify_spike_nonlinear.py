#!/usr/bin/env python3
"""Independent stdlib audit of saved nonlinear600 primary quote-path ledgers.

Reads historical outcomes only when completed results.json exists. Imports no
ML, research metrics, feature generator, fitting code or replay engine. Saved
scores, p-values and confidence intervals are not independently regenerated.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_spike_payoff import Audit, FLAGS, causal_atr, epoch, sha256
from scripts.verify_spike_learned import number, read_json, safety_inputs, source_path
from scripts.verify_spike_native300 import PRIMARY, MODES, mixed_summary, observed_value, path_or_censor, read_ohlc
from scripts.verify_spike_multiframe import holm_values, timestamp

SYMBOLS = ("BOOM600", "CRASH600")
FAMILIES = ("BOOST44", "RIDGE44", "RIDGE19")
COHORT = "later_temporal180"
BOOST_PARAMS = {"n_trees": 100, "learning_rate": .05, "max_depth": 3,
                "min_leaf": 200, "n_bins": 16, "leaf_regularization": 20., "quantile": .75}
PUBLIC_PAGE_KEYS = {"ticks_history", "count", "end", "style", "granularity", "req_id"}


def model_keys():
    return [f"{family}_{mode}" for mode in MODES for family in FAMILIES]


def hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def audit_pages(symbol, source, manifest, audit, label):
    pages_path = source_path(source["page_audit"])
    pages = read_json(pages_path)
    audit.equal(f"{label}/page_hash", sha256(pages_path), source["page_audit_sha256"])
    audit.equal(f"{label}/page_count", len(pages), manifest["page_count"])
    previous_oldest = None
    covered_ends = {}
    for ordinal, page in enumerate(pages):
        request = page["request"]
        item = f"{label}/page/{ordinal}"
        audit.equal(f"{item}/public_fields", sorted(request), sorted(PUBLIC_PAGE_KEYS))
        audit.equal(f"{item}/symbol", request["ticks_history"], symbol)
        audit.equal(f"{item}/style", request["style"], "candles")
        audit.equal(f"{item}/granularity", request["granularity"], 60)
        audit.equal(f"{item}/count", 0 < page["returned_rows"] <= request["count"], True)
        audit.equal(f"{item}/bounds",
                    page["oldest_epoch"] <= page["newest_epoch"] <= request["end"], True)
        audit.equal(f"{item}/hash", hex_digest(page["response_sha256"]), True)
        expected_end = (manifest["requested_cutoff_exclusive_epoch"] - 1 if previous_oldest is None
                        else previous_oldest - 1)
        audit.equal(f"{item}/chronological_chain", request["end"], expected_end)
        previous_oldest = page["oldest_epoch"]
        covered_ends[request["end"]] = covered_ends.get(request["end"], 0) + 1
    audit.equal(f"{label}/history_reaches_start",
                previous_oldest is not None and previous_oldest <= manifest["requested_start_epoch"], True)
    errors = manifest["request_errors"]
    audit.equal(f"{label}/retained_errors", source["recovered_request_errors"], errors)
    for ordinal, error in enumerate(errors):
        item = f"{label}/recovered_error/{ordinal}"
        audit.equal(f"{item}/schema", sorted(error), ["attempt", "end", "error"])
        audit.equal(f"{item}/attempt", isinstance(error["attempt"], int) and 1 <= error["attempt"] <= 4, True)
        prefix, payload = error["error"].split(": ", 1)
        payload = json.loads(payload)
        audit.equal(f"{item}/error_kind", prefix, "RuntimeError")
        audit.equal(f"{item}/code", payload["code"], "RateLimit")
        audit.equal(f"{item}/same_end_success", covered_ends.get(error["end"], 0), 1)
    return len(errors)


def audit_source(symbol, kind, source, audit):
    label = f"{symbol}/{kind}/source"
    clean_path = source_path(source["path"])
    manifest_path = source_path(source["manifest"])
    manifest = read_json(manifest_path)
    audit.safety(manifest, label)
    audit.equal(f"{label}/flags", manifest["safety"], dict.fromkeys(FLAGS, False))
    audit.equal(f"{label}/clean_hash", sha256(clean_path), source["sha256"])
    audit.equal(f"{label}/manifest_hash", sha256(manifest_path), source["manifest_sha256"])
    audit.equal(f"{label}/manifest_clean_hash", manifest["normalized_sha256"], source["sha256"])
    audit.equal(f"{label}/raw_path", source["raw_file"], manifest["raw_file"])
    audit.equal(f"{label}/raw_hash", sha256(source_path(manifest["raw_file"])), manifest["raw_sha256"])
    audit.equal(f"{label}/frozen_raw_hash", source["raw_sha256"], manifest["raw_sha256"])
    audit.equal(f"{label}/symbol", manifest["symbol"], symbol)
    for field in ("normalization_valid", "finite_positive_ohlc", "canonical_utc_minute_grid"):
        audit.equal(f"{label}/{field}", manifest[field], True)
    for field in ("fills_or_interpolations", "authentication_used"):
        audit.equal(f"{label}/{field}", manifest[field], False)
    minutes, excluded, duplicates = read_ohlc(clean_path)
    raw, raw_excluded, raw_duplicates = read_ohlc(source_path(manifest["raw_file"]), allow_off_grid=True)
    audit.equal(f"{label}/raw_normalization", minutes == raw, True)
    audit.equal(f"{label}/raw_excluded", raw_excluded, [row["epoch"] for row in manifest["excluded_rows"]])
    audit.equal(f"{label}/raw_duplicates", raw_duplicates, manifest["equal_duplicates_removed"])
    audit.equal(f"{label}/raw_rows", len(raw) + len(raw_excluded) + raw_duplicates, manifest["raw_rows"])
    del raw
    audit.equal(f"{label}/clean_excluded", excluded, [])
    audit.equal(f"{label}/clean_duplicates", duplicates, 0)
    start, end = min(minutes), max(minutes) + 60
    audit.equal(f"{label}/requested_start", start, manifest["requested_start_epoch"])
    audit.equal(f"{label}/requested_end", end, manifest["requested_cutoff_exclusive_epoch"])
    audit.equal(f"{label}/first", start, source["first_epoch"])
    audit.equal(f"{label}/last", end - 60, source["last_epoch"])
    audit.equal(f"{label}/duration180", end - start, 180 * 86400)
    audit.equal(f"{label}/row_count", len(minutes), source["rows"])
    audit.equal(f"{label}/manifest_count", len(minutes), manifest["normalized_rows"])
    missing = sorted(set(range(start, end, 60)) - set(minutes))
    declared = sorted(t for gap in source["declared_gaps"] for t in gap["missing_epochs"])
    audit.equal(f"{label}/missing_epochs", missing, declared)
    audit.equal(f"{label}/missing_count", len(missing), source["declared_missing_minutes"])
    audit.equal(f"{label}/complete_grid", len(minutes) + len(missing), source["expected_grid_rows"])
    audit.equal(f"{label}/full180_grid", source["expected_grid_rows"], 180 * 1440)
    audit.equal(f"{label}/gap_metadata", manifest["gaps"], source["declared_gaps"])
    audit.equal(f"{label}/excluded_metadata", manifest["excluded_rows"], source["excluded_raw_rows"])
    for ordinal, gap in enumerate(manifest["gaps"]):
        item = f"{label}/gap/{ordinal}"
        expected = list(range(gap["previous_epoch"] + 60, gap["next_epoch"], 60))
        audit.equal(f"{item}/epochs", gap["missing_epochs"], expected)
        audit.equal(f"{item}/count", gap["missing_bars"], len(expected))
        audit.equal(f"{item}/bounds_present", gap["previous_epoch"] in minutes and gap["next_epoch"] in minutes, True)
    error_count = audit_pages(symbol, source, manifest, audit, label)
    return minutes, {"symbol": symbol, "kind": kind, "clean_sha256": source["sha256"],
                     "raw_sha256": source["raw_sha256"], "rows": len(minutes),
                     "missing_minutes": len(missing), "recovered_rate_limit_errors": error_count}


def audit_fit(fit, family, old, development_end, audit, label):
    estimator = fit["estimator"]
    count = 19 if family == "RIDGE19" else 44
    audit.equal(f"{label}/family", fit["family"], family)
    audit.equal(f"{label}/feature_count", len(fit["feature_names"]), count)
    audit.equal(f"{label}/nested_order", estimator["feature_names"], fit["feature_names"])
    audit.equal(f"{label}/fit_rows", estimator["fit_rows"], fit["training_completed_labels"])
    audit.equal(f"{label}/minimum_training_labels", fit["training_completed_labels"] >= 1000, True)
    audit.equal(f"{label}/training_start", epoch(fit["training_start"]), old["first_epoch"])
    audit.equal(f"{label}/training_end", epoch(fit["training_end"]), development_end)
    audit.equal(f"{label}/purge", fit["label_purge_minutes"], 31)
    audit.equal(f"{label}/latest_training_issue",
                epoch(fit["training_latest_issue"]) + 31 * 60 <= development_end, True)
    audit.equal(f"{label}/latest_training_planned_end",
                epoch(fit["training_latest_planned_end"]) <= development_end, True)
    audit.equal(f"{label}/threshold_nonnegative", number(estimator["threshold"], label) >= 0, True)
    audit.equal(f"{label}/score_kind", estimator["score_kind"], "continuous_uncalibrated_score")
    for field in ("features_clipped", "target_clipped"):
        audit.equal(f"{label}/{field}", estimator[field], False)
    audit.equal(f"{label}/matrix_target_hash", hex_digest(fit["matrix_and_target_sha256"]), True)
    if family == "BOOST44":
        audit.equal(f"{label}/boost_parameters", estimator["parameters"], BOOST_PARAMS)
        audit.equal(f"{label}/tree_count", len(estimator["trees"]), BOOST_PARAMS["n_trees"])
        payload = {key: value for key, value in estimator.items() if key != "integrity_sha256"}
        integrity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode("utf-8")).hexdigest()
        audit.equal(f"{label}/serialized_integrity", estimator["integrity_sha256"], integrity)
    else:
        audit.equal(f"{label}/ridge_penalty", estimator["penalty"], .1)
        audit.equal(f"{label}/ridge_quantile", estimator["quantile"], .75)
    return number(estimator["threshold"], label)


def ledger(study, symbol, family, mode, row, fit, minutes, start, end, audit):
    key = f"{family}_{mode}"
    label = f"{symbol}/{key}/{COHORT}"
    before = len(audit.failures)
    prefix = f"{symbol}_{COHORT}_{key}"
    signals_path = study / f"{prefix}_signals.csv"
    trades_path = study / f"{prefix}_trades.csv"
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    threshold = number(fit["estimator"]["threshold"], label)
    audit.equal(f"{label}/training_before_later", epoch(fit["training_end"]) < start, True)
    audit.equal(f"{label}/family", row["family"], family)
    audit.equal(f"{label}/mode", row["mode"], mode)
    audit.equal(f"{label}/primary_config", row["config"], PRIMARY)
    execution = row["audit"]
    audit.equal(f"{label}/execution_config", execution["config"], PRIMARY)
    audit.equal(f"{label}/execution_purge", execution["purge_minutes"], 31)
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
            audit.equal(f"{item}/variant", saved["variant"], f"NONLINEAR_{key}")
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
            audit.equal(f"{item}/variant", saved["variant"], f"NONLINEAR_{key}")
            atr = atr_cache.get(signal)
            if atr is None:
                atr = causal_atr(minutes, signal)
            calculated = path_or_censor(minutes, signal, atr, side, PRIMARY)
            if previous_signal is not None:
                audit.equal(f"{item}/spacing", signal - previous_signal >= 1800, True)
                audit.equal(f"{item}/occupancy", calculated["entry_time"] >= previous_exit, True)
            for field, expected in calculated.items():
                audit.equal(f"{item}/{field}", observed_value(saved, field, expected, item), expected)
            reconstructed.append(calculated)
            observed.append(signal)
            previous_signal, previous_exit = signal, calculated["exit_time"]
    audit.equal(f"{label}/signal_members", sorted(observed), sorted(expected_signals))
    audit.equal(f"{label}/signal_unique", len(observed), len(set(observed)))
    censored = sum(item["censored"] for item in reconstructed)
    expected_execution = {"issued": len(signals), "purged": len(signals) - len(admitted),
                          "missing_entry": len(missing_entries), "filled": len(reconstructed),
                          "completed": len(reconstructed) - censored, "censored": censored,
                          "outside_partition": 0, "overlap_skipped": 0, "ambiguous": 0}
    for field, expected in expected_execution.items():
        audit.equal(f"{label}/execution/{field}", execution[field], expected)
    independent = mixed_summary(reconstructed, start, end, PRIMARY["stop_atr"],
                                row["metrics"]["illustrative_risk_fraction"])
    for field, expected in independent.items():
        audit.equal(f"{label}/metrics/{field}", row["metrics"][field], expected)
    audit.equal(f"{label}/observed_target", row["user_target_observed"],
                independent["completed"] >= 1000 and (independent["profit_factor"] or 0) >= 1.5)
    audit.equal(f"{label}/actual_money_profit", row["actual_money_profit"], "NOT TESTED")
    audit.equal(f"{label}/historical_rejections", row["historical_candidate"], not row["rejection_reasons"])
    if row["historical_candidate"]:
        audit.equal(f"{label}/eligible_promotion", row["development_eligible"], True)
        audit.equal(f"{label}/minimum_promotion_n", independent["completed"] >= 1000, True)
    if row["supports_expected_pf_1_5"]:
        audit.equal(f"{label}/expected_pf_requires_candidate", row["historical_candidate"], True)
    return {"symbol": symbol, "family": family, "mode": mode, "cohort": COHORT,
            "trades": len(reconstructed), "completed": len(reconstructed) - censored,
            "censored": censored, "issued_signals": len(signals), "purged": len(signals) - len(admitted),
            "missing_entries": len(missing_entries), "ledger_sha256": sha256(trades_path),
            "signals_sha256": sha256(signals_path), "mean_net_R": independent["mean_net_R"],
            "profit_factor": independent["profit_factor"], "mismatches": len(audit.failures) - before}


def verify(study, audit):
    result, selection, declaration = [read_json(study / name) for name in
                                      ("results.json", "selection.json", "declaration.json")]
    for label, payload in (("results", result), ("selection", selection), ("declaration", declaration)):
        audit.safety(payload, label)
        audit.equal(f"{label}/safety", payload["safety"], dict.fromkeys(FLAGS, False))
    audit.equal("result/stage", result["stage"], "nonlinear_chronological_evaluation")
    audit.equal("selection/stage", selection["stage"], "frozen_nonlinear_development")
    audit.equal("declaration/stage", declaration["stage"], "nonlinear_predevelopment_declaration")
    audit.equal("result/symbols", sorted(result["symbols"]), sorted(SYMBOLS))
    audit.equal("selection/symbols", sorted(selection["symbols"]), sorted(SYMBOLS))
    audit.equal("selection/families", selection["families"], list(FAMILIES))
    audit.equal("selection/boost_parameters", selection["boost_parameters"], BOOST_PARAMS)
    audit.equal("selection/bootstrap", selection["bootstrap_repeats"], 9999)
    audit.equal("result/family", result["joint_fresh_hypotheses"], 12)
    audit.equal("selection/family", selection["joint_fresh_hypotheses"], 12)
    audit.equal("selection/config", selection["config"], PRIMARY)
    audit.equal("selection/user_target", selection["user_target"],
                {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000})
    selection_hash = sha256(study / "selection.json")
    declaration_hash = sha256(study / "declaration.json")
    for name, observed in (("selection", selection_hash), ("declaration", declaration_hash)):
        audit.equal(f"hash/{name}", observed, result[f"{name}_sha256"])
        audit.equal(f"hash/{name}_file", observed, (study / f"{name}.sha256").read_text().strip())
    audit.equal("hash/selection_declaration", selection["declaration_sha256"], declaration_hash)
    audit.equal("hash/result_code", result["code_hashes"], selection["code_hashes"])
    for field in declaration:
        if field != "stage":
            audit.equal(f"declaration/{field}", selection[field], declaration[field])
    for field in ("fresh_features_evaluated", "fresh_payoffs_evaluated"):
        audit.equal(f"declaration/{field}", declaration[field], False)
        audit.equal(f"selection/{field}", selection[field], False)
    for name, expected in selection["code_hashes"].items():
        audit.equal(f"hash/code/{name}", sha256(source_path(name)), expected)
    audit.equal("chronology/declaration_before_frozen",
                timestamp(declaration["run_utc"]) <= timestamp(selection["frozen_utc"]), True)
    audit.equal("chronology/evaluation_after_frozen",
                timestamp(result["run_utc"]) >= timestamp(selection["frozen_utc"]), True)
    audit.equal("result/chronological_old_to_new", result["chronological_old_to_new"], True)
    for field in ("prospective_paper", "forward_test"):
        audit.equal(f"result/{field}", result[field], False)
    audit.equal("result/actual_money", result["actual_money_profit"], "NOT TESTED")
    ledgers, sources_audited, family_rows = [], [], []
    for symbol in SYMBOLS:
        sources = selection["sources"][symbol]
        old, fresh = sources["old"], sources["fresh"]
        audit.equal(f"{symbol}/adjacent", old["last_epoch"] + 60, fresh["first_epoch"])
        old_minutes, old_audit = audit_source(symbol, "old", old, audit)
        sources_audited.append(old_audit)
        del old_minutes
        minutes, fresh_audit = audit_source(symbol, "fresh", fresh, audit)
        sources_audited.append(fresh_audit)
        chosen = selection["symbols"][symbol]["selected_model"]
        candidates = selection["symbols"][symbol]["models"]
        audit.equal(f"{symbol}/frozen_model_ids", sorted(candidates), sorted(model_keys()))
        audit.equal(f"{symbol}/selected_model_copy", result["symbols"][symbol]["selected_model"], chosen)
        eligible = [(key, item) for key, item in candidates.items() if item["development_eligible"]]
        expected_choice = (sorted(eligible, key=lambda pair: (-pair[1]["validation"]["selection_score"], pair[0]))[0][0]
                           if eligible else None)
        audit.equal(f"{symbol}/development_only_selection", chosen, expected_choice)
        development_end = epoch(selection["symbols"][symbol]["partitions"]["development"][1])
        audit.equal(f"{symbol}/development70", development_end - old["first_epoch"], 126 * 86400)
        development_audit = selection["symbols"][symbol]["audit"]
        audit.equal(f"{symbol}/development_features_only", development_audit["development_features_only"], True)
        audit.equal(f"{symbol}/development_feature_end", epoch(development_audit["feature_end_exclusive"]), development_end)
        audit.equal(f"{symbol}/development_feature_rows", development_audit["feature_source_rows"], 126 * 1440)
        cohort = result["symbols"][symbol]["cohorts"][COHORT]
        start, end = epoch(cohort["start"]), epoch(cohort["end"])
        audit.equal(f"{symbol}/cohort_source", cohort["source"], fresh)
        audit.equal(f"{symbol}/cohort_symbol", cohort["symbol"], symbol)
        audit.equal(f"{symbol}/cohort_model_ids", sorted(cohort["models"]), sorted(model_keys()))
        audit.equal(f"{symbol}/cohort_hash", cohort["audit"]["sha256"], fresh["sha256"])
        audit.equal(f"{symbol}/cohort_start", start, fresh["first_epoch"])
        audit.equal(f"{symbol}/cohort_end", end, fresh["last_epoch"] + 60)
        audit.equal(f"{symbol}/cohort_rows", cohort["audit"]["source_rows"], fresh["rows"])
        audit.equal(f"{symbol}/cohort_gaps", cohort["audit"]["missing_minutes"], fresh["declared_missing_minutes"])
        audit.equal(f"{symbol}/common_input_count", cohort["audit"]["feature_count"], 44)
        for mode in MODES:
            names44 = candidates[f"RIDGE44_{mode}"]["final_model"]["feature_names"]
            audit.equal(f"{symbol}/{mode}/boost_ridge_order", candidates[f"BOOST44_{mode}"]["final_model"]["feature_names"], names44)
            audit.equal(f"{symbol}/{mode}/base19_prefix", candidates[f"RIDGE19_{mode}"]["final_model"]["feature_names"], names44[:19])
            for estimator_family in FAMILIES:
                key = f"{estimator_family}_{mode}"
                candidate, row = candidates[key], cohort["models"][key]
                fit = candidate["final_model"]
                audit_fit(fit, estimator_family, old, development_end, audit, f"{symbol}/{key}/final_fit")
                audit.equal(f"{symbol}/{key}/development_eligibility", row["development_eligible"], candidate["development_eligible"])
                audit.equal(f"{symbol}/{key}/selected", row["selected_model"], key == chosen)
                ledgers.append(ledger(study, symbol, estimator_family, mode, row, fit, minutes, start, end, audit))
                family_rows.append(row)
                metric = row["metrics"]
                audit.equal(f"{symbol}/{key}/clock_conjunction", metric["clock_conjunction_p"], max(metric["day_p"], metric["weekly_p"]))
                if estimator_family != "BOOST44":
                    audit.equal(f"{symbol}/{key}/conjunction", metric["p"], metric["clock_conjunction_p"])
            boost = cohort["models"][f"BOOST44_{mode}"]
            reference_p = [boost["metrics"]["clock_conjunction_p"]]
            audit.equal(f"{symbol}/{mode}/linear_reference_ids", sorted(boost["linear_references"]), ["RIDGE19", "RIDGE44"])
            for reference_family, reference in boost["linear_references"].items():
                linear = cohort["models"][f"{reference_family}_{mode}"]
                audit.equal(f"{symbol}/{mode}/{reference_family}/copied_metrics", reference["metrics"], linear["metrics"])
                audit.equal(f"{symbol}/{mode}/{reference_family}/copied_audit", reference["audit"], linear["audit"])
                reference_p.extend((reference["day_inference"]["p"], reference["weekly_inference"]["weekly_p"]))
            audit.equal(f"{symbol}/{mode}/boost_conjunction", boost["metrics"]["p"], max(reference_p))
            audit.equal(f"{symbol}/{mode}/nonlinear_added_value", boost["nonlinear_added_value"], boost["historical_candidate"])
    audit.equal("joint_holm/family_size", len(family_rows), 12)
    adjusted_values = holm_values([row["metrics"]["p"] for row in family_rows])
    for ordinal, (row, adjusted) in enumerate(zip(family_rows, adjusted_values)):
        audit.equal(f"joint_holm/{ordinal}", row["metrics"]["holm_p"], adjusted)
    return selection_hash, ledgers, sources_audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_nonlinear_20261005")
    args = parser.parse_args()
    if not (args.study / "results.json").is_file():
        parser.error("Completed results.json required before reading nonlinear600 outcomes")
    audit, selection_hash, ledgers, sources = Audit(), None, [], []
    try:
        safety_inputs(audit)
        if audit.failures:
            raise ValueError("Requires all four runtime safety flags=false")
        selection_hash, ledgers, sources = verify(args.study, audit)
    except Exception as exc:
        audit.failures.append({"check": "audit_exception", "error": f"{type(exc).__name__}: {exc}"})
    helpers = ("scripts/verify_spike_payoff.py", "scripts/verify_spike_timed.py",
               "scripts/verify_spike_learned.py", "scripts/verify_spike_native300.py",
               "scripts/verify_spike_multiframe.py")
    report = {"status": "FAIL" if audit.failures else "PASS", "pass": not audit.failures,
              "run_utc": datetime.now(timezone.utc).isoformat(), "safety": dict.fromkeys(FLAGS, False),
              "implementation": "Independent standard-library quote-path and summary arithmetic; no ML, features, fitting, engine or research metrics imports",
              "script_sha256": sha256(Path(__file__)), "helper_sha256": {name: sha256(ROOT / name) for name in helpers},
              "selection_sha256": selection_hash, "results_sha256": sha256(args.study / "results.json"),
              "scope": ["twelve saved later-period signal/trade ledgers on the declared common44-input clock",
                        "frozen code/source/declaration/selection hashes and recorded chronology",
                        "raw normalization, missing-minute preservation, source grid and recovered same-end RateLimit pages",
                        "serialized model integrity and training/cutoff metadata, without independently fitting or predicting",
                        "closed M5 ATR14, saved score/cutoff, side, close, UTC00/30 and 31-minute purge",
                        "saved issuance-to-ledger completeness, scalar quote-path R and censored unknowns",
                        "counts/means/PF/day-cluster SE/illustrative closed equity arithmetic",
                        "copied same-clock linear-reference consistency and joint Holm-twelve arithmetic from saved p-values"],
              "not_verified": ["44 or19 ML features independently regenerated", "model/scaler fitting or prediction scores independently regenerated",
                               "all possible ML opportunities or common44-feature eligibility", "clock-baseline replay",
                               "bootstrap confidence intervals or p-values", "secondary older-last30 ledgers",
                               "sensitivity ledgers", "actual broker fills/costs/cash profits"],
              "checks": audit.checks, "saved_false_safety_values": audit.safety_values_checked,
              "total_trades": sum(row["trades"] for row in ledgers), "ledgers": ledgers, "sources": sources,
              "error_count": len(audit.failures), "errors": audit.failures,
              "max_numeric_differences": audit.max_numeric_error}
    (args.study / "independent_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "pass", "checks", "total_trades", "error_count")}))
    raise SystemExit(bool(audit.failures))


if __name__ == "__main__":
    main()
