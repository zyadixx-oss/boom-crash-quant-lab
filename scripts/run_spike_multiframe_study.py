#!/usr/bin/env python3
"""Frozen combined H4/H1/M15/M5/M1 native300 study with joint M5 comparison."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.research.multiframe_signal import causal_multiframe_inputs, FEATURE_FORMULAS
from app.research.spike_hunter import load_m1, partitions
from app.research.payoff_metrics import paired_inference
from scripts.run_spike_timed_study import weekly_inference
from scripts.run_spike_learned_study import prepare as native_prepare
from scripts.run_spike_native300_study import (
    CFG, MODES, SYMBOLS, WALK_FORWARD, BOOTSTRAP_REPEATS, SOURCES as NATIVE_SOURCES,
    assert_offline, digest, save, source, validate_intervals, verify_prices,
    check_prepared, choose_direction, guard_stage, issued, fit_at,
    development_eligible, evaluate_one, gate, replay_timed, summarize, window, holm,
    frozen_selection as native_frozen_selection,
)


SOURCES = (*NATIVE_SOURCES, "backend/app/research/multiframe_signal.py",
           "backend/tests/test_multiframe_signal.py", "docs/SPIKE_MULTIFRAME_PROTOCOL.md",
           "scripts/run_spike_multiframe_study.py", "backend/tests/test_spike_multiframe_study.py")
DEFAULT_NATIVE_STUDY = ROOT / "docs/spike_native300_20261005"


def hashes() -> dict:
    return {name: digest(ROOT / name) for name in SOURCES}


def prepare(path, symbol):
    m1, audit = load_m1(path)
    direction = "boom" if symbol.startswith("BOOM") else "crash"
    m5, names = causal_multiframe_inputs(m1, direction)
    issues = m5.index + pd.Timedelta(minutes=5)
    clock = (issues.minute % 30 == 0) & (issues.second == 0)
    mask = m5.feature_valid & clock
    rows = m5.loc[mask].copy()
    rows.index = issues[mask]
    rows.index.name = "signal_time"
    split = partitions(m1)
    split.update({"train40": (m1.index[0], split["wf1"][0]),
                  "train50": (m1.index[0], split["wf2"][0]),
                  "train60": (m1.index[0], split["wf3"][0])})
    audit.update(causal_input_rows=len(m5), eligible_clock_rows=len(rows),
                 feature_count=len(names), frames=["H4", "H1", "M15", "M5", "M1"],
                 clock="UTC close minute00/30; complete-as-of44-input mask")
    return m1, rows, names, audit, split


def native_study_path(value) -> Path:
    path = (ROOT / Path(value)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Native reference study must remain inside the repository") from error
    return path


def native_lineage(path: Path, unevaluated: bool = False) -> tuple[dict, dict]:
    path = native_study_path(path)
    selection, fingerprint = native_frozen_selection(path)
    if unevaluated and (path / "results.json").exists():
        raise ValueError("Native fresh results preceded the required combined-timeframe freeze")
    return selection, {"study_dir": str(path.relative_to(ROOT.resolve())), "selection_sha256": fingerprint,
                       "declaration_sha256": selection["declaration_sha256"],
                       "code_hashes": selection["code_hashes"]}


def develop(args) -> None:
    guard_stage(args.output, "develop")
    safety = assert_offline()
    native, lineage = native_lineage(args.native_study, unevaluated=True)
    values, intervals = {}, {}
    for symbol in SYMBOLS:
        pair = {"old": source("data/spike_learned_transfer", symbol),
                "fresh": source("data/spike_native300_recent", symbol)}
        intervals[symbol] = validate_intervals(pair["old"], pair["fresh"])
        if pair != native["sources"][symbol]:
            raise ValueError("Combined/native studies must share identical declared source bytes")
        values[symbol] = pair
    code_hashes = hashes()
    verify_prices(values)
    declaration = {
        "stage": "combined_multiframe_predevelopment_declaration", "adaptive_round": 5,
        "run_utc": datetime.now(timezone.utc).isoformat(), "safety": safety,
        "code_hashes": code_hashes, "sources": values, "intervals": intervals,
        "native_reference": lineage, "config": asdict(CFG), "feature_formulas": FEATURE_FORMULAS,
        "fresh_features_evaluated": False, "fresh_payoffs_evaluated": False,
        "older300_payoffs_previously_seen": True, "prospective_paper": False,
        "feature_count": 44, "joint_fresh_hypotheses": 8, "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "user_target": {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000},
    }
    save(args.output / "declaration.json", declaration)
    declared_hash = digest(args.output / "declaration.json")
    (args.output / "declaration.sha256").write_text(declared_hash + "\n")
    selection = {**declaration, "stage": "frozen_combined_multiframe_development",
                 "declaration_sha256": declared_hash, "symbols": {}}
    for symbol in SYMBOLS:
        value = values[symbol]["old"]
        m1, rows, names, audit, split = prepare(ROOT / value["path"], symbol)
        check_prepared(m1, audit, value)
        if len(names) != 44:
            raise ValueError("Combined study requires exactly44 declared features")
        candidates = {}
        for mode in MODES:
            labels, label_audit = replay_timed(m1, issued(rows, symbol, mode), CFG,
                                              *split["development"], purge_minutes=31)
            folds, ledgers = [], []
            for training, test in WALK_FORWARD:
                model = fit_at(rows, labels, names, *split[training])
                trades, execution = replay_timed(m1, issued(rows, symbol, mode, model), CFG,
                                                 *split[test], purge_minutes=31)
                folds.append({"training": training, "test": test, "model": model,
                              "metrics": summarize(trades, *split[test], stop_atr=CFG.stop_atr),
                              "baseline": summarize(window(labels, *split[test]), *split[test],
                                                    stop_atr=CFG.stop_atr), "audit": execution})
                ledgers.append(trades)
            combined = pd.concat(ledgers, ignore_index=True)
            candidate = {"mode": mode, "config": asdict(CFG), "walk_forward": folds,
                         "validation": summarize(combined, split["wf1"][0], split["wf3"][1],
                                                 stop_atr=CFG.stop_atr),
                         "training_label_audit": label_audit,
                         "final_model": fit_at(rows, labels, names, *split["development"])}
            candidate["development_eligible"] = development_eligible(candidate)
            candidates[mode] = candidate
            combined.to_csv(args.output / f"{symbol}_{mode}_validation_trades.csv", index=False)
            print(f"{symbol}/{mode}: combined44 validation n={candidate['validation']['completed']} "
                  f"PF={candidate['validation']['profit_factor']} "
                  f"eligible={candidate['development_eligible']}", flush=True)
        selection["symbols"][symbol] = {
            "audit": audit, "partitions": split, "models": candidates,
            "selected_direction": choose_direction(candidates),
            "eligible_directions": sum(c["development_eligible"] for c in candidates.values()),
        }
    if hashes() != code_hashes:
        raise ValueError("Combined study code changed during development")
    verify_prices(values)
    _, current_lineage = native_lineage(args.native_study, unevaluated=True)
    if current_lineage != lineage:
        raise ValueError("Native reference changed during combined development")
    selection["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output / "selection.json", selection)
    fingerprint = digest(args.output / "selection.json")
    (args.output / "selection.sha256").write_text(fingerprint + "\n")
    print(f"MULTIFRAME_SELECTION_FROZEN {fingerprint}; both selections frozen before fresh features/payoffs", flush=True)


def frozen_selection(output: Path) -> tuple[dict, str, dict]:
    fingerprint = (output / "selection.sha256").read_text().strip()
    if digest(output / "selection.json") != fingerprint:
        raise ValueError("Frozen combined selection changed")
    selection = json.loads((output / "selection.json").read_text())
    declared_hash = (output / "declaration.sha256").read_text().strip()
    if digest(output / "declaration.json") != declared_hash or declared_hash != selection["declaration_sha256"]:
        raise ValueError("Frozen combined declaration changed")
    declaration = json.loads((output / "declaration.json").read_text())
    for name in ("code_hashes", "sources", "intervals", "native_reference", "config",
                 "feature_formulas", "bootstrap_repeats", "joint_fresh_hypotheses"):
        if declaration[name] != selection[name]:
            raise ValueError(f"Combined selection diverges from declaration: {name}")
    if selection["code_hashes"] != hashes():
        raise ValueError("Frozen combined protocol/source changed")
    for pair in selection["sources"].values():
        validate_intervals(pair["old"], pair["fresh"])
    verify_prices(selection["sources"])
    native, lineage = native_lineage(Path(selection["native_reference"]["study_dir"]))
    if lineage != selection["native_reference"] or native["sources"] != selection["sources"]:
        raise ValueError("Frozen combined native-reference lineage changed")
    return selection, fingerprint, native


def native_result(selection: dict) -> tuple[dict, str]:
    path = native_study_path(selection["native_reference"]["study_dir"]) / "results.json"
    results = json.loads(path.read_text())
    if (results["selection_sha256"] != selection["native_reference"]["selection_sha256"] or
            results["code_hashes"] != selection["native_reference"]["code_hashes"]):
        raise ValueError("Native results do not match the frozen reference selection")
    if pd.Timestamp(results["run_utc"]) < pd.Timestamp(selection["frozen_utc"]):
        raise ValueError("Native fresh evaluation started before both model selections froze")
    return results, digest(path)


def common_clock(mtf_rows, native_rows):
    common = mtf_rows.index.intersection(native_rows.index, sort=False)
    if len(common) != len(mtf_rows):
        raise ValueError("Combined44 eligible clock must retain base19 feature availability")
    return mtf_rows.loc[common], native_rows.loc[common]


def compare_m5_reference(row, trades, m1, native_rows, symbol, mode, model, start, end, repeats):
    signals = issued(native_rows, symbol, mode, model)
    reference, audit = replay_timed(m1, signals, CFG, start, end, purge_minutes=31)
    day = paired_inference(trades, reference, start, end, repeats=repeats, seed=20261005)
    week = weekly_inference(trades, reference, start, end, repeats)
    metric = row["metrics"]
    metric["clock_conjunction_p"] = metric["p"]
    metric["m5_reference_day_p"] = day["p"]
    metric["m5_reference_weekly_p"] = week["weekly_p"]
    metric["p"] = max(metric["clock_conjunction_p"], day["p"], week["weekly_p"])
    row["m5_reference"] = {"metrics": summarize(reference, start, end, stop_atr=CFG.stop_atr),
                           "audit": audit, "day_inference": day, "weekly_inference": week,
                           "comparison_basis": "per-completed-trade expectancy on common feature-valid UTCclock"}
    signal_frame = pd.DataFrame(signals, columns=["signal_time", "atr", "side", "signal_close", "variant", "score"])
    if not signal_frame.empty:
        signal_frame = signal_frame.loc[(signal_frame.signal_time >= start) & (signal_frame.signal_time < end)]
    return reference, signal_frame


def combined_gate(row, eligible) -> None:
    gate(row, eligible)
    reference = row["m5_reference"]
    lower_day = reference["day_inference"]["baseline_difference_ci95"][0]
    lower_week = reference["weekly_inference"]["weekly_difference_ci95"][0]
    excess = lower_day is not None and lower_day > 0 and lower_week is not None and lower_week > 0
    valid = (reference["metrics"]["censored"] == 0 and reference["metrics"]["invalid_uncensored"] == 0 and
             reference["audit"]["missing_entry"] == 0)
    if not excess:
        row["rejection_reasons"].append("no_positive_day_week_excess_over_common_clock_m5")
    if not valid:
        row["rejection_reasons"].append("missing_or_censored_m5_reference")
    row["historical_candidate"] = not row["rejection_reasons"]
    row["supports_expected_pf_1_5"] = row["supports_expected_pf_1_5"] and row["historical_candidate"]
    row["adds_value_over_m5_reference"] = row["historical_candidate"] and excess


def evaluate(args) -> None:
    guard_stage(args.output, "evaluate")
    if args.bootstrap != BOOTSTRAP_REPEATS:
        raise ValueError("Combined primary inference requires the declared9999 replicates")
    safety = assert_offline()
    selection, fingerprint, native = frozen_selection(args.output)
    native_results, native_results_hash = native_result(selection)
    result = {"stage": "combined_multiframe_chronological_evaluation", "adaptive_round": 5,
              "run_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": fingerprint,
              "declaration_sha256": selection["declaration_sha256"], "code_hashes": selection["code_hashes"],
              "native_reference": selection["native_reference"], "native_results_sha256": native_results_hash,
              "safety": safety, "symbols": {}, "forward_test": False, "prospective_paper": False,
              "chronological_old_to_new": True, "methodology_adaptive": True,
              "actual_money_profit": "NOT TESTED", "joint_fresh_hypotheses": 8}
    native_family, mtf_family = [], []
    joint_native = {"original_family_size": 4, "joint_family_size": 8,
                    "round4_internal_status": "REFERENCE_ONLY; nativeHolm4 is not a joint discovery claim", "symbols": {}}
    for symbol in SYMBOLS:
        models = {}
        for mode in MODES:
            row = deepcopy(native_results["symbols"][symbol]["cohorts"]["fresh_temporal180"]["models"][mode])
            row["original_native_holm_p"] = row["metrics"]["holm_p"]
            models[mode] = row
            native_family.append(row)
        joint_native["symbols"][symbol] = {"models": models}
    for symbol in SYMBOLS:
        payload = {"selected_direction": selection["symbols"][symbol]["selected_direction"], "cohorts": {}}
        for cohort, kind in (("reused_final30", "old"), ("fresh_temporal180", "fresh")):
            value = selection["sources"][symbol][kind]
            m1, rows, names, audit, split = prepare(ROOT / value["path"], symbol)
            check_prepared(m1, audit, value)
            native_m1, native_rows, native_names, native_audit, _ = native_prepare(ROOT / value["path"], symbol)
            check_prepared(native_m1, native_audit, value)
            start, end = (split["final_test"] if kind == "old" else (m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)))
            source_mtf_clock_rows, source_native_clock_rows = len(rows), len(native_rows)
            rows = rows.loc[(rows.index >= start) & (rows.index < end)]
            native_rows = native_rows.loc[(native_rows.index >= start) & (native_rows.index < end)]
            rows, common_native = common_clock(rows, native_rows)
            comparison_clock = {"mtf_eligible_clock_rows": len(rows), "full_native_eligible_clock_rows": len(native_rows),
                                "common_clock_rows": len(common_native),
                                "native_opportunities_excluded_by_mtf_features": len(native_rows) - len(common_native),
                                "source_mtf_clock_rows": source_mtf_clock_rows,
                                "source_native_clock_rows": source_native_clock_rows,
                                "count_basis": "cohort[start,end), before execution/end-purge filtering",
                                "intersection_uses_future_outcomes": False,
                                "first_mtf_issue": None if rows.empty else rows.index.min(),
                                "first_native_issue": None if native_rows.empty else native_rows.index.min()}
            models = {}
            for mode in MODES:
                candidate = selection["symbols"][symbol]["models"][mode]
                native_candidate = native["symbols"][symbol]["models"][mode]
                if names != candidate["final_model"]["feature_names"] or native_names != native_candidate["final_model"]["feature_names"]:
                    raise ValueError("Combined/native frozen feature order changed")
                row, trades, signal_frame = evaluate_one(m1, rows, symbol, mode,
                                                         candidate["final_model"], start, end, args.bootstrap)
                reference, reference_signals = compare_m5_reference(
                    row, trades, m1, common_native, symbol, mode, native_candidate["final_model"], start, end, args.bootstrap,
                )
                row.update(development_eligible=candidate["development_eligible"],
                           selected_direction=mode == payload["selected_direction"])
                models[mode] = row
                prefix = f"{symbol}_{cohort}_{mode}"
                for suffix, frame in (("trades", trades), ("signals", signal_frame),
                                      ("m5_reference_trades", reference), ("m5_reference_signals", reference_signals)):
                    frame.to_csv(args.output / f"{prefix}_{suffix}.csv", index=False)
                if kind == "fresh":
                    mtf_family.append(row)
                print(f"{symbol}/{cohort}/{mode}: combined n={row['metrics']['completed']} PF={row['metrics']['profit_factor']} "
                      f"commonM5 n={row['m5_reference']['metrics']['completed']}", flush=True)
            payload["cohorts"][cohort] = {"symbol": symbol, "start": start, "end": end, "source": value,
                                         "audit": audit, "native_feature_audit": native_audit,
                                         "comparison_clock": comparison_clock, "models": models}
        result["symbols"][symbol] = payload
    if len(native_family) != 4 or len(mtf_family) != 4:
        raise ValueError("Joint discovery family must contain all4native and all4combined hypotheses")
    holm(native_family + mtf_family)
    for symbol, payload in joint_native["symbols"].items():
        for mode, row in payload["models"].items():
            gate(row, row["development_eligible"])
            row["inference_basis"] = "jointHolm8; original native inputs/results remain immutable"
    result["native_reference_joint_inference"] = joint_native
    flat = []
    for symbol, payload in result["symbols"].items():
        for cohort, data in payload["cohorts"].items():
            for mode, row in data["models"].items():
                if cohort == "fresh_temporal180":
                    combined_gate(row, row["development_eligible"])
                else:
                    row.update(user_target_observed=(row["metrics"]["completed"] >= 1000 and
                               (row["metrics"]["profit_factor"] or 0) >= 1.5), historical_candidate=False,
                               supports_expected_pf_1_5=False, adds_value_over_m5_reference=False,
                               rejection_reasons=["reused_exploratory_cohort"], actual_money_profit="NOT TESTED")
                flat.append({"symbol": symbol, "cohort": cohort, "mode": mode,
                             "historical_candidate": row["historical_candidate"],
                             "adds_value_over_m5_reference": row["adds_value_over_m5_reference"], **row["metrics"]})
    _, terminal_fingerprint, _ = frozen_selection(args.output)
    _, terminal_native_results_hash = native_result(selection)
    if terminal_fingerprint != fingerprint or terminal_native_results_hash != native_results_hash:
        raise ValueError("Combined/native inference inputs changed during evaluation")
    save(args.output / "results.json", result)
    pd.DataFrame(flat).to_csv(args.output / "metrics.csv", index=False)
    print("MULTIFRAME_EVALUATION_COMPLETE; jointHolm8, no refitting/reselection or enabled execution", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_multiframe_20261005")
    parser.add_argument("--native-study", type=Path, default=DEFAULT_NATIVE_STUDY)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert_offline()
    (develop if args.stage == "develop" else evaluate)(args)


if __name__ == "__main__":
    main()
