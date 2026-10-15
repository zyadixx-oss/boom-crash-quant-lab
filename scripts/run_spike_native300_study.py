#!/usr/bin/env python3
"""Fixed native300 old-to-new replication composed from frozen learned helpers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from scripts.run_spike_learned_study import (
    CFG, MODES, SOURCES as INHERITED_SOURCES, FEATURE_FORMULAS,
    assert_offline, audited_source, digest, save, prepare, issued, fit_at,
    development_eligible, evaluate_one, gate, replay_timed, summarize, window, holm,
)


SYMBOLS = ("BOOM300N", "CRASH300N")
WALK_FORWARD = (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3"))
SOURCES = (*INHERITED_SOURCES, "docs/SPIKE_NATIVE300_PROTOCOL.md",
           "scripts/run_spike_native300_study.py", "backend/tests/test_spike_native300_study.py")
EXPECTED_INTERVALS = {
    "old": (pd.Timestamp("2025-10-09T11:08:00Z"), pd.Timestamp("2026-04-07T11:08:00Z")),
    "fresh": (pd.Timestamp("2026-04-07T11:08:00Z"), pd.Timestamp("2026-10-04T11:08:00Z")),
}
BOOTSTRAP_REPEATS = 9999


def hashes() -> dict:
    return {name: digest(ROOT / name) for name in SOURCES}


def source(folder: str, symbol: str) -> dict:
    """Audit acquisition bytes/metadata only; do not prepare features or labels."""
    value = audited_source(folder, symbol)
    path = ROOT / value["manifest"]
    document = json.loads(path.read_text())
    if document.get("symbol") != symbol:
        raise ValueError("Normalization manifest symbol changed")
    if not all(document.get(key) is True for key in (
        "normalization_valid", "finite_positive_ohlc", "canonical_utc_minute_grid",
    )):
        raise ValueError("Native source normalization audit failed")
    if document.get("authentication_used") is not False or document.get("fills_or_interpolations") is not False:
        raise ValueError("Native source must be public and uninterpolated")
    flags = assert_offline()
    if any(document.get("safety", {}).get(flag) is not False for flag in flags):
        raise ValueError("Acquisition safety flags must all remain false")
    if document.get("request_errors"):
        raise ValueError("Native source has unresolved request errors")
    first, last = value["first_epoch"], value["last_epoch"]
    if first % 60 or last % 60 or first > last:
        raise ValueError("Native source interval must use UTC minute openings")
    if (document.get("requested_start_epoch") != first or
            document.get("requested_cutoff_exclusive_epoch") != last + 60):
        raise ValueError("Acquisition does not cover the declared endpoint interval")
    gaps = document.get("gaps", [])
    missing = sum(int(item["missing_bars"]) for item in gaps)
    expected = (last + 60 - first) // 60
    if value["rows"] + missing != expected:
        raise ValueError("Clean source rows and declared gaps do not cover the requested grid")
    value.update(manifest_sha256=digest(path), expected_grid_rows=expected,
                 declared_missing_minutes=missing, declared_gaps=gaps,
                 excluded_raw_rows=document.get("excluded_rows", []),
                 from_utc=str(pd.Timestamp(first, unit="s", tz="UTC")),
                 to_exclusive_utc=str(pd.Timestamp(last + 60, unit="s", tz="UTC")))
    return value


def validate_intervals(old: dict, fresh: dict) -> dict:
    intervals = {
        role: (pd.Timestamp(value["first_epoch"], unit="s", tz="UTC"),
               pd.Timestamp(value["last_epoch"] + 60, unit="s", tz="UTC"))
        for role, value in (("old", old), ("fresh", fresh))
    }
    if intervals["old"][1] > intervals["fresh"][0]:
        raise ValueError("Older development and fresh evaluation intervals overlap")
    if intervals["old"][1] != intervals["fresh"][0]:
        raise ValueError("Native study requires the declared adjacent intervals")
    if intervals != EXPECTED_INTERVALS:
        raise ValueError("Native study interval differs from the declared old-to-new periods")
    return {"old_start": intervals["old"][0], "old_end": intervals["old"][1],
            "fresh_start": intervals["fresh"][0], "fresh_end": intervals["fresh"][1],
            "overlap_minutes": 0, "adjacent": True, "chronological_old_to_new": True}


def verify_prices(sources: dict) -> None:
    for values in sources.values():
        for value in values.values():
            if digest(ROOT / value["path"]) != value["sha256"]:
                raise ValueError("Frozen native price source changed")
            if digest(ROOT / value["manifest"]) != value["manifest_sha256"]:
                raise ValueError("Frozen native normalization manifest changed")


def check_prepared(m1, audit: dict, value: dict) -> None:
    expected = (pd.Timestamp(value["first_epoch"], unit="s", tz="UTC"),
                pd.Timestamp(value["last_epoch"] + 60, unit="s", tz="UTC"))
    actual = (m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1))
    if actual != expected or audit["sha256"] != value["sha256"]:
        raise ValueError("Prepared native source differs from declared bytes or interval")
    if audit["missing_minutes"] != value["declared_missing_minutes"]:
        raise ValueError("Prepared native gaps differ from acquisition declaration")


def choose_direction(candidates: dict) -> str | None:
    eligible = [candidate for candidate in candidates.values() if candidate["development_eligible"]]
    if not eligible:
        return None
    return sorted(eligible, key=lambda candidate: (-candidate["validation"]["selection_score"],
                                                  candidate["mode"]))[0]["mode"]


def guard_stage(output: Path, stage: str) -> None:
    if stage not in ("develop", "evaluate"):
        raise ValueError("Unknown native research stage")
    if (output / "results.json").exists():
        raise ValueError("Refusing post-result selection or evaluation overwrite")
    if stage == "develop" and any((output / name).exists() for name in (
        "declaration.json", "declaration.sha256", "selection.json", "selection.sha256",
    )):
        raise ValueError("Refusing to overwrite the native declaration or frozen selection")


def develop(args) -> None:
    guard_stage(args.output, "develop")
    safety = assert_offline()
    declared_sources, intervals = {}, {}
    for symbol in SYMBOLS:
        old = source("data/spike_learned_transfer", symbol)
        fresh = source("data/spike_native300_recent", symbol)
        intervals[symbol] = validate_intervals(old, fresh)
        declared_sources[symbol] = {"old": old, "fresh": fresh}
    code_hashes = hashes()
    verify_prices(declared_sources)
    declaration = {
        "stage": "native300_predevelopment_declaration", "adaptive_round": 4,
        "run_utc": datetime.now(timezone.utc).isoformat(), "safety": safety,
        "code_hashes": code_hashes, "sources": declared_sources, "intervals": intervals,
        "config": asdict(CFG), "feature_formulas": FEATURE_FORMULAS,
        "fresh_features_evaluated": False, "fresh_payoffs_evaluated": False,
        "older300_payoffs_previously_seen": True, "prospective_paper": False,
        "user_target": {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000},
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
    }
    save(args.output / "declaration.json", declaration)
    declaration_hash = digest(args.output / "declaration.json")
    (args.output / "declaration.sha256").write_text(declaration_hash + "\n")
    result = {**declaration, "stage": "frozen_native300_development",
              "declaration_sha256": declaration_hash, "symbols": {}}
    for symbol in SYMBOLS:
        value = declared_sources[symbol]["old"]
        m1, rows, names, audit, split = prepare(ROOT / value["path"], symbol)
        check_prepared(m1, audit, value)
        candidates = {}
        for mode in MODES:
            labels, label_audit = replay_timed(m1, issued(rows, symbol, mode), CFG,
                                              *split["development"], purge_minutes=31)
            folds, validation_ledgers = [], []
            for training, test in WALK_FORWARD:
                model = fit_at(rows, labels, names, *split[training])
                trades, execution = replay_timed(m1, issued(rows, symbol, mode, model), CFG,
                                                 *split[test], purge_minutes=31)
                folds.append({"training": training, "test": test, "model": model,
                              "metrics": summarize(trades, *split[test], stop_atr=CFG.stop_atr),
                              "baseline": summarize(window(labels, *split[test]), *split[test],
                                                    stop_atr=CFG.stop_atr), "audit": execution})
                validation_ledgers.append(trades)
            validation_start, validation_end = split["wf1"][0], split["wf3"][1]
            combined = pd.concat(validation_ledgers, ignore_index=True)
            candidate = {"mode": mode, "config": asdict(CFG), "walk_forward": folds,
                         "validation": summarize(combined, validation_start, validation_end,
                                                 stop_atr=CFG.stop_atr),
                         "training_label_audit": label_audit,
                         "final_model": fit_at(rows, labels, names, *split["development"])}
            candidate["development_eligible"] = development_eligible(candidate)
            candidates[mode] = candidate
            combined.to_csv(args.output / f"{symbol}_{mode}_validation_trades.csv", index=False)
            print(f"{symbol}/{mode}: native validation n={candidate['validation']['completed']} "
                  f"PF={candidate['validation']['profit_factor']} "
                  f"eligible={candidate['development_eligible']}", flush=True)
        result["symbols"][symbol] = {
            "audit": audit, "partitions": split, "models": candidates,
            "selected_direction": choose_direction(candidates),
            "eligible_directions": sum(c["development_eligible"] for c in candidates.values()),
        }
    # Detect concurrent source/protocol drift rather than silently freezing it.
    if hashes() != code_hashes:
        raise ValueError("Native study source changed during development")
    verify_prices(declared_sources)
    save(args.output / "selection.json", result)
    fingerprint = digest(args.output / "selection.json")
    (args.output / "selection.sha256").write_text(fingerprint + "\n")
    print(f"NATIVE300_SELECTION_FROZEN {fingerprint}; fresh features/payoffs NOT EVALUATED", flush=True)


def frozen_selection(output: Path) -> tuple[dict, str]:
    selected_path = output / "selection.json"
    fingerprint = (output / "selection.sha256").read_text().strip()
    if digest(selected_path) != fingerprint:
        raise ValueError("Frozen native selection changed")
    selection = json.loads(selected_path.read_text())
    declaration_path = output / "declaration.json"
    declared_hash = (output / "declaration.sha256").read_text().strip()
    if digest(declaration_path) != declared_hash or declared_hash != selection["declaration_sha256"]:
        raise ValueError("Frozen native declaration changed")
    declaration = json.loads(declaration_path.read_text())
    for name in ("code_hashes", "sources", "intervals", "config", "feature_formulas", "bootstrap_repeats"):
        if declaration[name] != selection[name]:
            raise ValueError(f"Native selection diverges from predevelopment declaration: {name}")
    if selection["code_hashes"] != hashes():
        raise ValueError("Frozen native protocol/source changed")
    for values in selection["sources"].values():
        validate_intervals(values["old"], values["fresh"])
    verify_prices(selection["sources"])
    return selection, fingerprint


def evaluate(args) -> None:
    guard_stage(args.output, "evaluate")
    if args.bootstrap != BOOTSTRAP_REPEATS:
        raise ValueError("Native primary inference requires the declared9999 bootstrap replicates")
    safety = assert_offline()
    selection, fingerprint = frozen_selection(args.output)
    result = {"stage": "native300_chronological_evaluation",
              "run_utc": datetime.now(timezone.utc).isoformat(), "adaptive_round": 4,
              "selection_sha256": fingerprint, "declaration_sha256": selection["declaration_sha256"],
              "code_hashes": selection["code_hashes"], "safety": safety, "symbols": {},
              "forward_test": False, "prospective_paper": False,
              "chronological_old_to_new": True, "methodology_adaptive": True,
              "actual_money_profit": "NOT TESTED", "bootstrap_repeats": args.bootstrap}
    fresh_family, flat = [], []
    for symbol in SYMBOLS:
        selected_direction = selection["symbols"][symbol]["selected_direction"]
        payload = {"selected_direction": selected_direction, "cohorts": {}}
        for cohort, kind in (("reused_final30", "old"), ("fresh_temporal180", "fresh")):
            value = selection["sources"][symbol][kind]
            m1, rows, names, audit, split = prepare(ROOT / value["path"], symbol)
            check_prepared(m1, audit, value)
            start, end = (split["final_test"] if kind == "old" else
                          (m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)))
            models = {}
            for mode in MODES:
                candidate = selection["symbols"][symbol]["models"][mode]
                if names != candidate["final_model"]["feature_names"]:
                    raise ValueError("Prepared native feature order differs from frozen model")
                row, trades, signal_frame = evaluate_one(
                    m1, rows, symbol, mode, candidate["final_model"], start, end, args.bootstrap,
                )
                row.update(development_eligible=candidate["development_eligible"],
                           selected_direction=mode == selected_direction)
                models[mode] = row
                trades.to_csv(args.output / f"{symbol}_{cohort}_{mode}_trades.csv", index=False)
                signal_frame.to_csv(args.output / f"{symbol}_{cohort}_{mode}_signals.csv", index=False)
                if kind == "fresh":
                    fresh_family.append(row)
                print(f"{symbol}/{cohort}/{mode}: n={row['metrics']['completed']} "
                      f"PF={row['metrics']['profit_factor']}", flush=True)
            payload["cohorts"][cohort] = {
                "symbol": symbol, "start": start, "end": end, "source": value,
                "audit": audit, "models": models,
            }
        result["symbols"][symbol] = payload
    if len(fresh_family) != 4:
        raise ValueError("The frozen native multiplicity family must contain all four hypotheses")
    holm(fresh_family)
    for symbol, payload in result["symbols"].items():
        for cohort, data in payload["cohorts"].items():
            for mode, row in data["models"].items():
                if cohort == "fresh_temporal180":
                    gate(row, row["development_eligible"])
                else:
                    row.update(user_target_observed=(row["metrics"]["completed"] >= 1000 and
                               (row["metrics"]["profit_factor"] or 0) >= 1.5),
                               historical_candidate=False, supports_expected_pf_1_5=False,
                               rejection_reasons=["reused_exploratory_cohort"],
                               actual_money_profit="NOT TESTED", prospective_validation="NOT TESTED")
                flat.append({"training_symbol": symbol, "evaluation_symbol": symbol,
                             "cohort": cohort, "mode": mode,
                             "user_target_observed": row["user_target_observed"],
                             "historical_candidate": row["historical_candidate"], **row["metrics"]})
    # Freeze integrity again before committing the terminal evaluation record.
    _, current_fingerprint = frozen_selection(args.output)
    if current_fingerprint != fingerprint:
        raise ValueError("Native selection changed during evaluation")
    save(args.output / "results.json", result)
    pd.DataFrame(flat).to_csv(args.output / "metrics.csv", index=False)
    print("NATIVE300_EVALUATION_COMPLETE; no fitting/reselection or enabled execution", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_native300_20261005")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert_offline()
    (develop if args.stage == "develop" else evaluate)(args)


if __name__ == "__main__":
    main()
