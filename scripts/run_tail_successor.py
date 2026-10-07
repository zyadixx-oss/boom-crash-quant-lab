#!/usr/bin/env python3
"""Frozen, already-exposed-history successor-feed measurement; never trades."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.tick_tail import FixedTailDetector
from app.research.tail_successor import collect_successor_days, summarize_successor
from scripts import run_spike_tick_tail as previous

OUTPUT = ROOT / "docs/tail_successor_20261007"
FLAGS = previous.FLAGS
SYMBOLS, DATES = previous.SYMBOLS, previous.DATES
SCIENCE = (
    "docs/TAIL_SUCCESSOR_PREREQUISITE.md",
    "backend/app/research/tail_successor.py",
    "backend/tests/test_tail_successor.py",
    "scripts/run_tail_successor.py",
    "backend/tests/test_tail_successor_study.py",
    "scripts/verify_tail_successor.py",
    "backend/tests/test_tail_successor_verifier.py",
)
CONFIG = {
    "response": "side*log(P_t_plus_2/P_t_plus_1)",
    "event": "side*log(P_t/P_t_minus_1)>10*frozen_round8_median",
    "detector_refit": False,
    "reference_includes_events": True,
    "reference_day_weight": "eligible_event_anchors",
    "endpoint_rule": "t_plus_2_strictly_before_partition_end_and_source_midnight",
    "unknown_rule": "retain_eligible_denominators_without_substitution",
    "bootstrap_repeats": 9999,
    "bootstrap_seed": 20261007,
    "minimum_eligible_event_anchors_for_conditional_rejection": 100,
    "history_status": "previously_exposed_not_fresh_price_OOS",
    "study_kind": "feed_morphology_prerequisite_not_strategy_validation",
}
digest, read_json, save, offline = previous.digest, previous.read_json, previous.save, previous.offline


def path_for(label):
    """Only canonical repository-relative regular files, without symlink aliases."""
    if not isinstance(label, str) or not label or Path(label).is_absolute():
        raise ValueError("Repository-relative input required")
    parts = Path(label).parts
    if any(part in (".", "..") for part in parts) or str(Path(label)) != label:
        raise ValueError("Canonical source path required")
    path = ROOT
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Source symlinks forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Regular repository source required")
    return path


def science_hashes():
    return {label: digest(path_for(label)) for label in SCIENCE}


def verified_lineage():
    """Read only metadata and hash bytes; never decode historical quote values."""
    offline()
    declaration = previous.frozen(previous.OUTPUT)
    results_path = previous.OUTPUT / "results.json"
    audit_path = previous.OUTPUT / "independent_audit.json"
    results, audit = read_json(results_path), read_json(audit_path)
    previous.safety(results); previous.safety(audit)
    if (digest(results_path) != (previous.OUTPUT / "results.sha256").read_text().strip()
            or results["declaration_sha256"] != digest(previous.OUTPUT / "declaration.json")
            or results["historical_strategy_candidate"] is not False
            or results["goal_achieved"] is not False
            or audit["passed"] is not True or audit["errors"]
            or audit["stage"] != "independent_fixed_tail_tick_pilot_audit"):
        raise ValueError("The exact frozen round8 measurement and audit must be valid")
    pins = dict(declaration["lineage"]["inherited_files"])
    pins.update(declaration["science_code_hashes"])
    for path in (previous.OUTPUT / "declaration.json", previous.OUTPUT / "declaration.sha256",
                 results_path, previous.OUTPUT / "results.sha256", audit_path):
        pins[previous.relative(path)] = digest(path)
    verifier = path_for(audit["verifier_file"])
    if digest(verifier) != audit["verifier_sha256"]:
        raise ValueError("Round8 independent verifier bytes changed")
    pins[previous.relative(verifier)] = digest(verifier)
    for path in (previous.OUTPUT / "declaration.json", results_path):
        previous.audited_input_hash(audit, path)
    detectors = {}
    for symbol in SYMBOLS:
        value = results["symbols"][symbol]
        detector = value["detector"]
        side = 1 if symbol == "BOOM600" else -1
        fixed = FixedTailDetector(side, detector["median_abs_log_return"])
        if (value["side"] != side or value["observed_rows"] != declaration["observed_rows"][symbol]
                or value["row_cutoffs"] != declaration["row_cutoffs"][symbol]
                or detector["adequate"] is not True or detector["calibration_events"] < 100
                or detector["threshold"] != fixed.threshold or detector["threshold_multiplier"] != 10.
                or detector["calibration_source"] != "earliest40_percent_observed_rows_only"):
            raise ValueError("Fixed detector or partition identity changed")
        detectors[symbol] = detector
    for label, fingerprint in pins.items():
        if digest(path_for(label)) != fingerprint:
            raise ValueError("Pinned ancestor bytes changed: " + label)
    return {"input_sha256": pins, "sources": declaration["lineage"]["sources"],
            "detectors": detectors, "observed_rows": declaration["observed_rows"],
            "row_cutoffs": declaration["row_cutoffs"]}


def declare(output):
    offline()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Declaration output must be empty; refusing overwrite")
    hashes, lineage = science_hashes(), verified_lineage()
    document = {"stage": "tail_successor_frozen_before_response_measurement",
                "run_utc": datetime.now(timezone.utc).isoformat(), "safety": offline(),
                "config": CONFIG, "dates": list(DATES), "symbols": list(SYMBOLS),
                "science_code_hashes": hashes, "lineage": lineage,
                "historical_quotes_decoded_for_this_response": False,
                "successor_response_evaluated": False, "detector_refit": False,
                "goal_achieved": False, "historical_strategy_candidate": False}
    if science_hashes() != hashes or verified_lineage() != lineage:
        raise ValueError("Source bytes changed during declaration")
    output.mkdir(parents=True, exist_ok=True)
    save(output / "declaration.json", document)
    with (output / "declaration.sha256").open("x") as stream:
        stream.write(digest(output / "declaration.json") + "\n")
    print("TAIL_SUCCESSOR_DECLARED", digest(output / "declaration.json"), flush=True)
    return document


def frozen(output):
    offline()
    output = Path(output)
    document = read_json(output / "declaration.json")
    if digest(output / "declaration.json") != (output / "declaration.sha256").read_text().strip():
        raise ValueError("Declaration bytes changed")
    previous.safety(document)
    if (document["stage"] != "tail_successor_frozen_before_response_measurement"
            or document["config"] != CONFIG or document["dates"] != list(DATES)
            or document["symbols"] != list(SYMBOLS)
            or any(document[k] is not False for k in ("historical_quotes_decoded_for_this_response",
                "successor_response_evaluated", "detector_refit", "goal_achieved", "historical_strategy_candidate"))
            or document["science_code_hashes"] != science_hashes()
            or document["lineage"] != verified_lineage()):
        raise ValueError("Frozen protocol, source lineage or premeasurement state changed")
    return document


def evaluate_symbol(ticks, symbol, lineage):
    cuts = lineage["row_cutoffs"][symbol]
    detector_dict = lineage["detectors"][symbol]
    side = 1 if symbol == "BOOM600" else -1
    detector = FixedTailDetector(side, detector_dict["median_abs_log_return"])
    result = {"side": side, "observed_rows": len(ticks), "row_cutoffs": cuts,
              "detector": detector_dict, "detector_fit_attempts": 0, "no_detector_refit": True,
              "strategy_eligible": False, "profit_factor": None, "segments": {}}
    for name, (start, end) in previous.segment_slices(cuts).items():
        if name == "calibration40":
            continue
        first = int(ticks.index[start].timestamp())
        endpoint = (int(ticks.index[end].timestamp()) if end < len(ticks)
                    else int((ticks.index[-1].normalize() + pd.Timedelta(days=1)).timestamp()))
        days = collect_successor_days(ticks, detector, first, endpoint)
        result["segments"][name] = {"row_start_inclusive": start, "row_end_exclusive": end,
            "first_observed_utc": ticks.index[start].isoformat(),
            "last_observed_utc": ticks.index[end - 1].isoformat(),
            "start_epoch": first, "end_exclusive_epoch": endpoint, "daily": days,
            "summary": summarize_successor(days, repeats=9999, seed=20261007)}
    return result


def evaluate(output):
    offline()
    output = Path(output)
    if any((output / name).exists() for name in ("results.json", "results.sha256")):
        raise ValueError("Refusing outcome overwrite")
    document = frozen(output)
    result = {"stage": "tail_successor_response_results", "run_utc": datetime.now(timezone.utc).isoformat(),
              "declaration_sha256": digest(output / "declaration.json"), "safety": offline(),
              "config": CONFIG, "dates": list(DATES), "symbols": {}, "goal_achieved": False,
              "historical_strategy_candidate": False, "actual_money_profit": "NOT TESTED",
              "prospective_paper": "NOT TESTED", "actual_execution_costs": "NOT TESTED",
              "physical_spike_census": False, "profit_factor": None}
    for symbol in SYMBOLS:
        sources = document["lineage"]["sources"][symbol]
        frames = [previous.load_tick_day(sources[day]) for day in DATES]
        ticks = pd.concat(frames)
        if (len(ticks) != document["lineage"]["observed_rows"][symbol]
                or not ticks.index.is_unique or not ticks.index.is_monotonic_increasing):
            raise ValueError("Decoded source chronology/count mismatch")
        result["symbols"][symbol] = evaluate_symbol(ticks, symbol, document["lineage"])
        print("TAIL_SUCCESSOR_MEASURED", symbol, flush=True)
    frozen(output)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(output / "results.json", result)
    with (output / "results.sha256").open("x") as stream:
        stream.write(digest(output / "results.json") + "\n")
    print("TAIL_SUCCESSOR_SAVED", digest(output / "results.json"), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("declare", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    (declare if args.stage == "declare" else evaluate)(args.output)


if __name__ == "__main__":
    main()
