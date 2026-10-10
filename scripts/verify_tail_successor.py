#!/usr/bin/env python3
"""Separate scalar reconstruction of the frozen successor-feed measurement.

Does not import its helper/runner or refit a detector. NumPy supplies only the
declared shared day resamples and percentile interpolation, not quote logic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SYMBOLS = ("BOOM600", "CRASH600")
DATES = ("2026-04-11", "2026-04-26", "2026-05-12", "2026-05-28", "2026-06-13", "2026-06-29",
         "2026-07-15", "2026-07-31", "2026-08-16", "2026-09-01", "2026-09-17", "2026-10-03")
COUNTS = ("anchor_rows", "detector_pair_exclusions", "detector_eligible_anchors",
    "detected_event_anchors", "boundary_excluded_anchors", "boundary_excluded_events",
    "reference_eligible", "reference_known", "reference_unknown", "event_eligible",
    "event_known", "event_unknown", "overlapping_consecutive_event_windows")
SUMS = ("reference_response_sum", "event_response_sum")
SCIENCE = ("docs/TAIL_SUCCESSOR_PREREQUISITE.md", "backend/app/research/tail_successor.py",
    "backend/tests/test_tail_successor.py", "scripts/run_tail_successor.py",
    "backend/tests/test_tail_successor_study.py", "scripts/verify_tail_successor.py",
    "backend/tests/test_tail_successor_verifier.py")
CONFIG = {"response": "side*log(P_t_plus_2/P_t_plus_1)",
    "event": "side*log(P_t/P_t_minus_1)>10*frozen_round8_median", "detector_refit": False,
    "reference_includes_events": True, "reference_day_weight": "eligible_event_anchors",
    "endpoint_rule": "t_plus_2_strictly_before_partition_end_and_source_midnight",
    "unknown_rule": "retain_eligible_denominators_without_substitution", "bootstrap_repeats": 9999,
    "bootstrap_seed": 20261007, "minimum_eligible_event_anchors_for_conditional_rejection": 100,
    "history_status": "previously_exposed_not_fresh_price_OOS",
    "study_kind": "feed_morphology_prerequisite_not_strategy_validation"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def safe_path(label):
    if not isinstance(label, str) or not label or Path(label).is_absolute() or str(Path(label)) != label:
        raise ValueError("Canonical repository-relative path required")
    if any(x in (".", "..") for x in Path(label).parts):
        raise ValueError("Path traversal refused")
    path = ROOT
    for part in Path(label).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Source symlinks refused")
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Regular repository input required")
    return path


def offline():
    if any(os.environ.get(flag, "false").lower() != "false" for flag in FLAGS):
        raise ValueError("All execution flags must be false")
    return dict.fromkeys(FLAGS, False)


class Audit:
    def __init__(self):
        self.checks = 0; self.errors = []; self.inputs = {}; self.false_values = 0
        self.max_numeric_error = 0.

    def require(self, condition, label):
        self.checks += 1
        if condition is not True:
            raise ValueError(label)

    def pin(self, label, fingerprint=None):
        value = digest(safe_path(label)); self.inputs[label] = value
        if fingerprint is not None:
            self.require(value == fingerprint, "Changed source hash: " + label)
        return value

    def compare(self, actual, expected, label="result"):
        self.checks += 1
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), label + " keys")
            for key in expected:
                self.compare(actual[key], expected[key], label + "/" + key)
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), label + " list")
            for a, b in zip(actual, expected, strict=True):
                self.compare(a, b, label)
        elif isinstance(expected, bool):
            self.require(type(actual) is bool and actual is expected, label + " boolean")
        elif expected is None:
            self.require(actual is None, label + " null")
        elif isinstance(expected, float):
            self.require(type(actual) in (int, float) and math.isfinite(actual), label + " finite number")
            self.max_numeric_error = max(self.max_numeric_error, abs(actual - expected))
            self.require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-15), label + " numeric mismatch")
        else:
            self.require(type(actual) is type(expected) and actual == expected, label + " mismatch")

    def safety(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FLAGS or key in ("goal_achieved", "historical_strategy_candidate", "strategy_eligible", "physical_spike_census"):
                    self.false_values += 1; self.require(item is False, "False gate: " + key)
                self.safety(item)
        elif isinstance(value, list):
            for item in value:
                self.safety(item)


def log_change(before, after):
    relative = (after - before) / before
    return math.log1p(relative) if math.isfinite(relative) and relative > -1 else math.log(after) - math.log(before)


def read_quotes(path, metadata):
    rows = []
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["epoch", "quote"]:
            raise ValueError("Exact epoch/quote CSV required")
        for row in reader:
            if len(row) != 2:
                raise ValueError("Ragged quote row")
            epoch = int(row[0]); quote = float(row[1])
            if (str(epoch) != row[0] or (rows and epoch <= rows[-1][0])
                    or not metadata["start_epoch"] <= epoch < metadata["end_exclusive_epoch"]
                    or not math.isfinite(quote) or quote <= 0):
                raise ValueError("Invalid canonical quote row")
            rows.append((epoch, quote))
    if (len(rows) != metadata["rows"] or len(rows) + metadata["missing_seconds"] != 86400
            or metadata["end_exclusive_epoch"] - metadata["start_epoch"] != 86400
            or metadata["gap_free"] is not (metadata["missing_seconds"] == 0)):
        raise ValueError("Source coverage mismatch")
    return rows


def reconstruct_days(rows, side, median, start, end):
    """Independent epoch lookup, rather than the production adjacent-row checks."""
    lookup = dict(rows)
    days = {}; responses = defaultdict(list); event_responses = defaultdict(list); previous_events = {}
    for t, quote in rows:
        if not start <= t < end:
            continue
        midnight = t // 86400 * 86400
        label = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
        if label not in days:
            days[label] = {"date": label, **dict.fromkeys(COUNTS, 0)}
        d = days[label]; d["anchor_rows"] += 1
        if t - 1 not in lookup or t - 1 < midnight:
            d["detector_pair_exclusions"] += 1
            continue
        event = side * log_change(lookup[t - 1], quote) > 10 * median
        d["detector_eligible_anchors"] += 1; d["detected_event_anchors"] += int(event)
        if t + 2 >= min(end, midnight + 86400):
            d["boundary_excluded_anchors"] += 1; d["boundary_excluded_events"] += int(event)
            continue
        d["reference_eligible"] += 1
        if event:
            d["event_eligible"] += 1
            if label in previous_events and t - previous_events[label] <= 2:
                d["overlapping_consecutive_event_windows"] += 1
            previous_events[label] = t
        known = t + 1 in lookup and t + 2 in lookup
        d["reference_known" if known else "reference_unknown"] += 1
        if event:
            d["event_known" if known else "event_unknown"] += 1
        if known:
            y = side * log_change(lookup[t + 1], lookup[t + 2]); responses[label].append(y)
            if event:
                event_responses[label].append(y)
    for label, d in days.items():
        d["reference_response_sum"] = math.fsum(responses[label])
        d["event_response_sum"] = math.fsum(event_responses[label])
    return list(days.values())


def reconstruct_summary(days):
    """Recompute each resample's weighting via scalar sums."""
    totals = {key: sum(day[key] for day in days) for key in COUNTS}
    totals.update({key: math.fsum(day[key] for day in days) for key in SUMS})
    def means(multiplicity):
        known = sum(int(n) * d["event_known"] for n, d in zip(multiplicity, days, strict=True))
        eligible = sum(int(n) * d["event_eligible"] for n, d in zip(multiplicity, days, strict=True))
        event = (math.fsum(int(n) * d["event_response_sum"] for n, d in zip(multiplicity, days, strict=True)) / known
                 if known else None)
        missing = any(n and d["event_eligible"] and not d["reference_known"] for n, d in zip(multiplicity, days, strict=True))
        reference = (math.fsum(int(n) * d["event_eligible"] * (d["reference_response_sum"] / d["reference_known"])
                              for n, d in zip(multiplicity, days, strict=True) if n and d["event_eligible"])
                     / eligible if eligible and not missing else None)
        return event, reference, event - reference if event is not None and reference is not None else None
    point = means([1] * len(days))
    distributions = [[], [], []]
    resamples = np.random.default_rng(20261007).multinomial(len(days), [1 / len(days)] * len(days), size=9999)
    for multiplicity in resamples:
        for bucket, value in zip(distributions, means(multiplicity), strict=True):
            if value is not None:
                bucket.append(value)
    uncertainty = {}
    for key, values in zip(("event_mean", "day_matched_reference_mean", "excess_mean"), distributions, strict=True):
        uncertainty[key] = {"finite_draws": len(values), "undefined_draws": 9999 - len(values),
                            "conditional_ci95": np.quantile(values, [.025, .975]).tolist() if values else None}
    complete = totals["event_unknown"] == totals["reference_unknown"] == 0
    ci = uncertainty["excess_mean"]["conditional_ci95"]
    rejected = totals["event_eligible"] >= 100 and complete and len(distributions[2]) == 9999 and ci is not None and ci[1] <= 0
    return {"observed_days": [d["date"] for d in days], "observed_day_clusters": len(days), "totals": totals,
        "event_mean_known_subset": point[0], "day_matched_reference_mean_known_subset": point[1],
        "excess_mean_known_subset": point[2], "complete_responses": complete,
        "bootstrap_repeats": 9999, "bootstrap_seed": 20261007, "bootstrap": uncertainty,
        "conditional_positive_excess_rejected": bool(rejected),
        "interpretation": "specified_positive_excess_conditionally_rejected" if rejected else "insufficient_for_bounded_rejection",
        "historical_strategy_candidate": False, "strategy_eligible": False, "profit_factor": None,
        "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED"}


def verify(output, results_ready=False):
    offline()
    if results_ready is not True:
        raise ValueError("Explicit --results-ready required before historical reconstruction")
    output = Path(output); audit = Audit()
    for name in ("declaration.json", "declaration.sha256", "results.json", "results.sha256"):
        audit.pin(str((output / name).resolve().relative_to(ROOT.resolve())))
    declaration, result = read_json(output / "declaration.json"), read_json(output / "results.json")
    audit.safety(declaration); audit.safety(result)
    audit.require(digest(output / "declaration.json") == (output / "declaration.sha256").read_text().strip(), "Declaration digest")
    audit.require(digest(output / "results.json") == (output / "results.sha256").read_text().strip(), "Results digest")
    audit.require(result["declaration_sha256"] == digest(output / "declaration.json"), "Result declaration identity")
    audit.require(declaration["stage"] == "tail_successor_frozen_before_response_measurement", "Declaration stage")
    audit.require(result["stage"] == "tail_successor_response_results", "Result stage")
    audit.compare(declaration["config"], CONFIG, "declaration/config"); audit.compare(result["config"], CONFIG, "result/config")
    for key in ("historical_quotes_decoded_for_this_response", "successor_response_evaluated", "detector_refit"):
        audit.require(declaration[key] is False, "Frozen-before-measurement field " + key)
    for value in (declaration, result):
        audit.require(value["safety"] == offline(), "All4 saved flags")
        for key in ("goal_achieved", "historical_strategy_candidate"):
            audit.require(value.get(key) is False, "Required false gate " + key)
        audit.compare(value["dates"], list(DATES), "dates")
    audit.require(result.get("physical_spike_census") is False, "Required false physical-event identity claim")
    audit.compare(declaration["symbols"], list(SYMBOLS), "declaration/symbols")
    audit.require(set(result["symbols"]) == set(SYMBOLS), "Result symbol identities")
    audit.require(set(declaration["science_code_hashes"]) == set(SCIENCE), "Seven frozen source files")
    for mapping in (declaration["science_code_hashes"], declaration["lineage"]["input_sha256"]):
        for label, fingerprint in mapping.items():
            audit.pin(label, fingerprint)
    lineage = declaration["lineage"]
    old_prefix = "docs/spike_tick_tail_20261005/"
    old = read_json(safe_path(old_prefix + "results.json"))
    old_declaration = read_json(safe_path(old_prefix + "declaration.json"))
    old_audit = read_json(safe_path(old_prefix + "independent_audit.json"))
    audit.require(old_audit["passed"] is True and not old_audit["errors"]
                  and old_audit["stage"] == "independent_fixed_tail_tick_pilot_audit", "Passed prior detector audit")
    audit.compare(lineage["sources"], old_declaration["lineage"]["sources"], "Immutable24 source metadata")
    audit.compare(lineage["row_cutoffs"], old_declaration["row_cutoffs"], "Inherited row cutoffs")
    audit.compare(lineage["observed_rows"], old_declaration["observed_rows"], "Inherited counts")
    for label in (old_prefix + "declaration.json", old_prefix + "results.json"):
        matches = [v for k, v in old_audit["input_sha256"].items() if k == label or k.endswith("/" + label)]
        audit.require(matches == [digest(safe_path(label))], "Prior independently audited detector input")
    sources_count = 0
    for symbol in SYMBOLS:
        audit.require(set(lineage["sources"][symbol]) == set(DATES), "Exact12 scheduled source days")
        rows = []
        for day in DATES:
            source = lineage["sources"][symbol][day]
            audit.safety(source)
            audit.require(source["symbol"] == symbol and source["date"] == day, "Source identity")
            midnight = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())
            audit.require(source["start_epoch"] == midnight and source["end_exclusive_epoch"] == midnight + 86400
                          and source["expected_grid_rows"] == 86400 and source["cadence_seconds"] == 1,
                          "Source UTC grid")
            for filekey, shakey in (("clean_file", "clean_sha256"), ("raw_pages_file", "raw_pages_sha256"),
                                    ("page_audit_file", "page_audit_sha256"), ("manifest", "manifest_sha256")):
                audit.pin(source[filekey], source[shakey])
            rows.extend(read_quotes(safe_path(source["clean_file"]), source)); sources_count += 1
        audit.require(all(rows[i][0] < rows[i + 1][0] for i in range(len(rows) - 1)), "Combined sorted source chronology")
        count = len(rows); cuts = {"40": 4 * count // 10, "50": 5 * count // 10, "60": 6 * count // 10,
                                  "70": 7 * count // 10, "100": count}
        audit.compare(lineage["row_cutoffs"][symbol], cuts, symbol + "/cutoffs")
        audit.require(lineage["observed_rows"][symbol] == count, "Observed row total")
        detector = old["symbols"][symbol]["detector"]
        audit.compare(lineage["detectors"][symbol], detector, symbol + "/frozen_detector")
        side = 1 if symbol == "BOOM600" else -1
        expected = {"side": side, "observed_rows": count, "row_cutoffs": cuts, "detector": detector,
                    "detector_fit_attempts": 0, "no_detector_refit": True, "strategy_eligible": False,
                    "profit_factor": None, "segments": {}}
        partitions = {"wf1": (cuts["40"], cuts["50"]), "wf2": (cuts["50"], cuts["60"]),
                      "wf3": (cuts["60"], cuts["70"]), "dev_validation": (cuts["40"], cuts["70"]),
                      "final30": (cuts["70"], count)}
        for name, (lo, hi) in partitions.items():
            start = rows[lo][0]; end = rows[hi][0] if hi < count else (rows[-1][0] // 86400 + 1) * 86400
            daily = reconstruct_days(rows, side, detector["median_abs_log_return"], start, end)
            expected["segments"][name] = {"row_start_inclusive": lo, "row_end_exclusive": hi,
                "first_observed_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                "last_observed_utc": datetime.fromtimestamp(rows[hi - 1][0], timezone.utc).isoformat(),
                "start_epoch": start, "end_exclusive_epoch": end, "daily": daily,
                "summary": reconstruct_summary(daily)}
        audit.compare(result["symbols"][symbol], expected, symbol)
        print("TAIL_SUCCESSOR_AUDITED", symbol, flush=True)
    for key in ("actual_money_profit", "prospective_paper", "actual_execution_costs"):
        audit.require(result[key] == "NOT TESTED", "Untested economic quantity " + key)
    audit.require(result["profit_factor"] is None, "No PF from feed morphology")
    for label, fingerprint in audit.inputs.items():
        audit.require(digest(safe_path(label)) == fingerprint, "Unchanged inputs after reconstruction")
    return {"stage": "independent_scalar_tail_successor_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "checks": audit.checks, "errors": [], "safety": offline(), "sources": sources_count,
        "symbols": list(SYMBOLS), "saved_false_values_checked": audit.false_values,
        "input_sha256": audit.inputs, "declaration_sha256": digest(output / "declaration.json"),
        "results_sha256": digest(output / "results.json"), "max_numeric_error": audit.max_numeric_error,
        "verifier_file": "scripts/verify_tail_successor.py", "verifier_sha256": digest(Path(__file__)),
        "scope": "Exact frozen source hashes, scalar anchors/events/reference/unknown/boundary/overlap, point metrics, shared-day resamples and percentile arithmetic; excludes inferential validity, physical jump identity, fresh OOS, strategy payoff, broker execution and profit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/tail_successor_20261007")
    parser.add_argument("--results-ready", action="store_true")
    args = parser.parse_args(); offline()
    destination = args.output / "independent_audit.json"
    if destination.exists() or (args.output / "independent_audit.sha256").exists():
        raise ValueError("Refusing audit overwrite")
    report = verify(args.output, args.results_ready)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    with (args.output / "independent_audit.sha256").open("x") as stream:
        stream.write(digest(destination) + "\n")
    print("TAIL_SUCCESSOR_AUDIT_PASS", report["checks"], "checks", digest(destination), flush=True)


if __name__ == "__main__":
    main()
