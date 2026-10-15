#!/usr/bin/env python3
"""Declared exploratory tail-tick mechanism pilot; no trading or payoff model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.tick_tail import fit_fixed_tail_detector, describe_tick_tail
from scripts.run_spike_tick_execution import frozen_declaration as frozen7_declaration

ROUND7 = ROOT / "docs/spike_tick_execution_20261005"
OUTPUT = ROOT / "docs/spike_tick_tail_20261005"
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SYMBOLS = ("BOOM600", "CRASH600")
DATES = tuple((pd.Timestamp("2026-04-11", tz="UTC") + pd.Timedelta(days=k*175//11)).strftime("%Y-%m-%d") for k in range(12))
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
BOOTSTRAP, SEED, NOMINAL_N = 9999, 20261005, 600
SCIENCE_SOURCES = ("docs/SPIKE_TICK_TAIL_PROTOCOL.md", "backend/app/research/tick_tail.py",
                   "backend/tests/test_tick_tail.py", "scripts/run_spike_tick_tail.py",
                   "backend/tests/test_spike_tick_tail_study.py")
BANDS = ("lt_N", "ge_N")
CONFIG = {"calibration_fraction": .4, "development_fraction": .7, "threshold_multiplier": 10.,
          "nominal_age_boundary_seconds": NOMINAL_N, "minimum_calibration_events": 100,
          "minimum_events_per_band": 100, "bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED,
          "large_overdue_ratio": 1.5, "undefined_bootstrap_draws_block_falsification": True}


def offline():
    if any(os.environ.get(flag, "false").lower() != "false" for flag in FLAGS):
        raise ValueError("All four live/execution flags must be false")
    return dict.fromkeys(FLAGS, False)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON metadata: "+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def path_for(value):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("Source paths must be repository relative")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Source path escapes repository")
    return path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def safety(value):
    if value.get("safety") != offline() or any(value["safety"].get(flag) is not False for flag in FLAGS):
        raise ValueError("Saved source safety flags must all be false")


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name+" must be an integer >= "+str(minimum))
    return value


def science_hashes():
    return {path: digest(path_for(path)) for path in SCIENCE_SOURCES}


def cutoffs(n):
    n = integer(n, "observed rows", 1)
    # Exact rational floors avoid binary floating-point cutoff rounding.
    return {"40": 4*n//10, "50": 5*n//10, "60": 6*n//10, "70": 7*n//10, "100": n}


def segment_slices(cuts):
    return {"calibration40": (0, cuts["40"]), "wf1": (cuts["40"], cuts["50"]),
            "wf2": (cuts["50"], cuts["60"]), "wf3": (cuts["60"], cuts["70"]),
            "dev_validation": (cuts["40"], cuts["70"]), "final30": (cuts["70"], cuts["100"])}


def audited_input_hash(report, path):
    label = relative(path)
    matches = [value for key, value in report["input_sha256"].items() if key == label or key.endswith("/"+label)]
    if len(matches) != 1 or matches[0] != digest(path):
        raise ValueError("Prior source audit does not anchor current bytes: "+label)


def verified_round7_metadata():
    """Only metadata parsing and byte hashing; never call the quote-reading guard."""
    offline()
    old = frozen7_declaration(ROUND7)
    declaration_path = ROUND7 / "declaration.json"
    sources_path, sources_sha_path = ROUND7 / "tick_sources.json", ROUND7 / "tick_sources.sha256"
    audit_path, results_path = ROUND7 / "independent_audit.json", ROUND7 / "results.json"
    if digest(sources_path) != sources_sha_path.read_text().strip():
        raise ValueError("Round7 source declaration hash changed")
    sources, prior_audit = read_json(sources_path), read_json(audit_path)
    safety(old); safety(sources); safety(prior_audit)
    if (old["dates"] != list(DATES) or old["symbols"] != list(SYMBOLS) or
            sources["stage"] != "tick_sources_frozen_before_payoff" or
            sources["declaration_sha256"] != digest(declaration_path) or sources["tick_outcomes_evaluated"] is not False):
        raise ValueError("Round7 source identity or chronology changed")
    if (prior_audit["stage"] != "independent_primary_tick_execution_audit" or prior_audit["passed"] is not True or
            prior_audit["errors"] or len(prior_audit["sources"]) != 24 or len(prior_audit["groups"]) != 16):
        raise ValueError("The exact prior24-source audit must have passed")
    for path in (declaration_path, sources_path, results_path):
        audited_input_hash(prior_audit, path)
    verifier = path_for(prior_audit["verifier_file"])
    if digest(verifier) != prior_audit["verifier_sha256"]:
        raise ValueError("Prior source verifier changed")
    pins = {relative(path): digest(path) for path in (declaration_path, ROUND7/"declaration.sha256",
            sources_path, sources_sha_path, results_path, audit_path, verifier)}
    for mapping in (old["code_hashes"], old["round6_source_files"], sources["acquisition_files"]):
        for label, fingerprint in mapping.items():
            if digest(path_for(label)) != fingerprint:
                raise ValueError("Inherited source/code bytes changed: "+label)
            pins[label] = fingerprint
    if set(sources["sources"]) != set(SYMBOLS):
        raise ValueError("Exactly the two declared600 symbols are required")
    identities = set()
    file_names = []
    for symbol in SYMBOLS:
        if set(sources["sources"][symbol]) != set(DATES):
            raise ValueError("Exactly the twelve scheduled day sources are required")
        for date in DATES:
            value = sources["sources"][symbol][date]
            safety(value)
            start = int(pd.Timestamp(date, tz="UTC").timestamp())
            rows = integer(value["rows"], "source rows", 1)
            missing = integer(value["missing_seconds"], "missing seconds")
            if (value["symbol"] != symbol or value["date"] != date or value["start_epoch"] != start or
                    value["end_exclusive_epoch"] != start+86400 or value["expected_grid_rows"] != 86400 or
                    value["cadence_seconds"] != 1 or rows+missing != 86400 or
                    value["gap_free"] is not (missing == 0) or value["endpoint"] != ENDPOINT or
                    value["normalization_valid"] is not True or value["authentication_used"] is not False or
                    value["fills_or_interpolations"] is not False or value["features_labels_or_tick_outcomes_computed"] is not False or
                    value["collector_sha256"] != old["code_hashes"]["scripts/collect_spike_ticks.py"] or
                    value["reconciliation"]["price_match"] is not True or
                    value["reconciliation"]["complete_minutes"]+value["reconciliation"]["unknown_minutes"] != 1440):
                raise ValueError("Round7 source coverage/public-data identity changed")
            for filekey, shakey in (("manifest", "manifest_sha256"), ("clean_file", "clean_sha256"),
                                    ("raw_pages_file", "raw_pages_sha256"), ("page_audit_file", "page_audit_sha256")):
                label, fingerprint = value[filekey], value[shakey]
                path = path_for(label)
                if digest(path) != fingerprint:
                    raise ValueError("Immutable tick source bytes changed: "+label)
                audited_input_hash(prior_audit, path)
                pins[label] = fingerprint
                file_names.append(label)
            manifest = read_json(path_for(value["manifest"]))
            if any(value.get(key) != item for key, item in manifest.items()):
                raise ValueError("Frozen source differs from its manifest")
            identities.add((symbol, date))
    if len(set(file_names)) != len(file_names) or len(identities) != 24:
        raise ValueError("Tick source file aliases are forbidden")
    if {(value["symbol"], value["date"]) for value in prior_audit["sources"]} != identities:
        raise ValueError("Prior audit does not cover exactly these24 sources")
    return {"inherited_files": pins, "round6_selection_sha256": old["round6_selection_sha256"],
            "round7_declaration_sha256": digest(declaration_path), "round7_tick_sources_sha256": digest(sources_path),
            "round7_results_sha256": digest(results_path), "round7_audit_sha256": digest(audit_path),
            "sources": sources["sources"]}


def declare(output):
    offline()
    output = Path(output)
    if any((output/name).exists() for name in ("declaration.json", "declaration.sha256", "results.json", "results.sha256", "metrics.csv")):
        raise ValueError("Refusing declaration or outcome overwrite")
    hashes = science_hashes()
    metadata = verified_round7_metadata()
    counts = {symbol: sum(metadata["sources"][symbol][date]["rows"] for date in DATES) for symbol in SYMBOLS}
    document = {"stage": "tail_tick_precalibration_declaration", "run_utc": datetime.now(timezone.utc).isoformat(),
                "safety": offline(), "dates": list(DATES), "symbols": list(SYMBOLS), "config": CONFIG,
                "science_code_hashes": hashes, "lineage": metadata, "observed_rows": counts,
                "row_cutoffs": {symbol: cutoffs(n) for symbol, n in counts.items()},
                "partition_basis": "time_ordered_observed_quote_rows_not_intervening_calendar_span",
                "quotes_parsed": False, "detector_calibrated": False, "events_or_hazards_evaluated": False,
                "study_kind": "adaptive_known_history_tail_tick_mechanism_pilot_not_strategy_validation"}
    if science_hashes() != hashes or verified_round7_metadata() != metadata:
        raise ValueError("Source or science bytes changed during declaration")
    output.mkdir(parents=True, exist_ok=True)
    save(output/"declaration.json", document)
    with (output/"declaration.sha256").open("x") as stream:
        stream.write(digest(output/"declaration.json")+"\n")
    print("TAIL_TICK_DECLARED", digest(output/"declaration.json"), "no quotes parsed or detector calibrated", flush=True)
    return document


def frozen(output):
    offline()
    output = Path(output)
    document = read_json(output/"declaration.json")
    if digest(output/"declaration.json") != (output/"declaration.sha256").read_text().strip():
        raise ValueError("Frozen tail-tick declaration changed")
    safety(document)
    if (document["stage"] != "tail_tick_precalibration_declaration" or document["dates"] != list(DATES) or
            document["symbols"] != list(SYMBOLS) or document["config"] != CONFIG or
            document["quotes_parsed"] is not False or document["detector_calibrated"] is not False or
            document["events_or_hazards_evaluated"] is not False):
        raise ValueError("Declared pilot rules or premeasurement state changed")
    if science_hashes() != document["science_code_hashes"] or verified_round7_metadata() != document["lineage"]:
        raise ValueError("Frozen pilot science or inherited inputs changed")
    for symbol in SYMBOLS:
        count = sum(document["lineage"]["sources"][symbol][day]["rows"] for day in DATES)
        if document["observed_rows"][symbol] != count or document["row_cutoffs"][symbol] != cutoffs(count):
            raise ValueError("Observed-row counts or fixed cutoffs changed")
    return document


def load_tick_day(value):
    """Decode canonical CSV quote text with Python float, without parser rounding."""
    for filekey, shakey in (("clean_file", "clean_sha256"), ("raw_pages_file", "raw_pages_sha256"),
                            ("page_audit_file", "page_audit_sha256")):
        if digest(path_for(value[filekey])) != value[shakey]:
            raise ValueError("Immutable tick source bytes changed")
    epochs, quotes = [], []
    with path_for(value["clean_file"]).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["epoch", "quote"]:
            raise ValueError("Exact epoch/quote source schema required")
        for row in reader:
            stamp = int(row["epoch"])
            quote = float(row["quote"])
            if (str(stamp) != row["epoch"] or (epochs and stamp <= epochs[-1]) or
                    not value["start_epoch"] <= stamp < value["end_exclusive_epoch"] or
                    not math.isfinite(quote) or quote <= 0):
                raise ValueError("Ticks require canonical sorted seconds and finite positive quotes")
            epochs.append(stamp); quotes.append(quote)
    missing = 86400-len(epochs)
    if (not epochs or len(epochs) != value["rows"] or value["end_exclusive_epoch"]-value["start_epoch"] != 86400 or
            value["expected_grid_rows"] != 86400 or value["missing_seconds"] != missing or
            value["gap_free"] is not (missing == 0)):
        raise ValueError("Tick day coverage differs from frozen metadata")
    index = pd.to_datetime(epochs, unit="s", utc=True).astype("datetime64[ns, UTC]")
    return pd.DataFrame({"quote": quotes}, index=index)


def band_summary(values, events):
    values = np.asarray(values, dtype=float)
    events = np.asarray(events, dtype=bool)
    if values.ndim != 1 or events.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional band increments are required")
    count, event_count = len(values), int(events.sum())
    event_sum = math.fsum(values[events])
    non_event_sum = math.fsum(values[~events])
    total = math.fsum(values)
    event_mean = event_sum/event_count if event_count else None
    non_event_count = count-event_count
    non_event_mean = non_event_sum/non_event_count if non_event_count else None
    hazard = event_count/count if count else None
    mean = total/count if count else None
    reconstruct = hazard*event_mean+(1-hazard)*non_event_mean if event_count and non_event_count else None
    error = abs(mean-reconstruct) if reconstruct is not None else None
    if error is not None and not math.isclose(mean, reconstruct, rel_tol=1e-11, abs_tol=1e-15):
        raise ValueError("Tail/non-tail mean decomposition identity failed")
    return {"exposure_seconds": count, "events": event_count, "hazard": hazard,
            "event_return_sum": event_sum, "event_mean_d": event_mean,
            "non_event_seconds": non_event_count, "non_event_return_sum": non_event_sum,
            "non_event_mean_d": non_event_mean, "return_sum": total, "unconditional_mean_d": mean,
            "decomposition_identity_available": reconstruct is not None,
            "decomposition_reconstructed_mean_d": reconstruct, "decomposition_identity_abs_error": error}


def directed_quantiles(values):
    values = np.asarray(values, dtype=float)
    probabilities = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
    names = ("p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100")
    if not np.isfinite(values).all():
        raise ValueError("Finite directed increments required for quantiles")
    return dict(zip(names, np.quantile(values, probabilities).tolist() if len(values) else [None]*len(names), strict=True))


def segment_summary(description, start, end):
    integer(start, "segment start"); integer(end, "segment end")
    if not 0 <= start <= end <= len(description):
        raise ValueError("Segment must be a positional slice of the described stream")
    part = description.iloc[start:end]
    days = part.index.strftime("%Y-%m-%d")
    observed_days = list(dict.fromkeys(days.tolist()))
    if any(day not in DATES for day in observed_days):
        raise ValueError("Unscheduled quote day in pilot segment")
    valid = part.increment_known.to_numpy(bool)
    returns = part.directed_log_return.to_numpy(float)
    events = part.tail_event.fillna(False).to_numpy(bool)
    known_age = part.pre_event_age_seconds.notna().to_numpy(bool)
    if (not np.isfinite(returns[valid]).all() or part.tail_event.isna().to_numpy()[valid].any() or
            np.any(known_age & ~valid)):
        raise ValueError("Descriptor validity/event/age states are inconsistent")
    band_labels = part.age_band.fillna("").to_numpy(str)
    if np.any(known_age & ~np.isin(band_labels, BANDS)) or np.any(~known_age & (band_labels != "")):
        raise ValueError("Known prior age requires exactly one declared band")
    summaries = {band: band_summary(returns[known_age & (band_labels == band)], events[known_age & (band_labels == band)]) for band in BANDS}
    daily = []
    for day in observed_days:
        day_mask = np.asarray(days == day)
        daily.append({"date": day, "observed_rows": int(day_mask.sum()), "bands": {
            band: band_summary(returns[day_mask & known_age & (band_labels == band)], events[day_mask & known_age & (band_labels == band)])
            for band in BANDS}})
    return {"status": "MEASURED_EXPLORATORY", "row_start_inclusive": start, "row_end_exclusive": end,
            "observed_rows": len(part), "first_observed_utc": part.index[0].isoformat() if len(part) else None,
            "last_observed_utc": part.index[-1].isoformat() if len(part) else None,
            "observed_scheduled_days": observed_days, "observed_cluster_count": len(observed_days),
            "valid_increments": int(valid.sum()), "initial_or_gap_pair_exclusions": int((~valid).sum()),
            "unknown_age_exclusions": int((valid & ~known_age).sum()),
            "known_age_increments": int(known_age.sum()), "detected_events": int(events[valid].sum()),
            "events_with_unknown_prior_age": int((events & valid & ~known_age).sum()),
            "directed_sign_counts": {"positive": int((returns[valid] > 0).sum()),
                                     "negative": int((returns[valid] < 0).sum()), "zero": int((returns[valid] == 0).sum())},
            "directed_return_quantiles": directed_quantiles(returns[valid]), "bands": summaries, "daily_bands": daily}


def paired_hazard_comparison(segment, repeats=BOOTSTRAP, seed=SEED):
    integer(repeats, "bootstrap repeats", 1); integer(seed, "bootstrap seed")
    daily = segment["daily_bands"]
    days = [row["date"] for row in daily]
    if days != segment["observed_scheduled_days"] or len(set(days)) != len(days) or any(day not in DATES for day in days):
        raise ValueError("Bootstrap units must be the segment's observed scheduled days")
    totals = {}
    for band in BANDS:
        exposures = np.asarray([integer(row["bands"][band]["exposure_seconds"], "day exposure") for row in daily], dtype=float)
        counts = np.asarray([integer(row["bands"][band]["events"], "day events") for row in daily], dtype=float)
        if np.any(counts > exposures):
            raise ValueError("Day event count cannot exceed exposure")
        expected = segment["bands"][band]
        if exposures.sum() != expected["exposure_seconds"] or counts.sum() != expected["events"]:
            raise ValueError("Day sufficient statistics do not reconstruct the segment")
        totals[band] = (exposures, counts)
    ratios = np.full(repeats, np.nan)
    if days:
        # A single shared draw matrix preserves day pairing between age bands.
        weights = np.random.default_rng(seed).multinomial(len(days), np.full(len(days), 1/len(days)), size=repeats)
        young_e, young_j = (weights @ array for array in totals["lt_N"])
        old_e, old_j = (weights @ array for array in totals["ge_N"])
        young = np.divide(young_j, young_e, out=np.full(repeats, np.nan), where=young_e > 0)
        old = np.divide(old_j, old_e, out=np.full(repeats, np.nan), where=old_e > 0)
        ratios = np.divide(old, young, out=np.full(repeats, np.nan), where=np.isfinite(young) & (young > 0) & np.isfinite(old))
    young_hazard, old_hazard = (segment["bands"][band]["hazard"] for band in BANDS)
    point = old_hazard/young_hazard if young_hazard is not None and young_hazard > 0 and old_hazard is not None else None
    finite = ratios[np.isfinite(ratios)]
    return {"status": "DESCRIPTIVE_EXPLORATORY", "older_over_younger_hazard_ratio": point,
            "hazard_ratio_ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None],
            "bootstrap_repeats": repeats, "bootstrap_seed": seed, "valid_ratio_replicates": len(finite),
            "undefined_ratio_replicates": repeats-len(finite), "paired_scheduled_day_resampling": True,
            "same_day_draws_for_both_bands": True, "observed_cluster_count": len(days),
            "observed_scheduled_days": days, "zero_band_exposure_days": {
                band: int((totals[band][0] == 0).sum()) for band in BANDS},
            "interval_omits_undefined_draws": True, "undefined_draws_block_falsification": True,
            "inference_kind": "conditional_on_fixed_detector_descriptive_iid_observed_scheduled_days",
            "training_median_refits_in_bootstrap": False, "no_contiguous_weekly_claim": True,
            "no_independent_tick_inference_claim": True, "no_strategy_or_profitability_claim": True}


def falsification_gate(segment, comparison):
    reasons = []
    counts_ok = all(segment["bands"][band]["events"] >= 100 for band in BANDS)
    all_defined = (comparison["bootstrap_repeats"] == BOOTSTRAP and comparison["valid_ratio_replicates"] == BOOTSTRAP and
                   comparison["undefined_ratio_replicates"] == 0)
    lower, upper = comparison["hazard_ratio_ci95"]
    point = comparison["older_over_younger_hazard_ratio"]
    finite_ratio = lambda value: (not isinstance(value, bool) and isinstance(value, (int, float)) and
                                 math.isfinite(value) and value >= 0)
    if not counts_ok:reasons.append("fewer_than100_events_in_at_least_one_band")
    if not all_defined:reasons.append("not_all9999_bootstrap_ratios_defined")
    if not all(finite_ratio(value) for value in (point, lower, upper)) or lower > upper:
        reasons.append("undefined_or_invalid_ratio_or_interval")
    insufficient = bool(reasons)
    falsified = not insufficient and upper < 1.5
    if not insufficient and not falsified:reasons.append("upper_bound_not_below1.5")
    return {"status": "INSUFFICIENT_EVIDENCE" if insufficient else
            ("REJECT_LARGE_OVERDUE_EFFECT" if falsified else "LARGE_OVERDUE_EFFECT_NOT_REJECTED"),
            "large_overdue_effect_falsified": falsified, "reasons": reasons,
            "minimum_events_per_band": 100, "hazard_ratio_upper_threshold": 1.5,
            "all9999_ratios_defined": all_defined, "does_not_establish_independence": True,
            "does_not_test_profitability": True, "strategy_eligible": False}


def not_tested_segments(cuts, reason):
    return {name: {"status": "NOT TESTED", "reason": reason, "row_start_inclusive": start,
                   "row_end_exclusive": end} for name, (start, end) in segment_slices(cuts).items() if name != "calibration40"}


def evaluate_symbol(ticks, symbol, cuts):
    if symbol not in SYMBOLS or cuts != cutoffs(len(ticks)):
        raise ValueError("Declared symbol and observed-row cutoffs required")
    side = 1 if symbol == "BOOM600" else -1
    prefix = ticks.iloc[:cuts["40"]]
    result = {"side": side, "observed_rows": len(ticks), "row_cutoffs": cuts,
              "detector_fit_attempts": 1, "no_detector_refit": True,
              "physical_spike_census": False, "strategy_eligible": False,
              "profit_factor": "NOT TESTED", "actual_money_profit": "NOT TESTED"}
    if len(prefix) < 2:
        reason = "fewer_than_two_calibration_quotes"
        result.update(detector={"adequate": False, "median_abs_log_return": None, "threshold": None,
                                "threshold_multiplier": 10., "reason": reason, "calibration_events": None},
                      calibration40={"status": "NOT TESTED", "reason": reason,
                                     "row_start_inclusive": 0, "row_end_exclusive": cuts["40"]},
                      segments=not_tested_segments(cuts, reason), comparisons={"status": "NOT TESTED", "reason": reason})
        return result
    try:
        detector = fit_fixed_tail_detector(prefix, side)
    except ValueError as exc:
        if not (str(exc).startswith("median_abs_log_return must be finite and positive") or
                str(exc) == "At least one observed consecutive training increment is required"):
            raise
        reason = "zero_or_invalid_calibration_scale_or_no_consecutive_increment"
        result.update(detector={"adequate": False, "median_abs_log_return": None, "threshold": None,
                                "threshold_multiplier": 10., "reason": reason, "calibration_events": None},
                      calibration40={"status": "NOT TESTED", "reason": reason,
                                     "row_start_inclusive": 0, "row_end_exclusive": cuts["40"]},
                      segments=not_tested_segments(cuts, reason), comparisons={"status": "NOT TESTED", "reason": reason})
        return result
    calibration = segment_summary(describe_tick_tail(prefix, detector, NOMINAL_N), 0, len(prefix))
    adequate = calibration["detected_events"] >= 100
    result["detector"] = {"adequate": adequate, "median_abs_log_return": detector.median_abs_log_return,
                          "threshold": detector.threshold, "threshold_multiplier": 10.,
                          "calibration_events": calibration["detected_events"],
                          "calibration_source": "earliest40_percent_observed_rows_only"}
    result["calibration40"] = calibration
    if not adequate:
        reason = "fewer_than100_calibration_tail_events"
        result["detector"]["reason"] = reason
        result.update(segments=not_tested_segments(cuts, reason), comparisons={"status": "NOT TESTED", "reason": reason})
        return result
    description = describe_tick_tail(ticks, detector, NOMINAL_N)
    segments = {name: segment_summary(description, start, end) for name, (start, end) in segment_slices(cuts).items() if name != "calibration40"}
    comparisons = {}
    for name in ("dev_validation", "final30"):
        comparison = paired_hazard_comparison(segments[name])
        comparison["falsification"] = falsification_gate(segments[name], comparison)
        comparisons[name] = comparison
    result.update(segments=segments, comparisons=comparisons)
    return result


def evaluate(output):
    offline()
    output = Path(output)
    if any((output/name).exists() for name in ("results.json", "results.sha256", "metrics.csv")):
        raise ValueError("Refusing outcome overwrite")
    document = frozen(output)
    result = {"stage": "exploratory_tail_tick_mechanism_results", "run_utc": datetime.now(timezone.utc).isoformat(),
              "declaration_sha256": digest(output/"declaration.json"), "safety": offline(), "config": CONFIG,
              "dates": list(DATES), "symbols": {}, "goal_achieved": False,
              "historical_strategy_candidate": False, "actual_money_profit": "NOT TESTED",
              "prospective_paper": "NOT TESTED", "physical_spike_census": False,
              "study_kind": "adaptive_known_history_tail_tick_mechanism_pilot_not_strategy_validation"}
    rows = []
    for symbol in SYMBOLS:
        sources = document["lineage"]["sources"][symbol]
        frames = [load_tick_day(sources[day]) for day in DATES]
        ticks = pd.concat(frames)
        if len(ticks) != document["observed_rows"][symbol] or not ticks.index.is_unique or not ticks.index.is_monotonic_increasing:
            raise ValueError("Loaded quotes differ from the immutable chronological source")
        value = evaluate_symbol(ticks, symbol, document["row_cutoffs"][symbol])
        result["symbols"][symbol] = value
        if value["detector"]["adequate"]:
            for name, segment in {"calibration40": value["calibration40"], **value["segments"]}.items():
                for band, metrics in segment["bands"].items():
                    rows.append({"symbol": symbol, "segment": name, "band": band, **metrics})
        print(symbol, "detector_adequate", value["detector"]["adequate"], flush=True)
    frozen(output)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(output/"results.json", result)
    with (output/"results.sha256").open("x") as stream:
        stream.write(digest(output/"results.json")+"\n")
    fields = ["symbol", "segment", "band", *band_summary([], []).keys()]
    with (output/"metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print("TAIL_TICK_RESULTS_SAVED", digest(output/"results.json"), "strategy/PF goal remains NOT TESTED", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("declare", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    (declare if args.stage == "declare" else evaluate)(args.output)


if __name__ == "__main__":
    main()
