#!/usr/bin/env python3
"""Independent stdlib audit of the fixed M5 age/mark economic pilot.

Only pinned independent scalar verifiers are reused. Research engines, model
fitting and bootstrap generation are not imported. Historical execution needs
separate results-ready authorization; this module is safe to test synthetically.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_spike_tick_execution as v7
from scripts import verify_spike_tick_tail as v8

FLAGS, SYMBOLS, DATES = v7.FLAGS, v7.SYMBOLS, v7.DATES
MODES, FAMILIES = ("SPIKE", "DRIFT"), ("RIDGE44", "RIDGE46")
COUNTERS = ("issued", "filled", "completed", "censored", "missing_entry", "missing_path",
            "outside_partition", "purged", "overlap_skipped", "ambiguous")
DEPENDENCIES = {
    "scripts/verify_spike_tick_execution.py": "f2360d5556fbc43ef3bb140df4e6dfd5d80767f859fb37f38934bd11872b476b",
    "scripts/verify_spike_tick_tail.py": "a3b94835b857e3bcb839bb8100f1c5ca037e56456bf954b777c743b069c54a5a",
}
BASE_NAMES = ("aligned_return_1_atr", "aligned_return_3_atr", "aligned_return_6_atr", "aligned_return_12_atr",
              "aligned_body_atr", "range_atr", "favorable_wick_atr", "adverse_wick_atr", "favorable_close_position",
              "atr_5_over_30", "bb_width_over_prior_median_100", "range_over_prior_median_20", "body_over_range",
              "mean_range_3_over_prior_median_20", "large_completed_bar", "log1p_large_bar_age", "log_atr_over_price",
              "aligned_distance_prior_favorable_12_atr", "aligned_distance_prior_adverse_12_atr")
EXTRA_NAMES = ("h4_aligned_return_3_atr", "h4_range_atr", "h4_favorable_close_position",
               "h4_aligned_distance_prior_favorable_12_atr", "h1_aligned_body_atr", "h1_range_atr",
               "h1_favorable_close_position", "h1_aligned_distance_favorable_boundary_m5_atr",
               "h1_aligned_distance_adverse_boundary_m5_atr", "h1_recent_m5_sweep_reclaim", "m15_aligned_return_3_atr",
               "m15_atr_5_over_30", "m15_range_atr", "m15_aligned_body_atr", "m15_favorable_close_position",
               "m15_aligned_distance_prior_favorable_5_atr", "m15_favorable_fvg_atr", "m1_aligned_body_atr",
               "m1_range_atr", "m1_aligned_return_3_atr", "m1_atr_5_over_30", "m1_body_over_range",
               "m1_aligned_distance_prior_favorable_5_atr", "m1_favorable_fvg_atr", "m1_body_over_prior_median_20")
NAMES44 = (*BASE_NAMES, *EXTRA_NAMES)
NAMES46 = (*NAMES44, "tail_age_log1p", "prior_tail_mark")
CONFIG = {"exit": v7.CONFIG, "purge_minutes": 31, "cadence_minutes": 5,
          "ridge_penalty": .1, "training_quantile": .75, "minimum_training_labels": 1000,
          "bootstrap_repeats": 9999, "bootstrap_seed": 20261005, "age_boundary_seconds": 600,
          "detector_recalibration": False, "training_labels_may_overlap": True,
          "unknown_payoffs_are_zero": False, "undefined_draws_block_bounded_inference": True}
digest, source_path, read_json = v8.digest, v8.source_path, v8.read_json
iso, date_of = v8.iso, v8.date_of
compare_points = v8.compare_points


class Audit(v8.Audit):
    def equal(self, label, actual, expected):
        if any(type(v) is float and not math.isfinite(v) for v in (actual, expected)):
            self.checks += 1
            safe = lambda v: repr(v) if type(v) is float and not math.isfinite(v) else v
            self.failures.append({"check": label, "actual": safe(actual), "expected": safe(expected)})
            return
        # Scalar sums can differ from BLAS/vector reductions by a few ulps.
        if type(actual) is float and type(expected) in (int, float):
            self.checks += 1
            error = abs(actual - expected)
            key = label.rsplit("/", 1)[-1]
            self.max_numeric_error[key] = max(self.max_numeric_error.get(key, 0), error)
            if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=2e-11):
                self.failures.append({"check": label, "actual": actual, "expected": expected})
        else:
            super().equal(label, actual, expected)


def stamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0 or parsed.microsecond:
        raise ValueError("Whole aware UTC timestamps required")
    return int(parsed.timestamp())


def finite(value, optional=False):
    if optional and value in (None, "", "NaT"):
        return None
    return v7.number(value)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def observed_days(start, end, dates=DATES):
    if type(start) is not int or type(end) is not int or start >= end:
        raise ValueError("Ordered integer UTC partition bounds required")
    return [d for d in dates if max(start, v7.date_bounds(d)[0]) < min(end, v7.date_bounds(d)[1])]


def day_bounds(start, end, dates=DATES):
    return [(d, max(start, v7.date_bounds(d)[0]), min(end, v7.date_bounds(d)[1])) for d in observed_days(start, end, dates)]


def read_signals(path):
    signals = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["signal_time", "atr", "side", "variant", "signal_close", "prior_age_seconds", "score"]
        if reader.fieldnames != expected:
            raise ValueError("Exact M5 issuance CSV schema required")
        for raw in reader:
            row = {"signal_time": stamp(raw["signal_time"]), "atr": finite(raw["atr"]), "side": int(raw["side"]),
                   "variant": raw["variant"], "signal_close": finite(raw["signal_close"]),
                   "prior_age_seconds": finite(raw["prior_age_seconds"])}
            if raw["score"] != "":
                row["score"] = finite(raw["score"])
            if row["signal_time"] % 300 or row["atr"] <= 0 or row["side"] not in (-1, 1) or row["signal_close"] <= 0:
                raise ValueError("Invalid closed UTC M5 issuance")
            if row["prior_age_seconds"] < 1 or not row["prior_age_seconds"].is_integer():
                raise ValueError("Strictly prior whole-second age required")
            if signals and row["signal_time"] <= signals[-1]["signal_time"]:
                raise ValueError("Issuances must be sorted and unique")
            signals.append(row)
    return signals


def issuance_hash(signals):
    return canonical_hash([{**s, "signal_time": iso(s["signal_time"])} for s in signals])


def replay_partition(ticks, signals, start, end, *, overlapping=False, dates=DATES):
    """Singleton or one-open scalar paths; original effective bounds stay fixed."""
    total, ledgers, daily = dict.fromkeys(COUNTERS, 0), [], {}
    for date, left, right in day_bounds(start, end, dates):
        chosen = [s for s in signals if left <= s["signal_time"] < right]
        day_total, rows = dict.fromkeys(COUNTERS, 0), []
        batches = [[s] for s in chosen] if overlapping else [chosen]
        for batch in batches:
            part, counts = v7.scalar_replay(ticks, batch, left, right)
            rows.extend(part)
            for key in COUNTERS:
                day_total[key] += counts[key]
        daily[date] = {"start": iso(left), "end": iso(right), **day_total}
        for key in COUNTERS:
            total[key] += day_total[key]
        ledgers.extend(rows)
    return ledgers, {**total, "days": daily}


def planned_eligible(signals, start, end, dates=DATES):
    return [s for _, left, right in day_bounds(start, end, dates) for s in signals
            if left <= s["signal_time"] < right and s["signal_time"] + 1860 <= right
            and s["signal_time"] + 961 <= right]


def day_sums(rows, days):
    grouped = {day: [] for day in days}
    for row in rows:
        day = date_of(row["signal_time"])
        if day not in grouped:
            raise ValueError("Ledger day is outside observed partition")
        if not row["censored"] and row["net_R"] is not None:
            grouped[day].append(finite(row["net_R"]))
    return [[math.fsum(values), len(values), math.fsum(v for v in values if v > 0), -math.fsum(v for v in values if v < 0)]
            for values in grouped.values()]


def point_metrics(rows, accounting, days):
    sums = day_sums(rows, days)
    n = sum(s[1] for s in sums)
    total, gain, loss = (math.fsum(s[i] for s in sums) for i in (0, 2, 3))
    mean = total / n if n else None
    se = math.sqrt(len(days) / (len(days) - 1) * math.fsum((s[0] - mean * s[1]) ** 2 for s in sums)) / n if n and len(days) > 1 else None
    return {"completed": n, "censored": accounting.get("censored", 0), "missing_entry": accounting.get("missing_entry", 0),
            "invalid_uncensored": sum(not r["censored"] and r["net_R"] is None for r in rows),
            "issued": accounting.get("issued", 0), "overlap_skipped": accounting.get("overlap_skipped", 0),
            "mean_net_R": mean, "profit_factor": gain / loss if loss else None, "sum_net_R": total,
            "sum_gains_R": gain, "sum_losses_R": loss, "active_days": sum(s[1] > 0 for s in sums),
            "observed_days": days, "observed_day_clusters": len(days), "day_ratio_cluster_se": se,
            "day_sums_return_count_gain_loss": sums, "bootstrap_repeats": 9999, "bootstrap_seed": 20261005,
            "interval_scope": "iid_observed_source_days_inside_partition_only", "weeks_or_unsampled_calendar_days_padded": False}


def inference_policy(saved, sums, denominator, audit, label):
    unknown = saved["undefined_draws"]
    audit.require(label + "/undefined_integer", type(unknown) is int and 0 <= unknown <= 9999)
    compare_points(audit, label, saved, {"all_draws_defined": unknown == 0, "conditional_on_defined_draws": unknown > 0})
    interval = saved["ci95"]
    if unknown == 9999:
        audit.equal(label + "/null_interval", interval, None)
    else:
        audit.require(label + "/ordered_finite_interval", isinstance(interval, list) and len(interval) == 2
                      and all(type(v) in (int, float) and math.isfinite(v) for v in interval) and interval[0] <= interval[1])
    if sums is not None:
        if not any(s[denominator] > 0 for s in sums):
            audit.equal(label + "/necessarily_undefined", unknown, 9999)
        if sums and all(s[denominator] > 0 for s in sums):
            audit.equal(label + "/necessarily_defined", unknown, 0)
    return unknown == 0


def audit_metrics(saved, rows, accounting, days, audit, label, *, overlapping=False):
    points = point_metrics(rows, accounting, days)
    if overlapping:
        for key in ("profit_factor", "sum_gains_R", "sum_losses_R"):
            points.pop(key)
        points["day_sums_return_count"] = [s[:2] for s in points.pop("day_sums_return_count_gain_loss")]
        points["profit_factor_evidence"] = "NOT TESTED_overlapping_targets_are_not_a_strategy_portfolio"
    compare_points(audit, label, saved, points)
    sums = day_sums(rows, days)
    all_mean = inference_policy(saved["mean_inference"], sums, 1, audit, label + "/mean_inference")
    audit.equal(label + "/same_mean_ci", saved["day_mean_ci95"], saved["mean_inference"]["ci95"])
    score = points["mean_net_R"] - 1.96 * points["day_ratio_cluster_se"] if points["mean_net_R"] is not None and points["day_ratio_cluster_se"] is not None and all_mean else None
    audit.equal(label + "/selection_score", saved["selection_score"], score)
    if not overlapping:
        all_pf = inference_policy(saved["pf_inference"], sums, 3, audit, label + "/pf_inference")
        audit.equal(label + "/bounded_inference", saved["bounded_inference_allowed"], all_mean and all_pf)


def paired_policy(saved, left, right, days, audit, label):
    if "partial_descriptive_only" in saved:
        audit.equal(label + "/untested_status", saved["status"], "NOT TESTED")
        saved = saved["partial_descriptive_only"]
    a, b = day_sums(left, days), day_sums(right, days)
    na, nb = sum(s[1] for s in a), sum(s[1] for s in b)
    point = math.fsum(s[0] for s in a) / na - math.fsum(s[0] for s in b) / nb if na and nb else None
    compare_points(audit, label, saved, {"mean_difference_R": point, "observed_days": days,
                   "observed_day_clusters": len(days), "bootstrap_repeats": 9999,
                   "interpretation": "pooled_path_mean_difference_paired_observed_day_draws", "discovery_claim": False})
    inference_policy(saved, None, 1, audit, label)
    if not na or not nb:
        audit.equal(label + "/zero_total_forces_undefined", saved["undefined_draws"], 9999)
    if a and all(x[1] > 0 and y[1] > 0 for x, y in zip(a, b, strict=True)):
        audit.equal(label + "/every_day_both_defined", saved["undefined_draws"], 0)


def prediction(features, estimator):
    names = estimator["feature_names"]
    vectors = [estimator[key] for key in ("means", "std", "coefs")]
    if any(len(v) != len(names) for v in vectors) or any(finite(s) <= 0 for s in vectors[1]):
        raise ValueError("Saved scaler/coefficient dimensions or scales invalid")
    score = finite(estimator["intercept"]) + math.fsum((finite(features[n]) - finite(m)) / finite(s) * finite(c)
               for n, m, s, c in zip(names, *vectors, strict=True))
    return finite(score)


def matrix_hash(names, rows, labels):
    ordered = sorted((r for r in labels if not r["censored"] and r["net_R"] is not None), key=lambda r: r["signal_time"])
    times = [r["signal_time"] for r in ordered]
    if len(times) != len(set(times)):
        raise ValueError("Duplicate completed training identity")
    x = b"".join(struct.pack("=d", finite(rows[t][n])) for t in times for n in names)
    y = b"".join(struct.pack("=d", finite(r["net_R"])) for r in ordered)
    clock = b"".join(struct.pack("=q", t * 1_000_000_000) for t in times)
    return (hashlib.sha256(json.dumps(list(names)).encode() + x + y + clock).hexdigest(),
            hashlib.sha256(clock + y).hexdigest(), ordered)


def validate_model(model, inputs, labels, start, end, audit, label):
    names = NAMES44 if model["family"] == "RIDGE44" else NAMES46
    audit.require(label + "/family", model["family"] in FAMILIES)
    compare_points(audit, label, model, {"feature_names": list(names), "training_start": iso(start), "training_end": iso(end),
                   "training_labels_may_overlap": True, "statistically_independent_labels": False,
                   "counts_toward_profit_sample_target": False, "label_purge_minutes": 31})
    matrix, target, known = matrix_hash(names, inputs, labels)
    planned = {r["signal_time"] for r in planned_eligible(labels, start, end)}
    audit.require(label + "/causal_training_labels", all(r["signal_time"] in planned for r in labels))
    n = len(known)
    compare_points(audit, label + "/training_identity", model, {"matrix_and_target_sha256": matrix,
                   "common_timestamp_target_sha256": target, "training_completed_labels": n,
                   "training_censored_labels": sum(r["censored"] for r in labels),
                   "training_invalid_labels": sum(not r["censored"] and r["net_R"] is None for r in labels),
                   "training_latest_issue": iso(known[-1]["signal_time"]) if n else None})
    if n < 1000:
        compare_points(audit, label + "/insufficient_fit", model, {"status": "NOT TESTED", "estimator": None,
                       "serialized_estimator_sha256": None})
        return
    audit.equal(label + "/tested_fit", model["status"], "TESTED")
    estimator = model["estimator"]
    compare_points(audit, label + "/estimator", estimator, {"version": 1, "feature_names": list(names), "fit_rows": n,
                   "penalty": .1, "quantile": .75, "score_kind": "continuous_uncalibrated_score",
                   "scaler_fitted_on": "training_rows_only", "target_clipped": False, "features_clipped": False})
    audit.equal(label + "/serialized_estimator", model["serialized_estimator_sha256"], canonical_hash(estimator))
    audit.equal(label + "/intercept", estimator["intercept"], statistics.fmean(r["net_R"] for r in known))
    for index, name in enumerate(names):
        values = [inputs[r["signal_time"]][name] for r in known]
        audit.equal(label + "/mean/" + name, estimator["means"][index], statistics.fmean(values))
        saved_mean = finite(estimator["means"][index])
        std = math.sqrt(math.fsum((v - saved_mean) ** 2 for v in values) / n) or 1.
        audit.equal(label + "/std/" + name, estimator["std"][index], std)
    scores = [prediction(inputs[r["signal_time"]], estimator) for r in known]
    audit.equal(label + "/training_only_q75_threshold", estimator["threshold"], max(0., v8.quantile(scores, .75)))


def artifact(value, audit):
    if value.get("local_ignored_artifact") is not True:
        raise ValueError("Saved local evidence artifact required")
    path = source_path(value["path"])
    audit.pin(path, value["sha256"])
    return path


def pin_artifacts(value, audit):
    if isinstance(value, dict):
        if value.get("local_ignored_artifact") is True:
            artifact(value, audit)
        for child in value.values():
            pin_artifacts(child, audit)
    elif isinstance(value, list):
        for child in value:
            pin_artifacts(child, audit)


def read_inputs(value, end, minute, tail, audit, label, fingerprints=None):
    """Stream all rows, retain only the shared available44/46 matrix."""
    path = artifact(value["inputs"], audit)
    eligible, previous, count = {}, None, 0
    frame_cache = {}
    start = v7.date_bounds(DATES[0])[0]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"m5_open_time", "signal_time", *NAMES46, "atr", "close", "prior_age_seconds",
                    "original_feature_valid", "original44_all_finite", "multiframe_feature_valid",
                    "tail_feature_valid", "common_available", "feature_valid"}
        required |= {name + suffix for name in ("h4", "h1", "m15", "m1") for suffix in ("_closed_at", "_row_valid")}
        audit.require(label + "/input_schema", required == set(reader.fieldnames))
        for raw in reader:
            t, opening = stamp(raw["signal_time"]), stamp(raw["m5_open_time"])
            audit.require(label + "/fixed_clock", t % 300 == 0 and t == opening + 300 and start <= t < end
                          and t == (start if previous is None else previous + 300))
            previous, count = t, count + 1
            if fingerprints is not None:
                fingerprint = canonical_hash(raw)
                if t in fingerprints:
                    audit.equal(label + "/full_grid_prefix_consistency", fingerprint, fingerprints[t])
                else:
                    fingerprints[t] = fingerprint
            values = {n: finite(raw[n], True) for n in NAMES46}
            flags = {n: v7.boolean(raw[n]) for n in ("original_feature_valid", "original44_all_finite", "multiframe_feature_valid",
                     "tail_feature_valid", "common_available", "feature_valid")}
            frame_valid = flags["original_feature_valid"] and all(values[n] is not None for n in NAMES44)
            state = tail.get(t)
            tail_valid = state is not None and state.known and state.age is not None and state.prior_mark is not None
            common = frame_valid and tail_valid
            compare_points(audit, label + "/mask", flags, {"original44_all_finite": all(values[n] is not None for n in NAMES44),
                           "multiframe_feature_valid": frame_valid, "tail_feature_valid": tail_valid,
                           "common_available": common, "feature_valid": common})
            age, mark = (state.age, state.prior_mark) if state is not None else (None, None)
            audit.equal(label + "/strict_prior_age", finite(raw["prior_age_seconds"], True), age)
            audit.equal(label + "/strict_prior_mark", values["prior_tail_mark"], mark)
            audit.equal(label + "/age_transform", values["tail_age_log1p"], math.log1p(age / 600) if age is not None else None)
            contexts_known = True
            for name, step in (("h4", 14400), ("h1", 3600), ("m15", 900), ("m1", 60)):
                closed = stamp(raw[name + "_closed_at"]) if raw[name + "_closed_at"] not in ("", "NaT") else None
                expected_close = t // step * step
                audit.equal(label + "/latest_closed/" + name, closed, expected_close)
                key = (name, expected_close)
                if key not in frame_cache:
                    frame_cache[key] = all(s in minute for s in range(expected_close - step, expected_close, 60))
                valid = frame_cache[key]
                audit.equal(label + "/context_valid/" + name, v7.boolean(raw[name + "_row_valid"]), valid)
                contexts_known &= valid
            if flags["original_feature_valid"]:
                audit.require(label + "/available_context", contexts_known)
            if common:
                atr = v7.causal_atr(minute, t)
                close = minute[t - 60][3]
                audit.equal(label + "/causal_atr", finite(raw["atr"]), atr)
                audit.equal(label + "/closed_signal_price", finite(raw["close"]), close)
                eligible[t] = {**values, "atr": finite(raw["atr"]), "close": finite(raw["close"]), "prior_age_seconds": age}
    compare_points(audit, label + "/prefix_metadata", value, {"feature_end_exclusive": iso(end),
                   "suffix_calculations": False, "detector_refit": False, "clock": "every closed UTC M5",
                   "eligible_common_clock_rows": len(eligible), "canonical_boundary_day_quotes_decoded": True})
    audit.equal(label + "/saved_input_rows", value["inputs"]["rows"], count)
    audit.equal(label + "/complete_original_M5_grid", count, len(range(start, end, 300)))
    audit.require(label + "/last_closed_m1", stamp(value["last_m1_close"]) <= end)
    audit.require(label + "/last_prefix_tick", stamp(value["tick_last_used"]) < end)
    return eligible


def m1_prefix_inventory(minutes, end):
    """Count closed UTC slots separately from available prices, without filling.

    The frozen loader retains a dense calendar grid with NaN at source gaps.
    The independent price dictionary remains sparse so a gap cannot become a
    valid context candle or contribute a fabricated true range.
    """
    if not minutes or type(end) is not int or any(type(t) is not int or t % 60 for t in minutes):
        raise ValueError("Nonempty UTC minute openings and whole-second cutoff required")
    first, last = min(minutes), max(minutes)
    closed_end = max(first, min(last + 60, end // 60 * 60))
    slots = (closed_end - first) // 60
    available = sum(t < closed_end for t in minutes)
    return {"closed_grid_slots": slots, "available_closed_candles": available,
            "unavailable_closed_slots": slots - available,
            "last_grid_close": closed_end if slots else None}


def expected_signals(inputs, symbol, mode, family="CLOCK", model=None, start=None, end=None):
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    result = []
    for t, row in sorted(inputs.items()):
        if start is not None and not start <= t < end:
            continue
        score = None
        if family != "CLOCK":
            if model["status"] == "NOT TESTED":
                continue
            score = prediction(row, model["estimator"])
            if score < model["estimator"]["threshold"] or score <= 0:
                continue
        signal = {"signal_time": t, "atr": row["atr"], "side": side, "variant": family + "_" + mode,
                  "signal_close": row["close"], "prior_age_seconds": float(row["prior_age_seconds"])}
        if score is not None:
            signal["score"] = score
        result.append(signal)
    return result


def audit_signals(value, expected, audit, label):
    saved = read_signals(artifact(value, audit))
    audit.equal(label + "/rows", value["rows"], len(saved))
    compare_points(audit, label + "/issued", saved, expected)
    audit.equal(label + "/canonical_saved_hash", value["canonical_issuance_sha256"], issuance_hash(saved))
    return saved


def audit_paths(value, expected, audit, label):
    saved = v7.read_ledger(artifact(value, audit))
    audit.equal(label + "/rows", value["rows"], len(saved))
    compare_points(audit, label + "/scalar_paths", saved, expected)
    return saved


def audit_accounting(saved, expected, audit, label, overlapping):
    compare_points(audit, label, saved, {k: expected[k] for k in COUNTERS})
    audit.require(label + "/count_conservation", saved["issued"] == sum(saved[k] for k in
                  ("outside_partition", "purged", "overlap_skipped", "missing_entry", "filled")) and
                  saved["filled"] == saved["completed"] + saved["censored"])
    compare_points(audit, label + "/policy", saved, {"overlapping_labels": overlapping,
                   "statistically_independent_labels": False if overlapping else None, "strategy_returns": not overlapping,
                   "counts_toward_profit_sample_target": not overlapping, "effective_day_end_and_planned_purge": True})
    audit.require(label + "/exact_day_set", set(saved["days"]) == set(expected["days"]))
    for day, counts in expected["days"].items():
        compare_points(audit, label + "/day/" + day, saved["days"][day], counts)
        if overlapping:
            compare_points(audit, label + "/target_policy/" + day, saved["days"][day], {
                "label_kind": "individually_replayed_overlapping_opportunity_training_targets",
                "no_statistical_independence_claim": True, "labels_may_overlap": True,
                "profit_factor_evidence": False, "counts_toward_profit_sample_target": False})


def audit_report(report, expected, ticks, start, end, audit, label):
    signals = audit_signals(report["signals"], expected, audit, label)
    audit.equal(label + "/issuance_hash", report["issuance_sha256"], issuance_hash(signals))
    paths, counts = replay_partition(ticks, signals, start, end)
    saved = audit_paths(report["ledger"], paths, audit, label)
    audit_accounting(report["audit"], counts, audit, label + "/accounting", False)
    audit_metrics(report["metrics"], saved, counts, observed_days(start, end), audit, label + "/metrics")
    return saved, counts


def audit_labels(model, inputs, ticks, symbol, mode, start, end, audit, label):
    expected = expected_signals(inputs, symbol, mode, start=start, end=end)
    signals = audit_signals(model["training_signals"], expected, audit, label + "/training_clock")
    audit.equal(label + "/clock_hash", model["training_clock_issuance_sha256"], issuance_hash(signals))
    rows, counts = replay_partition(ticks, signals, start, end, overlapping=True)
    saved = audit_paths(model["training_labels"], rows, audit, label + "/training_labels")
    audit_accounting(model["training_label_audit"], counts, audit, label + "/label_accounting", True)
    validate_model(model, inputs, saved, start, end, audit, label + "/model")
    return saved


def audit_comparisons(saved, ledgers, days, audit, label, fit_statuses=None):
    expected = {"RIDGE46_minus_RIDGE44", "RIDGE44_minus_CLOCK", "RIDGE46_minus_CLOCK"}
    audit.require(label + "/exact_comparisons", set(saved) == expected)
    for key in sorted(expected):
        left, right = key.split("_minus_")
        if fit_statuses is not None:
            tested = all(fit_statuses[family] == "TESTED" for family in (left, right))
            audit.equal(label + "/fit_status/" + key, saved[key]["status"], "TESTED" if tested else "NOT TESTED")
            audit.equal(label + "/partial_wrapper/" + key, "partial_descriptive_only" in saved[key], not tested)
        paired_policy(saved[key], ledgers[left], ledgers[right], days, audit, label + "/" + key)


def age_gate(completed, all_defined, missing, censored, invalid, interval):
    counts = (completed, missing, censored, invalid)
    if any(type(n) is not int or n < 0 for n in counts) or type(all_defined) is not bool:
        raise ValueError("Explicit nonnegative counts and inference flag required")
    if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in interval) or interval[0] > interval[1]):
        raise ValueError("Ordered finite interval or null required")
    adequate = completed >= 100 and all_defined and not (missing or censored or invalid) and interval is not None
    status = ("REJECT_POSITIVE_FIXED_OVERLAPPING_LABEL_MEAN" if adequate and interval[1] <= 0 else
              "NOT_REJECTED_POSITIVE_MEAN" if adequate else "INSUFFICIENT_EVIDENCE")
    return {"status": status, "whole_eligible_policy_rejection_allowed": bool(adequate)}


def audit_age_diagnostic(value, inputs, ticks, symbol, mode, start, end, audit, label):
    expected = expected_signals(inputs, symbol, mode, start=start, end=end)
    signals = audit_signals(value["signals"], expected, audit, label + "/clock")
    rows, counts = replay_partition(ticks, signals, start, end, overlapping=True)
    saved = audit_paths(value["labels"], rows, audit, label + "/labels")
    audit_accounting(value["audit"], counts, audit, label + "/accounting", True)
    eligible = planned_eligible(signals, start, end)
    days = observed_days(start, end)
    for band in ("lt600", "ge600"):
        chosen = [s for s in eligible if (s["prior_age_seconds"] < 600) == (band == "lt600")]
        times = {s["signal_time"] for s in chosen}
        selected = [r for r in saved if r["signal_time"] in times]
        missing = len(times - {r["signal_time"] for r in selected})
        censored = sum(r["censored"] for r in selected)
        invalid = sum(not r["censored"] and r["net_R"] is None for r in selected)
        record = value["bands"][band]
        compare_points(audit, label + "/" + band, record, {"planned_eligible_signals": len(chosen), "missing_entry_labels": missing,
                       "censored_labels": censored, "invalid_labels": invalid, "known_subset_only": bool(missing or censored or invalid),
                       "labels_may_overlap": True, "statistically_independent_labels": False, "strategy_pf_evidence": False,
                       "discovery_claim": False, "scope": "fixed_overlapping_label_mean_not_all46_interactions_or_occupancy_policies"})
        band_counts = {"issued": len(chosen), "censored": censored, "missing_entry": missing}
        audit_metrics(record["metrics"], selected, band_counts, days, audit, label + "/" + band + "/metrics", overlapping=True)
        metrics = record["metrics"]
        compare_points(audit, label + "/" + band + "/gate", record, age_gate(
            metrics["completed"], metrics["mean_inference"]["all_draws_defined"], missing, censored, invalid, metrics["day_mean_ci95"]))
    return len(saved)


def metadata(study, audit):
    declaration, selection, result = (read_json(study / n) for n in ("declaration.json", "selection.json", "results.json"))
    for name, value in (("declaration", declaration), ("selection", selection), ("results", result)):
        audit.pin(study / (name + ".json"), (study / (name + ".sha256")).read_text().strip())
        audit.safety(value, name)
        compare_points(audit, name, value, {"config": CONFIG, "goal_achieved": False, "historical_strategy_candidate": False,
                       "safety": dict.fromkeys(FLAGS, False)})
    compare_points(audit, "declaration", declaration, {"stage": "tick_age_payoff_premeasurement_declaration", "dates": list(DATES),
                   "symbols": list(SYMBOLS), "families": list(FAMILIES), "modes": list(MODES), "quotes_parsed": False,
                   "new_features_labels_models_computed": False})
    compare_points(audit, "selection", selection, {"stage": "frozen_tick_age_payoff_development",
                   "declaration_sha256": digest(study / "declaration.json"), "final30_features_or_payoffs_evaluated": False})
    compare_points(audit, "results", result, {"stage": "fixed_tick_age_payoff_final30_diagnostics",
                   "selection_sha256": digest(study / "selection.json"), "declaration_sha256": digest(study / "declaration.json"),
                   "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED", "price_oos": False,
                   "live_candidate": False, "new_label_release_after_frozen_selection": True})
    audit.require("chronology", v8.timestamp(declaration["run_utc"]) <=
                  v8.timestamp(selection["run_utc"]) <= v8.timestamp(result["run_utc"]))
    for value in (selection, result):
        compare_points(audit, "frozen_lineage", value, {"science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"]})
    for name, expected in {**declaration["science_code_hashes"], **declaration["lineage"]["files"], **DEPENDENCIES}.items():
        audit.pin(source_path(name), expected)
    old_declaration, old_result = v8.metadata(ROOT / "docs/spike_tick_tail_20261005", audit)
    old_audit = read_json(ROOT / "docs/spike_tick_tail_20261005/independent_audit.json")
    audit.require("prior_independent_audit", old_audit["passed"] is True and old_audit["errors"] == [])
    audit.require("prior_tick_source_identity", declaration["lineage"]["tick_sources"] == old_declaration["lineage"]["sources"])
    for symbol in SYMBOLS:
        audit.equal("detector/" + symbol, declaration["lineage"]["detectors"][symbol], old_result["symbols"][symbol]["detector"])
        bounds = declaration["lineage"]["boundaries"][symbol]
        audit.equal("outer_boundary_start/" + symbol, stamp(bounds["0"]), v7.date_bounds(DATES[0])[0])
        audit.equal("outer_boundary_end/" + symbol, stamp(bounds["100"]), v7.date_bounds(DATES[-1])[1])
    pin_artifacts(selection, audit)
    pin_artifacts(result, audit)
    return declaration, selection, result, old_declaration


def verify_symbol(declaration, selected, result, old_declaration, symbol, audit):
    times, quotes, sources = v8.load_symbol(old_declaration, symbol, audit)
    ticks = dict(zip(times, quotes, strict=True))
    cut = v8.cutoffs(len(times))
    detector = declaration["lineage"]["detectors"][symbol]
    scale, _ = v8.fit_scale(times, quotes, cut["40"])
    audit.equal(symbol + "/frozen_training_scale", detector["median_abs_log_return"], scale)
    full = v8.decode(times, quotes, scale, 1 if symbol.startswith("BOOM") else -1)
    tail = {r.epoch: r for r in full if r.epoch % 300 == 0}
    del full
    bounds = {k: stamp(v) for k, v in declaration["lineage"]["boundaries"][symbol].items()}
    for key in ("40", "50", "60", "70"):
        audit.equal(symbol + "/observed_row_cut/" + key, bounds[key], times[cut[key]])
    audit.require(symbol + "/adequate_detector", detector["adequate"] is True)
    m1_value = declaration["lineage"]["m1_sources"][symbol]
    audit.pin(source_path(m1_value["path"]), m1_value["sha256"])
    minute = v7.read_m1(source_path(m1_value["path"]))
    contexts, path_count, training_count, fingerprints, m1_coverage = {}, 0, 0, {}, {}
    full_inventory = m1_prefix_inventory(minute, max(minute) + 60)
    for key in ("40", "50", "60", "70", "100"):
        value = selected["audits"][key] if key != "100" else result["audit"]
        context = read_inputs(value, bounds[key], minute, tail, audit, symbol + "/prefix" + key, fingerprints)
        contexts[key] = context
        audit.equal(symbol + "/prefix_tick_rows/" + key, value["tick_prefix_rows"], sum(t < bounds[key] for t in times))
        audit.equal(symbol + "/prefix_last_tick/" + key, stamp(value["tick_last_used"]), max(t for t in times if t < bounds[key]))
        inventory = m1_prefix_inventory(minute, bounds[key])
        m1_coverage[key] = inventory
        audit.equal(symbol + "/prefix_closed_m1_grid_slots/" + key, value["m1_closed_rows_used"], inventory["closed_grid_slots"])
        audit.equal(symbol + "/prefix_last_m1_close/" + key, stamp(value["last_m1_close"]), inventory["last_grid_close"])
        compare_points(audit, symbol + "/source_grid_inventory/" + key, value,
                       {"source_rows": len(minute), "grid_minutes": full_inventory["closed_grid_slots"],
                        "missing_minutes": full_inventory["unavailable_closed_slots"]})
    for mode in MODES:
        folds, pooled_counts = {}, {}
        for family in FAMILIES:
            candidate = selected["candidates"][mode + "_" + family]
            audit.equal(symbol + "/development_eligible", candidate["development_eligible"], False)
            audit.require(symbol + "/three_folds", len(candidate["walk_forward"]) == 3)
            pooled_counts[family] = dict.fromkeys(COUNTERS, 0)
            for index, (train, test) in enumerate((("40", "50"), ("50", "60"), ("60", "70"))):
                fold = candidate["walk_forward"][index]
                model = {**fold["model"], "training_labels": fold["training_labels"], "training_label_audit": fold["training_label_audit"]}
                label = f"{symbol}/{mode}/{family}/wf{index+1}"
                audit.equal(label + "/training_input_pin", model["training_inputs"], selected["audits"][train]["inputs"])
                labels = audit_labels(model, contexts[train], ticks, symbol, mode, bounds["0"], bounds[train], audit, label)
                training_count += len(labels)
                expected = expected_signals(contexts[test], symbol, mode, family, model, bounds[train], bounds[test])
                ledger, counts = audit_report(fold, expected, ticks, bounds[train], bounds[test], audit, label)
                audit.equal(label + "/common_availability", fold["common_available_rows"],
                            sum(bounds[train] <= t < bounds[test] for t in contexts[test]))
                audit.equal(label + "/fit_report_status", fold["status"], model["status"])
                path_count += len(ledger)
                folds.setdefault(family, []).extend(ledger)
                for counter in COUNTERS:
                    pooled_counts[family][counter] += counts[counter]
        # Per-fold clock files are not referenced by selection. Reconstruct
        # each fold independently, then compare the saved pooled clock ledger.
        clock = []
        pooled_counts["CLOCK"] = dict.fromkeys(COUNTERS, 0)
        for train, test in (("40", "50"), ("50", "60"), ("60", "70")):
            expected = expected_signals(contexts[test], symbol, mode, start=bounds[train], end=bounds[test])
            rows, counts = replay_partition(ticks, expected, bounds[train], bounds[test])
            clock.extend(rows)
            for counter in COUNTERS:
                pooled_counts["CLOCK"][counter] += counts[counter]
        folds["CLOCK"] = clock
        days = observed_days(bounds["40"], bounds["70"])
        fit_statuses = {"CLOCK": "TESTED"}
        for family in ("CLOCK", *FAMILIES):
            pooled = selected["validation"][mode][family]
            saved = audit_paths(pooled["ledger"], folds[family], audit, f"{symbol}/{mode}/{family}/pooled")
            compare_points(audit, symbol + "/" + family + "/pooled_accounting", pooled["audit"], pooled_counts[family])
            if family == "CLOCK":
                path_count += len(saved)
            else:
                statuses = [f["model"]["status"] for f in selected["candidates"][mode + "_" + family]["walk_forward"]]
                status = ("TESTED" if all(s == "TESTED" for s in statuses) else
                          "PARTIALLY TESTED" if any(s == "TESTED" for s in statuses) else "NOT TESTED")
                compare_points(audit, symbol + "/pooled_fit_status/" + family, pooled,
                               {"fold_fit_statuses": statuses, "status": status})
                fit_statuses[family] = status
            audit_metrics(pooled["metrics"], saved, pooled_counts[family], days, audit, f"{symbol}/{mode}/{family}/pooled_metrics")
        audit_comparisons(selected["validation"][mode]["comparisons"], folds, days, audit,
                          symbol + "/" + mode + "/pooled_comparisons", fit_statuses)
        final_ledgers = {}
        for family in FAMILIES:
            model = selected["final_models"][mode + "_" + family]
            audit.equal(symbol + "/final_training_input_pin", model["training_inputs"], selected["audits"]["70"]["inputs"])
            labels = audit_labels(model, contexts["70"], ticks, symbol, mode, bounds["0"], bounds["70"], audit, f"{symbol}/{mode}/{family}/final_model")
            training_count += len(labels)
        pair = [selected["final_models"][mode + "_" + family] for family in FAMILIES]
        for field in ("common_timestamp_target_sha256", "training_clock_issuance_sha256", "training_completed_labels", "training_start", "training_end"):
            audit.equal(symbol + "/matched_final_training/" + field, pair[0][field], pair[1][field])
        for family in ("CLOCK", *FAMILIES):
            model = None if family == "CLOCK" else selected["final_models"][mode + "_" + family]
            expected = expected_signals(contexts["100"], symbol, mode, family, model, bounds["70"], bounds["100"])
            report = result["models"][mode]["reports"][family]
            rows, _ = audit_report(report, expected, ticks, bounds["70"], bounds["100"], audit, f"{symbol}/{mode}/{family}/final30")
            audit.equal(symbol + "/final_common_availability", report["common_available_rows"],
                        sum(bounds["70"] <= t < bounds["100"] for t in contexts["100"]))
            audit.equal(symbol + "/final_report_status", report["status"], "TESTED" if model is None else model["status"])
            final_ledgers[family] = rows
            path_count += len(rows)
            if family != "CLOCK":
                compare_points(audit, symbol + "/frozen_final_report", report, {"development_eligible": False,
                               "historical_gate": False, "live_candidate": False, "model_sha256": model["serialized_estimator_sha256"]})
        audit_comparisons(result["models"][mode]["comparisons"], final_ledgers, observed_days(bounds["70"], bounds["100"]), audit, symbol + "/" + mode + "/final_comparisons")
        for value, lo, hi, key in ((selected["age_diagnostics"][mode], "40", "70", "development"),
                                   (result["age_diagnostics"][mode], "70", "100", "final30")):
            training_count += audit_age_diagnostic(value, contexts[hi], ticks, symbol, mode, bounds[lo], bounds[hi], audit, f"{symbol}/{mode}/age/{key}")
    audit.equal(symbol + "/no_selected_candidate", selected["selected_candidate"], None)
    audit.equal(symbol + "/no_final_candidate", result["selected_candidate"], None)
    audit.equal(symbol + "/zero_eligible_candidates", selected["eligible_candidates"], 0)
    return {"symbol": symbol, "observed_rows": len(times), "available_prefix_rows": {k: len(v) for k, v in contexts.items()},
            "prefix_m1_coverage": m1_coverage,
            "strategy_paths_checked": path_count, "overlapping_label_paths_checked_including_repeated_family_artifacts": training_count}, sources


def verify_metrics_csv(path, documents, development, audit):
    audit.pin(path)
    expected = []
    for symbol in SYMBOLS:
        if documents["symbols"][symbol]["status"] == "NOT TESTED":
            expected.append({"symbol": symbol, "status": "NOT TESTED"})
            continue
        for mode in MODES:
            reports = documents["symbols"][symbol]["validation"][mode] if development else documents["symbols"][symbol]["models"][mode]["reports"]
            for family in ("CLOCK", *FAMILIES):
                report = reports[family]
                expected.append({"symbol": symbol, "mode": mode, "family": family, "status": report.get("status", "TESTED"),
                    **{k: report["metrics"][k] for k in ("completed", "censored", "missing_entry", "profit_factor", "mean_net_R", "active_days", "observed_day_clusters")},
                    "historical_gate": False, "live_candidate": False})
    with path.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    audit.require("table/complete_rows", len(table) == len(expected))
    for raw, row in zip(table, expected, strict=True):
        parsed = {}
        for key, value in row.items():
            parsed[key] = v7.boolean(raw[key]) if type(value) is bool else csv_integer(raw[key]) if type(value) is int else finite(raw[key], True) if type(value) is float or value is None else raw[key]
        if row["status"] == "NOT TESTED" and len(row) == 2:
            audit.require("table/no_untested_metrics", all(value == "" for key, value in raw.items() if key not in row))
        compare_points(audit, "table/" + path.name, parsed, row)


def csv_integer(value):
    number = finite(value)
    if not number.is_integer() or abs(number) > 2 ** 53:
        raise ValueError("Exactly represented integral CSV counter required")
    return int(number)


def verify(study, audit):
    declaration, selection, result, old = metadata(study, audit)
    symbols, sources = [], []
    for symbol in SYMBOLS:
        if declaration["lineage"]["detectors"][symbol]["adequate"] is not True:
            selected, final = selection["symbols"][symbol], result["symbols"][symbol]
            audit_untested_symbol(selected, final, audit, symbol)
            times, _, source = v8.load_symbol(old, symbol, audit)
            symbols.append({"symbol": symbol, "status": "NOT TESTED", "observed_rows": len(times),
                            "strategy_paths_checked": 0, "overlapping_label_paths_checked_including_repeated_family_artifacts": 0})
            sources.extend(source)
            continue
        report, source = verify_symbol(declaration, selection["symbols"][symbol], result["symbols"][symbol], old, symbol, audit)
        symbols.append(report)
        sources.extend(source)
    verify_metrics_csv(study / "development_metrics.csv", selection, True, audit)
    verify_metrics_csv(study / "metrics.csv", result, False, audit)
    for path, expected in tuple(audit.inputs.items()):
        audit.require("unchanged_during_audit/" + path, digest(path) == expected)
    return symbols, sources


def audit_untested_symbol(selected, final, audit, label):
    compare_points(audit, label + "/untested_development", selected, {"status": "NOT TESTED",
                   "reason": "inadequate_fixed_round8_detector", "final_models": {}, "candidates": {}, "selected_candidate": None})
    compare_points(audit, label + "/untested_final", final, {"status": "NOT TESTED",
                   "reason": "inadequate_fixed_round8_detector", "models": {}})
    for field in ("final_models", "candidates"):
        audit.equal(label + "/no_untested_" + field, selected[field], {})
    audit.equal(label + "/no_untested_final_models", final["models"], {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_tick_age_payoff_20261005")
    args = parser.parse_args()
    audit = Audit()
    for flag in FLAGS:
        audit.require("environment/" + flag, os.environ.get(flag, "false").lower() == "false")
    target = args.output / "independent_audit.json"
    if target.exists():
        raise ValueError("Refusing audit overwrite; preserve failed attempt before a repair")
    symbols, sources = [], []
    try:
        symbols, sources = verify(args.output.resolve(), audit)
    except Exception as exc:
        audit.failures.append({"check": "fatal_exception", "type": type(exc).__name__, "message": str(exc)})
    report = {"stage": "independent_tick_age_payoff_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
              "passed": not audit.failures, "checks": audit.checks, "errors": audit.failures, "symbols": symbols,
              "sources": sources, "safety": dict.fromkeys(FLAGS, False), "saved_false_values_checked": audit.safety_values_checked,
              "input_sha256": audit.inputs, "max_numeric_error": audit.max_numeric_error,
              "verifier_file": "scripts/verify_spike_tick_tail_signal.py", "verifier_sha256": digest(Path(__file__)),
              "scope": {"included": ["all24 immutable clean quote/gap grids anchored by passing raw provenance audits",
                         "fixedfirst40 scale, strict-prior age/mark, exact observed-row cuts and saved common M5 availability",
                         "causal ATR/closed context stamps, saved44/46 rows, training identity/matrix/model hashes, scalers and scalar scores",
                         "individually replayed overlapping labels and jointly replayed one-open model/clock paths with planned31minute purge",
                         "observed-day point means/PF/paired differences and saved null/replicate/nonpromotion gate arithmetic"],
                        "excluded": ["regenerating all44 indicators or proving original feature-valid rolling calculations",
                                     "refitting ridge coefficients, bootstrap draw regeneration or inferential validity",
                                     "rechecking rawwire normalization already anchored by the passed immutable round7 audit",
                                     "new price-OOS, statistical independence, broker fills, measured costs, prospective or actual profit"]}}
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"passed": report["passed"], "checks": audit.checks, "sources": len(sources), "symbols": len(symbols),
                      "errors": len(audit.failures), "audit": str(target)}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
