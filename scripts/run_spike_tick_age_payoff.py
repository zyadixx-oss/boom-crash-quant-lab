#!/usr/bin/env python3
"""Frozen round9 M5-clock economic pilot; quote proxies, never orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.learned_signal import fit_ridge, predict_ridge
from app.research.multiframe_signal import causal_multiframe_inputs, FEATURE_NAMES
from app.research.payoff_ticks import replay_ticks, TICK_TRADE_COLUMNS
from app.research.spike_hunter import load_m1
from app.research.tick_tail import FixedTailDetector, describe_tick_tail
from app.research.tick_tail_signal import join_tick_tail_inputs
from app.research.tick_tail_labels import independent_tick_labels, COUNTERS
from scripts import run_spike_tick_tail as round8
from scripts import run_spike_nonlinear_study as round6
from scripts.run_spike_native300_study import check_prepared
from scripts.run_spike_learned_study import development_eligible
from scripts.run_spike_tick_execution import CONFIG as EXIT_CONFIG, NEW_SOURCES as ROUND7_SOURCES

OUTPUT = ROOT / "docs/spike_tick_age_payoff_20261005"
ROUND8 = ROOT / "docs/spike_tick_tail_20261005"
ROUND6 = ROOT / "docs/spike_nonlinear_20261005"
SYMBOLS, DATES = round8.SYMBOLS, round8.DATES
MODES, FAMILIES = ("SPIKE", "DRIFT"), ("RIDGE44", "RIDGE46")
BOOTSTRAP, SEED, PURGE, MIN_TRAIN = 9999, 20261005, 31, 1000
START, END = pd.Timestamp("2026-04-11", tz="UTC"), pd.Timestamp("2026-10-04", tz="UTC")
NEW_SOURCES = ("docs/SPIKE_TICK_AGE_PAYOFF_PROTOCOL.md", "scripts/run_spike_tick_age_payoff.py",
               "backend/tests/test_spike_tick_age_payoff.py", "backend/app/research/tick_tail_signal.py",
               "backend/tests/test_tick_tail_signal.py", "backend/app/research/tick_tail_labels.py",
               "backend/tests/test_tick_tail_labels.py")
SOURCES = tuple(dict.fromkeys((*round6.SOURCES, *ROUND7_SOURCES, *round8.SCIENCE_SOURCES, *NEW_SOURCES)))
CONFIG = {"exit": asdict(EXIT_CONFIG), "purge_minutes": PURGE, "cadence_minutes": 5,
          "ridge_penalty": .1, "training_quantile": .75, "minimum_training_labels": MIN_TRAIN,
          "bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED, "age_boundary_seconds": 600,
          "detector_recalibration": False, "training_labels_may_overlap": True,
          "unknown_payoffs_are_zero": False, "undefined_draws_block_bounded_inference": True}


def digest(path):
    return round8.digest(path)


def canonical(value):
    """Normalize saved scalar metadata, retaining nulls rather than inventing zero."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite saved metadata must be represented explicitly as null")
    return value


def json_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    round8.save(path, canonical(value))


def stamp(value):
    result = pd.Timestamp(value)
    if result.tz is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("Aware UTC boundaries are required")
    return result.tz_convert("UTC")


def hashes():
    return {name: digest(round8.path_for(name)) for name in SOURCES}


def verified_round8_metadata():
    """Read saved metadata and hash bytes only; never parse any prices here."""
    prior = round8.frozen(ROUND8)
    paths = [ROUND8 / name for name in ("declaration.json", "declaration.sha256", "results.json",
                                       "results.sha256", "independent_audit.json", "independent_audit.sha256")]
    result, audit = round8.read_json(paths[2]), round8.read_json(paths[4])
    for document in (result, audit):
        round8.safety(document)
    for path, hashpath in ((paths[0], paths[1]), (paths[2], paths[3]), (paths[4], paths[5])):
        if digest(path) != hashpath.read_text().strip():
            raise ValueError("Round8 immutable artifact hash changed")
    if (result["stage"] != "exploratory_tail_tick_mechanism_results" or
            result["declaration_sha256"] != digest(paths[0]) or result["dates"] != list(DATES) or
            set(result["symbols"]) != set(SYMBOLS) or result["config"] != round8.CONFIG or
            audit["stage"] != "independent_fixed_tail_tick_pilot_audit" or audit["passed"] is not True or audit["errors"]):
        raise ValueError("Exact round8 pilot and passing audit are required")
    for path in (paths[0], paths[2]):
        round8.audited_input_hash(audit, path)
    verifier = round8.path_for(audit["verifier_file"])
    if digest(verifier) != audit["verifier_sha256"]:
        raise ValueError("The passing round8 audit verifier changed")
    selection, fingerprint = round6.frozen(ROUND6)
    if fingerprint != prior["lineage"]["round6_selection_sha256"]:
        raise ValueError("M1 lineage differs from the immutable round8 lineage")
    pins = {round8.relative(p): digest(p) for p in (*paths, verifier, ROUND6 / "selection.json",
                                                   ROUND6 / "selection.sha256", ROUND6 / "declaration.json",
                                                   ROUND6 / "declaration.sha256")}
    pins.update(prior["science_code_hashes"])
    pins.update(prior["lineage"]["inherited_files"])
    bounds, detectors, m1_sources = {}, {}, {}
    for symbol in SYMBOLS:
        saved = result["symbols"][symbol]
        if saved["observed_rows"] != prior["observed_rows"][symbol] or saved["row_cutoffs"] != prior["row_cutoffs"][symbol]:
            raise ValueError("Observed-row round8 partition changed")
        bounds[symbol] = {"0": START.isoformat(), "100": END.isoformat(), **{
            key: stamp(saved["segments"][segment]["first_observed_utc"]).isoformat()
            for key, segment in (("40", "wf1"), ("50", "wf2"), ("60", "wf3"), ("70", "final30"))}}
        ordered = [stamp(bounds[symbol][key]) for key in ("0", "40", "50", "60", "70", "100")]
        if any(a >= b for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Round8 observed-row chronology is invalid")
        detector = saved["detector"]
        if detector["adequate"] is True:
            inherited = FixedTailDetector(1 if symbol.startswith("BOOM") else -1, detector["median_abs_log_return"])
            if (detector["threshold"] != inherited.threshold or detector["threshold_multiplier"] != 10. or
                    detector["calibration_events"] < 100 or saved["no_detector_refit"] is not True):
                raise ValueError("First40 fixed detector changed")
        detectors[symbol] = detector
        m1_sources[symbol] = selection["sources"][symbol]["fresh"]
    return {"files": pins, "round8_declaration_sha256": digest(paths[0]),
            "round8_results_sha256": digest(paths[2]), "round8_audit_sha256": digest(paths[4]),
            "round8_verifier_sha256": digest(verifier), "round6_selection_sha256": fingerprint,
            "tick_sources": prior["lineage"]["sources"], "m1_sources": m1_sources,
            "boundaries": bounds, "detectors": detectors}


def guard(output, stage):
    output = Path(output)
    if stage not in ("declare", "develop", "evaluate"):
        raise ValueError("Unknown round9 stage")
    forbidden = ["results.json", "results.sha256", "metrics.csv"]
    if stage in ("declare", "develop"):
        forbidden += ["selection.json", "selection.sha256", "development_metrics.csv"]
    if stage == "declare":
        forbidden += ["declaration.json", "declaration.sha256"]
    if any((output / name).exists() for name in forbidden):
        raise ValueError("Refusing round9 frozen artifact or result overwrite")


def declare(output):
    round8.offline(); guard(output, "declare")
    science, lineage = hashes(), verified_round8_metadata()
    document = {"stage": "tick_age_payoff_premeasurement_declaration", "run_utc": datetime.now(timezone.utc),
                "safety": round8.offline(), "config": CONFIG, "dates": DATES, "symbols": SYMBOLS,
                "families": FAMILIES, "modes": MODES, "science_code_hashes": science, "lineage": lineage,
                "quotes_parsed": False, "new_features_labels_models_computed": False,
                "goal_achieved": False, "historical_strategy_candidate": False,
                "study_kind": "adaptive_known_history_discontinuous_tick_economic_feasibility_pilot",
                "quote_decode_policy": "whole_canonical_boundary_day_decode_then_immediate_prefix_clip_no_suffix_calculation",
                "user_target": {"profit_factor": 1.5, "completed_strategy_paths_per_symbol_model": 1000,
                                "active_heldout_days": 60, "pilot_can_meet_target": False},
                "model_scores": "continuous_uncalibrated_scores_not_probabilities"}
    if hashes() != science or verified_round8_metadata() != lineage:
        raise ValueError("Sources or science changed during declaration")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    save(output / "declaration.json", document)
    with (output / "declaration.sha256").open("x") as stream:
        stream.write(digest(output / "declaration.json") + "\n")
    print("TICK_AGE_PAYOFF_DECLARED", digest(output / "declaration.json"), "metadata only", flush=True)
    return canonical(document)


def frozen_declaration(output):
    round8.offline()
    output = Path(output)
    value = round8.read_json(output / "declaration.json")
    if digest(output / "declaration.json") != (output / "declaration.sha256").read_text().strip():
        raise ValueError("Frozen round9 declaration changed")
    round8.safety(value)
    if (value["stage"] != "tick_age_payoff_premeasurement_declaration" or value["config"] != CONFIG or
            value["dates"] != list(DATES) or value["symbols"] != list(SYMBOLS) or
            value["families"] != list(FAMILIES) or value["modes"] != list(MODES) or
            value["quotes_parsed"] is not False or value["new_features_labels_models_computed"] is not False or
            value["goal_achieved"] is not False or value["historical_strategy_candidate"] is not False or
            value["science_code_hashes"] != hashes() or value["lineage"] != verified_round8_metadata()):
        raise ValueError("Declared round9 science or immutable lineage changed")
    return value


def frozen_selection(output):
    declaration = frozen_declaration(output)
    output = Path(output)
    value = round8.read_json(output / "selection.json")
    fingerprint = digest(output / "selection.json")
    if fingerprint != (output / "selection.sha256").read_text().strip():
        raise ValueError("Frozen round9 selection changed")
    round8.safety(value)
    if (value["stage"] != "frozen_tick_age_payoff_development" or
            value["declaration_sha256"] != digest(output / "declaration.json") or
            value["science_code_hashes"] != declaration["science_code_hashes"] or
            value["lineage"] != declaration["lineage"] or value["config"] != CONFIG or
            value["final30_features_or_payoffs_evaluated"] is not False or
            value["goal_achieved"] is not False or value["historical_strategy_candidate"] is not False):
        raise ValueError("Selection chronology/configuration changed")
    if set(value["symbols"]) != set(SYMBOLS):
        raise ValueError("Exactly the two declared symbol records are required")
    verify_artifacts(value)
    for symbol in SYMBOLS:
        saved = value["symbols"][symbol]
        if saved["selected_candidate"] is not None:
            raise ValueError("This sparse diagnostic pilot cannot select a live candidate")
        if saved["status"] not in ("TESTED", "NOT TESTED"):
            raise ValueError("Unknown frozen symbol status")
        if saved["status"] == "NOT TESTED":
            if saved["final_models"] or saved["candidates"]:
                raise ValueError("Untested detector cannot have fitted models")
            continue
        expected = {mode + "_" + family for mode in MODES for family in FAMILIES}
        if set(saved["final_models"]) != expected or set(saved["candidates"]) != expected:
            raise ValueError("Exactly both families and both modes must be frozen")
        for mode in MODES:
            a, b = (saved["final_models"][mode + "_" + family] for family in FAMILIES)
            if any(a[key] != b[key] for key in ("status", "training_completed_labels", "common_timestamp_target_sha256",
                                               "training_clock_issuance_sha256", "training_start", "training_end")):
                raise ValueError("44/46 final fits must share the exact training clock and targets")
            if a["training_start"] != START.isoformat() or a["training_end"] != value["lineage"]["boundaries"][symbol]["70"]:
                raise ValueError("Final models must train only the declared first70 prefix")
        for key, model in saved["final_models"].items():
            if model["family"] != key.split("_")[1] or saved["candidates"][key]["development_eligible"] is not False:
                raise ValueError("Pilot model identity or impossible development eligibility changed")
            validate_model(model)
    return value, fingerprint



def validate_model(model):
    family, status = model["family"], model["status"]
    names = list(FEATURE_NAMES) + (["tail_age_log1p", "prior_tail_mark"] if family == "RIDGE46" else [])
    if family not in FAMILIES or model["feature_names"] != names or status not in ("TESTED", "NOT TESTED"):
        raise ValueError("Frozen model feature family/schema/status changed")
    n = model["training_completed_labels"]
    if status == "NOT TESTED":
        if n >= MIN_TRAIN or model["estimator"] is not None or model["serialized_estimator_sha256"] is not None:
            raise ValueError("Insufficient models must stay NOT TESTED without an estimator")
        return
    estimator = model["estimator"]
    if (n < MIN_TRAIN or estimator["fit_rows"] != n or estimator["feature_names"] != names or
            estimator["penalty"] != .1 or estimator["quantile"] != .75 or estimator["threshold"] < 0 or
            estimator["score_kind"] != "continuous_uncalibrated_score" or
            json_hash(estimator) != model["serialized_estimator_sha256"]):
        raise ValueError("Saved model/scaler/cutoff or training floor changed")


def verify_artifacts(value, seen=None):
    """Frozen local evidence files must retain exact bytes before label release."""
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if value.get("local_ignored_artifact") is True:
            identity = (value["path"], value["sha256"])
            if identity not in seen:
                if digest(round8.path_for(value["path"])) != value["sha256"]:
                    raise ValueError("Frozen local inputs/signals/labels/ledger bytes changed")
                seen.add(identity)
        if "family" in value and "estimator" in value and "status" in value:
            validate_model(value)
        for child in value.values():
            verify_artifacts(child, seen)
    elif isinstance(value, list):
        for child in value:
            verify_artifacts(child, seen)

def observed_days(start, end):
    start, end = stamp(start), stamp(end)
    if start >= end:
        raise ValueError("start must precede end")
    return [date for date in DATES if max(start, pd.Timestamp(date, tz="UTC")) <
            min(end, pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=1))]


def partition_days(start, end):
    for date in observed_days(start, end):
        opening = pd.Timestamp(date, tz="UTC")
        yield date, max(stamp(start), opening), min(stamp(end), opening + pd.Timedelta(days=1))


def prepare_prefix(declaration, symbol, end):
    """Withhold suffix calculations, retaining exact gap/partial rows before rolling."""
    if symbol not in SYMBOLS:
        raise ValueError("Only the declared600 symbols are allowed")
    end = stamp(end)
    value = declaration["lineage"]["m1_sources"][symbol]
    m1, audit = load_m1(round8.path_for(value["path"]))
    check_prepared(m1, audit, value)
    # A split can lie inside a minute: only already CLOSED source candles enter.
    m1 = m1.loc[m1.index + pd.Timedelta(minutes=1) <= end].copy()
    parts = []
    for date in DATES:
        if pd.Timestamp(date, tz="UTC") >= end:
            continue
        day = round8.load_tick_day(declaration["lineage"]["tick_sources"][symbol][date])
        parts.append(day.loc[day.index < end])
    if not parts:
        raise ValueError("No observed prefix quote days")
    ticks = pd.concat(parts).sort_index()
    saved = declaration["lineage"]["detectors"][symbol]
    if saved["adequate"] is not True:
        raise ValueError("Inadequate fixed round8 detector: NOT TESTED")
    detector = FixedTailDetector(1 if symbol.startswith("BOOM") else -1, saved["median_abs_log_return"])
    if detector.threshold != saved["threshold"]:
        raise ValueError("Inherited detector threshold changed")
    description = describe_tick_tail(ticks, detector, 600)
    m5, names = causal_multiframe_inputs(m1, "boom" if symbol.startswith("BOOM") else "crash")
    table, names44, names46 = join_tick_tail_inputs(m5, names, description)
    table["m5_open_time"] = m5.index
    table["original_feature_valid"] = m5.feature_valid.to_numpy(bool)
    table["original44_all_finite"] = np.isfinite(m5[names44].to_numpy(float)).all(axis=1)
    table["prior_age_seconds"] = description.pre_event_age_seconds.reindex(pd.DatetimeIndex(table.signal_time)).to_numpy()
    mask = (table.feature_valid & (table.signal_time >= START) & (table.signal_time < end) &
            (table.signal_time.dt.minute % 5 == 0) & (table.signal_time.dt.second == 0))
    rows = table.loc[mask].copy()
    rows.index = pd.DatetimeIndex(rows.signal_time).as_unit("ns")
    rows.index.name = "signal_time"
    if rows.index.has_duplicates or not np.isfinite(rows[names46].to_numpy(float)).all():
        raise ValueError("Common eligible feature rows must be finite and unique")
    audit.update(feature_end_exclusive=end.isoformat(), m1_closed_rows_used=len(m1),
                 last_m1_close=(m1.index[-1] + pd.Timedelta(minutes=1)).isoformat() if len(m1) else None,
                 tick_prefix_rows=len(ticks), tick_last_used=ticks.index[-1].isoformat(),
                 canonical_boundary_day_quotes_decoded=True, suffix_calculations=False,
                 eligible_common_clock_rows=len(rows), original_m5_rows=len(table),
                 clock="every closed UTC M5", detector_refit=False)
    inputs = table.loc[(table.signal_time >= START) & (table.signal_time < end)].copy()
    ordered = ["m5_open_time", "signal_time", *names46, "atr", "close", "prior_age_seconds",
               "original_feature_valid", "original44_all_finite", "multiframe_feature_valid",
               "tail_feature_valid", "common_available", "feature_valid"]
    ordered += [name for name in table if name.endswith("_closed_at") or name.endswith("_row_valid")]
    inputs = inputs[list(dict.fromkeys(ordered))]
    return {"ticks": ticks, "rows": rows, "inputs": inputs, "names44": names44, "names46": names46, "audit": audit}


def window_rows(rows, start, end):
    return rows.loc[(rows.index >= stamp(start)) & (rows.index < stamp(end))]


def issue(rows, symbol, mode, model=None):
    if symbol not in SYMBOLS or mode not in MODES:
        raise ValueError("Unknown declared symbol/direction")
    chosen, scores = rows, None
    if model is not None:
        if model["status"] == "NOT TESTED":
            return []
        if model["status"] != "TESTED" or model["family"] not in FAMILIES:
            raise ValueError("Unknown frozen estimator status/family")
        names = model["feature_names"]
        if len(names) != (44 if model["family"] == "RIDGE44" else 46):
            raise ValueError("Frozen feature schema length changed")
        scores = predict_ridge(rows[names], model["estimator"], feature_names=names) if len(rows) else np.array([])
        mask = (scores >= model["estimator"]["threshold"]) & (scores > 0)
        chosen, scores = rows.loc[mask], scores[mask]
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    variant = (model["family"] if model else "CLOCK") + "_" + mode
    result = [{"signal_time": t, "atr": float(r.atr), "side": side, "variant": variant,
               "signal_close": float(r.close), "prior_age_seconds": float(r.prior_age_seconds)}
              for t, r in chosen.iterrows()]
    if scores is not None:
        for record, score in zip(result, scores, strict=True):
            record["score"] = float(score)
    return result


def planned_eligible(signals, start, end):
    """Return planned eligibility BEFORE viewing any realized outcome."""
    result = []
    for _, left, right in partition_days(start, end):
        for record in signals:
            t = stamp(record["signal_time"])
            deadline = t + pd.Timedelta(minutes=EXIT_CONFIG.entry_delay_minutes + EXIT_CONFIG.max_hold_minutes)
            if left <= t < right and t + pd.Timedelta(minutes=PURGE) <= right and deadline + pd.Timedelta(seconds=1) <= right:
                result.append(record)
    return result


def replay_partition(ticks, signals, start, end, *, overlapping=False):
    parts, days = [], {}
    totals = dict.fromkeys(COUNTERS, 0)
    for date, left, right in partition_days(start, end):
        source = ticks.loc[(ticks.index >= left) & (ticks.index < right)]
        issued = [r for r in signals if left <= stamp(r["signal_time"]) < right]
        replay = independent_tick_labels if overlapping else replay_ticks
        ledger, audit = replay(source, issued, EXIT_CONFIG, left, right, purge_minutes=PURGE)
        days[date] = {"start": left.isoformat(), "end": right.isoformat(), **audit}
        for name in COUNTERS:
            totals[name] += audit[name]
        if len(ledger):
            parts.append(ledger)
    if parts:
        ledger = pd.concat(parts, ignore_index=True)
    else:
        ledger, _ = replay_ticks(ticks.iloc[:0], [], EXIT_CONFIG, stamp(start), stamp(end), PURGE)
    totals.update(days=days, safety=round8.offline(), overlapping_labels=overlapping,
                  statistically_independent_labels=False if overlapping else None,
                  strategy_returns=not overlapping, counts_toward_profit_sample_target=not overlapping,
                  effective_day_end_and_planned_purge=True)
    return ledger, totals


def training_pair(rows, labels, names44, names46, start, end):
    """Fit44/46 on exactly matching known planned-purged rows; no fallback floor."""
    if list(names44) != list(FEATURE_NAMES) or list(names46) != [*names44, "tail_age_log1p", "prior_tail_mark"]:
        raise ValueError("Exact declared44/46 schema required")
    start, end = stamp(start), stamp(end)
    times = pd.to_datetime(labels.signal_time, utc=True)
    mask = ((times >= start) & (times < end) & (times + pd.Timedelta(minutes=PURGE) <= end) &
            times.dt.strftime("%Y-%m-%d").isin(observed_days(start, end)))
    # Replay also applied each effective sampled-day end; retained labels must
    # still satisfy that planned guard, even when an early stop was observed.
    for _, left, right in partition_days(start, end):
        belongs = (times >= left) & (times < right)
        mask &= ~belongs | (times + pd.Timedelta(minutes=PURGE) <= right)
    past = labels.loc[mask].copy()
    finite = np.isfinite(past.net_R.to_numpy(float))
    known = past.loc[~past.censored & finite].set_index("signal_time").sort_index()
    known.index = pd.DatetimeIndex(known.index).as_unit("ns")
    if known.index.has_duplicates or rows.index.has_duplicates:
        raise ValueError("Training labels and common clocks must be unique")
    if not known.index.isin(rows.index).all():
        raise ValueError("Every completed label must align with the common exact clock")
    X46 = rows.loc[known.index, names46]
    if not np.isfinite(X46.to_numpy(float)).all():
        raise ValueError("Training feature inputs must all be finite")
    y = known.net_R.to_numpy(float)
    target_hash = hashlib.sha256(np.ascontiguousarray(known.index.asi8).tobytes() +
                                 np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest()
    common = {"training_start": start.isoformat(), "training_end": end.isoformat(),
              "training_completed_labels": len(known), "training_censored_labels": int(past.censored.sum()),
              "training_invalid_labels": int((~past.censored & ~finite).sum()),
              "common_timestamp_target_sha256": target_hash, "training_labels_may_overlap": True,
              "statistically_independent_labels": False, "counts_toward_profit_sample_target": False,
              "label_purge_minutes": PURGE,
              "training_latest_issue": known.index[-1].isoformat() if len(known) else None}
    result = {}
    for family, names in zip(FAMILIES, (names44, names46), strict=True):
        X = X46[names]
        matrix_hash = hashlib.sha256(json.dumps(names).encode() + np.ascontiguousarray(X.to_numpy(float)).tobytes() +
                                     np.ascontiguousarray(y).tobytes() + np.ascontiguousarray(known.index.asi8).tobytes()).hexdigest()
        model = {**common, "family": family, "feature_names": list(names), "matrix_and_target_sha256": matrix_hash}
        if len(known) < MIN_TRAIN:
            model.update(status="NOT TESTED", reason="fewer_than1000_completed_finite_individually_replayed_training_labels",
                         estimator=None, serialized_estimator_sha256=None)
        else:
            estimator = fit_ridge(X, y, names, penalty=.1, quantile=.75)
            model.update(status="TESTED", estimator=estimator, serialized_estimator_sha256=json_hash(estimator))
        result[family] = model
    return result


def weights(start, end):
    days = observed_days(start, end)
    if not days:
        raise ValueError("A sampled observed-day partition is required")
    rng = np.random.default_rng(SEED)
    return days, rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=BOOTSTRAP)


def day_sums(ledger, days):
    """Pool finite known paths within each observed day; unknowns stay unknown."""
    result = np.zeros((len(days), 4), dtype=float)  # return sum,count,gain,loss
    if not len(ledger):
        return result
    stamps = pd.to_datetime(ledger.signal_time, utc=True)
    labels = stamps.dt.strftime("%Y-%m-%d")
    if not labels.isin(days).all():
        raise ValueError("Ledger includes a day outside the inference partition")
    finite = np.isfinite(ledger.net_R.to_numpy(float))
    known = ~ledger.censored.to_numpy(bool) & finite
    for i, day in enumerate(days):
        values = ledger.net_R.to_numpy(float)[known & (labels.to_numpy() == day)]
        result[i] = values.sum(), len(values), values[values > 0].sum(), -values[values < 0].sum()
    return result


def draw_means(sums, draws):
    pooled = draws @ sums
    means = np.divide(pooled[:, 0], pooled[:, 1], out=np.full(len(draws), np.nan), where=pooled[:, 1] > 0)
    pf = np.divide(pooled[:, 2], pooled[:, 3], out=np.full(len(draws), np.nan), where=pooled[:, 3] > 0)
    return means, pf


def interval(values):
    finite = np.isfinite(values)
    return {"ci95": np.quantile(values[finite], [.025, .975]).tolist() if finite.any() else None,
            "undefined_draws": int((~finite).sum()), "all_draws_defined": bool(finite.all()),
            "conditional_on_defined_draws": bool((~finite).any())}


def metric(ledger, audit, days, draws):
    sums = day_sums(ledger, days)
    n = int(sums[:, 1].sum()); total = float(sums[:, 0].sum()); loss = float(sums[:, 3].sum())
    mean, pf = draw_means(sums, draws)
    mean_inference, pf_inference = interval(mean), interval(pf)
    center = total / n if n else None
    se = (float(np.sqrt(len(days) / (len(days) - 1) * np.square(sums[:, 0] - center * sums[:, 1]).sum()) / n)
          if n and len(days) > 1 else None)
    invalid = int((~ledger.censored & ~np.isfinite(ledger.net_R)).sum()) if len(ledger) else 0
    return {"completed": n, "censored": int(audit.get("censored", 0)),
            "missing_entry": int(audit.get("missing_entry", 0)), "invalid_uncensored": invalid,
            "issued": int(audit.get("issued", 0)), "overlap_skipped": int(audit.get("overlap_skipped", 0)),
            "mean_net_R": center, "profit_factor": float(sums[:, 2].sum() / loss) if loss else None,
            "sum_net_R": total, "sum_gains_R": float(sums[:, 2].sum()), "sum_losses_R": loss,
            "active_days": int((sums[:, 1] > 0).sum()), "observed_days": days, "observed_day_clusters": len(days),
            "day_mean_ci95": mean_inference["ci95"], "mean_inference": mean_inference, "pf_inference": pf_inference,
            "selection_score": center - 1.96 * se if center is not None and se is not None and mean_inference["all_draws_defined"] else None,
            "day_ratio_cluster_se": se, "bootstrap_repeats": len(draws), "bootstrap_seed": SEED,
            "day_sums_return_count_gain_loss": sums.tolist(), "interval_scope": "iid_observed_source_days_inside_partition_only",
            "weeks_or_unsampled_calendar_days_padded": False,
            "bounded_inference_allowed": mean_inference["all_draws_defined"] and pf_inference["all_draws_defined"]}


def paired_difference(left, right, days, draws):
    a, _ = draw_means(day_sums(left, days), draws)
    b, _ = draw_means(day_sums(right, days), draws)
    diff = a - b
    result = interval(diff)
    sa, sb = day_sums(left, days).sum(axis=0), day_sums(right, days).sum(axis=0)
    result.update(mean_difference_R=(float(sa[0] / sa[1] - sb[0] / sb[1]) if sa[1] and sb[1] else None),
                  observed_days=days, observed_day_clusters=len(days), bootstrap_repeats=len(draws),
                  interpretation="pooled_path_mean_difference_paired_observed_day_draws", discovery_claim=False)
    return result


def age_diagnostics(labels, planned, start, end, days, draws):
    times = pd.DatetimeIndex(pd.to_datetime(labels.signal_time, utc=True))
    if times.has_duplicates:
        raise ValueError("Individually replayed diagnostic labels must have unique clocks")
    reports = {}
    for band in ("lt600", "ge600"):
        members = [r for r in planned if (r["prior_age_seconds"] < 600) == (band == "lt600")]
        chosen = pd.DatetimeIndex([stamp(r["signal_time"]) for r in members])
        ledger = labels.loc[times.isin(chosen)].copy()
        missing = int((~chosen.isin(times)).sum())
        censored = int(ledger.censored.sum())
        invalid = int((~ledger.censored & ~np.isfinite(ledger.net_R)).sum())
        summary = metric(ledger, {"issued": len(members), "censored": censored, "missing_entry": missing}, days, draws)
        for key in ("profit_factor", "pf_inference", "sum_gains_R", "sum_losses_R", "bounded_inference_allowed"):
            summary.pop(key)
        summary["profit_factor_evidence"] = "NOT TESTED_overlapping_targets_are_not_a_strategy_portfolio"
        summary["day_sums_return_count"] = [row[:2] for row in summary.pop("day_sums_return_count_gain_loss")]
        adequate = summary["completed"] >= 100 and summary["mean_inference"]["all_draws_defined"] and not (missing or censored or invalid)
        upper = summary["day_mean_ci95"][1] if summary["day_mean_ci95"] else None
        status = ("REJECT_POSITIVE_FIXED_OVERLAPPING_LABEL_MEAN" if adequate and upper <= 0 else
                  "NOT_REJECTED_POSITIVE_MEAN" if adequate else "INSUFFICIENT_EVIDENCE")
        reports[band] = {"planned_eligible_signals": len(members), "missing_entry_labels": missing,
                         "censored_labels": censored, "invalid_labels": invalid, "known_subset_only": bool(missing or censored or invalid),
                         "status": status, "whole_eligible_policy_rejection_allowed": adequate,
                         "metrics": summary, "labels_may_overlap": True, "statistically_independent_labels": False,
                         "strategy_pf_evidence": False, "discovery_claim": False,
                         "scope": "fixed_overlapping_label_mean_not_all46_interactions_or_occupancy_policies"}
    return reports


def dump_frame(output, category, name, frame):
    folder = Path(output) / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".csv")
    with path.open("x") as stream:
        frame.to_csv(stream, index=False)
    return {"path": round8.relative(path), "sha256": digest(path), "rows": len(frame),
            "local_ignored_artifact": True}



def dump_signals(output, name, signals):
    columns = ["signal_time", "atr", "side", "variant", "signal_close", "prior_age_seconds", "score"]
    frame = pd.DataFrame(signals).reindex(columns=columns)
    result = dump_frame(output, "signals", name, frame)
    result["canonical_issuance_sha256"] = issuance_hash(signals)
    return result


def persist_context(context, output, symbol, prefix):
    artifact = dump_frame(output, "inputs", prefix + "_" + symbol, context["inputs"])
    context["audit"]["inputs"] = artifact
    return artifact

def issuance_hash(signals):
    return json_hash(signals)


def evaluate_partition(context, models, symbol, mode, start, end, output, prefix):
    rows = window_rows(context["rows"], start, end)
    days, draws = weights(start, end)
    ledgers, reports = {}, {}
    for family in ("CLOCK", *FAMILIES):
        model = None if family == "CLOCK" else models[family]
        signals = issue(rows, symbol, mode, model)
        ledger, audit = replay_partition(context["ticks"], signals, start, end)
        ledgers[family] = ledger
        reports[family] = {"status": "TESTED" if model is None else model["status"],
                           "metrics": metric(ledger, audit, days, draws), "audit": audit,
                           "issuance_sha256": issuance_hash(signals),
                           "signals": dump_signals(output, f"{prefix}_{symbol}_{mode}_{family}", signals),
                           "common_available_rows": len(rows),
                           "ledger": dump_frame(output, "ledgers", f"{prefix}_{symbol}_{mode}_{family}", ledger),
                           "strategy_returns": True, "quoteproxy_not_actual_fills": True}
    comparisons = {"RIDGE46_minus_RIDGE44": paired_difference(ledgers["RIDGE46"], ledgers["RIDGE44"], days, draws)}
    for family in FAMILIES:
        comparisons[family + "_minus_CLOCK"] = paired_difference(ledgers[family], ledgers["CLOCK"], days, draws)
    return reports, comparisons, ledgers


def develop(output):
    round8.offline(); guard(output, "develop")
    declaration = frozen_declaration(output)
    result = {"stage": "frozen_tick_age_payoff_development", "run_utc": datetime.now(timezone.utc),
              "declaration_sha256": digest(Path(output) / "declaration.json"), "safety": round8.offline(),
              "science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"],
              "config": CONFIG, "final30_features_or_payoffs_evaluated": False,
              "goal_achieved": False, "historical_strategy_candidate": False, "symbols": {}}
    for symbol in SYMBOLS:
        bounds = {key: stamp(value) for key, value in declaration["lineage"]["boundaries"][symbol].items()}
        if declaration["lineage"]["detectors"][symbol]["adequate"] is not True:
            result["symbols"][symbol] = {"status": "NOT TESTED", "reason": "inadequate_fixed_round8_detector",
                                         "final_models": {}, "candidates": {}, "selected_candidate": None}
            continue
        cache = {}
        def context_at(key):
            if key not in cache:
                cache[key] = prepare_prefix(declaration, symbol, bounds[key])
                persist_context(cache[key], output, symbol, "prefix" + key)
            return cache[key]
        candidates = {mode + "_" + family: {"mode": mode, "family": family, "walk_forward": []}
                      for mode in MODES for family in FAMILIES}
        pooled = {mode: {family: [] for family in ("CLOCK", *FAMILIES)} for mode in MODES}
        pooled_audits = {mode: {family: dict.fromkeys(COUNTERS, 0) for family in ("CLOCK", *FAMILIES)} for mode in MODES}
        for ordinal, (trainkey, testkey) in enumerate((("40", "50"), ("50", "60"), ("60", "70")), 1):
            train, test = context_at(trainkey), context_at(testkey)
            for mode in MODES:
                signals = issue(window_rows(train["rows"], START, bounds[trainkey]), symbol, mode)
                labels, label_audit = replay_partition(train["ticks"], signals, START, bounds[trainkey], overlapping=True)
                artifact = dump_frame(output, "training_labels", f"train{trainkey}_{symbol}_{mode}", labels)
                models = training_pair(train["rows"], labels, train["names44"], train["names46"], START, bounds[trainkey])
                train_signals_artifact = dump_signals(output, f"train{trainkey}_{symbol}_{mode}", signals)
                for model in models.values():
                    model["training_clock_issuance_sha256"] = issuance_hash(signals)
                    model["training_signals"] = train_signals_artifact
                    model["training_inputs"] = train["audit"]["inputs"]
                reports, comparisons, ledgers = evaluate_partition(test, models, symbol, mode, bounds[trainkey], bounds[testkey], output, f"wf{ordinal}")
                for family in ("CLOCK", *FAMILIES):
                    pooled[mode][family].append(ledgers[family])
                    for counter in COUNTERS:
                        pooled_audits[mode][family][counter] += reports[family]["audit"][counter]
                for family in FAMILIES:
                    candidates[mode + "_" + family]["walk_forward"].append({"training": "train" + trainkey,
                        "test": "wf" + str(ordinal), "model": models[family], "training_labels": artifact,
                        "training_label_audit": label_audit, **reports[family], "comparisons": comparisons})
        final = context_at("70")
        days, draws = weights(bounds["40"], bounds["70"])
        final_models, diagnostics, validation = {}, {}, {}
        for mode in MODES:
            all_signals = issue(window_rows(final["rows"], START, bounds["70"]), symbol, mode)
            labels, label_audit = replay_partition(final["ticks"], all_signals, START, bounds["70"], overlapping=True)
            artifact = dump_frame(output, "training_labels", f"train70_{symbol}_{mode}", labels)
            pair = training_pair(final["rows"], labels, final["names44"], final["names46"], START, bounds["70"])
            train_signals_artifact = dump_signals(output, f"train70_{symbol}_{mode}", all_signals)
            for model in pair.values():
                model["training_clock_issuance_sha256"] = issuance_hash(all_signals)
                model["training_signals"] = train_signals_artifact
                model["training_inputs"] = final["audit"]["inputs"]
            final_models.update({mode + "_" + family: {**pair[family], "training_labels": artifact,
                                                      "training_label_audit": label_audit} for family in FAMILIES})
            pooled_ledgers = {family: pd.concat(frames, ignore_index=True) for family, frames in pooled[mode].items()}
            validation[mode] = {family: {"metrics": metric(ledger, pooled_audits[mode][family], days, draws),
                                       "audit": pooled_audits[mode][family],
                                       "ledger": dump_frame(output, "ledgers", f"validation_{symbol}_{mode}_{family}", ledger)}
                                for family, ledger in pooled_ledgers.items()}
            validation[mode]["comparisons"] = {"RIDGE46_minus_RIDGE44": paired_difference(pooled_ledgers["RIDGE46"], pooled_ledgers["RIDGE44"], days, draws),
                **{family + "_minus_CLOCK": paired_difference(pooled_ledgers[family], pooled_ledgers["CLOCK"], days, draws) for family in FAMILIES}}
            for family in FAMILIES:
                candidate = candidates[mode + "_" + family]
                candidate["validation"] = validation[mode][family]["metrics"]
                candidate["final_model_status"] = pair[family]["status"]
                fold_statuses = [fold["status"] for fold in candidate["walk_forward"]]
                validation[mode][family]["fold_fit_statuses"] = fold_statuses
                validation[mode][family]["status"] = ("TESTED" if all(s == "TESTED" for s in fold_statuses) else
                    "PARTIALLY TESTED" if any(s == "TESTED" for s in fold_statuses) else "NOT TESTED")
                candidate["development_eligible"] = bool(development_eligible(candidate) and
                    pair[family]["status"] == "TESTED" and len(days) >= 30)
                candidate["eligibility_limitation"] = "only" + str(len(days)) + "_observed_validation_days_less_than30"
            for comparison, inference in validation[mode]["comparisons"].items():
                involved = FAMILIES if comparison == "RIDGE46_minus_RIDGE44" else (comparison.split("_minus_")[0],)
                if not all(validation[mode][family]["status"] == "TESTED" for family in involved):
                    validation[mode]["comparisons"][comparison] = {"status": "NOT TESTED",
                        "reason": "some_validation_fits_NOT_TESTED_unmatched_fit_availability",
                        "partial_descriptive_only": inference}
                else:
                    inference["status"] = "TESTED"
            diag_signals = issue(window_rows(final["rows"], bounds["40"], bounds["70"]), symbol, mode)
            diag_labels, diag_audit = replay_partition(final["ticks"], diag_signals, bounds["40"], bounds["70"], overlapping=True)
            diag_artifact = dump_frame(output, "diagnostic_labels", f"validation_{symbol}_{mode}", diag_labels)
            diagnostics[mode] = {"audit": diag_audit, "labels": diag_artifact,
                "signals": dump_signals(output, f"validation_diagnostic_{symbol}_{mode}", diag_signals),
                "bands": age_diagnostics(diag_labels, planned_eligible(diag_signals, bounds["40"], bounds["70"]),
                                         bounds["40"], bounds["70"], days, draws)}
        result["symbols"][symbol] = {"status": "TESTED", "audits": {key: item["audit"] for key, item in cache.items()},
            "candidates": candidates, "final_models": final_models, "validation": validation,
            "age_diagnostics": diagnostics, "selected_candidate": None, "eligible_candidates": 0}
        print(symbol, "development diagnostics complete; no candidate can meet30 active days", flush=True)
    if declaration != frozen_declaration(output):
        raise ValueError("Inputs or declaration changed during development")
    verify_artifacts(result)
    save(Path(output) / "selection.json", result)
    with (Path(output) / "selection.sha256").open("x") as stream:
        stream.write(digest(Path(output) / "selection.json") + "\n")
    write_metrics(Path(output) / "development_metrics.csv", result, development=True)
    print("TICK_AGE_PAYOFF_SELECTION_FROZEN", digest(Path(output) / "selection.json"), "final30 NOT EVALUATED", flush=True)
    return canonical(result)


def evaluate(output):
    round8.offline(); guard(output, "evaluate")
    selection, fingerprint = frozen_selection(output)
    result = {"stage": "fixed_tick_age_payoff_final30_diagnostics", "run_utc": datetime.now(timezone.utc),
              "selection_sha256": fingerprint, "declaration_sha256": selection["declaration_sha256"],
              "science_code_hashes": selection["science_code_hashes"], "lineage": selection["lineage"],
              "safety": round8.offline(), "config": CONFIG, "symbols": {},
              "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False,
              "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED",
              "price_oos": False, "new_label_release_after_frozen_selection": True,
              "eight_fixed_models_descriptive_not_profit_discovery": True}
    for symbol in SYMBOLS:
        saved = selection["symbols"][symbol]
        if saved["status"] == "NOT TESTED":
            result["symbols"][symbol] = {"status": "NOT TESTED", "reason": saved["reason"], "models": {}}
            continue
        bounds = {key: stamp(value) for key, value in selection["lineage"]["boundaries"][symbol].items()}
        context = prepare_prefix(selection, symbol, bounds["100"])
        persist_context(context, output, symbol, "prefix100")
        days, draws = weights(bounds["70"], bounds["100"])
        models, diagnostics = {}, {}
        for mode in MODES:
            pair = {family: saved["final_models"][mode + "_" + family] for family in FAMILIES}
            reports, comparisons, _ = evaluate_partition(context, pair, symbol, mode, bounds["70"], bounds["100"], output, "final30")
            for family in FAMILIES:
                reports[family].update(development_eligible=saved["candidates"][mode + "_" + family]["development_eligible"],
                                      historical_gate=False, live_candidate=False, model_sha256=pair[family]["serialized_estimator_sha256"])
            models[mode] = {"reports": reports, "comparisons": comparisons}
            signals = issue(window_rows(context["rows"], bounds["70"], bounds["100"]), symbol, mode)
            labels, audit = replay_partition(context["ticks"], signals, bounds["70"], bounds["100"], overlapping=True)
            diagnostics[mode] = {"audit": audit, "labels": dump_frame(output, "diagnostic_labels", f"final30_{symbol}_{mode}", labels),
                "signals": dump_signals(output, f"final30_diagnostic_{symbol}_{mode}", signals),
                "bands": age_diagnostics(labels, planned_eligible(signals, bounds["70"], bounds["100"]),
                                         bounds["70"], bounds["100"], days, draws)}
        result["symbols"][symbol] = {"status": "TESTED", "audit": context["audit"], "models": models,
                                      "age_diagnostics": diagnostics, "selected_candidate": None}
    if frozen_selection(output)[1] != fingerprint:
        raise ValueError("Frozen selection/source bytes changed during evaluation")
    save(Path(output) / "results.json", result)
    with (Path(output) / "results.sha256").open("x") as stream:
        stream.write(digest(Path(output) / "results.json") + "\n")
    write_metrics(Path(output) / "metrics.csv", result)
    print("TICK_AGE_PAYOFF_DIAGNOSTICS_COMPLETE; actual profit/prospective evidence NOT TESTED", flush=True)
    return canonical(result)


def write_metrics(path, result, *, development=False):
    rows = []
    for symbol, value in result["symbols"].items():
        if value["status"] != "TESTED":
            rows.append({"symbol": symbol, "status": "NOT TESTED"}); continue
        for mode in MODES:
            reports = value["validation"][mode] if development else value["models"][mode]["reports"]
            for family in ("CLOCK", *FAMILIES):
                m = reports[family]["metrics"]
                rows.append({"symbol": symbol, "mode": mode, "family": family, "status": reports[family].get("status", "TESTED"),
                             **{key: m[key] for key in ("completed", "censored", "missing_entry", "profit_factor", "mean_net_R",
                                                        "active_days", "observed_day_clusters")},
                             "historical_gate": False, "live_candidate": False})
    with Path(path).open("x") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("declare", "develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"declare": declare, "develop": develop, "evaluate": evaluate}[args.stage](args.output)


if __name__ == "__main__":
    main()
