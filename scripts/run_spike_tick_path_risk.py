#!/usr/bin/env python3
"""Declared44/45 native-adverse path-risk pilot on known prices AND payoffs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.learned_signal import fit_ridge, predict_ridge
from app.research.multiframe_signal import causal_multiframe_inputs, FEATURE_NAMES as NAMES44
from app.research.spike_hunter import load_m1
from app.research.tick_path_risk import (describe_tick_path_risk, join_tick_path_risk_inputs,
    FEATURE_NAMES as NAMES45, FEATURE_FORMULAS, PATH_FEATURE_NAME)
from scripts import run_spike_tick_age_payoff as round9
from scripts.run_spike_native300_study import check_prepared

# These immutable primitives contain no feature preparation or model selection.
digest, canonical, json_hash, save, stamp = round9.digest, round9.canonical, round9.json_hash, round9.save, round9.stamp
window_rows, weights, metric = round9.window_rows, round9.weights, round9.metric
paired_difference, replay_partition = round9.paired_difference, round9.replay_partition
planned_eligible, dump_frame, issuance_hash = round9.planned_eligible, round9.dump_frame, round9.issuance_hash
persist_context, development_eligible = round9.persist_context, round9.development_eligible
read_json, path_for, offline = round9.round8.read_json, round9.round8.path_for, round9.round8.offline
load_tick_day, frozen_round9_selection = round9.round8.load_tick_day, round9.frozen_selection
SYMBOLS, MODES, DATES = round9.SYMBOLS, round9.MODES, round9.DATES
START, END, PURGE, MIN_TRAIN = round9.START, round9.END, round9.PURGE, round9.MIN_TRAIN
FAMILIES = ("RIDGE44", "RIDGE45")
OUTPUT = ROOT / "docs/spike_tick_path_risk_20261006"
ROUND9 = ROOT / "docs/spike_tick_age_payoff_20261005"
NEW_SOURCES = ("docs/SPIKE_TICK_PATH_RISK_PROTOCOL.md", "backend/app/research/tick_path_risk.py",
               "backend/tests/test_tick_path_risk.py", "scripts/run_spike_tick_path_risk.py",
               "backend/tests/test_spike_tick_path_risk.py")
SOURCES = (*round9.SOURCES, *NEW_SOURCES)
CONFIG = {"exit": round9.CONFIG["exit"], "purge_minutes": PURGE, "cadence_minutes": 5,
          "ridge_penalty": .1, "training_quantile": .75, "minimum_training_labels": MIN_TRAIN,
          "bootstrap_repeats": round9.BOOTSTRAP, "bootstrap_seed": round9.SEED,
          "risk_increments": 600, "required_prior_quotes": 601, "scale_refitted": False,
          "native_direction_independent_of_strategy_mode": True, "training_labels_may_overlap": True,
          "unknown_payoffs_are_zero": False, "undefined_draws_block_bounded_inference": True}
HISTORY = {"known_prices": True, "known_payoffs": True, "price_oos": False,
           "new_label_holdout": False, "prospective": False,
           "adaptive_history": "round9_final_payoffs_already_exposed_before_round10"}


def hashes():
    return {name: digest(path_for(name)) for name in SOURCES}



def audited_relative_path(label, audited_prefix=None):
    """Normalize only paths under the uniquely hash-anchored old repository."""
    if not isinstance(label, str) or not label or "\\" in label:
        raise ValueError("Audited path labels must be canonical POSIX strings")
    lexical = PurePosixPath(label)
    if ".." in lexical.parts or str(lexical) != label:
        raise ValueError("Audited paths must not contain traversal or noncanonical components")
    if lexical.is_absolute():
        if audited_prefix is None:
            if not Path(label).is_relative_to(ROOT):
                raise ValueError("Audited absolute input needs a uniquely anchored repository prefix")
            label = str(Path(label).relative_to(ROOT))
        elif label.startswith(audited_prefix + "/"):
            label = label[len(audited_prefix) + 1:]
        else:
            raise ValueError("Audited absolute input is outside its anchored repository")
    path_for(label)
    return label


def audited_repository_prefix(audit):
    label = str((ROUND9 / "declaration.json").resolve().relative_to(ROOT.resolve()))
    fingerprint = digest(ROUND9 / "declaration.json")
    matching = [key for key, value in audit["input_sha256"].items()
                if value == fingerprint and (key == label or key.endswith("/" + label))]
    if len(matching) != 1:
        raise ValueError("Audited repository prefix needs one exact declaration/hash anchor")
    key = matching[0]
    if key == label:
        return None
    if not PurePosixPath(key).is_absolute():
        raise ValueError("Audited declaration anchor must be relative or an absolute POSIX path")
    prefix = key[:-(len(label) + 1)]
    if not prefix or ".." in PurePosixPath(prefix).parts or str(PurePosixPath(prefix)) != prefix:
        raise ValueError("Invalid anchored audit repository prefix")
    return prefix


def audited_anchor(audit, file, prefix):
    relative = str(Path(file).resolve().relative_to(ROOT.resolve()))
    matching = [value for label, value in audit["input_sha256"].items()
                if audited_relative_path(label, prefix) == relative]
    if len(matching) != 1 or matching[0] != digest(file):
        raise ValueError("Round9 audit does not anchor this exact artifact")


def verified_round9_metadata():
    """Metadata/hash-only gate: an independently passing9 audit is mandatory."""
    offline()
    audit_path, audit_sha = ROUND9 / "independent_audit.json", ROUND9 / "independent_audit.sha256"
    if not audit_path.exists() or not audit_sha.exists():
        raise ValueError("Round10 requires the completed PASS round9 audit and its SHA sidecar")
    audit = read_json(audit_path)
    round9.round8.safety(audit)
    if (audit["stage"] != "independent_tick_age_payoff_audit" or audit["passed"] is not True or audit["errors"] or
            digest(audit_path) != audit_sha.read_text().strip()):
        raise ValueError("Round9 independent audit has not passed unchanged")
    verifier = path_for(audit["verifier_file"])
    if digest(verifier) != audit["verifier_sha256"]:
        raise ValueError("Passing round9 verifier source changed")
    selection, selection_sha = frozen_round9_selection(ROUND9)
    result_path, result_sha_path = ROUND9 / "results.json", ROUND9 / "results.sha256"
    if digest(result_path) != result_sha_path.read_text().strip():
        raise ValueError("Known round9 results changed")
    result = read_json(result_path)
    round9.round8.safety(result)
    if (result["stage"] != "fixed_tick_age_payoff_final30_diagnostics" or result["selection_sha256"] != selection_sha or
            result["declaration_sha256"] != digest(ROUND9 / "declaration.json") or
            result["lineage"] != selection["lineage"] or result["config"] != round9.CONFIG or
            result["goal_achieved"] is not False or result["historical_strategy_candidate"] is not False or
            result["live_candidate"] is not False or result["price_oos"] is not False):
        raise ValueError("Exact known-history round9 result identity is required")
    # Audited input files include local inputs/issuances/targets/ledgers as well
    # as science, canonical raw/clean quote provenance and source metadata.
    pins = {}
    audited_prefix = audited_repository_prefix(audit)
    for raw_label, fingerprint in audit["input_sha256"].items():
        label = audited_relative_path(raw_label, audited_prefix)
        if label in pins:
            raise ValueError("Duplicate audited repository file identities")
        if digest(path_for(label)) != fingerprint:
            raise ValueError("An independently audited round9 source/artifact changed")
        pins[label] = fingerprint
    artifacts = [ROUND9 / name for name in ("declaration.json", "declaration.sha256", "selection.json", "selection.sha256",
                  "results.json", "results.sha256", "development_metrics.csv", "metrics.csv")]
    for file in (ROUND9 / "declaration.json", ROUND9 / "selection.json", result_path):
        audited_anchor(audit, file, audited_prefix)
    for file in (*artifacts, audit_path, audit_sha, verifier):
        pins[str(Path(file).resolve().relative_to(ROOT.resolve()))] = digest(file)
    pins.update(selection["science_code_hashes"])
    pins.update(selection["lineage"]["files"])
    return {"files": pins, "round9_selection_sha256": selection_sha, "round9_results_sha256": digest(result_path),
            "round9_audit_sha256": digest(audit_path), "round9_verifier_sha256": digest(verifier),
            "boundaries": selection["lineage"]["boundaries"], "tick_sources": selection["lineage"]["tick_sources"],
            "m1_sources": selection["lineage"]["m1_sources"], "detectors": selection["lineage"]["detectors"]}


def guard(output, stage):
    if stage not in ("declare", "develop", "evaluate"):
        raise ValueError("Unknown round10 stage")
    forbidden = ["results.json", "results.sha256", "metrics.csv"]
    if stage in ("declare", "develop"):
        forbidden += ["selection.json", "selection.sha256", "development_metrics.csv"]
    if stage == "declare":
        forbidden += ["declaration.json", "declaration.sha256"]
    if any((Path(output) / name).exists() for name in forbidden):
        raise ValueError("Refusing round10 frozen artifact or output overwrite")


def declare(output):
    offline(); guard(output, "declare")
    science, lineage = hashes(), verified_round9_metadata()
    document = {"stage": "tick_path_risk_premeasurement_declaration", "run_utc": datetime.now(timezone.utc),
        "safety": offline(), "config": CONFIG, "history": HISTORY, "dates": DATES, "symbols": SYMBOLS,
        "families": FAMILIES, "modes": MODES, "feature_names45": NAMES45, "new_feature_formulas": FEATURE_FORMULAS,
        "science_code_hashes": science, "lineage": lineage, "quotes_parsed": False,
        "new_risk_features_models_or_ledgers_computed": False, "goal_achieved": False,
        "historical_strategy_candidate": False, "live_candidate": False,
        "quote_decode_policy": "whole_canonical_boundary_day_decode_then_immediate_prefix_clip_no_suffix_calculation",
        "user_target": {"profit_factor": 1.5, "completed_strategy_paths_per_symbol_model": 1000,
                        "active_heldout_days": 60, "pilot_can_meet_target": False},
        "model_scores": "continuous_uncalibrated_scores_not_probabilities"}
    if hashes() != science or verified_round9_metadata() != lineage:
        raise ValueError("Science or known-history inputs changed during declaration")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    save(output / "declaration.json", document)
    with (output / "declaration.sha256").open("x") as stream:
        stream.write(digest(output / "declaration.json") + "\n")
    print("TICK_PATH_RISK_DECLARED", digest(output / "declaration.json"), "metadata only; known prices/payoffs", flush=True)
    return canonical(document)


def frozen_declaration(output):
    offline()
    output = Path(output)
    value = read_json(output / "declaration.json")
    if digest(output / "declaration.json") != (output / "declaration.sha256").read_text().strip():
        raise ValueError("Frozen round10 declaration changed")
    round9.round8.safety(value)
    if (value["stage"] != "tick_path_risk_premeasurement_declaration" or value["config"] != CONFIG or
            value["history"] != HISTORY or value["dates"] != list(DATES) or value["symbols"] != list(SYMBOLS) or
            value["families"] != list(FAMILIES) or value["modes"] != list(MODES) or value["feature_names45"] != list(NAMES45) or
            value["new_feature_formulas"] != FEATURE_FORMULAS or value["quotes_parsed"] is not False or
            value["new_risk_features_models_or_ledgers_computed"] is not False or value["goal_achieved"] is not False or
            value["historical_strategy_candidate"] is not False or value["live_candidate"] is not False or
            value["science_code_hashes"] != hashes() or value["lineage"] != verified_round9_metadata()):
        raise ValueError("Frozen round10 rules, history or inherited evidence changed")
    return value


def validate_model(model):
    family, status = model["family"], model["status"]
    names = list(NAMES44 if family == "RIDGE44" else NAMES45)
    if family not in FAMILIES or model["feature_names"] != names or status not in ("TESTED", "NOT TESTED"):
        raise ValueError("Exact44/45 family/schema/status required")
    n = model["training_completed_labels"]
    if status == "NOT TESTED":
        if n >= MIN_TRAIN or model["estimator"] is not None or model["serialized_estimator_sha256"] is not None:
            raise ValueError("Insufficient training fits must remain NOT TESTED")
        return
    est = model["estimator"]
    if (type(est.get("version")) is not int or est["version"] != 1 or
            est.get("scaler_fitted_on") != "training_rows_only" or
            est.get("target_clipped") is not False or est.get("features_clipped") is not False or
            est.get("objective") != "mean_squared_error_plus_penalty_times_squared_coefficients"):
        raise ValueError("Frozen ridge semantic fields differ from training-only unclipped mean-SSE protocol")
    vectors = [np.asarray(est[key], dtype=float) for key in ("means", "std", "coefs")]
    if (any(vector.shape != (len(names),) for vector in vectors) or
            not np.isfinite(np.r_[*vectors, float(est["intercept"]), float(est["threshold"])]).all() or
            (vectors[1] <= 0).any()):
        raise ValueError("Frozen44/45 scaler/coefficient dimensions or finite values changed")
    if (n < MIN_TRAIN or est["fit_rows"] != n or est["feature_names"] != names or est["penalty"] != .1 or
            est["quantile"] != .75 or est["threshold"] < 0 or est["score_kind"] != "continuous_uncalibrated_score" or
            json_hash(est) != model["serialized_estimator_sha256"]):
        raise ValueError("Frozen ridge training size/scaler/cutoff changed")


def verify_artifacts(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if value.get("local_ignored_artifact") is True:
            identity = (value["path"], value["sha256"])
            if identity not in seen:
                if digest(path_for(value["path"])) != value["sha256"]:
                    raise ValueError("Frozen10 input/matrix/issuance/label/ledger bytes changed")
                seen.add(identity)
        if "family" in value and "estimator" in value and "status" in value:
            validate_model(value)
        for child in value.values():
            verify_artifacts(child, seen)
    elif isinstance(value, list):
        for child in value:
            verify_artifacts(child, seen)


def frozen_selection(output):
    declaration = frozen_declaration(output)
    output = Path(output)
    value, fingerprint = read_json(output / "selection.json"), digest(output / "selection.json")
    if fingerprint != (output / "selection.sha256").read_text().strip():
        raise ValueError("Frozen round10 selection changed")
    round9.round8.safety(value)
    if (value["stage"] != "frozen_tick_path_risk_development" or value["config"] != CONFIG or value["history"] != HISTORY or
            value["declaration_sha256"] != digest(output / "declaration.json") or
            value["science_code_hashes"] != declaration["science_code_hashes"] or value["lineage"] != declaration["lineage"] or
            value["final45_features_scores_or_ledgers_computed"] is not False or value["goal_achieved"] is not False or
            value["historical_strategy_candidate"] is not False or value["live_candidate"] is not False or
            set(value["symbols"]) != set(SYMBOLS)):
        raise ValueError("Round10 frozen selection chronology/history changed")
    verify_artifacts(value)
    expected = {mode + "_" + family for mode in MODES for family in FAMILIES}
    for symbol, saved in value["symbols"].items():
        if saved["selected_candidate"] is not None or saved["status"] not in ("TESTED", "NOT TESTED"):
            raise ValueError("Sparse known-history pilot cannot select a strategy")
        if saved["status"] == "NOT TESTED":
            if saved["final_models"] or saved["candidates"]:
                raise ValueError("Inadequate inherited scale cannot have trained models")
            continue
        if set(saved["final_models"]) != expected or set(saved["candidates"]) != expected:
            raise ValueError("Both44/45 families and both modes must be frozen")
        for mode in MODES:
            a, b = (saved["final_models"][mode + "_" + family] for family in FAMILIES)
            keys = ("status", "training_completed_labels", "common_timestamp_target_sha256", "training_clock_issuance_sha256",
                    "training_start", "training_end", "training_matrix")
            if any(a[key] != b[key] for key in keys) or a["training_start"] != START.isoformat() or a["training_end"] != value["lineage"]["boundaries"][symbol]["70"]:
                raise ValueError("44/45 must use identical first70 rows/targets/matrix and clock")
        for key, model in saved["final_models"].items():
            if model["family"] != key.split("_")[1] or saved["candidates"][key]["development_eligible"] is not False:
                raise ValueError("Frozen family identity or impossible pilot eligibility changed")
    return value, fingerprint


def prepare_prefix(declaration, symbol, end):
    if symbol not in SYMBOLS:
        raise ValueError("Only declared native600 symbols are allowed")
    end = stamp(end)
    source = declaration["lineage"]["m1_sources"][symbol]
    m1, audit = load_m1(path_for(source["path"]))
    check_prepared(m1, audit, source)
    m1 = m1.loc[m1.index + pd.Timedelta(minutes=1) <= end].copy()
    pieces = []
    for date in DATES:
        if pd.Timestamp(date, tz="UTC") < end:
            full_day = load_tick_day(declaration["lineage"]["tick_sources"][symbol][date])
            pieces.append(full_day.loc[full_day.index < end])
    if not pieces:
        raise ValueError("No observed prefix quote source")
    ticks = pd.concat(pieces).sort_index()
    inherited = declaration["lineage"]["detectors"][symbol]
    if inherited["adequate"] is not True:
        raise ValueError("Inadequate independently audited first40 scale: NOT TESTED")
    native_side = 1 if symbol.startswith("BOOM") else -1
    described = describe_tick_path_risk(ticks, native_side, inherited["median_abs_log_return"])
    m5, names = causal_multiframe_inputs(m1, "boom" if native_side == 1 else "crash")
    table, names44, names45 = join_tick_path_risk_inputs(m5, names, described)
    table["m5_open_time"] = m5.index
    table["original_feature_valid"] = m5.feature_valid.to_numpy(bool)
    table["original44_all_finite"] = np.isfinite(m5[names44].to_numpy(float)).all(axis=1)
    mask = table.feature_valid & (table.signal_time >= START) & (table.signal_time < end)
    rows = table.loc[mask].copy()
    rows.index = pd.DatetimeIndex(rows.signal_time).as_unit("ns"); rows.index.name = "signal_time"
    if rows.index.has_duplicates or not np.isfinite(rows[names45].to_numpy(float)).all():
        raise ValueError("Common44/45 rows must be unique and finite")
    inputs = table.loc[(table.signal_time >= START) & (table.signal_time < end)].copy()
    columns = ["m5_open_time", "signal_time", *names45, "atr", "close", "original_feature_valid", "original44_all_finite",
               "multiframe_feature_valid", "tick_path_feature_valid", "common_available", "feature_valid"]
    columns += [name for name in table if name.endswith("_closed_at") or name.endswith("_row_valid")]
    inputs = inputs[list(dict.fromkeys(columns))]
    audit.update(feature_end_exclusive=end.isoformat(), m1_closed_rows_used=len(m1),
                 m1_available_closed_rows_used=int(m1.close.notna().sum()),
                 m1_unknown_closed_rows_retained=int(m1.close.isna().sum()),
                 last_m1_close=(m1.index[-1] + pd.Timedelta(minutes=1)).isoformat() if len(m1) else None,
                 tick_prefix_rows=len(ticks), tick_last_used=ticks.index[-1].isoformat(),
                 eligible_common_clock_rows=len(rows), original_m5_rows=len(table), native_side=native_side,
                 frozen_first40_scale=inherited["median_abs_log_return"], scale_refit=False,
                 canonical_boundary_day_quotes_decoded=True, suffix_calculations=False,
                 age_or_mark_availability_used=False, clock="every closed UTC M5")
    return {"ticks": ticks, "rows": rows, "inputs": inputs, "names44": names44, "names45": names45, "audit": audit}


def issue(rows, symbol, mode, model=None):
    if symbol not in SYMBOLS or mode not in MODES:
        raise ValueError("Unknown fixed symbol/strategy mode")
    chosen, scores = rows, None
    if model is not None:
        if model["status"] == "NOT TESTED":
            return []
        validate_model(model)
        names = model["feature_names"]
        scores = predict_ridge(rows[names], model["estimator"], feature_names=names) if len(rows) else np.array([])
        mask = (scores >= model["estimator"]["threshold"]) & (scores > 0)
        chosen, scores = rows.loc[mask], scores[mask]
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    result = [{"signal_time": t, "atr": float(row.atr), "side": side,
               "variant": (model["family"] if model else "CLOCK") + "_" + mode,
               "signal_close": float(row.close), PATH_FEATURE_NAME: float(row[PATH_FEATURE_NAME])}
              for t, row in chosen.iterrows()]
    if scores is not None:
        for record, score in zip(result, scores, strict=True):
            record["score"] = float(score)
    return result


def dump_signals(output, name, records):
    columns = ["signal_time", "atr", "side", "variant", "signal_close", PATH_FEATURE_NAME, "score"]
    value = dump_frame(output, "signals", name, pd.DataFrame(records).reindex(columns=columns))
    value["canonical_issuance_sha256"] = issuance_hash(records)
    return value


def aligned_training(rows, labels, start, end):
    start, end = stamp(start), stamp(end)
    times = pd.to_datetime(labels.signal_time, utc=True)
    mask = ((times >= start) & (times < end) & (times + pd.Timedelta(minutes=PURGE) <= end) &
            times.dt.strftime("%Y-%m-%d").isin(round9.observed_days(start, end)))
    for _, left, right in round9.partition_days(start, end):
        belongs = (times >= left) & (times < right)
        mask &= ~belongs | (times + pd.Timedelta(minutes=PURGE) <= right)
    past = labels.loc[mask].copy()
    finite = np.isfinite(past.net_R.to_numpy(float))
    known = past.loc[~past.censored & finite].set_index("signal_time").sort_index()
    known.index = pd.DatetimeIndex(known.index).as_unit("ns")
    if known.index.has_duplicates or rows.index.has_duplicates or not known.index.isin(rows.index).all():
        raise ValueError("Every unique completed target must align with the unique common44/45 clock")
    X = rows.loc[known.index, list(NAMES45)]
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError("Training45 inputs must be finite")
    return X, known.net_R.to_numpy(float), past, known


def training_pair(rows, labels, names44, names45, start, end, output, prefix):
    if list(names44) != list(NAMES44) or list(names45) != list(NAMES45):
        raise ValueError("Exact ordered44/45 schemas required; no age/mark additions")
    X45, y, past, known = aligned_training(rows, labels, start, end)
    matrix = X45.copy(); matrix.insert(0, "signal_time", known.index); matrix["net_R"] = y
    artifact = dump_frame(output, "matrices", prefix, matrix)
    clock_bytes, y_bytes = np.ascontiguousarray(known.index.asi8).tobytes(), np.ascontiguousarray(y).tobytes()
    common = {"training_start": stamp(start).isoformat(), "training_end": stamp(end).isoformat(),
              "training_completed_labels": len(known), "training_censored_labels": int(past.censored.sum()),
              "training_invalid_labels": int((~past.censored & ~np.isfinite(past.net_R)).sum()),
              "common_timestamp_target_sha256": hashlib.sha256(clock_bytes + y_bytes).hexdigest(),
              "training_labels_may_overlap": True, "statistically_independent_labels": False,
              "counts_toward_profit_sample_target": False, "label_purge_minutes": PURGE,
              "training_latest_issue": known.index[-1].isoformat() if len(known) else None,
              "training_matrix": artifact}
    models = {}
    for family, names in (("RIDGE44", names44), ("RIDGE45", names45)):
        X = X45[names]
        matrix_hash = hashlib.sha256(json.dumps(names).encode() + np.ascontiguousarray(X.to_numpy(float)).tobytes() + y_bytes + clock_bytes).hexdigest()
        model = {**common, "family": family, "feature_names": names, "matrix_and_target_sha256": matrix_hash}
        if len(known) < MIN_TRAIN:
            model.update(status="NOT TESTED", reason="fewer_than1000_completed_finite_individually_replayed_targets",
                         estimator=None, serialized_estimator_sha256=None)
        else:
            estimator = fit_ridge(X, y, names, penalty=.1, quantile=.75)
            model.update(status="TESTED", estimator=estimator, serialized_estimator_sha256=json_hash(estimator))
        models[family] = model
    return models


def comparison_reports(ledgers, reports, days, draws):
    result = {}
    for key, left, right in (("RIDGE45_minus_RIDGE44", "RIDGE45", "RIDGE44"),
                              ("RIDGE44_minus_CLOCK", "RIDGE44", "CLOCK"), ("RIDGE45_minus_CLOCK", "RIDGE45", "CLOCK")):
        inference = paired_difference(ledgers[left], ledgers[right], days, draws)
        if any(reports[family]["status"] != "TESTED" for family in (left, right)):
            result[key] = {"status": "NOT TESTED", "reason": "unmatched_fit_availability", "partial_descriptive_only": inference}
        else:
            result[key] = {"status": "TESTED", **inference}
    return result


def opportunity_diagnostics(context, symbol, mode, start, end, output, prefix):
    records = issue(window_rows(context["rows"], start, end), symbol, mode)
    labels, audit = replay_partition(context["ticks"], records, start, end, overlapping=True)
    planned = planned_eligible(records, start, end)
    planned_times = pd.DatetimeIndex([stamp(record["signal_time"]) for record in planned])
    present = pd.DatetimeIndex(pd.to_datetime(labels.signal_time, utc=True))
    missing = int((~planned_times.isin(present)).sum())
    days, draws = weights(start, end)
    summary = metric(labels, audit, days, draws)
    for key in ("profit_factor", "pf_inference", "sum_gains_R", "sum_losses_R", "bounded_inference_allowed"):
        summary.pop(key)
    summary["day_sums_return_count"] = [row[:2] for row in summary.pop("day_sums_return_count_gain_loss")]
    summary["profit_factor_evidence"] = "NOT TESTED_overlapping_training_targets"
    return {"audit": audit, "metrics": summary, "planned_eligible_signals": len(planned), "missing_entry_labels": missing,
            "known_subset_only": bool(missing or audit["censored"] or summary["invalid_uncensored"]),
            "labels_may_overlap": True, "strategy_pf_evidence": False, "discovery_claim": False,
            "labels": dump_frame(output, "diagnostic_labels", f"{prefix}_{symbol}_{mode}", labels),
            "signals": dump_signals(output, f"{prefix}_diagnostic_{symbol}_{mode}", records)}


def evaluate_partition(context, models, symbol, mode, start, end, output, prefix):
    rows = window_rows(context["rows"], start, end)
    days, draws = weights(start, end)
    reports, ledgers = {}, {}
    for family in ("CLOCK", *FAMILIES):
        model = None if family == "CLOCK" else models[family]
        records = issue(rows, symbol, mode, model)
        ledger, audit = replay_partition(context["ticks"], records, start, end)
        ledgers[family] = ledger
        reports[family] = {"status": "TESTED" if model is None else model["status"], "metrics": metric(ledger, audit, days, draws),
            "audit": audit, "issuance_sha256": issuance_hash(records), "common_available_rows": len(rows),
            "signals": dump_signals(output, f"{prefix}_{symbol}_{mode}_{family}", records),
            "ledger": dump_frame(output, "ledgers", f"{prefix}_{symbol}_{mode}_{family}", ledger),
            "strategy_returns": True, "quoteproxy_not_actual_fills": True}
    return reports, comparison_reports(ledgers, reports, days, draws), ledgers


def develop(output):
    offline(); guard(output, "develop")
    declaration = frozen_declaration(output)
    result = {"stage": "frozen_tick_path_risk_development", "run_utc": datetime.now(timezone.utc),
        "declaration_sha256": digest(Path(output) / "declaration.json"), "safety": offline(), "config": CONFIG, "history": HISTORY,
        "science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"],
        "final45_features_scores_or_ledgers_computed": False, "goal_achieved": False,
        "historical_strategy_candidate": False, "live_candidate": False, "symbols": {}}
    for symbol in SYMBOLS:
        bounds = {key: stamp(value) for key, value in declaration["lineage"]["boundaries"][symbol].items()}
        if declaration["lineage"]["detectors"][symbol]["adequate"] is not True:
            result["symbols"][symbol] = {"status": "NOT TESTED", "reason": "inadequate_independently_audited_scale",
                                         "final_models": {}, "candidates": {}, "selected_candidate": None}
            continue
        cache = {}
        def context_at(key):
            if key not in cache:
                cache[key] = prepare_prefix(declaration, symbol, bounds[key])
                persist_context(cache[key], output, symbol, "prefix" + key)
            return cache[key]
        candidates = {mode + "_" + family: {"mode": mode, "family": family, "walk_forward": []} for mode in MODES for family in FAMILIES}
        pooled = {mode: {family: [] for family in ("CLOCK", *FAMILIES)} for mode in MODES}
        pooled_audit = {mode: {family: dict.fromkeys(round9.COUNTERS, 0) for family in ("CLOCK", *FAMILIES)} for mode in MODES}
        for ordinal, (trainkey, testkey) in enumerate((("40", "50"), ("50", "60"), ("60", "70")), 1):
            train, test = context_at(trainkey), context_at(testkey)
            for mode in MODES:
                records = issue(window_rows(train["rows"], START, bounds[trainkey]), symbol, mode)
                labels, label_audit = replay_partition(train["ticks"], records, START, bounds[trainkey], overlapping=True)
                labels_file = dump_frame(output, "training_labels", f"train{trainkey}_{symbol}_{mode}", labels)
                models = training_pair(train["rows"], labels, train["names44"], train["names45"], START, bounds[trainkey], output,
                                       f"train{trainkey}_{symbol}_{mode}")
                signals_file = dump_signals(output, f"train{trainkey}_{symbol}_{mode}", records)
                for model in models.values():
                    model.update(training_clock_issuance_sha256=issuance_hash(records), training_signals=signals_file,
                                 training_inputs=train["audit"]["inputs"], training_labels=labels_file, training_label_audit=label_audit)
                reports, comparisons, ledgers = evaluate_partition(test, models, symbol, mode, bounds[trainkey], bounds[testkey], output, f"wf{ordinal}")
                for family in ("CLOCK", *FAMILIES):
                    pooled[mode][family].append(ledgers[family])
                    for counter in round9.COUNTERS:
                        pooled_audit[mode][family][counter] += reports[family]["audit"][counter]
                for family in FAMILIES:
                    candidates[mode + "_" + family]["walk_forward"].append({"training": "train" + trainkey,
                        "test": "wf" + str(ordinal), "model": models[family], **reports[family], "comparisons": comparisons})
        final = context_at("70")
        days, draws = weights(bounds["40"], bounds["70"])
        final_models, validation, diagnostics = {}, {}, {}
        for mode in MODES:
            records = issue(window_rows(final["rows"], START, bounds["70"]), symbol, mode)
            labels, audit = replay_partition(final["ticks"], records, START, bounds["70"], overlapping=True)
            labels_file = dump_frame(output, "training_labels", f"train70_{symbol}_{mode}", labels)
            pair = training_pair(final["rows"], labels, final["names44"], final["names45"], START, bounds["70"], output,
                                 f"train70_{symbol}_{mode}")
            signals_file = dump_signals(output, f"train70_{symbol}_{mode}", records)
            for family, model in pair.items():
                model.update(training_clock_issuance_sha256=issuance_hash(records), training_signals=signals_file,
                             training_inputs=final["audit"]["inputs"], training_labels=labels_file, training_label_audit=audit)
                final_models[mode + "_" + family] = model
            ledgers = {family: pd.concat(frames, ignore_index=True) for family, frames in pooled[mode].items()}
            validation[mode] = {}
            for family, ledger in ledgers.items():
                statuses = ([fold["status"] for fold in candidates[mode + "_" + family]["walk_forward"]] if family != "CLOCK" else ["TESTED"] * 3)
                status = "TESTED" if all(s == "TESTED" for s in statuses) else "PARTIALLY TESTED" if any(s == "TESTED" for s in statuses) else "NOT TESTED"
                validation[mode][family] = {"status": status, "fold_fit_statuses": statuses,
                    "metrics": metric(ledger, pooled_audit[mode][family], days, draws), "audit": pooled_audit[mode][family],
                    "ledger": dump_frame(output, "ledgers", f"validation_{symbol}_{mode}_{family}", ledger)}
            validation[mode]["comparisons"] = comparison_reports(ledgers, validation[mode], days, draws)
            for family in FAMILIES:
                candidate = candidates[mode + "_" + family]
                candidate["validation"] = validation[mode][family]["metrics"]
                candidate["final_model_status"] = pair[family]["status"]
                candidate["development_eligible"] = bool(development_eligible(candidate) and pair[family]["status"] == "TESTED" and len(days) >= 30)
                candidate["eligibility_limitation"] = "only" + str(len(days)) + "_observed_validation_days_less_than30"
            diagnostics[mode] = opportunity_diagnostics(final, symbol, mode, bounds["40"], bounds["70"], output, "validation")
        result["symbols"][symbol] = {"status": "TESTED", "audits": {key: value["audit"] for key, value in cache.items()},
            "final_models": final_models, "candidates": candidates, "validation": validation, "opportunity_diagnostics": diagnostics,
            "selected_candidate": None, "eligible_candidates": 0}
        print(symbol, "known-history path-risk development complete; pilot candidate false", flush=True)
    if declaration != frozen_declaration(output):
        raise ValueError("Frozen10 evidence changed during development")
    verify_artifacts(result)
    save(Path(output) / "selection.json", result)
    with (Path(output) / "selection.sha256").open("x") as stream:
        stream.write(digest(Path(output) / "selection.json") + "\n")
    write_metrics(Path(output) / "development_metrics.csv", result, development=True)
    print("TICK_PATH_RISK_SELECTION_FROZEN; final45 features/scores/ledgers not calculated; payoffs already known", flush=True)
    return canonical(result)


def expansion_diagnostic(reports, comparisons):
    """Descriptive economic prerequisite, never permission or a strategy gate."""
    risk = reports["RIDGE45"]
    m = risk["metrics"]
    increments = [comparisons[key] for key in ("RIDGE45_minus_RIDGE44", "RIDGE45_minus_CLOCK")]
    mean_ci = m["day_mean_ci95"]
    positive = (risk["status"] == "TESTED" and m["mean_inference"]["all_draws_defined"] and mean_ci is not None and mean_ci[0] > 0 and
                all(reports[family]["metrics"]["censored"] == 0 and reports[family]["metrics"]["missing_entry"] == 0 and
                    reports[family]["metrics"]["invalid_uncensored"] == 0 for family in ("CLOCK", *FAMILIES)) and
                all(row["status"] == "TESTED" and row["all_draws_defined"] and row["ci95"] is not None and row["ci95"][0] > 0 for row in increments))
    return {"positive_absolute_and_incremental_day_bounds": bool(positive), "expansion_authorized": False,
            "historical_gate": False, "live_candidate": False,
            "scope": "known_history_economic_prerequisite_only_not_stability_or_profit_discovery"}


def evaluate(output):
    offline(); guard(output, "evaluate")
    selection, fingerprint = frozen_selection(output)
    result = {"stage": "known_history_tick_path_risk_final30_diagnostics", "run_utc": datetime.now(timezone.utc),
        "selection_sha256": fingerprint, "declaration_sha256": selection["declaration_sha256"], "safety": offline(),
        "config": CONFIG, "history": HISTORY, "science_code_hashes": selection["science_code_hashes"], "lineage": selection["lineage"],
        "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False,
        "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED", "symbols": {}}
    for symbol in SYMBOLS:
        saved = selection["symbols"][symbol]
        if saved["status"] == "NOT TESTED":
            result["symbols"][symbol] = {"status": "NOT TESTED", "reason": saved["reason"], "models": {}}; continue
        bounds = {key: stamp(value) for key, value in selection["lineage"]["boundaries"][symbol].items()}
        context = prepare_prefix(selection, symbol, bounds["100"])
        persist_context(context, output, symbol, "prefix100")
        models, diagnostics = {}, {}
        for mode in MODES:
            pair = {family: saved["final_models"][mode + "_" + family] for family in FAMILIES}
            reports, comparisons, _ = evaluate_partition(context, pair, symbol, mode, bounds["70"], bounds["100"], output, "final30")
            for family in FAMILIES:
                reports[family].update(development_eligible=saved["candidates"][mode + "_" + family]["development_eligible"],
                                      historical_gate=False, live_candidate=False, model_sha256=pair[family]["serialized_estimator_sha256"])
            models[mode] = {"reports": reports, "comparisons": comparisons,
                           "expansion_diagnostic": expansion_diagnostic(reports, comparisons)}
            diagnostics[mode] = opportunity_diagnostics(context, symbol, mode, bounds["70"], bounds["100"], output, "final30")
        result["symbols"][symbol] = {"status": "TESTED", "audit": context["audit"], "models": models,
                                     "opportunity_diagnostics": diagnostics, "selected_candidate": None}
    if frozen_selection(output)[1] != fingerprint:
        raise ValueError("Frozen10 selection/evidence changed during evaluation")
    verify_artifacts(result)
    save(Path(output) / "results.json", result)
    with (Path(output) / "results.sha256").open("x") as stream:
        stream.write(digest(Path(output) / "results.json") + "\n")
    write_metrics(Path(output) / "metrics.csv", result)
    print("TICK_PATH_RISK_KNOWN_HISTORY_DIAGNOSTICS_COMPLETE; no profit/discovery/prospective claim", flush=True)
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
                rows.append({"symbol": symbol, "mode": mode, "family": family, "status": reports[family]["status"],
                    **{key: m[key] for key in ("completed", "censored", "missing_entry", "profit_factor", "mean_net_R", "active_days", "observed_day_clusters")},
                    "known_prices": True, "known_payoffs": True, "price_oos": False, "historical_gate": False, "live_candidate": False})
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
