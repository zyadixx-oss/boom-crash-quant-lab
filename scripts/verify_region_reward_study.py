#!/usr/bin/env python3
"""Independent conditional-region labels, ridge, quote paths and inference audit.

Only independently written audit helpers are imported. Cached feature formulas
and intrinsic validity retain the prior audit's explicit scope limitation.
Nothing here fits production learners, calls replay engines or opens an account.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
sys.path.insert(0, str(ROOT))
from scripts.verify_zone_study import (Audit, FLAGS, TIME_FIELDS, epoch,
    independent_path, metric_checks, read_events, sha)
from scripts.verify_hybrid_event_study import (MODEL_ATOL, MODEL_RTOL, array,
    clock_rows, compare_record, frame, independent_ridge, inference_oracle, read_ledger)

FOLDER = ROOT / "docs/region_reward_20261008"
PARENT = ROOT / "docs/hybrid_event_regions_20261008"
PARENT_DECLARATION = "6ade87b7207754cc21468d68ab315c3f66f5b84bedc2e274907b453759e03af5"
PARENT_RESULT = "fb345c068a899157e15ec6393a72a7eeb5d78f932b7f7b7f0fac0640823b4abd"
FAMILIES = ("CLOCK", "RAW_TIMED", "HYBRID_TIMED", "RAW_REGION", "HYBRID_REGION")
NEW = ("RAW_REGION", "HYBRID_REGION")
PARTS = ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180")
UNKNOWN = ("waiting_gap", "entry_gap", "path_gap")
NONFILL = ("invalidated", "expired")
LABEL_COLUMNS = ("signal_time", "net_R", "completed", "known_nonfill", "unknown_path",
                 "exposure_skipped", "status", "planned_end")
REFERENCES = {"RAW_REGION": ("CLOCK", "RAW_TIMED"),
              "HYBRID_REGION": ("CLOCK", "HYBRID_TIMED", "RAW_REGION")}
FIXED_CONFIG = {"activation_delay_minutes": 1, "wait_minutes": 15, "hold_minutes": 15,
                "stop_atr": 2., "cost_atr": .1,
                "purge_minutes": 31}


def safety():
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() not in ("false", "0", "no", "off", ""):
            raise ValueError("Live flag is not false: " + flag)
    return dict.fromkeys(FLAGS, False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def booleans(values, name):
    if not values.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Known actual boolean values required: " + name)
    return values.to_numpy(bool)


def timestamps(values):
    result = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    # Parsing utc=True must not silently repair an unqualified or fractional time.
    for value in values:
        epoch(value)
    if result.hasnans or result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("Exact sorted unique UTC timestamps required")
    return result


def decode_sources(audit, sources):
    """Independent NumPy decoder with daily bounds, schema and original gaps."""
    total = sum(s["rows"] for s in sources)
    times, prices = np.empty(total, np.int64), np.empty(total, float)
    cursor, last, dates = 0, None, []
    for source in sources:
        p = audit.pin(source["path"], source["sha256"])
        with p.open() as f:
            if f.readline().strip() != "epoch,quote":
                raise ValueError("Exact original quote schema required")
        data = np.loadtxt(p, delimiter=",", skiprows=1,
                          dtype=[("epoch", np.int64), ("quote", np.float64)], ndmin=1)
        if len(data) != source["rows"]:
            raise ValueError("Original source count changed")
        t, q = data["epoch"], data["quote"]
        first = epoch(pd.Timestamp(source["date"], tz="UTC"))
        if (np.any(np.diff(t) <= 0) or np.any(t < first) or np.any(t >= first + 86400)
                or (len(t) and last is not None and t[0] <= last)
                or not np.isfinite(q).all() or np.any(q <= 0)):
            raise ValueError("Original source chronology, daily bounds or prices changed")
        dates.append(source["date"])
        n = len(data); times[cursor:cursor+n], prices[cursor:cursor+n] = t, q
        cursor += n
        if n: last = int(t[-1])
    if dates != sorted(set(dates)):
        raise ValueError("Unique chronological source dates required")
    return times, prices


def region(signal, symbol, family):
    issue, atr, p = epoch(signal["signal_time"]), float(signal["atr"]), float(signal["signal_close"])
    side = 1 if symbol == "BOOM600" else -1 if symbol == "CRASH600" else 0
    if (not side or isinstance(signal["side"], (bool, np.bool_)) or signal["side"] != side
            or issue % 1800 or not np.isfinite([atr, p]).all() or min(atr, p) <= 0):
        raise ValueError("Exact native-side clock, original price and raw ATR required")
    lo, hi = sorted((p - side * .55 * atr, p - side * .45 * atr))
    inv = lo - .25 * atr if side == 1 else hi + .25 * atr
    if min(lo, inv) <= 0:
        raise ValueError("Positive original region required")
    stamp = pd.Timestamp(issue, unit="s", tz="UTC")
    return {"signal_time": stamp, "side": side, "atr": atr, "issue_price": p,
            "zone_low": lo, "zone_high": hi, "invalidation": inv,
            "variant": family, "zone_id": f"{symbol}:{family}:{stamp.isoformat()}"}


def oracle_events(times, prices, signals, symbol, family, bounds, cfg):
    start, end = map(epoch, bounds); busy, result = start, []
    timestamps(signals.signal_time)
    for signal in signals.to_dict("records"):
        z = region(signal, symbol, family)
        event, busy = independent_path(times, prices, z, cfg, start, end, busy)
        result.append({**z, **event})
    return result


def conditional_labels(issues, events):
    """Derive every disposition from the independent path, never saved labels."""
    issues = timestamps(issues)
    if len(issues) != len(events) or [epoch(x) for x in issues] != [x["signal_time"] for x in events]:
        raise ValueError("Independent training events must equal the full common clock")
    labels = []
    for issue, event in zip(issues, events, strict=True):
        status = event["status"]
        if status not in (*UNKNOWN, *NONFILL, "overlap_skipped", "completed"):
            raise ValueError("Unexpected unpurged CLOCK training disposition")
        complete, unknown = status == "completed", status in UNKNOWN
        value = event["net_R"]
        if (bool(event["censored"]) != unknown or (complete and (value is None or not math.isfinite(value)))
                or (not complete and value is not None and math.isfinite(value))
                or (complete and event["entry_time"] is None)):
            raise ValueError("Independent conditional label semantics differ")
        labels.append({"signal_time": issue, "net_R": value if complete else np.nan,
            "completed": complete, "known_nonfill": status in NONFILL,
            "unknown_path": unknown, "exposure_skipped": status == "overlap_skipped",
            "status": status, "planned_end": issue + pd.Timedelta(minutes=30, seconds=1)})
    return pd.DataFrame(labels, columns=LABEL_COLUMNS)


def check_labels(audit, actual, expected, train_end):
    if list(actual.columns) != list(LABEL_COLUMNS):
        raise ValueError("Exact saved conditional label schema required")
    array(audit, timestamps(actual.signal_time).asi8, timestamps(expected.signal_time).asi8,
          "full ordered conditional CLOCK labels")
    for name in ("completed", "known_nonfill", "unknown_path", "exposure_skipped"):
        array(audit, booleans(actual[name], name), expected[name], "conditional category " + name)
    array(audit, actual.status, expected.status, "conditional saved status")
    array(audit, actual.net_R, expected.net_R, "independent original-quote region reward", close=True)
    array(audit, timestamps(actual.planned_end).asi8, timestamps(expected.planned_end).asi8,
          "exact conditional planned horizon")
    audit.check(bool((timestamps(actual.planned_end) < pd.Timestamp(train_end)).all()), "conditional planned purge")
    audit.check(bool(np.all(actual[["completed", "known_nonfill", "unknown_path", "exposure_skipped"]].sum(axis=1) == 1)),
                "entire CLOCK category population partitions exactly")
    complete = booleans(actual.completed, "completed")
    values = actual.net_R.to_numpy(float)
    audit.check(bool(np.isfinite(values[complete]).all() and np.isnan(values[~complete]).all()),
                "excluded labels never fabricated zero rewards")


def target_hash(issues, y):
    h = hashlib.sha256(b"paired-region-filled-target-v1\0")
    h.update(np.asarray(pd.DatetimeIndex(issues).as_unit("ns").asi8, dtype=">i8").tobytes())
    h.update(np.asarray(y, dtype=">f8").tobytes())
    return h.hexdigest()


def check_events(audit, actual, expected, replay_audit, cfg, label):
    audit.equal(len(actual), len(expected), label + " full event population")
    for saved, event in zip(actual.to_dict("records"), expected, strict=True):
        compare_record(audit, saved, event, label)
    statuses = dict(Counter(x["status"] for x in expected))
    audit.equal(replay_audit["statuses"], statuses, label + " dispositions")
    audit.equal(replay_audit["issued"], len(expected), label + " issued count")
    audit.equal(replay_audit["unknown"], sum(x["censored"] for x in expected), label + " all unknowns")
    audit.equal(replay_audit["filled"], sum(x["entry_time"] is not None for x in expected), label + " filled count")
    audit.equal(replay_audit["config"], cfg, label + " fixed replay config")
    audit.equal(replay_audit["safety"], safety(), label + " safe replay")


def check_ledger(audit, saved, events, expected, label):
    filled = [e for e in expected if e["entry_time"] is not None]
    audit.equal(len(saved), len(filled), label + " filled ledger size")
    for actual, event in zip(saved.to_dict("records"), filled, strict=True):
        compare_record(audit, actual, event, label + " filled quote path")
    from_events = events.loc[events.entry_time.ne("")].reset_index(drop=True)
    audit.check(list(saved.columns) == list(from_events.columns), label + " ledger schema equals events")
    # Both readers normalize blanks/booleans identically; compare NaN as unknown.
    for name in saved.columns:
        array(audit, saved[name], from_events[name], label + " ledger equals events " + name)


def check_model(audit, state, labels, raw, hybrid, positions, names, symbol, bounds):
    complete = labels.completed.to_numpy(bool); count = int(complete.sum())
    counts = {"completed": count, "known_nonfill": int(labels.known_nonfill.sum()),
              "unknown_path": int(labels.unknown_path.sum()), "exposure_skipped": int(labels.exposure_skipped.sum())}
    for key, value in counts.items(): audit.equal(state[key], value, "training state " + key)
    audit.equal(state["training_dispositions"], dict(Counter(labels.status)), "training disposition counts")
    audit.equal(state["safety"], safety(), "safe conditional fit state")
    audit.check(state["QUALIFIED"] is False, "conditional fit never qualified")
    if count < 1000:
        audit.equal(state["status"], "NOT_FIT_INSUFFICIENT_REGION_LABELS", "insufficient region training")
        audit.check(state["models"] is None and state["model_metadata"] is None, "no insufficient-sample fallback")
        return {}
    audit.equal(state["status"], "FITTED", "minimum completed conditional training")
    audit.equal(state["error"], None, "fitted model has no error")
    if set(state["models"]) != set(NEW): raise ValueError("Exactly two new conditional estimators required")
    meta = state["model_metadata"]
    audit.equal(meta["metadata_sha256"], fingerprint({k: v for k, v in meta.items() if k != "metadata_sha256"}), "fit metadata hash")
    audit.equal(meta["symbol"], symbol, "fit symbol")
    audit.equal(meta["native_side"], 1 if symbol == "BOOM600" else -1, "fit native side")
    for k, v in (("training_start", bounds[0]), ("training_end", bounds[1])):
        audit.equal(epoch(meta[k]), epoch(v), k)
    for k, v in counts.items(): audit.equal(meta[k], v, "fit metadata " + k)
    audit.equal(meta["opportunities"], len(labels), "entire CLOCK fit metadata population")
    audit.equal(meta["feature_names"], names, "fit ordered44 names")
    audit.equal(meta["penalty"], .1, "fixed mean-SSE ridge penalty")
    audit.equal(meta["quantile"], .75, "fixed completed-training quantile")
    audit.equal(meta["safety"], safety(), "safe fit metadata")
    audit.equal(meta["target"], "conditional_completed_original_quote_CLOCK_region_net_R", "conditional target definition")
    audit.check(meta["training_is_independent_trade_sample"] is False and meta["label_selection_is_conditional_on_CLOCK_fill"] is True,
                "conditional overlapping-prefix exposure acknowledged")
    y = labels.net_R.to_numpy(float)[complete]
    issues = timestamps(labels.signal_time)[complete]
    digest = target_hash(issues, y)
    audit.equal(meta["target_sha256"], digest, "paired completed target hash")
    oracles = {}
    for family, features in (("RAW_REGION", raw), ("HYBRID_REGION", hybrid)):
        x = features.loc[positions[complete], names].to_numpy(float)
        audit.check(bool(np.isfinite(x).all()), "finite complete conditional training matrix")
        oracle = independent_ridge(x, y); model = state["models"][family]
        for key in ("means", "std", "coefs", "intercept", "threshold"):
            array(audit, model[key], oracle[key], "independent conditional ridge " + family + key, close=True)
        for key, value in (("feature_names", names), ("fit_rows", count), ("penalty", .1), ("quantile", .75), ("version", 1)):
            audit.equal(model[key], value, "fixed conditional model " + key)
        audit.equal(meta["model_sha256"][family], fingerprint(model), "conditional model hash")
        matrix = hashlib.sha256(digest.encode()); matrix.update(np.asarray(x, dtype=">f8").tobytes())
        audit.equal(meta["matrix_sha256"][family], matrix.hexdigest(), "paired completed matrix hash")
        oracles[family] = oracle
    return oracles


def scores_and_selection(features, positions, names, model):
    x = features.loc[positions, names].to_numpy(float)
    scores = model["intercept"] + ((x - np.asarray(model["means"])) / np.asarray(model["std"])) @ np.asarray(model["coefs"])
    return scores, (scores >= model["threshold"]) & (scores > 0)


def check_reference(audit, actual, prior, label):
    """Renaming a frozen stream must preserve every other saved value exactly."""
    columns = [c for c in actual if c not in ("variant", "zone_id")]
    audit.check(set(actual.columns) == set(prior.columns), label + " reference schema")
    for col in columns:
        array(audit, actual[col], prior[col], label + " frozen reference " + col)


def compare_inference(audit, saved, values, week, label):
    ci, difference, prob = values
    keys = ("weekly_mean_net_R_ci95", "weekly_difference_ci95", "weekly_p") if week else (
        "mean_net_R_ci95", "baseline_difference_ci95", "p")
    for key, targets in ((keys[0], ci), (keys[1], difference)):
        for actual, expected in zip(saved[key], targets, strict=True): audit.equal(actual, expected, label + key)
    audit.equal(saved[keys[2]], float(prob), label + keys[2])
    return float(prob)


def holm(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Finite probabilities in [0,1] required")
    order = np.argsort(values, kind="stable"); out, maximum = np.empty(len(values)), 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1., (len(order) - rank) * values[i])); out[i] = maximum
    return out


def load_features(audit, manifest, prior, names):
    audit.equal(manifest["features"], prior["features"], "exact immutable prior cached features")
    audit.equal(manifest["timed_fit"], prior["fit"], "exact immutable prior timed fit")
    raw, hybrid = [frame(audit, manifest["features"][v], features=True) for v in ("RAW44", "HYBRID44")]
    for f in (raw, hybrid):
        if list(f.columns) != ["m5_open", *names, "feature_valid", "atr", "close"]:
            raise ValueError("Exact full cached44 schema required")
        index = timestamps(f.m5_open)
        if len(index) > 1 and not np.all(np.diff(index.asi8) == 300 * 10**9):
            raise ValueError("Full uncompressed M5 feature grid required")
        booleans(f.feature_valid, "intrinsic cached feature_valid")
    array(audit, raw.m5_open, hybrid.m5_open, "paired full feature grid")
    array(audit, hybrid.log1p_large_bar_age, raw.log1p_large_bar_age, "unchanged raw age in hybrid44")
    available = frame(audit, manifest["features"]["availability"], features=True)
    all_issues = timestamps(raw.m5_open) + pd.Timedelta(minutes=5)
    mask = all_issues.asi8 % (1800 * 10**9) == 0
    array(audit, timestamps(available.issue_time).asi8, all_issues[mask].asi8, "entire saved scheduled availability")
    array(audit, timestamps(available.m5_open).asi8, timestamps(raw.m5_open)[mask].asi8, "scheduled opening grid")
    for key, expected in (("raw_feature_valid", raw.feature_valid.to_numpy(bool)[mask]),
                          ("transformed_feature_valid", hybrid.feature_valid.to_numpy(bool)[mask]),
                          ("common_feature_valid", (raw.feature_valid & hybrid.feature_valid).to_numpy(bool)[mask])):
        array(audit, booleans(available[key], key), expected, "saved availability " + key)
    array(audit, available.raw_execution_atr, raw.atr.loc[mask], "separate raw execution ATR")
    array(audit, available.transformed_feature_atr, hybrid.atr.loc[mask], "separate hybrid feature ATR")
    known = available.common_feature_valid.to_numpy(bool)
    runs = pd.to_numeric(available.run_id, errors="raise").to_numpy(float)
    audit.check(bool(np.isfinite(runs[known]).all() and np.all(runs[known] >= 0) and np.all(runs[known] % 1 == 0)),
                "known native run identity on common scheduled rows")
    for f in (raw, hybrid):
        valid = f.feature_valid.to_numpy(bool)
        audit.check(bool(np.isfinite(f.loc[valid, names].to_numpy(float)).all()), "enabled cached44 finite")
    model_path = audit.pin(manifest["timed_fit"]["path"], manifest["timed_fit"]["sha256"])
    timed = json.loads(model_path.read_text())
    audit.equal(timed["status"], "FITTED", "complete prior frozen timed pair")
    return raw, hybrid, timed


def verify(audit):
    safety()
    decl_path, result_path = audit.pin(FOLDER / "declaration.json"), audit.pin(FOLDER / "results.json")
    d, result = json.loads(decl_path.read_text()), json.loads(result_path.read_text())
    audit.equal((FOLDER / "declaration.sha256").read_text().strip(), sha(decl_path), "declaration sidecar")
    audit.equal((FOLDER / "results.sha256").read_text().strip(), sha(result_path), "result sidecar")
    audit.equal(result["declaration_sha256"], sha(decl_path), "result frozen declaration")
    for name, pin in d["code_sha256"].items(): audit.pin(name, pin)
    for p in (__file__, "backend/tests/test_region_reward_audit.py", "scripts/verify_hybrid_event_study.py", "scripts/verify_zone_study.py"):
        audit.pin(p)
    parent_d_path = audit.pin(PARENT / "declaration.json", PARENT_DECLARATION)
    parent_result_path = audit.pin(PARENT / "results.json", PARENT_RESULT)
    parent_d, parent = json.loads(parent_d_path.read_text()), json.loads(parent_result_path.read_text())
    parent_audit_path = audit.pin(PARENT / "independent_audit.json", d["inputs"]["parent_audit_sha256"])
    prior_audit = json.loads(parent_audit_path.read_text())
    audit.check(prior_audit["passed"] is True and not prior_audit["errors"] and prior_audit["models_checked"] == 16,
                "prior independent sources, timed target and ridge audit PASS")
    audit.equal(d["inputs"]["parent_declaration_sha256"], PARENT_DECLARATION, "declared prior declaration")
    audit.equal(d["inputs"]["parent_result_sha256"], PARENT_RESULT, "declared prior result")
    audit.equal(d["sources"], parent_d["sources"], "same original quote population")
    audit.pin(d["sources"]["audit_path"], d["sources"]["audit_sha256"])
    for obj in (d, result):
        audit.equal(obj["safety"], safety(), "all four live gates false")
        audit.check(obj["QUALIFIED"] is False, "adaptive history cannot qualify")
    spec, old_spec = d["specification"], parent_d["specification"]
    names, cfg = spec["feature_names"], spec["region_config"]
    audit.equal(names, old_spec["feature_names"], "unchanged ordered feature schema")
    audit.check(len(names) == 44 and len(set(names)) == 44, "exact44 feature names")
    audit.equal(spec["families"], list(FAMILIES), "fixed five streams")
    audit.equal(spec["folds"], old_spec["folds"], "unchanged temporal folds")
    audit.equal(cfg, old_spec["region_config"], "unchanged region execution")
    # Config may acquire no new trade rule merely because the runner saved it.
    for key, value in FIXED_CONFIG.items(): audit.equal(cfg[key], value, "declared fixed " + key)
    for key, value in (("features_recomputed", False), ("minimum_completed_training_labels", 1000),
                       ("ridge_penalty", .1), ("training_quantile", .75), ("bootstrap_repeats", 9999),
                       ("bootstrap_seed", 20261008), ("joint_heldout_Holm_family", 8),
                       ("fresh_out_of_sample", False), ("known_history_adaptive", True)):
        audit.equal(spec[key], value, "fixed specification " + key)
    expected_cells = {(s, v, p) for s in ("BOOM600", "CRASH600") for v in FAMILIES for p in PARTS}
    rows = result["rows"]
    index = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    audit.check(len(rows) == 60 and set(index) == expected_cells, "all60 unique result cells")
    if len(rows) != 60 or set(index) != expected_cells: raise ValueError("Incomplete result population")
    old_index = {(r["symbol"], r["variant"], r["partition"]): r for r in parent["rows"] if r["endpoint"] == "REGION"}
    ledgers, models_checked, issued, native_paths, training_paths = {}, 0, 0, 0, 0
    training_categories, frozen_references_checked = {}, 0
    source_rows, source_gaps = 0, 0
    for symbol in ("BOOM600", "CRASH600"):
        print(symbol + ": independent source and conditional-label reconstruction", flush=True)
        times, prices = decode_sources(audit, d["sources"]["symbols"][symbol])
        source_rows += len(times); source_gaps += int(np.maximum(np.diff(times) - 1, 0).sum())
        for fold, contract in spec["folds"].items():
            manifest = d["inputs"]["fold_inputs"][symbol][fold]
            raw, hybrid, timed = load_features(audit, manifest, parent["symbols"][symbol]["folds"][fold], names)
            bounds = contract["train"]; positions, issues = clock_rows(raw, hybrid, *bounds)
            side = 1 if symbol == "BOOM600" else -1
            clock = pd.DataFrame({"signal_time": issues, "side": side, "atr": raw.atr.iloc[positions].to_numpy(float),
                                  "signal_close": raw.close.iloc[positions].to_numpy(float)})
            expected = oracle_events(times, prices, clock, symbol, "CLOCK", bounds, cfg)
            labels = conditional_labels(issues, expected)
            training_categories.setdefault(symbol, {})[fold] = {
                "opportunities": len(labels), "statuses": dict(Counter(labels.status)),
                **{k: int(labels[k].sum()) for k in ("completed", "known_nonfill", "unknown_path", "exposure_skipped")},
                "prefixes_overlap_not_independent_samples": True}
            training_paths += len(expected)
            info = result["symbols"][symbol]["folds"][fold]
            saved_labels = frame(audit, info["labels"], features=True)
            check_labels(audit, saved_labels, labels, bounds[1])
            events = read_events(audit.pin(info["events"]["path"], info["events"]["sha256"]))
            state = json.loads(audit.pin(info["models"]["path"], info["models"]["sha256"]).read_text())
            check_events(audit, events, expected, state["training_audit"], cfg, symbol + fold + " training")
            oracle = check_model(audit, state, labels, raw, hybrid, positions, names, symbol, bounds)
            models_checked += len(oracle)
            audit.equal(info["status"], state["status"], "fold saved fit status")
            for part, bounds in contract["evaluate"].items():
                positions, issues = clock_rows(raw, hybrid, *bounds)
                for family in FAMILIES:
                    row = index[(symbol, family, part)]
                    audit.equal([row["start"], row["end_exclusive"]], bounds, "exact evaluation bounds")
                    audit.equal(row["endpoint"], "REGION", "fixed region application")
                    signals = frame(audit, row["artifacts"]["signals"])
                    features = hybrid if family.startswith("HYBRID") else raw
                    scores = np.full(len(issues), np.nan)
                    selected = np.ones(len(issues), bool) if family == "CLOCK" else np.zeros(len(issues), bool)
                    model = None
                    if family in oracle: model = oracle[family]
                    elif family in ("RAW_TIMED", "HYBRID_TIMED"):
                        model = timed["models"]["RAW44" if family == "RAW_TIMED" else "TRANSFORMED44"]
                    if model is not None: scores, selected = scores_and_selection(features, positions, names, model)
                    array(audit, timestamps(signals.signal_time).asi8, issues[selected].asi8, "exact past-only selected issuance")
                    array(audit, signals.atr, raw.atr.iloc[positions[selected]], "unchanged raw execution ATR")
                    array(audit, signals.signal_close, raw.close.iloc[positions[selected]], "original quote issue price")
                    array(audit, signals.side, np.full(int(selected.sum()), side), "fixed native side")
                    expected_variant = family + "_SPIKE"
                    array(audit, signals.variant, np.full(len(signals), expected_variant), "fixed signal family")
                    array(audit, pd.to_numeric(signals.score, errors="coerce"), scores[selected], "frozen issue score", close=True)
                    issued += len(signals)
                    expected = oracle_events(times, prices, signals, symbol, family, bounds, cfg)
                    events = read_events(audit.pin(row["artifacts"]["events"]["path"], row["artifacts"]["events"]["sha256"]))
                    check_events(audit, events, expected, row["replay_audit"], cfg, symbol + part + family)
                    ledger = read_events(audit.pin(row["artifacts"]["ledger"]["path"], row["artifacts"]["ledger"]["sha256"]))
                    check_ledger(audit, ledger, events, expected, symbol + part + family)
                    native_paths += len(expected)
                    metric_checks(audit, row, ledger, cfg, 9999, 20261008)
                    holding = ledger.loc[~ledger.censored & np.isfinite(ledger.net_R), "holding_minutes"].to_numpy(float)
                    audit.equal(row["metrics"]["median_holding_minutes"], float(np.median(holding)) if len(holding) else None, "median completed holding time")
                    ledgers[(symbol, family, part)] = ledger
                    if family in ("CLOCK", "RAW_TIMED", "HYBRID_TIMED"):
                        old_family = {"CLOCK": "CLOCK", "RAW_TIMED": "RAW44", "HYBRID_TIMED": "HYBRID44"}[family]
                        previous = old_index[(symbol, old_family, part)]
                        check_reference(audit, signals, frame(audit, previous["artifacts"]["signals"]), family + " signals")
                        old_events = read_events(audit.pin(previous["artifacts"]["events"]["path"], previous["artifacts"]["events"]["sha256"]))
                        old_ledger = read_events(audit.pin(previous["artifacts"]["ledger"]["path"], previous["artifacts"]["ledger"]["sha256"]))
                        check_reference(audit, events, old_events, family + " events")
                        check_reference(audit, ledger, old_ledger, family + " ledger")
                        frozen_references_checked += 1
                print(symbol + "/" + part + ": five region streams independently checked", flush=True)
            del raw, hybrid
        del times, prices
    audit.equal(source_rows, 62380130, "same62,380,130 native quotes")
    audit.equal(source_gaps, 670, "same670 missing seconds preserved")
    for symbol in ("BOOM600", "CRASH600"):
        for family in FAMILIES:
            row = index[(symbol, family, "walk_forward_combined")]
            ledger = pd.concat([ledgers[(symbol, family, p)] for p in ("wf1", "wf2", "wf3")], ignore_index=True)
            ledgers[(symbol, family, "walk_forward_combined")] = ledger
            metric_checks(audit, row, ledger, cfg, 9999, 20261008)
            audit.equal(row["replay_audit"]["unknown"], sum(index[(symbol, family, p)]["replay_audit"]["unknown"] for p in ("wf1", "wf2", "wf3")), "union unknown count")
    held, probs, comparisons = [], [], 0
    for row in rows:
        symbol, family, part = row["symbol"], row["variant"], row["partition"]
        ledger = ledgers[(symbol, family, part)]; references = REFERENCES.get(family, ("CLOCK",))
        if family in NEW:
            if set(row.get("target_comparisons", {})) != set(references) - {"CLOCK"}:
                raise ValueError("Every required target comparison must be saved, including the union")
        joint = []
        for reference in references:
            baseline = ledgers[(symbol, reference, part)]
            for week in (False, True):
                saved = row["weekly_inference" if week else "day_inference"] if reference == "CLOCK" else row["target_comparisons"][reference]["week" if week else "day"]
                values = inference_oracle(ledger, baseline, row["start"], row["end_exclusive"], 9999, 20261008, week)
                joint.append(compare_inference(audit, saved, values, week, symbol + family + part + reference))
                comparisons += 1
        if family in NEW and part in ("final_test", "later180"):
            held.append(row); probs.append(max(joint))
    audit.equal(len(held), 8, "joint eight new held-out hypotheses")
    for row, expected in zip(held, holm(probs), strict=True): audit.equal(row["holm_p"], float(expected), "joint8 Holm conjunction")
    for name, expected in tuple(audit.pins.items()): audit.equal(sha(ROOT / name), expected, "final stable pin " + name)
    return {"models_checked": models_checked, "issued_scores_checked": issued,
        "training_CLOCK_paths_checked_including_overlapping_prefixes": training_paths,
        "evaluation_region_dispositions_checked": native_paths, "cells_checked": len(rows),
        "paired_day_week_comparisons_checked": comparisons, "native_quote_rows_checked": source_rows,
        "training_categories": training_categories,
        "frozen_timed_and_CLOCK_reference_cells_checked": frozen_references_checked,
        "missing_seconds_preserved": source_gaps, "declaration_sha256": sha(decl_path),
        "scope": "immutable_cached44_lineage_common_clock_original_training_region_paths_conditional_categories_independent_ridge_target_matrix_model_hashes_frozen_timed_reference_identity_issuance_native_paths_R_PF_day_week_CIs_required_conjunctions_joint8_Holm",
        "not_independently_recomputed": ["43_original_feature_formulas", "intrinsic_feature_validity",
            "qualification_gate_logic", "drawdown_and_thirds"],
        "fresh_out_of_sample": False, "known_history_adaptive": True,
        "broker_execution": "NOT TESTED", "measured_historical_costs": "NOT TESTED",
        "prospective_paper": "NOT TESTED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FOLDER / "independent_audit.json"))
    args = parser.parse_args(); output = Path(args.output).resolve(); output.relative_to(ROOT)
    if output.exists() or output.with_suffix(".sha256").exists(): raise ValueError("Exclusive audit output")
    audit, detail = Audit(), {}
    try: detail = verify(audit)
    except Exception as error: audit.errors.append(f"{type(error).__name__}: {error}")
    report = {"stage": "independent_conditional_region_reward_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not audit.errors, "checks": audit.checks, "errors": audit.errors,
        "input_sha256": audit.pins, "verifier_sha256": sha(__file__),
        "model_atol": MODEL_ATOL, "model_rtol": MODEL_RTOL,
        "safety": dict.fromkeys(FLAGS, False), "QUALIFIED": False, **detail}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f: f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with output.with_suffix(".sha256").open("x") as f: f.write(sha(output) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__": sys.exit(main())
