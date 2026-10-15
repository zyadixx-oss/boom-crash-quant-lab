#!/usr/bin/env python3
"""Fixed nonlinear native600 chronological research; no trading or refitting OOS."""
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
from app.research.learned_signal import FEATURE_NAMES as BASE_NAMES, fit_ridge, predict_ridge
from app.research.nonlinear_signal import fit_histogram_boost, predict_histogram_boost
from app.research.learned_metrics import profit_factor_inference
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.payoff_metrics import summarize, paired_inference
from app.research.spike_hunter import assert_offline, load_m1, partitions
from app.research.multiframe_signal import causal_multiframe_inputs
from scripts.run_spike_payoff_study import window, holm
from scripts.run_spike_timed_study import weekly_inference, tail_metrics, thirds
from scripts.run_spike_learned_study import development_eligible, gate, audited_source
from scripts.run_spike_native300_study import validate_intervals, verify_prices, check_prepared
from scripts.run_spike_multiframe_study import FEATURE_FORMULAS, SOURCES as INHERITED_SOURCES

SYMBOLS = ("BOOM600", "CRASH600")
MODES = ("SPIKE", "DRIFT")
FAMILIES = ("BOOST44", "RIDGE44", "RIDGE19")
WALK_FORWARD = (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3"))
CFG = TimedExitConfig(stop_atr=2.0, max_hold_minutes=15)
BOOST_PARAMS = {"n_trees": 100, "learning_rate": .05, "max_depth": 3,
                "min_leaf": 200, "n_bins": 16, "leaf_regularization": 20., "quantile": .75}
BOOTSTRAP = 9999
SOURCES = (*INHERITED_SOURCES, "backend/app/research/nonlinear_signal.py",
           "backend/tests/test_nonlinear_signal.py", "docs/SPIKE_NONLINEAR_PROTOCOL.md",
           "scripts/run_spike_nonlinear_study.py", "backend/tests/test_spike_nonlinear_study.py",
           "scripts/collect_spike_history.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def guard(output, stage):
    if (output / "results.json").exists():
        raise ValueError("Refusing post-result overwrite or development")
    if stage == "develop" and any((output / name).exists() for name in
                                   ("selection.json", "selection.sha256", "declaration.json", "declaration.sha256")):
        raise ValueError("Refusing to overwrite the frozen nonlinear declaration/selection")


def source(folder, symbol):
    """Metadata-only600 source audit, preserving proven recovered rate limits."""
    if symbol not in SYMBOLS:
        raise ValueError("Nonlinear study requires the declared600 symbols")
    value = audited_source(folder, symbol)
    manifest_path = ROOT / value["manifest"]
    document = json.loads(manifest_path.read_text())
    if document.get("symbol") != symbol or not all(document.get(key) is True for key in
                                                   ("normalization_valid", "finite_positive_ohlc", "canonical_utc_minute_grid")):
        raise ValueError("600 source normalization audit failed")
    flags = assert_offline()
    if (document.get("authentication_used") is not False or document.get("fills_or_interpolations") is not False
            or any(document.get("safety", {}).get(flag) is not False for flag in flags)):
        raise ValueError("600 source must remain public, uninterpolated and offline")
    page_path = manifest_path.parent / f"{symbol}_page_audit.json"
    pages = json.loads(page_path.read_text())
    if len(pages) != document["page_count"]:
        raise ValueError("600 public page count changed")
    request_ends = {}
    expected_end = value["last_epoch"] + 59
    for page in pages:
        request = page["request"]
        if (set(request) != {"ticks_history", "count", "end", "style", "granularity", "req_id"} or
                request.get("ticks_history") != symbol or request.get("style") != "candles" or
                request.get("granularity") != 60 or request.get("count") != 1000 or
                not 0 < page["returned_rows"] <= 1000 or len(page["response_sha256"]) != 64 or
                any(c not in "0123456789abcdef" for c in page["response_sha256"]) or
                request["end"] != expected_end or
                not page["oldest_epoch"] <= page["newest_epoch"] <= request["end"]):
            raise ValueError("600 successful-page audit invalid")
        request_ends[request["end"]] = request_ends.get(request["end"], 0) + 1
        expected_end = page["oldest_epoch"] - 1
    if not pages or pages[-1]["oldest_epoch"] > value["first_epoch"]:
        raise ValueError("600 public pagination did not reach the declared start")
    errors = document.get("request_errors", [])
    for error in errors:
        try:
            payload = json.loads(error["error"].split(": ", 1)[1])
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError("600 unrecognized request error") from exc
        if not isinstance(payload, dict) or payload.get("code") != "RateLimit" or request_ends.get(error["end"]) != 1:
            raise ValueError("600 request error has no proven successful retry")
    first, last = value["first_epoch"], value["last_epoch"]
    if (first % 60 or last % 60 or first > last or document["requested_start_epoch"] != first or
            document["requested_cutoff_exclusive_epoch"] != last + 60):
        raise ValueError("600 source endpoints differ from the requested UTC interval")
    gaps = document.get("gaps", [])
    missing = sum(int(item["missing_bars"]) for item in gaps)
    expected = (last + 60 - first) // 60
    if value["rows"] + missing != expected:
        raise ValueError("600 clean rows and declared gaps do not cover the requested grid")
    if digest(ROOT / document["raw_file"]) != document["raw_sha256"]:
        raise ValueError("600 raw source changed")
    value.update(manifest_sha256=digest(manifest_path), expected_grid_rows=expected,
                 declared_missing_minutes=missing, declared_gaps=gaps,
                 excluded_raw_rows=document.get("excluded_rows", []),
                 raw_file=document["raw_file"], raw_sha256=document["raw_sha256"],
                 recovered_request_errors=errors, page_audit=str(page_path.relative_to(ROOT)),
                 page_audit_sha256=digest(page_path), from_utc=str(pd.Timestamp(first, unit="s", tz="UTC")),
                 to_exclusive_utc=str(pd.Timestamp(last + 60, unit="s", tz="UTC")))
    return value


def verify_sources(sources):
    verify_prices(sources)
    for pair in sources.values():
        for value in pair.values():
            if digest(ROOT / value["raw_file"]) != value["raw_sha256"]:
                raise ValueError("Frozen600 raw source changed")
            if digest(ROOT / value["page_audit"]) != value["page_audit_sha256"]:
                raise ValueError("Frozen600 successful-page audit changed")


def prepare(value, symbol, development=False):
    """Clip older last30 before any feature calculation during development."""
    full_m1, audit = load_m1(ROOT / value["path"])
    check_prepared(full_m1, audit, value)
    split = partitions(full_m1)
    split.update({"train40": (full_m1.index[0], split["wf1"][0]),
                  "train50": (full_m1.index[0], split["wf2"][0]),
                  "train60": (full_m1.index[0], split["wf3"][0])})
    m1 = full_m1.loc[full_m1.index < split["development"][1]].copy() if development else full_m1
    direction = "boom" if symbol.startswith("BOOM") else "crash"
    m5, names = causal_multiframe_inputs(m1, direction)
    issues = m5.index + pd.Timedelta(minutes=5)
    mask = m5.feature_valid & (issues.minute % 30 == 0) & (issues.second == 0)
    rows = m5.loc[mask].copy()
    rows.index, rows.index.name = issues[mask], "signal_time"
    audit.update(causal_input_rows=len(m5), eligible_clock_rows=len(rows), feature_count=len(names),
                 frames=["H4", "H1", "M15", "M5", "M1"], clock="UTC close minute00/30",
                 feature_source_rows=len(m1), feature_end_exclusive=m1.index[-1] + pd.Timedelta(minutes=1),
                 development_features_only=development)
    return m1, rows, names, audit, split


def feature_order(family, names):
    if family not in FAMILIES or len(names) != 44 or list(names[:19]) != list(BASE_NAMES):
        raise ValueError("Expected declared estimator family and exact base19 prefix of44 inputs")
    return list(BASE_NAMES) if family == "RIDGE19" else list(names)


def predict(rows, model):
    names = model["feature_names"]
    X = rows[names]
    if model["family"] == "BOOST44":
        return predict_histogram_boost(X, model["estimator"], feature_names=names)
    if model["family"] in ("RIDGE44", "RIDGE19"):
        return predict_ridge(X, model["estimator"], feature_names=names)
    raise ValueError("Unknown frozen estimator family")


def issued(rows, symbol, mode, model=None):
    if mode not in MODES:
        raise ValueError("Unknown direction")
    scores = predict(rows, model) if model is not None else None
    chosen = rows
    if scores is not None:
        cutoff = model["estimator"]["threshold"]
        mask = (scores >= cutoff) & (scores > 0)
        chosen, scores = rows.loc[mask], scores[mask]
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    variant = f"NONLINEAR_{model['family']}_{mode}" if model else f"CLOCK_{mode}"
    result = [{"signal_time": time, "atr": float(row.atr), "side": side,
               "signal_close": float(row.close), "variant": variant}
              for time, row in chosen.iterrows()]
    if scores is not None:
        for item, score in zip(result, scores, strict=True):
            item["score"] = float(score)
    return result


def fit_at(rows, labels, names, family, start, end):
    names = feature_order(family, names)
    past = window(labels, start, end)
    known = past.loc[~past.censored & np.isfinite(past.net_R)].set_index("signal_time")
    if len(known) < 1000:
        raise ValueError("Every estimator requires1000 completed training labels")
    X, y = rows.loc[known.index, names], known.net_R.to_numpy(float)
    estimator = (fit_histogram_boost(X, y, names, **BOOST_PARAMS) if family == "BOOST44"
                 else fit_ridge(X, y, names, penalty=.1, quantile=.75))
    serialized = (np.ascontiguousarray(X.to_numpy(float)).tobytes() +
                  np.ascontiguousarray(y).tobytes() + np.ascontiguousarray(known.index.asi8).tobytes())
    return {"family": family, "feature_names": names, "estimator": estimator,
            "training_start": start, "training_end": end, "label_purge_minutes": 31,
            "matrix_and_target_sha256": hashlib.sha256(serialized).hexdigest(),
            "training_completed_labels": len(known), "training_censored_labels": int(past.censored.sum()),
            "training_invalid_labels": len(past) - len(known) - int(past.censored.sum()),
            "training_latest_issue": known.index.max(), "training_latest_planned_end": known.planned_end.max()}


def choose(candidates):
    eligible = [(key, row) for key, row in candidates.items()
                if row["development_eligible"] and row["family"] in ("BOOST44", "RIDGE44")]
    return (sorted(eligible, key=lambda pair: (-pair[1]["validation"]["selection_score"], pair[0]))[0][0]
            if eligible else None)


def develop(args):
    guard(args.output, "develop")
    safety = assert_offline()
    sources, intervals = {}, {}
    for symbol in SYMBOLS:
        pair = {"old": source("data/spike_nonlinear600_old", symbol),
                "fresh": source("data/spike_nonlinear600_recent", symbol)}
        intervals[symbol] = validate_intervals(pair["old"], pair["fresh"])
        sources[symbol] = pair
    code = hashes()
    verify_sources(sources)
    declaration = {"stage": "nonlinear_predevelopment_declaration", "adaptive_round": 6,
                   "run_utc": datetime.now(timezone.utc).isoformat(), "safety": safety,
                   "code_hashes": code, "sources": sources, "intervals": intervals,
                   "config": asdict(CFG), "feature_formulas": FEATURE_FORMULAS,
                   "families": list(FAMILIES), "boost_parameters": BOOST_PARAMS,
                   "joint_fresh_hypotheses": 12, "bootstrap_repeats": BOOTSTRAP,
                   "fresh_features_evaluated": False, "fresh_payoffs_evaluated": False,
                   "prospective_paper": False, "cached_unverified_boom600_claim": True,
                   "user_target": {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000}}
    save(args.output / "declaration.json", declaration)
    declaration_hash = digest(args.output / "declaration.json")
    (args.output / "declaration.sha256").write_text(declaration_hash + "\n")
    selection = {**declaration, "stage": "frozen_nonlinear_development",
                 "declaration_sha256": declaration_hash, "symbols": {}}
    for symbol in SYMBOLS:
        value = sources[symbol]["old"]
        m1, rows, names, audit, split = prepare(value, symbol, development=True)
        candidates = {}
        for mode in MODES:
            labels, label_audit = replay_timed(m1, issued(rows, symbol, mode), CFG,
                                              *split["development"], purge_minutes=31)
            for family in FAMILIES:
                folds, ledgers = [], []
                for training, test in WALK_FORWARD:
                    model = fit_at(rows, labels, names, family, *split[training])
                    trades, execution = replay_timed(m1, issued(rows, symbol, mode, model), CFG,
                                                     *split[test], purge_minutes=31)
                    folds.append({"training": training, "test": test, "model": model,
                                  "metrics": summarize(trades, *split[test], stop_atr=CFG.stop_atr),
                                  "baseline": summarize(window(labels, *split[test]), *split[test], stop_atr=CFG.stop_atr),
                                  "audit": execution})
                    ledgers.append(trades)
                combined = pd.concat(ledgers, ignore_index=True)
                candidate = {"family": family, "mode": mode, "config": asdict(CFG),
                             "walk_forward": folds,
                             "validation": summarize(combined, split["wf1"][0], split["wf3"][1], stop_atr=CFG.stop_atr),
                             "training_label_audit": label_audit,
                             "final_model": fit_at(rows, labels, names, family, *split["development"])}
                candidate["development_eligible"] = development_eligible(candidate)
                key = f"{family}_{mode}"
                candidates[key] = candidate
                combined.to_csv(args.output / f"{symbol}_{key}_validation_trades.csv", index=False)
                print(f"{symbol}/{key}: validation n={candidate['validation']['completed']} "
                      f"PF={candidate['validation']['profit_factor']} eligible={candidate['development_eligible']}", flush=True)
        selection["symbols"][symbol] = {"audit": audit, "partitions": split, "models": candidates,
                                         "selected_model": choose(candidates)}
    if hashes() != code:
        raise ValueError("Nonlinear code changed during development")
    verify_sources(sources)
    selection["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output / "selection.json", selection)
    fingerprint = digest(args.output / "selection.json")
    (args.output / "selection.sha256").write_text(fingerprint + "\n")
    print(f"NONLINEAR_SELECTION_FROZEN {fingerprint}; later features/payoffs not read", flush=True)


def frozen(output):
    selection = json.loads((output / "selection.json").read_text())
    fingerprint = digest(output / "selection.json")
    if fingerprint != (output / "selection.sha256").read_text().strip():
        raise ValueError("Frozen nonlinear selection changed")
    declaration = json.loads((output / "declaration.json").read_text())
    declared_hash = digest(output / "declaration.json")
    if declared_hash != (output / "declaration.sha256").read_text().strip() or declared_hash != selection["declaration_sha256"]:
        raise ValueError("Frozen nonlinear declaration changed")
    for key in declaration:
        if key != "stage" and selection[key] != declaration[key]:
            raise ValueError(f"Nonlinear selection diverges from declaration: {key}")
    if selection["code_hashes"] != hashes():
        raise ValueError("Frozen nonlinear source/protocol changed")
    if (selection["config"] != asdict(CFG) or selection["boost_parameters"] != BOOST_PARAMS or
            selection["families"] != list(FAMILIES) or selection["joint_fresh_hypotheses"] != 12):
        raise ValueError("Declared nonlinear fixed configuration changed")
    for pair in selection["sources"].values():
        validate_intervals(pair["old"], pair["fresh"])
    verify_sources(selection["sources"])
    return selection, fingerprint


def evaluate_model(m1, rows, symbol, mode, model, start, end, baseline, baseline_audit):
    signals = issued(rows, symbol, mode, model)
    trades, audit = replay_timed(m1, signals, CFG, start, end, purge_minutes=31)
    metric = summarize(trades, start, end, stop_atr=CFG.stop_atr)
    metric.update(paired_inference(trades, baseline, start, end, repeats=BOOTSTRAP, seed=20261005))
    metric["day_p"] = metric["p"]
    metric.update(weekly_inference(trades, baseline, start, end, BOOTSTRAP))
    metric["clock_conjunction_p"] = max(metric["day_p"], metric["weekly_p"])
    metric["p"] = metric["clock_conjunction_p"]
    metric.update(profit_factor_inference(trades, start, end, repeats=BOOTSTRAP, seed=20261005))
    row = {"family": model["family"], "mode": mode, "config": asdict(CFG), "metrics": metric,
           "audit": audit, "baseline": summarize(baseline, start, end, stop_atr=CFG.stop_atr),
           "baseline_audit": baseline_audit,
           "tail": tail_metrics(trades, start, end, CFG.stop_atr, CFG.round_trip_cost_atr),
           "thirds": thirds(trades, start, end, CFG.stop_atr), "sensitivities": []}
    for fill, delay in (("adverse_extreme", 1), ("barrier_proxy", 1), ("adverse_extreme", 0)):
        zero_cfg = TimedExitConfig(**{**asdict(CFG), "fill_mode": fill, "entry_delay_minutes": delay,
                                     "round_trip_cost_atr": 0.})
        gross, _ = replay_timed(m1, signals, zero_cfg, start, end, purge_minutes=31)
        for cost in (0., .025, .05, .10, .20, .40):
            adjusted = gross.copy()
            adjusted["net_R"] = adjusted.gross_R - cost / zero_cfg.stop_atr
            row["sensitivities"].append({"fill_mode": fill, "entry_delay_minutes": delay,
                                         "round_trip_cost_atr": cost,
                                         **summarize(adjusted, start, end, stop_atr=zero_cfg.stop_atr)})
    signal_frame = pd.DataFrame(signals, columns=["signal_time", "atr", "side", "signal_close", "variant", "score"])
    return row, trades, signal_frame


def compare_references(rows, ledgers, mode, start, end):
    target = rows[f"BOOST44_{mode}"]
    target["linear_references"] = {}
    for family in ("RIDGE44", "RIDGE19"):
        key = f"{family}_{mode}"
        day = paired_inference(ledgers[f"BOOST44_{mode}"], ledgers[key], start, end,
                               repeats=BOOTSTRAP, seed=20261005)
        week = weekly_inference(ledgers[f"BOOST44_{mode}"], ledgers[key], start, end, BOOTSTRAP)
        target["linear_references"][family] = {"day_inference": day, "weekly_inference": week,
                                                "metrics": rows[key]["metrics"], "audit": rows[key]["audit"],
                                                "comparison_basis": "per-completed-trade mean R on same44-input available clock"}
        target["metrics"]["p"] = max(target["metrics"]["p"], day["p"], week["weekly_p"])


def nonlinear_gate(row):
    gate(row, row["development_eligible"])
    row["satisfies_requested_timeframes"] = row["family"] in ("BOOST44", "RIDGE44")
    if row["family"] == "RIDGE19":
        row["rejection_reasons"].append("m5_reference_not_combined_strategy")
        row["historical_candidate"] = False
        row["supports_expected_pf_1_5"] = False
        return
    if row["family"] != "BOOST44":
        return
    references = row.get("linear_references", {})
    if set(references) != {"RIDGE44", "RIDGE19"}:
        row["rejection_reasons"].append("missing_or_undeclared_linear_reference_family")
    for family in ("RIDGE44", "RIDGE19"):
        if family not in references:
            continue
        reference = references[family]
        day = reference["day_inference"]["baseline_difference_ci95"][0]
        week = reference["weekly_inference"]["weekly_difference_ci95"][0]
        if day is None or week is None or not np.isfinite(day) or not np.isfinite(week) or day <= 0 or week <= 0:
            row["rejection_reasons"].append(f"no_positive_day_week_excess_over_{family}")
        metric = reference["metrics"]
        if metric["censored"] or metric["invalid_uncensored"] or reference["audit"]["missing_entry"]:
            row["rejection_reasons"].append(f"missing_or_censored_{family}_reference")
    row["historical_candidate"] = not row["rejection_reasons"]
    row["supports_expected_pf_1_5"] = row["supports_expected_pf_1_5"] and row["historical_candidate"]
    row["nonlinear_added_value"] = row["historical_candidate"]


def evaluate(args):
    guard(args.output, "evaluate")
    safety = assert_offline()
    if args.bootstrap != BOOTSTRAP:
        raise ValueError("Nonlinear primary inference requires9999 bootstrap replicates")
    selection, fingerprint = frozen(args.output)
    result = {"stage": "nonlinear_chronological_evaluation", "adaptive_round": 6,
              "run_utc": datetime.now(timezone.utc).isoformat(), "selection_sha256": fingerprint,
              "declaration_sha256": selection["declaration_sha256"], "code_hashes": selection["code_hashes"],
              "safety": safety, "symbols": {}, "joint_fresh_hypotheses": 12,
              "prospective_paper": False, "forward_test": False,
              "chronological_old_to_new": True, "methodology_adaptive": True, "actual_money_profit": "NOT TESTED"}
    family = []
    for symbol in SYMBOLS:
        payload = {"selected_model": selection["symbols"][symbol]["selected_model"], "cohorts": {}}
        for cohort, kind in (("old_final30", "old"), ("later_temporal180", "fresh")):
            value = selection["sources"][symbol][kind]
            m1, opportunities, names, audit, split = prepare(value, symbol)
            start, end = (split["final_test"] if kind == "old" else
                          (m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)))
            opportunities = opportunities.loc[(opportunities.index >= start) & (opportunities.index < end)]
            models, ledgers = {}, {}
            for mode in MODES:
                baseline, baseline_audit = replay_timed(m1, issued(opportunities, symbol, mode), CFG,
                                                       start, end, purge_minutes=31)
                for estimator_family in FAMILIES:
                    key = f"{estimator_family}_{mode}"
                    candidate = selection["symbols"][symbol]["models"][key]
                    model = candidate["final_model"]
                    if model["feature_names"] != feature_order(estimator_family, names):
                        raise ValueError("Nonlinear frozen feature order changed")
                    row, trades, signals = evaluate_model(m1, opportunities, symbol, mode, model,
                                                         start, end, baseline, baseline_audit)
                    row.update(development_eligible=candidate["development_eligible"],
                               selected_model=key == payload["selected_model"])
                    models[key], ledgers[key] = row, trades
                    prefix = f"{symbol}_{cohort}_{key}"
                    trades.to_csv(args.output / f"{prefix}_trades.csv", index=False)
                    signals.to_csv(args.output / f"{prefix}_signals.csv", index=False)
                    if kind == "fresh":
                        family.append(row)
                    print(f"{symbol}/{cohort}/{key}: n={row['metrics']['completed']} "
                          f"PF={row['metrics']['profit_factor']}", flush=True)
                compare_references(models, ledgers, mode, start, end)
            payload["cohorts"][cohort] = {"symbol": symbol, "start": start, "end": end,
                                         "source": value, "audit": audit, "eligible_clock_rows": len(opportunities),
                                         "models": models}
        result["symbols"][symbol] = payload
    if len(family) != 12:
        raise ValueError("All12 declared later hypotheses must enter joint inference")
    holm(family)
    flat = []
    for symbol, payload in result["symbols"].items():
        for cohort, value in payload["cohorts"].items():
            for key, row in value["models"].items():
                if cohort == "later_temporal180":
                    nonlinear_gate(row)
                else:
                    row.update(user_target_observed=(row["metrics"]["completed"] >= 1000 and
                               (row["metrics"]["profit_factor"] or 0) >= 1.5), historical_candidate=False,
                               supports_expected_pf_1_5=False, rejection_reasons=["secondary_old_holdout"],
                               actual_money_profit="NOT TESTED", prospective_validation="NOT TESTED")
                flat.append({"symbol": symbol, "cohort": cohort, "model": key,
                             "historical_candidate": row["historical_candidate"], **row["metrics"]})
    _, final_hash = frozen(args.output)
    if final_hash != fingerprint:
        raise ValueError("Nonlinear declaration changed during evaluation")
    save(args.output / "results.json", result)
    pd.DataFrame(flat).to_csv(args.output / "metrics.csv", index=False)
    print("NONLINEAR_EVALUATION_COMPLETE; no refitting, new selection or enabled trading", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_nonlinear_20261005")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    assert_offline()
    args.output.mkdir(parents=True, exist_ok=True)
    (develop if args.stage == "develop" else evaluate)(args)


if __name__ == "__main__":
    main()
