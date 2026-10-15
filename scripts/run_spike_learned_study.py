#!/usr/bin/env python3
"""Frozen causal ridge payoff replication; public saved prices, never orders."""
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
from app.research.learned_signal import causal_inputs, fit_ridge, predict_ridge, FEATURE_FORMULAS
from app.research.learned_metrics import profit_factor_inference
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.payoff_metrics import summarize, paired_inference
from app.research.spike_hunter import assert_offline, load_m1, partitions
from scripts.run_spike_payoff_study import window, holm
from scripts.run_spike_timed_study import weekly_inference, tail_metrics, thirds

SYMBOLS = ("BOOM500", "CRASH500")
MODES = ("SPIKE", "DRIFT")
TRANSFER = {"BOOM500": "BOOM300N", "CRASH500": "CRASH300N"}
SOURCES = ("docs/SPIKE_LEARNED_PROTOCOL.md", "backend/app/research/learned_signal.py",
           "backend/app/research/learned_metrics.py", "backend/app/research/spike_hunter.py",
           "backend/app/research/payoff.py", "backend/app/research/payoff_timed.py",
           "backend/app/research/payoff_metrics.py", "scripts/run_spike_payoff_study.py",
           "scripts/run_spike_timed_study.py", "scripts/run_spike_learned_study.py")
CFG = TimedExitConfig(stop_atr=2.0, max_hold_minutes=15)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=str) + "\n")


def hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def audited_source(folder, symbol, original=False):
    suffix = "_m1_clean.csv" if original else "_m1_180d_clean.csv"
    path = ROOT / folder / f"{symbol.lower()}{suffix}"
    value = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
    if not original:
        manifest = path.parent / f"{symbol}_normalization_manifest.json"
        document = json.loads(manifest.read_text())
        if value["sha256"] != document["normalized_sha256"]:
            raise ValueError(f"Changed acquisition input: {symbol}")
        value.update(rows=document["normalized_rows"], first_epoch=document["first_epoch"],
                     last_epoch=document["last_epoch"], manifest=str(manifest.relative_to(ROOT)))
    return value


def prepare(path, symbol):
    m1, audit = load_m1(path)
    direction = "boom" if symbol.startswith("BOOM") else "crash"
    m5, names = causal_inputs(m1, direction)
    issue_time = m5.index + pd.Timedelta(minutes=5)
    fixed_clock = (issue_time.minute % 30 == 0) & (issue_time.second == 0)
    rows = m5.loc[m5.feature_valid & fixed_clock].copy()
    rows.index = issue_time[m5.feature_valid & fixed_clock]
    rows.index.name = "signal_time"
    split = partitions(m1)
    split.update({"train40": (m1.index[0], split["wf1"][0]),
                  "train50": (m1.index[0], split["wf2"][0]),
                  "train60": (m1.index[0], split["wf3"][0])})
    audit.update(causal_input_rows=len(m5), eligible_clock_rows=len(rows),
                 clock="UTC close minute00/30; finite causal inputs only")
    return m1, rows, names, audit, split


def issued(rows, symbol, mode, model=None):
    if mode not in MODES:
        raise ValueError("Unknown mode")
    chosen = rows
    scores = None
    if model is not None:
        scores = predict_ridge(rows[model["feature_names"]], model)
        mask = (scores >= model["threshold"]) & (scores > 0)
        chosen = rows.loc[mask]
        scores = scores[mask]
    side = (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)
    result = [{"signal_time": time, "atr": float(row.atr), "side": side,
               "signal_close": float(row.close),
               "variant": f"LEARNED_{mode}" if model else f"CLOCK_{mode}"}
              for time, row in chosen.iterrows()]
    if scores is not None:
        for row, score in zip(result, scores, strict=True):
            row["score"] = float(score)
    return result


def fit_at(rows, labels, names, start, end):
    past = window(labels, start, end)
    known = past.loc[~past.censored & np.isfinite(past.net_R)].set_index("signal_time")
    inputs = rows.loc[known.index, names]
    target = known.net_R.to_numpy(float)
    model = fit_ridge(inputs, target, names, penalty=.1, quantile=.75)
    training_bytes = (np.ascontiguousarray(inputs.to_numpy(float)).tobytes() +
                      np.ascontiguousarray(target).tobytes() +
                      np.ascontiguousarray(known.index.asi8).tobytes())
    model.update(training_start=start, training_end=end, label_purge_minutes=31,
                 matrix_and_target_sha256=hashlib.sha256(training_bytes).hexdigest(),
                 training_completed_labels=len(known),
                 training_censored_labels=int(past.censored.sum()),
                 training_invalid_labels=len(past) - len(known) - int(past.censored.sum()),
                 training_latest_issue=known.index.max(),
                 training_latest_planned_end=known.planned_end.max())
    return model


def development_eligible(candidate):
    folds = candidate["walk_forward"]
    m = candidate["validation"]
    identities = [(f.get("training"), f.get("test")) for f in folds]
    return (identities == [("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")] and
            m["completed"] >= 500 and m["active_days"] >= 30 and
            m["censored"] == 0 and m["invalid_uncensored"] == 0 and
            m["selection_score"] is not None and m["selection_score"] > 0 and
            all(f["metrics"]["completed"] >= 100 and f["metrics"]["censored"] == 0 and
                f["metrics"]["invalid_uncensored"] == 0 and
                (f["metrics"]["mean_net_R"] or -np.inf) > 0 and
                (f["metrics"]["profit_factor"] or 0) > 1 for f in folds))


def develop(args):
    result = {"stage": "frozen_learned_development", "run_utc": datetime.now(timezone.utc).isoformat(),
              "safety": assert_offline(), "code_hashes": hashes(), "config": asdict(CFG),
              "feature_formulas": FEATURE_FORMULAS, "adaptive_round": 3,
              "user_target": {"net_profit_factor": 1.5, "min_completed_per_symbol_model": 1000},
              "sources": {}, "symbols": {}}
    for symbol in SYMBOLS:
        original = audited_source("data/spike_hunter", symbol, original=True)
        older = audited_source("data/spike_payoff_external", symbol)
        transfer = audited_source("data/spike_learned_transfer", TRANSFER[symbol])
        result["sources"][symbol] = {"original": original, "reused_older": older, "transfer": transfer}
        m1, rows, names, audit, split = prepare(ROOT / original["path"], symbol)
        candidates = {}
        for mode in MODES:
            labels, label_audit = replay_timed(m1, issued(rows, symbol, mode), CFG,
                                              *split["development"], purge_minutes=31)
            validation_ledgers, folds = [], []
            for training, test in (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")):
                model = fit_at(rows, labels, names, *split[training])
                trades, execution = replay_timed(m1, issued(rows, symbol, mode, model), CFG,
                                                *split[test], purge_minutes=31)
                base = window(labels, *split[test])
                folds.append({"training": training, "test": test, "model": model,
                              "metrics": summarize(trades, *split[test], stop_atr=CFG.stop_atr),
                              "baseline": summarize(base, *split[test], stop_atr=CFG.stop_atr),
                              "audit": execution})
                validation_ledgers.append(trades)
            start, end = split["wf1"][0], split["wf3"][1]
            combined = pd.concat(validation_ledgers, ignore_index=True)
            candidate = {"mode": mode, "config": asdict(CFG), "walk_forward": folds,
                         "validation": summarize(combined, start, end, stop_atr=CFG.stop_atr),
                         "training_label_audit": label_audit,
                         "final_model": fit_at(rows, labels, names, *split["development"])}
            candidate["development_eligible"] = development_eligible(candidate)
            candidates[mode] = candidate
            combined.to_csv(args.output / f"{symbol}_{mode}_validation_trades.csv", index=False)
            print(f"{symbol}/{mode}: validation n={candidate['validation']['completed']} "
                  f"PF={candidate['validation']['profit_factor']} "
                  f"eligible={candidate['development_eligible']}", flush=True)
        pool = [c for c in candidates.values() if c["development_eligible"]]
        selected = (sorted(pool, key=lambda c: (-c["validation"]["selection_score"], c["mode"]))[0]["mode"]
                    if pool else None)
        result["symbols"][symbol] = {"audit": audit, "partitions": split, "models": candidates,
                                     "selected_direction": selected, "eligible_directions": len(pool)}
    save(args.output / "selection.json", result)
    fingerprint = digest(args.output / "selection.json")
    (args.output / "selection.sha256").write_text(fingerprint + "\n")
    print(f"LEARNED_SELECTION_FROZEN {fingerprint}; transfer features/payoff NOT EVALUATED", flush=True)


def evaluate_one(m1, rows, symbol, mode, model, start, end, bootstrap):
    signals = issued(rows, symbol, mode, model)
    trades, audit = replay_timed(m1, signals, CFG, start, end, purge_minutes=31)
    baseline, base_audit = replay_timed(m1, issued(rows, symbol, mode), CFG,
                                      start, end, purge_minutes=31)
    metric = summarize(trades, start, end, stop_atr=CFG.stop_atr)
    metric.update(paired_inference(trades, baseline, start, end, repeats=bootstrap, seed=20261005))
    metric["day_p"] = metric["p"]
    metric.update(weekly_inference(trades, baseline, start, end, bootstrap))
    metric["p"] = max(metric["day_p"], metric["weekly_p"])
    metric.update(profit_factor_inference(trades, start, end, repeats=bootstrap, seed=20261005))
    row = {"mode": mode, "config": asdict(CFG), "metrics": metric, "audit": audit,
           "baseline": summarize(baseline, start, end, stop_atr=CFG.stop_atr),
           "baseline_audit": base_audit,
           "tail": tail_metrics(trades, start, end, CFG.stop_atr, CFG.round_trip_cost_atr),
           "thirds": thirds(trades, start, end, CFG.stop_atr)}
    sensitivities = []
    for fill, delay in (("adverse_extreme", 1), ("barrier_proxy", 1), ("adverse_extreme", 0)):
        zero_cfg = TimedExitConfig(**{**asdict(CFG), "fill_mode": fill, "entry_delay_minutes": delay,
                                     "round_trip_cost_atr": 0.0})
        gross, _ = replay_timed(m1, signals, zero_cfg, start, end, purge_minutes=31)
        for cost in (0., .025, .05, .10, .20, .40):
            adjusted = gross.copy()
            adjusted["net_R"] = adjusted.gross_R - cost / zero_cfg.stop_atr
            sensitivities.append({"fill_mode": fill, "entry_delay_minutes": delay,
                                  "round_trip_cost_atr": cost,
                                  **summarize(adjusted, start, end, stop_atr=zero_cfg.stop_atr)})
    row["sensitivities"] = sensitivities
    signal_frame = pd.DataFrame(signals, columns=["signal_time", "atr", "side", "signal_close", "variant", "score"])
    if not signal_frame.empty:
        signal_frame = signal_frame.loc[(signal_frame.signal_time >= start) & (signal_frame.signal_time < end)]
    return row, trades, signal_frame


def gate(row, eligible):
    m = row["metrics"]
    observed = m["completed"] >= 1000 and (m["profit_factor"] or 0) >= 1.5
    def lower(key, threshold=0):
        value = m[key][0]
        return value is not None and value > threshold
    double_cost = next(s for s in row["sensitivities"] if s["fill_mode"] == "adverse_extreme" and
                       s["entry_delay_minutes"] == 1 and s["round_trip_cost_atr"] == .20)
    checks = [(observed, "user_pf1_5_n1000_not_met"), (eligible, "development_rejected"),
              (m["censored"] == 0 and m["invalid_uncensored"] == 0 and row["audit"]["missing_entry"] == 0,
              "missing_or_censored_outcomes"), (m["active_days"] >= 60, "under60_active_days"),
              (m["day_undefined_replicates"] == 0 and m["weekly_undefined_replicates"] == 0,
               "undefined_pf_bootstrap_ratios"),
              (lower("day_profit_factor_ci95", 1) and lower("weekly_profit_factor_ci95", 1), "pf_ci_not_above1"),
              (lower("mean_net_R_ci95") and lower("weekly_mean_net_R_ci95"), "mean_ci_not_positive"),
              (lower("baseline_difference_ci95") and lower("weekly_difference_ci95"), "no_clock_advantage"),
              (m["holm_p"] < .05, "multiplicity_adjusted_not_significant"),
              (all(t["metrics"]["completed"] >= 200 and (t["metrics"]["mean_net_R"] or -np.inf) > 0
                   for t in row["thirds"]), "chronological_thirds_unstable"),
              (not m["equity_ruin"] and m["closed_trade_max_drawdown"] <= .10, "drawdown_over10pct_or_ruin"),
              ((double_cost["mean_net_R"] or -np.inf) > 0, "doubled_cost_not_positive")]
    reasons = [reason for passed, reason in checks if not passed]
    row.update(user_target_observed=observed, historical_candidate=not reasons,
               supports_expected_pf_1_5=(not reasons and all(m[key][0] is not None and m[key][0] >= 1.5
                       for key in ("day_profit_factor_ci95", "weekly_profit_factor_ci95"))),
               rejection_reasons=reasons, actual_money_profit="NOT TESTED", prospective_validation="NOT TESTED")


def evaluate(args):
    selected_path = args.output / "selection.json"
    frozen_hash = (args.output / "selection.sha256").read_text().strip()
    if digest(selected_path) != frozen_hash:
        raise ValueError("Frozen learned selection changed")
    selection = json.loads(selected_path.read_text())
    if selection["code_hashes"] != hashes():
        raise ValueError("Frozen learned protocol/source changed; new declaration required")
    # Verify every source before preparing any evaluation features.
    for source in selection["sources"].values():
        for value in source.values():
            if digest(ROOT / value["path"]) != value["sha256"]:
                raise ValueError("Frozen price source changed")
    result = {"stage": "learned_evaluation", "run_utc": datetime.now(timezone.utc).isoformat(),
              "selection_sha256": frozen_hash, "code_hashes": hashes(), "safety": assert_offline(),
              "symbols": {}, "forward_test": False, "actual_money_profit": "NOT TESTED"}
    transfer_family = []
    flat = []
    for symbol in SYMBOLS:
        payload = {"selected_direction": selection["symbols"][symbol]["selected_direction"], "cohorts": {}}
        for cohort, kind, eval_symbol in (("reused_final30", "original", symbol),
                                          ("reused_older180", "reused_older", symbol),
                                          ("cross_symbol_transfer", "transfer", TRANSFER[symbol])):
            source = selection["sources"][symbol][kind]
            m1, rows, names, audit, split = prepare(ROOT / source["path"], eval_symbol)
            start, end = (split["final_test"] if kind == "original" else
                          (m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)))
            models = {}
            for mode in MODES:
                candidate = selection["symbols"][symbol]["models"][mode]
                row, trades, signal_frame = evaluate_one(m1, rows, eval_symbol, mode,
                                                         candidate["final_model"], start, end, args.bootstrap)
                row["development_eligible"] = candidate["development_eligible"]
                row["selected_direction"] = mode == payload["selected_direction"]
                models[mode] = row
                trades.to_csv(args.output / f"{eval_symbol}_{cohort}_{mode}_trades.csv", index=False)
                signal_frame.to_csv(args.output / f"{eval_symbol}_{cohort}_{mode}_signals.csv", index=False)
                if kind == "transfer":
                    transfer_family.append(row)
                print(f"{eval_symbol}/{cohort}/{mode}: n={row['metrics']['completed']} "
                      f"PF={row['metrics']['profit_factor']}", flush=True)
            payload["cohorts"][cohort] = {"symbol": eval_symbol, "start": start, "end": end,
                                         "source": source, "audit": audit, "models": models}
        result["symbols"][symbol] = payload
    holm(transfer_family)
    for symbol, payload in result["symbols"].items():
        for cohort, data in payload["cohorts"].items():
            for mode, row in data["models"].items():
                if cohort == "cross_symbol_transfer":
                    gate(row, row["development_eligible"])
                else:
                    row.update(user_target_observed=(row["metrics"]["completed"] >= 1000 and
                               (row["metrics"]["profit_factor"] or 0) >= 1.5),
                               historical_candidate=False, supports_expected_pf_1_5=False,
                               rejection_reasons=["reused_exploratory_cohort"])
                flat.append({"training_symbol": symbol, "evaluation_symbol": data["symbol"],
                             "cohort": cohort, "mode": mode, "user_target_observed": row["user_target_observed"],
                             "historical_candidate": row["historical_candidate"], **row["metrics"]})
    save(args.output / "results.json", result)
    pd.DataFrame(flat).to_csv(args.output / "metrics.csv", index=False)
    print("LEARNED_EVALUATION_COMPLETE; execution remains disabled", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_learned_20261005")
    parser.add_argument("--bootstrap", type=int, default=9999)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    assert_offline()
    if args.stage == "develop" and (args.output / "selection.json").exists():
        raise ValueError("Refusing to overwrite frozen selection")
    if args.stage == "evaluate" and (args.output / "results.json").exists():
        raise ValueError("Refusing to overwrite evaluation; declare a separate output")
    (develop if args.stage == "develop" else evaluate)(args)


if __name__ == "__main__":
    main()
