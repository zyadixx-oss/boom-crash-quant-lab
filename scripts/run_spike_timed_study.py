#!/usr/bin/env python3
"""Adaptive, frozen two-stage uncapped quote-path research; never places orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.payoff_metrics import summarize, paired_inference
from app.research.spike_hunter import VARIANTS, assert_offline
from scripts.run_spike_payoff_study import prepare, window, holm

SYMBOLS = ("BOOM500", "CRASH500")
PARTS = ("train40", "train50", "train60", "development", "wf1", "wf2", "wf3")
SOURCES = ("docs/SPIKE_TIMED_PROTOCOL.md", "backend/app/research/spike_hunter.py",
           "backend/app/research/payoff.py", "backend/app/research/payoff_metrics.py",
           "backend/app/research/payoff_timed.py", "scripts/run_spike_payoff_study.py",
           "scripts/run_spike_timed_study.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def config(stop=2.0, hold=15, **overrides):
    return TimedExitConfig(stop_atr=stop, max_hold_minutes=hold, **overrides)


def identifier(variant, mode, cfg):
    return f"{mode}::{variant}__sl{cfg.stop_atr:g}_h{cfg.max_hold_minutes}"


def issued_for(issued, variant, mode):
    if mode not in ("SPIKE", "DRIFT"):
        raise ValueError("Unknown candidate direction")
    multiplier = 1 if mode == "SPIKE" else -1
    return [{**row, "side": multiplier * row["side"], "variant": f"{mode}::{variant}"}
            for row in issued[variant]]


def adequate(row, part):
    m = row["parts"][part]
    return (m["completed"] >= 300 and m["active_days"] >= 30 and
            m["censored"] == 0 and m["invalid_uncensored"] == 0 and
            m["selection_score"] is not None)


def stable(row):
    return all(row["parts"][part]["completed"] >= 30 and
               row["parts"][part]["censored"] == 0 and
               row["parts"][part]["invalid_uncensored"] == 0 and
               row["parts"][part]["mean_net_R"] is not None and
               row["parts"][part]["mean_net_R"] > 0 for part in ("wf1", "wf2", "wf3"))


def choose(rows, training, robust=False):
    enough = [r for r in rows if adequate(r, training)]
    eligible = [r for r in enough if not robust or stable(r)]
    pool = eligible or enough
    if not pool:
        return None
    picked = sorted(pool, key=lambda r: (-r["parts"][training]["selection_score"], r["id"]))[0]
    return {k: picked[k] for k in ("id", "variant", "mode", "config", "parts")} | {
        "development_eligible": bool(eligible), "training_partition": training,
        "selection_score": picked["parts"][training]["selection_score"]}


def audited_sources(folder, symbols):
    result = {}
    for symbol in symbols:
        path = ROOT / folder / f"{symbol.lower()}_m1_180d_clean.csv"
        manifest = json.loads((path.parent / f"{symbol}_normalization_manifest.json").read_text())
        source_hash = digest(path)
        if source_hash != manifest["normalized_sha256"]:
            raise ValueError("Transfer source changed from audited acquisition")
        result[symbol] = {"path": str(path.relative_to(ROOT)), "sha256": source_hash,
                          "rows": manifest["normalized_rows"], "first_epoch": manifest["first_epoch"],
                          "last_epoch": manifest["last_epoch"]}
    return result


def transfer_sources():
    return audited_sources("data/spike_timed_transfer", ("BOOM1000", "CRASH1000"))


def develop(args):
    result = {"stage": "frozen_timed_development", "run_utc": datetime.now(timezone.utc).isoformat(),
              "safety": assert_offline(), "code_hashes": hashes(), "transfer_sources": transfer_sources(),
              "reused_older_sources": audited_sources("data/spike_payoff_external", SYMBOLS),
              "adaptive_round": True, "symbols": {}}
    flat = []
    for symbol in SYMBOLS:
        m1, audit, issued, split = prepare(ROOT / "data/spike_hunter" / f"{symbol.lower()}_m1_clean.csv", symbol)
        rows = []
        for ordinal, (variant, mode, stop, hold) in enumerate(itertools.product(
                VARIANTS, ("SPIKE", "DRIFT"), (1.0, 2.0, 4.0), (1, 5, 15, 30)), 1):
            cfg = config(stop, hold)
            trades, execution = replay_timed(m1, issued_for(issued, variant, mode), cfg,
                                            *split["development"], purge_minutes=31)
            metrics = {part: summarize(window(trades, *split[part]), *split[part], stop_atr=stop)
                       for part in PARTS}
            row = {"id": identifier(variant, mode, cfg), "variant": variant, "mode": mode,
                   "config": asdict(cfg), "parts": metrics, "audit": execution}
            rows.append(row)
            flat.extend({"symbol": symbol, "candidate": row["id"], "partition": part, **m}
                        for part, m in metrics.items())
            if ordinal % 48 == 0:
                print(f"{symbol}: {ordinal}/384 development configurations", flush=True)
        primary = next(r for r in rows if r["id"] == identifier("CRT", "SPIKE", config()))
        folds = []
        for train, test in (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")):
            picked = choose(rows, train)
            folds.append({"train": train, "test": test, "id": None if picked is None else picked["id"],
                          "development_eligible": False if picked is None else picked["development_eligible"],
                          "validation": None if picked is None else picked["parts"][test]})
        selected = choose(rows, "development", robust=True)
        result["symbols"][symbol] = {"audit": audit, "partitions": split, "primary": primary,
                                     "selected": selected, "walk_forward": folds, "candidate_count": len(rows),
                                     "fully_eligible_count": sum(adequate(r, "development") and stable(r) for r in rows)}
        save(args.output / f"{symbol}_development_grid.json", rows)
    pd.DataFrame(flat).to_csv(args.output / "development_grid.csv", index=False)
    save(args.output / "selection.json", result)
    fingerprint = digest(args.output / "selection.json")
    (args.output / "selection.sha256").write_text(fingerprint + "\n")
    print(f"TIMED_SELECTION_FROZEN {fingerprint}; transfer payoffs not evaluated", flush=True)


def completed(trades):
    return trades.loc[~trades.censored & np.isfinite(trades.net_R)].copy()


def tail_metrics(trades, start, end, stop_atr, cost_atr):
    done = completed(window(trades, start, end))
    values = done.net_R.to_numpy(float)
    n = len(values)
    if not n:
        return {"tail_status": "NO_COMPLETED_TRADES", "worst_leave_one_day_out_mean_R": None}
    positive = np.sort(values[values > 0])[::-1]
    positive_sum = float(positive.sum())
    daily = done.assign(day=done.entry_time.dt.floor("D")).groupby("day").net_R.agg(["sum", "count"])
    positive_days = np.sort(daily.loc[daily["sum"] > 0, "sum"].to_numpy(float))[::-1]
    day_positive_sum = float(positive_days.sum())
    total = float(values.sum())
    loo = [(total - row["sum"]) / (n - row["count"]) for _, row in daily.iterrows() if n > row["count"]]
    best_days = daily.sort_values("sum", ascending=False).head(5)
    remaining_n = n - int(best_days["count"].sum())
    kept = np.sort(values)[:-5] if n > 5 else np.array([])
    raw = done.gross_R.to_numpy(float) * done.atr.to_numpy(float) * stop_atr
    lower = float(np.quantile(values, .05))
    out = {"tail_status": "AVAILABLE", "raw_mean_directional_quote_points": float(raw.mean()),
           "raw_mean_after_modeled_cost_points": float((raw - cost_atr * done.atr.to_numpy(float)).mean()),
           "signal_atr_quantiles25_50_75": np.quantile(done.atr.to_numpy(float), [.25, .5, .75]).tolist(),
           "winner_effective_n": positive_sum ** 2 / float(np.sum(positive ** 2)) if len(positive) else None,
           "largest_winner_R": float(positive[0]) if len(positive) else None,
           "largest_winner_positive_profit_share": float(positive[0]) / positive_sum if len(positive) else None,
           "largest_day_positive_day_profit_share": float(positive_days[0]) / day_positive_sum if len(positive_days) else None,
           "mean_R_without_best5_trades": float(kept.mean()) if len(kept) else None,
           "mean_R_without_best5_days": (total - float(best_days["sum"].sum())) / remaining_n if remaining_n else None,
           "worst_leave_one_day_out_mean_R": min(loo) if loo else None,
           "worst_trade_R": float(values.min()), "lower5pct_R": lower,
           "mean_lower5pct_R": float(values[values <= lower].mean())}
    for fraction in (.01, .05):
        label = str(int(100 * fraction))
        out[f"top{label}pct_trades_positive_profit_share"] = (
            float(positive[:math.ceil(n * fraction)].sum()) / positive_sum if positive_sum else None)
        out[f"top{label}pct_active_days_positive_day_profit_share"] = (
            float(positive_days[:math.ceil(len(daily) * fraction)].sum()) / day_positive_sum if day_positive_sum else None)
    return out


def weekly_inference(trades, baseline, start, end, repeats, seed=20261005):
    days = pd.date_range(start.floor("D"), (end - pd.Timedelta(nanoseconds=1)).floor("D"), freq="D")
    k = len(days)
    def daily(frame):
        frame = completed(frame)
        ids = ((frame.entry_time.dt.floor("D") - days[0]) // pd.Timedelta(days=1)).to_numpy(int)
        if ((ids < 0) | (ids >= k)).any():
            raise ValueError("Entry falls outside inference days")
        return (np.bincount(ids, minlength=k).astype(float),
                np.bincount(ids, weights=frame.net_R, minlength=k).astype(float))
    mn, ms = daily(trades)
    bn, bs = daily(baseline)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, k, size=(repeats, math.ceil(k / 7)))
    indices = ((starts[:, :, None] + np.arange(7)) % k).reshape(repeats, -1)[:, :k]
    mcounts, bcounts = mn[indices].sum(axis=1), bn[indices].sum(axis=1)
    m = np.divide(ms[indices].sum(axis=1), mcounts, out=np.full(repeats, np.nan), where=mcounts > 0)
    b = np.divide(bs[indices].sum(axis=1), bcounts, out=np.full(repeats, np.nan), where=bcounts > 0)
    diffs = m - b
    mean = float(ms.sum() / mn.sum()) if mn.sum() else None
    difference = mean - float(bs.sum() / bn.sum()) if mean is not None and bn.sum() else None
    def ci(values):
        finite = values[np.isfinite(values)]
        return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
    def p(values, estimate):
        if estimate is None or estimate <= 0:
            return 1.0
        return (int((~np.isfinite(values) | (values - estimate >= estimate)).sum()) + 1) / (repeats + 1)
    return {"weekly_mean_net_R_ci95": ci(m), "weekly_difference_ci95": ci(diffs),
            "weekly_p": max(p(m, mean), p(diffs, difference)), "weekly_block_days": 7,
            "weekly_bootstrap_repeats": repeats, "weekly_bootstrap_seed": seed,
            "weekly_valid_mean_replicates": int(np.isfinite(m).sum()),
            "weekly_valid_difference_replicates": int(np.isfinite(diffs).sum())}


def thirds(trades, start, end, stop_atr):
    duration = end - start
    return [{"start": start + duration * i / 3, "end": start + duration * (i + 1) / 3,
             "metrics": summarize(window(trades, start + duration * i / 3, start + duration * (i + 1) / 3),
                                  start + duration * i / 3, start + duration * (i + 1) / 3, stop_atr=stop_atr)}
            for i in range(3)]


def evaluate_one(m1, issued, picked, start, end, bootstrap):
    cfg = TimedExitConfig(**picked["config"])
    trades, audit = replay_timed(m1, issued_for(issued, picked["variant"], picked["mode"]), cfg,
                                start, end, purge_minutes=31)
    base, base_audit = replay_timed(m1, issued_for(issued, "CLOCK_BASELINE", picked["mode"]), cfg,
                                  start, end, purge_minutes=31)
    m = summarize(trades, start, end, stop_atr=cfg.stop_atr)
    m.update(paired_inference(trades, base, start, end, repeats=bootstrap, seed=20261005))
    m["day_p"] = m["p"]
    m.update(weekly_inference(trades, base, start, end, bootstrap))
    m["p"] = max(m["day_p"], m["weekly_p"])
    row = {"id": picked["id"], "variant": picked["variant"], "mode": picked["mode"], "config": picked["config"],
           "metrics": m, "audit": audit, "baseline": summarize(base, start, end, stop_atr=cfg.stop_atr),
           "baseline_audit": base_audit, "tail": tail_metrics(trades, start, end, cfg.stop_atr, cfg.round_trip_cost_atr),
           "baseline_tail": tail_metrics(base, start, end, cfg.stop_atr, cfg.round_trip_cost_atr),
           "thirds": thirds(trades, start, end, cfg.stop_atr)}
    sensitivities = []
    for fill, delay in (("adverse_extreme", 1), ("barrier_proxy", 1), ("adverse_extreme", 0)):
        zero_cfg = TimedExitConfig(**{**picked["config"], "fill_mode": fill, "entry_delay_minutes": delay,
                                     "round_trip_cost_atr": 0.0})
        gross, _ = replay_timed(m1, issued_for(issued, picked["variant"], picked["mode"]), zero_cfg,
                                start, end, purge_minutes=31)
        for cost in (0., .025, .05, .10, .20, .40):
            adjusted = gross.copy()
            adjusted["net_R"] = adjusted.gross_R - cost / zero_cfg.stop_atr
            sensitivities.append({"fill_mode": fill, "entry_delay_minutes": delay, "round_trip_cost_atr": cost,
                                  **summarize(adjusted, start, end, stop_atr=zero_cfg.stop_atr)})
    row["sensitivity"] = sensitivities
    return row, trades


def gate(row, picked, source_days):
    m = row["metrics"]
    def lower_positive(field):
        return m[field][0] is not None and m[field][0] > 0
    checks = [
        (adequate(picked, "development") and stable(picked) and picked.get("development_eligible", True), "development_rejected"),
        (source_days >= 170, "source_under170days"), (m["completed"] >= 300, "sample_under300"),
        (m["active_days"] >= 30, "days_under30"), (m["censored"] == 0 and m["invalid_uncensored"] == 0, "unknown_trades"),
        (m["mean_net_R"] is not None and m["mean_net_R"] >= .10, "expectancy_under0.10R"),
        (m["profit_factor"] is not None and m["profit_factor"] >= 1.30, "PF_under1.30"),
        (all(lower_positive(f) for f in ("mean_net_R_ci95", "baseline_difference_ci95",
                                       "weekly_mean_net_R_ci95", "weekly_difference_ci95")), "CI_not_positive"),
        (m["holm_p"] < .05, "holm_not_significant"),
        (m["closed_trade_max_drawdown"] <= .10 and not m["equity_ruin"], "drawdown_or_ruin"),
        (all(t["metrics"]["completed"] >= 30 and t["metrics"]["mean_net_R"] is not None and
             t["metrics"]["mean_net_R"] > 0 for t in row["thirds"]), "transfer_thirds_not_stable"),
        (row["tail"]["worst_leave_one_day_out_mean_R"] is not None and
         row["tail"]["worst_leave_one_day_out_mean_R"] > 0, "single_day_fragility")]
    reasons = [reason for passed, reason in checks if not passed]
    return not reasons, reasons


def evaluate(args):
    path = args.output / "selection.json"
    fingerprint = digest(path)
    if fingerprint != (args.output / "selection.sha256").read_text().strip():
        raise ValueError("Selection hash changed")
    frozen = json.loads(path.read_text())
    if (frozen["code_hashes"] != hashes() or frozen["transfer_sources"] != transfer_sources()
            or frozen["reused_older_sources"] != audited_sources("data/spike_payoff_external", SYMBOLS)):
        raise ValueError("Source or transfer data changed after selection")
    result = {"stage": "locked_uncapped_timed_evaluation", "adaptive_round": True,
              "run_utc": datetime.now(timezone.utc).isoformat(), "safety": assert_offline(),
              "selection_sha256": fingerprint, "code_hashes": hashes(), "symbols": {},
              "actual_fills": "NOT TESTED", "monetary_net_profit": "NOT TESTED", "forward": "NOT TESTED",
              "multiple_testing_scope": "within this adaptive round only; not all project searches"}
    inference_rows = []
    for source_symbol in SYMBOLS:
        picks = {"primary": frozen["symbols"][source_symbol]["primary"],
                 "selected": frozen["symbols"][source_symbol]["selected"]}
        transfer_symbol = source_symbol.replace("500", "1000")
        sym = {"development": frozen["symbols"][source_symbol], "cohorts": {}}
        for cohort, target in (("reused_recent30", source_symbol), ("reused_older500", source_symbol),
                               ("cross_symbol_replication", transfer_symbol)):
            if cohort == "reused_recent30":
                source = ROOT / "data/spike_hunter" / f"{target.lower()}_m1_clean.csv"
            elif cohort == "reused_older500":
                source = ROOT / "data/spike_payoff_external" / f"{target.lower()}_m1_180d_clean.csv"
            else:
                source = ROOT / frozen["transfer_sources"][target]["path"]
            m1, audit, issued, split = prepare(source, target)
            if cohort == "reused_recent30" and audit["sha256"] != frozen["symbols"][source_symbol]["audit"]["sha256"]:
                raise ValueError("Development source changed")
            start, end = (split["final_test"] if cohort == "reused_recent30"
                          else (m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)))
            c = {"symbol": target, "audit": audit, "start": start, "end": end,
                 "source_days": (end - start).total_seconds() / 86400, "models": {}}
            evaluated = {}
            for role, picked in picks.items():
                if picked is None:
                    c["models"][role] = None
                    continue
                if picked["id"] in evaluated:
                    c["models"][role] = evaluated[picked["id"]]
                    continue
                print(f"{target} {cohort} locked {role}: {picked['id']}", flush=True)
                row, trades = evaluate_one(m1, issued, picked, start, end, args.bootstrap)
                trades.to_csv(args.output / f"{target}_{cohort}_{role}_trades.csv", index=False)
                c["models"][role] = row
                evaluated[picked["id"]] = row
                if cohort == "cross_symbol_replication":
                    inference_rows.append(row)
            sym["cohorts"][cohort] = c
        result["symbols"][source_symbol] = sym
    holm(inference_rows)
    flat = []
    for source_symbol, sym in result["symbols"].items():
        c = sym["cohorts"]["cross_symbol_replication"]
        for role, row in list(c["models"].items()):
            if row is None:
                continue
            passed, reasons = gate(row, frozen["symbols"][source_symbol][role], c["source_days"])
            c["models"][role] = {**row, "prospective_paper_candidate": passed, "gate_reasons": reasons}
        for cohort, c in sym["cohorts"].items():
            for role, row in c["models"].items():
                if row is not None:
                    flat.append({"training_symbol": source_symbol, "evaluation_symbol": c["symbol"],
                                 "cohort": cohort, "role": role, "candidate": row["id"],
                                 **{k: v for k, v in row["metrics"].items() if not isinstance(v, (list, dict))}})
    save(args.output / "results.json", result)
    pd.DataFrame(flat).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "REPORT.ar.md").write_text(markdown(result), encoding="utf-8")
    print("TIMED_EVALUATION_COMPLETE; all trading flags false", flush=True)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(result):
    lines = ["# اختبار خروج زمني دون سقف ربح — الجولة البحثية الثانية", "",
             "محاكاة فعلية لمسار الأسعار؛ تكلفة0.10ATR افتراضية، دخول بعد دقيقة، وقف عند أسوأ سعر الدقيقة.",
             "الربح المالي والتنفيذ الحقيقي والاختبار الأمامي: **NOT TESTED**. الأعلام الأربعة false.",
             "اختبار1000 نقل لقواعد اختيرت على500، ضمن فترة تاريخية أخرى معروفة على500؛ لا يثبت استقلال مولّد المؤشرات.",
             "هذه جولة بحث تكيفية؛ تصحيح Holm داخل الجولة فقط، ولا يثبت احتمالية اكتشاف عالمية عبر جميع المحاولات.", "",
             f"بصمة الاختيار قبل التقييم: `{result['selection_sha256']}`.", ""]
    for symbol, sym in result["symbols"].items():
        dev = sym["development"]
        selected = dev["selected"]
        lines += [f"## قواعد {symbol}", "",
                  f"الإعدادات المؤهلة في التطوير: {dev['fully_eligible_count']}/384. الاختيار: `{None if selected is None else selected['id']}`؛ أهلية: {False if selected is None else selected['development_eligible']}.", "",
                  "| مؤشر الاختبار | الفترة | الدور | n | متوسط R | CI95 يومي | CI95 أسبوعي | PF | تراجع مغلق % |",
                  "|---|---|---|---:|---:|---|---|---:|---:|"]
        for cohort, c in sym["cohorts"].items():
            for role, row in c["models"].items():
                if row is None:
                    continue
                m = row["metrics"]
                lines.append(f"| {c['symbol']} | {cohort} | {role} | {m['completed']} | {fmt(m['mean_net_R'])} | {fmt(m['mean_net_R_ci95'][0])}–{fmt(m['mean_net_R_ci95'][1])} | {fmt(m['weekly_mean_net_R_ci95'][0])}–{fmt(m['weekly_mean_net_R_ci95'][1])} | {fmt(m['profit_factor'])} | {fmt(100*m['closed_trade_max_drawdown'],2)} |")
        lines += ["", "التحقق المتحرك؛ اختير كل إعداد من الماضي قبل فترة التحقق:", "",
                  "| الفترة | الاختيار | n | متوسط R |", "|---|---|---:|---:|"]
        for fold in dev["walk_forward"]:
            m = fold["validation"]
            lines.append(f"| {fold['test']} | {fold['id']} | {0 if m is None else m['completed']} | {fmt(None if m is None else m['mean_net_R'])} |")
        lines += ["", "قرار نقل القواعد:", ""]
        for role, row in sym["cohorts"]["cross_symbol_replication"]["models"].items():
            if row is None:
                lines.append(f"- {role}: لا يوجد اختيار بعينة كافية.")
                continue
            m, tail = row["metrics"], row["tail"]
            lines.append(f"- {role}: {'PASS للبحث الورقي فقط' if row['prospective_paper_candidate'] else 'FAIL'}؛ Holm p={fmt(m['holm_p'],5)}؛ الفرق عن التوقيت غير المشروط={fmt(m['baseline_difference'])}R؛ أسوأ حذف يوم={fmt(tail['worst_leave_one_day_out_mean_R'])}R؛ أسوأ صفقة={fmt(tail.get('worst_trade_R'))}R.")
            if row["gate_reasons"]:
                lines.append("  أسباب الرفض: " + ", ".join(row["gate_reasons"]) + ".")
        lines.append("")
    lines += ["## حدود", "",
              "- الفترة الأقدم الجديدة لـ500 غير متاحة كاملة:6170 دقيقة فقط. لم تُقدّم كاختبار مستقل كافٍ.",
              "- نتائج500 المعاد استخدامها استكشافية. نتائج1000 تحقق عبر مؤشرات مختلفة، وليست Forward أو ملء صفقات.",
              "- متوسط R يتأثر بـATR وحجم المخاطرة؛ نقاط السعر الخام وتوزيع ATR والذيل محفوظة لتمييز هذا الأثر.",
              "- وقف الدقيقة والخروج عند إغلاق المدة افتراضات؛ تراجع الصفقات المغلقة لا يشمل كل المخاطرة داخل الصفقة.",
              "- تفاصيل الأثلاث، حذف أفضل5 صفقات/أيام، تركّز الربح وحساسية التكاليف والتنفيذ محفوظة في results.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_timed_20261005")
    parser.add_argument("--bootstrap", type=int, default=9999)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    assert_offline()
    args.output.mkdir(parents=True, exist_ok=True)
    (develop if args.stage == "develop" else evaluate)(args)


if __name__ == "__main__":
    main()
