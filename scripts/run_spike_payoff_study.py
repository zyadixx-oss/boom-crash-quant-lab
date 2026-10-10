#!/usr/bin/env python3
"""Frozen two-stage quote-path payoff research. Never connects to an account."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.research.spike_hunter import VARIANTS, assert_offline, features, load_m1, partitions, signals
from app.research.payoff import BracketConfig, replay_brackets
from app.research.payoff_metrics import summarize, paired_inference

SYMBOLS = ("BOOM500", "CRASH500")
PARTS = ("train40", "train50", "train60", "development", "wf1", "wf2", "wf3")
SOURCE_FILES = ("docs/SPIKE_PAYOFF_PROTOCOL.md", "backend/app/research/spike_hunter.py",
                "backend/app/research/payoff.py", "backend/app/research/payoff_metrics.py",
                "scripts/run_spike_payoff_study.py")


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def prepare(path, symbol):
    m1, audit = load_m1(path)
    direction = "boom" if symbol.startswith("BOOM") else "crash"
    m5 = features(m1, direction)
    index = signals(m5, direction)
    clock, last = [], None
    for i in np.flatnonzero(m5.feature_valid.to_numpy(bool)):
        time = m5.index[i] + pd.Timedelta(minutes=5)
        if last is None or time - last >= pd.Timedelta(minutes=30):
            clock.append(i)
            last = time
    index["CLOCK_BASELINE"] = np.asarray(clock, dtype=int)
    issued = {variant: [{"signal_time": m5.index[i] + pd.Timedelta(minutes=5),
                        "atr": float(m5.atr.iloc[i]), "signal_close": float(m5.close.iloc[i]),
                        "side": 1 if direction == "boom" else -1, "variant": variant}
                       for i in positions] for variant, positions in index.items()}
    split = partitions(m1)
    split.update({"train40": (m1.index[0], split["wf1"][0]),
                  "train50": (m1.index[0], split["wf2"][0]),
                  "train60": (m1.index[0], split["wf3"][0])})
    return m1, audit, issued, split


def configuration(stop=1.0, target=2.0, hold=15, **overrides):
    return BracketConfig(stop_atr=stop, target_atr=target, max_hold_minutes=hold,
                         **overrides)


def key(variant, config):
    return f"{variant}__sl{config.stop_atr:g}_tp{config.target_atr:g}_h{config.max_hold_minutes}"


def window(trades, start, end):
    if trades.empty:
        return trades
    return trades.loc[(trades.signal_time >= start) &
                      (trades.signal_time + pd.Timedelta(minutes=31) <= end)].copy()


def choose(rows, training, robust=False):
    def enough(row):
        r = row["parts"][training]
        base = r["completed"] >= 300 and r["active_days"] >= 30 and r["censored"] == 0
        if robust:
            base = base and all(row["parts"][p]["completed"] >= 30 and
                                (row["parts"][p]["mean_net_R"] or -np.inf) > 0 and
                                row["parts"][p]["censored"] == 0 for p in ("wf1", "wf2", "wf3"))
        return base
    candidates = [r for r in rows if enough(r)]
    eligible = bool(candidates)
    pool = candidates or [r for r in rows if r["parts"][training]["selection_score"] is not None]
    if not pool:
        raise ValueError("No candidate has a finite development score")
    selected = sorted(pool, key=lambda r: (-r["parts"][training]["selection_score"], r["id"]))[0]
    return {"id": selected["id"], "variant": selected["variant"], "config": selected["config"],
            "development_eligible": eligible, "training_partition": training,
            "selection_score": selected["parts"][training]["selection_score"],
            "parts": selected["parts"]}


def develop(args):
    result = {"stage": "frozen_development_selection", "run_utc": datetime.now(timezone.utc).isoformat(),
              "safety": assert_offline(), "code_hashes": hashes(), "symbols": {}}
    flat = []
    for symbol in SYMBOLS:
        print(f"{symbol}: loading old development data", flush=True)
        path = ROOT / "data/spike_hunter" / f"{symbol.lower()}_m1_clean.csv"
        m1, audit, issued, split = prepare(path, symbol)
        rows = []
        combinations = itertools.product(VARIANTS, (0.5, 1.0, 1.5), (1.0, 2.0, 3.0), (5, 15, 30))
        for ordinal, (variant, stop, target, hold) in enumerate(combinations, 1):
            config = configuration(stop, target, hold)
            trades, replay_audit = replay_brackets(m1, issued[variant], config, *split["development"], purge_minutes=31)
            metrics = {part: summarize(window(trades, *split[part]), *split[part], stop_atr=stop) for part in PARTS}
            row = {"id": key(variant, config), "variant": variant, "config": asdict(config),
                   "parts": metrics, "replay_audit": replay_audit}
            rows.append(row)
            for part, metric in metrics.items():
                flat.append({"symbol": symbol, "candidate": row["id"], "partition": part, **metric})
            if ordinal % 54 == 0:
                print(f"{symbol}: {ordinal}/432 development configurations", flush=True)
        primary_cfg = configuration()
        primary = next(r for r in rows if r["id"] == key("SR_ALIGNMENT", primary_cfg))
        wf = []
        for train, test in (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")):
            selected = choose(rows, train)
            wf.append({"train": train, "test": test, "id": selected["id"],
                       "training_score": selected["selection_score"],
                       "development_eligible": selected["development_eligible"],
                       "validation": selected["parts"][test]})
        result["symbols"][symbol] = {"audit": audit, "partitions": split,
                                     "primary": primary, "selected": choose(rows, "development", robust=True),
                                     "walk_forward": wf, "candidate_count": len(rows)}
        save(args.output / f"{symbol}_development_grid.json", rows)
    pd.DataFrame(flat).to_csv(args.output / "development_grid.csv", index=False)
    save(args.output / "selection.json", result)
    digest = hashlib.sha256((args.output / "selection.json").read_bytes()).hexdigest()
    (args.output / "selection.sha256").write_text(digest + "\n")
    print(f"SELECTION_FROZEN sha256={digest}; external payoff data not evaluated", flush=True)


def evaluate_config(m1, issued, variant, config, start, end, bootstrap):
    trades, audit = replay_brackets(m1, issued[variant], config, start, end, purge_minutes=31)
    base, base_audit = replay_brackets(m1, issued["CLOCK_BASELINE"], config, start, end, purge_minutes=31)
    metric = summarize(trades, start, end, stop_atr=config.stop_atr)
    baseline = summarize(base, start, end, stop_atr=config.stop_atr)
    metric.update(paired_inference(trades, base, start, end, repeats=bootstrap))
    return {"metrics": metric, "audit": audit, "baseline": baseline, "baseline_audit": base_audit}, trades


def holm(rows):
    ordered = sorted(rows, key=lambda r: r["metrics"]["p"])
    running = 0.0
    for i, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * row["metrics"]["p"]))
        row["metrics"]["holm_p"] = running


def gate(row, selection):
    m = row["metrics"]
    reasons = []
    checks = [(m["censored"] == 0, "censored_trades"), (m["completed"] >= 300, "sample_under_300"),
              (selection.get("development_eligible", True), "development_selection_rejected"),
              (m["active_days"] >= 30, "active_days_under_30"),
              ((m["mean_net_R"] or -np.inf) >= .10, "expectancy_under_0.10R"),
              ((m["profit_factor"] or 0) >= 1.30, "profit_factor_under_1.30"),
              ((m["mean_net_R_ci95"][0] or -np.inf) > 0, "mean_ci_not_positive"),
              ((m["baseline_difference_ci95"][0] or -np.inf) > 0, "baseline_difference_ci_not_positive"),
              (m["holm_p"] < .05, "holm_not_significant"),
              (m["closed_trade_max_drawdown"] is not None and m["closed_trade_max_drawdown"] <= .10, "drawdown_over_10pct"),
              (all(selection["parts"][p]["completed"] >= 30 and
                   (selection["parts"][p]["mean_net_R"] or -np.inf) > 0 for p in ("wf1", "wf2", "wf3")), "development_folds_not_stable")]
    reasons.extend(reason for passed, reason in checks if not passed)
    return not reasons, reasons


def evaluate(args):
    selection_path = args.output / "selection.json"
    digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    if digest != (args.output / "selection.sha256").read_text().strip():
        raise ValueError("Frozen selection hash mismatch")
    selection = json.loads(selection_path.read_text())
    if selection["code_hashes"] != hashes():
        raise ValueError("Research code or protocol changed after development selection")
    result = {"stage": "locked_historical_payoff_evaluation", "run_utc": datetime.now(timezone.utc).isoformat(),
              "safety": assert_offline(), "selection_sha256": digest, "code_hashes": hashes(),
              "monetary_net_profit": "NOT TESTED", "actual_fills": "NOT TESTED", "forward": "NOT TESTED", "symbols": {}}
    external_rows, flat = [], []
    for symbol in SYMBOLS:
        frozen = selection["symbols"][symbol]
        chosen = {"primary": frozen["primary"], "selected": frozen["selected"]}
        sym = {"development_selection": frozen["selected"], "walk_forward": frozen["walk_forward"], "cohorts": {}}
        for cohort in ("reused_oos_exploratory", "external_older_replication"):
            path = (ROOT / "data/spike_hunter" / f"{symbol.lower()}_m1_clean.csv" if cohort.startswith("reused")
                    else ROOT / "data/spike_payoff_external" / f"{symbol.lower()}_m1_180d_clean.csv")
            m1, audit, issued, split = prepare(path, symbol)
            if cohort.startswith("reused"):
                if audit["sha256"] != frozen["audit"]["sha256"]:
                    raise ValueError("Old data changed after selection")
                start, end = split["final_test"]
            else:
                start, end = m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)
                if end > pd.Timestamp(frozen["audit"]["from_utc"]):
                    raise ValueError("External and old data overlap")
            c = {"audit": audit, "start": start, "end": end, "models": {}}
            by_id = {}
            for role, picked in chosen.items():
                if picked["id"] in by_id:
                    c["models"][role] = by_id[picked["id"]]
                    continue
                config = BracketConfig(**picked["config"])
                print(f"{symbol} {cohort}: locked {role} {picked['id']}", flush=True)
                row, trades = evaluate_config(m1, issued, picked["variant"], config, start, end, args.bootstrap)
                row.update(id=picked["id"], variant=picked["variant"], config=picked["config"])
                trades.to_csv(args.output / f"{symbol}_{cohort}_{role}_trades.csv", index=False)
                if cohort == "external_older_replication":
                    external_rows.append(row)
                sensitivities = []
                for fill_mode, policy, delay in (("adverse_extreme", "stop_first", 1),
                                                ("barrier_proxy", "stop_first", 1),
                                                ("barrier_proxy", "target_first", 1),
                                                ("adverse_extreme", "stop_first", 0)):
                    # Costs alter arithmetic only, never triggers, chronology or selection.
                    zero_cfg = BracketConfig(**{**picked["config"], "round_trip_cost_atr": 0.0,
                                                "fill_mode": fill_mode, "same_bar_policy": policy,
                                                "entry_delay_minutes": delay})
                    gross, stress_audit = replay_brackets(m1, issued[picked["variant"]], zero_cfg, start, end, purge_minutes=31)
                    for cost in (0.0, .025, .05, .10, .20, .40):
                        adjusted = gross.copy()
                        adjusted["net_R"] = adjusted.gross_R - cost / zero_cfg.stop_atr
                        metrics = summarize(adjusted, start, end, stop_atr=zero_cfg.stop_atr)
                        sensitivities.append({"fill_mode": fill_mode, "same_bar_policy": policy, "entry_delay_minutes": delay,
                                              "round_trip_cost_atr": cost, **metrics})
                row["sensitivity"] = sensitivities
                c["models"][role] = row
                by_id[picked["id"]] = row
            sym["cohorts"][cohort] = c
        result["symbols"][symbol] = sym
    holm(external_rows)
    for symbol, sym in result["symbols"].items():
        models = sym["cohorts"]["external_older_replication"]["models"]
        for role, row in list(models.items()):
            passed, reasons = gate(row, selection["symbols"][symbol][role])
            # Shared statistical hypothesis, but independent role eligibility.
            models[role] = {**row, "forward_research_candidate": passed, "gate_reasons": reasons}
        for cohort, c in sym["cohorts"].items():
            for role, row in c["models"].items():
                flat.append({"symbol": symbol, "cohort": cohort, "role": role, "candidate": row["id"],
                             **{k: v for k, v in row["metrics"].items() if not isinstance(v, (list, dict))}})
    save(args.output / "results.json", result)
    pd.DataFrame(flat).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "REPORT.ar.md").write_text(markdown(result), encoding="utf-8")
    print("EVALUATION_COMPLETE: quote-path hypotheses only; all trading flags false", flush=True)


def fmt(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def markdown(result):
    lines = ["# اختبار العائد — نتائج محاكاة فعلية على الأسعار العامة", "",
             "هذه محاكاة لمسار أسعار المؤشر بوحدات R بعد تكلفة افتراضية، وليست أرباح حساب منفّذة.",
             "**الربح المالي الصافي والتنفيذ الحقيقي والاختبار الأمامي: NOT TESTED.**",
             "الأعلام الأربعة false. مصدر القواعد: `docs/SPIKE_PAYOFF_PROTOCOL.md`.", "",
             f"تجميد الاختيار SHA256: `{result['selection_sha256']}`.",
             "الأساسي: تأخير دقيقة، تكلفة إجمالية0.10ATR، تنفيذ الوقف عند أسوأ سعر في الدقيقة، والوقف أولًا عند الغموض.", ""]
    for symbol, sym in result["symbols"].items():
        lines += [f"## {symbol}", "", f"اختيار التطوير: `{sym['development_selection']['id']}`؛ أهلية التطوير: {sym['development_selection']['development_eligible']}.", "",
                  "| الفترة | الفرضية | الصفقات | الدقة الربحية | متوسط R | CI95 للمتوسط | PF | عائد توضيحي % | تراجع الصفقات المغلقة % |",
                  "|---|---|---:|---:|---:|---|---:|---:|---:|"]
        for cohort, c in sym["cohorts"].items():
            label = "آخر30% المعاد استخدامه" if cohort.startswith("reused") else "العينة الأقدم المستقلة"
            for role, row in c["models"].items():
                m = row["metrics"]
                lo, hi = m["mean_net_R_ci95"]
                lines.append(f"| {label} | {role} | {m['completed']} | {fmt(None if m['win_rate'] is None else 100*m['win_rate'],2)}% | {fmt(m['mean_net_R'])} | {fmt(lo)}–{fmt(hi)} | {fmt(m['profit_factor'])} | {fmt(None if m['closed_trade_return'] is None else 100*m['closed_trade_return'],2)} | {fmt(None if m['closed_trade_max_drawdown'] is None else 100*m['closed_trade_max_drawdown'],2)} |")
        lines += ["", "### التحقق المتحرك باختيار من الماضي فقط", "",
                  "| فترة التحقق | النموذج المختار قبلها | n | متوسط R | PF |", "|---|---|---:|---:|---:|"]
        for fold in sym["walk_forward"]:
            m = fold["validation"]
            lines.append(f"| {fold['test']} | {fold['id']} | {m['completed']} | {fmt(m['mean_net_R'])} | {fmt(m['profit_factor'])} |")
        lines += ["", "### قرار المتابعة بالعينة الخارجية", ""]
        for role, row in sym["cohorts"]["external_older_replication"]["models"].items():
            m = row["metrics"]
            lines.append(f"- {role}: بوابة البحث {'PASS' if row['forward_research_candidate'] else 'FAIL'}؛ Holm p={fmt(m['holm_p'],5)}؛ الفرق عن التوقيت غير المشروط={fmt(m['baseline_difference'])}R؛ تكلفة التعادل الافتراضية={fmt(m['break_even_cost_atr'])}ATR؛ صفقات غير محسومة={m['censored']}؛ غموض داخل الدقيقة={m['ambiguous']}.")
            if row["gate_reasons"]:
                lines.append("  الأسباب: " + ", ".join(row["gate_reasons"]) + ".")
        lines.append("")
    lines += ["## تفسير وحدود", "",
              "- العائد التوضيحي يفترض مخاطرة0.25% من الرصيد النظري لكل مسافة وقف؛ لا يمثل lots أو رصيدًا أو أرباحًا بالدولار. التراجع محسوب من إغلاق الصفقات، ولا يثبت الحد الأقصى للتراجع داخلها.",
              "- الفترة الخارجية أقدم: تحقق تاريخي في فترة مختلفة، وليست Forward. آخر30% كان معروفًا من الدراسة السابقة ويظل استكشافيًا.",
              "- كلفة0.10ATR نموذج حساسية وليست سبريدًا مقاسًا. جميع تكاليف0–0.40ATR وسياسات التنفيذ البديلة محفوظة في results.json.",
              "- تختلف وتيرة الصفقات عن خط الأساس. المقارنة الإحصائية للمتوسط لكل صفقة، وليست مقارنة عائد محافظ متساوية التعرض.",
              "- أي نتيجة تمر البوابة تحتاج بيانات تنفيذ وتكاليف فعلية ثم اختبارًا ورقيًا مستقبليًا جديدًا؛ لا يُفتح أي تداول.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/spike_payoff_20261004")
    parser.add_argument("--bootstrap", type=int, default=9999)
    args = parser.parse_args()
    assert_offline()
    args.output.mkdir(parents=True, exist_ok=True)
    (develop if args.stage == "develop" else evaluate)(args)


if __name__ == "__main__":
    main()
