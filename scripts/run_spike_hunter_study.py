#!/usr/bin/env python3
"""Run reproducible offline excursion classification, never trading simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.research.spike_hunter import (VARIANTS, assert_offline, bootstrap_context,
    eligible, features, holm_adjust, load_m1, outcomes, partitions, research_gate,
    signals, summarize)


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def ci(values: list) -> str:
    return "—" if values[0] is None else f"{values[0]:.3f}–{values[1]:.3f}"


def markdown(result: dict) -> str:
    lines = ["# Spike Hunter — نتائج فعلية، بحث فقط", "",
             f"تاريخ التشغيل UTC: {result['run_utc']}. المصدر: بيانات شموع Deriv العامة M1.", "",
             "**لا توجد استراتيجية مثبتة الربحية.** الأرقام تقيس وصول السعر إلى مسافة محددة؛",
             "الصفقات والتكاليف والربح والخسارة وForward/Shadow: **NOT TESTED**.", "",
             "الأعلام الأربعة: `LIVE_TRADING=false`, `READY_FOR_LIVE=false`,",
             "`LIVE_ALLOWED=false`, `OPENED_TRADES=false`.", "",
             "التعريف الأساسي: حركة اتجاهية **2×ATR من M5 خلال 15 دقيقة** بعد إغلاق",
             "الإشارة. آخر 30% اختبار نهائي؛ تحقق متحرك 40–50% و50–60% و60–70% داخل التطوير.", "",
             "فترات الثقة الرئيسية تسحب أيام UTC معًا للإشارات والخط الأساسي، لمعالجة تداخل",
             "النوافذ. التصحيح Holm يشمل كل الرموز والنماذج والتعريفات المختبرة.", "",
             "Recall هنا تغطية فرص إغلاق M5 الناجحة، وليس نسبة الانفجارات المنفصلة.", ""]
    for symbol, item in result["symbols"].items():
        audit = item["audit"]
        lines += [f"## {symbol}", "",
                  f"شموع M1 صالحة: **{audit['source_rows']:,}**؛ دقائق مفقودة: **{audit['missing_minutes']}**.",
                  f"من {audit['from_utc']} إلى {audit['to_exclusive_utc']} (نهاية غير مشمولة).",
                  f"الفصل: {item['partitions']['final_test'][0]}.", "",
                  "### مقارنة التعريف الأساسي في الاختبار النهائي", "",
                  "| النموذج | نجاح/إشارات | Precision | Base rate | Lift | CI95 Lift أيام | Recall الفرص | TTS دقيقة | Holm p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        primary = item["definitions"]["2.0ATR_15m"]
        for name in VARIANTS:
            r = primary[name]["final_test"]
            lines.append(f"| {name} | {r['hits']}/{r['signals']} | {percent(r['precision'])} | {percent(r['base_rate'])} | {number(r['lift'])} | {ci(r['block_lift_ci95'])} | {percent(r['opportunity_recall'])} | {number(r['median_time_to_spike_min_upper_bound'])} | {number(r['holm_adjusted_p'])} |")
        lines += ["", "### التحقق الزمني للتعريف الأساسي", "",
                  "| النموذج | التطوير n / Lift | طية1 n / Lift | طية2 n / Lift | طية3 n / Lift | البوابة البحثية |",
                  "|---|---:|---:|---:|---:|---|"]
        for name in VARIANTS:
            rows = primary[name]
            cells = [f"{rows[p]['signals']} / {number(rows[p]['lift'])}" for p in ("development", "wf1", "wf2", "wf3")]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {'PASS_RESEARCH_ONLY' if rows['final_test']['research_gate_pass'] else 'REJECTED / يحتاج دليلًا جديدًا'} |")
        lines += ["", "### جميع تعريفات CRT الخام في الاختبار النهائي", "",
                  "| ATR | دقائق | n | Precision | Base | Lift | CI95 Lift | TTS |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, models in item["definitions"].items():
            r = models["CRT"]["final_test"]
            lines.append(f"| {r['atr_multiplier']} | {r['horizon_minutes']} | {r['signals']} | {percent(r['precision'])} | {percent(r['base_rate'])} | {number(r['lift'])} | {ci(r['block_lift_ci95'])} | {number(r['median_time_to_spike_min_upper_bound'])} |")
        lines += ["", "### حساسية طول الكتلة — الاختبار الأساسي", "",
                  "| النموذج | CI95 Lift: يوم | CI95 Lift: 6 ساعات |",
                  "|---|---:|---:|"]
        for name in VARIANTS:
            r = primary[name]["final_test"]
            lines.append(f"| {name} | {ci(r['block_lift_ci95'])} | {ci(r['six_hour_sensitivity']['block_lift_ci95'])} |")
        lines += ["", "### ترتيب المتابعة من التطوير فقط", "",
                  "الترتيب التالي يعتمد على أسوأ Lift في طيات التطوير، بشرط >=30 إشارة في كل طية.",
                  "ولا يمنح نتيجة الاختبار النهائي صفة تأكيد مستقل لأي اختيار جديد.", ""]
        for candidate in item["development_ranking_primary"][:5]:
            name = candidate["variant"]
            r = primary[name]["final_test"]
            lines.append(f"- {name}: أسوأ Lift تطوير {number(candidate['worst_fold_lift'])}؛ نهائي {number(r['lift'])}؛ أسباب عدم الاجتياز: {', '.join(r['research_gate_reasons']) or 'لا توجد، لكن الربحية غير مختبرة'}.")
        lines.append("")
    lines += ["## حدود النتيجة وإعادة التشغيل", "",
              "- شموع M1 لا تحدد ثانية حدوث الانفجار؛ TTS نهاية أول دقيقة ناجحة، وبين الناجحين فقط.",
              "- 16 نموذجًا × 12 تعريفًا × عدد الرموز؛ أعلى نتيجة من هذه الشبكة استكشافية.",
              "- فترات الثقة pointwise؛ قيم p bootstrap تقريبية وليست ضمانًا ضد كل أنواع الاعتماد الزمني.",
              "- شرط 200 إشارة و30 نجاحًا و20 يومًا حد بحثي عملي، وليس إثباتًا للربحية.",
              "- لا يمكن نقل إعدادات Boom500 إلى Crash500 أو رموز أخرى دون اختبار.",
              "- الملف metrics.csv يحوي جميع النتائج، وresults.json جميع فترات الثقة وأسباب الرفض.",
              "- الفجوات لا تُملأ؛ الساعات والنوافذ غير الكاملة تُستبعد. انظر acquisition manifests.", "",
              "```bash", "python -m venv .venv", ".venv/bin/pip install -r backend/requirements.txt",
              "LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \\",
              "  .venv/bin/python scripts/run_spike_hunter_study.py \\",
              "  --dataset BOOM500=data/spike_hunter/boom500_m1_clean.csv \\",
              "  --dataset CRASH500=data/spike_hunter/crash500_m1_clean.csv \\",
              "  --output docs/spike_hunter_20261004 --bootstrap 9999", "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="SYMBOL=CSV; M1 only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=9999)
    args = parser.parse_args()
    safe = assert_offline()
    if args.bootstrap < 2000:
        parser.error("Use at least 2000 block bootstrap replicates")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / "docs" / "SPIKE_HUNTER_PROTOCOL.md"
    result = {"run_utc": datetime.now(timezone.utc).isoformat(), "safety": safe,
              "status": "OFFLINE_RESEARCH_NO_PNL_TEST", "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "bootstrap_replicates": args.bootstrap,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "symbols": {}, "final_family_size": len(args.dataset) * len(VARIANTS) * 12}
    final_rows, flat_rows = [], []
    for dataset in args.dataset:
        symbol, source = dataset.split("=", 1)
        if symbol in result["symbols"]:
            parser.error("Duplicate symbol")
        direction = "boom" if symbol.upper().startswith("BOOM") else "crash" if symbol.upper().startswith("CRASH") else None
        if direction is None:
            parser.error("Only separate Boom/Crash symbols accepted")
        print(f"Loading {symbol}: {source}", flush=True)
        m1, audit = load_m1(Path(source))
        m5 = features(m1, direction)
        sigs = signals(m5, direction)
        part = partitions(m1)
        _, _, valid30 = outcomes(m1, m5, direction, 2, 30)
        sym = {"audit": audit, "direction": direction,
               "partitions": {p: [str(a), str(b)] for p, (a, b) in part.items()},
               "signals_before_outcome_filter": {name: len(v) for name, v in sigs.items()},
               "definitions": {}}
        result["symbols"][symbol] = sym
        # Persist issue times for independent audit. No outcomes are used to issue.
        issued = [{"variant": name, "signal_close_utc": str(m5.index[i] + pd.Timedelta(minutes=5)),
                   "entry_reference": float(m5.close.iloc[i]), "atr_m5": float(m5.atr.iloc[i])}
                  for name, indices in sigs.items() for i in indices]
        pd.DataFrame(issued).to_csv(args.output / f"{symbol.lower()}_signals.csv", index=False)
        labels = {(mult, horizon): outcomes(m1, m5, direction, mult, horizon)
                  for mult in (1.5, 2.0, 3.0) for horizon in (5, 10, 15, 30)}
        for mult in (1.5, 2.0, 3.0):
            for horizon in (5, 10, 15, 30):
                key = f"{mult}ATR_{horizon}m"
                sym["definitions"][key] = {name: {} for name in VARIANTS}
        for partition, (start, end) in part.items():
            mask = eligible(m5, valid30, start, end)
            if int(mask.sum()) < 100:
                raise RuntimeError(f"Insufficient eligible opportunities in {symbol} {partition}; verify clock units and data")
            print(f"{symbol} {partition}: {mask.sum():,} opportunities", flush=True)
            ctx = bootstrap_context(m5, mask, start, end, args.bootstrap)
            for (mult, horizon), (hit, tts, valid) in labels.items():
                if (mask & ~valid).any():
                    raise RuntimeError("A shorter horizon cannot be invalid when 30 minutes is valid")
                for name in VARIANTS:
                    row = summarize(hit, tts, mask, sigs[name], ctx, m5)
                    row.update(symbol=symbol, variant=name, partition=partition,
                               atr_multiplier=mult, horizon_minutes=horizon)
                    sym["definitions"][f"{mult}ATR_{horizon}m"][name][partition] = row
                    flat_rows.append(row)
                    if partition == "final_test":
                        final_rows.append(row)
            if partition == "final_test":
                ctx6 = bootstrap_context(m5, mask, start, end, args.bootstrap, block_hours=6)
                hit, tts, _ = labels[(2.0, 15)]
                for name in VARIANTS:
                    sens = summarize(hit, tts, mask, sigs[name], ctx6, m5)
                    sym["definitions"]["2.0ATR_15m"][name][partition]["six_hour_sensitivity"] = sens
        ranking = []
        primary = sym["definitions"]["2.0ATR_15m"]
        for name in VARIANTS:
            folds = [primary[name][p] for p in ("wf1", "wf2", "wf3")]
            if all(f["signals"] >= 30 and f["lift"] is not None for f in folds):
                ranking.append({"variant": name, "worst_fold_lift": min(f["lift"] for f in folds),
                                "pooled_validation_signals": sum(f["signals"] for f in folds)})
        sym["development_ranking_primary"] = sorted(ranking, key=lambda x: x["worst_fold_lift"], reverse=True)
    holm_adjust(final_rows)
    for sym in result["symbols"].values():
        for models in sym["definitions"].values():
            for rows in models.values():
                ok, reasons = research_gate(rows["final_test"], [rows[p] for p in ("wf1", "wf2", "wf3")])
                rows["final_test"].update(research_gate_pass=ok, research_gate_reasons=reasons)
    result["research_gate_pass_count"] = sum(r["research_gate_pass"] for r in final_rows)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    csv_rows = []
    for row in flat_rows:
        record = {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
        for key in ("wilson_precision_ci95", "block_precision_ci95", "block_base_rate_ci95", "block_lift_ci95", "block_difference_ci95"):
            record[key + "_low"], record[key + "_high"] = row[key]
        csv_rows.append(record)
    pd.DataFrame(csv_rows).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "REPORT.ar.md").write_text(markdown(result), encoding="utf-8")
    print(f"DONE: {len(flat_rows)} metric rows; research gate passes={result['research_gate_pass_count']}; all live flags false.", flush=True)


if __name__ == "__main__":
    main()
