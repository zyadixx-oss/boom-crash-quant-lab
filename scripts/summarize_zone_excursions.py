#!/usr/bin/env python3
"""Export the complete already-computed grid without fitting or row selection."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/zone_excursions_20261008"
VARIANTS = ("CRT_RETEST", "FIB_RETRACE", "TREND_RETEST")
NAMES = {"CRT_RETEST": "CRT", "FIB_RETRACE": "Fibonacci", "TREND_RETEST": "Trend"}


def number(x, places=3):
    return "غير معرّف" if x is None else f"{x:.{places}f}"


def percent(x):
    return "غير معرّف" if x is None else f"{100*x:.2f}%"


def ci(x):
    return "[" + ", ".join(number(a) for a in x) + "]"


def flatten(value, prefix=""):
    result = {}
    for k, v in value.items():
        key = prefix + k
        if isinstance(v, dict): result.update(flatten(v, key + "."))
        elif isinstance(v, list):
            for i, part in enumerate(v): result[key + f".{i}"] = part
        else: result[key] = v
    return result


def main():
    index = json.loads((OUT / "result_index.json").read_text())
    payload = (ROOT / index["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == index["sha256"]
    r = json.loads(gzip.decompress(payload))
    audit = json.loads((OUT / "independent_audit_attempt2.json").read_text())
    assert audit["passed"] and audit["errors"] == [] and len(r["rows"]) == 2016
    economic_path = ROOT / "docs/zone_study_20261008/results.json"
    assert hashlib.sha256(economic_path.read_bytes()).hexdigest() == r["economic_result_sha256"]
    economic = json.loads(economic_path.read_text())
    metrics = [flatten(x) for x in r["rows"]]
    fields = list(dict.fromkeys(k for row in metrics for k in row))
    with (OUT / "metrics.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(metrics)
    lookup = {(x["symbol"], x["endpoint"], x["partition"], x["variant"], x["baseline_variant"],
        x["atr_multiplier"], x["horizon_minutes"]): x for x in r["rows"]}
    def selected(symbol, endpoint, variant, a=2, h=15, baseline=None):
        return lookup[(symbol, endpoint, "later180", variant,
            baseline or ("AVAILABILITY_CLOCK" if endpoint == "ISSUE_DELAYED" else "CONTEXT_GEOMETRIC"), a, h)]
    text = ["# قياس الحركة بعد مناطق CRT وFibonacci والترند — 8 أكتوبر 2026", "",
        "**لم تتأهل أي استراتيجية، ولم يتحقق هدف PF≥1.5 بعينة كبيرة.** هذا قياس إضافي على نفس المناطق والأسعار التاريخية المكشوفة سابقًا، لا اختبار مستقل جديد ولا تعديل للصفقات الخاسرة. جميع حدود المناطق، توقيت الدخول، الوقف، الخروج والتكاليف بقيت مطابقة للملفات المجمّدة.", "",
        "اختُبرت كل التعريفات الاثني عشر: حركة في الاتجاه الأصلي لـBoom/Crash بمقدار 1.5/2/3×ATR خلال 5/10/15/30 دقيقة. ATR هو M5 ATR14 المعلوم عند إصدار المنطقة. القياس هو وصول السعر إلى مستوى؛ لا يثبت وقوع انفجار منفصل أو تحقيق ربح.", "",
        "المصدر 62,380,130 سعرًا أصليًا، 361 يومًا لكل من Boom600 وCrash600، مع إبقاء 670 ثانية ناقصة مجهولة. أولوية Boom500 ومقارنات CRT الخام/MSS/Displacement/FVG ومرشحات الضغط اختُبرت في الجولات السابقة؛ هذه الإضافة تخص مناطق600 ولا تُعرض كإعادة اختبار500 على tick جديد.", "",
        "الفترات: أول70% من التاريخ الأقدم للتطوير، ثلاث نوافذ وصف walk-forward ثابتة40–50/50–60/60–70، وآخر30% ثم180 يومًا لاحقة. لا يوجد تدريب أو اختيار جديد. جميع الفترات تاريخ معروف بالفعل؛ لا تُسمى fresh OOS. لا تُجمع النماذج أو الرموز لاستيفاء العينة.", "",
        "## التعريف المرجعي 2×ATR خلال15 دقيقة، آخر180 يومًا", "",
        "مرجع الإشارة هو السعر الأصلي عند الإصدار+61 ثانية، والاستجابة تبدأ بعده. يشمل هذا الجدول جميع المناطق الصادرة، حتى التي لم تلمس لاحقًا؛ ليست هذه صفقات منفذة. خط الأساس ساعات الإصدار التي كانت بيانات الفريمات وATR متاحة فيها قبل معرفة المستقبل.", "",
        "| الرمز | المنطقة | نجاح/معلوم | مجهول | Precision | Base rate | Lift | CI95 أسبوعي للـLift | Opportunity recall | وسيط الدقائق من المرجع |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for symbol in ("BOOM600", "CRASH600"):
        for v in VARIANTS:
            x = selected(symbol, "ISSUE_DELAYED", v)
            text.append(f"| {symbol} | {NAMES[v]} | {x['model']['hits']}/{x['model']['known']} | {x['model']['unknown']} | {percent(x['precision'])} | {percent(x['base_rate'])} | {number(x['lift'])} | {ci(x['inference']['week']['lift_ci95'])} | {percent(x['opportunity_recall'])} | {number(x['median_time_to_excursion_minutes_from_anchor'],2)} |")
    text += ["", "Opportunity recall هو نسبة فرص الساعة الإيجابية التي اختارها النموذج، وليس نسبة الانفجارات الفريدة التي اصطادها. Unique-event recall غير معرّف. أُجريت أيضًا مقارنة بمرجع المناطق الهندسية الذي يشترط السياق نفسه؛ كل أرقامه وفتراته محفوظة في metrics.csv.", "",
        "## الحركة بعد الدخول الفعلي المحاكى، منفصلة عن الربح", "",
        "المرجع هنا سعر الدخول الأصلي لكل منطقة لمسها السعر؛ أي حركة قبله لا تُحتسب. حركة تتحقق بعد الوقف أو انتهاء الصفقة تظل نتيجة تصنيف فقط، ولا تدخل في الأرباح. المرجع المقارن دخول المناطق الهندسية وله توقيت مختلف؛ لذلك recall هنا null.", "",
        "| الرمز | المنطقة | نجاح/دخول معلوم | Precision | Base rate | Lift | CI95 أسبوعي للـLift | وسيط الدقائق من الدخول | PF الصافي الأصلي | صفقات مكتملة |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for symbol in ("BOOM600", "CRASH600"):
        for v in VARIANTS:
            x = selected(symbol, "ENTRY_CONDITIONAL", v)
            e = next(y for y in economic["rows"] if y["symbol"] == symbol and y["variant"] == v and y["partition"] == "later180")["metrics"]
            text.append(f"| {symbol} | {NAMES[v]} | {x['model']['hits']}/{x['model']['known']} | {percent(x['precision'])} | {percent(x['base_rate'])} | {number(x['lift'])} | {ci(x['inference']['week']['lift_ci95'])} | {number(x['median_time_to_excursion_minutes_from_anchor'],2)} | {number(e['profit_factor'])} | {e['completed']} |")
    text += ["", "Crash Trend يبقى PF=1.477 مع60 صفقة/51 يومًا نشطًا، وCI95 أسبوعي للـPF=[0.733,2.758]. لا يستوفي PF≥1.5 ولا1000 صفقة و60 يومًا، وفواصل متوسط العائد والفرق مع المرجع تشمل الصفر. لا تُخفَّض التكاليف ولا تُعدَّل القواعد بعد النتيجة لإنقاذه.", "",
        "## كل التعريفات، آخر180 يومًا", "",
        "كل خلية: Precision / Lift. هذه مصفوفة وصفية شاملة وليست اختيارًا لأفضل إعداد. فواصل اليوم/الأسبوع، العينة، القيم المجهولة، حدود عدم اليقين، p وHolm والفترات الأخرى محفوظة بالكامل في metrics.csv وresults.json.gz.", ""]
    for endpoint in ("ISSUE_DELAYED", "ENTRY_CONDITIONAL"):
        text += ["### " + endpoint, "", "| ATR | دقائق | Boom CRT | Boom Fibonacci | Boom Trend | Crash CRT | Crash Fibonacci | Crash Trend |", "|---:|---:|---|---|---|---|---|---|"]
        for a in (1.5, 2., 3.):
            for h in (5, 10, 15, 30):
                cells = [f"{percent(selected(s, endpoint, v, a, h)['precision'])} / {number(selected(s, endpoint, v, a, h)['lift'])}"
                    for s in ("BOOM600", "CRASH600") for v in VARIANTS]
                text.append(f"| {a:g} | {h} | " + " | ".join(cells) + " |")
        text.append("")
    significant = sum(x.get("holm_p", 1) < .05 for x in r["rows"])
    text += ["## التفسير والقرار", "",
        f"لا توجد مقارنة متأهلة بعد تصحيح تعدد المقارنات: {significant} خلية لها Holm p<0.05. طُبّقت عائلة144 مقارنة مستقلة لكل endpoint، مع اشتراط اختباري اليوم والأسبوع وكلا المرجعين في القياس الأساسي. 2016 خلية إجمالية تشمل تكرارات وصفية ومراجع؛ ليست2016 تجارب مستقلة أو صفقات.", "",
        "نرفض اعتماد هذه النسخ المحددة من CRT/Fibonacci/Trend كمناطق دخول مثبتة الربحية. CRT قليل الإشارات؛ Fibonacci دون المرجع في عدة مقاييس؛ Trend قريب من المرجع الهندسي، ولا يثبت توقيت الانفجار. متابعة Crash Trend ممكنة كفرضية غير مؤهلة فقط، دون ادعاء أفضلية أو إعادة ضبط النتائج. لا تثبت النتيجة استحالة كل طرق ICT أو Fibonacci.", "",
        "كل ثانية حتى نهاية الأفق مطلوبة حتى إذا تحقق الهدف مبكرًا؛ الفجوات مجهولة لا خسائر. الحذف المخطط ثابت32 دقيقة من الإصدار للقياس الأساسي و46 دقيقة للدخول، لكل التعريفات. CI95 من9999 إعادة سحب على أيامUTC وكتل سبعة أيام، seed20261008. تكرارات النسبة غير المعرّفة معدودة، والفواصل مشروطة بالتكرارات المعرّفة. حدود unknown منفصلة عن هذه الفواصل.", "",
        f"التدقيق المستقل نجح: **{audit['checks']:,} فحصًا، {audit['definition_labels_checked']:,} نتيجة تعريف، وكل2016 خلية وفواصلها وتصحيحاتها**. قرأ المدقق كل الأسعار عبر NumPy دون استيراد محرك الاستراتيجية/التصنيف؛ طابق عضوية الفرص، التوقيت، أول عبور، missing/purge، الأرقام وفواصل اليوم/الأسبوع وHolm. هذا لا يعيد اشتقاق خصائص CRT/Fibonacci/EMA الأصلية. سجل فشل القارئ الأول محفوظ؛ إصلاح نوع عمود ATR الفارغ يخص المدقق وحده، والنتائج المجمّدة لم تتغير.", "",
        "LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false. التداول الفعلي، تنفيذCFD، تكاليف الوسيط المقاسة، الأرباح النقدية وprospective paper: **NOT TESTED**.", "",
        "## الملفات وإعادة التشغيل", "",
        "- [البروتوكول المسبق](PROTOCOL.md)، [التصريح](declaration.json)، [فهرس النتائج](result_index.json).",
        "- [كل2016 خليةCSV](metrics.csv)، [JSON كامل مضغوط](results.json.gz)، ملفات *_labels.csv.gz تحفظ كل الفرص حتى الفارغة والمجهولة.",
        "- [التدقيق المستقل](independent_audit_attempt2.json)، [فشل القارئ المحفوظ](independent_audit_failure_initial.json).",
        "- [تقرير الربح الأصلي](../zone_study_20261008/REPORT.ar.md)، [نتائج500 السابقة](../spike_hunter_20261004/REPORT.ar.md).", "",
        "```sh", "# Requires the exact locally retained ignored native sources and restored candidate CSVs.",
        ".venv/bin/python scripts/restore_zone_candidates.py", ".venv/bin/python scripts/verify_zone_excursions.py --output /tmp/zone-excursion-audit-new.json",
        "# Frozen execution is exclusive; do not overwrite existing output or silently refreeze.", "```", ""]
    with (OUT / "REPORT.ar.md").open("x") as f: f.write("\n".join(text))
    print(json.dumps({"rows": len(metrics), "report": str((OUT / "REPORT.ar.md").relative_to(ROOT)), "significant_cells": significant}))


if __name__ == "__main__": main()
