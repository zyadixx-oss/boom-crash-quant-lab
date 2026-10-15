#!/usr/bin/env python3
"""Render saved nonlinear-region evidence without fitting or replaying a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT_SHA256 = "4664f8d2ad7282acf1336e449efa6376f467452d86df8a3b2e70a83ab04cf4d7"
DECLARATION_SHA256 = "35756c799689ef6a03f76e3aeda86a4568994a328f79af13685a929f91bee09c"
FIRST_AUDIT_SHA256 = "43e7c94d6a0dab5e97b7ba95be6763424e6a7fcde2497bcaa8034d1b6fe4f297"
SAFETY = {
    "LIVE_TRADING": False,
    "READY_FOR_LIVE": False,
    "LIVE_ALLOWED": False,
    "OPENED_TRADES": False,
}
SYMBOLS = ("BOOM600", "CRASH600")
VARIANTS = (
    "CLOCK", "RAW_REGION", "HYBRID_REGION", "RAW_BOOST_REGION", "HYBRID_BOOST_REGION"
)
NEW_VARIANTS = VARIANTS[-2:]
PARTITIONS = ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned(path: Path, sha256: str) -> dict:
    require(len(sha256) == 64 and set(sha256) <= set("0123456789abcdef"), "Invalid SHA256 pin")
    require(digest(path) == sha256, f"SHA256 mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(audit_path: Path, audit_sha256: str) -> tuple[dict, dict, dict, dict]:
    result = load_pinned(HERE / "results.json", RESULT_SHA256)
    declaration = load_pinned(HERE / "declaration.json", DECLARATION_SHA256)
    audit = load_pinned(audit_path, audit_sha256)
    first_audit = load_pinned(HERE / "independent_audit.json", FIRST_AUDIT_SHA256)
    require(audit.get("passed") is True and audit.get("errors") == [], "Independent audit is not PASS")
    require(audit.get("result_sha256") == RESULT_SHA256, "Audit result identity mismatch")
    require(audit.get("declaration_sha256") == DECLARATION_SHA256, "Audit declaration identity mismatch")
    require(result.get("declaration_sha256") == DECLARATION_SHA256, "Result declaration identity mismatch")
    for name, expected in (("results.json", RESULT_SHA256), ("declaration.json", DECLARATION_SHA256)):
        key = str((HERE / name).relative_to(ROOT))
        require(audit.get("input_sha256", {}).get(key) == expected, f"Audit input pin missing: {key}")
    require(audit.get("verifier_sha256") == digest(HERE / "review_study.py"), "Reviewer source pin mismatch")
    require(audit.get("cells_checked") == 60, "Audit does not cover all 60 cells")
    require(audit.get("heldout_gates_checked") == 8, "Audit does not cover all eight held-out gates")
    for label, item in (("result", result), ("declaration", declaration), ("audit", audit)):
        require(item.get("safety") == SAFETY, f"Unsafe {label} flags")
        require(item.get("QUALIFIED") is False, f"Unexpected {label} qualification")
    require(result.get("fresh_out_of_sample") is False, "Fresh OOS flag mismatch")
    require(result.get("known_history_adaptive") is True, "Adaptive-history flag mismatch")
    require(first_audit.get("passed") is False and len(first_audit.get("errors", [])) == 93,
            "First failed audit record changed")
    rows = result["rows"]
    identities = {(r["symbol"], r["variant"], r["partition"]) for r in rows}
    expected = {(s, v, p) for s in SYMBOLS for v in VARIANTS for p in PARTITIONS}
    require(len(rows) == 60 and identities == expected, "Cell identity set mismatch")
    gates = [r for r in rows if "qualification" in r]
    require(len(gates) == 8, "Qualification cell count mismatch")
    require(all(r["qualification"]["historical_criteria_passed"] is False for r in gates),
            "Saved result no longer supports the stated rejection")
    require(all(r["qualification"]["point_criterion"] == "finite PF > 1 AND finite mean_net_R > 0"
                for r in gates), "Unexpected point criterion")
    return result, declaration, audit, first_audit


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "غير معرّف"
    if isinstance(value, bool):
        return "نعم" if value else "لا"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "غير معرّف"
        return str(value) if isinstance(value, int) else f"{value:.{digits}f}"
    return str(value)


def interval(values: list | None) -> str:
    if values is None or len(values) != 2:
        return "غير معرّف"
    return f"[{fmt(values[0])}, {fmt(values[1])}]"


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ])


def ordered_rows(result: dict) -> list[dict]:
    lookup = {(r["symbol"], r["variant"], r["partition"]): r for r in result["rows"]}
    return [lookup[s, v, p] for s in SYMBOLS for p in PARTITIONS for v in VARIANTS]


def metrics_row(row: dict, identity: bool = True) -> list[object]:
    m = row["metrics"]
    values = [
        m["completed"], m["active_days"], fmt(m["profit_factor"]),
        interval(row["profit_factor_inference"]["weekly_profit_factor_ci95"]),
        fmt(m["mean_net_R"]), interval(row["weekly_inference"]["weekly_mean_net_R_ci95"]),
        fmt(row["double_cost_metrics"]["mean_net_R"]), row["replay_audit"]["unknown"],
        fmt(m["median_holding_minutes"], 2),
    ]
    return ([row["symbol"], row["variant"]] if identity else []) + values


METRIC_HEADERS = ["المكتمل n", "أيام نشطة", "PF صافٍ", "CI95 أسبوعي لـPF",
                  "متوسط R صافٍ", "CI95 أسبوعي لمتوسط R", "متوسط R مع تكلفة مضاعفة",
                  "مسارات مجهولة", "وسيط الاحتفاظ/دقيقة"]


def render_report(result: dict, declaration: dict, audit: dict, first_audit: dict,
                  audit_path: Path, audit_sha256: str) -> str:
    rows = ordered_rows(result)
    lookup = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    lead = [lookup[s, v, "later180"] for s in SYMBOLS for v in NEW_VARIANTS]
    crash = lookup["CRASH600", "RAW_BOOST_REGION", "later180"]
    old = lookup["CRASH600", "RAW_BOOST_REGION", "final_test"]
    wf = lookup["CRASH600", "RAW_BOOST_REGION", "walk_forward_combined"]
    passed = sum(r.get("qualification", {}).get("historical_criteria_passed") is True for r in rows)
    spec = declaration["specification"]
    lines = [
        "# اختبار العائد الشرطي غير الخطي للمناطق — 10 أكتوبر 2026",
        "",
        f"**النتيجة: {passed}/8 خلايا اجتازت الشروط التاريخية الكاملة. لا توجد استراتيجية مؤهلة من هذه التجربة.** "
        "اختُبر فعليًا الجمع الذي لم تغطّه المقارنة السابقة: تعلّم غير خطي لعائد المنطقة المملوءة، "
        "مع إبقاء البيانات والخصائص والمناطق والتكلفة والمعلمات ثابتة. التاريخ مكشوف سابقًا؛ هذه تجربة تطوير تكيفية وليست OOS جديدة.",
        "",
        "## النتيجة في الفترة اللاحقة ذات 180 يومًا",
        "",
        table(["الرمز", "النموذج", *METRIC_HEADERS], [metrics_row(r) for r in lead]),
        "",
        f"حقق `CRASH600 / RAW_BOOST_REGION` تقديرًا نقطيًا موجبًا: PF={fmt(crash['metrics']['profit_factor'])} "
        f"ومتوسط R={fmt(crash['metrics']['mean_net_R'])}، مع {crash['metrics']['completed']} مسارًا مكتملًا "
        f"و{crash['metrics']['active_days']} يومًا نشطًا. هذا يحقق شرط النقطة الجديد رغم PF الأقل من 1.1، "
        "ولا يُرفض بسبب انخفاض PF عن 1.1. لكنه لا يثبت الاستقرار: CI95 الأسبوعي يعبر التعادل في PF ومتوسط R، "
        f"والتكلفة المضاعفة تجعل المتوسط {fmt(crash['double_cost_metrics']['mean_net_R'])}، "
        f"مع {crash['replay_audit']['unknown']} مسارات مجهولة. الأثلاث الزمنية تعطي PF="
        + "/".join(fmt(t["metrics"]["profit_factor"]) for t in crash["thirds"]) + ".",
        "",
        f"لم تتكرر النقطة الموجبة في الاختبار الأقدم: PF={fmt(old['metrics']['profit_factor'])} "
        f"ومتوسط R={fmt(old['metrics']['mean_net_R'])}، ولا في اتحاد walk-forward: "
        f"PF={fmt(wf['metrics']['profit_factor'])} ومتوسط R={fmt(wf['metrics']['mean_net_R'])}. "
        "لذلك يُرفض النموذج كاستراتيجية ربح مستقرة؛ تُحفظ النقطة الموجبة كملاحظة تطوير تكيفية فقط. "
        "هذه النتائج لا تثبت استحالة كل سياسات Boom/Crash ولا تحدد منطقة مستقبلية موثوقة.",
        "",
        "## استقرار Crash RAW_BOOST عبر الفترات",
        "",
        table(["الفترة", *METRIC_HEADERS], [
            [p, *metrics_row(lookup["CRASH600", "RAW_BOOST_REGION", p], identity=False)]
            for p in PARTITIONS
        ]),
        "",
        "اتحاد walk-forward يجمع سجلات الطيات المنفصلة مع حدودها المحذوفة؛ لا يضيف بيانات مستقلة "
        "ولا تُجمع فتراته مع الاختبارين لتجاوز شرط حجم العينة.",
        "",
        "## المقارنة مع ridge الموافق والساعة المشتركة",
        "",
        "الأرقام التالية تخص `later180`. فروق المتوسط واختبارات المقارنة من القيم المحفوظة "
        "لإعادة أخذ عينات أيام UTC المزدوجة، مع كتل أسبوعية دائرية. المقارنة وصفية لسياسات مختلفة "
        "في الانتقاء والإشغال؛ ليست أثرًا سببيًا محددًا.",
        "",
    ]
    comparisons = []
    for row in lead:
        same_arm = "RAW_REGION" if row["variant"] == "RAW_BOOST_REGION" else "HYBRID_REGION"
        for ref in ("CLOCK", same_arm):
            reference = lookup[row["symbol"], ref, "later180"]
            if ref == "CLOCK":
                delta = row["day_inference"]["baseline_difference"]
                ci = row["weekly_inference"]["weekly_difference_ci95"]
            else:
                comp = row["target_comparisons"][ref]
                delta = comp["day"]["baseline_difference"]
                ci = comp["week"]["weekly_difference_ci95"]
            comparisons.append([
                row["symbol"], row["variant"], ref, fmt(reference["metrics"]["profit_factor"]),
                reference["metrics"]["completed"], fmt(delta), interval(ci), reference["replay_audit"]["unknown"],
            ])
    lines += [
        table(["الرمز", "النموذج", "المرجع", "PF المرجع", "n المرجع", "فرق متوسط R",
               "CI95 أسبوعي للفرق", "مجهول المرجع"], comparisons),
        "",
        "`RAW_REGION` و`HYBRID_REGION` مرجعا ridge من الدراسة السابقة، منسوخان بالأرقام نفسها "
        "دون إعادة تدريب أو إعادة سحب إحصاءاتهما. تشترط بوابة النموذج الهجين أيضًا تفوقًا على "
        "`RAW_BOOST_REGION`؛ تُحفظ تلك المقارنة كاملة في `results.json`.",
        "",
        "## التصميم وحدود الدليل",
        "",
        "- الرمزان في هذه الجولة: Boom 600 وCrash 600. استخدمت مصادر الأصل نفسها: 722 ملف يومي، "
        "62,380,130 سعرًا عامًا و670 ثانية مفقودة محفوظة دون استيفاء. لا تتضمن الجولة اختبارًا جديدًا لـBoom 500.",
        "- الخصائص المغلقة تجمع H4/H1/M15/M5/M1. RAW44 يستخدم التمثيل الأصلي؛ HYBRID44 "
        "يستخدم 43 خاصية محوّلة وعمر الشمعة الكبيرة المغلقة نفسه. التنفيذ وATR من الأسعار الأصلية دائمًا.",
        f"- المتعلّم ثابت: {spec['parameters']['n_trees']} شجرة، عمق {spec['parameters']['max_depth']}، "
        f"معدل تعلم {spec['parameters']['learning_rate']}، أصغر ورقة {spec['parameters']['min_leaf']}، "
        f"{spec['parameters']['n_bins']} صندوقًا، تنظيم {fmt(spec['parameters']['leaf_regularization'], 0)}، "
        "وحد اختيار q75 من درجات النموذج على صفوف التدريب المكتملة فقط، مع اشتراط درجة موجبة. الدرجات تقديرات R غير معايرة وليست احتمالات.",
        "- التدريب على أول 40%/50%/60% ثم اختبار الـ10% التالية لكل طية. تدريب 70% واحد "
        "لآخر 30% من الفترة القديمة وللفترة اللاحقة، دون إعادة تدريب لاحقة. فترة الاختبار القديم "
        "[2026-02-12T11:08Z, 2026-04-07T11:08Z)، واللاحقة "
        "[2026-04-07T11:08Z, 2026-10-04T11:08Z). حدود الفترات UTC ونهاياتها مستبعدة.",
        "- عند السعر المغلق p وATR الأصلي A: منطقة Boom=[p−0.55A,p−0.45A]، "
        "ومنطقة Crash=[p+0.45A,p+0.55A]. إبطال وراء الحد المعاكس بمقدار 0.25A، تفعيل بعد دقيقة، "
        "لمس مرصود خلال 15 دقيقة، دخول بالسعر الأصلي في الثانية التالية، وقف 2A واحتفاظ 15 دقيقة من اللمس. "
        "الخسائر غير محدودة بسعر الوقف، وتعريض واحد منتظر/مفتوح لكل مسار، وحذف 31 دقيقة عند الحدود.",
        "- التكلفة الأساسية المفترضة 0.10A، والمضاعفة 0.20A. هذه تكلفة نموذجية وليست تكلفة CFD تاريخية مقاسة. "
        "R هو العائد نسبة إلى مسافة الوقف الاسمية؛ لا يمثل ربحًا نقديًا محققًا.",
        "- CI95 من 9,999 إعادة سحب يومية وكتل سبعة أيام، ببذرة 20261008. الجدول يعرض الأسبوعية؛ "
        "اليومية محفوظة في النتائج وCSV. وسيط الاحتفاظ هو مدة المسار بعد الدخول، وليس time-to-spike. "
        "هذه التجربة تستهدف العائد الشرطي للمنطقة؛ precision/recall وتعريفات spike الاثنا عشر ليست قياسات جديدة هنا.",
        "- القبول النقطي `PF>1 AND mean_net_R>0`؛ لا أرضية PF1.1 أو PF1.5. تبقى شروط التطوير "
        "والعينة ≥1000 مسار مكتمل و≥60 يومًا نشطًا لكل رمز/نموذج/فترة، وفواصل الثقة، والتفوق على المراجع، "
        "وثبات الأثلاث والتكلفة والمخاطر وعدم الجهل بالمسارات قائمة. عائلة Holm مشتركة من ثماني خلايا فقط.",
        "",
        "## البوابات الثماني",
        "",
    ]
    gate_rows = []
    for row in rows:
        if "qualification" not in row:
            continue
        m = row["metrics"]
        q = row["qualification"]
        point = m["profit_factor"] is not None and m["mean_net_R"] is not None and m["profit_factor"] > 1 and m["mean_net_R"] > 0
        gate_rows.append([
            row["symbol"], row["variant"], row["partition"], fmt(point),
            fmt(m["completed"] >= 1000 and m["active_days"] >= 60),
            fmt(q["development_eligible"]), fmt(row["holm_p"]), fmt(q["historical_criteria_passed"]),
        ])
    lines += [
        table(["الرمز", "النموذج", "الفترة", "نقطة موجبة", "n والأيام كافيان",
               "التطوير مؤهل", "Holm p", "البوابة كاملة"], gate_rows),
        "",
        "كل خلايا التطوير مرفوضة أو غير كافية. فواصل الثقة وشروط التفوق والأثلاث والتكلفة لا تحقق "
        "التلازم المطلوب. كانت مسارات CLOCK المجهولة معروفة قبل القياس: 3 في الاختبار القديم و8 في "
        "الفترة اللاحقة لكل رمز؛ تمنع بوابة عدم الجهل في المرجع التأهيل حتى لو بدا تقدير أحد المرشحين موجبًا. "
        "لم تُحذف هذه المسارات أو تُستبدل بعائد صفري. أسباب كل رفض كاملة في `qualification.historical_rejection_reasons`.",
        "",
        "## المراجعة وقابلية الإعادة",
        "",
        f"المراجعة المستقلة النهائية `PASS`: {audit['checks']:,} فحصًا، "
        f"{audit['models_checked']} نموذجًا، {audit['cells_checked']} خلية و{audit['heldout_gates_checked']} بوابات. "
        "يشمل نطاقها تطابق الأهداف والخصائص المحفوظة، حساب الشجرات الموجّهة والدرجات والاختيار، "
        "ومسارات التنفيذ الأصلية الجديدة، والإحصاءات وإعادة أخذ العينات والبوابات. "
        "لا يعاد بها البحث عن أفضل انقسام لكل شجرة، ولا اشتقاق صيغ الخصائص الأربع والأربعين، "
        "ولا مسارات أسعار تدريب ومراجع الدراسة الأصلية في هذه الجولة. PASS يصف هذا النطاق فقط ولا يثبت ربحًا مستقرًا.",
        "",
        f"حُفظت المحاولة الأولى للمراجعة بنتيجة `FAIL`: {first_audit['checks']:,} فحصًا "
        f"و{len(first_audit['errors'])} اختلافًا في مقاييس الأثلاث. لم تُمحَ هذه المحاولة؛ "
        "أغفلت أداة المراجعة في البداية حذف الأفق المخطط: وقت الإشارة +31 دقيقة يجب ألا يتجاوز نهاية الثلث. "
        "أُثبت الفرق باختبار حدود، ثم أُصلحت أداة المراجعة وحدها. التفاصيل في [auditor_notes.md](auditor_notes.md). "
        "يعتمد هذا التقرير على المراجعة النهائية المثبتة أدناه، مع بقاء بصمتي النتائج والإعلان الأصليتين.",
        "",
        f"- الإعلان: `{DECLARATION_SHA256}`.",
        f"- النتائج: `{RESULT_SHA256}`.",
        f"- المراجعة الأولى: `{FIRST_AUDIT_SHA256}`.",
        f"- المراجعة النهائية `{audit_path.name}`: `{audit_sha256}`.",
        f"- مصدر المراجع: `{audit['verifier_sha256']}`.",
        "",
        "ينشئ `summarize.py` هذا التقرير و`summary.csv` من القيم المحفوظة، بعد اشتراط PASS "
        "وتطابق بصمات الإعلان والنتائج ومصدر المراجع. يحفظ CSV القيم العددية الأصلية دون التقريب المرئي في الجداول. "
        "يمكن إعادة التحقق دون كتابة ملفات بالأمر:",
        "",
        "```sh",
        "python3 docs/nonlinear_region_20261010/summarize.py \\",
        f"  --audit docs/nonlinear_region_20261010/{audit_path.name} \\",
        f"  --audit-sha256 {audit_sha256} --check",
        "```",
        "",
        "`LIVE_TRADING=false`، `READY_FOR_LIVE=false`، `LIVE_ALLOWED=false`، `OPENED_TRADES=false`. "
        "لم يُفتح أي تداول. تنفيذ CFD، تكاليفه التاريخية المقاسة، الربح النقدي والاختبار الورقي المستقبلي: **NOT TESTED**.",
        "",
        "## ملحق: جميع الخلايا الستين",
        "",
        "تتضمن الخلايا مراجع الساعة وridge المنسوخة، والطيات واتحادها المتداخل؛ ليست 60 عينة مستقلة. "
        "المجهول يشمل حالات الانتظار أو المسار المفقود، وقد يزيد على عدد الصفقات المحجوبة بعد الدخول.",
        "",
        table(["الرمز", "الفترة", "النموذج", *METRIC_HEADERS], [
            [r["symbol"], r["partition"], r["variant"], *metrics_row(r, identity=False)] for r in rows
        ]),
        "",
    ]
    return "\n".join(lines)


def render_csv(result: dict) -> str:
    output = io.StringIO(newline="")
    fields = [
        "symbol", "variant", "partition", "start", "end_exclusive", "completed", "trades",
        "active_days", "unknown_region_outcomes", "censored", "invalid_uncensored", "profit_factor",
        "day_PF_ci95_low", "day_PF_ci95_high", "weekly_PF_ci95_low", "weekly_PF_ci95_high",
        "mean_net_R", "day_mean_ci95_low", "day_mean_ci95_high", "weekly_mean_ci95_low",
        "weekly_mean_ci95_high", "double_cost_profit_factor", "double_cost_mean_net_R",
        "median_holding_minutes", "closed_trade_max_drawdown", "point_positive", "large_sample_and_days",
        "development_eligible", "historical_criteria_passed", "holm_p", "fresh_out_of_sample", "QUALIFIED",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in ordered_rows(result):
        m = row["metrics"]
        q = row.get("qualification", {})
        pf = row["profit_factor_inference"]
        record = {k: row[k] for k in ("symbol", "variant", "partition", "start", "end_exclusive")}
        record.update({k: m[k] for k in ("completed", "trades", "active_days", "censored", "invalid_uncensored",
                                       "profit_factor", "mean_net_R", "median_holding_minutes", "closed_trade_max_drawdown")})
        for prefix, values in (
            ("day_PF", pf["day_profit_factor_ci95"]), ("weekly_PF", pf["weekly_profit_factor_ci95"]),
            ("day_mean", row["day_inference"]["mean_net_R_ci95"]),
            ("weekly_mean", row["weekly_inference"]["weekly_mean_net_R_ci95"]),
        ):
            record[f"{prefix}_ci95_low"], record[f"{prefix}_ci95_high"] = values
        record.update({
            "unknown_region_outcomes": row["replay_audit"]["unknown"],
            "double_cost_profit_factor": row["double_cost_metrics"]["profit_factor"],
            "double_cost_mean_net_R": row["double_cost_metrics"]["mean_net_R"],
            "point_positive": m["profit_factor"] is not None and m["mean_net_R"] is not None and m["profit_factor"] > 1 and m["mean_net_R"] > 0,
            "large_sample_and_days": m["completed"] >= 1000 and m["active_days"] >= 60,
            "development_eligible": q.get("development_eligible", "NOT_APPLICABLE"),
            "historical_criteria_passed": q.get("historical_criteria_passed", "NOT_APPLICABLE"),
            "holm_p": row.get("holm_p", "NOT_APPLICABLE"),
            "fresh_out_of_sample": result["fresh_out_of_sample"], "QUALIFIED": row["QUALIFIED"],
        })
        writer.writerow({k: "UNKNOWN" if v is None else v for k, v in record.items()})
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path, help="Passed independent audit JSON")
    parser.add_argument("--audit-sha256", required=True, help="Exact SHA256 of that audit")
    parser.add_argument("--check", action="store_true", help="Compare existing report/CSV without writing")
    parser.add_argument("--validate-only", action="store_true", help="Verify inputs without emitting artifacts")
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    result, declaration, audit, first_audit = load_inputs(audit_path, args.audit_sha256)
    if args.validate_only:
        print("PASS: exact pins and independent audit; no report emitted")
        return
    artifacts = {
        HERE / "REPORT.ar.md": render_report(result, declaration, audit, first_audit, audit_path, args.audit_sha256),
        HERE / "summary.csv": render_csv(result),
    }
    if args.check:
        for path, content in artifacts.items():
            require(path.read_bytes() == content.encode("utf-8"), f"Saved summary does not reproduce: {path}")
        print("PASS: REPORT.ar.md and all 60 summary.csv cells reproduce exactly")
        return
    require(all(not p.exists() for p in artifacts), "Refusing to overwrite an existing report or CSV")
    for path, content in artifacts.items():
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    print("Created REPORT.ar.md and summary.csv from pinned, independently audited evidence")


if __name__ == "__main__":
    main()
