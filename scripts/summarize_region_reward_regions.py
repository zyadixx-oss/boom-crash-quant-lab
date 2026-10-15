#!/usr/bin/env python3
"""Write the Arabic region-reward report from completed audited artifacts only.

This script reads result/model/audit JSON, never quotes, features or ledgers. It
does not rerun statistics or rank a new parameter choice. Every fixed result cell
and required held-out comparison is retained. A passing independent audit is
required before publishing, with its actual scope and limitations quoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/region_reward_20261008"
SYMBOLS = ("BOOM600", "CRASH600")
FAMILIES = ("CLOCK", "RAW_TIMED", "HYBRID_TIMED", "RAW_REGION", "HYBRID_REGION")
NEW = ("RAW_REGION", "HYBRID_REGION")
FOLDS = ("fit40", "fit50", "fit60", "fit70")
PARTS = ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180")
SAFETY = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
PART_LABELS = {
    "wf1": "نافذة التحقق الأولى بعد تدريب أول 40%",
    "wf2": "نافذة التحقق الثانية بعد تدريب أول 50%",
    "wf3": "نافذة التحقق الثالثة بعد تدريب أول 60%",
    "walk_forward_combined": "اتحاد نوافذ التحقق الثلاث المستقلة في الإصدار",
    "final_test": "آخر 30% من الفترة الأقدم",
    "later180": "فترة 180 يومًا اللاحقة",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=3):
    if value is None:
        return "غير معرّف"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Report statistics must be finite numbers or explicit null")
    return f"{value:.{digits}f}"


def interval(values):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("Every saved interval must contain exactly two endpoints")
    return "[" + ", ".join(number(value, 4) for value in values) + "]"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def repository_path(value):
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Report input must remain within the repository") from error
    return path


def read_checked(path, expected=None, sidecar=False):
    path = repository_path(path)
    fingerprint = digest(path)
    if expected is not None and fingerprint != expected:
        raise ValueError(f"Pinned report input changed: {path.name}")
    if sidecar and path.with_suffix(".sha256").read_text().strip() != fingerprint:
        raise ValueError(f"Report input sidecar differs: {path.name}")
    return json.loads(path.read_text()), fingerprint


def require_safety(document, name):
    if any(document.get("safety", {}).get(flag) is not False for flag in SAFETY):
        raise ValueError(f"All four offline flags must be explicitly false in {name}")


def validate_result(result, declaration):
    require_safety(result, "results")
    require_safety(declaration, "declaration")
    if (result.get("stage") != "executed_region_conditional_reward"
            or result.get("QUALIFIED") is not False
            or result.get("fresh_out_of_sample") is not False
            or result.get("known_history_adaptive") is not True):
        raise ValueError("The saved result must retain adaptive-history and qualification limits")
    rows = result.get("rows", [])
    expected = {(symbol, family, part) for symbol in SYMBOLS for family in FAMILIES for part in PARTS}
    observed = [(r.get("symbol"), r.get("variant"), r.get("partition")) for r in rows]
    if len(rows) != 60 or len(set(observed)) != 60 or set(observed) != expected:
        raise ValueError("Report requires every one of the exact60 fixed cells")
    for row in rows:
        if row.get("endpoint") != "REGION" or row.get("QUALIFIED") is not False:
            raise ValueError("All cells must remain region experiments with QUALIFIED=false")
        if row["variant"] in NEW:
            wanted = {"RAW_TIMED"} if row["variant"] == "RAW_REGION" else {"HYBRID_TIMED", "RAW_REGION"}
            if set(row.get("target_comparisons", {})) != wanted:
                raise ValueError("A required target comparison is missing, including fold unions")
            if row["partition"] in ("final_test", "later180"):
                qualification = row.get("qualification", {})
                if qualification.get("qualified") is not False or qualification.get("fresh_out_of_sample") is not False:
                    raise ValueError("Held-out historical criteria cannot imply final qualification")
                if set(row.get("required_reference_checks", {})) != wanted:
                    raise ValueError("Saved required-reference gate checks are incomplete")
    return {(r["symbol"], r["variant"], r["partition"]): r for r in rows}


def load_report_inputs(folder):
    folder = repository_path(folder)
    declaration, declared_sha = read_checked(folder / "declaration.json", sidecar=True)
    result, result_sha = read_checked(folder / "results.json", sidecar=True)
    if result.get("declaration_sha256") != declared_sha:
        raise ValueError("Result belongs to a different declaration")
    validate_result(result, declaration)
    audit, audit_sha = read_checked(folder / "independent_audit.json", sidecar=True)
    if audit.get("passed") is not True or audit.get("errors") != []:
        raise ValueError("The independent economic audit must pass before report generation")
    pins = audit.get("input_sha256", {})
    for filename, fingerprint in (("declaration.json", declared_sha), ("results.json", result_sha)):
        relative = str((folder / filename).relative_to(ROOT))
        if pins.get(relative) != fingerprint:
            raise ValueError("The independent audit did not pin these exact declaration/result bytes")
    scope = audit.get("verified_scope", audit.get("scope"))
    limitations = audit.get("limitations", audit.get("unverified_scope", audit.get("not_independently_recomputed")))
    if scope is None or limitations is None:
        raise ValueError("Audit must state its verified scope and limitations explicitly")
    models = {}
    for symbol in SYMBOLS:
        folds = result["symbols"][symbol]["folds"]
        if set(folds) != set(FOLDS):
            raise ValueError("Exactly four declared training prefixes are required per symbol")
        for fold in FOLDS:
            artifact = folds[fold]["models"]
            model, _ = read_checked(artifact["path"], artifact["sha256"])
            require_safety(model, "training model")
            if model.get("QUALIFIED") is not False:
                raise ValueError("Training fit cannot qualify live use")
            dispositions = model["training_dispositions"]
            category_total = sum(model[name] for name in ("completed", "known_nonfill", "unknown_path", "exposure_skipped"))
            if sum(dispositions.values()) != category_total:
                raise ValueError("Saved training disposition categories do not cover the whole population")
            models[(symbol, fold)] = model
    return result, declaration, audit, models, {
        "declaration_sha256": declared_sha, "results_sha256": result_sha, "audit_sha256": audit_sha,
        "scope": scope, "limitations": limitations,
    }


def textual_scope(value):
    translations = {
        "immutable_cached44_lineage_common_clock_original_training_region_paths_conditional_categories_independent_ridge_target_matrix_model_hashes_frozen_timed_reference_identity_issuance_native_paths_R_PF_day_week_CIs_required_conjunctions_joint8_Holm":
            "بصمات الخصائص الـ44 والتوافر المشترك؛ مسارات مناطق التدريب الأصلية وتصنيفها؛ حل ridge مستقل وبصمات الأهداف والمصفوفات والنماذج؛ ثبات المراجع الزمنية السابقة؛ درجات الإصدار وعضوية الإشارات ومسارات الأسعار الأصلية؛ حساب R وPF وفواصل المتوسط والفروق يوميًا وأسبوعيًا، والمقارنات المطلوبة وتصحيح Holm لثماني حالات.",
        "43_original_feature_formulas": "لم يُعد اشتقاق صيغ الخصائص الأصلية الـ43 من الصفر.",
        "intrinsic_feature_validity": "لم يُعد بناء شرط الصلاحية الأصلي لكل خاصية من الصفر؛ تحقق من الملفات والتوافر المحفوظين.",
        "qualification_gate_logic": "لم يُعد تنفيذ منطق بوابات التأهيل مستقلًا؛ حالات البوابات المعروضة هي مخرجات المحرك المجمد.",
        "drawdown_and_thirds": "لم يُعد حساب التراجع والأثلاث الزمنية مستقلًا.",
    }
    if isinstance(value, str):
        return [translations.get(value, value)]
    if isinstance(value, list):
        return [translations.get(str(item), str(item)) for item in value]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()]
    raise ValueError("Audit scope/limitations must be human-readable structured text")


def disposition_counts(row, index):
    if row["partition"] == "walk_forward_combined":
        components = [index[(row["symbol"], row["variant"], fold)]["replay_audit"].get("statuses", {})
                      for fold in ("wf1", "wf2", "wf3")]
    else:
        components = [row["replay_audit"].get("statuses", {})]
    return {
        "known_nonfill": sum(item.get("invalidated", 0) + item.get("expired", 0) for item in components),
        "exposure_skipped": sum(item.get("overlap_skipped", 0) for item in components),
    }


def render(result, declaration, audit, models, provenance):
    index = validate_result(result, declaration)
    heldout = [index[(s, v, p)] for s in SYMBOLS for v in NEW for p in ("final_test", "later180")]
    passed = sum(row["qualification"]["historical_criteria_passed"] for row in heldout)
    text = ["# تدريب المناطق على عائدها الفعلي المشروط بالتنفيذ", "",
        f"**اكتمل اختبار 60 حالة. اجتاز كل الشروط التاريخية {passed}/8 من حالات الاختبار للنموذجين الجديدين. QUALIFIED=false؛ التاريخ مستخدم سابقًا، ولم يثبت ربح قابل للتداول الحقيقي.**", "",
        "تختبر هذه التجربة فرضية جديدة محددة: تدريب بوابة المناطق على عوائد مناطق CLOCK المكتملة نفسها، بدل هدف الدخول الزمني السابق. تتغير أيضًا عينة التدريب المشروطة بحدوث التنفيذ؛ لذلك لا تعزل التجربة الأثر السببي لتغيير قيم الهدف وحدها. عدم اللمس والإبطال ليسا صفقة بعائد صفر، والنتائج المجهولة تبقى مجهولة.", "",
        "تستخدم RAW_REGION المؤشرات الأصلية الـ44، وتستخدم HYBRID_REGION المؤشرات المحوّلة الـ43 مع عمر شمعة M5 الكبيرة الأصلي. الفريمات المغلقة H4/H1/M15/M5/M1 والكاشف والخصائص محفوظة من التجربة السابقة ولم يُعد ضبطها. RAW_TIMED وHYBRID_TIMED نموذجان مرجعيان سابقان مجمدان؛ جميع الأذرع الخمسة تطبق مناطق السعر نفسها في هذا التقرير.", "",
        "مصادر التجربة: 722 ملفًا، و62,380,130 سعرًا أصليًا مع إبقاء 670 ثانية ناقصة. تدريب أول 40/50/60% ثم التحقق في 10% تالية لكل إعداد؛ تدريب 70% لآخر 30% و180 يومًا لاحقة دون إعادة ملاءمة. الفترة الأقدم تبدأ 9 أكتوبر 2025 الساعة 11:08 UTC وتنتهي 7 أبريل 2026 الساعة 11:08 UTC، واللاحقة تنتهي 4 أكتوبر 2026 الساعة 11:08 UTC. هذه اختبارات زمنية ضمن تاريخ مكشوف، وليست fresh OOS أو تجربة استباقية.", "",
        "## المجتمع الفعلي للتدريب", "",
        "تتداخل فترات التدريب؛ لا تُجمع الأعداد بوصفها صفقات مستقلة. الصفقة المكتملة فقط تحمل netR في الانحدار. شرط التدريب 1000 صفقة مشتركة مكتملة؛ ridge بعقوبة 0.1 وعتبة الربع الأعلى q75 ودرجة موجبة، دون بحث عن إعداد أفضل.", "",
        "| الرمز | الإعداد | كل الفرص | مكتملة | لم تنفذ/أبطلت | مسار مجهول | تخطيت لانشغال التعرض | الحالة |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for symbol in SYMBOLS:
        for fold in FOLDS:
            model = models[(symbol, fold)]
            text.append(f"| {symbol} | {fold} | {sum(model['training_dispositions'].values())} | {model['completed']} | {model['known_nonfill']} | {model['unknown_path']} | {model['exposure_skipped']} | {cell(model['status'])} |")
    text += ["", "## إصدار المنطقة وحساب العائد", "",
        "عند السعر الأصلي المغلق p وATR الأصلي A: شراء Boom داخل[p−0.55A,p−0.45A]، وبيع Crash داخل[p+0.45A,p+0.55A]. تثبت الحدود والإبطال وقت الإصدار؛ الإبطال 0.25A خارج الطرف المعاكس. يراقب فورًا، ويتفعل بعد دقيقة، وينتظر سعرًا مرصودًا داخل المنطقة خلال 15 دقيقة، ثم يدخل بالسعر الأصلي في الثانية التالية. لا يُنشأ لمس افتراضي عندما يقفز السعر فوق المنطقة.", "",
        "وقف 2A من سعر الدخول، ومدة 15 دقيقة من اللمس، وأولوية للوقف عند نهاية المهلة ثم خروج بالثانية التالية. التعرض واحد لكل ذراع، والنتيجة المجهولة تحجز أقصى المدة المخططة. التكلفة المفترضة 0.10A لكل المسار والتطهير 31 دقيقة؛ R وحدة مخاطرة 2A، وليست ربحًا نقديًا. لم تُستخدم الأسعار أو ATR المحوّلان في الدخول أو الخروج أو التكلفة.", ""]
    for part in PARTS:
        text += ["", f"## {PART_LABELS[part]}", "",
            "| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        for symbol in SYMBOLS:
            for family in FAMILIES:
                row = index[(symbol, family, part)]
                metric = row["metrics"]
                counts = disposition_counts(row, index)
                text.append(f"| {symbol} | {family} | {metric['completed']} | {metric['active_days']} | {counts['known_nonfill']} | {counts['exposure_skipped']} | {row['replay_audit']['unknown']} | {number(metric['profit_factor'])} | {interval(row['profit_factor_inference']['weekly_profit_factor_ci95'])} | {number(metric['mean_net_R'], 4)} | {number(metric['mean_gross_R'], 4)} |")
    text += ["", "الجدول يعرض جميع 60 خلية؛ اتحاد التحقق يعيد عرض الصفقات نفسها في النوافذ الثلاث ولا يمثل صفقات إضافية. لا تُجمع الأذرع أو الرموز أو الفترات للوصول إلى حد العينة.", "",
        "## التفوق على المراجع المطلوبة في الاختبار الزمني", "",
        "RAW_REGION يجب أن يتفوق على CLOCK وRAW_TIMED؛ وHYBRID_REGION على CLOCK وHYBRID_TIMED وRAW_REGION. فواصل الثقة من 9999 إعادة سحب بنفس البذرة 20261008، يوميًا ومع كتل 7 أيام. الأرقام التالية فروق متوسط R لكل صفقة مكتملة؛ لا تمثل احتمالات ربح معايرة.", "",
        "| الرمز | الذراع | الفترة | المرجع | فرق المتوسط | CI95 يومي للفرق | CI95 أسبوعي للفرق |",
        "|---|---|---|---|---:|---|---|"]
    for row in heldout:
        comparisons = {"CLOCK": {"day": row["day_inference"], "week": row["weekly_inference"]}, **row["target_comparisons"]}
        for reference, values in comparisons.items():
            day, week = values["day"], values["week"]
            text.append(f"| {row['symbol']} | {row['variant']} | {row['partition']} | {reference} | {number(day['baseline_difference'], 4)} | {interval(day['baseline_difference_ci95'])} | {interval(week['weekly_difference_ci95'])} |")
    text += ["", "## قرار الشروط التاريخية", "",
        "الشروط تشمل PF≥1.5 و1000 نتيجة مكتملة و60 يومًا نشطًا لكل رمز/نموذج/فترة، ونجاح التطوير، وغياب نتائج المناطق المجهولة في النموذج ومراجعه، وفواصل ثقة موجبة للمتوسط والتفوق، وحد PF السفلي>1 يوميًا وأسبوعيًا، وتصحيح Holm لثماني مقارنات. تضاف استقرارية الأثلاث الزمنية، وخسارة إغلاق قصوى≤10%، ومتوسط موجب مع ضعف التكلفة. حتى اجتيازها جميعًا يبقى بحثًا تاريخيًا تكيفيًا.", "",
        "| الرمز | الذراع | الفترة | أهلية التطوير | اجتاز كل الشروط | Holm p | أسباب عدم الاجتياز |",
        "|---|---|---|---|---|---:|---|"]
    for row in heldout:
        gate = row["qualification"]
        reasons = ", ".join(gate["historical_rejection_reasons"]) or "لا رفض تاريخي؛ يلزم دليل جديد"
        text.append(f"| {row['symbol']} | {row['variant']} | {row['partition']} | {gate['development_eligible']} | {gate['historical_criteria_passed']} | {number(row['holm_p'], 4)} | {cell(reasons)} |")
    text += ["", "| الرمز | الذراع | الفترة | متوسط R بتكلفة 0.20ATR | وسيط الاحتفاظ بالدقائق | أقصى تراجع إغلاقات توضيحي |",
        "|---|---|---|---:|---:|---:|"]
    for row in heldout:
        metric = row["metrics"]
        drawdown = metric["closed_trade_max_drawdown"]
        text.append(f"| {row['symbol']} | {row['variant']} | {row['partition']} | {number(row['double_cost_metrics']['mean_net_R'], 4)} | {number(metric['median_holding_minutes'], 2)} | {number(100 * drawdown if drawdown is not None else None, 2)}% |")
    text += ["", "التراجع توضيحي على الصفقات المغلقة فقط وبالمخاطرة الكسرية المحفوظة؛ لا يقيس خسارة عائمة فعلية أو مالًا متداولًا. الأثلاث والنتائج غير المعرفة وعدد إعادات السحب غير المعرفة محفوظة كاملة فيresults.json.", "",
        "## التدقيق المستقل وحدوده", "",
        f"حالة التدقيق PASS؛ عدد الفحوص المحفوظ {audit.get('checks', 'غير مذكور')}. هذه أعداد عمل المدقق، وليست صفقات مستقلة إضافية. تُربط نسخة النتيجة والتصريح بحدودهما الدقيقة أدناه.", "",
        f"أعاد المدقق {audit['models_checked']} نموذجًا، و{audit['training_CLOCK_paths_checked_including_overlapping_prefixes']:,} تصرفًا تدريبيًا ضمن فترات متداخلة، و{audit['evaluation_region_dispositions_checked']:,} منطقة اختبار ودرجة إصدار. راجع كل {audit['cells_checked']} خلية و{audit['paired_day_week_comparisons_checked']} مقارنة يومية/أسبوعية، وطابق {audit['frozen_timed_and_CLOCK_reference_cells_checked']} خلية CLOCK ومرجع زمني مع التجربة السابقة. قرأ {audit['native_quote_rows_checked']:,} سعرًا أصليًا وأبقى {audit['missing_seconds_preserved']} ثانية مجهولة.", "",
        "النطاق الذي يعلنه المدقق صراحة:", ""]
    text += ["- " + cell(item) for item in textual_scope(provenance["scope"])]
    text += ["", "الحدود أو الأجزاء غير المتحقق منها التي يعلنها:", ""]
    text += ["- " + cell(item) for item in textual_scope(provenance["limitations"])]
    text += ["", "التحقق البرمجي: نجحت 122 حالة قبل القياس بعد إصلاح شروط المراجع الإضافية، و73 حالة خاصة بالمدقق. التشغيل الكامل النهائي نجح في 2627 حالة مع تحذير Starlette موجود سابقًا؛ حالات المدقق ضمن هذا العدد، ولا تُضاف إليه. هذه نتائج محلية، وحالة CI على النسخة المنشورة تُثبت منفصلًا.", "",
        "تنفيذ CFD الفعلي، الفروق التاريخية المقاسة، تكاليف وسيط فعلية، الربح النقدي، والتجربة الورقية الاستباقية: **NOT TESTED**. نجاح التدقيق لا يبدل هذه الحدود ولا ينشئ أفضلية قابلة للتداول.", "",
        "LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false.", "",
        "## الملفات والبصمات", "",
        "- [البروتوكول](PROTOCOL.md)، [التصريح](declaration.json)، [كل 60 خلية](results.json)، [التدقيق المستقل](independent_audit.json)، [سجل المراجعة والتحقق](review_and_validation_notes.md).",
        "- ملفات training_labels/training_events/models تحفظ المجتمع المشروط وكل تصرفات التدريب؛ ملفات signals/events/ledger تحفظ الفرص والمناطق والدخول والصفقات. الملفات الأصلية والخصائص المخبأة تبقى محلية ببصماتSHA256.",
        "- [مرجع الهدف الزمني السابق](../hybrid_event_regions_20261008/REPORT.ar.md)، [صيغة الـ44 المحوّلة المرفوضة](../representation_regions_20261008/REPORT.ar.md).",
        f"- declaration SHA256: `{provenance['declaration_sha256']}`",
        f"- results SHA256: `{provenance['results_sha256']}`",
        f"- audit SHA256: `{provenance['audit_sha256']}`", ""]
    return "\n".join(text), {"cells": len(result["rows"]), "historical_criteria_passed": passed, "heldout_new_cells": len(heldout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    values = load_report_inputs(args.folder)
    content, summary = render(*values)
    output = repository_path(args.folder) / "REPORT.ar.md"
    if not args.check_only:
        with output.open("x", encoding="utf-8") as target:
            target.write(content)
    print(json.dumps({**summary, "report": str(output.relative_to(ROOT)), "written": not args.check_only}))


if __name__ == "__main__":
    main()
