#!/usr/bin/env python3
"""Publish all fixed hybrid arms and economic limits after independent audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "docs/hybrid_event_regions_20261008"


def number(value, digits=3):
    return "غير معرّف" if value is None else f"{value:.{digits}f}"


def interval(values):
    return "[" + ", ".join(number(v) for v in values) + "]"


def main():
    result = json.loads((FOLDER / "results.json").read_text())
    audit = json.loads((FOLDER / "independent_audit.json").read_text())
    assert audit["passed"] and not audit["errors"] and len(result["rows"]) == 72
    heldout = [r for r in result["rows"] if r["variant"] != "CLOCK" and r["partition"] in ("final_test", "later180")]
    passed = sum(r["qualification"]["historical_criteria_passed"] for r in heldout)
    rows = ["# مؤشرات محوّلة مع عمر القفزة الأصلية: اختبار مناطق ودخول زمني", "",
        f"**اكتمل التشغيل والتدقيق. عدد الحالات المستوفية لكل الشروط التاريخية: {passed}/16. لا توجد ترقية إلى استراتيجية رابحة معتمدة؛ التاريخ مكشوف سابقًا وQUALIFIED=false.**", "",
        "هذه فرضية تكيفية منفصلة عن صيغة44المحوّلة المرفوضة. تستخدم HYBRID44 الخصائص المحوّلة الـ43 الأخرى كما هي، وتستخدم عمر آخر شمعة M5 كبيرة مرصودة على السعر الأصلي. RAW44 يستخدم العمر الأصلي نفسه. العمر المجهول يبقى مجهولًا. لا تتغير النتيجة السابقة ذات الصفر من الفرص المشتركة، ولا تتحول الأسعار المعروفة إلى fresh OOS.", "",
        "دُمج H4/H1/M15/M5/M1، مع تدريب أول40/50/60% والتحقق في10%تالية لكل نافذة، ثم تدريب70%واختبار30%و180يومًا لاحقة دون إعادة ملاءمة. كاشف التغيرات الكبيرة يثبت من تدريب كل نافذة فقط. مصادر722ملفًا و62,380,130سعرًا أصليًا، مع670ثانية ناقصة. الفجوات تعيد تاريخ كل فريم؛ لا تعويض أو تقريب.", "",
        "قبل القياس نجحت192حالة، وثُبّت التصريح والكود والمصادر والبروتوكول. أربعة إعدادات تدريب زوجي لكل رمز؛ ridge بعقوبة.1 وعتبةq75لدرجات التدريب موجبة فقط، مع مقياس مستقل لكل ذراع. الهدف المشترك هو netRلدخول زمني أصلي، ولا تدخل النتائج المجهولة التدريب.", "",
        "## التدريب الفعلي", "",
        "فترات التدريب متداخلة؛ لا تُجمع أعدادها بوصفها صفقات مستقلة أو تحققًا خارج العينة.", "",
        "| الرمز | الإعداد | فرص أصلية | فرص Hybrid | فرص مشتركة | مكتملة | مجهولة | الحالة |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for symbol, data in result["symbols"].items():
        for fold, info in data["folds"].items():
            model = json.loads((ROOT / info["fit"]["path"]).read_text()); clock = model["training_clock"]
            rows.append(f"| {symbol} | {fold} | {clock['raw_valid_clock_rows']} | {clock['transformed_valid_clock_rows']} | {clock['common_clock_opportunities']} | {model['training_completed']} | {model['training_unknown']} | {model['status']} |")
    rows += ["", "## قواعد المناطق والتنفيذ", "",
        "REGION: سعر الإغلاق الأصليpوATRالأصليA يثبتان النطاق مسبقًا: Boom[p−.55A,p−.45A]؛ Crash[p+.45A,p+.55A]. إبطال إضافي.25A، مراقبة فور الإصدار، تفعيل بعد دقيقة ولمسة حقيقية خلال15دقيقة، ثم السعر التالي بالثانية للدخول. وقف2A وخروج بعد15دقيقة من اللمسة، مع أولوية الوقف عند نهاية المهلة وخروج بالسعر التالي وخسارة غير مقيدة اصطناعيًا.", "",
        "TIMED: الإشارات نفسها دون انتظار المنطقة؛ تأخير دقيقة ثم دخول الثانية التالية، خروج عند issue+16دقيقة بالثانية التالية، ووقف2ATR. في التطبيقين: تكلفة افتراضية كلية.10ATR، تطهير31دقيقة، تعرض واحد لكل ذراع، والأسعار الأصلية فقط للعائد والتنفيذ. Unknownيبقى مجهولًا ويحجز التعرض المخطط. CLOCKيستخدم نفس التوافر المشترك دون بوابة درجة.", ""]
    for part, label in (("walk_forward_combined", "التحقق المتدرج المجمع بعد ثلاث ملاءمات مستقلة"),
                        ("final_test", "الجزء الأخير30%من الفترة الأقدم"), ("later180", "فترة180يومًا اللاحقة")):
        rows += [f"## {label}", "", "كل صف ذراع مستقل؛ لا تُجمع الأذرع أو الرموز للوصول إلى العينة المطلوبة.", "",
            "| الرمز | التطبيق | الذراع | مكتملة | أيام نشطة | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ |",
            "|---|---|---|---:|---:|---:|---:|---|---:|"]
        for r in result["rows"]:
            if r["partition"] != part: continue
            m = r["metrics"]
            rows.append(f"| {r['symbol']} | {r['endpoint']} | {r['variant']} | {m['completed']} | {m['active_days']} | {r['replay_audit']['unknown']} | {number(m['profit_factor'])} | {interval(r['profit_factor_inference']['weekly_profit_factor_ci95'])} | {number(m['mean_net_R'], 4)} |")
    rows += ["", "## القرار على كل ذراع", "",
        "المطلوب PFصافٍ≥1.5، و1000نتيجة مكتملة و60يومًا نشطًا لكل رمز/نموذج/تطبيق وفترة منفردة، مع نجاح التطوير وثبات الثلث الزمني وفواصل المتوسط والتفوق على المرجع وتصحيح16مقارنة وقيود الخسارة والكلفة المضاعفة. الدرجة المستمرة ليست احتمالًا معايرًا.", "",
        "| الرمز | التطبيق | الذراع | الفترة | أهلية التطوير | شروط التاريخ كلها | Holm p | أسباب الرفض |",
        "|---|---|---|---|---|---|---:|---|"]
    for r in heldout:
        q = r["qualification"]
        rows.append(f"| {r['symbol']} | {r['endpoint']} | {r['variant']} | {r['partition']} | {q['development_eligible']} | {q['historical_criteria_passed']} | {number(r['holm_p'], 4)} | {', '.join(q['historical_rejection_reasons']) or 'لا يوجد رفض تاريخي؛ يلزم تحقق جديد'} |")
    rows += ["", "## التدقيق المستقل وحدوده", "",
        f"نجح التدقيق في{audit['checks']:,}فحصًا دون أخطاء، وأعاد{audit['models_checked']}نموذجًا و{audit['issued_scores_checked']:,}إشارة/درجة، و{audit['native_paths_checked_including_overlapping_training_prefixes']:,}فرصة/مسارًا تتضمن فترات تدريب متداخلة. هذه أعداد عمل المدقق، وليست صفقات مستقلة جديدة.", "",
        "أُعيد حساب أهداف التدريب من المصادر الأصلية، والمقياس ومعاملاتridgeبحل مربعات صغرى معزز مستقل، والعتبات وعضوية الإشارات، والنطاقات والثانية الفعلية للدخول/الوقف/الخروج، وR/PFوفواصلPFوالمتوسط والفروق يوميًا وأسبوعيًا وتصحيحHolm. تحقق أيضًا ثبات الخصائص الأصلية والـ43المحوّلة مقابل الملفات السابقة، وعمر الحدث الأصلي والتوافر المشترك المحفوظ.", "",
        "لم يُعد التدقيق اشتقاق صيغ الخصائص الـ43 أو شرط صلاحيتها الأصلي من الصفر، ولا منطق بوابات التأهيل أو حسابdrawdownوالأثلاث. اختبارات السببية والفجوات مستقلة عن هذا النطاق، وتدقيق السلسلة/العمر السابق يبقى محفوظًا. تنفيذCFDوالفروق والكلفة التاريخية المقاسة والربح النقدي والتجربة الورقية الاستباقية: NOT TESTED.", "",
        "فحص مستقل مبكر لأول تدريبBoomطابق2416هدفًا مكتملًا ومعاملات النموذجين. نجحت2489حالةbackendكاملة مع تحذيرStarletteموجود، و19حالة للمدقق مستقلة. لم يتغير كود القياس أو العتبات بعد التشغيل. نتائج CRT/Fibonacci/Trend السابقة محفوظة.", "",
        "فشل التدقيق الأول عند حقل غائب لأربع مقارناتHYBRID44معRAW44في اتحاد التحقق المتدرج؛ أكمل قبل ذلك مراجعة المسارات والنماذج. حُفظت محاولة الفشل وبصمة مدققها. النتيجة الأصلية بقيت بالـSHAنفسها؛ أُضيفت المقارنات الوصفية الناقصة من الدفاتر نفسها داخل ملحق التدقيق، دون إشارات أو معلمات جديدة. المقارنات المنفردة وخلايا الاختبار النهائية وتصحيحها مكتملة في النتيجة الأصلية.", "",
        "نسخة المصدر الدقيقة التي استُخدمت في الفحص المبكر محفوظة فيprecheck_verifier_snapshot.py.txt؛ تطور المدقق الكامل بعدها بإضافة فحوص التوافر ومعالجة نقص التقرير. هذا لا يعدل تجربة السوق المجمدة.", "",
        "## ملحق المقارنات المجمعة الناقصة في النتيجة الأصلية", "",
        "| الرمز | التطبيق | فرق متوسطHYBRID44عنRAW44: CI95أسبوعي | pأسبوعي |",
        "|---|---|---|---:|"]
    for item in audit["supplemental_union_comparisons"]:
        week = item["week"]
        rows.append(f"| {item['symbol']} | {item['endpoint']} | {interval(week['weekly_difference_ci95'])} | {number(week['weekly_p'], 4)} |")
    rows += ["",
        "LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false. هدف الربح يبقى غير متحقق حتى تستوفي سياسة فعلية شروط الدليل المحددة.", "",
        "## الملفات", "",
        "- [البروتوكول](PROTOCOL.md)، [التصريح](declaration.json)، [كل72خلية](results.json)، [التدقيق](independent_audit.json)، [الفحص المبكر](independent_precheck.json).",
        "- ملفاتsignals/events/ledger/training_labelsوmodelsتحفظ الإشارات والمناطق والمجهول والصفقات والتدريب. معاملاتHYBRID44تظل تحت اسمTRANSFORMED44داخليًا في محركridgeالمجمد؛ mappingموثق صراحة.",
        "- الخصائص الكاملة محلية تحتdata/hybrid_event_regions_20261008معSHA256؛ المصادر المحلية ضرورية لإعادة الحساب.",
        "- [صيغة44المحوّلة المرفوضة](../representation_regions_20261008/REPORT.ar.md)، [المناطق السابقة](../zone_study_20261008/REPORT.ar.md)، [تعريفات الحركة الاثنا عشر](../zone_excursions_20261008/REPORT.ar.md).", ""]
    with (FOLDER / "REPORT.ar.md").open("x") as f: f.write("\n".join(rows))
    print(json.dumps({"report": str((FOLDER / "REPORT.ar.md").relative_to(ROOT)), "historical_criteria_passed": passed, "audit_checks": audit["checks"]}))


if __name__ == "__main__": main()
