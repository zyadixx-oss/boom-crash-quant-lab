"""Produce a readable supplement from already verified descriptive outputs."""
import json
from pathlib import Path

from verify import digest

HERE = Path(__file__).resolve().parent


def main():
    r = json.loads((HERE / "validation_sensitivity.json").read_text())
    v = json.loads((HERE / "independent_verification.json").read_text())
    if v["status"] != "PASS" or v["errors"]:
        raise ValueError("Verified numerical descriptive outputs required")
    lines = ["# قراءة وصفية لحساسية الإعدادات", "",
             "المصدر الوحيد للحساب هو الأعمدة المسموح بها من validation. لا قراءة لنتائج final أو أسباب قبوله، ولا إعادة اختيار أو ترتيب أو اختبار استراتيجية. جميع الأرقام أدناه تصف نقاط معلمات محفوظة؛ لا تمثل عينة صفقات مستقلة.", "",
             "تحققت أداة منفصلة عن المولد من المصدر والمصفوفات والملخصات: **1,762,579 فحصًا، صفر اختلاف**. نجح 16 اختبارًا للمولد و3 اختبارات للفاحص قبل التنفيذ؛ تشمل الحدود: شبكة خمسة أبعاد، aliases، نقص نقطة، PF=NULL، تعارض التكرارات، منع حقول final، هويات الإعدادات، والفروق المتجاورة.", "",
             "التغطية: **1,300 مجموعة، 41,600 محاولة، 38,400 قياس علمي مختلف، 3,200 aliases؛ صفر نقص**. لا تغيير للعائلات الـ13 أو الإعدادات الـ32 الأصلية. الشبكة 2×2×2×2×2، ويزال threshold غير المستخدم من الهوية العلمية لعائلتي ICT/trend فقط؛ لذلك لكل خلية 416 محاولة و384 قياسًا علميًا.", "",
             "توزيع صافي الربح المفترض للنقاط العلمية: **4,754 موجبًا، 30,162 سالبًا، 3,484 صفرًا**. الوسيط −774.009738، والمدى تقريبًا −10,000 إلى 1,327.479106 وحدة حساب مفترضة. وسيط PF للقيم المعرفة 0.679954؛ توجد **3,884 قيمة PF=NULL** محفوظة دون تحويل إلى صفر أو مالانهاية. النهايات العشرية الأدنى من −10,000 بفارق نحو 1e−11 محفوظة في البيانات الأصلية، ولم تُقص أو تُصحح هنا.", "",
             "## الفروق بين قيم متجاورة", "",
             "تُغيّر المقارنة محورًا واحدًا مع تثبيت بقية المحاور الفعالة. من 104,000 حافة معلنة، توجد 94,400 مقارنة فعلية؛ استُبعدت 6,400 حافة مكررة علميًا و3,200 حافة للمحور الخامل. الجدول يتبع ترتيب المحاور الأصلي وليس ترتيب أداء:", "",
             "| المحور | مقارنات فعلية | وسيط الفرق الموقّع في صافي الربح | وسيط الفرق المطلق | انقلاب موجب/سالب صريح |", "|---|---:|---:|---:|---:|"]
    for axis in ("lookback", "threshold", "stop_atr", "target_atr", "max_hold_bars"):
        d = v["descriptive_axis_distributions"][axis]
        lines.append(f"| {axis} | {d['net_delta']['points']:,} | {d['net_delta']['median']:.6f} | {d['absolute_net_delta']['median']:.6f} | {d['strict_net_sign_reversals']:,} |")
    lines.extend(["", "الفرق الموقّع هو قيمة المقياس عند الإعداد الأعلى ناقص قيمته عند الأدنى؛ ليس ربح صفقة أو توصية بتغيير الإعداد. قد يعني الفرق الموجب تقليل خسارة مع بقاء النقطتين سالبتين. stop_atr يغير أيضًا الكمية وفق المخاطرة المثبتة؛ الفروق ليست تجارب سببية مستقلة. لا تُعدّل الاختيارات أو تُستحدث حدود قبول من هذا التحليل.", "",
                  "## توزيع الفرضيات دون ترتيب جديد", "",
                  "| العائلة | نقاط علمية | صافي موجب | سالب | صفر | وسيط صافي الربح المفترض | PF=NULL |", "|---|---:|---:|---:|---:|---:|---:|"])
    for family in r["family_distributions"]:
        # JSON was canonically sorted. Lexical order is deliberate, not ranking.
        d = r["family_distributions"][family]
        net, pf = d["metrics"]["net_profit"], d["metrics"]["profit_factor"]
        lines.append(f"| {family} | {d['scientific_points']:,} | {net['positive']:,} | {net['negative']:,} | {net['zero']:,} | {net['median']:.6f} | {pf['null']:,} |")
    lines.extend(["", "الجدول أبجدي بحسب اسم العائلة، ولا يعيد ترتيب السياسات. تُحفظ كل مجموعة symbol/timeframe/family منفصلة في المصفوفات والملخصات؛ جمع توزيعات النقاط هنا ليس جمع صفقات أو PF أو أرباح في استراتيجية واحدة.", "",
                  "الفرضية الإضافية empirical_conditional_return لها 3,200 نقطة: 133 موجبة،1,147 سالبة،1,920 بلا صفقات وبصافي صفر وPF=NULL. لذلك وسيطها الصفري لا يثبت ثباتًا أو ربحية. هذه مساهمة موثقة للفرضية المعلنة مسبقًا، ولا تثبت ابتكار عائلة في كل دورة أو تأهيلًا خارج العينة.", "",
                  "## حدود الإغلاق", "",
                  "تقوي المصفوفات دليل تحليل استقرار الإعدادات 8.7 وتمييز الإعدادات الجديدة والتكرار 6.4 ووصف مساهمة الفرضية 3.2. لا تحول هذه البنود آليًا إلى VERIFIED، ولا تستبدل متطلبات الابتكار التاريخي أو التكاليف الحقيقية أو الثبات المستقبلي. النتائج من بيانات تطوير مكشوفة وبنموذج تكلفة مفترض؛ ولا يوجد هنا حكم على ربحية قابلة للتنفيذ.", "",
                  "لم تتغير مدخلات التحليل المثبتة أو الكود العلمي أو حالة الحملة أو النتائج أو اللوحة. كل أعلام التداول الحقيقي ونظائرها الصغيرة false. صفر إشارات أو backtests أو أوامر جديدة."])
    with (HERE / "DESCRIPTIVE_FINDINGS.ar.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    files = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name != "delivery_manifest.json":
            files[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {"schema_version": 1, "scope": "Descriptive validation sensitivity only; producer, verifier, tests and original output pins", "files": files,
                "producer_receipt_sha256": digest(HERE / "receipt.json"), "independent_verification_sha256": digest(HERE / "independent_verification.json"),
                "source_sha256": r["input"]["sha256"], "final_results_read": False, "new_backtests": 0, "reranking": False,
                "requirements_auto_promoted": False}
    with (HERE / "delivery_manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"manifest": str(HERE / "delivery_manifest.json"), "manifest_sha256": digest(HERE / "delivery_manifest.json"), "files": len(files), "independent_checks": v["check_count"], "errors": len(v["errors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
