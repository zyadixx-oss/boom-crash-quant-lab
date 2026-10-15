# بحث متعدد الفريمات على أسعار bid/ask

جُمّد البروتوكول في 14 أكتوبر 2026 ، الساعة 13:52:27UTC ، قبل قياس الاستراتيجيات
الثلاث في هذه الجولة. اكتملت طيات التطوير والتحقق عند 14:02:34UTC:
64 حسابًا فعليًا، وصفر قواعد مؤهلة لفتح النهائي. لا توجد استراتيجية ربحية مؤهلة
أو توصية دخول حاليًا. [التقرير بالأرقام وأسباب الاستبعاد](DEVELOPMENT_REPORT.ar.md).

الفرضيات: `CRT_SWEEP_RECLAIM` و`SQUEEZE_BREAKOUT` و`PULLBACK_CONTINUATION`،
لكل من Boom500 و Crash500. تجمع H4/H1 للسياق و M15 للإعداد و M5 للكسر/التأكيد
و M1 لتوقيت القرار. كل قاعدة محددة مسبقًا؛ لا بحث عن عشرات العتبات لإظهار فائز.

التنفيذ يستخدم أسعار bid/ask التاريخية المقاسة وفرضيات تأخير وانزلاق ثابتة،
مع معاملة الفجوات كمجهولة. الرسوم التاريخية والتعبئات الفعلية غير موثقة؛
`cash_net=null` و`profit_qualified=false` حتى عند ظهور نتيجة شرطية موجبة.

الفترة 150 يومًا من 2025-01-02 إلى 2025-06-01 حصريًا. أول 120 يومًا تطوير معاد
استخدامه بعد فشل الجولة السابقة؛ لا نسميه دليلًا مستقلًا جديدًا بالكامل.
النهائي 30 يومًا ظل غير مقاس استراتيجيًا في الجولة السابقة، ولا يُفتح إلا لقاعدة
تجتاز التطوير ويثبت اختيارها. لا تُغيَّر الملفات المجمدة بعد ظهور النتائج.

## الأدلة

- [البروتوكول المجمد](PROTOCOL_FROZEN.ar.md).
- [تصريح القياس والبصمات](measurement_declaration.json):
  `a3f9b73838b63abfcb0e85cc0fce076c5e5ae7c31fd8545ad05038f208dbe0e6`.
- [الفحص قبل القياس](premeasurement_validation.json):3515 اختبارًا كاملًا
  و 164 اختبارًا مركزًا؛ تحذير Starlette واحد موجود سابقًا.
- [المراجعة المستقلة قبل القياس](premeasurement_review.json): سلامة السببية
  وربط الأدلة والبوابات؛ ليست إثباتًا للربح.
- [نتائج التطوير](DEVELOPMENT_RESULTS.json): عينات المرشحين 6–39 صفقة فقط.
- [تدقيق WF1 المستقل](independent_WF1_accounting_audit.json):586 صفقة مكتملة
  من السجلات المحفوظة؛ لا إعادة إصدار للإشارات أو إثبات لتعبئة وسيط.
- [تشخيص عدم اليقين اللاحق](development_uncertainty_posthoc.json): بيانات تطوير
  معاد استخدامها، غير مسجل قبل القياس، لا يغيّر الاختيار ولا يفتح النهائي.

U من 8 ملاءمات TRAIN سابقة مثبتة، ولا ندّعي إعادة حسابها. مرجع CLOCK واحد لكل
رمز/مرحلة/سيناريو يستخدمه الثلاثة. العائلة الإحصائية 24 مطالبة تضم 12 مطالبة v1
غير مقاسة، ولا تعالج كل البحث التكيفي السابق. عدادات استبعاد run تخص الإشارات
المستهلكة فيه؛ مرشحالحدود الخارجي لا يسجل عدادات استبعاد إضافية.

## التشغيل المحلي

يتطلب أرشيف الأسعار المحلي المحتفظ به وإيصالات v1 المجمدة؛ لا تُحمّل أسعار أو
حسابات من الشبكة بالأوامر التالية. البيانات الكبيرة والسجل المالي الخاص غير منشورين.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py run --stage WF1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py run --stage WF2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py run --stage WF3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py run --stage VALIDATION
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py select
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py run --stage FINAL
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_multiframe_quote_study.py summarize
```

يمنع القفل تشغيل مرحلتين معًا. الإيصال المعاد استخدامه موسوم`REUSED`، وليس
حسابًا جديدًا. فشل التطوير يمنع قراءة أسعار النهائي؛ UNKNOWN يوقف بقية المرحلة
ويجعل مجاميعها الأساسية null. لا تُحسب خسارة غير معروفة كصفر.

الرايات الثماني دائمًا false: `LIVE_TRADING`, `READY_FOR_LIVE`, `LIVE_ALLOWED`,
`OPENED_TRADES` ونظائرها الصغيرة. لا أوامر تداول أو تغيير حساب أو تصدير بياناته.
