# بحث متعدد الفريمات على أسعار bid/ask

جُمّد البروتوكول في 14 أكتوبر 2026، الساعة13:52:27UTC، قبل قياس الاستراتيجيات
الثلاث في هذه الجولة. بدأت طيات التطوير والتحقق فعليًا؛ نتائجها لم تكتمل عند
كتابة هذه الصفحة. لا توجد استراتيجية ربحية مؤهلة أو توصية دخول حاليًا.

الفرضيات: `CRT_SWEEP_RECLAIM` و`SQUEEZE_BREAKOUT` و`PULLBACK_CONTINUATION`،
لكل من Boom500 وCrash500. تجمع H4/H1 للسياق وM15 للإعداد وM5 للكسر/التأكيد
وM1 لتوقيت القرار. كل قاعدة محددة مسبقًا؛ لا بحث عن عشرات العتبات لإظهار فائز.

التنفيذ يستخدم أسعار bid/ask التاريخية المقاسة وفرضيات تأخير وانزلاق ثابتة،
مع معاملة الفجوات كمجهولة. الرسوم التاريخية والتعبئات الفعلية غير موثقة؛
`cash_net=null` و`profit_qualified=false` حتى عند ظهور نتيجة شرطية موجبة.

الفترة150يومًا من2025-01-02 إلى2025-06-01 حصريًا. أول120يومًا تطوير معاد
استخدامه بعد فشل الجولة السابقة؛ لا نسميه دليلًا مستقلًا جديدًا بالكامل.
النهائي30يومًا ظل غير مقاس استراتيجيًا في الجولة السابقة، ولا يُفتح إلا لقاعدة
تجتاز التطوير ويثبت اختيارها. لا تُغيَّر الملفات المجمدة بعد ظهور النتائج.

## الأدلة

- [البروتوكول المجمد](PROTOCOL_FROZEN.ar.md).
- [تصريح القياس والبصمات](measurement_declaration.json):
  `a3f9b73838b63abfcb0e85cc0fce076c5e5ae7c31fd8545ad05038f208dbe0e6`.
- [الفحص قبل القياس](premeasurement_validation.json):3515اختبارًا كاملًا
  و164اختبارًا مركزًا؛ تحذيرStarletteواحد موجود سابقًا.
- [المراجعة المستقلة قبل القياس](premeasurement_review.json): سلامة السببية
  وربط الأدلة والبوابات؛ ليست إثباتًا للربح.

Uمن8ملاءماتTRAINسابقة مثبتة، ولا ندّعي إعادة حسابها. مرجعCLOCKواحد لكل
رمز/مرحلة/سيناريو يستخدمه الثلاثة. العائلة الإحصائية24مطالبة تضم12مطالبةv1
غير مقاسة، ولا تعالج كل البحث التكيفي السابق. عدادات استبعادrun تخص الإشارات
المستهلكة فيه؛ مرشحالحدود الخارجي لا يسجل عدادات استبعاد إضافية.

## التشغيل المحلي

يتطلب أرشيف الأسعار المحلي المحتفظ به وإيصالاتv1 المجمدة؛ لا تُحمّل أسعار أو
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
حسابًا جديدًا. فشل التطوير يمنع قراءة أسعار النهائي؛UNKNOWNيوقف بقية المرحلة
ويجعل مجاميعها الأساسيةnull. لا تُحسب خسارة غير معروفة كصفر.

الرايات الثماني دائمًاfalse: `LIVE_TRADING`, `READY_FOR_LIVE`, `LIVE_ALLOWED`,
`OPENED_TRADES` ونظائرها الصغيرة. لا أوامر تداول أو تغيير حساب أو تصدير بياناته.
