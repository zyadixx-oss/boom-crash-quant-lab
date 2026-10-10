# مؤشرات محوّلة مع عمر القفزة الأصلية: اختبار مناطق ودخول زمني

**اكتمل التشغيل والتدقيق. عدد الحالات المستوفية لكل الشروط التاريخية: 0/16. لا توجد ترقية إلى استراتيجية رابحة معتمدة؛ التاريخ مكشوف سابقًا وQUALIFIED=false.**

هذه فرضية تكيفية منفصلة عن صيغة44المحوّلة المرفوضة. تستخدم HYBRID44 الخصائص المحوّلة الـ43 الأخرى كما هي، وتستخدم عمر آخر شمعة M5 كبيرة مرصودة على السعر الأصلي. RAW44 يستخدم العمر الأصلي نفسه. العمر المجهول يبقى مجهولًا. لا تتغير النتيجة السابقة ذات الصفر من الفرص المشتركة، ولا تتحول الأسعار المعروفة إلى fresh OOS.

دُمج H4/H1/M15/M5/M1، مع تدريب أول40/50/60% والتحقق في10%تالية لكل نافذة، ثم تدريب70%واختبار30%و180يومًا لاحقة دون إعادة ملاءمة. كاشف التغيرات الكبيرة يثبت من تدريب كل نافذة فقط. مصادر722ملفًا و62,380,130سعرًا أصليًا، مع670ثانية ناقصة. الفجوات تعيد تاريخ كل فريم؛ لا تعويض أو تقريب.

قبل القياس نجحت192حالة، وثُبّت التصريح والكود والمصادر والبروتوكول. أربعة إعدادات تدريب زوجي لكل رمز؛ ridge بعقوبة.1 وعتبةq75لدرجات التدريب موجبة فقط، مع مقياس مستقل لكل ذراع. الهدف المشترك هو netRلدخول زمني أصلي، ولا تدخل النتائج المجهولة التدريب.

## التدريب الفعلي

فترات التدريب متداخلة؛ لا تُجمع أعدادها بوصفها صفقات مستقلة أو تحققًا خارج العينة.

| الرمز | الإعداد | فرص أصلية | فرص Hybrid | فرص مشتركة | مكتملة | مجهولة | الحالة |
|---|---|---:|---:|---:|---:|---:|---|
| BOOM600 | fit40 | 2418 | 2418 | 2418 | 2416 | 2 | FITTED |
| BOOM600 | fit50 | 3282 | 3282 | 3282 | 3280 | 2 | FITTED |
| BOOM600 | fit60 | 3740 | 3740 | 3740 | 3736 | 4 | FITTED |
| BOOM600 | fit70 | 4288 | 4288 | 4288 | 4283 | 5 | FITTED |
| CRASH600 | fit40 | 2418 | 2418 | 2418 | 2416 | 2 | FITTED |
| CRASH600 | fit50 | 3282 | 3282 | 3282 | 3280 | 2 | FITTED |
| CRASH600 | fit60 | 3740 | 3740 | 3740 | 3736 | 4 | FITTED |
| CRASH600 | fit70 | 4288 | 4288 | 4288 | 4283 | 5 | FITTED |

## قواعد المناطق والتنفيذ

REGION: سعر الإغلاق الأصليpوATRالأصليA يثبتان النطاق مسبقًا: Boom[p−.55A,p−.45A]؛ Crash[p+.45A,p+.55A]. إبطال إضافي.25A، مراقبة فور الإصدار، تفعيل بعد دقيقة ولمسة حقيقية خلال15دقيقة، ثم السعر التالي بالثانية للدخول. وقف2A وخروج بعد15دقيقة من اللمسة، مع أولوية الوقف عند نهاية المهلة وخروج بالسعر التالي وخسارة غير مقيدة اصطناعيًا.

TIMED: الإشارات نفسها دون انتظار المنطقة؛ تأخير دقيقة ثم دخول الثانية التالية، خروج عند issue+16دقيقة بالثانية التالية، ووقف2ATR. في التطبيقين: تكلفة افتراضية كلية.10ATR، تطهير31دقيقة، تعرض واحد لكل ذراع، والأسعار الأصلية فقط للعائد والتنفيذ. Unknownيبقى مجهولًا ويحجز التعرض المخطط. CLOCKيستخدم نفس التوافر المشترك دون بوابة درجة.

## التحقق المتدرج المجمع بعد ثلاث ملاءمات مستقلة

كل صف ذراع مستقل؛ لا تُجمع الأذرع أو الرموز للوصول إلى العينة المطلوبة.

| الرمز | التطبيق | الذراع | مكتملة | أيام نشطة | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ |
|---|---|---|---:|---:|---:|---:|---|---:|
| BOOM600 | REGION | CLOCK | 1501 | 45 | 5 | 0.941 | [0.832, 1.055] | -0.0240 |
| BOOM600 | REGION | RAW44 | 322 | 44 | 0 | 0.935 | [0.710, 1.206] | -0.0256 |
| BOOM600 | REGION | HYBRID44 | 222 | 40 | 0 | 0.970 | [0.732, 1.207] | -0.0117 |
| BOOM600 | TIMED | CLOCK | 1864 | 47 | 3 | 0.915 | [0.815, 1.018] | -0.0343 |
| BOOM600 | TIMED | RAW44 | 383 | 45 | 0 | 0.774 | [0.640, 0.928] | -0.0951 |
| BOOM600 | TIMED | HYBRID44 | 286 | 42 | 0 | 0.984 | [0.827, 1.195] | -0.0062 |
| CRASH600 | REGION | CLOCK | 1504 | 45 | 5 | 0.801 | [0.738, 0.869] | -0.0875 |
| CRASH600 | REGION | RAW44 | 346 | 42 | 0 | 0.700 | [0.578, 0.841] | -0.1458 |
| CRASH600 | REGION | HYBRID44 | 286 | 41 | 1 | 0.727 | [0.617, 0.843] | -0.1384 |
| CRASH600 | TIMED | CLOCK | 1864 | 47 | 3 | 0.802 | [0.718, 0.881] | -0.0851 |
| CRASH600 | TIMED | RAW44 | 406 | 42 | 0 | 0.736 | [0.599, 0.928] | -0.1264 |
| CRASH600 | TIMED | HYBRID44 | 342 | 41 | 1 | 0.713 | [0.619, 0.845] | -0.1385 |
## الجزء الأخير30%من الفترة الأقدم

كل صف ذراع مستقل؛ لا تُجمع الأذرع أو الرموز للوصول إلى العينة المطلوبة.

| الرمز | التطبيق | الذراع | مكتملة | أيام نشطة | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ |
|---|---|---|---:|---:|---:|---:|---|---:|
| BOOM600 | REGION | CLOCK | 1444 | 45 | 3 | 0.931 | [0.833, 1.048] | -0.0277 |
| BOOM600 | REGION | RAW44 | 297 | 43 | 2 | 1.011 | [0.794, 1.362] | 0.0047 |
| BOOM600 | REGION | HYBRID44 | 266 | 43 | 1 | 0.766 | [0.537, 1.022] | -0.0982 |
| BOOM600 | TIMED | CLOCK | 1810 | 45 | 2 | 0.933 | [0.840, 1.056] | -0.0266 |
| BOOM600 | TIMED | RAW44 | 363 | 44 | 1 | 0.960 | [0.789, 1.195] | -0.0170 |
| BOOM600 | TIMED | HYBRID44 | 336 | 43 | 0 | 0.775 | [0.617, 0.944] | -0.0941 |
| CRASH600 | REGION | CLOCK | 1448 | 45 | 3 | 0.887 | [0.819, 0.962] | -0.0459 |
| CRASH600 | REGION | RAW44 | 217 | 41 | 0 | 0.925 | [0.723, 1.127] | -0.0286 |
| CRASH600 | REGION | HYBRID44 | 174 | 35 | 0 | 1.014 | [0.768, 1.323] | 0.0055 |
| CRASH600 | TIMED | CLOCK | 1810 | 45 | 2 | 0.870 | [0.805, 0.936] | -0.0528 |
| CRASH600 | TIMED | RAW44 | 259 | 42 | 0 | 0.747 | [0.542, 0.937] | -0.1093 |
| CRASH600 | TIMED | HYBRID44 | 233 | 35 | 0 | 1.158 | [0.873, 1.425] | 0.0592 |
## فترة180يومًا اللاحقة

كل صف ذراع مستقل؛ لا تُجمع الأذرع أو الرموز للوصول إلى العينة المطلوبة.

| الرمز | التطبيق | الذراع | مكتملة | أيام نشطة | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ |
|---|---|---|---:|---:|---:|---:|---|---:|
| BOOM600 | REGION | CLOCK | 5394 | 157 | 8 | 0.858 | [0.795, 0.921] | -0.0581 |
| BOOM600 | REGION | RAW44 | 1088 | 155 | 1 | 0.841 | [0.704, 1.006] | -0.0689 |
| BOOM600 | REGION | HYBRID44 | 984 | 144 | 1 | 0.865 | [0.724, 1.031] | -0.0547 |
| BOOM600 | TIMED | CLOCK | 6635 | 157 | 7 | 0.849 | [0.791, 0.907] | -0.0619 |
| BOOM600 | TIMED | RAW44 | 1316 | 156 | 1 | 0.852 | [0.736, 0.988] | -0.0646 |
| BOOM600 | TIMED | HYBRID44 | 1221 | 146 | 1 | 0.891 | [0.777, 1.024] | -0.0441 |
| CRASH600 | REGION | CLOCK | 5353 | 157 | 8 | 0.935 | [0.874, 1.001] | -0.0262 |
| CRASH600 | REGION | RAW44 | 760 | 147 | 3 | 0.869 | [0.706, 1.053] | -0.0564 |
| CRASH600 | REGION | HYBRID44 | 687 | 138 | 0 | 0.982 | [0.799, 1.178] | -0.0073 |
| CRASH600 | TIMED | CLOCK | 6635 | 157 | 7 | 0.918 | [0.872, 0.962] | -0.0331 |
| CRASH600 | TIMED | RAW44 | 965 | 150 | 3 | 0.906 | [0.746, 1.080] | -0.0386 |
| CRASH600 | TIMED | HYBRID44 | 858 | 142 | 0 | 0.920 | [0.779, 1.069] | -0.0330 |

## القرار على كل ذراع

المطلوب PFصافٍ≥1.5، و1000نتيجة مكتملة و60يومًا نشطًا لكل رمز/نموذج/تطبيق وفترة منفردة، مع نجاح التطوير وثبات الثلث الزمني وفواصل المتوسط والتفوق على المرجع وتصحيح16مقارنة وقيود الخسارة والكلفة المضاعفة. الدرجة المستمرة ليست احتمالًا معايرًا.

| الرمز | التطبيق | الذراع | الفترة | أهلية التطوير | شروط التاريخ كلها | Holm p | أسباب الرفض |
|---|---|---|---|---|---|---:|---|
| BOOM600 | REGION | RAW44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| BOOM600 | REGION | HYBRID44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| BOOM600 | TIMED | RAW44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| BOOM600 | TIMED | HYBRID44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| BOOM600 | REGION | RAW44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |
| BOOM600 | REGION | HYBRID44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |
| BOOM600 | TIMED | RAW44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |
| BOOM600 | TIMED | HYBRID44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |
| CRASH600 | REGION | RAW44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| CRASH600 | REGION | HYBRID44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| CRASH600 | TIMED | RAW44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| CRASH600 | TIMED | HYBRID44 | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, Holm_not_significant, chronological_thirds_not_stable |
| CRASH600 | REGION | RAW44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |
| CRASH600 | REGION | HYBRID44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive |
| CRASH600 | TIMED | RAW44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |
| CRASH600 | TIMED | HYBRID44 | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive |

## التدقيق المستقل وحدوده

نجح التدقيق في1,240,052فحصًا دون أخطاء، وأعاد16نموذجًا و27,617إشارة/درجة، و82,690فرصة/مسارًا تتضمن فترات تدريب متداخلة. هذه أعداد عمل المدقق، وليست صفقات مستقلة جديدة.

أُعيد حساب أهداف التدريب من المصادر الأصلية، والمقياس ومعاملاتridgeبحل مربعات صغرى معزز مستقل، والعتبات وعضوية الإشارات، والنطاقات والثانية الفعلية للدخول/الوقف/الخروج، وR/PFوفواصلPFوالمتوسط والفروق يوميًا وأسبوعيًا وتصحيحHolm. تحقق أيضًا ثبات الخصائص الأصلية والـ43المحوّلة مقابل الملفات السابقة، وعمر الحدث الأصلي والتوافر المشترك المحفوظ.

لم يُعد التدقيق اشتقاق صيغ الخصائص الـ43 أو شرط صلاحيتها الأصلي من الصفر، ولا منطق بوابات التأهيل أو حسابdrawdownوالأثلاث. اختبارات السببية والفجوات مستقلة عن هذا النطاق، وتدقيق السلسلة/العمر السابق يبقى محفوظًا. تنفيذCFDوالفروق والكلفة التاريخية المقاسة والربح النقدي والتجربة الورقية الاستباقية: NOT TESTED.

فحص مستقل مبكر لأول تدريبBoomطابق2416هدفًا مكتملًا ومعاملات النموذجين. نجحت2489حالةbackendكاملة مع تحذيرStarletteموجود، و19حالة للمدقق مستقلة. لم يتغير كود القياس أو العتبات بعد التشغيل. نتائج CRT/Fibonacci/Trend السابقة محفوظة.

فشل التدقيق الأول عند حقل غائب لأربع مقارناتHYBRID44معRAW44في اتحاد التحقق المتدرج؛ أكمل قبل ذلك مراجعة المسارات والنماذج. حُفظت محاولة الفشل وبصمة مدققها. النتيجة الأصلية بقيت بالـSHAنفسها؛ أُضيفت المقارنات الوصفية الناقصة من الدفاتر نفسها داخل ملحق التدقيق، دون إشارات أو معلمات جديدة. المقارنات المنفردة وخلايا الاختبار النهائية وتصحيحها مكتملة في النتيجة الأصلية.

نسخة المصدر الدقيقة التي استُخدمت في الفحص المبكر محفوظة فيprecheck_verifier_snapshot.py.txt؛ تطور المدقق الكامل بعدها بإضافة فحوص التوافر ومعالجة نقص التقرير. هذا لا يعدل تجربة السوق المجمدة.

## ملحق المقارنات المجمعة الناقصة في النتيجة الأصلية

| الرمز | التطبيق | فرق متوسطHYBRID44عنRAW44: CI95أسبوعي | pأسبوعي |
|---|---|---|---:|
| BOOM600 | REGION | [-0.094, 0.115] | 1.0000 |
| BOOM600 | TIMED | [0.018, 0.172] | 1.0000 |
| CRASH600 | REGION | [-0.069, 0.073] | 1.0000 |
| CRASH600 | TIMED | [-0.092, 0.063] | 1.0000 |

LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false. هدف الربح يبقى غير متحقق حتى تستوفي سياسة فعلية شروط الدليل المحددة.

## الملفات

- [البروتوكول](PROTOCOL.md)، [التصريح](declaration.json)، [كل72خلية](results.json)، [التدقيق](independent_audit.json)، [الفحص المبكر](independent_precheck.json).
- ملفاتsignals/events/ledger/training_labelsوmodelsتحفظ الإشارات والمناطق والمجهول والصفقات والتدريب. معاملاتHYBRID44تظل تحت اسمTRANSFORMED44داخليًا في محركridgeالمجمد؛ mappingموثق صراحة.
- الخصائص الكاملة محلية تحتdata/hybrid_event_regions_20261008معSHA256؛ المصادر المحلية ضرورية لإعادة الحساب.
- [صيغة44المحوّلة المرفوضة](../representation_regions_20261008/REPORT.ar.md)، [المناطق السابقة](../zone_study_20261008/REPORT.ar.md)، [تعريفات الحركة الاثنا عشر](../zone_excursions_20261008/REPORT.ar.md).
