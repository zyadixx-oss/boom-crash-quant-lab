# تدريب المناطق على عائدها الفعلي المشروط بالتنفيذ

**اكتمل اختبار 60 حالة. اجتاز كل الشروط التاريخية 0/8 من حالات الاختبار للنموذجين الجديدين. QUALIFIED=false؛ التاريخ مستخدم سابقًا، ولم يثبت ربح قابل للتداول الحقيقي.**

تختبر هذه التجربة فرضية جديدة محددة: تدريب بوابة المناطق على عوائد مناطق CLOCK المكتملة نفسها، بدل هدف الدخول الزمني السابق. تتغير أيضًا عينة التدريب المشروطة بحدوث التنفيذ؛ لذلك لا تعزل التجربة الأثر السببي لتغيير قيم الهدف وحدها. عدم اللمس والإبطال ليسا صفقة بعائد صفر، والنتائج المجهولة تبقى مجهولة.

تستخدم RAW_REGION المؤشرات الأصلية الـ44، وتستخدم HYBRID_REGION المؤشرات المحوّلة الـ43 مع عمر شمعة M5 الكبيرة الأصلي. الفريمات المغلقة H4/H1/M15/M5/M1 والكاشف والخصائص محفوظة من التجربة السابقة ولم يُعد ضبطها. RAW_TIMED وHYBRID_TIMED نموذجان مرجعيان سابقان مجمدان؛ جميع الأذرع الخمسة تطبق مناطق السعر نفسها في هذا التقرير.

مصادر التجربة: 722 ملفًا، و62,380,130 سعرًا أصليًا مع إبقاء 670 ثانية ناقصة. تدريب أول 40/50/60% ثم التحقق في 10% تالية لكل إعداد؛ تدريب 70% لآخر 30% و180 يومًا لاحقة دون إعادة ملاءمة. الفترة الأقدم تبدأ 9 أكتوبر 2025 الساعة 11:08 UTC وتنتهي 7 أبريل 2026 الساعة 11:08 UTC، واللاحقة تنتهي 4 أكتوبر 2026 الساعة 11:08 UTC. هذه اختبارات زمنية ضمن تاريخ مكشوف، وليست fresh OOS أو تجربة استباقية.

## المجتمع الفعلي للتدريب

تتداخل فترات التدريب؛ لا تُجمع الأعداد بوصفها صفقات مستقلة. الصفقة المكتملة فقط تحمل netR في الانحدار. شرط التدريب 1000 صفقة مشتركة مكتملة؛ ridge بعقوبة 0.1 وعتبة الربع الأعلى q75 ودرجة موجبة، دون بحث عن إعداد أفضل.

| الرمز | الإعداد | كل الفرص | مكتملة | لم تنفذ/أبطلت | مسار مجهول | تخطيت لانشغال التعرض | الحالة |
|---|---|---:|---:|---:|---:|---:|---|
| BOOM600 | fit40 | 2418 | 1948 | 466 | 4 | 0 | FITTED |
| BOOM600 | fit50 | 3282 | 2640 | 638 | 4 | 0 | FITTED |
| BOOM600 | fit60 | 3740 | 3012 | 721 | 7 | 0 | FITTED |
| BOOM600 | fit70 | 4288 | 3450 | 829 | 9 | 0 | FITTED |
| CRASH600 | fit40 | 2418 | 1940 | 473 | 4 | 1 | FITTED |
| CRASH600 | fit50 | 3282 | 2660 | 617 | 4 | 1 | FITTED |
| CRASH600 | fit60 | 3740 | 3014 | 718 | 7 | 1 | FITTED |
| CRASH600 | fit70 | 4288 | 3447 | 831 | 9 | 1 | FITTED |

## إصدار المنطقة وحساب العائد

عند السعر الأصلي المغلق p وATR الأصلي A: شراء Boom داخل[p−0.55A,p−0.45A]، وبيع Crash داخل[p+0.45A,p+0.55A]. تثبت الحدود والإبطال وقت الإصدار؛ الإبطال 0.25A خارج الطرف المعاكس. يراقب فورًا، ويتفعل بعد دقيقة، وينتظر سعرًا مرصودًا داخل المنطقة خلال 15 دقيقة، ثم يدخل بالسعر الأصلي في الثانية التالية. لا يُنشأ لمس افتراضي عندما يقفز السعر فوق المنطقة.

وقف 2A من سعر الدخول، ومدة 15 دقيقة من اللمس، وأولوية للوقف عند نهاية المهلة ثم خروج بالثانية التالية. التعرض واحد لكل ذراع، والنتيجة المجهولة تحجز أقصى المدة المخططة. التكلفة المفترضة 0.10A لكل المسار والتطهير 31 دقيقة؛ R وحدة مخاطرة 2A، وليست ربحًا نقديًا. لم تُستخدم الأسعار أو ATR المحوّلان في الدخول أو الخروج أو التكلفة.


## نافذة التحقق الأولى بعد تدريب أول 40%

| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BOOM600 | CLOCK | 692 | 19 | 171 | 0 | 0 | 0.908 | [0.7612, 1.0464] | -0.0369 | 0.0131 |
| BOOM600 | RAW_TIMED | 153 | 19 | 31 | 0 | 0 | 0.973 | [0.6569, 1.4080] | -0.0102 | 0.0398 |
| BOOM600 | HYBRID_TIMED | 129 | 19 | 37 | 0 | 0 | 1.025 | [0.9146, 1.1401] | 0.0094 | 0.0594 |
| BOOM600 | RAW_REGION | 117 | 19 | 21 | 0 | 0 | 1.051 | [0.9173, 1.2245] | 0.0192 | 0.0692 |
| BOOM600 | HYBRID_REGION | 84 | 18 | 21 | 0 | 0 | 0.700 | [0.4155, 1.0772] | -0.1252 | -0.0752 |
| CRASH600 | CLOCK | 719 | 19 | 144 | 0 | 0 | 0.802 | [0.7208, 0.8880] | -0.0887 | -0.0387 |
| CRASH600 | RAW_TIMED | 236 | 19 | 33 | 0 | 0 | 0.697 | [0.6157, 0.7777] | -0.1502 | -0.1002 |
| CRASH600 | HYBRID_TIMED | 160 | 19 | 31 | 0 | 0 | 0.743 | [0.6569, 0.8391] | -0.1319 | -0.0819 |
| CRASH600 | RAW_REGION | 209 | 19 | 40 | 0 | 0 | 0.760 | [0.6801, 0.8529] | -0.1021 | -0.0521 |
| CRASH600 | HYBRID_REGION | 110 | 19 | 16 | 0 | 0 | 0.943 | [0.7334, 1.1787] | -0.0233 | 0.0267 |

## نافذة التحقق الثانية بعد تدريب أول 50%

| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BOOM600 | CLOCK | 372 | 13 | 82 | 0 | 3 | 1.072 | [0.9049, 1.3412] | 0.0272 | 0.0772 |
| BOOM600 | RAW_TIMED | 65 | 11 | 12 | 0 | 0 | 0.710 | [0.3317, 0.9875] | -0.1070 | -0.0570 |
| BOOM600 | HYBRID_TIMED | 31 | 11 | 7 | 0 | 0 | 1.010 | [0.7854, 1.6176] | 0.0037 | 0.0537 |
| BOOM600 | RAW_REGION | 60 | 11 | 8 | 0 | 0 | 0.301 | [0.1893, 0.4814] | -0.2937 | -0.2437 |
| BOOM600 | HYBRID_REGION | 48 | 11 | 8 | 0 | 0 | 1.438 | [0.9795, 2.4630] | 0.1405 | 0.1905 |
| CRASH600 | CLOCK | 353 | 13 | 101 | 0 | 3 | 0.809 | [0.7029, 0.9535] | -0.0811 | -0.0311 |
| CRASH600 | RAW_TIMED | 43 | 13 | 9 | 0 | 0 | 1.161 | [0.7353, 1.5768] | 0.0580 | 0.1080 |
| CRASH600 | HYBRID_TIMED | 54 | 11 | 12 | 0 | 1 | 0.830 | [0.6907, 0.9335] | -0.0767 | -0.0267 |
| CRASH600 | RAW_REGION | 40 | 11 | 15 | 0 | 0 | 1.284 | [0.6429, 2.7166] | 0.0907 | 0.1407 |
| CRASH600 | HYBRID_REGION | 33 | 9 | 6 | 0 | 1 | 0.582 | [0.2024, 0.9960] | -0.2006 | -0.1506 |

## نافذة التحقق الثالثة بعد تدريب أول 60%

| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BOOM600 | CLOCK | 437 | 15 | 108 | 0 | 2 | 0.891 | [0.7887, 0.9725] | -0.0472 | 0.0028 |
| BOOM600 | RAW_TIMED | 104 | 15 | 18 | 0 | 0 | 1.006 | [0.6380, 1.5093] | 0.0025 | 0.0525 |
| BOOM600 | HYBRID_TIMED | 62 | 12 | 20 | 0 | 0 | 0.866 | [0.3813, 1.5745] | -0.0633 | -0.0133 |
| BOOM600 | RAW_REGION | 62 | 14 | 15 | 0 | 1 | 0.936 | [0.4714, 1.6181] | -0.0266 | 0.0234 |
| BOOM600 | HYBRID_REGION | 70 | 12 | 20 | 0 | 0 | 0.875 | [0.4882, 1.5437] | -0.0482 | 0.0018 |
| CRASH600 | CLOCK | 432 | 15 | 113 | 0 | 2 | 0.792 | [0.6715, 0.9127] | -0.0906 | -0.0406 |
| CRASH600 | RAW_TIMED | 67 | 12 | 18 | 0 | 0 | 0.511 | [0.2520, 0.8097] | -0.2610 | -0.2110 |
| CRASH600 | HYBRID_TIMED | 72 | 13 | 13 | 0 | 0 | 0.625 | [0.4219, 0.9408] | -0.1992 | -0.1492 |
| CRASH600 | RAW_REGION | 64 | 14 | 24 | 0 | 0 | 0.846 | [0.4633, 1.0985] | -0.0580 | -0.0080 |
| CRASH600 | HYBRID_REGION | 36 | 10 | 8 | 0 | 0 | 0.834 | [0.5961, 1.0151] | -0.0787 | -0.0287 |

## اتحاد نوافذ التحقق الثلاث المستقلة في الإصدار

| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BOOM600 | CLOCK | 1501 | 45 | 361 | 0 | 5 | 0.941 | [0.8323, 1.0547] | -0.0240 | 0.0260 |
| BOOM600 | RAW_TIMED | 322 | 44 | 61 | 0 | 0 | 0.935 | [0.7097, 1.2063] | -0.0256 | 0.0244 |
| BOOM600 | HYBRID_TIMED | 222 | 40 | 64 | 0 | 0 | 0.970 | [0.7322, 1.2074] | -0.0117 | 0.0383 |
| BOOM600 | RAW_REGION | 239 | 43 | 44 | 0 | 1 | 0.821 | [0.5245, 1.0900] | -0.0712 | -0.0212 |
| BOOM600 | HYBRID_REGION | 202 | 39 | 49 | 0 | 0 | 0.908 | [0.6375, 1.3133] | -0.0354 | 0.0146 |
| CRASH600 | CLOCK | 1504 | 45 | 358 | 0 | 5 | 0.801 | [0.7385, 0.8692] | -0.0875 | -0.0375 |
| CRASH600 | RAW_TIMED | 346 | 42 | 60 | 0 | 0 | 0.700 | [0.5775, 0.8414] | -0.1458 | -0.0958 |
| CRASH600 | HYBRID_TIMED | 286 | 41 | 56 | 0 | 1 | 0.727 | [0.6168, 0.8430] | -0.1384 | -0.0884 |
| CRASH600 | RAW_REGION | 313 | 42 | 79 | 0 | 0 | 0.830 | [0.7175, 1.0372] | -0.0685 | -0.0185 |
| CRASH600 | HYBRID_REGION | 179 | 36 | 30 | 0 | 1 | 0.845 | [0.6499, 0.9887] | -0.0671 | -0.0171 |

## آخر 30% من الفترة الأقدم

| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BOOM600 | CLOCK | 1444 | 45 | 364 | 1 | 3 | 0.931 | [0.8325, 1.0482] | -0.0277 | 0.0223 |
| BOOM600 | RAW_TIMED | 297 | 43 | 65 | 0 | 2 | 1.011 | [0.7939, 1.3623] | 0.0047 | 0.0547 |
| BOOM600 | HYBRID_TIMED | 266 | 43 | 69 | 0 | 1 | 0.766 | [0.5372, 1.0220] | -0.0982 | -0.0482 |
| BOOM600 | RAW_REGION | 245 | 41 | 51 | 0 | 0 | 0.964 | [0.6558, 1.2703] | -0.0148 | 0.0352 |
| BOOM600 | HYBRID_REGION | 268 | 43 | 59 | 0 | 1 | 0.688 | [0.5065, 0.9418] | -0.1365 | -0.0865 |
| CRASH600 | CLOCK | 1448 | 45 | 361 | 0 | 3 | 0.887 | [0.8186, 0.9623] | -0.0459 | 0.0041 |
| CRASH600 | RAW_TIMED | 217 | 41 | 42 | 0 | 0 | 0.925 | [0.7230, 1.1268] | -0.0286 | 0.0214 |
| CRASH600 | HYBRID_TIMED | 174 | 35 | 59 | 0 | 0 | 1.014 | [0.7681, 1.3232] | 0.0055 | 0.0555 |
| CRASH600 | RAW_REGION | 171 | 42 | 44 | 0 | 0 | 0.896 | [0.6288, 1.3030] | -0.0380 | 0.0120 |
| CRASH600 | HYBRID_REGION | 111 | 33 | 23 | 0 | 0 | 1.272 | [0.8781, 2.0147] | 0.0890 | 0.1390 |

## فترة 180 يومًا اللاحقة

| الرمز | الذراع | مكتملة | أيام نشطة | بلا تنفيذ | تخطٍّ | مجهولة | PF صافٍ | CI95 أسبوعي لـPF | متوسط R صافٍ | متوسط R قبل التكلفة |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BOOM600 | CLOCK | 5394 | 157 | 1239 | 1 | 8 | 0.858 | [0.7952, 0.9212] | -0.0581 | -0.0081 |
| BOOM600 | RAW_TIMED | 1088 | 155 | 228 | 0 | 1 | 0.841 | [0.7038, 1.0062] | -0.0689 | -0.0189 |
| BOOM600 | HYBRID_TIMED | 984 | 144 | 237 | 0 | 1 | 0.865 | [0.7242, 1.0305] | -0.0547 | -0.0047 |
| BOOM600 | RAW_REGION | 1047 | 151 | 224 | 0 | 1 | 0.763 | [0.6507, 0.8936] | -0.1061 | -0.0561 |
| BOOM600 | HYBRID_REGION | 987 | 151 | 245 | 0 | 1 | 0.902 | [0.7689, 1.0572] | -0.0388 | 0.0112 |
| CRASH600 | CLOCK | 5353 | 157 | 1281 | 0 | 8 | 0.935 | [0.8744, 1.0011] | -0.0262 | 0.0238 |
| CRASH600 | RAW_TIMED | 760 | 147 | 205 | 0 | 3 | 0.869 | [0.7059, 1.0533] | -0.0564 | -0.0064 |
| CRASH600 | HYBRID_TIMED | 687 | 138 | 171 | 0 | 0 | 0.982 | [0.7988, 1.1780] | -0.0073 | 0.0427 |
| CRASH600 | RAW_REGION | 569 | 142 | 196 | 0 | 1 | 0.879 | [0.7391, 1.0475] | -0.0432 | 0.0068 |
| CRASH600 | HYBRID_REGION | 399 | 122 | 98 | 0 | 0 | 1.076 | [0.8546, 1.3302] | 0.0303 | 0.0803 |

الجدول يعرض جميع 60 خلية؛ اتحاد التحقق يعيد عرض الصفقات نفسها في النوافذ الثلاث ولا يمثل صفقات إضافية. لا تُجمع الأذرع أو الرموز أو الفترات للوصول إلى حد العينة.

## التفوق على المراجع المطلوبة في الاختبار الزمني

RAW_REGION يجب أن يتفوق على CLOCK وRAW_TIMED؛ وHYBRID_REGION على CLOCK وHYBRID_TIMED وRAW_REGION. فواصل الثقة من 9999 إعادة سحب بنفس البذرة 20261008، يوميًا ومع كتل 7 أيام. الأرقام التالية فروق متوسط R لكل صفقة مكتملة؛ لا تمثل احتمالات ربح معايرة.

| الرمز | الذراع | الفترة | المرجع | فرق المتوسط | CI95 يومي للفرق | CI95 أسبوعي للفرق |
|---|---|---|---|---:|---|---|
| BOOM600 | RAW_REGION | final_test | CLOCK | 0.0129 | [-0.0989, 0.1153] | [-0.1458, 0.1153] |
| BOOM600 | RAW_REGION | final_test | RAW_TIMED | -0.0195 | [-0.1316, 0.0843] | [-0.1550, 0.0719] |
| BOOM600 | RAW_REGION | later180 | CLOCK | -0.0480 | [-0.0994, 0.0037] | [-0.0982, 0.0046] |
| BOOM600 | RAW_REGION | later180 | RAW_TIMED | -0.0373 | [-0.0966, 0.0198] | [-0.0969, 0.0171] |
| BOOM600 | HYBRID_REGION | final_test | CLOCK | -0.1088 | [-0.1908, -0.0268] | [-0.2009, -0.0125] |
| BOOM600 | HYBRID_REGION | final_test | HYBRID_TIMED | -0.0383 | [-0.1275, 0.0501] | [-0.1017, 0.0231] |
| BOOM600 | HYBRID_REGION | final_test | RAW_REGION | -0.1218 | [-0.2610, 0.0305] | [-0.2868, 0.1147] |
| BOOM600 | HYBRID_REGION | later180 | CLOCK | 0.0194 | [-0.0333, 0.0716] | [-0.0298, 0.0687] |
| BOOM600 | HYBRID_REGION | later180 | HYBRID_TIMED | 0.0159 | [-0.0415, 0.0728] | [-0.0478, 0.0804] |
| BOOM600 | HYBRID_REGION | later180 | RAW_REGION | 0.0674 | [-0.0085, 0.1412] | [-0.0144, 0.1460] |
| CRASH600 | RAW_REGION | final_test | CLOCK | 0.0079 | [-0.0982, 0.1253] | [-0.1194, 0.1491] |
| CRASH600 | RAW_REGION | final_test | RAW_TIMED | -0.0093 | [-0.1438, 0.1221] | [-0.1235, 0.1290] |
| CRASH600 | RAW_REGION | later180 | CLOCK | -0.0170 | [-0.0831, 0.0537] | [-0.0665, 0.0357] |
| CRASH600 | RAW_REGION | later180 | RAW_TIMED | 0.0132 | [-0.0640, 0.0902] | [-0.0548, 0.0809] |
| CRASH600 | HYBRID_REGION | final_test | CLOCK | 0.1349 | [-0.0146, 0.3010] | [0.0030, 0.3115] |
| CRASH600 | HYBRID_REGION | final_test | HYBRID_TIMED | 0.0835 | [-0.1386, 0.3039] | [-0.1367, 0.3473] |
| CRASH600 | HYBRID_REGION | final_test | RAW_REGION | 0.1270 | [-0.0654, 0.3357] | [-0.0634, 0.3887] |
| CRASH600 | HYBRID_REGION | later180 | CLOCK | 0.0565 | [-0.0388, 0.1534] | [-0.0304, 0.1514] |
| CRASH600 | HYBRID_REGION | later180 | HYBRID_TIMED | 0.0376 | [-0.0625, 0.1383] | [-0.0593, 0.1361] |
| CRASH600 | HYBRID_REGION | later180 | RAW_REGION | 0.0735 | [-0.0455, 0.1937] | [-0.0339, 0.1858] |

## قرار الشروط التاريخية

الشروط تشمل PF≥1.5 و1000 نتيجة مكتملة و60 يومًا نشطًا لكل رمز/نموذج/فترة، ونجاح التطوير، وغياب نتائج المناطق المجهولة في النموذج ومراجعه، وفواصل ثقة موجبة للمتوسط والتفوق، وحد PF السفلي>1 يوميًا وأسبوعيًا، وتصحيح Holm لثماني مقارنات. تضاف استقرارية الأثلاث الزمنية، وخسارة إغلاق قصوى≤10%، ومتوسط موجب مع ضعف التكلفة. حتى اجتيازها جميعًا يبقى بحثًا تاريخيًا تكيفيًا.

| الرمز | الذراع | الفترة | أهلية التطوير | اجتاز كل الشروط | Holm p | أسباب عدم الاجتياز |
|---|---|---|---|---|---:|---|
| BOOM600 | RAW_REGION | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive, required_reference_RAW_TIMED_no_unknown_reference_outcomes, required_reference_RAW_TIMED_day_mean_CI_positive, required_reference_RAW_TIMED_day_advantage_CI_positive, required_reference_RAW_TIMED_week_mean_CI_positive, required_reference_RAW_TIMED_week_advantage_CI_positive |
| BOOM600 | RAW_REGION | later180 | False | False | 1.0000 | development_rejected_or_insufficient, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive, required_reference_RAW_TIMED_no_unknown_reference_outcomes, required_reference_RAW_TIMED_day_mean_CI_positive, required_reference_RAW_TIMED_day_advantage_CI_positive, required_reference_RAW_TIMED_week_mean_CI_positive, required_reference_RAW_TIMED_week_advantage_CI_positive |
| BOOM600 | HYBRID_REGION | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive, required_reference_HYBRID_TIMED_no_unknown_reference_outcomes, required_reference_HYBRID_TIMED_day_mean_CI_positive, required_reference_HYBRID_TIMED_day_advantage_CI_positive, required_reference_HYBRID_TIMED_week_mean_CI_positive, required_reference_HYBRID_TIMED_week_advantage_CI_positive, required_reference_RAW_REGION_day_mean_CI_positive, required_reference_RAW_REGION_day_advantage_CI_positive, required_reference_RAW_REGION_week_mean_CI_positive, required_reference_RAW_REGION_week_advantage_CI_positive |
| BOOM600 | HYBRID_REGION | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, incomplete_payoff, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, drawdown_or_ruin, doubled_cost_not_positive, required_reference_HYBRID_TIMED_no_unknown_reference_outcomes, required_reference_HYBRID_TIMED_day_mean_CI_positive, required_reference_HYBRID_TIMED_day_advantage_CI_positive, required_reference_HYBRID_TIMED_week_mean_CI_positive, required_reference_HYBRID_TIMED_week_advantage_CI_positive, required_reference_RAW_REGION_no_unknown_reference_outcomes, required_reference_RAW_REGION_day_mean_CI_positive, required_reference_RAW_REGION_day_advantage_CI_positive, required_reference_RAW_REGION_week_mean_CI_positive, required_reference_RAW_REGION_week_advantage_CI_positive |
| CRASH600 | RAW_REGION | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive, required_reference_RAW_TIMED_day_mean_CI_positive, required_reference_RAW_TIMED_day_advantage_CI_positive, required_reference_RAW_TIMED_week_mean_CI_positive, required_reference_RAW_TIMED_week_advantage_CI_positive |
| CRASH600 | RAW_REGION | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive, required_reference_RAW_TIMED_no_unknown_reference_outcomes, required_reference_RAW_TIMED_day_mean_CI_positive, required_reference_RAW_TIMED_day_advantage_CI_positive, required_reference_RAW_TIMED_week_mean_CI_positive, required_reference_RAW_TIMED_week_advantage_CI_positive |
| CRASH600 | HYBRID_REGION | final_test | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, fewer_than_60_active_days, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, required_reference_HYBRID_TIMED_day_mean_CI_positive, required_reference_HYBRID_TIMED_day_advantage_CI_positive, required_reference_HYBRID_TIMED_week_mean_CI_positive, required_reference_HYBRID_TIMED_week_advantage_CI_positive, required_reference_RAW_REGION_day_mean_CI_positive, required_reference_RAW_REGION_day_advantage_CI_positive, required_reference_RAW_REGION_week_mean_CI_positive, required_reference_RAW_REGION_week_advantage_CI_positive |
| CRASH600 | HYBRID_REGION | later180 | False | False | 1.0000 | development_rejected_or_insufficient, fewer_than_1000_completed, PF_below_1.5_or_unknown, unknown_region_outcomes, day_PF_CI_not_above_1, week_PF_CI_not_above_1, day_mean_CI_not_positive, week_mean_CI_not_positive, day_control_advantage_not_positive, week_control_advantage_not_positive, Holm_not_significant, chronological_thirds_not_stable, doubled_cost_not_positive, required_reference_HYBRID_TIMED_day_mean_CI_positive, required_reference_HYBRID_TIMED_day_advantage_CI_positive, required_reference_HYBRID_TIMED_week_mean_CI_positive, required_reference_HYBRID_TIMED_week_advantage_CI_positive, required_reference_RAW_REGION_no_unknown_reference_outcomes, required_reference_RAW_REGION_day_mean_CI_positive, required_reference_RAW_REGION_day_advantage_CI_positive, required_reference_RAW_REGION_week_mean_CI_positive, required_reference_RAW_REGION_week_advantage_CI_positive |

| الرمز | الذراع | الفترة | متوسط R بتكلفة 0.20ATR | وسيط الاحتفاظ بالدقائق | أقصى تراجع إغلاقات توضيحي |
|---|---|---|---:|---:|---:|
| BOOM600 | RAW_REGION | final_test | -0.0648 | 15.00 | 4.46% |
| BOOM600 | RAW_REGION | later180 | -0.1561 | 15.00 | 24.86% |
| BOOM600 | HYBRID_REGION | final_test | -0.1865 | 15.00 | 10.85% |
| BOOM600 | HYBRID_REGION | later180 | -0.0888 | 15.00 | 13.54% |
| CRASH600 | RAW_REGION | final_test | -0.0880 | 15.00 | 3.24% |
| CRASH600 | RAW_REGION | later180 | -0.0932 | 15.00 | 6.73% |
| CRASH600 | HYBRID_REGION | final_test | 0.0390 | 15.00 | 1.39% |
| CRASH600 | HYBRID_REGION | later180 | -0.0197 | 15.00 | 3.42% |

التراجع توضيحي على الصفقات المغلقة فقط وبالمخاطرة الكسرية المحفوظة؛ لا يقيس خسارة عائمة فعلية أو مالًا متداولًا. الأثلاث والنتائج غير المعرفة وعدد إعادات السحب غير المعرفة محفوظة كاملة فيresults.json.

## التدقيق المستقل وحدوده

حالة التدقيق PASS؛ عدد الفحوص المحفوظ 2300121. هذه أعداد عمل المدقق، وليست صفقات مستقلة إضافية. تُربط نسخة النتيجة والتصريح بحدودهما الدقيقة أدناه.

أعاد المدقق 16 نموذجًا، و27,456 تصرفًا تدريبيًا ضمن فترات متداخلة، و33,495 منطقة اختبار ودرجة إصدار. راجع كل 60 خلية و192 مقارنة يومية/أسبوعية، وطابق 30 خلية CLOCK ومرجع زمني مع التجربة السابقة. قرأ 62,380,130 سعرًا أصليًا وأبقى 670 ثانية مجهولة.

النطاق الذي يعلنه المدقق صراحة:

- بصمات الخصائص الـ44 والتوافر المشترك؛ مسارات مناطق التدريب الأصلية وتصنيفها؛ حل ridge مستقل وبصمات الأهداف والمصفوفات والنماذج؛ ثبات المراجع الزمنية السابقة؛ درجات الإصدار وعضوية الإشارات ومسارات الأسعار الأصلية؛ حساب R وPF وفواصل المتوسط والفروق يوميًا وأسبوعيًا، والمقارنات المطلوبة وتصحيح Holm لثماني حالات.

الحدود أو الأجزاء غير المتحقق منها التي يعلنها:

- لم يُعد اشتقاق صيغ الخصائص الأصلية الـ43 من الصفر.
- لم يُعد بناء شرط الصلاحية الأصلي لكل خاصية من الصفر؛ تحقق من الملفات والتوافر المحفوظين.
- لم يُعد تنفيذ منطق بوابات التأهيل مستقلًا؛ حالات البوابات المعروضة هي مخرجات المحرك المجمد.
- لم يُعد حساب التراجع والأثلاث الزمنية مستقلًا.

التحقق البرمجي: نجحت 122 حالة قبل القياس بعد إصلاح شروط المراجع الإضافية، و73 حالة خاصة بالمدقق. التشغيل الكامل النهائي نجح في 2627 حالة مع تحذير Starlette موجود سابقًا؛ حالات المدقق ضمن هذا العدد، ولا تُضاف إليه. هذه نتائج محلية، وحالة CI على النسخة المنشورة تُثبت منفصلًا.

تنفيذ CFD الفعلي، الفروق التاريخية المقاسة، تكاليف وسيط فعلية، الربح النقدي، والتجربة الورقية الاستباقية: **NOT TESTED**. نجاح التدقيق لا يبدل هذه الحدود ولا ينشئ أفضلية قابلة للتداول.

LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false.

## الملفات والبصمات

- [البروتوكول](PROTOCOL.md)، [التصريح](declaration.json)، [كل 60 خلية](results.json)، [التدقيق المستقل](independent_audit.json)، [سجل المراجعة والتحقق](review_and_validation_notes.md).
- ملفات training_labels/training_events/models تحفظ المجتمع المشروط وكل تصرفات التدريب؛ ملفات signals/events/ledger تحفظ الفرص والمناطق والدخول والصفقات. الملفات الأصلية والخصائص المخبأة تبقى محلية ببصماتSHA256.
- [مرجع الهدف الزمني السابق](../hybrid_event_regions_20261008/REPORT.ar.md)، [صيغة الـ44 المحوّلة المرفوضة](../representation_regions_20261008/REPORT.ar.md).
- declaration SHA256: `5a5cd147eefc7d12ac83c8ef839c931292c5fb338e07a1f083001c893c5196e5`
- results SHA256: `6e470b505d4f32ffb28600cd136ae886001f35e34631663553d8807b114d8115`
- audit SHA256: `a6194dd9178d2a14a7dee4d317429a1aaaad28d3d3ce4c195d9378f8e588f25b`
