# اختبار خروج زمني دون سقف ربح — الجولة البحثية الثانية

محاكاة فعلية لمسار الأسعار؛ تكلفة0.10ATR افتراضية، دخول بعد دقيقة، وقف عند أسوأ سعر الدقيقة.
الربح المالي والتنفيذ الحقيقي والاختبار الأمامي: **NOT TESTED**. الأعلام الأربعة false.
اختبار1000 نقل لقواعد اختيرت على500، ضمن فترة تاريخية أخرى معروفة على500؛ لا يثبت استقلال مولّد المؤشرات.
هذه جولة بحث تكيفية؛ تصحيح Holm داخل الجولة فقط، ولا يثبت احتمالية اكتشاف عالمية عبر جميع المحاولات.

بصمة الاختيار قبل التقييم: `7681dd680cef2f9372c11ea0df5cf23e7377d561d5c4f512e2a707d1a3c39dbe`.

## قواعد BOOM500

الإعدادات المؤهلة في التطوير: 0/384. الاختيار: `SPIKE::BB_SQUEEZE__sl4_h1`؛ أهلية: False.

| مؤشر الاختبار | الفترة | الدور | n | متوسط R | CI95 يومي | CI95 أسبوعي | PF | تراجع مغلق % |
|---|---|---|---:|---:|---|---|---:|---:|
| BOOM500 | reused_recent30 | primary | 229 | -0.013 | -0.124–0.099 | -0.151–0.118 | 0.967 | 4.32 |
| BOOM500 | reused_recent30 | selected | 1141 | -0.024 | -0.031–-0.017 | -0.031–-0.016 | 0.533 | 6.74 |
| BOOM500 | reused_older500 | primary | 728 | -0.072 | -0.141–-0.000 | -0.142–-0.001 | 0.831 | 13.08 |
| BOOM500 | reused_older500 | selected | 3822 | -0.025 | -0.029–-0.021 | -0.028–-0.021 | 0.523 | 21.34 |
| BOOM1000 | cross_symbol_replication | primary | 505 | -0.103 | -0.206–0.006 | -0.218–0.018 | 0.807 | 13.94 |
| BOOM1000 | cross_symbol_replication | selected | 3803 | -0.021 | -0.026–-0.016 | -0.026–-0.017 | 0.592 | 18.45 |

التحقق المتحرك؛ اختير كل إعداد من الماضي قبل فترة التحقق:

| الفترة | الاختيار | n | متوسط R |
|---|---|---:|---:|
| wf1 | DRIFT::ATR_BB_COMPRESSION__sl4_h30 | 181 | -0.000 |
| wf2 | DRIFT::ATR_BB_COMPRESSION__sl4_h30 | 176 | -0.030 |
| wf3 | SPIKE::SR_ALIGNMENT__sl4_h5 | 215 | -0.058 |

قرار نقل القواعد:

- primary: FAIL؛ Holm p=1.00000؛ الفرق عن التوقيت غير المشروط=-0.011R؛ أسوأ حذف يوم=-0.121R؛ أسوأ صفقة=-1.133R.
  أسباب الرفض: development_rejected, expectancy_under0.10R, PF_under1.30, CI_not_positive, holm_not_significant, drawdown_or_ruin, transfer_thirds_not_stable, single_day_fragility.
- selected: FAIL؛ Holm p=1.00000؛ الفرق عن التوقيت غير المشروط=0.003R؛ أسوأ حذف يوم=-0.022R؛ أسوأ صفقة=-0.079R.
  أسباب الرفض: development_rejected, expectancy_under0.10R, PF_under1.30, CI_not_positive, holm_not_significant, drawdown_or_ruin, transfer_thirds_not_stable, single_day_fragility.

## قواعد CRASH500

الإعدادات المؤهلة في التطوير: 0/384. الاختيار: `SPIKE::CRT_DISPLACEMENT__sl4_h5`؛ أهلية: False.

| مؤشر الاختبار | الفترة | الدور | n | متوسط R | CI95 يومي | CI95 أسبوعي | PF | تراجع مغلق % |
|---|---|---|---:|---:|---|---|---:|---:|
| CRASH500 | reused_recent30 | primary | 216 | -0.204 | -0.302–-0.095 | -0.304–-0.105 | 0.573 | 11.41 |
| CRASH500 | reused_recent30 | selected | 145 | -0.041 | -0.079–0.001 | -0.087–0.014 | 0.656 | 2.10 |
| CRASH500 | reused_older500 | primary | 736 | -0.031 | -0.098–0.037 | -0.104–0.043 | 0.926 | 8.20 |
| CRASH500 | reused_older500 | selected | 535 | -0.033 | -0.056–-0.009 | -0.055–-0.010 | 0.739 | 4.51 |
| CRASH1000 | cross_symbol_replication | primary | 484 | -0.102 | -0.212–0.009 | -0.200–0.003 | 0.812 | 13.38 |
| CRASH1000 | cross_symbol_replication | selected | 399 | -0.057 | -0.088–-0.023 | -0.089–-0.023 | 0.614 | 5.82 |

التحقق المتحرك؛ اختير كل إعداد من الماضي قبل فترة التحقق:

| الفترة | الاختيار | n | متوسط R |
|---|---|---:|---:|
| wf1 | SPIKE::ATR_COMPRESSION__sl4_h30 | 427 | -0.019 |
| wf2 | SPIKE::ATR_COMPRESSION__sl4_h30 | 433 | -0.067 |
| wf3 | DRIFT::CANDLE_STRUCTURE__sl4_h5 | 147 | -0.053 |

قرار نقل القواعد:

- primary: FAIL؛ Holm p=1.00000؛ الفرق عن التوقيت غير المشروط=-0.035R؛ أسوأ حذف يوم=-0.117R؛ أسوأ صفقة=-1.141R.
  أسباب الرفض: development_rejected, expectancy_under0.10R, PF_under1.30, CI_not_positive, holm_not_significant, drawdown_or_ruin, transfer_thirds_not_stable, single_day_fragility.
- selected: FAIL؛ Holm p=1.00000؛ الفرق عن التوقيت غير المشروط=-0.026R؛ أسوأ حذف يوم=-0.063R؛ أسوأ صفقة=-0.270R.
  أسباب الرفض: development_rejected, expectancy_under0.10R, PF_under1.30, CI_not_positive, holm_not_significant, transfer_thirds_not_stable, single_day_fragility.

## حدود

- الفترة الأقدم الجديدة لـ500 غير متاحة كاملة:6170 دقيقة فقط. لم تُقدّم كاختبار مستقل كافٍ.
- نتائج500 المعاد استخدامها استكشافية. نتائج1000 تحقق عبر مؤشرات مختلفة، وليست Forward أو ملء صفقات.
- متوسط R يتأثر بـATR وحجم المخاطرة؛ نقاط السعر الخام وتوزيع ATR والذيل محفوظة لتمييز هذا الأثر.
- وقف الدقيقة والخروج عند إغلاق المدة افتراضات؛ تراجع الصفقات المغلقة لا يشمل كل المخاطرة داخل الصفقة.
- تفاصيل الأثلاث، حذف أفضل5 صفقات/أيام، تركّز الربح وحساسية التكاليف والتنفيذ محفوظة في results.json.
