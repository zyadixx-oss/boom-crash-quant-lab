# تصدير أدلة الأسعار للبحث — 2026-10-13

هذه الأدوات تجمع أسعارًا وتفحص الملفات المحفوظة. تظل التكاليف التاريخية، وملاءمة مواصفات العقد للفترة، والتنفيذ الفعلي، والربح: **NOT TESTED / NOT VERIFIED**. لا تغيّر الأدوات نتائج الحملة أو أهلية أي استراتيجية.

نجاح ترجمة MQL5 مثبت في سجل البناء أدناه. تشغيل مُصدّر Python مع منصة Windows فعلية: **NOT TESTED**؛ اختبارات fake backend لا تثبت اتصالًا فعليًا. نتائج محاولات جمع الأسعار، ومنها `probe02`، تُوثّق في تقرير أدلة منفصل؛ هذا الدليل يصف طريقة التشغيل وعقد الملفات.

## الملفات

| الملف | الوظيفة |
|---|---|
| [export_mt5_quotes.py](/Users/zyadalasmri/Documents/Codex/2026-10-04/referenced-chatgpt-conversation-this-is-an/outputs/boom-crash-quant-lab/scripts/export_mt5_quotes.py) | مُصدّر Python على Windows؛ يتصل بعملية MT5 موجودة بالفعل. |
| [ExportResearchQuotes.mq5](/Users/zyadalasmri/Documents/Codex/2026-10-04/referenced-chatgpt-conversation-this-is-an/outputs/boom-crash-quant-lab/scripts/mt5/ExportResearchQuotes.mq5) | سكربت داخل MT5 لطلب نافذة محدودة وحفظ TSV في `FILE_COMMON`. |
| [verify_mt5_quote_export.py](/Users/zyadalasmri/Documents/Codex/2026-10-04/referenced-chatgpt-conversation-this-is-an/outputs/boom-crash-quant-lab/scripts/verify_mt5_quote_export.py) | تحقق مستقل دون اتصال من ملفات MQL5 المحفوظة. |
| [prospective_quote_archive.py](/Users/zyadalasmri/Documents/Codex/2026-10-04/referenced-chatgpt-conversation-this-is-an/outputs/boom-crash-quant-lab/scripts/prospective_quote_archive.py) | أرشيف محدود للتغذية العامة غير المصادق عليها، مع وضع تحقق دون اتصال. |

نفّذ أوامر Python الواردة هنا من جذر المستودع، باستخدام بيئة Python المناسبة. الأمثلة أدناه تعليمات تشغيل؛ وجودها لا يعني أنها نُفِّذت على Windows.

## حواجز الأمان المشتركة

تظل الرايات الثماني قيمًا منطقية `false`:

```json
{
  "LIVE_TRADING": false,
  "READY_FOR_LIVE": false,
  "LIVE_ALLOWED": false,
  "OPENED_TRADES": false,
  "live_trading": false,
  "ready_for_live": false,
  "live_allowed": false,
  "opened_trades": false
}
```

مُصدّر Python يرفض متغيرات البيئة المقابلة إن وُجدت بقيمة تختلف عن النص الحرفي `false`. في إيصال MQL5، تمثل الحقول الأربعة `safety_upper_*` النظائر المكتوبة بالأحرف الكبيرة، إلى جانب `live_trading` و`ready_for_live` و`live_allowed` و`opened_trades`؛ جميعها النص `false` لأن الصيغة TSV.

- الحساب يجب أن يكون موجودًا ومتصلًا مسبقًا بخادم Deriv. لا تُقبل بيانات دخول أو كلمة مرور، ولا تُصدَّر هوية صاحب الحساب أو رقم الدخول أو الرصيد.
- يجب تعطيل التداول الآلي في المنصة وفي السكربت. لا يغيّر المصدران هذا الإعداد ولا يرسل أي منهما أوامر.
- يُرفض الرمز المخصص. تُقرأ مواصفات الرمز الحالي المتاح دون تغيير الحساب أو إضافة رمز إلى Market Watch صراحةً.
- يُحفظ ترتيب الصفوف وتعددها؛ تكرار `time_msc` أو تطابق صفين ليس سببًا لحذفهما.
- استخدم مخرجًا أو بادئة جديدة لكل محاولة. الملفات الموجودة لا تُستبدل، وتُستخدم أقفال لمنع تشغيلين متزامنين على المخرج نفسه.

مقارنة السياق تستخدم الخادم والشركة ونمط الحساب فقط. انتقال الحساب بين حسابين لهما الخادم والنمط نفسيهما لا يُثبت اكتشافه دون هوية شخصية، وهي لا تُقرأ لهذا الغرض.

## مُصدّر Python على Windows

يتطلب Windows وحزمة `MetaTrader5` وتثبيتًا موجودًا للمنصة. افتح المنصة الموجودة أولًا، واترك حساب Deriv التجريبي متصلًا، وعطّل التداول الآلي. يُتحقق من تطابق مسار ملف المنصة مع عملية Windows قائمة قبل الاستدعاء:

```text
initialize(existing_terminal_path, timeout=...)
```

لا يمرر المُصدّر `login` أو `password` أو `server`، ولا يقبل خيارات لتغيير الحساب. قائمة استدعاءات المنصة المسموح بها هي: `initialize` و`account_info` و`terminal_info` و`symbol_info` و`copy_ticks_range` و`last_error` و`shutdown`. عند إتاحة `tradeapi_disabled` يجب أن تكون `true`؛ غياب الحقل يُسجَّل كحالة غير متاحة.

مثال PowerShell، بعد وجود المجلد الأب `C:\research`، وبمجلد إخراج جديد:

```powershell
python scripts/export_mt5_quotes.py `
  --terminal-path "C:\Program Files\MetaTrader 5\terminal64.exe" `
  --symbol BOOM500 "Boom 500 Index" `
  --symbol CRASH500 "Crash 500 Index" `
  --window 2025-11-01T00:00:00Z 2025-11-01T00:01:00Z `
  --output "C:\research\quotes-example-001"
```

الافتراضي حساب DEMO. قراءة أسعار حساب حقيقي موجود تتطلب الخيار الصريح `--allow-existing-live-market-data`، مع بقاء حواجز التداول ذاتها. لا يسمح هذا الخيار بفتح صفقة.

يقبل المصدر نافذة أو نافذتين غير متداخلتين، كل واحدة في الماضي وبحد أقصى 24 ساعة. التواريخ UTC صريحة، بدقة لا تتجاوز الملّي ثانية، والنهاية حصرية. يطلب الاستدعاء حتى `end - 1ms` ثم يفحص الحدود الفعلية لكل صف. دقة حدود خادم/Python واكتمال التاريخ يبقيان غير موثقين.

يحفظ المخرج CSV خامًا، وCSV مشتقًا للـ Bid/Ask عند اجتياز التحقق، و`manifest.json` وبصمته. يتضمن الإيصال الأعداد، والحدود، والأخطاء الرقمية، وحالة المحافظة على الصفوف. تُحفظ الحقول الرقمية المتوقعة من الصفوف المعيبة للتشخيص؛ الحقول غير المتوقعة لا تُنسخ، والقيم غير الرقمية تُحجب مع عداد واضح. الحد 200,000 صف لكل استجابة؛ أي حذف بسبب هذا الحد يُصرَّح به.

رمز الخروج `2` يعني رفضًا أو بيانات جزئية/فارغة/معيبة تستلزم قراءة الإيصال. رمز `0` يعني تسجيل أسعار اجتازت فحوص هذا المُصدّر، ولا يثبت اكتمال التغطية. `--max-call-seconds` حد ملاحظة لزمن الاستدعاء؛ لا يلغي استدعاءً أصليًا محجوبًا داخل المكتبة.

## ترجمة سكربت MQL5

المصدر المثبت لهذه المرحلة:

```text
source_sha256 = 98a3d5dd209853cadd9072f0eee822b20d8724801f312d8eea15e8535d435bc5
ex5_sha256    = 178485a368f30d1ccfa97cba2c8ae2588de688453ff4fa0c7656d3499272e113
compiler_log  = 0 errors, 0 warnings
```

مرجع البناء: [mql5_compile_verified.json](/Users/zyadalasmri/Documents/Codex/2026-10-04/referenced-chatgpt-conversation-this-is-an/outputs/boom-crash-quant-lab/docs/quote_evidence_followup_20261013/mql5_compile_verified.json). خرج مغلّف Wine برمز `1`، لكن سجل الترجمة الخاص بالمصدر وملف EX5 الناتج موجودان ومثبتان بالبصمات؛ نجاح البناء لا يُستنتج من رمز مغلّف Wine وحده.

خطوات إعادة البناء:

1. انسخ المصدر إلى `C:\ResearchQuoteCompile\ExportResearchQuotes.mq5` داخل تثبيت Windows/Wine الموجود.
2. استخدم MetaEditor الخاص بالتثبيت نفسه. مثال Windows:

   ```powershell
   & "C:\Program Files\MetaTrader 5\MetaEditor64.exe" `
     "/compile:C:\ResearchQuoteCompile\ExportResearchQuotes.mq5" /log
   ```

3. افحص سجل `ExportResearchQuotes.log` الناتج بجوار المصدر، ووجود `ExportResearchQuotes.ex5`. احفظ السجل وبصمات المصدر والناتج مع المحاولة. إعادة البناء قد تنتج بصمة EX5 أخرى بحسب إصدار المترجم؛ لا تنسب بصمة هذه المرحلة لبناء آخر.
4. ضع المصدر وEX5 الموافق له داخل `MQL5\Scripts\Research` في مجلد بيانات المنصة الفعلي. يمكن الوصول إليه من قائمة المنصة «فتح مجلد البيانات».

المصدر لا يحتوي `script_show_inputs` كي لا يعلّق تشغيل إعدادات StartUp على نافذة إدخال.

## تشغيل MQL5 بإعدادات صريحة

مدخلات السكربت: `InpSymbol` و`InpFromDate` و`InpEndDate` و`InpPrefix` و`InpConnectionWaitSeconds`. الافتراضي 60 ثانية، والحد الأعلى للنافذة 24 ساعة. انتظار الاتصال محدود بين صفر و30 ثانية؛ لا يحاول السكربت تسجيل الدخول أو تغيير الحساب.

ضع ملفًا جديدًا باسم `ResearchQuotes_Example_Boom500_20251101_60s.set` في `MQL5\Presets`. اكتب التواريخ **كأعداد صحيحة بالثواني منذ UTC epoch**:

```ini
InpSymbol=Boom 500 Index
InpFromDate=1761955200
InpEndDate=1761955260
InpPrefix=ResearchQuotes_Example_Boom500_20251101_60s
InpConnectionWaitSeconds=30
```

هذه النافذة هي `[2025-11-01T00:00:00Z, 2025-11-01T00:01:00Z)`. كتابة `2025.11.01 00:00:00` كنص داخل `.set` فُسِّرت في المحاولة الأولى كعدد `2025`، وأنتجت `FAILED_INPUT` دون طلب أسعار. ملف `.set` ليس موضعًا لاستخدام صياغة التاريخ النصية. احفظ ملفات الإعداد بصيغة تدعمها المنصة؛ استُخدمت UTF-16 في مسار التشغيل المحلي لهذه المرحلة.

مثال ملف `C:\ResearchQuoteCompile\quotes-example.ini`:

```ini
[Charts]
ProfileLast=ResearchQuotesReadOnly
MaxBars=1000
SaveDeleted=0
[Experts]
Enabled=0
AllowLiveTrading=0
AllowDllImport=0
Account=1
Profile=1
[StartUp]
Script=Research\ExportResearchQuotes
ScriptParameters=ResearchQuotes_Example_Boom500_20251101_60s.set
Symbol=Boom 500 Index
Period=M1
ShutdownTerminal=1
```

يجب أن يكون ملف تعريف البحث مخصصًا لهذا التشغيل، مع عدم تشغيل نسخة أخرى من المنصة على التثبيت نفسه. لا تضف حقول بيانات دخول إلى ملف الإعداد. يستخدم MQL5 الحساب الموجود في تثبيت المنصة؛ بخلاف مُصدّر Python الذي يشترط عملية قائمة، قد يبدأ مسار StartUp عملية المنصة الموجودة ثم يغلقها بعد السكربت.

تشغيل Windows باستخدام التثبيت الموجود:

```powershell
& "C:\Program Files\MetaTrader 5\terminal64.exe" `
  "/config:C:\ResearchQuoteCompile\quotes-example.ini"
```

إذا انتهت مهلة مراقبة المُشغّل وبقيت العملية حية، افحص العملية والملفات أولًا. انقضاء مهلة المراقب لا يثبت توقف `CopyTicksRange`، ولا يبرر بدء نسخة ثانية أو استبدال الملفات.

## عقد ملفات MQL5

يستخدم السكربت `CopyTicksRange(..., COPY_TICKS_ALL, from_msc, end_msc - 1)` مع ثلاثة استدعاءات كحد أقصى. قد يعيد الاستدعاء بيانات جزئية مع خطأ مزامنة/مهلة؛ يسجل الإيصال عدد الصفوف والخطأ والزمن لكل محاولة.

تُحفظ الملفات في مجلد MetaTrader المشترك `Terminal\Common\Files` باستخدام `FILE_COMMON`:

```text
<InpPrefix>_ticks.tsv
<InpPrefix>_receipt.tsv
```

رأس ملف الأسعار، بترتيبه الحرفي:

```text
time\ttime_msc\tbid\task\tlast\tvolume\tflags\tvolume_real
```

يمثل `\t` هنا فاصل TAB فعليًا في الملف. رأس الإيصال `field<TAB>value`، ومفاتيحه `lower_snake_case` بلا تكرار. تُحفظ صفوف المحاولة المختارة كما عادت، دون فرز أو حذف تكرار. نطاق الاحتفاظ صريح:

```text
raw_retention_scope = SELECTED_ATTEMPT_ONLY
all_attempt_raw_arrays_retained = false
```

لا تُحفظ المصفوفة الخام لكل محاولة. `selected_attempt` يحدد مصدر الملف الخام، بينما يحتفظ الإيصال بملخصات المحاولات الأخرى. بعد خطأ كتابة قد يبقى ملف `.part.tsv` للتشخيص؛ لا تعِد استخدام البادئة ذاتها. النشر النهائي يرفض استبدال ملف موجود، ويُنشر الإيصال أخيرًا.

`SAVED_NOT_PROVEN` يعني أن المحاولة المختارة عادت دون خطأ واجتازت فحوص الصفوف. تبقى `synchronization_status=NOT_PROVEN` و`historical_tick_completeness=NOT_PROVEN`. حالات `PARTIAL` و`EMPTY` و`FAILED_*` أدلة تشخيصية؛ لا تُعامل كتغطية مكتملة. الفصل أو السماح بالتداول أو الرمز المخصص يُسجَّل كفشل صريح.

## مواصفات العقد الحالية

يسجل الإيصال لقطة حالية للنقطة والأرقام العشرية وحجم العقد وحجم/قيمة التكة وحدود الحجم وخطوته وحقول swap وطريقة الحساب. تُجمع مواصفات MQL5 في الذاكرة بعد فحص السياق، ثم يُعاد فحص الخادم والشركة والنمط والاتصال وحواجز التداول قبل قبولها. إذا تغيّر السياق تُحذف اللقطة ويظهر `UNSTABLE_NOT_CAPTURED`؛ كتابة الإيصال تستخدم اللقطة المقبولة دون قراءات مواصفات جديدة.

هذه لقطة **CURRENT_ONLY_NOT_HISTORICAL**، ولا يجوز إسقاطها تلقائيًا على نوفمبر 2025. Bid/Ask يتيح وصف السبريد المرصود فقط؛ لا يثبت commission أو slippage أو swap التاريخي أو أسعار التعبئة أو الأرباح. تظل `costs_verified` و`execution_verified` و`historical_costs_verified` و`cfd_fills_verified` و`cash_contract_mapping_verified` جميعها `false`.

## التحقق دون اتصال

انسخ ملفَي TSV كزوج متجاور إلى مجلد أدلة جديد، واحفظ الأصل وبصماته. مثال، مع مجلد أب موجود ومخرج JSON جديد:

```powershell
python scripts/verify_mt5_quote_export.py `
  --receipt "C:\research\retained\ResearchQuotes_Example_Boom500_20251101_60s_receipt.tsv" `
  --output "C:\research\reports\ResearchQuotes_Example_Boom500_20251101_60s-verification.json"
```

المتحقق يستخدم المكتبة القياسية و`Decimal`، ولا يحتاج MT5 ولا يتصل بالخادم. يفحص الرؤوس، وتفرّد مفاتيح الإيصال، والرايات، والأعداد، وحدود UTC، والترتيب، واتساق `time` و`time_msc`، والاقتباسات المعكوسة، وأخطاء الإيصال، ويحسب وصف السبريد والفواصل وبصمات الملفات.

حالة `QUOTES_VALIDATED_COVERAGE_NOT_PROVEN` تعني صلاحية هذه الملفات وفق الفحوص. الفواصل المرصودة ليست عددًا مثبتًا للتكات المفقودة. رمز الخروج `0` مخصص لحالة `QUOTES_VALIDATED_COVERAGE_NOT_PROVEN`؛ الملف الفارغ أو الجزئي أو المعيب يخرج بالرمز `2`. حتى الحالة الصالحة لا تثبت اكتمال التغطية أو التكلفة.

## الأرشيف العام المستقبلي

جامع التغذية العامة منفصل عن MT5. يشترك دون مصادقة في `BOOM500` و`CRASH500`، ويحفظ الرسائل الخام قبل تحليلها، مع إعلانات جلسات وإيصالات وبصمات. الافتراضي 60 ثانية، والحد الأعلى 900 ثانية وثلاث محاولات إعادة اتصال. يتطلب بيئة POSIX للمسار الحالي الذي يستخدم `fcntl`، مع تبعيات المشروع `websockets` و`certifi`.

جمع جديد محدود؛ هذا الأمر يتصل بالتغذية العامة عند تنفيذه:

```bash
python3 scripts/prospective_quote_archive.py \
  --output work/public-quotes-example-001 \
  --duration 60 --receive-timeout 10 --max-reconnects 3
```

تحقق دون شبكة من الأرشيف نفسه:

```bash
python3 scripts/prospective_quote_archive.py \
  --output work/public-quotes-example-001 --verify
```

للاستئناف استخدم `--resume` مع الإعدادات نفسها. يفحص الاستئناف الجلسات المختومة أولًا ثم ينشئ جلسة محدودة جديدة؛ لا يصلح فجوات جلسة سابقة ولا يصطنع الجوانب التاريخية الناقصة. تُحفظ الجلسة غير المختومة للمراجعة. أسعار التغذية العامة لا تُستبدل بها أسعار MT5 التنفيذية، ولا تُدمج بمخرجاته لإثبات تكلفة أو تعبئة CFD.

## الاختبارات وحدود الإثبات

```bash
python3 -m pytest backend/tests/test_export_mt5_quotes.py -q
python3 -m pytest backend/tests/test_prospective_quote_archive.py -q
```

اختبارات fake backend تفحص السلوك والحواجز والملفات، بما في ذلك DEMO/LIVE، وحالات الفصل والمهلة، والصفوف المعيبة والمتكررة، وعدم تسريب الهوية، وعدم استبدال المخرجات. نجاح هذه الاختبارات لا يثبت سلوك مكتبة MT5 على Windows فعليًا.

| طبقة الإثبات | الحالة في هذه المرحلة من الدليل |
|---|---|
| اختبارات مُصدّر Python باستخدام fake backend | PASS وفق نتيجة تنفيذ الاختبارات المسجلة للمهمة. |
| اتصال مُصدّر Python بمنصة Windows فعلية | NOT TESTED. |
| ترجمة مصدر MQL5 المثبت أعلاه | مثبتة: 0 أخطاء و0 تحذيرات وEX5 محفوظ. |
| نتائج محاولات MQL5 وجمع التغذية العامة | تُقرأ من تقرير الأدلة المنفصل وإيصالات كل محاولة. |
| اكتمال التاريخ، التكاليف التاريخية، التعبئات والربح | NOT VERIFIED / NOT TESTED. |

## مراجع الواجهات الرسمية

آلية StartUp وScriptParameters وإغلاق المنصة موثقة في [MetaTrader 5](https://www.metatrader5.com/en/terminal/help/start_advanced/start). حدود CopyTicksRange بالملّي ثانية والنهايات الشاملة وحالات المزامنة موثقة في [MQL5](https://www.mql5.com/en/docs/series/copyticksrange). أوامر ترجمة MetaEditor موثقة في [دليل MetaEditor](https://www.metatrader5.com/en/metaeditor/help/beginning/integration_ide). النتائج الفعلية هنا مصدرها الإيصالات المحفوظة، ولا تُستنتج من أمثلة التوثيق.
