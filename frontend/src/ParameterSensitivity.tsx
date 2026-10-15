import { useEffect, useState } from 'react';

type Distribution = { min: number | null; median: number | null; max: number | null; null: number; positive: number; negative: number; zero: number };
type Axis = { axis: string; active: boolean; comparisons: number; net_delta_median: number | null; net_abs_delta_median: number | null; sign_reversals: number };
type Group = { attempts: number; scientific_points: number; aliases: number; missing: number; metrics: Record<string, Distribution>; axes: Axis[] };
type Dataset = { schema_version: number; scope: string; campaign_id: string; source_report_sha256: string; source_csv_sha256: string; axes: Record<string, number[]>; groups: Record<string, Group> };
const LABELS: Record<string, string> = { lookback: 'فترة النطاق', threshold: 'حد الفلتر', stop_atr: 'وقف الخسارة × ATR', target_atr: 'الهدف × ATR', max_hold_bars: 'أقصى احتفاظ · شموع' };
const METRICS: [string, string][] = [['profit_factor', 'PF'], ['net_profit', 'الصافي · وحدات افتراضية'], ['max_drawdown', 'التراجع'], ['completed_trades', 'الصفقات لكل إعداد']];
const format = (value: number | null, percent = false) => value === null || !Number.isFinite(value) ? '—' : (value * (percent ? 100 : 1)).toLocaleString('en-US', { maximumFractionDigits: 3 }) + (percent ? '%' : '');

export default function ParameterSensitivity({ campaignId, group }: { campaignId: string; group: string }) {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch('/research-audits/validation-sensitivity-20261012.json', { signal: controller.signal, cache: 'no-cache' });
        if (!response.ok) throw new Error('لم يُرفق تحليل حساسية محفوظ لهذه الحملة.');
        const value: Dataset = await response.json();
        if (value.schema_version !== 1 || value.scope !== 'VALIDATION_ONLY' || value.campaign_id !== campaignId || !value.groups || !value.axes) throw new Error('تحليل الحساسية لا يطابق الحملة المعروضة.');
        if (!controller.signal.aborted) {
          setError('');
          setDataset(value);
        }
      } catch (reason) {
        if (!controller.signal.aborted) {
          setDataset(null);
          setError(reason instanceof Error ? reason.message : 'تعذّر تحميل تحليل الحساسية.');
        }
      }
    })();
    return () => controller.abort();
  }, [campaignId]);
  const data = dataset?.campaign_id === campaignId ? dataset.groups[group] : undefined;
  const exportGroup = () => {
    if (!dataset || !data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify({ scope: dataset.scope, campaign_id: campaignId, group, axes: dataset.axes, source_report_sha256: dataset.source_report_sha256, source_csv_sha256: dataset.source_csv_sha256, data }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = group.replaceAll('/', '-') + '-validation-sensitivity.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <section className="rs-parameter-sensitivity" aria-label="حساسية الإعدادات في التحقق التطويري">
    <h3>حساسية الإعدادات · التحقق التطويري فقط</h3>
    <p>مقارنة وصفية للشبكة الأصلية ذات خمسة أبعاد، بتكاليف مفترضة. هذه أرقام تطوير؛ ليست اختبارًا نهائيًا جديدًا، ولا تغيّر التأهيل أو اختيار الإعدادات.</p>
    {error ? <p role="status">{error}</p> : !dataset ? <p role="status">جارٍ تحميل المصفوفة المحفوظة…</p> : !data ? <p>لا توجد مصفوفة محفوظة لهذه المجموعة؛ لا تُستعار أرقام من استراتيجية أخرى.</p> : <>
      <p><strong>{data.attempts} محاولة · {data.scientific_points} إعدادًا علميًا مختلفًا · {data.aliases} تكرارًا · {data.missing} نقص</strong></p>
      <p>الصافي الموجب: {data.metrics.net_profit.positive} · السالب: {data.metrics.net_profit.negative} · الصفر: {data.metrics.net_profit.zero}. هذه أعداد إعدادات، وليست صفقات مستقلة.</p>
      <div className="rs-scroll"><table className="rs-table"><thead><tr><th>المقياس عبر الإعدادات</th><th>الأدنى</th><th>الوسيط</th><th>الأعلى</th><th>غير متاح</th></tr></thead><tbody>{METRICS.map(([key, label]) => <tr key={key}><td>{label}</td>{(['min', 'median', 'max'] as const).map(stat => <td dir="ltr" key={stat}>{format(data.metrics[key][stat], key === 'max_drawdown')}</td>)}<td>{data.metrics[key].null}</td></tr>)}</tbody></table></div>
      <div className="rs-scroll"><table className="rs-table"><thead><tr><th>المحور</th><th>القيم المختبرة</th><th>مقارنات فعلية</th><th>وسيط فرق الصافي</th><th>وسيط الفرق المطلق</th><th>انقلاب موجب/سالب</th></tr></thead><tbody>{data.axes.map(axis => <tr key={axis.axis}><td>{LABELS[axis.axis] || axis.axis}{!axis.active && <small>خامل · لا يدخل هذه العائلة</small>}</td><td dir="ltr">{dataset.axes[axis.axis].join(' / ')}</td><td>{axis.comparisons}</td><td dir="ltr">{format(axis.net_delta_median)}</td><td dir="ltr">{format(axis.net_abs_delta_median)}</td><td>{axis.sign_reversals}</td></tr>)}</tbody></table></div>
      <p>الفرق = قيمة الصافي عند الإعداد الأعلى ناقص الأدنى، مع تثبيت بقية المحاور الفعالة. قد يعني الموجب تقليل خسارة فقط؛ لا يُستخدم لإعادة ترتيب السياسات. ثُبّت ATR عند 14، ووقف الخسارة المتحرك معطّل.</p>
      <button className="rs-text-button" onClick={exportGroup}>تصدير حساسية هذه المجموعة JSON</button>
      <details><summary>مصدر المصفوفة وبصماتها</summary><p>من سجل التحقق التطويري الأصلي، بعد إزالة التكرارات العلمية؛ PF غير المتاح محفوظ دون تحويله إلى صفر.</p><code dir="ltr">التقرير: {dataset.source_report_sha256}<br />CSV: {dataset.source_csv_sha256}</code></details>
    </>}
  </section>;
}
