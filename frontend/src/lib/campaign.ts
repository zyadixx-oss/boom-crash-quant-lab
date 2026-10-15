export const CAMPAIGN_SYMBOLS = ['BOOM300', 'BOOM500', 'BOOM600', 'BOOM900', 'BOOM1000', 'CRASH300', 'CRASH500', 'CRASH600', 'CRASH900', 'CRASH1000'] as const;
export const CAMPAIGN_FRAMES = ['M1', 'M2', 'M3', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H4'] as const;
export const RESEARCH_SAFETY = Object.freeze({ LIVE_TRADING: false, READY_FOR_LIVE: false, LIVE_ALLOWED: false, OPENED_TRADES: false });
export type ResultStatus = 'QUALIFIED' | 'REJECTED' | 'NOT_TESTED' | 'INSUFFICIENT_DATA';
export type SeriesPoint = { time: string; value: number };
export type Trade = { id: string; entry_time: string; exit_time: string; side: string; entry_price: number | null; exit_price: number | null; net_profit: number | null; status: string };
export type MonthlyResult = { month: string; trades: number | null; net_profit: number | null; profit_factor: number | null };
export type ReferenceResult = { name: string; net_profit: number | null; profit_factor: number | null; comparison: string; status: string; reason: string };
export type StabilityResult = { label: string; value: number | null; status: string; reason: string };
export type CampaignResult = {
  id: string; symbol: string; timeframe: string; name: string; status: ResultStatus; reasons: string[];
  settings: Record<string, unknown>; tested_at: string; source: string; period_start: string; period_end: string; rank_score: number | null; details_url: string;
  oos_available: boolean;
  development: { training: CampaignResult['oos']; validation: CampaignResult['oos']; validation_score: number | null; walk_forward: StabilityResult[] };
  oos: { win_rate: number | null; profit_factor: number | null; completed_trades: number | null; net_profit: number | null;
    max_drawdown: number | null; mean_net_R: number | null; net_profit_unit: string; status: string;
    equity: SeriesPoint[]; drawdown: SeriesPoint[]; trades: Trade[]; monthly: MonthlyResult[]; references: ReferenceResult[]; stability: StabilityResult[]; };
};
export type CampaignCycle = { number: number | null; status: string; started_at: string; completed_at: string; tested_count: number | null; notes: string[] };
export type CampaignSnapshot = {
  schema_version: number; generated_at: string;
  campaign: { id: string; status: string; started_at: string; next_cycle_at: string; completed_cycles: number; declared_completed_cycles: number | null; current_cycle: number | null; target_cycles: number; interval_minutes: number };
  source: { provider: string; period_start: string; period_end: string; updated_at: string; cost_provenance: string; contract_status: string; data_notes: string[] };
  symbols: { symbol: string; label: string; status: string; reason: string; best_strategy_id: string }[];
  strategies: CampaignResult[]; cycles: CampaignCycle[]; reproduction: { commands: string[]; report_url: string; search_csv_url: string };
};
const record = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
const rows = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
const string = (value: unknown): string => typeof value === 'string' ? value : '';
const strings = (value: unknown): string[] => rows(value).filter((item): item is string => typeof item === 'string');
export const finite = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const count = (value: unknown): number | null => { const n = finite(value); return n !== null && n >= 0 && Number.isInteger(n) ? n : null; };
const fraction = (value: unknown): number | null => { const n = finite(value); return n !== null && n >= 0 && n <= 1 ? n : null; };
const nonnegative = (value: unknown): number | null => { const n = finite(value); return n !== null && n >= 0 ? n : null; };
const ranked = (value: unknown): number | null => { const n = finite(value); return n === -1e12 ? null : n; };
const series = (value: unknown): SeriesPoint[] => rows(value).flatMap(item => { const row = record(item), n = finite(row.value), time = string(row.time); return n !== null && time ? [{ time, value: n }] : []; });
const status = (value: unknown): ResultStatus => ['QUALIFIED', 'REJECTED', 'INSUFFICIENT_DATA'].includes(string(value).toUpperCase()) ? string(value).toUpperCase() as ResultStatus : 'NOT_TESTED';
export const displayResearchUnit = (unit: string): string => unit === 'assumed_account_units; CFD cash mapping NOT VERIFIED' ? 'وحدات حساب افتراضي؛ مطابقة عقد CFD غير موثقة' : unit;
export const displayCycleNote = (note: string): string => ({
  'Training/validation search only; final holdout remains sealed': 'بحث التدريب والتحقق فقط؛ الاختبار النهائي محجوب.',
  'Point-price execution and assumed costs do not verify CFD profits': 'تنفيذ بأسعار مفردة وتكاليف مفترضة؛ ربحية عقود CFD غير متحققة.',
} as Record<string, string>)[note] || note;
export const formatResearchDate = (value?: string, short = false): string => value && Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleString('ar-SA', { calendar: 'gregory', year: 'numeric', day: '2-digit', month: 'short', ...(short ? {} : { hour: '2-digit', minute: '2-digit' }), timeZone: 'Asia/Riyadh' }) : 'غير متاح';

function metrics(input: unknown): CampaignResult['oos'] {
  const wrapper = record(input), oos = { ...record(wrapper.metrics), ...wrapper };
  return { win_rate: fraction(oos.win_rate), profit_factor: nonnegative(oos.profit_factor), completed_trades: count(oos.completed_trades),
    net_profit: finite(oos.net_profit), max_drawdown: fraction(oos.max_drawdown), mean_net_R: finite(oos.mean_net_R), net_profit_unit: displayResearchUnit(string(oos.net_profit_unit) || string(oos.units)), status: string(oos.status),
    equity: series(oos.equity), drawdown: series(oos.drawdown),
    trades: rows(oos.trades).map((item, index) => { const trade = record(item); return { id: string(trade.id) || String(index + 1), entry_time: string(trade.entry_time), exit_time: string(trade.exit_time), side: string(trade.side), entry_price: finite(trade.entry_price), exit_price: finite(trade.exit_price), net_profit: finite(trade.net_profit), status: string(trade.status) }; }),
    monthly: rows(oos.monthly).map(item => { const month = record(item); return { month: string(month.month), trades: count(month.trades), net_profit: finite(month.net_profit), profit_factor: nonnegative(month.profit_factor) }; }),
    references: rows(oos.references).map(item => { const ref = record(item); return { name: string(ref.name), net_profit: finite(ref.net_profit), profit_factor: nonnegative(ref.profit_factor), comparison: string(ref.comparison), status: string(ref.status), reason: string(ref.reason) }; }),
    stability: rows(oos.stability).map(item => { const cell = record(item); return { label: string(cell.label), value: finite(cell.value), status: string(cell.status), reason: string(cell.reason) }; }),
  };
}
export type ResultScope = 'validation' | 'oos';
export const resultMetrics = (row: CampaignResult, scope: ResultScope) => scope === 'validation' ? row.development.validation : row.oos;
export const resultScore = (row: CampaignResult, scope: ResultScope): number | null => ranked(scope === 'validation' ? row.development.validation_score : row.rank_score);
export type ResultSortKey = 'risk_score' | 'profit_factor' | 'win_rate' | 'completed_trades' | 'net_profit' | 'max_drawdown';
export function compareCampaignResults(a: CampaignResult, b: CampaignResult, scope: ResultScope, key: ResultSortKey = 'risk_score', ascending = false): number {
  const av = key === 'risk_score' ? resultScore(a, scope) : resultMetrics(a, scope)[key];
  const bv = key === 'risk_score' ? resultScore(b, scope) : resultMetrics(b, scope)[key];
  if (av === null) return bv === null ? 0 : 1;
  if (bv === null) return -1;
  return (av - bv) * (ascending ? 1 : -1);
}
export function completedCycleTestCount(cycles: CampaignCycle[]): number | null {
  const receipts = new Map(cycles.filter(row => row.status === 'COMPLETED' && row.number !== null && Number.isFinite(Date.parse(row.completed_at))).map(row => [row.number, row]));
  if ([...receipts.values()].some(row => row.tested_count === null)) return null;
  return [...receipts.values()].reduce((sum, row) => sum + (row.tested_count ?? 0), 0);
}

export function parseCampaignSnapshot(input: unknown): CampaignSnapshot {
  const value = record(input);
  if (value.schema_version !== 1) throw new Error('صيغة نتائج البحث غير مدعومة. يلزم تحديث ملف النتائج.');
  const safety = record(value.safety);
  for (const key of ['LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES', 'ready_for_live', 'live_allowed', 'opened_trades']) {
    if (key in safety && safety[key] !== false) throw new Error('تعذّر عرض النتائج: ملف البحث لا يؤكد تعطيل التداول الحقيقي.');
  }
  const campaign = record(value.campaign), source = record(value.source), reproduction = record(value.reproduction);
  const cycles: CampaignCycle[] = rows(value.cycles).map(item => { const row = record(item); return {
    number: count(row.number), status: string(row.status).toUpperCase(), started_at: string(row.started_at), completed_at: string(row.completed_at), tested_count: count(row.tested_count), notes: strings(row.notes),
  }; });
  // Completion is backed by dated, unique cycle receipts, never by a projected counter.
  const completed = new Set(cycles.filter(row => row.status === 'COMPLETED' && row.number !== null && row.completed_at && Number.isFinite(Date.parse(row.completed_at))).map(row => row.number)).size;
  const strategies = rows(value.strategies).flatMap((item, index): CampaignResult[] => {
    const row = record(item), development = record(row.development), symbol = string(row.symbol), timeframe = string(row.timeframe);
    if (!(CAMPAIGN_SYMBOLS as readonly string[]).includes(symbol) || !(CAMPAIGN_FRAMES as readonly string[]).includes(timeframe)) return [];
    const finalAvailable = row.oos !== null && typeof row.oos === 'object' && !Array.isArray(row.oos);
    return [{ id: string(row.id) || `${symbol}-${timeframe}-${index}`, symbol, timeframe, name: string(row.name) || 'استراتيجية بلا اسم',
      status: !finalAvailable && status(row.status) === 'QUALIFIED' ? 'NOT_TESTED' : status(row.status), reasons: strings(row.reasons), settings: record(row.settings), tested_at: string(row.tested_at), source: string(row.source),
      period_start: string(row.period_start), period_end: string(row.period_end), rank_score: ranked(row.rank_score), details_url: string(row.details_url),
      oos_available: finalAvailable, oos: metrics(row.oos),
      development: { training: metrics(development.training), validation: metrics(development.validation), validation_score: ranked(development.validation_score), walk_forward: rows(development.walk_forward).map(item => { const cell = record(item); return { label: string(cell.label), value: finite(cell.value ?? cell.profit_factor), status: string(cell.status), reason: string(cell.reason) }; }) },
    }];
  });
  return { schema_version: 1, generated_at: string(value.generated_at), campaign: { id: string(campaign.id), status: string(campaign.status), started_at: string(campaign.started_at), next_cycle_at: string(campaign.next_cycle_at), completed_cycles: completed, declared_completed_cycles: count(campaign.completed_cycles), current_cycle: count(campaign.current_cycle), target_cycles: count(campaign.target_cycles) ?? 32, interval_minutes: count(campaign.interval_minutes) ?? 15 },
    source: { provider: string(source.provider), period_start: string(source.period_start), period_end: string(source.period_end), updated_at: string(source.updated_at), cost_provenance: string(source.cost_provenance), contract_status: string(source.contract_status), data_notes: strings(source.data_notes) },
    symbols: rows(value.symbols).map(item => { const symbol = record(item); return { symbol: string(symbol.symbol), label: string(symbol.label), status: string(symbol.status), reason: string(symbol.reason), best_strategy_id: string(symbol.best_strategy_id) }; }),
    strategies, cycles, reproduction: { commands: strings(reproduction.commands), report_url: string(reproduction.report_url), search_csv_url: string(reproduction.search_csv_url) },
  };
}

export function mergeCampaignDetails(row: CampaignResult, input: unknown): CampaignResult {
  const data = record(input), development = record(data.development);
  const combine = (base: CampaignResult['oos'], extra: unknown) => { const wrapper = record(extra); return { ...base, ...record(wrapper.metrics), ...wrapper }; };
  const merged = { ...row, oos: row.oos_available || data.oos != null ? combine(row.oos, data.oos) : null, development: { ...row.development, ...development, training: combine(row.development.training, development.training), validation: combine(row.development.validation, development.validation) } };
  return parseCampaignSnapshot({ schema_version: 1, safety: RESEARCH_SAFETY, strategies: [merged] }).strategies[0];
}

export function bestQualified(snapshot: CampaignSnapshot, symbol: string): CampaignResult | undefined {
  const candidates = snapshot.strategies.filter(row => row.symbol === symbol && row.status === 'QUALIFIED');
  const declared = snapshot.symbols.find(row => row.symbol === symbol)?.best_strategy_id;
  return candidates.find(row => row.id === declared) ?? candidates.sort((a, b) => (b.rank_score ?? -Infinity) - (a.rank_score ?? -Infinity))[0];
}

export function heatmapResult(snapshot: CampaignSnapshot, symbol: string, timeframe: string, scope: ResultScope = 'oos'): CampaignResult | undefined {
  const candidates = snapshot.strategies.filter(row => row.symbol === symbol && row.timeframe === timeframe);
  return candidates.sort((a, b) => Number(scope === 'oos' && b.status === 'QUALIFIED') - Number(scope === 'oos' && a.status === 'QUALIFIED') || compareCampaignResults(a, b, scope))[0];
}

// Spreadsheet exports preserve Arabic and quote untrusted text as text, not formulas.
const csvCell = (value: unknown) => { const raw = value == null ? '' : String(value); const safe = /^[\s]*[=+\-@]/.test(raw) && typeof value !== 'number' ? `'${raw}` : raw; return `"${safe.replaceAll('"', '""')}"`; };
export function resultCSV(results: CampaignResult[], scope: ResultScope = 'oos'): string {
  const fields = ['scope', 'symbol', 'timeframe', 'strategy', 'status', 'risk_score', 'win_rate', 'profit_factor', 'completed_trades', 'net_profit', 'net_profit_unit', 'max_drawdown', 'measurement_status', 'reasons', 'settings', 'tested_at'];
  const body = results.map(row => { const data = resultMetrics(row, scope); return [scope, row.symbol, row.timeframe, row.name, row.status, resultScore(row, scope), data.win_rate, data.profit_factor, data.completed_trades, data.net_profit, data.net_profit_unit, data.max_drawdown, data.status, row.reasons.join(' | '), JSON.stringify(row.settings), row.tested_at].map(csvCell).join(','); });
  return '\uFEFF' + [fields.join(','), ...body].join('\r\n');
}

export function tradeCSV(trades: Trade[]): string {
  const fields: (keyof Trade)[] = ['id', 'entry_time', 'exit_time', 'side', 'entry_price', 'exit_price', 'net_profit', 'status'];
  return '\uFEFF' + [fields.join(','), ...trades.map(row => fields.map(key => csvCell(row[key])).join(','))].join('\r\n');
}
