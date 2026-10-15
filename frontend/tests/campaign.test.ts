import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bestQualified, CAMPAIGN_FRAMES, CAMPAIGN_SYMBOLS, compareCampaignResults, completedCycleTestCount, displayCycleNote, formatResearchDate, heatmapResult, mergeCampaignDetails, parseCampaignSnapshot, RESEARCH_SAFETY, resultCSV, resultMetrics } from '../src/lib/campaign.ts';

const snapshot = (strategies: unknown[] = [], extra: Record<string, unknown> = {}) => parseCampaignSnapshot({ schema_version: 1, safety: RESEARCH_SAFETY, strategies, ...extra });
const row = (extra: Record<string, unknown> = {}) => ({ id: 'boom-m1-crt', symbol: 'BOOM500', timeframe: 'M1', name: 'CRT', status: 'NOT_TESTED', oos: null, ...extra });

test('coverage is ten requested symbols by ten independent frames', () => {
  assert.equal(CAMPAIGN_SYMBOLS.length, 10); assert.equal(CAMPAIGN_FRAMES.length, 10);
  assert.ok(CAMPAIGN_SYMBOLS.includes('CRASH1000')); assert.ok(CAMPAIGN_FRAMES.includes('H4'));
  const data = snapshot([row({ symbol: 'BTCUSD' }), row({ timeframe: 'H1/M5' })]);
  assert.equal(data.strategies.length, 0);
});
test('missing and nonfinite metrics never become zero or invented profit', () => {
  const data = snapshot([row({ oos: { win_rate: 40, profit_factor: '1.5', completed_trades: -2, net_profit: Infinity, max_drawdown: 22 } })]);
  const metrics = data.strategies[0].oos;
  assert.equal(metrics.win_rate, null); assert.equal(metrics.profit_factor, null); assert.equal(metrics.completed_trades, null);
  assert.equal(metrics.net_profit, null); assert.equal(metrics.max_drawdown, null); assert.deepEqual(metrics.equity, []);
});
test('campaign summary preserves explicit backend units before lazy details arrive', () => {
  const data = snapshot([row({ development: { validation: { units: 'assumed_account_units; CFD cash mapping NOT VERIFIED', net_profit: -10 } } }), row({ id: 'explicit', oos: { units: 'legacy unit', net_profit_unit: 'وحدات حساب افتراضي' } }), row({ id: 'absent' }), row({ id: 'unknown', oos: { units: 'unknown measured unit' } })]);
  assert.equal(data.strategies[0].development.validation.net_profit_unit, 'وحدات حساب افتراضي؛ مطابقة عقد CFD غير موثقة');
  assert.equal(data.strategies[1].oos.net_profit_unit, 'وحدات حساب افتراضي');
  assert.equal(data.strategies[2].oos.net_profit_unit, '');
  assert.equal(data.strategies[3].oos.net_profit_unit, 'unknown measured unit');
});
test('visible cycle notes localize known notices and retain unknown evidence', () => {
  assert.equal(displayCycleNote('Training/validation search only; final holdout remains sealed'), 'بحث التدريب والتحقق فقط؛ الاختبار النهائي محجوب.');
  assert.equal(displayCycleNote('Point-price execution and assumed costs do not verify CFD profits'), 'تنفيذ بأسعار مفردة وتكاليف مفترضة؛ ربحية عقود CFD غير متحققة.');
  assert.equal(displayCycleNote('new evidence'), 'new evidence');
});
test('short source periods and chart labels retain Gregorian years across a full-year study', () => {
  assert.match(formatResearchDate('2025-10-11T12:22:00Z', true), /2025|٢٠٢٥/);
  assert.match(formatResearchDate('2026-10-11T12:22:00Z', true), /2026|٢٠٢٦/);
  assert.match(formatResearchDate('2025-12-31T22:00:00Z', true), /2026|٢٠٢٦/);
  assert.equal(formatResearchDate('invalid', true), 'غير متاح');
});
test('default table ranking uses recorded risk score, hides sentinel, and preserves manual PF sort', () => {
  const data = snapshot([
    row({ id: 'tiny-high-pf', rank_score: -1e12, development: { validation_score: -1e12, validation: { profit_factor: 1044, completed_trades: 3 } } }),
    row({ id: 'stable', rank_score: 1, development: { validation_score: 5, validation: { profit_factor: 1.2, completed_trades: 1000 } } }),
    row({ id: 'lower-score', rank_score: 2, development: { validation_score: -1, validation: { profit_factor: .9, completed_trades: 500 } } }),
  ]);
  assert.equal(data.strategies[0].development.validation_score, null);
  assert.equal(data.strategies[0].rank_score, null);
  assert.deepEqual(data.strategies.slice().sort((a, b) => compareCampaignResults(a, b, 'validation')).map(row => row.id), ['stable', 'lower-score', 'tiny-high-pf']);
  assert.equal(data.strategies.slice().sort((a, b) => compareCampaignResults(a, b, 'validation', 'profit_factor'))[0].id, 'tiny-high-pf');
  assert.equal(data.strategies.slice().sort((a, b) => compareCampaignResults(a, b, 'oos'))[0].id, 'lower-score');
});
test('completed-cycle test totals use dated unique receipts and distinguish missing counts', () => {
  const data = snapshot([], { cycles: [
    { number: 1, status: 'COMPLETED', completed_at: '2026-10-11T10:00:00Z', tested_count: 1300 },
    { number: 1, status: 'COMPLETED', completed_at: '2026-10-11T10:00:00Z', tested_count: 1300 },
    { number: 2, status: 'COMPLETED', completed_at: '2026-10-11T10:15:00Z', tested_count: 1100 },
    { number: 3, status: 'RUNNING', tested_count: 900 },
  ] });
  assert.equal(completedCycleTestCount(data.cycles), 2400);
  assert.equal(completedCycleTestCount([]), 0);
  assert.equal(completedCycleTestCount(snapshot([], { cycles: [{ number: 1, status: 'COMPLETED', completed_at: '2026-10-11T10:00:00Z' }] }).cycles), null);
});
test('final sealed state preserves validation separately and cannot qualify', () => {
  const data = snapshot([row({ status: 'QUALIFIED', development: { validation: { metrics: { profit_factor: 1.8, completed_trades: 200, net_profit: 4 } } } })]);
  const result = data.strategies[0];
  assert.equal(result.status, 'NOT_TESTED'); assert.equal(result.oos_available, false);
  assert.equal(result.oos.profit_factor, null); assert.equal(resultMetrics(result, 'validation').profit_factor, 1.8);
  assert.equal(bestQualified(data, 'BOOM500'), undefined);
});
test('best selection considers qualified results only and honors the declared ID', () => {
  const data = snapshot([row({ id: 'failed', status: 'REJECTED', oos: { profit_factor: 9 } }), row({ id: 'qualified', status: 'QUALIFIED', rank_score: 1, oos: { profit_factor: 1.2 } }), row({ id: 'other', status: 'QUALIFIED', rank_score: 2, oos: { profit_factor: 1.3 } })], { symbols: [{ symbol: 'BOOM500', best_strategy_id: 'qualified' }] });
  assert.equal(bestQualified(data, 'BOOM500')?.id, 'qualified');
  assert.equal(bestQualified(data, 'CRASH500'), undefined);
});
test('heatmap scope never borrows validation values for final comparisons', () => {
  const data = snapshot([row({ id: 'a', development: { validation_score: 1, validation: { profit_factor: 1.9 } } }), row({ id: 'b', development: { validation_score: 2, validation: { profit_factor: 1.5 } } })]);
  assert.equal(heatmapResult(data, 'BOOM500', 'M1', 'validation')?.id, 'b');
  assert.equal(heatmapResult(data, 'BOOM500', 'M1', 'oos')?.oos.profit_factor, null);
  assert.equal(heatmapResult(data, 'CRASH300', 'M1'), undefined);
});
test('completion counts unique dated receipts only, never projected totals', () => {
  const data = snapshot([], { campaign: { completed_cycles: 32 }, cycles: [
    { number: 1, status: 'COMPLETED', completed_at: '2026-10-11T10:00:00Z' },
    { number: 1, status: 'COMPLETED', completed_at: '2026-10-11T10:00:00Z' },
    { number: 2, status: 'RUNNING', completed_at: '2026-10-11T10:15:00Z' },
    { number: 3, status: 'COMPLETED' }, { number: 4, status: 'COMPLETED', completed_at: 'invalid' },
  ] });
  assert.equal(data.campaign.completed_cycles, 1); assert.equal(data.campaign.declared_completed_cycles, 32);
});
test('unsafe live flags and incompatible schema are rejected', () => {
  for (const key of ['LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES', 'ready_for_live', 'live_allowed', 'opened_trades']) {
    assert.throws(() => snapshot([], { safety: { ...RESEARCH_SAFETY, [key]: true } }));
  }
  assert.throws(() => parseCampaignSnapshot({ schema_version: 2 }));
});
test('lazy detail merge preserves scalar metrics, full ledger and sealed final', () => {
  const original = snapshot([row({ details_url: '/campaign/details/a.json', development: { validation: { metrics: { profit_factor: 1.2, completed_trades: 120 } } } })]).strategies[0];
  const detail = mergeCampaignDetails(original, { development: { validation: { trades: [{ id: 'trade1', entry_time: '2026-10-11T10:00:00Z', net_profit: -1 }], equity: [{ time: '2026-10-11T10:00:00Z', value: 99 }] } } });
  assert.equal(detail.development.validation.profit_factor, 1.2); assert.equal(detail.development.validation.trades.length, 1);
  assert.equal(detail.development.validation.trades[0].net_profit, -1); assert.equal(detail.development.validation.equity[0].value, 99);
  assert.equal(detail.oos_available, false); assert.equal(detail.oos.net_profit, null);
});
test('CSV preserves unknowns, scope and unicode while neutralizing spreadsheet formulas', () => {
  const data = snapshot([row({ name: '=HYPERLINK("bad")', reasons: ['اختبار، سبب'], development: { validation: { net_profit: -1.5 } } })]);
  const csv = resultCSV(data.strategies, 'validation');
  assert.ok(csv.startsWith('\uFEFFscope,symbol')); assert.ok(csv.includes('"validation"'));
  assert.ok(csv.includes('"\'=HYPERLINK(""bad"")"')); assert.ok(csv.includes('"-1.5"')); assert.ok(csv.includes('اختبار، سبب'));
  assert.ok(resultCSV(data.strategies, 'oos').includes('"NOT_TESTED"'));
});
