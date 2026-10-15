import assert from 'node:assert/strict';
import {test} from 'node:test';
import {getCRT} from '../src/lib/crt.ts';
import {normalizeBar, normalizeSymbols, upsertBar, closedM5} from '../src/lib/deriv.ts';
import type {Bar} from '../src/lib/deriv.ts';

const start = Math.floor(1700000000 / 3600) * 3600;
const currentHour = start + 6 * 3600;
const baseline: Bar[] = Array.from({length: 72}, (_, i) => ({time: start + i * 300, open: 101, high: 102, low: 100, close: 101.4}));
const sweep: Bar = {time: currentHour, open: 100.2, high: 101.4, low: 99.5, close: 100.8};
const mss: Bar = {time: currentHour + 300, open: 100.8, high: 102.6, low: 100.7, close: 102.1};
const confirmation: Bar = {time: currentHour + 600, open: 102.7, high: 104.5, low: 102.5, close: 104.4};
const check = test;

check('complete closed H1 becomes reference; waiting setup uses the prior hour', () => {
 const crt = getCRT(baseline, 'boom', currentHour)!;
 assert.equal(crt.refStart, currentHour - 3600); assert.equal(crt.high, 102); assert.equal(crt.low, 100); assert.equal(crt.state, 'waiting');
});
check('missing M5 cannot manufacture a complete H1', () => {
 assert.equal(getCRT(baseline.filter((_, i) => i !== 65), 'boom', currentHour), null);
});
check('open M5 never emits sweep/reclaim, even when a future bar is supplied', () => {
 const crt = getCRT([...baseline, sweep], 'boom', currentHour + 299)!;
 assert.equal(crt.state, 'waiting'); assert.equal(crt.events.length, 0);
});
check('valid sweep and interior reclaim are observed after M5 close', () => {
 const crt = getCRT([...baseline, sweep], 'boom', currentHour + 300)!;
 assert.equal(crt.state, 'reclaimed'); assert.ok(Math.abs(crt.sweepDepthATR! - 0.25) < 1e-9);
 assert.deepEqual(crt.events.slice(0, 2).map(x => x.type), ['sweep', 'reclaim']);
 assert.ok(crt.events.every(event => event.time <= currentHour + 300));
});
check('MSS plus displacement without valid FVG cannot confirm', () => {
 const crt = getCRT([...baseline, sweep, mss], 'boom', currentHour + 600)!;
 assert.equal(crt.state, 'reclaimed'); assert.equal(crt.mss, true); assert.equal(crt.displacement, true); assert.equal(crt.fvg, null);
});
check('MSS, displacement and FVG on the same closed bar confirm', () => {
 const crt = getCRT([...baseline, sweep, mss, confirmation], 'boom', currentHour + 900)!;
 assert.equal(crt.state, 'confirmed'); assert.equal(crt.fvg?.low, 101.4); assert.equal(crt.fvg?.high, 102.5);
 assert.equal(crt.fvg?.time, currentHour + 900);
});
check('replay remains causal when additional future bars are supplied', () => {
 const prefix = getCRT([...baseline, sweep], 'boom', currentHour + 300);
 const full = getCRT([...baseline, sweep, mss, confirmation], 'boom', currentHour + 300);
 assert.deepEqual(full, prefix);
});
check('closing beyond sweep extreme invalidates before confirmation', () => {
 const invalid: Bar = {time: currentHour + 300, open: 100.8, high: 101, low: 98.8, close: 99.2};
 const crt = getCRT([...baseline, sweep, invalid], 'boom', currentHour + 600)!;
 assert.equal(crt.state, 'invalid'); assert.equal(crt.events.at(-1)?.type, 'invalid');
});
check('reclaim requires an interior close; equality at CRL is not sufficient', () => {
 const equal = {...sweep, close: 100};
 assert.equal(getCRT([...baseline, equal], 'boom', currentHour + 300)?.state, 'sweep');
});
check('expired reclaim cannot silently become a confirmed signal', () => {
 const equal = {...sweep, close: 100};
 const remaining = [1, 2].map(i => ({time: currentHour + i * 300, open: 100, high: 100.5, low: 99.8, close: 100}));
 const crt = getCRT([...baseline, equal, ...remaining], 'boom', currentHour + 900)!;
 assert.equal(crt.state, 'expired'); assert.equal(crt.events.at(-1)?.time, currentHour + 900);
});
check('Crash reverses the liquidity sweep and confirmation direction', () => {
 const mirror = (bar: Bar): Bar => ({time: bar.time, open: 202-bar.open, high: 202-bar.low, low: 202-bar.high, close: 202-bar.close});
 const crt = getCRT([...baseline, sweep, mss, confirmation].map(mirror), 'crash', currentHour + 900)!;
 assert.equal(crt.state, 'confirmed'); assert.equal(crt.direction, 'down'); assert.equal(crt.fvg?.high, 100.6); assert.equal(crt.fvg?.low, 99.5);
});
check('new and legacy symbol responses normalize without invented ids', () => {
 const rows = normalizeSymbols([
  {underlying_symbol:'BOOM150N',underlying_symbol_name:'Boom 150 Index'},
  {underlying_symbol:'CRASH50',underlying_symbol_name:'Crash 50 Index'},
  {symbol:'BOOM500',display_name:'Boom 500 Index'},
  {symbol:'R_100',display_name:'Volatility 100 Index'},
 ]);
 assert.deepEqual(rows.map(x => x.id), ['BOOM150N', 'BOOM500', 'CRASH50']);
 assert.equal(rows[0].label, 'بووم 150');
});
check('history and live candles normalize numbers and reject inconsistent OHLC', () => {
 assert.deepEqual(normalizeBar({epoch:300,open:'101',high:'102',low:'100',close:'101.5'}), {time:300,open:101,high:102,low:100,close:101.5});
 assert.equal(normalizeBar({open_time:300,epoch:310,open:101,high:100,low:99,close:101.5}, true), null);
 assert.equal(normalizeBar({epoch:300,open:'bad',high:102,low:100,close:101}), null);
});
check('live updates replace same candle without duplicates; M5 filters open candles', () => {
 const updated = upsertBar([sweep], {...sweep, close:100.9});
 assert.equal(updated.length, 1); assert.equal(updated[0].close,100.9);
 assert.deepEqual(closedM5([...baseline,sweep], currentHour+299), baseline);
});
check('confirmation expires while preserving the already observed events', () => {
 const remaining: Bar = {time:currentHour+900,open:104.4,high:105,low:104,close:104.5};
 const crt = getCRT([...baseline,sweep,mss,confirmation,remaining],'boom',currentHour+1200)!;
 assert.equal(crt.state,'expired'); assert.equal(crt.events.at(-1)?.type,'expired');
 assert.ok(crt.events.some(event => event.type === 'fvg'));
});
check('a confirmed setup is invalidated by a later close beyond the sweep extreme', () => {
 const invalid: Bar = {time:currentHour+900,open:104.4,high:104.5,low:99,close:99.2};
 const crt = getCRT([...baseline,sweep,mss,confirmation,invalid],'boom',currentHour+1200)!;
 assert.equal(crt.state,'invalid'); assert.equal(crt.events.at(-1)?.type,'invalid');
});
check('ATR history gaps prevent a new sweep rather than replacing missing data', () => {
 const missing = baseline.filter((_, i) => i !== 59);
 const crt = getCRT([...missing,sweep],'boom',currentHour+300)!;
 assert.equal(crt.state,'waiting'); assert.equal(crt.events.length,0);
});


check('a sweep after a missing M5 cannot reuse ATR from a non-adjacent candle', () => {
 const lateSweep = {...sweep,time:currentHour+300};
 const crt = getCRT([...baseline,lateSweep],'boom',currentHour+600)!;
 assert.equal(crt.state,'waiting'); assert.equal(crt.sweepDepthATR,null); assert.equal(crt.events.length,0);
 // The identical sweep is eligible when the missing preceding candle actually exists.
 const preceding: Bar = {time:currentHour,open:101,high:102,low:100,close:101.4};
 assert.equal(getCRT([...baseline,preceding,lateSweep],'boom',currentHour+600)?.state,'reclaimed');
});
