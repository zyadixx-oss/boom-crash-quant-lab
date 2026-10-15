import type {Bar, SymbolInfo} from './deriv';

export type CRTState = 'waiting' | 'sweep' | 'reclaimed' | 'confirmed' | 'invalid' | 'expired';
export type CRTEvent = {
  type: 'sweep' | 'reclaim' | 'mss' | 'displacement' | 'fvg' | 'invalid' | 'expired';
  time: number; price: number; referenceStart: number;
};
export type FVG = {low: number; high: number; time: number};
export type CRTAnalysis = {
  high: number; low: number; midpoint: number; refStart: number; refEnd: number;
  expiresAt: number; state: CRTState; direction: 'up' | 'down'; sweepDepthATR: number | null;
  events: CRTEvent[]; history: CRTEvent[]; mss: boolean; displacement: boolean;
  fvg: FVG | null; lastClosedTime: number;
};

type Features = {atr: number | null; previousATR: number | null; mss: boolean; displacement: boolean; fvg: FVG | null};
const M5 = 300;
const H1 = 3600;

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function completeHour(bars: Bar[], start: number): Bar | null {
  const hour = bars.filter(bar => bar.time >= start && bar.time < start + H1);
  if (hour.length !== 12 || hour.some((bar, i) => bar.time !== start + i * M5)) return null;
  return {time: start, open: hour[0].open, high: Math.max(...hour.map(bar => bar.high)),
    low: Math.min(...hour.map(bar => bar.low)), close: hour[11].close};
}

function buildFeatures(bars: Bar[], family: SymbolInfo['family']): Features[] {
  const trueRanges: (number | null)[] = bars.map((bar, i) => {
    if (!i || bar.time - bars[i - 1].time !== M5) return null;
    return Math.max(bar.high - bar.low, Math.abs(bar.high - bars[i - 1].close), Math.abs(bar.low - bars[i - 1].close));
  });
  const atr = bars.map((_, i) => {
    const recent = trueRanges.slice(Math.max(0, i - 13), i + 1);
    if (recent.length !== 14 || recent.some(value => value === null)) return null;
    return (recent as number[]).reduce((sum, value) => sum + value, 0) / 14;
  });
  return bars.map((bar, i) => {
    const previous = bars.slice(Math.max(0, i - 20), i);
    const contiguous = previous.length === 20 && previous.every((item, j) => item.time === bar.time - (20 - j) * M5);
    const previousFive = previous.slice(-5);
    const directional = family === 'boom' ? bar.close > bar.open : bar.close < bar.open;
    const bodyMedian = contiguous ? median(previous.map(item => Math.abs(item.close - item.open))) : 0;
    const mss = contiguous && (family === 'boom' ? bar.close > Math.max(...previousFive.map(item => item.high)) : bar.close < Math.min(...previousFive.map(item => item.low)));
    const displacement = contiguous && directional && bodyMedian > 0 && Math.abs(bar.close - bar.open) >= 1.3 * bodyMedian;
    let fvg: FVG | null = null;
    if (i >= 2 && bars[i - 2].time === bar.time - 2 * M5 && atr[i] !== null && directional) {
      const low = family === 'boom' ? bars[i - 2].high : bar.high;
      const high = family === 'boom' ? bar.low : bars[i - 2].low;
      if (high > low && high - low >= 0.08 * (atr[i] as number)) fvg = {low, high, time: bar.time + M5};
    }
    const previousATR = i > 0 && bars[i - 1].time === bar.time - M5 ? atr[i - 1] : null;
    return {atr: atr[i], previousATR, mss, displacement, fvg};
  });
}

function replayReference(bars: Bar[], features: Features[], ref: Bar, family: SymbolInfo['family'], now: number): CRTAnalysis {
  const analysis: CRTAnalysis = {
    high: ref.high, low: ref.low, midpoint: (ref.high + ref.low) / 2,
    refStart: ref.time, refEnd: ref.time + H1, expiresAt: ref.time + 2 * H1,
    state: 'waiting', direction: family === 'boom' ? 'up' : 'down', sweepDepthATR: null,
    events: [], history: [], mss: false, displacement: false, fvg: null,
    lastClosedTime: bars.at(-1)!.time + M5,
  };
  const add = (type: CRTEvent['type'], bar: Bar, price = bar.close) => {
    analysis.events.push({type, time: bar.time + M5, price, referenceStart: ref.time});
  };
  const depth = (bar: Bar) => family === 'boom' ? ref.low - bar.low : bar.high - ref.high;
  let sweepIndex = -1;
  let sweepATR = 0;
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (bar.time < ref.time + H1 || bar.time >= ref.time + 2 * H1) continue;
    const a = features[i].previousATR;
    if (a !== null && a > 0 && depth(bar) >= 0.10 * a && depth(bar) <= 0.65 * a) {
      sweepIndex = i; sweepATR = a; analysis.state = 'sweep';
      analysis.sweepDepthATR = depth(bar) / a;
      analysis.expiresAt = bar.time + 3 * M5;
      add('sweep', bar, family === 'boom' ? bar.low : bar.high);
      break;
    }
  }
  if (sweepIndex < 0) return analysis;
  let reclaimIndex = -1;
  let extreme = family === 'boom' ? bars[sweepIndex].low : bars[sweepIndex].high;
  for (let i = sweepIndex; i < Math.min(sweepIndex + 3, bars.length); i++) {
    const bar = bars[i];
    if (bar.time !== bars[sweepIndex].time + (i - sweepIndex) * M5 || depth(bar) > 0.65 * sweepATR) {
      analysis.state = 'invalid'; add('invalid', bar); return analysis;
    }
    extreme = family === 'boom' ? Math.min(extreme, bar.low) : Math.max(extreme, bar.high);
    if (bar.close > ref.low && bar.close < ref.high) {
      reclaimIndex = i; analysis.state = 'reclaimed'; add('reclaim', bar);
      analysis.expiresAt = bar.time + 4 * M5;
      break;
    }
  }
  if (reclaimIndex < 0) {
    if (now >= analysis.expiresAt) {
      analysis.state = 'expired';
      analysis.events.push({type: 'expired', time: analysis.expiresAt, price: bars[Math.min(sweepIndex + 2, bars.length - 1)].close, referenceStart: ref.time});
    }
    return analysis;
  }
  const recorded = new Set<CRTEvent['type']>();
  for (let i = reclaimIndex; i < Math.min(reclaimIndex + 4, bars.length); i++) {
    const bar = bars[i];
    if (bar.time !== bars[reclaimIndex].time + (i - reclaimIndex) * M5 ||
        (family === 'boom' ? bar.close < extreme : bar.close > extreme)) {
      analysis.state = 'invalid'; add('invalid', bar); return analysis;
    }
    const feature = features[i];
    for (const [type, present] of [['mss', feature.mss], ['displacement', feature.displacement], ['fvg', !!feature.fvg]] as const) {
      if (present && !recorded.has(type)) { add(type, bar); recorded.add(type); }
    }
    analysis.mss ||= feature.mss;
    analysis.displacement ||= feature.displacement;
    if (analysis.state !== 'confirmed') analysis.fvg = feature.fvg ?? analysis.fvg;
    // Research variant requires all three features on the same closed confirmation bar.
    if (feature.mss && feature.displacement && feature.fvg) {
      analysis.state = 'confirmed'; analysis.fvg = feature.fvg;
    }
  }
  if (now >= analysis.expiresAt) {
    analysis.state = 'expired';
    analysis.events.push({type: 'expired', time: analysis.expiresAt, price: bars[Math.min(reclaimIndex + 3, bars.length - 1)].close, referenceStart: ref.time});
  }
  return analysis;
}

/** Every observation is available only after its M5 candle closes; incomplete hours never become CRT ranges. */
export function getCRT(barsM5: Bar[], family: SymbolInfo['family'], nowEpoch = Math.floor(Date.now() / 1000)): CRTAnalysis | null {
  const byTime = new Map<number, Bar>();
  for (const bar of barsM5) {
    if (bar.time % M5 === 0 && bar.time + M5 <= nowEpoch &&
        Object.values(bar).every(Number.isFinite) && Math.min(bar.open, bar.high, bar.low, bar.close) > 0 && bar.high >= Math.max(bar.open, bar.close, bar.low) &&
        bar.low <= Math.min(bar.open, bar.close, bar.high)) byTime.set(bar.time, bar);
  }
  const bars = [...byTime.values()].sort((a, b) => a.time - b.time);
  if (bars.length < 12) return null;
  const referenceStart = Math.floor(nowEpoch / H1) * H1 - H1;
  const currentReference = completeHour(bars, referenceStart);
  if (!currentReference) return null;
  const features = buildFeatures(bars, family);
  const current = replayReference(bars, features, currentReference, family, nowEpoch);
  const historicalEvents: CRTEvent[] = [];
  // A bounded history provides chart observations without evaluating thousands of old ranges each tick.
  for (let hour = Math.max(Math.floor(bars[0].time / H1) * H1, referenceStart - 48 * H1); hour <= referenceStart; hour += H1) {
    const ref = completeHour(bars, hour);
    if (!ref) continue;
    historicalEvents.push(...(hour === referenceStart ? current.events : replayReference(bars, features, ref, family, nowEpoch).events));
  }
  current.history = historicalEvents.sort((a, b) => a.time - b.time || a.referenceStart - b.referenceStart).slice(-12);
  return current;
}
