import {useCallback, useEffect, useState} from 'react';

export type Bar = {time: number; open: number; high: number; low: number; close: number};
export type SymbolInfo = {id: string; label: string; family: 'boom' | 'crash'; displayName: string};
export type Timeframe = 'M1' | 'M5' | 'M15' | 'H1';
export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'error';
export type MarketState = {
  symbols: SymbolInfo[];
  bars: Bar[];
  analysisBars: Bar[];
  price: number | null;
  lastTickEpoch: number | null;
  status: ConnectionStatus;
  error: string | null;
  isStale: boolean;
};

export const PUBLIC_ENDPOINT = 'wss://api.derivws.com/trading/v1/options/ws/public';
export const GRANULARITY: Record<Timeframe, number> = {M1: 60, M5: 300, M15: 900, H1: 3600};
const LIMIT = 1000;
const STALE_AFTER_SECONDS = 20;

const initialState: MarketState = {
  symbols: [], bars: [], analysisBars: [], price: null, lastTickEpoch: null,
  status: 'connecting', error: null, isStale: true,
};

export function normalizeBar(raw: Record<string, unknown>, live = false): Bar | null {
  const bar = {
    time: Number(live ? raw.open_time : raw.epoch), open: Number(raw.open),
    high: Number(raw.high), low: Number(raw.low), close: Number(raw.close),
  };
  if (!Object.values(bar).every(Number.isFinite) || bar.time <= 0 ||
      Math.min(bar.open, bar.high, bar.low, bar.close) <= 0 ||
      bar.high < Math.max(bar.open, bar.close, bar.low) ||
      bar.low > Math.min(bar.open, bar.close, bar.high)) return null;
  return bar;
}

export function normalizeSymbols(rows: Record<string, unknown>[]): SymbolInfo[] {
  const found = new Map<string, SymbolInfo>();
  for (const row of rows) {
    const id = String(row.underlying_symbol || row.symbol || '');
    const displayName = String(row.underlying_symbol_name || row.display_name || '');
    const match = displayName.match(/\b(boom|crash)\s+(\d+)\b/i);
    if (!id || !match) continue;
    const family = match[1].toLowerCase() as SymbolInfo['family'];
    found.set(id, {id, displayName, family, label: `${family === 'boom' ? 'بووم' : 'كراش'} ${match[2]}`});
  }
  return [...found.values()].sort((a, b) =>
    a.family.localeCompare(b.family) ||
    Number(a.displayName.match(/\d+/)?.[0]) - Number(b.displayName.match(/\d+/)?.[0]));
}

export function upsertBar(bars: Bar[], next: Bar): Bar[] {
  const index = bars.findIndex(bar => bar.time === next.time);
  if (index >= 0) return [...bars.slice(0, index), next, ...bars.slice(index + 1)];
  return [...bars, next].sort((a, b) => a.time - b.time).slice(-LIMIT);
}

export function closedM5(bars: Bar[], epoch: number): Bar[] {
  return bars.filter(bar => bar.time % 300 === 0 && bar.time + 300 <= epoch);
}

/** A session owns one public socket and cancels every pending request and subscription on cleanup. */
export function createDerivMarketSession(
  symbol: string,
  timeframe: Timeframe,
  updateState: (update: (previous: MarketState) => MarketState) => void,
): () => void {
  let disposed = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  let connectionTimer: ReturnType<typeof setTimeout> | undefined;
  let attempt = 0;
  let requestId = 0;
  let chartBars: Bar[] = [];
  let m5Bars: Bar[] = [];
  let latestTick: number | null = null;
  let chartHistoryReady = false;
  let analysisHistoryReady = false;
  let chartTickEpoch: number | null = null;
  let analysisTickEpoch: number | null = null;
  let chartTickReceived: number | null = null;
  let analysisTickReceived: number | null = null;
  let chartFeedStarted: number | null = null;
  let analysisFeedStarted: number | null = null;
  let analysisWatermark: number | null = null;
  const analysisValidatedTimes = new Set<number>();
  const analysisLiveTimes = new Set<number>();
  const latestByKind = new Map<'chart' | 'analysis', number>();
  let lastReceive = 0;
  const selectedGranularity = GRANULARITY[timeframe];
  const pending = new Map<number, {kind: 'symbols' | 'chart' | 'analysis'; timer: ReturnType<typeof setTimeout>}>();
  const subscriptionKinds = new Map<string, 'chart' | 'analysis'>();
  const requestKinds = new Map<number, 'chart' | 'analysis'>();

  updateState(old => ({...initialState, symbols: old.symbols}));

  function clearRequests() {
    for (const request of pending.values()) clearTimeout(request.timer);
    pending.clear();
    requestKinds.clear();
    subscriptionKinds.clear();
    clearTimeout(connectionTimer);
  }

  function fail(message: string, close = true) {
    if (disposed) return;
    updateState(old => ({...old, status: 'error', error: message, isStale: true}));
    if (close && socket && socket.readyState < WebSocket.CLOSING) socket.close();
  }

  function request(payload: Record<string, unknown>, kind: 'symbols' | 'chart' | 'analysis') {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const reqId = ++requestId;
    if (kind !== 'symbols') requestKinds.set(reqId, kind);
    const timer = setTimeout(() => {
      pending.delete(reqId);
      fail('تأخر وصول بيانات السوق. جارٍ إعادة الاتصال…');
    }, 15000);
    pending.set(reqId, {kind, timer});
    socket.send(JSON.stringify({...payload, req_id: reqId}));
  }

  function isFreshTick(epoch: number | null, received: number | null) {
    if (epoch === null || received === null) return false;
    const ageSeconds = Date.now() / 1000 - epoch;
    return ageSeconds >= -5 && ageSeconds <= STALE_AFTER_SECONDS && Date.now() - received <= STALE_AFTER_SECONDS * 1000;
  }

  function isFreshMarket() {
    return socket?.readyState === WebSocket.OPEN && chartHistoryReady && analysisHistoryReady &&
      isFreshTick(chartTickEpoch, chartTickReceived) && isFreshTick(analysisTickEpoch, analysisTickReceived);
  }

  function publish(tickEpoch?: number, tickPrice?: number) {
    if (disposed) return;
    const acceptsTick = tickEpoch !== undefined && Number.isFinite(tickEpoch) && (latestTick === null || tickEpoch >= latestTick);
    if (acceptsTick) latestTick = tickEpoch;
    const firstRetainedTime = m5Bars[0]?.time;
    if (firstRetainedTime !== undefined) {
      for (const time of analysisValidatedTimes) if (time < firstRetainedTime) analysisValidatedTimes.delete(time);
      for (const time of analysisLiveTimes) if (time < firstRetainedTime) analysisLiveTimes.delete(time);
    }
    const cutoff = analysisWatermark ?? 0;
    const fresh = isFreshMarket();
    updateState(old => ({
      ...old, bars: chartBars, analysisBars: closedM5(m5Bars, cutoff).filter(bar => analysisValidatedTimes.has(bar.time)),
      price: acceptsTick ? tickPrice ?? old.price : latestTick === null ? chartBars.at(-1)?.close ?? old.price : old.price,
      lastTickEpoch: latestTick,
      status: fresh ? 'connected' : old.status,
      error: null,
      isStale: !fresh,
    }));
  }

  function connect() {
    if (disposed) return;
    clearRequests();
    chartHistoryReady = false;
    analysisHistoryReady = false;
    chartTickEpoch = null;
    analysisTickEpoch = null;
    chartTickReceived = null;
    analysisTickReceived = null;
    chartFeedStarted = null;
    analysisFeedStarted = null;
    analysisWatermark = null;
    analysisValidatedTimes.clear();
    analysisLiveTimes.clear();
    latestByKind.clear();
    updateState(old => ({...old, status: attempt ? 'reconnecting' : 'connecting', isStale: true}));
    const ws = new WebSocket(PUBLIC_ENDPOINT);
    socket = ws;
    connectionTimer = setTimeout(() => {
      if (socket === ws) fail('تعذر الاتصال ببيانات Deriv العامة.');
    }, 15000);
    ws.onopen = () => {
      if (disposed || socket !== ws) return;
      clearTimeout(connectionTimer);
      lastReceive = Date.now();
      request({active_symbols: 'brief'}, 'symbols');
    };
    ws.onmessage = event => {
      if (disposed || socket !== ws || ws.readyState !== WebSocket.OPEN) return;
      lastReceive = Date.now();
      let message: any;
      try { message = JSON.parse(event.data); } catch { return; }
      const reqId = Number(message.req_id);
      const waiting = pending.get(reqId);
      if (waiting && (message.error || message.msg_type === 'candles' || message.msg_type === 'active_symbols')) {
        clearTimeout(waiting.timer); pending.delete(reqId);
      }
      if (message.error) {
        const code = String(message.error.code || '');
        fail(code === 'InvalidSymbol' ? 'هذا الرمز غير متاح الآن لدى Deriv.' : 'تعذر تحميل بيانات السوق العامة. أعد الاتصال للمحاولة مجددًا.');
        return;
      }
      if (message.msg_type === 'active_symbols') {
        if (!Array.isArray(message.active_symbols)) { fail('لم تصل قائمة رموز صالحة من مزود البيانات.'); return; }
        const symbols = normalizeSymbols(message.active_symbols);
        updateState(old => ({...old, symbols}));
        if (!symbols.some(item => item.id === symbol)) {
          fail('الرمز المحدد غير موجود ضمن رموز بووم وكراش المتاحة.');
          return;
        }
        chartFeedStarted = Date.now();
        if (selectedGranularity === 300) analysisFeedStarted = chartFeedStarted;
        request({ticks_history: symbol, count: LIMIT, end: 'latest', style: 'candles',
          granularity: selectedGranularity, subscribe: 1}, 'chart');
        if (selectedGranularity !== 300) {
          analysisFeedStarted = Date.now();
          request({ticks_history: symbol, count: LIMIT, end: 'latest', style: 'candles',
            granularity: 300, subscribe: 1}, 'analysis');
        }
        return;
      }
      const subscriptionId = String(message.subscription?.id || message.ohlc?.id || '');
      const kind = subscriptionKinds.get(subscriptionId) || requestKinds.get(reqId);
      if (!kind) return;
      if (subscriptionId) subscriptionKinds.set(subscriptionId, kind);
      if (message.msg_type === 'candles' && Array.isArray(message.candles)) {
        const deduped = new Map<number, Bar>();
        for (const raw of message.candles) {
          const bar = normalizeBar(raw);
          if (bar) deduped.set(bar.time, bar);
        }
        const bars = [...deduped.values()].sort((a, b) => a.time - b.time).slice(-LIMIT);
        if (!bars.length) { fail('لم تصل شموع صالحة لهذا الرمز.'); return; }
        if (kind === 'chart') { chartBars = bars; chartHistoryReady = true; }
        if (kind === 'analysis' || selectedGranularity === 300) {
          // History's last candle may be a partial snapshot. Only the analysis stream can later validate it.
          const lastHistoryTime = bars.at(-1)!.time;
          for (const historical of bars) {
            if (historical.time < lastHistoryTime) analysisValidatedTimes.add(historical.time);
          }
          const liveBars = m5Bars.filter(bar => analysisLiveTimes.has(bar.time));
          m5Bars = bars;
          for (const live of liveBars) m5Bars = upsertBar(m5Bars, live);
          analysisWatermark = Math.max(analysisWatermark ?? 0, lastHistoryTime);
          analysisHistoryReady = true;
        }
        publish();
        return;
      }
      if (message.msg_type === 'ohlc' && message.ohlc) {
        if (message.ohlc.symbol !== symbol) return;
        const bar = normalizeBar(message.ohlc, true);
        const epoch = Number(message.ohlc.epoch);
        const granularity = kind === 'analysis' ? 300 : selectedGranularity;
        if (!bar || !Number.isInteger(epoch) || Number(message.ohlc.granularity) !== granularity ||
            epoch < bar.time || epoch >= bar.time + granularity || epoch < (latestByKind.get(kind) ?? 0)) return;
        latestByKind.set(kind, epoch);
        if (kind === 'chart') {
          chartBars = upsertBar(chartBars, bar);
          chartTickEpoch = epoch;
          chartTickReceived = Date.now();
        }
        if (kind === 'analysis' || selectedGranularity === 300) {
          m5Bars = upsertBar(m5Bars, bar);
          analysisTickEpoch = epoch;
          analysisTickReceived = Date.now();
          analysisLiveTimes.add(bar.time);
          analysisValidatedTimes.add(bar.time);
          analysisWatermark = Math.max(analysisWatermark ?? 0, epoch);
        }
        if (isFreshMarket()) attempt = 0;
        publish(epoch, bar.close);
      }
    };
    ws.onerror = () => {
      if (!disposed && socket === ws) fail('انقطع الاتصال ببيانات السوق العامة.');
    };
    ws.onclose = () => {
      if (disposed || socket !== ws) return;
      clearRequests();
      latestTick = null;
      const delay = Math.min(1000 * 2 ** attempt, 30000);
      attempt += 1;
      if (attempt > 8) { fail('تعذر استعادة الاتصال. اضغط إعادة الاتصال.', false); return; }
      updateState(old => ({...old, status: 'reconnecting', isStale: true}));
      reconnectTimer = setTimeout(connect, delay);
    };
  }

  const freshnessTimer = setInterval(() => {
    if (disposed) return;
    const stale = !isFreshMarket();
    updateState(old => old.isStale === stale ? old : {...old, isStale: stale});
    const chartSilent = chartFeedStarted !== null && Date.now() - (chartTickReceived ?? chartFeedStarted) > 30000;
    const analysisSilent = analysisFeedStarted !== null && Date.now() - (analysisTickReceived ?? analysisFeedStarted) > 30000;
    if (socket?.readyState === WebSocket.OPEN && ((lastReceive && Date.now() - lastReceive > 30000) || chartSilent || analysisSilent)) {
      fail('توقفت تحديثات السوق. جارٍ إعادة الاتصال…');
    }
  }, 2000);
  connect();
  return () => {
    if (disposed) return;
    disposed = true;
    clearTimeout(reconnectTimer);
    clearInterval(freshnessTimer);
    clearRequests();
    // Each selection owns its socket; closing cancels every subscription, including pending ones.
    socket?.close();
  };
}

/** Public prices only: no token, private endpoint, buy, sell or proposal request. */
export function useDerivMarket(symbol: string, timeframe: Timeframe) {
  const [state, setState] = useState<MarketState>(initialState);
  const [retryNonce, setRetryNonce] = useState(0);
  const retry = useCallback(() => setRetryNonce(value => value + 1), []);
  useEffect(() => createDerivMarketSession(symbol, timeframe, setState), [symbol, timeframe, retryNonce]);
  return {...state, retry};
}
