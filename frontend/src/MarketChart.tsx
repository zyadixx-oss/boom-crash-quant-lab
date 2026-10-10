import { useEffect, useRef } from 'react';
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineStyle,
  TickMarkType,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Logical,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';

export type ChartBar = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
};

export type CRTView = {
  range: { high: number; low: number; mid: number; start: number; end: number } | null;
  events: Array<{
    time: number;
    type: 'sweep' | 'reclaim' | 'mss' | 'displacement' | 'fvg';
    direction: 'boom' | 'crash';
  }>;
};

type Props = {
  bars: ChartBar[];
  crt: CRTView | null;
  showCRT: boolean;
  showMarkers: boolean;
  onCrosshair?: (bar: ChartBar | null) => void;
};

const eventStyle = {
  sweep: { color: '#e3b76e', text: 'سحب سيولة', shape: 'circle' },
  reclaim: { color: '#53c7e6', text: 'استعادة النطاق', shape: 'square' },
  mss: { color: '#bb9cff', text: 'تحوّل الهيكل', shape: 'arrowUp' },
  displacement: { color: '#33d1a0', text: 'اندفاع', shape: 'arrowUp' },
  fvg: { color: '#91a7ff', text: 'فجوة سعرية', shape: 'square' },
} as const;

const clockFormat = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'UTC', hour: '2-digit', minute: '2-digit',
});
const dateFormat = new Intl.DateTimeFormat('ar-SA-u-nu-latn', {
  timeZone: 'UTC', calendar: 'gregory', month: 'short', day: 'numeric',
  hour: '2-digit', minute: '2-digit', hour12: false,
});
const axisDateFormat = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'UTC', month: 'short', day: 'numeric',
});

function cleanBars(bars: ChartBar[]): ChartBar[] {
  const unique = new Map<number, ChartBar>();
  for (const bar of bars) {
    if ([bar.time, bar.open, bar.high, bar.low, bar.close].every(Number.isFinite)
      && bar.high >= Math.max(bar.open, bar.close, bar.low)
      && bar.low <= Math.min(bar.open, bar.close, bar.high)) {
      unique.set(bar.time, bar);
    }
  }
  return [...unique.values()].sort((a, b) => a.time - b.time);
}

// Events are known at candle close; a close on a boundary maps to the preceding candle.
function eventBarTime(eventTime: number, bars: ChartBar[]): number | null {
  if (!bars.length || eventTime <= bars[0].time) return null;
  const interval = bars.length > 1 ? bars[bars.length - 1].time - bars[bars.length - 2].time : 60;
  if (eventTime > bars[bars.length - 1].time + interval) return null;
  let low = 0;
  let high = bars.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (bars[middle].time < eventTime) low = middle + 1;
    else high = middle - 1;
  }
  const bar = bars[high];
  return bar && eventTime <= bar.time + interval ? bar.time : null;
}

export default function MarketChart({ bars, crt, showCRT, showMarkers, onCrosshair }: Props) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const barsRef = useRef<ChartBar[]>([]);
  const crosshairTimeRef = useRef<number | null>(null);
  const callbackRef = useRef(onCrosshair);
  const rangeRef = useRef<CRTView['range']>(null);
  const drawOverlayRef = useRef<() => void>(() => {});
  callbackRef.current = onCrosshair;
  rangeRef.current = showCRT ? crt?.range ?? null : null;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let active = true;
    const panel = wrapperRef.current?.parentElement;
    const chartDimensions = () => {
      const width = Math.max(1, container.clientWidth);
      let height = width < 600 ? 360 : 450;
      if (panel?.classList.contains('chart-expanded')) {
        const chromeHeight = [...panel.children]
          .filter(child => child !== wrapperRef.current)
          .reduce((sum, child) => sum + (child as HTMLElement).offsetHeight, 0);
        height = Math.max(100, panel.clientHeight - chromeHeight);
      }
      return { width, height };
    };
    const chart = createChart(container, {
      ...chartDimensions(),
      layout: {
        background: { type: ColorType.Solid, color: '#11151d' },
        textColor: '#909ba9', fontFamily: 'Arial, sans-serif', fontSize: 11,
        attributionLogo: true,
      },
      grid: { vertLines: { color: '#1c232d' }, horzLines: { color: '#1c232d' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#27303b', scaleMargins: { top: 0.15, bottom: 0.15 } },
      timeScale: {
        borderColor: '#27303b', timeVisible: true, secondsVisible: false,
        rightOffset: 8, barSpacing: 7, minBarSpacing: 2,
        tickMarkFormatter: (time: Time, tickType: TickMarkType) => {
          if (typeof time !== 'number') return '';
          const date = new Date(time * 1000);
          if (tickType === TickMarkType.Year) return String(date.getUTCFullYear());
          if (tickType === TickMarkType.Month || tickType === TickMarkType.DayOfMonth) return axisDateFormat.format(date);
          return clockFormat.format(date);
        },
      },
      localization: {
        locale: 'en-GB',
        timeFormatter: (time: Time) => typeof time === 'number'
          ? dateFormat.format(new Date(time * 1000)) : '',
      },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#33d1a0', downColor: '#ef7f80', borderVisible: false,
      wickUpColor: '#33d1a0', wickDownColor: '#ef7f80',
      priceFormat: { type: 'price', precision: 3, minMove: 0.001 },
      priceLineColor: '#7e8b9b', priceLineStyle: LineStyle.Dotted,
      lastValueVisible: true,
    });
    chartRef.current = chart;
    seriesRef.current = series;
    markersRef.current = createSeriesMarkers(series, []);

    const drawOverlay = () => {
      if (!active) return;
      const canvas = overlayRef.current;
      if (!canvas) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      const context = canvas.getContext('2d');
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      const range = rangeRef.current;
      if (!range || !barsRef.current.length) return;
      const timeToX = (time: number) => {
        const data = barsRef.current;
        const exact = chart.timeScale().timeToCoordinate(time as UTCTimestamp);
        if (exact !== null) return exact;
        const interval = data.length > 1 ? data[1].time - data[0].time : 60;
        // A reference closes at a boundary that may not have a plotted bar yet.
        if (time < data[0].time) return chart.timeScale().logicalToCoordinate(((time - data[0].time) / interval) as Logical);
        if (time > data[data.length - 1].time) return chart.timeScale().logicalToCoordinate((data.length - 1 + (time - data[data.length - 1].time) / interval) as Logical);
        const right = data.findIndex(bar => bar.time > time);
        if (right < 1) return null;
        const fraction = (time - data[right - 1].time) / (data[right].time - data[right - 1].time);
        return chart.timeScale().logicalToCoordinate((right - 1 + fraction) as Logical);
      };
      const x1 = timeToX(range.start);
      const x2 = timeToX(range.end);
      const y1 = series.priceToCoordinate(range.high);
      const y2 = series.priceToCoordinate(range.low);
      if (x1 === null || x2 === null || y1 === null || y2 === null) return;
      const paneWidth = chart.timeScale().width();
      const paneHeight = chart.panes()[0]?.getHeight() ?? height - 28;
      context.save();
      context.beginPath();
      context.rect(0, 0, paneWidth, paneHeight);
      context.clip();
      context.fillStyle = 'rgba(227, 183, 110, 0.075)';
      context.fillRect(x1, y1, x2 - x1, y2 - y1);
      context.strokeStyle = 'rgba(227, 183, 110, 0.42)';
      context.lineWidth = 1;
      context.setLineDash([4, 4]);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (x2 - x1 > 80) {
        context.setLineDash([]);
        context.font = '11px Arial, sans-serif';
        context.textAlign = 'right';
        context.direction = 'rtl';
        context.fillStyle = '#b9a078';
        context.fillText('الشمعة المرجعية', Math.min(x2 - 8, paneWidth - 8), Math.max(y1 + 18, 18));
      }
      context.restore();
    };
    drawOverlayRef.current = drawOverlay;
    const crosshairHandler = (param: MouseEventParams<Time>) => {
      crosshairTimeRef.current = typeof param.time === 'number' && param.point ? param.time : null;
      callbackRef.current?.(crosshairTimeRef.current === null ? null : barsRef.current.find(bar => bar.time === crosshairTimeRef.current) ?? null);
      drawOverlay();
    };
    chart.subscribeCrosshairMove(crosshairHandler);
    chart.timeScale().subscribeVisibleLogicalRangeChange(drawOverlay);
    // Price-axis gestures do not always emit crosshair/time-scale events.
    const redrawAfterInteraction = () => {
      requestAnimationFrame(() => {
        drawOverlay();
        requestAnimationFrame(drawOverlay);
      });
    };
    container.addEventListener('pointermove', redrawAfterInteraction, { passive: true });
    container.addEventListener('wheel', redrawAfterInteraction, { passive: true });
    container.addEventListener('dblclick', redrawAfterInteraction, { passive: true });
    const resizeChart = () => {
      if (!active) return;
      const dimensions = chartDimensions();
      const options = chart.options();
      if (options.width !== dimensions.width || options.height !== dimensions.height) chart.applyOptions(dimensions);
      requestAnimationFrame(drawOverlay);
    };
    const observer = new ResizeObserver(resizeChart);
    observer.observe(container);
    if (panel) observer.observe(panel);
    const panelObserver = new MutationObserver(resizeChart);
    if (panel) panelObserver.observe(panel, { attributes: true, attributeFilter: ['class'], childList: true });
    window.addEventListener('resize', resizeChart, { passive: true });
    return () => {
      active = false;
      observer.disconnect();
      panelObserver.disconnect();
      window.removeEventListener('resize', resizeChart);
      container.removeEventListener('pointermove', redrawAfterInteraction);
      container.removeEventListener('wheel', redrawAfterInteraction);
      container.removeEventListener('dblclick', redrawAfterInteraction);
      chart.unsubscribeCrosshairMove(crosshairHandler);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(drawOverlay);
      markersRef.current?.detach();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
      priceLinesRef.current = [];
      barsRef.current = [];
      crosshairTimeRef.current = null;
      callbackRef.current?.(null);
      drawOverlayRef.current = () => {};
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    const next = cleanBars(bars);
    const previous = barsRef.current;
    const previousLast = previous[previous.length - 1];
    const nextLast = next[next.length - 1];
    const resetView = previous.length === 0 || (previousLast && nextLast && nextLast.time < previousLast.time)
      || (previous.length > 1 && next.length > 1 && previous[1].time - previous[0].time !== next[1].time - next[0].time);
    barsRef.current = next;
    series.setData(next.map(bar => ({ ...bar, time: bar.time as UTCTimestamp })));
    if (crosshairTimeRef.current !== null) callbackRef.current?.(next.find(bar => bar.time === crosshairTimeRef.current) ?? null);
    if (resetView && next.length) {
      chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, next.length - 110), to: next.length + 7 });
    }
    requestAnimationFrame(drawOverlayRef.current);
  }, [bars]);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    for (const line of priceLinesRef.current) series.removePriceLine(line);
    priceLinesRef.current = [];
    if (showCRT && crt?.range) {
      const range = crt.range;
      priceLinesRef.current = [
        series.createPriceLine({ price: range.high, color: '#e3b76e', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'CRH · القمة' }),
        series.createPriceLine({ price: range.low, color: '#e3b76e', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'CRL · القاع' }),
        series.createPriceLine({ price: range.mid, color: '#687583', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: false, title: '50% · المنتصف' }),
      ];
    }
    const mapped: SeriesMarker<Time>[] = [];
    if (showMarkers && crt) {
      for (const event of crt.events) {
        const time = eventBarTime(event.time, barsRef.current);
        if (time === null) continue;
        const style = eventStyle[event.type];
        const arrow = style.shape === 'arrowUp';
        mapped.push({
          time: time as UTCTimestamp, color: style.color,
          position: event.direction === 'boom' ? 'belowBar' : 'aboveBar',
          shape: arrow && event.direction === 'crash' ? 'arrowDown' : style.shape,
          text: style.text,
        });
      }
      mapped.sort((a, b) => Number(a.time) - Number(b.time));
    }
    markersRef.current?.setMarkers(mapped);
    requestAnimationFrame(drawOverlayRef.current);
  }, [crt, showCRT, showMarkers, bars]);

  const latest = () => {
    const length = barsRef.current.length;
    if (length) chartRef.current?.timeScale().setVisibleLogicalRange({ from: Math.max(0, length - 110), to: length + 7 });
    drawOverlayRef.current();
  };

  return (
    <div ref={wrapperRef} className="market-chart" style={{ position: 'relative', width: '100%', direction: 'ltr', flex: '1 1 0', minHeight: 0 }}>
      <div ref={containerRef} role="img" aria-label="شارت الشموع اليابانية مع نطاق CRT ومراحل سحب السيولة. اسحب للتنقل واستخدم عجلة الفأرة للتكبير." style={{ width: '100%' }} />
      <canvas ref={overlayRef} aria-hidden="true" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 1 }} />
      <button className="chart-latest" onClick={latest} type="button" title="إعادة عرض آخر 110 شموع" style={{ direction: 'rtl' }}>
        أحدث الشموع <span aria-hidden="true">↗</span>
      </button>
    </div>
  );
}
