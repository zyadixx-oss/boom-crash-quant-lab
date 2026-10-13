import logging
import math
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from ..db import SessionLocal
from ..models import Tick, Candle, ShadowSignal, Signal, SystemEvent
from ..symbol_profiles import get_profile
from .deriv_client import DerivClient
from .candles import aggregate_ticks, TF_SECONDS
from .indicators import add_features
from .market_structure import add_market_structure
from .signal_engine import score_row

log=logging.getLogger(__name__)


def closed_tick_bars(ticks, timeframe):
    """Only complete, contiguous buckets of the one-second public tick feed."""
    if ticks['epoch'].duplicated().any() or not ticks['epoch'].is_monotonic_increasing:
        raise ValueError('Shadow ticks must be strictly ordered and unique')
    candles = aggregate_ticks(ticks, timeframe)
    if candles.empty:
        return candles
    seconds = TF_SECONDS[timeframe]
    as_of = int(ticks['epoch'].max())
    candles = candles[(candles['epoch'] + seconds <= as_of)
                      & (candles['tick_count'] == seconds)].copy()
    if candles.empty:
        return candles
    if int(candles.iloc[-1]['epoch']) + seconds != (as_of // seconds) * seconds:
        return candles.iloc[0:0]
    # A missing bucket resets indicator history instead of compressing time.
    gaps = candles['epoch'].diff().ne(seconds)
    groups = gaps.cumsum()
    return candles[groups == groups.iloc[-1]].reset_index(drop=True)


class ShadowRunner:
    def __init__(self,symbol:str):
        self.profile=get_profile(symbol)
        self.direction='UP' if self.profile.symbol.startswith('BOOM') else 'DOWN'
        self.buffer=[]

    def _accept_tick(self, row):
        if row['epoch'] <= 0 or not math.isfinite(row['price']) or row['price'] <= 0:
            raise ValueError('Invalid shadow tick')
        if self.buffer and row['epoch'] <= self.buffer[-1]['epoch']:
            if row == self.buffer[-1]:
                return False
            raise ValueError('Out-of-order or conflicting shadow tick; observation stopped')
        self.buffer.append(row)
        self.buffer=self.buffer[-65000:]
        return True

    async def run(self):
        c=DerivClient(); await c.connect(); api_symbol=await c.resolve_symbol(self.profile.display_name,self.profile.api_symbol)
        if not api_symbol: raise RuntimeError(f'Could not resolve Deriv symbol for {self.profile.display_name}')
        with SessionLocal() as db:
            db.add(SystemEvent(level='INFO',event_type='shadow_start',message=f'{self.profile.symbol}->{api_symbol}',payload={'mode':'SHADOW_ONLY'})); db.commit()
        try:
            async for tick in c.subscribe_ticks(api_symbol):
                row={'epoch':int(tick['epoch']),'price':float(tick['quote'])}
                # M15 context needs at least 60 complete bars, including warm-up.
                if not self._accept_tick(row): continue
                self._store_tick(row); self._store_closed_candles(); self._evaluate_pending(row); await self._evaluate_signal()
        finally: await c.close()

    def _store_tick(self,row):
        with SessionLocal() as db:
            try: db.add(Tick(symbol=self.profile.symbol,epoch=row['epoch'],price=row['price'])); db.commit()
            except IntegrityError: db.rollback()

    def _store_closed_candles(self):
        if len(self.buffer)<10: return
        ticks=pd.DataFrame(self.buffer)
        now_epoch=int(ticks.iloc[-1]['epoch'])
        with SessionLocal() as db:
            for tf,seconds in [('M1',60),('M5',300),('M15',900)]:
                c=closed_tick_bars(ticks,tf)
                # Persist only completed buckets; the current bucket is still changing.
                closed=c[c['epoch']+seconds<=now_epoch]
                if closed.empty: continue
                r=closed.iloc[-1]
                exists=db.scalar(select(Candle.id).where(Candle.symbol==self.profile.symbol,Candle.timeframe==tf,Candle.epoch==int(r.epoch)))
                if exists: continue
                db.add(Candle(symbol=self.profile.symbol,timeframe=tf,epoch=int(r.epoch),open=float(r.open),high=float(r.high),low=float(r.low),close=float(r.close),tick_count=int(r.tick_count)))
            db.commit()

    def _context(self,ticks,tf):
        c=closed_tick_bars(ticks,tf)
        if len(c)<60: return None
        x=add_market_structure(add_features(c,self.profile)); row=x.iloc[-1]
        aligned=(float(row['close'])>=float(row['ema20'])) if self.direction=='UP' else (float(row['close'])<=float(row['ema20']))
        return {'epoch':int(row['epoch']),'close':float(row['close']),'ema20':float(row['ema20']),'trend':int(row.get('trend',0)),'aligned':bool(aligned)}

    async def _evaluate_signal(self):
        if len(self.buffer)<500: return
        ticks=pd.DataFrame(self.buffer)
        m1=closed_tick_bars(ticks,'M1')
        if len(m1)<70: return
        x=add_market_structure(add_features(m1,self.profile)); row=x.iloc[-1].copy()
        m5=self._context(ticks,'M5'); m15=self._context(ticks,'M15')
        row['m5_confirm']=bool(m5 and m5['aligned']); row['m15_confirm']=bool(m15 and m15['aligned'])
        dec=score_row(row,self.profile,self.direction)
        if dec.score<self.profile.min_signal_score: return
        with SessionLocal() as db:
            last=db.execute(select(ShadowSignal).where(ShadowSignal.symbol==self.profile.symbol).order_by(ShadowSignal.epoch.desc()).limit(1)).scalar_one_or_none()
            if last and dec.epoch-last.epoch < self.profile.cooldown_bars*60: return
            issued_epoch = int(ticks['epoch'].max())
            issued_quote = float(ticks.iloc[-1]['price'])
            sh=ShadowSignal(symbol=self.profile.symbol,epoch=dec.epoch,price=issued_quote,score=dec.score,reasons=dec.reason_codes,features=dec.features_snapshot,result={'status':'PENDING','test_window_bars':self.profile.test_window_bars,'issued_epoch':issued_epoch,'last_observed_epoch':issued_epoch,'closed_bar_reference_price':dec.entry_reference_price,'measurement_kind':'DIRECTIONAL_MOVE_ONLY','cost_verified':False,'execution_verified':False})
            db.add(sh)
            db.add(Signal(symbol=self.profile.symbol,epoch=dec.epoch,direction=dec.direction,score=dec.score,reason_codes=dec.reason_codes,features_snapshot=dec.features_snapshot,m1_context={'close':dec.entry_reference_price,'issued_epoch':issued_epoch,'issued_reference_quote':issued_quote},m5_context=m5 or {},m15_context=m15 or {},entry_reference_price=issued_quote,expected_test_window=self.profile.test_window_bars))
            db.add(SystemEvent(level='INFO',event_type='shadow_signal',message=f'{self.profile.symbol} score={dec.score}',payload={'reasons':dec.reason_codes})); db.commit()

    def _evaluate_pending(self,tick):
        current_epoch=int(tick['epoch']); current_price=float(tick['price'])
        with SessionLocal() as db:
            pending=db.execute(select(ShadowSignal).where(ShadowSignal.symbol==self.profile.symbol, ShadowSignal.result['status'].as_string()=='PENDING').order_by(ShadowSignal.epoch)).scalars().all()
            changed=False
            for s in pending:
                if (s.result or {}).get('status')!='PENDING': continue
                issued_epoch = (s.result or {}).get('issued_epoch')
                if issued_epoch is None:
                    s.result={**(s.result or {}),'status':'UNKNOWN','reason':'LEGACY_ISSUANCE_TIME_NOT_RECORDED','cost_verified':False,'execution_verified':False}
                    changed=True
                    continue
                horizon = issued_epoch + (s.result or {}).get('test_window_bars', self.profile.test_window_bars)*60
                if current_epoch > horizon:
                    s.result={**s.result,'status':'UNKNOWN','reason':'MISSING_HORIZON_TICK','evaluated_epoch':current_epoch}
                    changed=True
                    continue
                previous_epoch = s.result.get('last_observed_epoch', issued_epoch)
                if current_epoch > previous_epoch + 1:
                    s.result={**s.result,'status':'UNKNOWN','reason':'MISSING_INTERMEDIATE_TICKS','evaluated_epoch':current_epoch,'last_observed_epoch':previous_epoch}
                    changed=True
                    continue
                if current_epoch <= previous_epoch:
                    continue
                s.result={**s.result,'last_observed_epoch':current_epoch}
                changed=True
                if current_epoch < horizon: continue
                pnl=(current_price-s.price) if self.direction=='UP' else (s.price-current_price)
                s.result={**s.result,'status':'WIN' if pnl>0 else 'LOSS','evaluated_epoch':current_epoch,'exit_price':current_price,'signed_move':pnl}
                changed=True
            if changed: db.commit()
