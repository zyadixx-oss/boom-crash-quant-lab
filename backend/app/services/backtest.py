import pandas as pd
from .indicators import add_features
from .market_structure import add_market_structure
from .spike import label_spikes
from .signal_engine import score_row
from .metrics import trading_metrics, classification_metrics

class BacktestEngine:
    def __init__(self, profile): self.profile=profile; self.direction='UP' if profile.symbol.startswith('BOOM') else 'DOWN'
    def prepare(self, candles):
        x=add_features(candles,self.profile); x=add_market_structure(x); x=label_spikes(x,self.profile,self.direction); return x
    def run(self, candles:pd.DataFrame, min_score=None):
        x=self.prepare(candles); threshold=min_score if min_score is not None else self.profile.min_signal_score; trades=[]; preds=[]; truths=[]; last_signal=-10**9
        warmup=max(60,self.profile.bb_period+40,self.profile.atr_period+40)
        horizon=self.profile.test_window_bars
        for i in range(warmup, len(x)-horizon):
            row=x.iloc[i]; dec=score_row(row,self.profile,self.direction); pred=dec.score>=threshold; preds.append(pred); truths.append(bool(row.get('pre_spike',False)))
            if not pred or i-last_signal<self.profile.cooldown_bars: continue
            entry=float(row['close']); exit_=float(x.iloc[i+horizon]['close']); pnl=(exit_-entry) if self.direction=='UP' else (entry-exit_)
            trades.append({'epoch':int(row['epoch']),'direction':self.direction,'entry':entry,'exit':exit_,'pnl':pnl,'score':dec.score,'reasons':dec.reason_codes}); last_signal=i
        m=trading_metrics([t['pnl'] for t in trades]); m.update(classification_metrics(truths,preds)); m['total_signals']=sum(preds); m['signal_frequency']=sum(preds)/len(preds) if preds else 0
        return {'metrics':m,'trades':trades,'rows_evaluated':len(preds),'classification':'NEEDS MORE DATA' if len(trades)<30 else ('PROMISING' if m['profit_factor']>1 and m['expectancy']>0 else 'FAIL')}

def chronological_split(df, train=0.6, validation=0.2):
    n=len(df); a=int(n*train); b=int(n*(train+validation)); return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()

def walk_forward(df, profile, train_size=500, val_size=150, test_size=150, step=150):
    engine=BacktestEngine(profile); out=[]; start=0
    while start+train_size+val_size+test_size<=len(df):
        tr=df.iloc[start:start+train_size]; va=df.iloc[start+train_size:start+train_size+val_size]; te=df.iloc[start+train_size+val_size:start+train_size+val_size+test_size]
        # train is reserved for feature history/future extensibility; threshold selection uses validation only.
        candidates=[profile.min_signal_score-5, profile.min_signal_score, profile.min_signal_score+5]
        scored=[(engine.run(va,s)['metrics']['expectancy'],s) for s in candidates]; best=max(scored,key=lambda z:z[0])[1]
        out.append({'start':start,'threshold':best,'validation':engine.run(va,best)['metrics'],'test':engine.run(te,best)['metrics']}); start+=step
    return out
