from copy import deepcopy
import numpy as np
import pandas as pd
from .backtest import BacktestEngine
from .metrics import trading_metrics


def _baseline_outcomes(df, direction, indexes, horizon=5):
    pnls=[]
    for i in indexes:
        if i+horizon>=len(df): continue
        entry=float(df.iloc[i]['close']); exit_=float(df.iloc[i+horizon]['close'])
        pnls.append(exit_-entry if direction=='UP' else entry-exit_)
    return trading_metrics(pnls)


def run_baselines(candles:pd.DataFrame, profile, seed=17):
    engine=BacktestEngine(profile); x=engine.prepare(candles); direction=engine.direction; warmup=max(60,profile.bb_period+40)
    eligible=np.arange(warmup,max(warmup,len(x)-profile.test_window_bars))
    rng=np.random.default_rng(seed); size=max(1,min(len(eligible)//30,50)) if len(eligible) else 0
    random_idx=sorted(rng.choice(eligible,size=size,replace=False).tolist()) if size else []
    atr_idx=x.index[(x['atr_ratio']<profile.atr_compression_threshold).fillna(False)].tolist()
    bb_idx=x.index[(x['bb_ratio']<profile.bb_squeeze_threshold).fillna(False)].tolist()
    if direction=='UP': rsi_mask=(x['rsi']<profile.rsi_oversold)
    else: rsi_mask=(x['rsi']>profile.rsi_overbought)
    rsi_idx=x.index[rsi_mask.fillna(False)].tolist()
    return {
      'random':_baseline_outcomes(x,direction,random_idx,profile.test_window_bars),
      'atr_only':_baseline_outcomes(x,direction,atr_idx,profile.test_window_bars),
      'bollinger_only':_baseline_outcomes(x,direction,bb_idx,profile.test_window_bars),
      'rsi_only':_baseline_outcomes(x,direction,rsi_idx,profile.test_window_bars),
    }


def ablation_test(candles:pd.DataFrame, profile):
    full=BacktestEngine(profile).run(candles)['metrics']
    out={'full':full}
    toggles={'without_rsi':'rsi','without_ict':'ict','without_bollinger':'bollinger','without_atr':'atr','without_tick_velocity':'tick_velocity','without_m15':'m15'}
    for label,flag in toggles.items():
        p=deepcopy(profile); p.feature_flags[flag]=False; out[label]=BacktestEngine(p).run(candles)['metrics']
    return out
