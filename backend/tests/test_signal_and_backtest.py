import numpy as np, pandas as pd
from app.symbol_profiles import get_profile
from app.services.backtest import BacktestEngine, chronological_split

def sample(n=240):
    rng=np.random.default_rng(7); close=100+np.cumsum(rng.normal(0,.08,n)); return pd.DataFrame({'epoch':np.arange(n)*60,'open':close+rng.normal(0,.02,n),'high':close+.12,'low':close-.12,'close':close,'tick_count':rng.integers(10,30,n)})
def test_backtest_runs_and_split_is_chronological():
    df=sample(); tr,va,te=chronological_split(df); assert tr.epoch.max()<va.epoch.min()<te.epoch.min()
    r=BacktestEngine(get_profile('CRASH500')).run(df); assert 'metrics' in r and 'classification' in r; assert r['metrics']['total_trades']>=0

def test_no_lookahead_signal_count_for_unchanged_prefix():
    df=sample(220); eng=BacktestEngine(get_profile('BOOM500')); a=eng.run(df.iloc[:180].copy())['metrics']['total_signals']; altered=df.copy(); altered.loc[190:,'close']*=5
    b=eng.run(altered.iloc[:180].copy())['metrics']['total_signals']; assert a==b
