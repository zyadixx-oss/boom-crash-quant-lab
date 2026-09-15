import numpy as np, pandas as pd
from app.symbol_profiles import get_profile
from app.services.indicators import add_features

def candles(n=120):
    c=100+np.cumsum(np.sin(np.arange(n)/7)*.05+.01); return pd.DataFrame({'epoch':np.arange(n)*60,'open':c-.02,'high':c+.1,'low':c-.1,'close':c,'tick_count':20})
def test_features_exist():
    x=add_features(candles(),get_profile('BOOM500'))
    for c in ['atr','bb_ratio','rsi','stoch','tick_velocity']: assert c in x.columns
    assert x['atr'].dropna().gt(0).all()
