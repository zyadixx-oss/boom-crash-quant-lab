import numpy as np
import pandas as pd

def add_market_structure(df:pd.DataFrame, swing=3, tolerance=0.0015)->pd.DataFrame:
    x=df.copy()
    win=2*swing+1
    x['swing_high']=x['high'].where(x['high'].eq(x['high'].rolling(win,center=True).max()))
    x['swing_low']=x['low'].where(x['low'].eq(x['low'].rolling(win,center=True).min()))
    prev_high=x['swing_high'].ffill().shift(1); prev_low=x['swing_low'].ffill().shift(1)
    x['bos_up']=x['close']>prev_high; x['bos_down']=x['close']<prev_low
    trend=np.where(x['bos_up'],1,np.where(x['bos_down'],-1,np.nan)); x['trend']=pd.Series(trend,index=x.index).ffill().fillna(0)
    x['choch']=((x['trend'].shift(1)==1)&x['bos_down'])|((x['trend'].shift(1)==-1)&x['bos_up'])
    rolling_high=x['high'].rolling(20,min_periods=5).max().shift(1); rolling_low=x['low'].rolling(20,min_periods=5).min().shift(1)
    x['liquidity_sweep_up']=(x['high']>rolling_high)&(x['close']<rolling_high)
    x['liquidity_sweep_down']=(x['low']<rolling_low)&(x['close']>rolling_low)
    x['support']=rolling_low; x['resistance']=rolling_high
    x['fvg_up']=x['low']>x['high'].shift(2); x['fvg_down']=x['high']<x['low'].shift(2)
    x['equal_highs']=((x['high']-x['high'].shift(1)).abs()/x['close']<tolerance)
    x['equal_lows']=((x['low']-x['low'].shift(1)).abs()/x['close']<tolerance)
    x['order_block_bull']=(x['close'].shift(1)<x['open'].shift(1)) & x['bos_up']
    x['order_block_bear']=(x['close'].shift(1)>x['open'].shift(1)) & x['bos_down']
    return x
