import numpy as np
import pandas as pd

def atr(df, period=14):
    prev=df['close'].shift(1)
    tr=pd.concat([(df['high']-df['low']).abs(),(df['high']-prev).abs(),(df['low']-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def rsi(close, period=14):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean(); al=loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs=ag/al.replace(0,np.nan); return 100-(100/(1+rs))

def bollinger(close, period=20, std_dev=2.0):
    mid=close.rolling(period).mean(); std=close.rolling(period).std(ddof=0); upper=mid+std_dev*std; lower=mid-std_dev*std
    width=(upper-lower)/mid.replace(0,np.nan)
    return mid,upper,lower,width

def stochastic(df, period=14):
    ll=df['low'].rolling(period).min(); hh=df['high'].rolling(period).max(); return 100*(df['close']-ll)/(hh-ll).replace(0,np.nan)

def add_features(df:pd.DataFrame, profile) -> pd.DataFrame:
    x=df.copy().sort_values('epoch').reset_index(drop=True)
    x['atr']=atr(x,profile.atr_period); x['atr_mean']=x['atr'].rolling(50,min_periods=20).mean(); x['atr_ratio']=x['atr']/x['atr_mean']
    mid,up,lo,width=bollinger(x['close'],profile.bb_period,profile.bb_std_dev); x['bb_mid']=mid; x['bb_upper']=up; x['bb_lower']=lo; x['bb_width']=width; x['bb_width_mean']=width.rolling(50,min_periods=20).mean(); x['bb_ratio']=width/x['bb_width_mean']
    x['rsi']=rsi(x['close'],profile.rsi_period); x['stoch']=stochastic(x,profile.stochastic_period); x['ema20']=x['close'].ewm(span=20,adjust=False).mean()
    x['body']=(x['close']-x['open']).abs(); x['range']=(x['high']-x['low']).abs(); x['upper_wick']=x['high']-x[['open','close']].max(axis=1); x['lower_wick']=x[['open','close']].min(axis=1)-x['low']
    x['rolling_vol']=x['close'].pct_change().rolling(20).std(); x['tick_velocity']=x['close'].diff()/x['atr'].replace(0,np.nan); x['tick_acceleration']=x['tick_velocity'].diff()
    if 'tick_count' in x: x['tick_frequency_z']=(x['tick_count']-x['tick_count'].rolling(30).mean())/x['tick_count'].rolling(30).std().replace(0,np.nan)
    else: x['tick_frequency_z']=np.nan
    return x
