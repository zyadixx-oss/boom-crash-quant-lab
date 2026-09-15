import pandas as pd

def label_spikes(df:pd.DataFrame, profile, direction:str)->pd.DataFrame:
    x=df.copy(); h=profile.spike_window
    future_close=x['close'].shift(-h)
    signed=(future_close-x['close']) if direction=='UP' else (x['close']-future_close)
    x['future_move_atr']=signed/x['atr'].replace(0,pd.NA)
    x['is_spike']=x['future_move_atr']>=profile.spike_atr_multiple
    x['pre_spike']=False
    spike_idx=x.index[x['is_spike'].fillna(False)]
    for i in spike_idx:
        x.loc[max(0,i-profile.pre_spike_window):i-1,'pre_spike']=True
    x['post_spike']=False
    for i in spike_idx:
        x.loc[i+1:min(len(x)-1,i+h),'post_spike']=True
    return x
