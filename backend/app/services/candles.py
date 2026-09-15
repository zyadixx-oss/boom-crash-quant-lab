import pandas as pd

TF_SECONDS={'M1':60,'M5':300,'M15':900}

def validate_ticks(df: pd.DataFrame) -> list[str]:
    errors=[]
    if df.empty: return ['empty_ticks']
    if df['epoch'].duplicated().any(): errors.append('duplicate_timestamps')
    if not df['epoch'].is_monotonic_increasing: errors.append('out_of_order')
    if (df['epoch'] > int(pd.Timestamp.now(tz='UTC').timestamp()) + 5).any(): errors.append('future_timestamp')
    return errors

def aggregate_ticks(ticks: pd.DataFrame, timeframe='M1') -> pd.DataFrame:
    if timeframe not in TF_SECONDS: raise ValueError('Unsupported timeframe')
    required={'epoch','price'}
    if not required.issubset(ticks.columns): raise ValueError('ticks require epoch and price')
    df=ticks[['epoch','price']].copy().drop_duplicates('epoch').sort_values('epoch')
    sec=TF_SECONDS[timeframe]
    df['bucket']=(df['epoch']//sec)*sec
    out=df.groupby('bucket')['price'].agg(open='first',high='max',low='min',close='last',tick_count='size').reset_index().rename(columns={'bucket':'epoch'})
    return out

def validate_ohlc(df:pd.DataFrame) -> list[str]:
    errors=[]
    if df.empty: return ['empty_candles']
    if ((df['open']>df['high']) | (df['close']>df['high']) | (df['low']>df['open']) | (df['low']>df['close'])).any(): errors.append('invalid_ohlc')
    if not df['epoch'].is_monotonic_increasing: errors.append('out_of_order')
    return errors
