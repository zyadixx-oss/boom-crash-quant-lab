import pandas as pd
from app.services.candles import aggregate_ticks, validate_ohlc

def test_aggregation_and_ohlc():
    ticks=pd.DataFrame({'epoch':[0,10,59,60,61],'price':[1,2,1.5,3,2.5]}); c=aggregate_ticks(ticks,'M1')
    assert len(c)==2; assert c.iloc[0]['open']==1; assert c.iloc[0]['high']==2; assert c.iloc[0]['close']==1.5; assert validate_ohlc(c)==[]
