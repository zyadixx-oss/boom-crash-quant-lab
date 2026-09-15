import pandas as pd
from app.services.market_structure import add_market_structure

def test_no_future_columns_required_for_bos():
    df=pd.DataFrame({'epoch':range(20),'open':[10]*20,'high':[10+i*.1 for i in range(20)],'low':[9+i*.05 for i in range(20)],'close':[9.5+i*.09 for i in range(20)]})
    x=add_market_structure(df,swing=2); assert {'bos_up','bos_down','choch','fvg_up','liquidity_sweep_up'}.issubset(x.columns)
