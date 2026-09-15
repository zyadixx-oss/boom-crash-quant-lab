from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(42); n=3000; close=100+np.cumsum(rng.normal(0,.08,n)); df=pd.DataFrame({'epoch':np.arange(n)*60+1700000000,'open':close+rng.normal(0,.02,n),'high':close+abs(rng.normal(.1,.03,n)),'low':close-abs(rng.normal(.1,.03,n)),'close':close,'tick_count':rng.integers(10,35,n)})
out=Path(__file__).resolve().parents[1]/'data'/'sample_candles.csv'; df.to_csv(out,index=False); print(out)
