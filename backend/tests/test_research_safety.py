import numpy as np, pandas as pd, pytest
from app.config import Settings
from app.symbol_profiles import get_profile
from app.services.research import run_baselines, ablation_test

def sample(n=180):
    rng=np.random.default_rng(13); close=100+np.cumsum(rng.normal(0,.05,n)); return pd.DataFrame({'epoch':np.arange(n)*60,'open':close,'high':close+.1,'low':close-.1,'close':close,'tick_count':20})

def test_research_baselines_and_ablation_execute():
    p=get_profile('BOOM300'); df=sample()
    assert set(run_baselines(df,p))=={'random','atr_only','bollinger_only','rsi_only'}
    out=ablation_test(df,p); assert 'full' in out and 'without_ict' in out and 'without_m15' in out

def test_safety_flags_cannot_be_enabled():
    with pytest.raises(RuntimeError): Settings(live_trading=True).assert_safe()
