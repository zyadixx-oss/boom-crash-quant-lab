import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.symbol_profiles import get_profile
from app.services.backtest import BacktestEngine
symbol=sys.argv[1] if len(sys.argv)>1 else 'BOOM500'; path=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'data'/'sample_candles.csv'
if not path.exists(): raise SystemExit(f'Missing {path}; run scripts/generate_sample_data.py')
r=BacktestEngine(get_profile(symbol)).run(pd.read_csv(path)); print(r['classification']); print(r['metrics'])
