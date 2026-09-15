import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.symbol_profiles import get_profile
from app.services.backtest import chronological_split
from app.services.optimizer import grid_optimize
symbol=sys.argv[1] if len(sys.argv)>1 else 'BOOM500'; path=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'data'/f'{symbol.lower()}_m1.csv'
df=pd.read_csv(path); train,validation,test=chronological_split(df)
result=grid_optimize(validation,get_profile(symbol)); print('BEST VALIDATION ONLY:',result['best']); print('TEST DATA RESERVED:',len(test),'rows')
