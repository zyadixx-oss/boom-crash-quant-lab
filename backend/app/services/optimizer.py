from itertools import product
from copy import deepcopy
from .backtest import BacktestEngine

def grid_optimize(validation_df, profile, search_space=None):
    space=search_space or {'min_signal_score':[55,60,65,70],'spike_atr_multiple':[2.0,2.5,3.0]}
    keys=list(space); best=None; results=[]
    for vals in product(*[space[k] for k in keys]):
        p=deepcopy(profile)
        for k,v in zip(keys,vals): setattr(p,k,v)
        m=BacktestEngine(p).run(validation_df)['metrics']; score=m['expectancy'] - 0.1*m['max_drawdown']
        row={'params':dict(zip(keys,vals)),'score':score,'metrics':m}; results.append(row)
        if best is None or score>best['score']: best=row
    return {'best':best,'results':sorted(results,key=lambda r:r['score'],reverse=True)}
