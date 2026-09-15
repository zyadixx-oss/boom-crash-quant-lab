import asyncio, logging
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func
from .config import settings
from .db import init_db, SessionLocal
from .models import Signal, BacktestRun, ShadowSignal, OptimizationRun
from .schemas import BacktestRequest, OptimizeRequest, HealthResponse
from .symbol_profiles import get_profile, list_profiles
from .services.backtest import BacktestEngine
from .services.optimizer import grid_optimize
from .services.research import run_baselines, ablation_test
from .services.deriv_client import DerivClient

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')

@asynccontextmanager
async def lifespan(app:FastAPI):
    settings.assert_safe(); init_db(); yield

app=FastAPI(title='Boom Crash Quant Lab',version='0.2.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[settings.frontend_origin],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])

@app.get('/health',response_model=HealthResponse)
def health(): return HealthResponse()

@app.get('/symbols')
def symbols(): return list_profiles()

@app.get('/symbols/{symbol}')
def symbol_detail(symbol:str):
    try: return get_profile(symbol).__dict__
    except KeyError as e: raise HTTPException(404,str(e))

@app.get('/signals')
def signals(limit:int=100):
    with SessionLocal() as db:
        rows=db.execute(select(Signal).order_by(Signal.epoch.desc()).limit(min(limit,500))).scalars().all()
        return [{'id':r.id,'symbol':r.symbol,'epoch':r.epoch,'direction':r.direction,'score':r.score,'reason_codes':r.reason_codes} for r in rows]

@app.get('/signals/latest')
def latest_signal():
    rows=signals(1); return rows[0] if rows else None

@app.get('/signals/{symbol}')
def symbol_signals(symbol:str,limit:int=100):
    with SessionLocal() as db:
        rows=db.execute(select(Signal).where(Signal.symbol==symbol.upper()).order_by(Signal.epoch.desc()).limit(min(limit,500))).scalars().all()
        return [{'id':r.id,'symbol':r.symbol,'epoch':r.epoch,'direction':r.direction,'score':r.score,'reason_codes':r.reason_codes} for r in rows]

@app.get('/stats')
def stats():
    with SessionLocal() as db:
        return {'signals':db.scalar(select(func.count()).select_from(Signal)) or 0,'backtests':db.scalar(select(func.count()).select_from(BacktestRun)) or 0,'shadow_signals':db.scalar(select(func.count()).select_from(ShadowSignal)) or 0}

@app.post('/backtest')
def backtest(req:BacktestRequest):
    try: p=get_profile(req.symbol); df=pd.DataFrame(req.candles); result=BacktestEngine(p).run(df,req.min_signal_score)
    except Exception as e: raise HTTPException(400,str(e))
    with SessionLocal() as db:
        run=BacktestRun(symbol=p.symbol,status=result['classification'],params={'min_signal_score':req.min_signal_score or p.min_signal_score},metrics=result['metrics']); db.add(run); db.commit(); db.refresh(run)
    return {'id':run.id,**result}

@app.get('/backtest/{run_id}')
def get_backtest(run_id:int):
    with SessionLocal() as db:
        r=db.get(BacktestRun,run_id)
        if not r: raise HTTPException(404,'not found')
        return {'id':r.id,'symbol':r.symbol,'status':r.status,'params':r.params,'metrics':r.metrics}

@app.post('/optimize')
def optimize(req:OptimizeRequest):
    try:
        p=get_profile(req.symbol); result=grid_optimize(pd.DataFrame(req.candles),p,req.search_space)
    except Exception as e: raise HTTPException(400,str(e))
    with SessionLocal() as db:
        row=OptimizationRun(symbol=p.symbol,search_space=req.search_space or {},best_params=(result['best'] or {}).get('params',{}),validation_metrics=(result['best'] or {}).get('metrics',{})); db.add(row); db.commit()
    return result

@app.post('/research/{symbol}/baselines')
def baselines(symbol:str, candles:list[dict]):
    try: return run_baselines(pd.DataFrame(candles),get_profile(symbol))
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/research/{symbol}/ablation')
def ablation(symbol:str, candles:list[dict]):
    try: return ablation_test(pd.DataFrame(candles),get_profile(symbol))
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/leaderboard')
def leaderboard():
    with SessionLocal() as db:
        rows=db.execute(select(BacktestRun).order_by(BacktestRun.id.desc()).limit(100)).scalars().all()
        out=[{'run_id':r.id,'symbol':r.symbol,'strategy':'weighted-pre-spike-v1','win_rate':(r.metrics or {}).get('win_rate',0),'profit_factor':(r.metrics or {}).get('profit_factor',0),'expectancy':(r.metrics or {}).get('expectancy',0),'max_drawdown':(r.metrics or {}).get('max_drawdown',0),'status':r.status} for r in rows]
        return sorted(out,key=lambda x:(x['expectancy'],x['profit_factor']),reverse=True)

@app.get('/shadow/stats')
def shadow_stats(): return {'mode':'SHADOW_ONLY','opened_trades':False,**stats()}

@app.get('/deriv/active-symbols')
async def active_symbols():
    c=DerivClient()
    try: await c.connect(); return await c.active_symbols()
    except Exception as e: raise HTTPException(502,f'Deriv connection failed: {e}')
    finally: await c.close()

@app.get('/market/history/{symbol}')
async def market_history(symbol:str,count:int=300,timeframe:str='M1'):
    seconds={'M1':60,'M5':300,'M15':900}
    if timeframe not in seconds: raise HTTPException(400,'timeframe must be M1, M5, or M15')
    try: profile=get_profile(symbol)
    except KeyError as e: raise HTTPException(404,str(e))
    c=DerivClient()
    try:
        await c.connect(); api_symbol=await c.resolve_symbol(profile.display_name,profile.api_symbol)
        if not api_symbol: raise RuntimeError('symbol unavailable in active_symbols')
        return await c.candles(api_symbol,count,seconds[timeframe])
    except Exception as e: raise HTTPException(502,f'Deriv market history failed: {e}')
    finally: await c.close()

@app.websocket('/ws/signals')
async def ws_signals(ws:WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(latest_signal()); await asyncio.sleep(2)
    except WebSocketDisconnect: pass

@app.websocket('/ws/market')
async def ws_market(ws:WebSocket):
    await ws.accept(); requested=ws.query_params.get('symbol','BOOM500'); c=DerivClient()
    try:
        profile=get_profile(requested); await c.connect(); api_symbol=await c.resolve_symbol(profile.display_name,profile.api_symbol)
        if not api_symbol: raise RuntimeError('symbol unavailable in active_symbols')
        async for tick in c.subscribe_ticks(api_symbol): await ws.send_json({'symbol':profile.symbol,'api_symbol':api_symbol,'epoch':tick.get('epoch'),'price':tick.get('quote')})
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await ws.send_json({'error':str(e)})
        except Exception: pass
    finally: await c.close()
