import asyncio
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from .config import settings
from .db import SessionLocal, init_db
from .models import BacktestRun, OptimizationRun, ShadowSignal, Signal
from .schemas import (
    ArenaBacktestRequest,
    ArenaOOSRequest,
    BacktestRequest,
    HealthResponse,
    OptimizeRequest,
    ShadowStartRequest,
    StrategyGenerateRequest,
)
from .services.arena.pipeline import ArenaPipeline, bootstrap_arena
from .services.arena.shadow import shadow_manager
from .services.backtest import BacktestEngine
from .services.deriv_client import DerivClient
from .services.optimizer import grid_optimize
from .services.research import ablation_test, run_baselines
from .services.safety import safety_snapshot
from .symbol_profiles import get_profile, list_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.assert_safe()
    init_db()
    bootstrap_arena()
    yield
    for task in list(shadow_manager.tasks.values()):
        task.cancel()


app = FastAPI(title="Boom Crash AI Strategy Arena", version="0.3.0", lifespan=lifespan)
frontend_origins = [origin.strip() for origin in settings.frontend_origin.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    settings.assert_safe()
    return HealthResponse()


@app.get("/symbols")
def symbols():
    return list_profiles()


@app.get("/symbols/{symbol}")
def symbol_detail(symbol: str):
    try:
        return get_profile(symbol).__dict__
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/signals")
def signals(limit: int = 100):
    with SessionLocal() as db:
        rows = db.execute(select(Signal).order_by(Signal.epoch.desc()).limit(min(limit, 500))).scalars().all()
        return [
            {
                "id": row.id,
                "symbol": row.symbol,
                "epoch": row.epoch,
                "direction": row.direction,
                "score": row.score,
                "reason_codes": row.reason_codes,
            }
            for row in rows
        ]


@app.get("/signals/latest")
def latest_signal():
    rows = signals(1)
    return rows[0] if rows else None


@app.get("/signals/{symbol}")
def symbol_signals(symbol: str, limit: int = 100):
    with SessionLocal() as db:
        rows = db.execute(
            select(Signal)
            .where(Signal.symbol == symbol.upper())
            .order_by(Signal.epoch.desc())
            .limit(min(limit, 500))
        ).scalars().all()
        return [
            {
                "id": row.id,
                "symbol": row.symbol,
                "epoch": row.epoch,
                "direction": row.direction,
                "score": row.score,
                "reason_codes": row.reason_codes,
            }
            for row in rows
        ]


@app.get("/stats")
def stats():
    with SessionLocal() as db:
        return {
            "signals": db.scalar(select(func.count()).select_from(Signal)) or 0,
            "backtests": db.scalar(select(func.count()).select_from(BacktestRun)) or 0,
            "shadow_signals": db.scalar(select(func.count()).select_from(ShadowSignal)) or 0,
        }


@app.post("/backtest")
def backtest(req: BacktestRequest):
    try:
        profile = get_profile(req.symbol)
        df = pd.DataFrame(req.candles)
        result = BacktestEngine(profile).run(df, req.min_signal_score)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    with SessionLocal() as db:
        run = BacktestRun(
            symbol=profile.symbol,
            status=result["classification"],
            params={"min_signal_score": req.min_signal_score or profile.min_signal_score},
            metrics=result["metrics"],
        )
        db.add(run)
        db.commit()
        db.refresh(run)
    return {"id": run.id, **result}


@app.get("/backtest/{run_id}")
def get_backtest(run_id: int):
    with SessionLocal() as db:
        row = db.get(BacktestRun, run_id)
        if not row:
            raise HTTPException(404, "not found")
        return {"id": row.id, "symbol": row.symbol, "status": row.status, "params": row.params, "metrics": row.metrics}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        profile = get_profile(req.symbol)
        result = grid_optimize(pd.DataFrame(req.candles), profile, req.search_space)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    with SessionLocal() as db:
        row = OptimizationRun(
            symbol=profile.symbol,
            search_space=req.search_space or {},
            best_params=(result["best"] or {}).get("params", {}),
            validation_metrics=(result["best"] or {}).get("metrics", {}),
        )
        db.add(row)
        db.commit()
    return result


@app.post("/research/{symbol}/baselines")
def baselines(symbol: str, candles: list[dict]):
    try:
        return run_baselines(pd.DataFrame(candles), get_profile(symbol))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/research/{symbol}/ablation")
def ablation(symbol: str, candles: list[dict]):
    try:
        return ablation_test(pd.DataFrame(candles), get_profile(symbol))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/leaderboard")
def legacy_leaderboard():
    with SessionLocal() as db:
        rows = db.execute(select(BacktestRun).order_by(BacktestRun.id.desc()).limit(100)).scalars().all()
        out = [
            {
                "run_id": row.id,
                "symbol": row.symbol,
                "strategy": "weighted-pre-spike-v1",
                "win_rate": (row.metrics or {}).get("win_rate", 0),
                "profit_factor": (row.metrics or {}).get("profit_factor", 0),
                "expectancy": (row.metrics or {}).get("expectancy", 0),
                "max_drawdown": (row.metrics or {}).get("max_drawdown", 0),
                "status": row.status,
            }
            for row in rows
        ]
        return sorted(out, key=lambda item: (item["expectancy"], item["profit_factor"]), reverse=True)


@app.get("/shadow/stats")
def shadow_stats():
    return {"mode": "SHADOW_ONLY", "opened_trades": False, **stats()}


@app.get("/deriv/active-symbols")
async def active_symbols():
    client = DerivClient()
    try:
        await client.connect()
        return await client.active_symbols()
    except Exception as exc:
        raise HTTPException(502, f"Deriv connection failed: {exc}")
    finally:
        await client.close()


@app.get("/market/history/{symbol}")
async def market_history(symbol: str, count: int = 300, timeframe: str = "M1"):
    seconds = {"M1": 60, "M5": 300, "M15": 900}
    if timeframe not in seconds:
        raise HTTPException(400, "timeframe must be M1, M5, or M15")
    try:
        profile = get_profile(symbol)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    client = DerivClient()
    try:
        await client.connect()
        api_symbol = await client.resolve_symbol(profile.display_name, profile.api_symbol)
        if not api_symbol:
            raise RuntimeError("symbol unavailable in active_symbols")
        return await client.candles(api_symbol, count, seconds[timeframe])
    except Exception as exc:
        raise HTTPException(502, f"Deriv market history failed: {exc}")
    finally:
        await client.close()


# ----------------------------
# AI Strategy Arena REST API
# ----------------------------


@app.get("/api/arena/agents")
def arena_agents():
    with SessionLocal() as db:
        return ArenaPipeline.agents(db)


@app.post("/api/arena/strategies/generate")
def arena_generate(req: StrategyGenerateRequest):
    try:
        with SessionLocal() as db:
            rows = ArenaPipeline.generate(db, req.symbol, req.agent_name, req.variants_per_agent)
            db.commit()
            return rows
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/arena/strategies")
def arena_strategies(
    symbol: str | None = None,
    agent: str | None = None,
    status: str | None = None,
    timeframe: str | None = None,
):
    try:
        with SessionLocal() as db:
            rows = ArenaPipeline.list_strategies(db, symbol=symbol, agent=agent, status=status)
            if timeframe:
                rows = [row for row in rows if timeframe in (row.get("timeframes") or [])]
            return rows
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/arena/strategies/{strategy_id}")
def arena_strategy_detail(strategy_id: str):
    try:
        with SessionLocal() as db:
            return ArenaPipeline.detail(db, strategy_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/arena/leaderboard")
def arena_leaderboard(
    symbol: str | None = None,
    agent: str | None = None,
    status: str | None = None,
):
    try:
        with SessionLocal() as db:
            return ArenaPipeline.leaderboard(db, symbol=symbol, agent=agent, status=status)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/arena/backtest")
def arena_backtest(req: ArenaBacktestRequest):
    settings.assert_safe()
    try:
        with SessionLocal() as db:
            return ArenaPipeline.backtest(db, req.strategy_id, req.candles)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/arena/oos-test")
def arena_oos_test(req: ArenaOOSRequest):
    settings.assert_safe()
    try:
        with SessionLocal() as db:
            return ArenaPipeline.oos_test(db, req.strategy_id, req.candles, req.train_ratio, req.validation_ratio)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/arena/shadow/start")
async def arena_shadow_start(req: ShadowStartRequest):
    settings.assert_safe()
    try:
        return await shadow_manager.start(req.strategy_id, req.duration_hours)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/arena/shadow/stop")
async def arena_shadow_stop(strategy_id: str = Query(...)):
    settings.assert_safe()
    try:
        return await shadow_manager.stop(strategy_id)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/arena/shadow/status")
def arena_shadow_status(strategy_id: str | None = None):
    return shadow_manager.status(strategy_id)


@app.get("/api/arena/activity")
def arena_activity(limit: int = 100):
    with SessionLocal() as db:
        return ArenaPipeline.activity(db, limit)


@app.get("/api/arena/stats")
def arena_stats():
    with SessionLocal() as db:
        return ArenaPipeline.stats(db)


@app.get("/api/arena/profiles")
def arena_profiles():
    return list_profiles()


@app.get("/api/arena/safety")
def arena_safety():
    return safety_snapshot()


@app.websocket("/ws/signals")
async def ws_signals(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(latest_signal())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/market")
async def ws_market(ws: WebSocket):
    await ws.accept()
    requested = ws.query_params.get("symbol", "BOOM500")
    client = DerivClient()
    try:
        profile = get_profile(requested)
        await client.connect()
        api_symbol = await client.resolve_symbol(profile.display_name, profile.api_symbol)
        if not api_symbol:
            raise RuntimeError("symbol unavailable in active_symbols")
        async for tick in client.subscribe_ticks(api_symbol):
            await ws.send_json(
                {
                    "symbol": profile.symbol,
                    "api_symbol": api_symbol,
                    "epoch": tick.get("epoch"),
                    "price": tick.get("quote"),
                }
            )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        await client.close()


@app.websocket("/ws/arena")
async def ws_arena(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with SessionLocal() as db:
                payload = {
                    "type": "arena_update",
                    "stats": ArenaPipeline.stats(db),
                    "activity": ArenaPipeline.activity(db, 20),
                    "leaderboard": ArenaPipeline.leaderboard(db)["rows"][:25],
                    "safety": safety_snapshot(),
                }
            await ws.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
