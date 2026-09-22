from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    symbol: str
    candles: list[dict]
    min_signal_score: float | None = None


class OptimizeRequest(BaseModel):
    symbol: str
    candles: list[dict]
    search_space: dict[str, list[float]] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str = "SHADOW_ONLY"
    live_trading: bool = False
    ready_for_live: bool = False
    live_allowed: bool = False
    opened_trades: bool = False


class StrategyGenerateRequest(BaseModel):
    symbol: str
    agent_name: str | None = None
    variants_per_agent: int = Field(default=2, ge=1, le=3)


class ArenaBacktestRequest(BaseModel):
    strategy_id: str
    candles: list[dict]


class ArenaOOSRequest(BaseModel):
    strategy_id: str
    candles: list[dict]
    train_ratio: float | None = Field(default=None, gt=0.3, lt=0.8)
    validation_ratio: float | None = Field(default=None, gt=0.05, lt=0.4)


class ShadowStartRequest(BaseModel):
    strategy_id: str
    duration_hours: int = Field(default=24)


class SafetyProbeRequest(BaseModel):
    action: str = "order"
