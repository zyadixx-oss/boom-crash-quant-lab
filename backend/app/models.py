from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text, Index
from sqlalchemy.sql import func
from .db import Base


class Tick(Base):
    __tablename__ = "ticks"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False)
    epoch = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    __table_args__ = (Index("ix_ticks_symbol_epoch", "symbol", "epoch", unique=True),)


class Candle(Base):
    __tablename__ = "candles"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False)
    timeframe = Column(String(8), nullable=False)
    epoch = Column(Integer, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    tick_count = Column(Integer, default=0)
    __table_args__ = (Index("ix_candles_symbol_tf_epoch", "symbol", "timeframe", "epoch", unique=True),)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), index=True, nullable=False)
    epoch = Column(Integer, index=True, nullable=False)
    direction = Column(String(8))
    score = Column(Float)
    reason_codes = Column(JSON)
    features_snapshot = Column(JSON)
    m1_context = Column(JSON)
    m5_context = Column(JSON)
    m15_context = Column(JSON)
    entry_reference_price = Column(Float)
    expected_test_window = Column(Integer)


class SignalResult(Base):
    __tablename__ = "signal_results"
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, index=True)
    outcome = Column(String(16))
    return_pct = Column(Float)
    evaluated_epoch = Column(Integer)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32))
    status = Column(String(32))
    params = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, index=True)
    epoch = Column(Integer)
    direction = Column(String(8))
    entry = Column(Float)
    exit = Column(Float)
    pnl = Column(Float)
    score = Column(Float)


class SymbolProfileModel(Base):
    __tablename__ = "symbol_profiles"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), unique=True, index=True)
    config = Column(JSON, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32))
    search_space = Column(JSON)
    best_params = Column(JSON)
    validation_metrics = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


class ShadowSignal(Base):
    __tablename__ = "shadow_signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), index=True)
    epoch = Column(Integer)
    price = Column(Float)
    score = Column(Float)
    reasons = Column(JSON)
    features = Column(JSON)
    result = Column(JSON)


class SystemEvent(Base):
    __tablename__ = "system_events"
    id = Column(Integer, primary_key=True)
    level = Column(String(16))
    event_type = Column(String(64), index=True)
    message = Column(Text)
    payload = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


# --- AI Strategy Arena tables ---
# These are additive only so existing SQLite data remains untouched.


class StrategyAgent(Base):
    __tablename__ = "strategy_agents"
    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=False)
    capabilities = Column(JSON, nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class StrategyCandidate(Base):
    __tablename__ = "strategy_candidates"
    id = Column(String(64), primary_key=True)
    name = Column(String(160), nullable=False)
    agent_name = Column(String(64), index=True, nullable=False)
    symbol = Column(String(32), index=True, nullable=False)
    version = Column(String(32), nullable=False)
    parameters = Column(JSON, nullable=False)
    rules = Column(JSON, nullable=False)
    indicators_used = Column(JSON, nullable=False)
    timeframes = Column(JSON, nullable=False)
    training_period = Column(JSON)
    validation_period = Column(JSON)
    status = Column(String(32), index=True, nullable=False, default="created")
    created_at = Column(DateTime, server_default=func.now())


class OOSRun(Base):
    __tablename__ = "oos_runs"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(64), index=True, nullable=False)
    split_config = Column(JSON, nullable=False)
    metrics = Column(JSON)
    spike_metrics = Column(JSON)
    status = Column(String(32), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ForwardRun(Base):
    __tablename__ = "forward_runs"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(64), index=True, nullable=False)
    mode = Column(String(64), nullable=False, default="real_24h_shadow_v2")
    duration_hours = Column(Integer, nullable=False, default=24)
    status = Column(String(32), nullable=False, default="created")
    metrics = Column(JSON)
    last_error = Column(Text)
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime)


class StrategyMetric(Base):
    __tablename__ = "strategy_metrics"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(64), index=True, nullable=False)
    stage = Column(String(32), index=True, nullable=False)
    metrics = Column(JSON, nullable=False)
    spike_metrics = Column(JSON, nullable=False)
    equity_curve = Column(JSON)
    drawdown_curve = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())


class SpikeEvent(Base):
    __tablename__ = "spike_events"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), index=True, nullable=False)
    strategy_id = Column(String(64), index=True)
    epoch = Column(Integer, index=True, nullable=False)
    side = Column(String(8), nullable=False)
    magnitude_atr = Column(Float)
    detected_by_signal = Column(Boolean, default=False)
    signal_epoch = Column(Integer)
    lead_time_seconds = Column(Integer)
    mae_before_spike = Column(Float)
    mfe_after_signal = Column(Float)


class LeaderboardEntry(Base):
    __tablename__ = "leaderboard"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(64), unique=True, index=True, nullable=False)
    symbol = Column(String(32), index=True, nullable=False)
    composite_score = Column(Float, nullable=False)
    components = Column(JSON, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class StrategyStatusHistory(Base):
    __tablename__ = "strategy_status_history"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(64), index=True, nullable=False)
    from_status = Column(String(32))
    to_status = Column(String(32), nullable=False)
    reason = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
