from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text, Index
from sqlalchemy.sql import func
from .db import Base

class Tick(Base):
    __tablename__='ticks'
    id=Column(Integer, primary_key=True); symbol=Column(String(32), nullable=False); epoch=Column(Integer, nullable=False); price=Column(Float, nullable=False)
    __table_args__=(Index('ix_ticks_symbol_epoch','symbol','epoch', unique=True),)

class Candle(Base):
    __tablename__='candles'
    id=Column(Integer, primary_key=True); symbol=Column(String(32), nullable=False); timeframe=Column(String(8), nullable=False); epoch=Column(Integer, nullable=False)
    open=Column(Float, nullable=False); high=Column(Float, nullable=False); low=Column(Float, nullable=False); close=Column(Float, nullable=False); tick_count=Column(Integer, default=0)
    __table_args__=(Index('ix_candles_symbol_tf_epoch','symbol','timeframe','epoch', unique=True),)

class Signal(Base):
    __tablename__='signals'
    id=Column(Integer, primary_key=True); symbol=Column(String(32), index=True, nullable=False); epoch=Column(Integer, index=True, nullable=False); direction=Column(String(8)); score=Column(Float)
    reason_codes=Column(JSON); features_snapshot=Column(JSON); m1_context=Column(JSON); m5_context=Column(JSON); m15_context=Column(JSON); entry_reference_price=Column(Float); expected_test_window=Column(Integer)

class SignalResult(Base):
    __tablename__='signal_results'
    id=Column(Integer, primary_key=True); signal_id=Column(Integer, index=True); outcome=Column(String(16)); return_pct=Column(Float); evaluated_epoch=Column(Integer)

class BacktestRun(Base):
    __tablename__='backtest_runs'
    id=Column(Integer, primary_key=True); symbol=Column(String(32)); status=Column(String(32)); params=Column(JSON); metrics=Column(JSON); created_at=Column(DateTime, server_default=func.now())

class BacktestTrade(Base):
    __tablename__='backtest_trades'
    id=Column(Integer, primary_key=True); run_id=Column(Integer, index=True); epoch=Column(Integer); direction=Column(String(8)); entry=Column(Float); exit=Column(Float); pnl=Column(Float); score=Column(Float)

class SymbolProfileModel(Base):
    __tablename__='symbol_profiles'
    id=Column(Integer, primary_key=True); symbol=Column(String(32), unique=True, index=True); config=Column(JSON, nullable=False); updated_at=Column(DateTime, server_default=func.now(), onupdate=func.now())

class OptimizationRun(Base):
    __tablename__='optimization_runs'
    id=Column(Integer, primary_key=True); symbol=Column(String(32)); search_space=Column(JSON); best_params=Column(JSON); validation_metrics=Column(JSON); created_at=Column(DateTime, server_default=func.now())

class ShadowSignal(Base):
    __tablename__='shadow_signals'
    id=Column(Integer, primary_key=True); symbol=Column(String(32), index=True); epoch=Column(Integer); price=Column(Float); score=Column(Float); reasons=Column(JSON); features=Column(JSON); result=Column(JSON)

class SystemEvent(Base):
    __tablename__='system_events'
    id=Column(Integer, primary_key=True); level=Column(String(16)); event_type=Column(String(64), index=True); message=Column(Text); payload=Column(JSON); created_at=Column(DateTime, server_default=func.now())
