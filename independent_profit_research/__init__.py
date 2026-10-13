"""Independent conditional execution arithmetic; no data access or trading APIs."""

from .execution import ExecutionConfig, Quote, TradeOutcome, simulate_trade

__all__ = ["Quote", "ExecutionConfig", "TradeOutcome", "simulate_trade"]
