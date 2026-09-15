# Architecture

`Deriv WebSocket -> tick validation/storage -> M1/M5/M15 aggregation -> feature engine -> market-structure/ICT heuristics -> signal score -> shadow signal storage -> delayed outcome evaluation`.

Historical research follows `chronological split -> baselines -> weighted strategy -> ablation -> validation-only optimization -> untouched test -> walk-forward`. No random time-series shuffle is used.

## Safety boundary
There is intentionally no order-placement service, contract purchase endpoint, or live trade route. The public Deriv client is used only for `active_symbols`, historical candles, and tick subscriptions. Startup fails if any live-safety flag is enabled.

## Database
SQLite models: ticks, candles, signals, signal_results, backtest_runs, backtest_trades, symbol_profiles, optimization_runs, shadow_signals, system_events. Composite indexes protect tick/candle uniqueness and common time-series access paths.

## Signal semantics
The 0-100 score is a configurable rule score, not a calibrated probability. Each profile has independent seed settings. Any probability shown in a future version must come from measured calibration and out-of-sample validation.
