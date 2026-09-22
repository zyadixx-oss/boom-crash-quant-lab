# AI Strategy Arena Architecture

The existing FastAPI + React/Vite/Tailwind + WebSocket stack is retained. Arena is an additive research layer rather than a rewrite.

## Data and safety boundary

`Deriv public WebSocket -> tick/candle data -> features -> Strategy Agents -> paper signal -> delayed observation`.

The Deriv client exposes market-data operations only: active symbols, historical candles, and tick subscriptions. There is no order-placement service, contract-purchase route, buy endpoint, or sell endpoint. Startup is fail-closed when any of these flags is true:

- `LIVE_TRADING`
- `READY_FOR_LIVE`
- `LIVE_ALLOWED`
- `OPENED_TRADES`

The UI safety panel reads the backend safety snapshot rather than hardcoded display state.

## Strategy factory

Six extensible Strategy Agents generate deterministic candidates:

1. ICT / Liquidity
2. Volatility Compression
3. Candle Structure
4. Support / Resistance
5. Multi-Timeframe
6. Ensemble

A candidate stores its ID, name, agent, symbol, version, parameters, rules, indicators, timeframes, train/validation period, and lifecycle status.

The default MVP seeds two variants from each agent for **Crash 500** and **Boom 500** only. Symbol Profiles support all ten requested Boom/Crash symbols, so later expansion does not require a schema change.

## Evaluation pipeline

`Generation -> In-sample Backtest -> Validation Qualification -> OOS -> Parameter Sensitivity / Robustness -> Forward/Shadow -> Leaderboard -> Qualified/Retired`.

Historical data uses chronological 60/20/20 train/validation/OOS partitions by default. No random time-series shuffle is used. OOS data is not used for candidate generation or sensitivity tuning.

The optimizer is deliberately bounded to at most three parameters and 64 combinations per experiment. Robustness perturbs a small parameter subset by ±5% and rejects candidates that collapse under small changes.

Market-structure pivots are causally confirmed; no centered rolling window is used for current-signal scoring.

## Metrics

Trading metrics include trade counts, win rate, net/gross P&L, profit factor, expectancy, drawdowns, average win/loss, risk/reward, consecutive losses, Sharpe-like, and stability.

Spike metrics include total/predicted/missed spikes, false spike signals, precision, recall, false-alert rate, lead time, MAE before spike, MFE after signal, and spike capture score.

Leaderboard scoring is configurable. Net profit is displayed but is not a direct score weight; the score combines bounded profit factor, expectancy quality, drawdown quality, spike precision/recall, false-alert quality, OOS quality, forward quality, and stability.

## Persistence

Existing tables remain intact. Arena adds:

`strategy_agents`, `strategy_candidates`, `oos_runs`, `forward_runs`, `strategy_metrics`, `spike_events`, `leaderboard`, and `strategy_status_history`.

SQLAlchemy `create_all` only creates missing tables, preserving existing SQLite rows.

## Real-time surfaces

REST endpoints live under `/api/arena/*`. WebSocket `/ws/arena` streams stats, activity, leaderboard updates, and safety status.

`real_24h_shadow_v2` accepts only candidates that already passed OOS + robustness. Supported durations are 24h, 72h (3 days), and 168h (7 days). It observes Deriv market data and records paper outcomes only.
