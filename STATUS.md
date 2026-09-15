# Build status

Implemented in this delivery:

- FastAPI/SQLAlchemy/SQLite backend with 10 independent Boom/Crash profiles.
- Safety invariants hard-defaulted false; no live order route exists.
- Deriv public WebSocket client with heartbeat, reconnect-compatible client lifecycle, rate spacing, active-symbol resolution, history and tick subscription.
- Tick/candle validation and M1/M5/M15 aggregation.
- ATR, Bollinger, RSI, Stochastic, EMA, candle anatomy, volatility, tick-derived features.
- BOS, CHOCH, liquidity sweep, equal highs/lows, FVG and heuristic order blocks.
- Mathematical ATR-based spike labelling with pre/post-spike windows.
- Configurable 0-100 signal scoring.
- Chronological backtest, classification/trading metrics, walk-forward helper, grid optimization, baselines, ablation.
- Shadow collector with database persistence and delayed outcome evaluation.
- FastAPI REST endpoints and signal/market WebSockets.
- React/Vite/TypeScript/Tailwind dashboard source with real Deriv history chart, signals, backtest summary and symbol comparison tabs.
- README, architecture and scientific results/failure-analysis docs.

Verification performed:

- `pytest -q`: **11 passed**.
- Python compileall: passed.
- FastAPI local startup: passed.
- `/health`: returned SHADOW_ONLY with all four live flags false.
- `/symbols`: returned all 10 profiles.
- Synthetic-data backtest smoke path: executed; result classified NEEDS MORE DATA. Synthetic metrics are not evidence of edge.

Environment limitations:

- Real Deriv WebSocket request could not be executed because outbound DNS is blocked in this build environment: **NOT TESTED**.
- `npm install` could not reach the npm registry, so the frontend production build is **NOT TESTED** here. Source and package configuration are included.
