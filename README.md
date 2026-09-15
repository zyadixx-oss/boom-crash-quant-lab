# boom-crash-quant-lab

Research-grade MVP for Deriv Boom/Crash indices. It is **analysis/backtesting/shadow-only**. It cannot place live trades and all live flags are hard-defaulted to false.

## Safety
`LIVE_TRADING=false`, `ready_for_live=false`, `live_allowed=false`, `opened_trades=false`. No buy/sell execution endpoint exists.

## Architecture
- `backend/`: FastAPI, SQLAlchemy/SQLite, Deriv public WebSocket client, feature/ICT/spike/signal/backtest/optimization engines.
- `frontend/`: React + Vite + TypeScript + Tailwind dashboard.
- `scripts/`: sample-data smoke test, CLI backtest, active-symbol verification.
- `docs/RESULTS.md`: scientific results template and failure analysis.

## Supported profiles
Boom and Crash 300, 500, 600, 900, 1000. Every symbol has an independent profile. Seed parameters are starting hypotheses only, not performance claims.

## Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```
API: `http://localhost:8000`, docs: `/docs`.

## Frontend
```bash
cd frontend
npm install
npm run dev
```

## Verify current Deriv symbol IDs
Deriv added new Crash/Boom ranges in 2026. Query `active_symbols` rather than trusting hard-coded names:
```bash
python scripts/check_deriv_symbols.py
```
Then update `api_symbol` in `backend/app/symbol_profiles.py` if needed.

## Generate smoke-test data and backtest
```bash
python scripts/generate_sample_data.py
python scripts/run_backtest.py BOOM500 data/sample_candles.csv
```
Synthetic sample results are only software smoke tests and must never be reported as trading performance.

## Tests
```bash
cd backend
pytest -q
```

## API
`GET /health`, `/symbols`, `/symbols/{symbol}`, `/signals`, `/signals/latest`, `/signals/{symbol}`, `/stats`, `/leaderboard`, `/shadow/stats`, `/deriv/active-symbols`; `POST /backtest`, `/optimize`; WebSocket `/ws/signals`, `/ws/market`.

## Data validation and bias controls
Ticks are sorted/deduplicated before aggregation; OHLC validation is provided. Backtesting is chronological, contains a test for prefix invariance, uses no random time-series shuffle, and the optimizer operates on the dataset explicitly passed as validation data. Walk-forward threshold selection is validation-only and evaluates the next test window.

## Scientific interpretation
Signal score is a weighted rule score, **not a probability**. ICT features can be disabled. No claim of profitability is valid until real out-of-sample and shadow data support it. See `docs/RESULTS.md`.

## Real data collection
```bash
python scripts/collect_data.py BOOM500 5000
```
The script resolves the current Deriv API symbol from `active_symbols` and stores real M1 candles under `data/`.

## Validation-only optimization
```bash
python scripts/run_optimize.py BOOM500 data/boom500_m1.csv
```
The script performs a chronological split and passes only validation data to the grid optimizer; the test partition remains reserved.

## Shadow mode
```bash
python scripts/run_shadow.py BOOM500
```
This subscribes to the public Deriv data feed, stores ticks and completed M1/M5/M15 candles, records scored signals, and evaluates them after the configured test window. It does not place orders.
