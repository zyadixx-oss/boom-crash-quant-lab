# boom-crash-quant-lab

Research-grade MVP for Deriv Boom/Crash indices. The project is **analysis/backtesting/shadow-only**. It does not place live trades.

## Safety invariants

These must remain false:

```env
LIVE_TRADING=false
READY_FOR_LIVE=false
LIVE_ALLOWED=false
OPENED_TRADES=false
```

There is no buy/sell execution endpoint.

## Stack

- Backend: FastAPI, SQLAlchemy/SQLite, pandas/numpy, Deriv public WebSocket API
- Frontend: React + Vite + TypeScript + TailwindCSS
- Live transport: WebSocket
- Deployment: Docker, Docker Compose, Vercel-ready frontend, Render Blueprint-ready backend

## Project layout

```text
backend/                 FastAPI application, research/backtest/shadow engines
frontend/                React/Vite dashboard
scripts/                 Data collection, backtest, optimization, shadow commands
data/                    Local runtime data (SQLite/CSV are gitignored)
docs/                    Architecture and results notes
Dockerfile               Backend container
frontend/Dockerfile      Production frontend container
frontend/vercel.json     Vercel frontend config
render.yaml              Render backend Blueprint
start.sh                 Simple launcher
Makefile                 Common local commands
docker-compose.yml       Local full-stack containers
```

## Local run without Docker

Requirements: Python 3.12+, Node 22+, npm.

```bash
git clone https://github.com/zyadixx-oss/boom-crash-quant-lab.git
cd boom-crash-quant-lab
make setup
```

Terminal 1:

```bash
make backend
```

Terminal 2:

```bash
make frontend
```

Open `http://localhost:5173`. API docs are at `http://localhost:8000/docs`.

Equivalent direct commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

```bash
cd frontend
npm install
npm run dev
```

## Local run with Docker

The easiest local full-stack startup is:

```bash
./start.sh docker
```

or:

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

Docker Compose stores the local SQLite database in the named volume `backend_data`.

## Frontend runtime configuration

The UI now supports separate frontend and backend hosts.

```env
VITE_API_BASE_URL=https://your-backend.example.com
VITE_WS_BASE_URL=wss://your-backend.example.com
```

For local Vite development these variables are optional because the Vite dev proxy handles `/api` and `/ws`.

## Recommended free preview deployment: Vercel + Render

This is intended for a demo/research preview, not production trading infrastructure.

### 1. Deploy the FastAPI backend on Render

The repository contains `render.yaml`.

1. Sign in to Render and connect GitHub.
2. Create a new **Blueprint** and select this repository.
3. Render reads `render.yaml` and creates `boom-crash-quant-lab-api` on the Free plan.
4. When prompted for `FRONTEND_ORIGIN`, you may initially use `http://localhost:5173`; update it after Vercel gives you the production URL.
5. Wait for `/health` to report healthy.
6. Copy the backend URL, for example `https://boom-crash-quant-lab-api.onrender.com`.

The start command is:

```bash
bash start.sh backend
```

### 2. Deploy the React/Vite frontend on Vercel

1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Vercel will use `frontend/vercel.json`.
4. Add production environment variables:

```env
VITE_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com
VITE_WS_BASE_URL=wss://YOUR-RENDER-SERVICE.onrender.com
```

5. Deploy and copy the resulting Vercel URL.

### 3. Finish CORS configuration on Render

Set the Render environment variable to the exact Vercel site URL:

```env
FRONTEND_ORIGIN=https://YOUR-PROJECT.vercel.app
```

Multiple origins can be comma-separated, for example:

```env
FRONTEND_ORIGIN=http://localhost:5173,https://YOUR-PROJECT.vercel.app
```

Redeploy the backend after changing the variable.

### 4. Verify

Check these in order:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
https://YOUR-RENDER-SERVICE.onrender.com/docs
https://YOUR-PROJECT.vercel.app
```

The dashboard should load symbol profiles, historical candles when Deriv is reachable, and a direct WebSocket connection to the Render backend.


## AI Strategy Arena

The Arena is an additive research layer over the existing project. On backend startup it creates the new Arena tables if missing, seeds the six Strategy Agents, and seeds two candidates per agent for the MVP symbols **CRASH500** and **BOOM500**. It does not automatically run heavy backtests.

Pipeline:

```text
Strategy Generation
  -> In-Sample Backtest
  -> Validation Qualification Gate
  -> Out-of-Sample Test
  -> Parameter Sensitivity / Robustness
  -> Forward / Shadow Test
  -> Leaderboard
  -> Qualified or Retired
```

The architecture supports all ten configured Boom/Crash symbols, but the first heavy evaluation scope is Crash 500 and Boom 500.

### Run the Arena locally

Start the backend and frontend as described above. The Arena UI is the default frontend dashboard. The backend automatically initializes the Arena tables and MVP candidates.

The Arena CLI is useful for local research:

```bash
python scripts/run_arena.py list
```

Use a chronological OHLC CSV with `epoch,open,high,low,close` columns.

Run train + validation backtest:

```bash
python scripts/run_arena.py backtest STRATEGY_ID data/your_m1_candles.csv
```

After the strategy passes the validation gate, run the untouched OOS + robustness stage:

```bash
python scripts/run_arena.py oos STRATEGY_ID data/your_m1_candles.csv
```

Only a strategy that passes OOS + robustness reaches `forward_testing`. Then run market-data-only shadow mode:

```bash
python scripts/run_arena.py shadow STRATEGY_ID --hours 24
python scripts/run_arena.py shadow STRATEGY_ID --hours 72
python scripts/run_arena.py shadow STRATEGY_ID --hours 168
```

These correspond to 24 hours, 3 days, and 7 days. Shadow mode records paper observations only; it never sends an order.

### Arena API

- `GET /api/arena/agents`
- `GET /api/arena/strategies`
- `POST /api/arena/strategies/generate`
- `GET /api/arena/strategies/{id}`
- `GET /api/arena/leaderboard`
- `POST /api/arena/backtest`
- `POST /api/arena/oos-test`
- `POST /api/arena/shadow/start`
- `POST /api/arena/shadow/stop?strategy_id=...`
- `GET /api/arena/shadow/status`
- `GET /api/arena/activity`
- `GET /api/arena/stats`
- `GET /api/arena/profiles`
- `GET /api/arena/safety`
- WebSocket `/ws/arena`

Qualification thresholds and composite-score weights are environment-configurable in `.env.example`. Net profit is displayed but is not a direct leaderboard weight.

## Free-hosting limitations

- Render Free web services can sleep after inactivity, so the first request can be slow.
- Render Free local files are ephemeral. The SQLite database can reset on restart/redeploy/spin-down. Use a persistent managed database for durable results.
- Vercel Hobby is appropriate for a personal/non-commercial frontend preview and has usage limits.
- This deployment does not change any trading-safety flags and does not enable real order execution.

## Alternative backend deployment

The root `Dockerfile` is portable to hosts that accept Docker images/repositories. Railway also auto-detects a root Dockerfile, but its pricing/free-credit model can change; Render is the documented zero-cost preview path for this repository.

## Verify current Deriv symbol IDs

Deriv symbol IDs should be queried rather than assumed:

```bash
python scripts/check_deriv_symbols.py
```

The client resolves symbols through `active_symbols` before history/shadow use.

## Generate smoke-test data and backtest

```bash
python scripts/generate_sample_data.py
python scripts/run_backtest.py BOOM500 data/sample_candles.csv
```

Synthetic results are software smoke tests only and must not be presented as trading performance.

## Real data collection

```bash
python scripts/collect_data.py BOOM500 5000
```

## Validation-only optimization

```bash
python scripts/run_optimize.py BOOM500 data/boom500_m1.csv
```

The script chronologically splits data and keeps the test partition reserved.

## Shadow mode

```bash
python scripts/run_shadow.py BOOM500
```

Shadow mode subscribes to public Deriv market data, stores observations, scores signals, and evaluates them later. It does not place orders.

## Tests

```bash
make test
```

or:

```bash
PYTHONPATH=backend pytest -q backend/tests tests
```

## API highlights

- `GET /health`
- `GET /symbols`
- `GET /signals`
- `GET /leaderboard`
- `GET /shadow/stats`
- `GET /deriv/active-symbols`
- `GET /market/history/{symbol}`
- `POST /backtest`
- `POST /optimize`
- WebSocket `/ws/signals`
- WebSocket `/ws/market`

## Scientific interpretation

Signal scores are weighted rule scores, **not probabilities**. A candidate is not promoted by win rate or net profit alone: qualification also evaluates trade count, profit factor, expectancy, drawdown, spike precision/recall, false-alert rate, OOS stability, and parameter sensitivity. No profitability claim is valid until supported by real out-of-sample and forward/shadow observations. See `docs/RESULTS.md` and `docs/ARCHITECTURE.md`.
