# boom-crash-quant-lab

Latest executed research: [Arabic consolidated results](docs/SPIKE_HUNTER_SUMMARY.ar.md).
No tested model meets net PF >= 1.5 with at least 1,000 completed held-out
simulations per model/symbol. The H4/H1 + M15/M5/M1 version is implemented and
tested on later historical periods; both linear300 and nonlinear600 versions
remain rejected. The [new tick execution report](docs/spike_tick_execution_20261005/REPORT.ar.md)
replays unchanged issuance on 24 predeclared public-tick days: 2,073,582 quotes,
3,083 independently checked paths and no qualified model. Two M5-reference PFs
above 1.5 have only 35 and 48 completed paths. The larger
[nonlinear600 study](docs/spike_nonlinear_20261005/REPORT.ar.md) remains the
full-period payoff evidence. The executed [tail-tick mechanism pilot](docs/spike_tick_tail_20261005/REPORT.ar.md)
now measures causal jump age on the same immutable tick sample. Last30% older/younger
hazard ratios are1.045 for Boom600 and1.226 for Crash600; neither is a profit factor.
The four declared >=1.5 overdue-hazard effects are rejected conditional on the
fixed detector and sparse observed-day resampling. No strategy is promoted.
All four live flags remain false.

## Live Arabic CRT chart

The `frontend` is now an Arabic, RTL standalone chart for all currently available Boom/Crash symbols. It uses the Deriv public WebSocket directly, without a trading account, API token or backend. Select M1/M5/M15/H1, inspect the closed H1 CRT range and closed M5 sweep/reclaim/confirmation observations, and review the frozen historical study. See `frontend/README.md` for precise rules and verification. All four trading flags remain false.

```bash
cd frontend
npm ci
npm test
npm run typecheck
npm run build
npm run dev
```

Node.js 22.18+ is required. The existing Python backend remains available for research APIs and offline studies.

## Offline Spike Hunter study (2026-10-04)

Frozen causal M5/H1 signals with real public Deriv M1 outcomes are implemented in
`scripts/run_spike_hunter_study.py`. See `docs/SPIKE_HUNTER_PROTOCOL.md` and
`docs/spike_hunter_20261004/REPORT.ar.md` for the actual 180-day BOOM500/CRASH500
study, all 12 outcome definitions, 70/30 split, three development validation
folds, block confidence intervals and limitations. This is excursion
classification; profitability and forward shadow performance remain NOT TESTED.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-research-lock.txt
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/collect_spike_history.py --days 180 \
  --symbols BOOM500 CRASH500 --cutoff-epoch 1791112080 --output-dir data/recollect
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_hunter_study.py \
  --dataset BOOM500=data/recollect/boom500_m1_180d_clean.csv \
  --dataset CRASH500=data/recollect/crash500_m1_180d_clean.csv \
  --output docs/reproduced-study --bootstrap 9999
```

The local deliverable includes the frozen source CSVs. GitHub contains code,
manifests and metrics; source CSVs are ignored to keep the repository small.
Historical server revisions can change recollected hashes. All four live flags
must remain false; the collector uses only the public data channel.

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

Requirements: Python 3.13+, Node 22+, npm.

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

## Frontend deployment

Build `frontend` with `npm ci && npm run build` and host `frontend/dist` on a static host. The current chart connects directly to the public Deriv endpoint and does not use `VITE_API_BASE_URL` or `VITE_WS_BASE_URL`. No backend CORS settings or account credentials are needed for the chart. A private Sites deployment is maintained separately from the offline Python research service.

## Optional backend deployment

The root Dockerfile, `render.yaml` and `start.sh backend` retain the research API deployment path. Verify `/health` and `/docs` independently of the chart. Persistent research storage and the historical study are backend concerns; a working live chart does not prove predictive advantage or profitability.

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

Signal scores are weighted rule scores, **not probabilities**. No profitability claim is valid until supported by real out-of-sample and shadow observations. See `docs/RESULTS.md`.

## Executed payoff study

The October 2026 extension tested 432 bracket/model configurations per symbol
on development data, froze selection, and evaluated BOOM500/CRASH500 on an
additional preceding 180-day sample. No configuration passed the development
eligibility requirements; all locked external research gates failed. The earlier
Crash resistance excursion lift did not establish profitable bracket returns.

See the [Arabic payoff report](docs/spike_payoff_20261004/REPORT.ar.md),
[interpretation and reproduction](docs/spike_payoff_20261004/README.md), and
[frozen protocol](docs/SPIKE_PAYOFF_PROTOCOL.md). These are public quote-path
simulations after hypothetical costs. Actual fills, monetary net profit and
forward performance remain **NOT TESTED**. All four trading flags stay false.

The [second round](docs/spike_timed_20261005/REPORT.ar.md) removes the profit cap
and tests 384 stop/time/direction/model configurations per500 symbol. No final
development candidate qualifies. Frozen500 rules transferred to1000 also fail:
primary CRT meanR=-0.103 (Boom1000,n505) and-0.102 (Crash1000,n484). This is
adaptive cross-symbol historical research, not forward validation. See its
[protocol](docs/SPIKE_TIMED_PROTOCOL.md) and
[reproduction/interpretation](docs/spike_timed_20261005/README.md).

The [learned-score round](docs/spike_learned_20261005/REPORT.ar.md) fits four
fixed causal ridge models, with no hyperparameter or exit search. No direction
qualifies in expanding validation. Frozen transfer to300 gives PF0.820(n8,354)
and0.854(n8,607) in spike direction; the opposite direction issues no signals.
The user target netPF>=1.5 and>=1,000 completed simulations per model/symbol is
not met. A source-only diagnostic shows severe input-distribution shift, without
refitting. All trading flags remain false. See the
[protocol](docs/SPIKE_LEARNED_PROTOCOL.md) and
[reproduction](docs/spike_learned_20261005/README.md).

The chronological [native300 M5 reference](docs/spike_native300_20261005/REPORT.ar.md)
and [combined H4/H1/M15/M5/M1 study](docs/spike_multiframe_20261005/REPORT.ar.md)
are now executed on the subsequent180-day300 period. Both selections froze
before its features/payoffs were read. The combined44-input models give
PF0.737(n460),0.824(n1,594),0.813(n1,036),0.812(n833); all rejected. JointHolm8
and a common-clock M5 comparison prevent timeframe/warmup differences from
being presented as an improvement. The user target remains unachieved.
See [protocol](docs/SPIKE_MULTIFRAME_PROTOCOL.md) and
[reproduction](docs/spike_multiframe_20261005/README.md). No trading is enabled.
