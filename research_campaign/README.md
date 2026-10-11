# Single-frame Boom/Crash research campaign

The 11 October request is preserved in
[REQUEST.ar.md](../docs/single_frame_campaign_20261011/REQUEST.ar.md).
The preimplementation checklist is
[ACCEPTANCE.md](../docs/single_frame_campaign_20261011/ACCEPTANCE.md).

This package searches ten actual public Deriv indices across ten independent
frames. Each strategy consumes one frame. It does not reuse the old multiframe
study as new evidence and cannot send account or order requests.

## Run on Ubuntu or macOS

From the repository root, with Python 3.12+:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-research-lock.txt
export LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false
.venv/bin/python -m research_campaign.data \
  --output-dir data/single_frame_campaign_20261011/history --days 365
```

Use the immutable `manifest_file` printed/recorded by acquisition as the
initial manifest. Preserve failed attempts and all raw wire files.

```bash
.venv/bin/python -m research_campaign run --initial-manifest /absolute/path/to/history/runs/MANIFEST.json
.venv/bin/python -m research_campaign status
.venv/bin/python -m research_campaign verify
.venv/bin/python -m research_campaign replay
```

The foreground command performs one actual cycle immediately, then waits for
the next 15-minute slot. Thirty-two starts span 7h45m; the final search window
is eight hours. Final evaluation begins only after all 32 cycles actually
complete. Overruns and outage recovery retain actual timestamps; they are never
reported as on-time execution or fabricated completed cycles.

To resume, run the same `run` command without an initial manifest. A kernel
lock prevents two workers. Each committed result is cached by source, scientific
parameters and split; equivalent parameters do not trigger another measurement.
`run --once` performs at most one due cycle. It does not accelerate the cadence.
Source or numerical runtime changes fail closed rather than reinterpret a
frozen campaign.

## Research design

- Fixed candidate catalogue: thirteen families, thirty-two declared parameter
  combinations. Settings are ranked independently for each symbol/frame/family
  using validation return/drawdown and a sample penalty.
- Each symbol's reliable observed endpoints determine its 60/20/20 calendar
  before outcomes are measured. Its ten frames share that calendar. Requested
  coverage, retention boundaries, exclusions and internal gaps remain recorded.
- Search sees training and validation only. Three expanding development folds
  are evaluated for each selected policy. These are selection-dependent
  development stability checks, not independent out-of-sample estimates.
  Selection is saved and hashed before
  opening final data; no final-based parameter changes.
- All selected final hypotheses enter one Holm family. Weekly block bootstrap
  uses 99,999 draws, sufficient p-value resolution for the planned family.
- Fixed 0.5% nominal equity risk, next-open entry, fixed ATR stop/target,
  conservative ambiguous-bar ordering, uncapped stop gaps, one position.
  Trailing stops are prohibited.
- Every result retains virtual trades/dispositions, equity, monthly/reference
  comparisons, settings, dates, source hashes and rejection reasons.

## Limits and qualification

The default 0.10 ATR round-trip cost is an **assumption scenario**, not measured
Deriv spread, slippage or fees. Public options catalogue fields have not proved
CFD mapping. Unverified contract costs or original execution keep qualification
false regardless of PF. The cash amounts are nominal accounting units, not
verified broker profits.

Final acceptance additionally requires at least 100 completions, PF at least
1.10, drawdown at most 20%, positive net return, outperformance of both original
costed buy-and-hold and a native-direction reference, source/causality validity,
weekly uncertainty/multiplicity and development stability. The buy-and-hold
model's CFD comparability remains explicitly unverified; its hurdle is retained.

Historical prices exposed in prior studies remain adaptive evidence. Newly
cached live updates never move the frozen split or rescue selected settings.
Original tick replay is supported by the engine but a candle-only run does not
claim it occurred. No strategy is promoted from assumed fills/costs.

## Dashboard and exports

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/#research`. The original CRT chart remains at
`/#chart`. The runner writes `frontend/public/campaign/latest.json`, lazy
selected-ledger details, and `search.csv`. Full compressed result artifacts,
all settings and cycle journals remain in the state directory. The dashboard
labels development and final results separately; no final metrics appear early.

## Verification

```bash
.venv/bin/python -m pytest -o addopts= -q research_campaign/tests
.venv/bin/python -m compileall -q research_campaign
cd frontend
npm test
npm run typecheck
npm run lint
npm run build
```

Unit fixtures are synthetic solely to reveal accounting, chronology and resume
errors; they are never product performance data. The actual completed-cycle
count comes from dated terminal cycle records. A live worker is established
by an actual held kernel lock and PID, not merely a state file.

`verify` checks recorded immutable artifact hashes. `replay` actually rebuilds
signals and reruns the frozen accounting from the pinned original price files,
then compares the complete ledgers, metrics, references, equity and signal
fingerprints. It saves a dated report in `data/single_frame_campaign_20261011/replays/`.
`replay --replay-limit 10` explicitly reports partial coverage. Both commands can
inspect committed measurements while the research worker waits for its next
cycle. Numerical replay uses the same production formulas; the separate source
audit independently reconstructs the original wire history.

After the authoritative state is `COMPLETED`, the final-family helper can
recompute every selected original final measurement and the three saved
development folds, then recalculate joint Holm and final acceptance gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/replay_campaign_final.py --include-walk-forward
```

The same command resumes from verified receipts in
`work/final_replay_20261011/`, reporting earlier verified replays separately
from computations performed now. It rejects an unfinished or sealed campaign
before reading final artifacts. `--max-new N` explicitly bounds new work and
reports partial coverage. The helper does not replay the entire development
search, independently implement the formulas, or verify CFD fills/costs.
It reads the compressed final family once, releases its large ledgers before
reconstruction, and retains only small acceptance summaries; bounded peak
memory is not claimed.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false` always. Lowercase display equivalents also remain false.
