# Executed round10 tick-path risk pilot

See [Arabic report](REPORT.ar.md), [results](results.json), [metrics](metrics.csv),
[frozen declaration](declaration.json), [selection](selection.json), and
[protocol](../SPIKE_TICK_PATH_RISK_PROTOCOL.md). The protocol retains its
pre-execution wording because its bytes were frozen before measurement.

RIDGE44 uses closed H4/H1 + M15/M5/M1 inputs. RIDGE45 adds exactly one
strict-prior native-adverse semivariance over 600 one-second increments,
normalized by the immutable round8 calibration median squared. Its 601 quotes
must end at t-1. A missing second invalidates the window until a new complete
window exists. Native direction is unchanged between SPIKE and DRIFT modes.
Both models and CLOCK share availability and closed M5 opportunities.

All development selections are ineligible. Final RIDGE45 point PFs are
0.758/0.761/0.763/1.033 with 105/132/121/83 completed one-open simulations.
All four 45-minus44 paired mean intervals cross zero. No declared expansion
condition passes. Four observed final days and these sample sizes cannot
qualify PF>=1.5 with 1,000 completed held-out paths and 60 active held-out days
per model/symbol. The 2,329 training targets per direction overlap and cannot
be counted toward that evidence requirement.

This is adaptive research on the same 12 discontinuous tick days per symbol,
with prices AND payoffs already known from prior research. New model freezing
before new feature/score/ledger calculations does not create fresh price OOS
or a new label holdout. Conditional observed-day bootstrap intervals do not
prove prospective inference or independence. All four live flags remain false.

Full reproduction needs the exact ignored local quotes, matrices, targets,
signals and ledgers plus the pinned ancestor artifacts. Git includes source,
protocols, tests, declarations, model metadata, results, logs and audit evidence.
A fresh checkout can run synthetic tests but cannot reconstruct the actual
historical counts without those source bytes. Public recollection is a new
dataset if its bytes differ. Use a separate empty output directory:

```bash
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_path_risk.py --stage declare --output docs/reproduced-path-risk
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_path_risk.py --stage develop --output docs/reproduced-path-risk
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_path_risk.py --stage evaluate --output docs/reproduced-path-risk
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/verify_spike_tick_path_risk.py --output docs/reproduced-path-risk
```

The independent stdlib audit passed on the first actual attempt: 13,939,724
checks, zero errors, 24 sources, 2,073,582 ticks and 3,918 strategy paths.
66,736 overlapping-label path records INCLUDE repeated family artifacts;
these are not unique trades. It checked 507 input hashes and 4,320 saved
false safety values. [Audit](independent_audit.json), [stdout](independent_audit.stdout.log).

Scope includes its own causal risk-window reconstruction, common availability,
closed frame/ATR timestamps, saved matrices, training-only scaler arithmetic,
scalar predictions and cutoffs, singleton/joint quote paths, point metrics,
unknown outcomes and policy gates. It excludes regeneration of all 44 inputs,
ridge refitting, bootstrap draw regeneration/inferential validity, independence,
fresh OOS, broker fills and actual profit. Raw provenance is anchored to the
passed round7 audit, without renormalizing raw sources here. Metadata path
portability is synthetically tested; a full historical Ubuntu reproduction
has not been executed.

Full local Python suite: 1,364 passed; compilation passed. Measured trading
costs, broker execution, cash profit and prospective paper results: NOT TESTED.
The complete research goal remains unmet; no strategy is promoted.
