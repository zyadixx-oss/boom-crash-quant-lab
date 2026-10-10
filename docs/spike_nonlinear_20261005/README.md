# Fixed nonlinear multiframe study — executed round6

Read [Arabic findings](REPORT.ar.md) and the
[frozen design](../SPIKE_NONLINEAR_PROTOCOL.md). No model met net PF>=1.5 with
at least1,000 completed held-out simulations per symbol/model. All12 models
failed development, both selections are null, and all12 later-period PFs<1.
These are modeled quote-path outcomes, not cash, broker fills or prospective
paper results. All four live/readiness/execution flags remain false.

## Files and evidence

- `declaration.json` / `.sha256`: source, parameter, protocol and code hashes
  recorded before development.
- `selection.json` / `.sha256`: all12 final models and three training-only
  validation refits; frozen before oldlast30 or later features/payoffs.
- `results.json` / `metrics.csv`: both held-out cohorts, all12 later hypotheses,
  jointHolm12, bootstrap intervals and18 predeclared sensitivities per model.
- `independent_audit.json`: PASS,12,621 paths,375,964 checks,zero errors.
  `independent_audit_attempt1.json` preserves a verifier-only obsolete metadata
  key failure; frozen code/models/prices/results were unchanged during repair.

Primary ledgers and saved issuance CSVs are present locally but git-ignored.
Price CSVs are cached under `data/spike_nonlinear600_old` and
`data/spike_nonlinear600_recent`; their raw/clean hashes, recovered successful
page audits, error lineage and whole-Crash retry metadata are tracked.
The later missing minute is unknown, never filled. Source first70 is clipped
before computing development indicators. RIDGE19 is a same-clock reference,
not a candidate satisfying the combined H4/H1/M15/M5/M1 requirement.

The standard-library independent audit covers source/frozen chronology,
saved issuance-to-ledger coverage, causal M5 ATR, scalar payoff/summary
arithmetic and copied reference/Holm consistency. It excludes regenerated ML
features/fits/scores, bootstrap inference, all possible common-clock
opportunities, clock-baseline replay, sensitivities and secondary ledgers.

## Local execution

From the repository root with Python3.12 and the cached audited data:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-research-lock.txt
export LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false
.venv/bin/python -m pytest
.venv/bin/python scripts/run_spike_nonlinear_study.py \
  --stage develop --output docs/spike_nonlinear_reproduction
.venv/bin/python scripts/run_spike_nonlinear_study.py \
  --stage evaluate --output docs/spike_nonlinear_reproduction --bootstrap 9999
.venv/bin/python scripts/verify_spike_nonlinear.py \
  --study docs/spike_nonlinear_reproduction
```

Do not overwrite the original frozen declaration/selection/results; the runner
refuses this.
Reproduction uses already-known historical outcomes and is not a new
validation experiment. No threshold/config changes belong in this frozen run.

GitHub stores metadata and models, not price/ledger CSVs. On a fresh clone,
restore the byte-identical local CSV archive first. Recollection should use a
separate acquisition directory and compare the recorded hashes; server history
or pagination revisions may change bytes. Changed bytes require a new declared
run and cannot be substituted into this evidence. No tokens/accounts/orders
are needed or permitted by the collector.
