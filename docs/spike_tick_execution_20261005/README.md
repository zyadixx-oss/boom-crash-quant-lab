# Sampled public-tick execution diagnosis

This round preserves round6 model issuance and evaluates quote ordering on 12
deterministically spaced UTC days for BOOM600 and CRASH600. The combined
H4/H1 + M15/M5/M1 signals are unchanged. See the
[frozen protocol](../SPIKE_TICK_EXECUTION_PROTOCOL.md) and
[preacquisition declaration](declaration.json).

Executed results: [Arabic report](REPORT.ar.md), [all primary metrics](metrics.csv),
[full results and sensitivities](results.json), and
[independent audit](independent_audit.json). The audit passed 2,364,376 checks on
24 sources and all 16 primary groups, with 3,083 paths and no errors. It does not
regenerate ML features/fits/scores, bootstrap or secondary ledgers/inference.

This is an adaptive execution diagnostic on known historical periods. At most
576 opportunities per model cannot establish the PF >= 1.5 and 1,000 completed
held-out simulations target. Previous development rejection remains binding.
Actual broker fills, measured spread/costs, monetary profit and prospective
paper performance remain **NOT TESTED**.

## Reproduction

Use Python 3.12 and `requirements-research-lock.txt` at the repository root.
The local source CSVs, raw wire pages, checkpoints and issuance/trade ledgers
are ignored by Git. GitHub contains their hashes and audit metadata. Historical
server revisions can change recollected bytes; a changed source needs a separate
declaration and must never overwrite this frozen run.

```bash
export LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false
.venv/bin/python scripts/collect_spike_ticks.py \
  --symbols BOOM600 CRASH600 \
  --dates 2026-04-11 2026-04-26 2026-05-12 2026-05-28 \
          2026-06-13 2026-06-29 2026-07-15 2026-07-31 \
          2026-08-16 2026-09-01 2026-09-17 2026-10-03 \
  --page-size 1000 --delay .35 \
  --output-dir data/spike_tick_execution_20261005
.venv/bin/python scripts/run_spike_tick_execution.py --stage freeze-data
.venv/bin/python scripts/run_spike_tick_execution.py --stage evaluate
.venv/bin/python scripts/verify_spike_tick_execution.py
```

Those commands require the original immutable declaration and round6 inputs.
Existing frozen outputs refuse overwrite. Resume a confirmed terminated
acquisition only with the same configuration and `--resume`, preserving the
previous acquisition summary first. An observation timeout does not justify
starting a second collector.

Only the public `active_symbols` and explicitly bounded `ticks_history` requests
are allowed. Effective spacing is at least 0.5 seconds. Missing ticks remain
unknown; prices are never interpolated. Complete UTC-minute OHLC must match the
frozen M1 source before the paired payoff calculation.

Primary entry, stop exit and timeout use the next observed quote according to
the fixed protocol. A stop trigger cannot be cancelled by a later recovery.
Costs are modeled in ATR; these historical indicative quotes are not executable
broker fills or cash results.
