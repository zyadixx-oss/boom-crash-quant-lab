# Executed round12: frozen SHORT1 tick-versus-M1 fidelity

See [Arabic report](REPORT.ar.md), [frozen protocol](../SPIKE_SHORT_TICK_PROTOCOL.md),
[declaration](declaration.json), [results](results.json) and [metrics](metrics.csv).

The exact saved round11 later180 SHORT44/LONG44/CLOCK issuance is replayed on
24 existing public tick sources/12 scheduled post-fit UTC days per symbol.
All models were trained before the first sampled day. No fitting, features,
predictions, threshold, source/date choice or cost/exit grid is added. The
same44 inputs combine closed H4/H1 + M15/M5/M1. Prices, prior payoffs and tick
ordering are known; this is adaptive execution diagnosis, not fresh OOS or
prospective profit evidence. The four final-day clusters of rounds9/10 were
their fitted tick-feature partitions, not this unchanged older-M1 model cohort.

Tick primary: first quote strictly after issue+1minute, SL2ATR, nominal expiry
atissue+2minutes, next quote after stop trigger/expiry, maxgap1second, cost.10ATR.
M1 comparison: unchanged adverse-minute-extreme stop/close timeout. Both have
common31-minute planned per-observed-day purge, one-open occupancy and frozen
ATR/signals. Entry/stop/expiry jointly differ; delta is not stop-only slippage.
All sampled filled paths completed, despite18 missing seconds in the complete
source sample. Source normalization is anchored to the passed round7 audit.

No model qualifies. SHORT has only8 CRASH600 SPIKE completions/PF.48968;
other SHORT streams are empty with null PF and9999 undefined PF draws. LONG
reference PFs are.68913/.18699/.29566/.47050 with59/49/54/114 completions.
CLOCK PFs are.53880/.20871/.31642/.34627, each564 paths/12 active days.
All observed net means are negative. Twelve days/max564 cannot meet1000/60;
prior development rejection is binding, and no reference can be promoted.

Aggregate CIs use9999 paired scheduled-day resamples/seed20261005, including
observed no-issuance days without padding unsampled dates as zero returns.
Per-day summaries are points only. Undefined draws/counts are retained and
finite-draw intervals never supply unconditional inference. Point estimates
and bootstrap bounds do not prove independence, a discovery or expected PF.

Full reproduction requires exact ignored local tick/M1 copies, saved signals,
ledgers and hash-pinned ancestors. Fresh checkout synthetic CI cannot recreate
historical numbers without those bytes; recollection may be a different dataset.
Use a separate research output directory; frozen stages refuse overwrite:

```bash
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_short_tick.py --stage declare --output docs/reproduced-short-tick
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_short_tick.py --stage evaluate --output docs/reproduced-short-tick
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/verify_spike_short_tick.py --study docs/reproduced-short-tick --output docs/reproduced-short-tick/independent_audit.json
```

Full predeclaration science tests:1645 passed/one existing warning; compilation
passed. Includes65 focused source/synthetic tests and a full synthetic
metadata-declare/evaluate integration with emptySHORT, blankCLOCK, gaps and
common purge. Final repaired-verifier suite:121 focused tests; full Python suite:
1766 passed/one existing warning, with compilation passed. See saved
[full test log](final_python_tests.stdout.txt), [repair test log](verifier_repair_tests.stdout.txt)
and [compile log](final_compile.stdout.txt), each with a SHA256 sidecar.

The [actual independent audit](independent_audit.json) passed the second attempt:
9,122,119 checks, zero errors, 2,540 tick policy records, 2,540 coarse records
and 2,540 matched-known records, across24 tick sources,2M1 sources and12 saved
issuance streams. These are policy paths across repeated strategy/reference
streams, not pooled independent trades.875 input hashes were verified;2,760
safety values were checked. Scope: saved issuance/ATR identity, scalar tick/M1
execution paths, point metrics, matched/control means, CI schema/counts and
refusal gates. It excludes raw-wire renormalization,44-feature/universe
regeneration, fitting/predictions, bootstrap draws/CI validity, new holdout,
independence and actual broker profit.

The [first failed audit](independent_audit.failed_attempt01.json) and its stdout
and SHA256 sidecars were archived before the verifier-only repair. An inherited
exact-match OHLC distance returned integer0 whereas the result schema requires
float0.0. The independent wrapper now preserves float quote-distance units;
a regression test also uses the runner's real reconciliation schema. Frozen
science, declaration and results were unchanged. Passing audit SHA256:
`588841f5779bbe0d1fcea72a1fa24b7eb509bd8aec20e7c722bda599263a6229`.
All four live flags remain false. Broker fills, measured costs, actual cash
profit and prospective paper results remain NOT TESTED. The full goal is unmet.
