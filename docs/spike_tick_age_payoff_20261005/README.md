# Executed round9 tick-age payoff pilot

See [Arabic report](REPORT.ar.md), [results](results.json), [metrics](metrics.csv),
[frozen declaration](declaration.json), [frozen selection](selection.json), and
[protocol](../SPIKE_TICK_AGE_PAYOFF_PROTOCOL.md). The protocol retains its
pre-execution wording because those bytes were frozen before measurement.

RIDGE44 and RIDGE46 share closed H4/H1 + M15/M5/M1 inputs, exact M5 decisions,
training targets and availability;46 adds strict-prior tail age and last mark.
All eight final point PFs are below1. The four incremental46-minus44 mean
intervals cross zero. Final model completions range71–136 across just four
observed day clusters. All development selections were rejected before final
labels. No strategy, actual profit or live readiness is established.

This is adaptive economic research on the same12discontinuous tick days per
symbol. Its individually replayed training/diagnostic labels may overlap and
are not strategy ledgers or the user's1,000-completion sample. Conditional iid
observed-day intervals do not establish week/prospective inference. Unknown
payoffs remain unknown; whole-band rejection is blocked when any planned
eligible outcome is missing.

All source and science hashes are pinned before new labels/fits. Selection,
models/scalers/cutoffs, matrices, inputs, labels and issuance are pinned before
final evaluation. The runner refuses changes and output overwrites. Reproduce
only on the exact preserved local source bytes in a new empty output directory:

```bash
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_age_payoff.py --stage declare --output docs/reproduced-age-payoff
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_age_payoff.py --stage develop --output docs/reproduced-age-payoff
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_age_payoff.py --stage evaluate --output docs/reproduced-age-payoff
```

Git includes metadata/results/code/tests; ignored local inputs, labels, signals,
ledgers and raw prices are necessary for full historical reproduction. Public
recollection may return changed history and constitutes a different dataset.
Actual broker fills, measured costs, monetary profit and prospective paper
performance remain NOT TESTED. All four live flags remain false.

Independent audit passed the second attempt:11,803,332 checks,24 sources,
3,866 strategy paths, zero errors.66,638 overlapping-label records include
repeated family artifacts and are not unique trades.333 input hashes and2,260
saved false safety values were checked. Full current Python suite:1,111 passed;
compilation passed. The first failed attempt is preserved: six grid-minute count
mismatches were repaired only in the verifier, without changing frozen science,
data, models or results. Its missing minute remained unknown.

Scope includes strict-prior age/mark/common availability, causal ATR/context
stamps, saved44/46 matrices/scalers/scalar predictions/cutoff, singleton/joint
paths and point/gate arithmetic. Full44 indicator regeneration, ridge refitting,
bootstrap regeneration/inferential validity, fresh OOS/independence and broker
profit are outside scope. Raw provenance is anchored to passed round7 audits.
[Passing audit](independent_audit.json), [first attempt](independent_audit.attempt1.json).

```bash
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/verify_spike_tick_tail_signal.py --output docs/reproduced-age-payoff
```
