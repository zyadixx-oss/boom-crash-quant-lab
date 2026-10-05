# Executed round8 fixed tail-tick mechanism pilot

See [Arabic report](REPORT.ar.md), [results](results.json), [metrics](metrics.csv),
[pre-calibration declaration](declaration.json) and the
[frozen protocol](../SPIKE_TICK_TAIL_PROTOCOL.md). The protocol retains its
pre-execution wording because its bytes were frozen before calibration.

Two independent prefix-only detectors were adequate:691 BOOM600 and684
CRASH600 calibration events. Last30% older/younger hazard ratios were1.045
CI95[0.987,1.137] and1.226 CI95[1.081,1.410]. All four declared large-overdue
ratio gates reject≥1.5 conditional on the fixed detector and observed-day
resampling. This does not establish independence or strategy profitability.

The sample is24 already-used discontinuous tick days, not fresh strategy
validation. Intervals condition on the calibrated median with just5/4observed
clusters in pooled validation/final. Profit factor, actual monetary profit and
prospective paper evidence are NOT TESTED. All four live flags remain false.

The local source quotes/raw provenance are ignored in Git; round7 retains their
hashes/manifests/page audit and passed independent raw reconstruction. The
runner refuses source/science drift, live flags or overwrites. Reproduction on
preserved local sources requires a separate empty output directory:

```bash
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_tail.py --stage declare --output docs/reproduced-tail
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_tick_tail.py --stage evaluate --output docs/reproduced-tail
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/verify_spike_tick_tail.py --output docs/reproduced-tail
```

A GitHub checkout without the exact local source bytes can run synthetic tests,
but cannot reproduce the historical event counts from the committed metadata
alone. Recollecting public history may return revisions and is a new dataset.

Independent stdlib audit passed first attempt:6,685 checks,24sources,12segment
summaries,zero mismatches. It independently recalculates clean grids, calibration,
events, prior ages, gap resets, all saved point band/day/drift arithmetic and
policy fields. It anchors raw integrity through the passed round7 audit; it does
not repeat raw normalization or regenerate bootstrap draws/validate inference,
compare an unsaved runtime last-mark table, test profit or broker execution.
[Audit](independent_audit.json) and [preserved stdout](independent_audit.stdout.log).
