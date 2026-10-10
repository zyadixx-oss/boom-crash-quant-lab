# Executed native M5 chronological research — rejected

See [Arabic report](REPORT.ar.md), [selection](selection.json), [declaration](declaration.json),
[metrics](metrics.csv), [structured results](results.json), and [independent audit](independent_audit.json).
No model satisfies netPF>=1.5 and>=1000 completed held-out trades per model/symbol.
All four live flags remain false; no orders/account/fills are used.

Older300 input: data/spike_learned_transfer; recent300 input: data/spike_native300_recent.
Their acquisition/normalization metadata, source hashes, request boundaries and
unfilled gaps are tracked. Raw price CSVs and simulated ledgers remain cached
locally and ignored byGit. Restore exact frozen CSV bytes before reproduction;
provider retention may prevent later recollection of the oldest period.

```sh
export LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false
# Install requirements-research-lock.txt in a Python 3.12 virtual environment.
python scripts/run_spike_native300_study.py --stage develop --output work/native-reproduction
python scripts/run_spike_multiframe_study.py --stage develop --output work/multiframe-reproduction --native-study work/native-reproduction
# Both native and multiframe selections must freeze before evaluating recent data.
# Evaluate native first, then multiframe; multiframe verifies that chronology.
python scripts/run_spike_native300_study.py --stage evaluate --output work/native-reproduction --bootstrap 9999
python scripts/run_spike_multiframe_study.py --stage evaluate --output work/multiframe-reproduction --native-study work/native-reproduction --bootstrap 9999
python scripts/verify_spike_native300.py --study work/native-reproduction
python scripts/verify_spike_multiframe.py --study work/multiframe-reproduction
```

Follow the [native protocol](../SPIKE_NATIVE300_PROTOCOL.md) and
[combined protocol](../SPIKE_MULTIFRAME_PROTOCOL.md) together when reproducing the
paired release. Use new output folders: the runners refuse rewriting frozen
selection/evaluation. Timestamp metadata change selection hashes on a new run;
model coefficients and numeric ledgers should reproduce from exact source/code.
The later multiframe report supplies the jointHolm8 inference; nativeHolm4 status alone is reference-only.

Historical quote-path stress and assumed cost are not broker execution or cash
profit. Actual spreads/fills/cash returns and prospective paper evidence remain
NOT TESTED. No retrospective result changes readiness or execution flags.
