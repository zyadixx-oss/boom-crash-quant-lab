# Learned ridge payoff study — executed, rejected

See [Arabic report](REPORT.ar.md), [frozen protocol](../SPIKE_LEARNED_PROTOCOL.md),
[selection](selection.json), [metrics](metrics.csv), [raw structured results](results.json),
[independent ledger audit](independent_audit.json), and [transfer shift](transfer_shift.json).

Four fixed models use nineteen causal completed-M5 inputs, train-only scaling,
mean-SSE ridge penalty0.1, and a frozen positive-floor q75 issuance cutoff.
No hyperparameter or exit search. The stop2ATR/time15m policy has no TP cap,
entry+1minute, primary adverse-extreme stress, modeled round-trip cost0.10ATR.
Both directions train independently on first70% of each500 input. Three expanding
out-of-fit validation folds precede final fitting and SHA256 freezing.

No development direction qualified. Frozen transfer to BOOM300N/CRASH300N gave
PF0.820(n8354) and0.854(n8607) in spike direction; drift issued no signals.
The user target PF>=1.5 and n>=1000 per model/symbol was not achieved.
All four transfer gates failed. This is older cross-symbol replication, not
future temporal confirmation. Reused500 periods are exploratory diagnostics.
Earlier conversation also contained an unverified Boom300 classification claim.

Transfer source CSVs remain local; their metadata and reproducible collection
parameters are tracked under data/spike_learned_transfer. Price ledgers are ignored
by Git; regenerate using the frozen source CSVs and installed research requirements.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-research-lock.txt
export LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false
python scripts/collect_spike_history.py --days 180 --symbols BOOM300N CRASH300N --cutoff-epoch 1775560080 --output-dir data/recollect300
# Copy the verified canonical CSVs/manifests into data/spike_learned_transfer,
# and restore the original500/older500 files with exact tracked hashes.
# Use a fresh output directory when reproducing the development stage:
python scripts/run_spike_learned_study.py --stage develop --output work/learned-reproduction
python scripts/run_spike_learned_study.py --stage evaluate --output work/learned-reproduction --bootstrap 9999
python scripts/verify_spike_learned.py --study work/learned-reproduction
python scripts/diagnose_learned_transfer.py --study work/learned-reproduction
python -m pytest
```

Timestamps in selection/run metadata vary on reproduction, so the selection file
fingerprint itself changes. Numerical models/ledgers should reproduce with exact
source prices and code; the saved selection fingerprint applies to this executed run.
The runner refuses overwriting frozen selection/evaluation and checks all code and
price hashes before evaluation. Never edit a frozen source to tune this run.

PF CI uses shared gain/loss day clusters, with undefined no-loss samples counted.
The scalar audit covers all16961 transfer trades and summary arithmetic, not ML
training/features/scores or bootstrap. Quote paths, stops, costs and closed-equity
risk are hypotheses; actual broker fills, cash returns and prospective paper
performance are NOT TESTED. No output enables execution.
