# Executed round11: fixed one-minute reward target ablation

See [Arabic report](REPORT.ar.md), [frozen protocol](../SPIKE_SHORT_TARGET_PROTOCOL.md),
[declaration](declaration.json), [selection](selection.json), [results](results.json),
[development metrics](development_metrics.csv) and [primary/secondary metrics](metrics.csv).
The protocol retains pre-execution wording because its bytes were frozen.

SHORT44 trains the existing closed H4/H1 + M15/M5/M1 inputs on one-minute reward;
LONG44 trains them on15-minute reward and is reference_only. Both and CLOCK
are evaluated on the fixed one-minute policy. Native BOOM600/CRASH600 SPIKE and
DRIFT stay separate. No feature/clock/cost/stop/entry/cutoff grid was searched.
All development SHORT selections were rejected. Later180-day SHORT point PFs
are0.000/0.265/0.285/0.809 with8/2/104/4 completed simulations. No model reaches
PF>=1.5/n>=1000; all means are negative, Holm p=1 and all economic gates fail.

The older180 days use three expanding40/50/60% fits/next10% validations and a
final70% fit before newly calculating older30% secondary/later180 primary.
Prices and prior15-minute payoffs were already exposed; this is adaptive
known-history research, not fresh price OOS, label holdout or prospective proof.
The final common training intersection has5926 targets per symbol/direction.
Targets/family copies cannot count toward strategy samples or independence.

Common closed M5 opportunities are UTC minute00/30. M1 proxy entry is+1minute,
SL2ATR, one-minute hold, noTP, adverse minute-extreme stop fill and last minute
close timeout; assumed total round-trip cost0.10ATR. Common31-minute purge,
one-open occupancy, exact missing-path handling and train-only scaling apply.
All primary ledgers have known outcomes; sparse SHORT samples still invalidate
positive inference. Day/week PF intervals use finite draws only: undefined
PF draws respectively are1/1,3734/3545,0/0,3569/3614 for the four SHORT groups.
These counts block bounded positive claims and are never silently normalized
away. No global discovery correction across adaptive rounds is claimed.

Selection's copied prices_decoded=false and
new_features_labels_models_or_ledgers_computed=false are premeasurement
DECLARATION SNAPSHOT fields, not runtime-state assertions after development.
Development/evaluation did decode prices and compute artifacts. Frozen bytes
remain unchanged to preserve the genuine declaration chronology.

Full reproduction requires exact ignored local sources/matrices/targets/
signals/ledgers and pinned ancestor artifacts. Git includes code, protocols,
synthetic tests, frozen models/metadata/results, logs and audit evidence.
A fresh checkout can run synthetic tests but cannot reproduce historical
numbers without those exact bytes. Recollection with different bytes is a new
dataset. Use a separate empty output directory; frozen stages refuse overwrite:

```bash
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_short_target.py --stage declare --output docs/reproduced-short-target
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_short_target.py --stage develop --output docs/reproduced-short-target
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/run_spike_short_target.py --stage evaluate --output docs/reproduced-short-target
LIVE_TRADING=false READY_FOR_LIVE=false LIVE_ALLOWED=false OPENED_TRADES=false \
  .venv/bin/python scripts/verify_spike_short_target.py --output docs/reproduced-short-target
```

Predeclaration full Python science suite:1476 passed, one existing warning;
compilation passed. Verifier synthetic suite:102 passed before actual audit.
Independent actual audit passed on its second attempt:9,040,587 checks,
zero errors,4 M1 sources/1,036,798 clean rows and444 saved false safety values.
It checked51,561 learned/evaluation strategy path records plus20,712 repeated
development CLOCK records.148,160 target records and74,080 common matrix rows
include repeated fits/modes/family artifacts; they cannot be pooled as strategy
sample evidence. [Passing audit](independent_audit.json), [stdout](independent_audit.stdout.txt).

The first attempt's258 third-period metric mismatches were caused by the
verifier omitting the inherited31-minute planned purge at third endpoints.
[Failed audit](independent_audit.attempt1.json) and its stdout/hashes were
preserved before changing only verifier logic and two boundary regression
cases. Frozen science/source/model/selection/result bytes were unchanged.
Final verifier synthetic suite:104 passed. Final full Python suite:1580 passed,
one existing warning; compilation passed. [Local full-suite log](final_python_tests.stdout.txt).

Scope covers pinned normalization/page lineage, saved closed-frame timestamps/
ATR/context/matrix bytes, dual-target paths/common membership/native row hashes,
train-only scaler and scalar score/cutoff arithmetic, issuance, all saved model/
CLOCK ledgers, point metrics, unknowns/purges/eligibility and Holm-four arithmetic
from saved p-values. It excludes independent regeneration of all44 features or
the complete feature-valid universe, ridge refitting, bootstrap draws/CI/p
regeneration or inferential validity, independence, tail diagnostics/export
copies, tick/bid-ask execution, broker costs, cash profit and prospective proof.
Post-PASS hash-only verification captured212 unique local CSV artifacts.
Metadata portability is synthetically tested; full historical Ubuntu
reproduction has not been executed.
Cash profit, broker fills, measured costs and prospective paper: NOT TESTED.
All four live flags remain false; the complete research goal remains unmet.
