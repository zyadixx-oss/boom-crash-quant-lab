# Results — actual offline Spike Hunter study

Executed 2026-10-04 using official public Deriv M1 candles. See
[full Arabic report](spike_hunter_20261004/REPORT.ar.md),
[frozen protocol](SPIKE_HUNTER_PROTOCOL.md), and
[all metric rows](spike_hunter_20261004/metrics.csv).

- BOOM500 and CRASH500: 259,199 valid M1 candles each, 2026-04-07 11:08 UTC to
  2026-10-04 11:08 UTC exclusive. One missing minute per symbol remains unfilled.
- Sixteen causal variants, 12 excursion definitions: 1.5/2/3 M5 ATR in 5/10/15/30m.
- Chronological 70/30, three expanding development validation windows, common
  30-minute label purge, paired day-block confidence intervals and Holm correction.
- Primary 2 ATR/15m BOOM500 raw CRT: 59/229 hits, precision25.76%, base22.22%,
  lift1.159, day-block CI95[0.926,1.392]. No confirmed timing advantage.
- Primary CRASH500 resistance alignment:195/639 hits, precision30.52%,
  base21.61%, lift1.412, day-block CI95[1.268,1.551]. Strongest research candidate.
- Post-hoc ATR matching: Crash resistance lift1.196 CI95[1.073,1.312];
  Boom CRT lift1.076 CI95[0.857,1.293]. The diagnostics use the same historical
  period and cannot serve as independent confirmation.
- An independent199,999-resample check supports the raw Crash resistance result;
  it does not test profitability or arrival of independently segmented tick spikes.
- Python validation:46passed. Independent scalar replay matches prices, causal
  features, signal timestamps,24,096labels, all384final comparisons and gates.

## Executed quote-path payoff extension

The locked [payoff study](spike_payoff_20261004/REPORT.ar.md) now tests normalized
bracket returns. It does not replace the earlier excursion results.

- 432 configurations per symbol; first70% development, three expanding
  walk-forward selections, final30% explicitly labelled reused/exploratory.
- Additional external period:2025-10-09 11:08 UTC to2026-04-07 11:08 UTC
  exclusive;259,200 genuine M1 candles per symbol, no gaps or overlap.
- No configuration satisfied all development eligibility rules. Diagnostic
  fallbacks remain rejected, regardless of later results.
- External fixed support/resistance model: BOOM500 n=2,216, mean netR=-0.317,
  CI95[-0.371,-0.263], PF0.587; CRASH500 n=2,259, mean netR=-0.304,
  CI95[-0.356,-0.250], PF0.603.
- Crash diagnostic CRT+MSS fallback: n=341, mean netR=-0.105,
  CI95[-0.172,-0.034], PF0.673. Boom compression fallback has only one external
  trade and provides no useful population estimate.
- All six independently selected walk-forward validation folds lost on average.
  All four external gates failed. Tested zero-cost and more favorable execution
  sensitivities did not reverse the locked configurations' negative averages.
- The primary model assumes entry one minute after signal,0.10ATR hypothetical
  total cost, stop-first ambiguity, and adverse minute-extreme stop fills.
  See [interpretation](spike_payoff_20261004/README.md) before using any metric.
- Full Python validation now:121passed. Original frontend code was unchanged
  by this research extension.
- Independent standard-library scalar reconstruction matched all4,817 external
  ledger trades, including causal ATR, execution path and summary arithmetic:
  91,733 checks, zero mismatches. Baseline, bootstrap and sensitivity replay are
  outside that independent ledger audit's scope.

## Evidence boundary

The subsequent [uncapped time-exit round](spike_timed_20261005/REPORT.ar.md)
tested384 configurations per500 symbol without a take-profit cap, in both
spike and opposite direction. No final development selection qualified, and
all six expanding validation folds lost on average. The primary spike-direction
CRT with SL2ATR/time15m transferred unchanged to Boom1000/Crash1000:

| Transfer symbol | n | Mean netR | Day-block CI95 | PF |
|---|---:|---:|---|---:|
| BOOM1000 | 505 | -0.103 | [-0.206,0.006] | 0.807 |
| CRASH1000 | 484 | -0.102 | [-0.212,0.009] | 0.812 |

All four within-round transfer gates failed. The rejected diagnostic
Boom BB-squeeze fallback has a tiny positive gross0.003738R at zero cost but
negative-0.002512R already at hypothetical0.025ATR; its cost-free interval
crosses zero. It does not establish an execution-cost-aware advantage.

This round is adaptive cross-symbol replication, not fresh temporal/forward
confirmation or a project-wide multiple-testing guarantee. The requested
older500 history was only4.28days available and was not used as a sufficiently
powered external test. Authentic1000 transfer history has259,200 M1 rows per
symbol and no gaps. Independent scalar replay verifies all5,191 transfer
trades in98,852 checks with zero mismatches. Full Python tests:184passed.

The first study uses future directional excursion labels; opportunity recall
is not unique-event recall. The extension tests quote-path R, win rate, profit
factor, expectancy and a closed-trade risk illustration under declared assumptions.
Monetary net profit, actual fills, measured historical spread/slippage,
independently defined tick-spike recall, and forward/shadow performance remain
**NOT TESTED**. The older weighted
backend strategy has not been evaluated by this isolated causal study. No
parameter should be promoted to production/live based on these outputs.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`, and
`OPENED_TRADES=false`. No accounts, credentials or order execution were used.

## Fixed learned-score extension — executed and rejected

See [Arabic report](spike_learned_20261005/REPORT.ar.md) and
[frozen protocol](SPIKE_LEARNED_PROTOCOL.md). Nineteen causal continuous inputs,
one fixed ridge model per direction/symbol, mean-SSE penalty0.1, train-only
standardization and a positive-floor training q75 cutoff. Fixed stop2ATR/time15m,
entry+1minute, adverse-extreme stop stress, modeled total cost0.10ATR. No
hyperparameter/exit search. First70% development with three expanding refits;
both final directions remain rejected when development eligibility fails.

User target: modeled net PF>=1.5 and>=1,000 completed out-of-selection trades
per model and symbol, with no pooling. None of the four models qualified during
development. After SHA256 freezing, complete official180-day300 inputs gave:

| Transfer spike direction | n | PF | Day PF CI95 | Weekly PF CI95 | Mean netR |
|---|---:|---:|---|---|---:|
| BOOM300N | 8,354 | 0.820 | [0.775,0.866] | [0.780,0.862] | -0.0625 |
| CRASH300N | 8,607 | 0.854 | [0.809,0.902] | [0.804,0.908] | -0.0502 |

Both opposite-direction models issued zero signals. All four transfer gates
failed; neither spike direction had positive chronological thirds or evidence
of an advantage over identical-clock entry. This is an older cross-symbol
replication, not future chronological confirmation. Earlier unverified Boom300
classification claims and all reused500 diagnostics are disclosed.

Post-result input-only diagnosis: log(ATR/price) shifts about6.94 trainingSD,
driving near-universal spike issuance and no opposite-direction issuance.
The diagnosis does not change the scaler, cutoff or predictions. Native-symbol
training followed by a later held-out period was the next declared question;
its subsequently executed results are reported in the section below.

Independent scalar ledger audit:16,961 transfer paths,492,290 checks,zero
errors; fits/features/bootstrap are outside that audit scope. Python validation:
287 passed. Actual money profit, broker execution/costs and prospective paper
performance remain NOT TESTED. All four safety flags remain false.

## Native300 and the user's combined-timeframe test — executed and rejected

The user clarified H4/H1 context plus M15/M5/M1 as one combined strategy,
rather than independent profit claims on each chart frame. Exactly44 fixed
causal inputs retain the prior ridge/cutoff/stop/time/cost parameters. Both
native19 and combined44 selections froze before preparing recent300 features.
Training used first70% of the older180-day300 period with three expanding
refits. Fresh chronology is2026-04-07 to2026-10-04,11:08UTC exclusive.
One missing minute per clean source remains unknown; no interpolation.

NativeM5 fresh PFs were0.821(n160),0.877(n610),0.909(n321),0.655(n170).
No development direction qualified. The combined model also had no eligible
direction and subsequently returned:

| Combined fresh model | n | Net PF | Day PF CI95 | Weekly PF CI95 |
|---|---:|---:|---|---|
| BOOM300N SPIKE | 460 | 0.737 | [0.581,0.929] | [0.592,0.907] |
| BOOM300N DRIFT | 1,594 | 0.824 | [0.733,0.932] | [0.719,0.956] |
| CRASH300N SPIKE | 1,036 | 0.813 | [0.694,0.947] | [0.680,0.957] |
| CRASH300N DRIFT | 833 | 0.812 | [0.681,0.960] | [0.663,0.989] |

Both cohorts meeting the1,000-trade sample threshold still failed the profit
target. All four mean-R estimates were negative with negative daily/weekly95%
intervals under the primary cost/stress model. Zero modeled cost did not reverse
their observed negative means. No clock or common-clock M5 advantage was proven.
JointHolm includes all8native/combined fresh hypotheses; all adjusted p=1.
NativeHolm4 results are retained as immutable references, not joint discoveries.

M5 reference predictions use the same44-input-eligible timestamps, intersecting
only feature availability. Each fresh source has8,393combined vs8,600fullM5
clock opportunities;207warmup/gap/context exclusions are disclosed. This tests
closed M1 inputs at UTC00/30 decisions, not minute-frequency order execution.

See the [combined Arabic report](spike_multiframe_20261005/REPORT.ar.md),
[frozen protocol](SPIKE_MULTIFRAME_PROTOCOL.md), and
[native reference](spike_native300_20261005/REPORT.ar.md). Full Python tests:
340passed. Independent nativeM5 ledger reconstruction verifies1,261trades in
37,065checks with zero errors. Independent combined/common-M5 reconstruction
verifies all5,165 paths in150,871 checks with zero errors, including saved
conjunction/ jointHolm8 arithmetic. ML fitting/features/scores and bootstrap
p-values/intervals are outside the independent ledger audit's scope.
Historical methodology is adaptive; prospective paper and cash profit remain
NOT TESTED. All four execution/readiness flags remain false.

## Fixed nonlinear native600 multiframe test — executed and rejected

One declared residual histogram boosting alternative on the same44 completed
H4/H1/M15/M5/M1 inputs, with same-clock ridge44 and ridge19 references.
No hyperparameter/threshold/exit/cost search. Two directions, two symbols,
three estimators:12 joint later hypotheses. Native BOOM600/CRASH600 use360 days
each; first70% of the older180 was clipped before indicator construction.
Three expanding refits preceded a frozen selection; no development candidate
qualified, including Crash drift validation PF1.151 with a negative daily
selection lower bound. The full later180 and olderlast30 were read afterwards.

| Later nonlinear model | n | Net PF | Day PF CI95 | Weekly PF CI95 |
|---|---:|---:|---|---|
| BOOM600 SPIKE | 1,475 | 0.834 | [0.722,0.960] | [0.735,0.948] |
| BOOM600 DRIFT | 1,490 | 0.827 | [0.732,0.937] | [0.730,0.932] |
| CRASH600 SPIKE | 1,306 | 0.938 | [0.814,1.082] | [0.831,1.063] |
| CRASH600 DRIFT | 2,084 | 0.851 | [0.773,0.938] | [0.772,0.941] |

All four satisfy the1,000 completed-path sample threshold, all fail PF>=1.5.
All12 later estimator PFs<1; jointHolm adjusted p=1. Every nonlinear vs linear
mean-R difference day/week interval crosses0. CrashSPIKE zero modeled cost
PF1.056 is a diagnostic sensitivity, not the primary result or a qualifying
strategy. The other three nonlinear gross observed means remain negative.
Both older600 grids are complete; each later grid has one declared missing
minute, preserved unknown. Recovered request errors and whole-Crash retry
lineage are retained; no fabricated candles or interpolation.

See [Arabic report](spike_nonlinear_20261005/REPORT.ar.md),
[frozen protocol](SPIKE_NONLINEAR_PROTOCOL.md) and
[reproduction instructions](spike_nonlinear_20261005/README.md).
Independent reconstruction:12,621 primary paths,375,964 checks,0 errors.
The verifier-only obsolete metadata-field failure is preserved; after its
repair all frozen prices/models/results hashes remained unchanged. Audit
excludes regenerated ML features/fits/scores, bootstrap inference, possible
clock completeness, clock-baseline replay, sensitivities and secondary ledgers.
Local Python validation484 passed (482 backend+2 acceptance), including144 new
synthetic cases. No money profit, broker fills/measured costs or prospective
paper claims. LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES=false.
