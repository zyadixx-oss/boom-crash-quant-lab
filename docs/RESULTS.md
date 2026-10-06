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


## Sampled public-tick execution diagnosis — executed, no promotion

See [Arabic report](spike_tick_execution_20261005/REPORT.ar.md) and
[frozen protocol](SPIKE_TICK_EXECUTION_PROTOCOL.md). Twelve predeclared UTC days
per native600 symbol, 24 sources, 2,073,582 exact public quotes, 2,088 successful
pages, 18 missing seconds retained. All34,556 complete minute OHLCs match the
frozen M1 prices exactly; four incomplete minutes remain unknown. No request
errors, interpolation, model refit, cutoff change or orders.

Frozen next-observed-quote entry/stop/expiry, SL2ATR/H15/delay1, modeled cost0.10ATR.
All12 round6 model issuances and four same-availability clock references replayed.
Primary BOOST44 results: Boom SPIKE n88/PF1.043, DRIFT n115/PF0.636;
Crash SPIKE n100/PF0.612, DRIFT n126/PF0.798. All clock-comparison mean-difference
intervals cross zero. Best combined44 point PF1.396 has only59 paths.
M5-reference RIDGE19_SPIKE Boom PF2.038/n35 and RIDGE19_DRIFT Crash PF1.765/n48
have mean intervals crossing zero and remain development-rejected. They do not
satisfy PF>=1.5 with >=1,000 paths or the combined-timeframe requirement.

Independent stdlib audit:2,364,376 checks,24 sources,16 groups,3,083 paths
(3,077 completed+6 censored),0 errors,1,956 saved false safety values checked.
It verifies raw Decimal/grid/retry provenance, causal ATR, primary quote paths,
point summaries and closed-trade risk. ML/features/fits/scores and complete
clock discovery, bootstrap validity, secondary/sensitivity ledgers and matched
inference are outside scope. Checkpoint and active_symbols hashes captured at
audit time are explicitly distinguished from the preexisting frozen hashes.

Local Python suite:795 passed,1 existing FastAPI deprecation warning. This includes
238 new tick-study tests and73 pure tail-helper tests. At that round7 release, the proposed tick-tail helper had only synthetic checks.
The subsequently executed round8 pilot is described below.

This is adaptive execution diagnosis on known historical prices, not new
strategy-OOS or prospective evidence. The discontinuous12-day sample is too
small by construction for the1,000-path target. Actual fills, measured costs,
monetary profit and prospective paper remain NOT TESTED. All four flagsfalse.


## Fixed tail-tick mechanism pilot — executed, no payoff claim

See [Arabic report](spike_tick_tail_20261005/REPORT.ar.md),
[precalibration declaration](spike_tick_tail_20261005/declaration.json) and
[frozen protocol](SPIKE_TICK_TAIL_PROTOCOL.md). The five science files and all
round7 source/provenance hashes were pinned before detector fitting. This reuses
known historical tick prices; it is not fresh strategy validation.

First40% observed quote rows calibrate the median absolute consecutive1s log
return once. Events strictly exceed10 times that fixed scale in the symbol's
spike direction.691 Boom and684 Crash calibration events make both detectors
adequate under the declared minimum100 rule. Gap/initial prior age stays unknown;
the current event cannot enter its own age covariate. Exactly two bands:<600s
and>=600s. Development ends70%, with three40–50/50–60/60–70 validation segments.

| Symbol/segment | Younger events/exposure | Older events/exposure | Hazard ratio | CI95 |
|---|---|---|---:|---|
| BOOM600 dev40–70 |281/188,072|195/115,656|1.128|[1.061,1.234]|
| BOOM600 final30 |316/193,893|197/115,707|1.045|[0.987,1.137]|
| CRASH600 dev40–70 |351/206,273|202/101,870|1.165|[1.010,1.452]|
| CRASH600 final30 |269/189,009|203/116,330|1.226|[1.081,1.410]|

The four declared large-overdue>=1.5 hazard effects reject under this detector
and conditional observed-day inference:>=100events in both bands and all9999
paired bootstrap ratios defined. This does not establish independence, rule
out smaller effects or test a strategy. The intervals condition on fixed
calibration with only5/4discontinuous observed-day clusters; training-median
uncertainty is not regenerated. Conditional and unconditional tick drift
arithmetic, folds, unknown-age exclusions and event sizes are saved in results.

Full local test run890 passed, followed by55 final verifier tests after four
additional guard cases; current suite894 unique tests. Existing FastAPI warning
only. Broker execution, measured costs, PF, monetary profit and prospective
paper for this pilot remain NOT TESTED; no prior model is promoted. All four
execution/readiness flags remain false.

Independent stdlib pilot audit passed first/only attempt:6,685checks,24sources,
12segment summaries,0errors,200savedfalseflags,146inputhashes. It reconstructs
prefix median/events/prior ages/gaps/point day-band-drift identities/metrics and
saved policy arithmetic. Raw integrity is anchored to the round7 passed audit,
not renormalized again. Bootstrap regeneration/inference validity, calibration
uncertainty, unsaved runtime prior-mark table, model fitting, profit and broker
execution remain outside scope. [Audit](spike_tick_tail_20261005/independent_audit.json).


## Fixed tick-age/mark economic pilot — executed, no qualified model

See [Arabic report](spike_tick_age_payoff_20261005/REPORT.ar.md),
[frozen protocol](SPIKE_TICK_AGE_PAYOFF_PROTOCOL.md),
[selection](spike_tick_age_payoff_20261005/selection.json) and
[results](spike_tick_age_payoff_20261005/results.json). Round9 retains44 closed
H4/H1 + M15/M5/M1 inputs and adds exactly strict-prior tail-age log1p and last
native tail mark in RIDGE46. Both families and clock use the same common
available closed M5 opportunities. The prior00/30 cadence changes for every
family/reference; comparison with round6 is not an isolated new-feature effect.

Four expanding fits use the fixed first40% detector, training-only ridge0.1,
q75 positive cutoff and individually replayed, potentially overlapping targets.
Final70% training has2,313/2,341 completed labels per Boom/Crash direction;
these are not non-overlapping strategy completion counts. Selection froze
before new final30% payoff calculation. All development selections were
ineligible, with just five observed validation days and unknown payoff cases.

| Final30 model | RIDGE44 n/PF | RIDGE46 n/PF | RIDGE46 day-conditional PF CI95 |
|---|---|---|---|
| BOOM600 SPIKE |103 /0.908|102 /0.751|[0.439,1.028]|
| BOOM600 DRIFT |133 /0.708|136 /0.636|[0.441,0.910]|
| CRASH600 SPIKE |107 /0.724|110 /0.694|[0.386,1.250]|
| CRASH600 DRIFT |73 /0.898|71 /0.898|[0.538,1.599]|

All four46-minus44 point mean differences are negative and their paired
observed-day intervals cross zero. Final ledgers have no censored/missing-entry/
invalid outcomes, but just four discontinuous observed-day clusters. The
final age-band overlapping-label mean intervals all cross zero; development
unknown labels block whole-policy rejection even where known-subset means have
negative intervals. No discovery, independent-price-OOS, full-period profit
or readiness claim is made. No result justifies expanding this fixed policy
solely to accumulate completions. The pilot cannot meet1,000 completed
held-out strategy paths or60activeheldout days; labels cannot be pooled to
manufacture those requirements.

Full local Python before declaration:1,045 passed, one existing warning.
Independent round9 audit passed the second attempt:11,803,332 checks,24sources,
3,866 strategy paths,0errors;66,638 overlapping-label records include repeated
family artifacts, not unique independent trades.333 input hashes/2,260 false
safety values were checked. First attempt's six dense-grid versus observed-M1
count mismatches are preserved; only verifier logic/tests were repaired.
Frozen science/data/models/results are unchanged. Full current Python:1,111
passed; compilation passed. Scope covers saved matrices/scalers/scalar scores/
cutoffs, strict-prior features/common clocks, ATR, singleton/joint paths and
point/gate arithmetic. Full44 regeneration/ridge refit/bootstrap regeneration/
inferential validity/fresh OOS/broker profit remain excluded. [Passing audit](spike_tick_age_payoff_20261005/independent_audit.json). All four flags remain false. Monetary profit, actual broker fills,
measured costs and prospective paper are NOT TESTED.


## Strict-prior tick-path risk economic pilot — executed, no qualified model

See [Arabic report](spike_tick_path_risk_20261006/REPORT.ar.md),
[frozen protocol](SPIKE_TICK_PATH_RISK_PROTOCOL.md),
[selection](spike_tick_path_risk_20261006/selection.json) and
[results](spike_tick_path_risk_20261006/results.json). Round10 retains the
44 closed H4/H1 + M15/M5/M1 inputs and adds one native-adverse semivariance
feature over exactly 600 prior second increments, using the frozen round8
median. All 601 consecutive quotes end at t-1; gaps remain unknown. No current
or future quote is used. Both families and CLOCK share closed M5 availability.

Models use fixed training-only ridge/scaling/cutoffs and the unchanged tick
execution policy. Expanding fits are 40/50/60%, validated on the next 10%, with
70% final training. The 2,329 final training targets per symbol/direction are
identical for both families and overlap; they are not strategy evidence.
All development selections are ineligible. Prices and payoffs were already
known: this is adaptive research, not fresh price OOS or a new label holdout.

| Final30 model | CLOCK n/PF | RIDGE44 n/PF | RIDGE45 n/PF | RIDGE45 conditional PF CI95 |
|---|---|---|---|---|
| BOOM600 SPIKE | 260 / 0.974 | 102 / 0.880 | 105 / 0.758 | [0.460,1.049] |
| BOOM600 DRIFT | 288 / 0.782 | 133 / 0.715 | 132 / 0.761 | [0.487,1.520] |
| CRASH600 SPIKE | 264 / 0.731 | 107 / 0.703 | 121 / 0.763 | [0.336,1.389] |
| CRASH600 DRIFT | 285 / 1.011 | 80 / 1.003 | 83 / 1.033 | [0.678,1.790] |

All 45-minus44 paired observed-day mean intervals cross zero. No positive
absolute-and-incremental expansion condition passes. Final paths have known
outcomes but only four discontinuous observed-day clusters. No model qualifies
1,000 held-out completed paths, 60 active held-out days or PF>=1.5. Neither
pooling labels nor expanding this fixed policy solely for count cures this.

Independent audit passed the first actual attempt: 13,939,724 checks, 24 sources,
3,918 strategy paths and zero errors. It checked 507 hashes and 4,320 saved false
values. 66,736 overlapping-label records include repeated 44/45 family artifacts
and are not unique trades. Full local Python suite: 1,364 passed, one existing
warning; compilation passed. Scope covers independent causal risk reconstruction,
saved matrices/scalers/scalar predictions, closed timestamps, common clocks,
quote paths and point/gate arithmetic. Full44 regeneration/refit/bootstrap
regeneration/inferential validity/independence/fresh OOS/broker profit remain
excluded. [Passing audit](spike_tick_path_risk_20261006/independent_audit.json).
All four live flags remain false. Monetary profit, actual broker fills, measured
costs and prospective paper results remain NOT TESTED.
