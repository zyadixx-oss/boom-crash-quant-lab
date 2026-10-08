# Paired native-tail representation and advance price regions

This distinct hypothesis asks whether old native spikes dominating rolling
indicators obscure useful context. The original44closed H4/H1/M15/M5/M1 inputs
are compared with those same44inputs after the specified tail removal. This is
an empirical hypothesis, not an assumed edge. It completes the previously
prepared representation machinery and tests actual pre-issued regions toward
the user's native Boom-rise/Crash-fall objective. No real trading is allowed.

Primary application: immutable adverse-retracement regions under the two learned
score gates. Secondary application: the previously proposed fixed timed-entry
representation ablation. Both are fixed before new historical features or
labels are computed. No school, indicator or transformed synthetic price is
declared profitable on theoretical grounds.

## Sources and chronology

Use exactly the722independently audited public clean native tick sources,
62,380,130quotes from9October2025 through4October2026. The670missing seconds stay
missing. Freeze source-audit lineage, all research engines, runner, adapter,
tests and this protocol before measurement. Full raw data remain local/ignored;
published SHA256 pins support reproduction with those inputs.

Older interval[2025-10-09T11:08Z,2026-04-07T11:08Z), later interval
[2026-04-07T11:08Z,2026-10-04T11:08Z). Refits use first40/50/60% of older history
and evaluate the following10%; the final70%fit evaluates older last30% and later
180days unchanged. No in-sample fitted prediction is called validation. The
three validation ledgers are concatenated only after separate purges/execution.
No models, symbols or held-out periods are pooled to attain evidence gates.
This history and previous returns are known: all results are adaptive historical
research, never fresh temporal OOS or prospective paper.

## Fold-specific feature preparation

Fit the unchanged native-tail detector only on original consecutive quote pairs
wholly within each training interval. Threshold=10×training median absolute log
increment. It remains frozen for that fold. The old detector fitted through
June2026 is not reused. Future prices cannot enter detector/scaler/model fitting.
Here future means after training end. Training features are rebuilt under their
training-fitted preprocessing and are not historical real-time decisions;
evaluation uses closed past quotes and the already frozen detector exclusively.

Rebuild each fold's feature chain from the planned UTC source start. For every
maximal exact-one-second run, Qstarts1, logQstarts0; replace only a detected
native-tail increment by zero and retain all others, using fixed Neumaier
accumulation. Gaps reset both raw and transformed rolling histories. The chain
is a specified feature representation, not an identified price without spikes.
Its ordinary drift component inside a detected increment is also removed.

Stream arbitrary quote chunks, with at most59pending seconds; produce only
exact60second/same-run UTC M1bars. Reindex to the entire planned minute grid,
including empty leading/trailing minutes. Keep raw and transformed minute
populations identical. Stop feature construction at that fold's evaluation end.
Existing numerically stable equivalent Bollinger arithmetic is unchanged.

Preserve all44feature definitions, including unknown large-bar age. Do not fill
unknown age with zero, drop a feature, loosen the common mask, change detector
scale, reanchor around outcomes, or substitute a raw-only fallback to obtain
more opportunities. Record all raw/transformed/common availability counts and
unknown large-bar-age counts. Insufficient common availability is a rejection,
not an invitation to modify this experiment.

## Fixed learning

Closed-M5 opening+5minutes is issue time, restricted to UTC00/30. At each issue,
raw/transformed44must both be intrinsically valid based only on past prices.
Use the common valid clock for both learners and their CLOCK reference. Preserve
all other clock rows in availability artifacts. Planned purge31minutes applies
to all training and evaluation paths before looking at outcomes.

Training target is the common-clock original-quote native timed-entry netR:
delay1minute, exact next-second entry, SL2rawM5ATR, hold15minutes from nominal
entry time, next-second stop/expiry quote, assumed total cost0.10rawATR. Every
clock opportunity stays in the training label table, including missing-entry
and path unknowns; unknowns are NaN, never zero. Fit only when>=1000completed
common training labels exist. Training labels are not counted as held-out trades.

Use the existing paired ridge implementation: independent training-only
standardizers, same original targets, SSEmean penalty0.1, trainingq75 cutoff
with positive score floor. No clipping, alternative learner, target, score
threshold, side or horizon search. If either common fit is unavailable, both
models remain NOT_FIT; issue no model regions/signals. CLOCK remains visible.
Evaluation issuance takes no labels and performs no fitting.

## Primary REGION and secondary TIMED execution

REGION: take each fixed model's selected opportunity and the ungated CLOCK
reference. The issue price p is the last original closed-M5 price, rawATR=A.
Boom band[p−.55A,p−.45A]; Crash band[p+.45A,p+.55A]. Invalidation is another.25A
outside the adverse boundary. These match the old geometric band's constants;
no native-body agreement filter is added. The learned44score supplies the
combined-frame context; no post-result discretionary CRT/Fibonacci label.

The region is immutable at issue. Monitor invalidation immediately, activate
after1minute, require an actual quote inside the band strictly after activation
and no later than issue+15minutes. Fill at exact next second even if price then
leaves the band. No fabricated crossing/interpolated touch. SL2rawATR measured
from actual entry; timeout is15minutes after touch. A stop at expiry wins priority;
exit exact next second, with uncapped gap loss and0.10rawATR assumed total cost.
Untouched/invalidated regions stay in disposition tables. Unknown before touch
or after entry stays unknown and reserves the full maximum planned exposure.
One pending region OR open position per independent model stream; no stacking.

TIMED: replay the same selected native signals on the unchanged strict-next-tick
fixed15minute convention used for training. No region touch/invalidation. The
paired comparison separates representation scoring from retracement filtering;
it cannot identify the causal effect of arbitrary geometry. Native direction
only; this does not establish superiority to opposite-direction trading.

Execution, labels, stops and costs always use original quote prices and raw ATR.
Transformed ATR supplies feature normalization only. No costs, delays, stops,
holds, waiting time or region width may be changed after results.

## Inference and rejection gates

Emit all72metric cells:2symbols×2endpoints×3streams×6partitions
(wf1,wf2,wf3,their union,final_test,later180). Save issuance, all region dispositions,
filled/censored ledgers, training labels, model coefficients/scalers/cutoffs,
detectors and availability. Store full paired feature tables locally with hashes.

Report completed/censored/missing entry/untouched/invalidated/overlap counts,
active days, net/gross meanR, netPF, median holding time, day/week mean and PF
CI95, paired CLOCK advantage, transformed-minus-RAW44advantage, chronological
thirds, doubled assumed cost and closed-trade drawdown illustration. Day/weekly
paired bootstrap9999replicates, seed20261008. Undefined denominators remain
explicit and cannot qualify a model. Profit results are not directional-spike
classification accuracy or unique-event recall.

One joint16comparison held-out family:2symbols×2models×2endpoints×2held-out
periods. RAW44p is the maximum of day/week CLOCK comparisons. TRANSFORMED44p is
the maximum of day/week comparisons against both CLOCK and RAW44. Holm adjusts
all16. No selected best outcome or posthoc fallback.

Retain the existing zone development/economic gates: combined walk-forward
>=500completed,>=30active days, positive conservative selection score, no unknown
candidate/control outcomes; each fold>=100completed, mean>0, PF>1. Held-out
PF>=1.5,>=1000completed per model/symbol,>=60active days, no unknowns, day/week
PF lower CI>1, positive day/week net-mean and control-advantage lower bounds,
Holm<.05, all thirds>=200positive-mean paths, closed-trade drawdown<=10%, no
illustrative ruin, doubled-cost mean>0. Never pool to meet these gates.

All QUALIFIED fields remainfalse because reused history and actual execution
cannot prove the full user goal. Even favorable historical results require
independent arithmetic/source review and genuinely new prospective evidence.
Broker execution, measured CFD costs, cash profit and prospective paper are
NOT TESTED. LIVE_TRADING=false, READY_FOR_LIVE=false, LIVE_ALLOWED=false,
OPENED_TRADES=false.

Motivation only: [Deriv product-team discussion of old spikes in indicators](https://experts.deriv.com/insights/boom-and-crash-the-drift-the-spike-and-what-a-spike-does-to-your-indicators).
That source supplies no validated profitable rule.
