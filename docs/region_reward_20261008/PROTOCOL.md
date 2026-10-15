# Conditional filled-region reward versus the frozen timed reward

This distinct adaptive hypothesis aligns training with the user's pre-issued
Boom-buy/Crash-sell region application. The preceding paired representation study
trained on timed-entry netR and then gated retracement regions. Here region
learning trains on the completed returns of those actual fixed regions. This
changes the reward definition and the natural conditional training population;
it is not an isolated causal effect of substituting numeric target values.
It is not a repair or parameter rescue of a previous frozen experiment.

No result is assumed profitable. Keep LIVE_TRADING=false, READY_FOR_LIVE=false,
LIVE_ALLOWED=false, OPENED_TRADES=false. No account, authentication or order API.
CFD execution, measured historical costs, cash profit and prospective paper
remain NOT TESTED. Historical quote-proxy arithmetic is the test endpoint.

## Immutable sources and cached features

Reuse exactly the preceding hybrid declaration6ade87b7207754cc21468d68ab315c3f66f5b84bedc2e274907b453759e03af5
and resultfb345c068a899157e15ec6393a72a7eeb5d78f932b7f7b7f0fac0640823b4abd.
Require its independent audit PASS. Freeze those file hashes, all input artifact
pins, research engines, new adapter/runner/tests, this protocol and passing
premeasurement log before new historical labels or scores are computed.

Use the same722audited daily native quote files and62,380,130quotes from
9October2025 through4October2026. Preserve670missing seconds. Decode pinned bytes
with pandas; enforce exact schema/count, original positive prices and UTC daily
chronology. Do not interpolate. Full raw data and feature caches stay local with
SHA256 pins. Prior independent source reconstruction remains authoritative.

No feature, detector, source window or representation is recalculated or tuned.
Read exact prior per-fold RAW44/HYBRID44 caches: unchanged closed H4/H1/M15/M5/M1.
HYBRID44retains43transformed features plus the same observed rawM5large-bar age
used by RAW44. Each cache's detector was fitted on that fold's training prefix
only. Its histories reset at missing seconds. The original all-transformed44
rejection is unchanged and cannot be replaced by this experiment.

Retain full M5 feature grids and intrinsic validity, including unknowns. Validate
scheduled availability against the prior saved UTC00/30 clock, including exact
raw execution ATR and transformed feature ATR. Only scheduled rows need run ids;
unscheduled rows receive no invented run id. Common availability always equals
raw-valid AND hybrid-valid. Transformed prices/ATR never supply execution prices,
risk or costs. Issue time is the closedM5opening+5minutes.

## Chronology and shared opportunity population

Older interval[2025-10-09T11:08Z,2026-04-07T11:08Z), later interval
[2026-04-07T11:08Z,2026-10-04T11:08Z). Fit first40/50/60%older and evaluate the
next10%each. Fit first70%once and evaluate older last30%and later180days without
refitting. Apply31minute planned purge to training and every evaluation window.
The same common valid scheduled clock supplies all five streams. No historical
in-sample prediction is called validation. Training prefixes overlap and are
never counted as independent held-out trades.

These dates, earlier returns and model outcomes are known. All results remain
adaptive historical research, never freshOOS or prospective evidence. Neither a
new declaration nor withholding dates within reused history restores freshness.

## Conditional training target

Replay the common training CLOCK under exactly the unchanged geometric region
and original-quote rules below, including one pending/open exposure. Preserve
every scheduled issue and its actual disposition. The training label table must
equal this exact clock in order, with planned_end=issue+30minutes+1second.

Only status=completed, filled, uncensored paths carry finite original-quote
netR and enter either regression. Invalidated/expired regions are known no-fills
with no regression reward; they are not zero-return trades. Overlap skips remain
a separate behavior-policy category. Waiting/entry/path gaps remain unknown.
Do not substitute zero for any excluded outcome, skip row identities or drop
unknown opportunities from the saved label population. Category counts must sum
to the entire clock. A future fill/outcome is used only as a training label,
never as an issuance feature or evaluation selection mask.

The conditional training population comes from the ungated CLOCK behavior.
Filtering earlier regions can alter later occupancy. The sparse31minute/30minute
boundary interaction therefore stays explicit; this study does not identify a
causal policy effect or assume CLOCK's missing labels would be filled under a
different policy. Actual selected evaluation streams are replayed independently.

Require>=1000completed common training region paths. If insufficient, neither
new model fits or issues signals; retain references. Fit RAW_REGION/HYBRID_REGION
on exactly the same completed timestamps and targets, with independent
training-only standardizers. Use the existing mean-SSE ridge penalty0.1,
unclipped target/features, q75of completed-training predictions and positive
score floor. No learner, feature, side, score threshold or parameter search.
Save matrix/target/model hashes, coefficients, scalers and cutoff populations.
Scores are continuous netR predictions, not calibrated probabilities.

## Fixed references and application

Five separately replayed streams:

- CLOCK: common valid opportunities with no model gate.
- RAW_TIMED/HYBRID_TIMED: exact frozen prior timed-target paired models for each
  fold. Do not refit, recalibrate, adjust thresholds or change feature inputs.
- RAW_REGION/HYBRID_REGION: the newly fitted conditional region-target pair.

All five issue the same fixed geometric region rule. Issue p is the original
last closedM5quote, Athe rawM5ATR14. Boom[p−.55A,p−.45A], Crash[p+.45A,p+.55A];
invalidation another.25Aoutside the adverse edge. No extra native-body filter.
Bounds, rawATR, invalidation and side are immutable when the region is issued.

Monitor invalidation immediately; activate after1minute; require an actual quote
inside the region strictly after activation and by issue+15minutes. Fill the
exact next-second original quote even if it leaves the band. No interpolated
crossing/touch. Stop2rawATRfrom actual entry; hold15minutes from touch. Stop at
expiry has priority, then exact next-second quote exit. Losses are not capped.
Assumed total cost.10rawATR. Preserve untouched/invalidated/censored/overlap
dispositions. Unknowns reserve full planned occupancy; one pending region or
open position per stream. No stacking. No opposite-native-direction rescue.

## Measurement, inference and decision

Emit all60cells:2symbols×5streams×6parts(wf1,wf2,wf3,their union,final_test,later180).
Save all training events/labels/models and evaluation signals/events/ledgers.
The union concatenates separately purged/replayed fold ledgers and must include
every declared target comparison; no missing union-reference field is accepted.

Report completed/unknown/no-fill/skips, active days, net/gross meanR, netPF,
day/weekPF CI95, mean/control advantage CI95, holding time, chronological thirds,
doubled modeled cost and illustrative closed-trade drawdown. Use9999replicates,
seed20261008: same-day paired multinomial and circular7day block inference.
Undefined statistics stay undefined. Do not pool symbols/models/periods to pass.

RAW_REGIONmust beat CLOCKandRAW_TIMED; HYBRID_REGIONmust beat CLOCK,
HYBRID_TIMEDandRAW_REGION. The conjunction p is max of mean/advantage day/week
pvalues over all required references. Holm jointly corrects the eight new
held-out tests:2symbols×2newmodels×2held-out periods. References are descriptive,
not extra winning candidates or replacement definitions after the result.

Retain existing development gates: each validation fold>=100completed,
mean>0,PF>1; union>=500completed,>=30active days, positive conservative selection
score, no unknown candidate/control outcomes. Held-out PF>=1.5,
>=1000completed one-open paths per model/symbol,>=60active days, no unknowns,
positive mean/control advantage lower CI, PF lower CI>1 daily/weekly, Holm<.05,
all chronological thirds>=200positive-mean paths, drawdown<=10%, no illustrative
ruin and doubled-cost positive mean. Reused history always QUALIFIED=false.
No post-result target/sample/horizon/cost/stop/geometry/threshold rescue.

An independent audit must reconstruct the original training region paths,
conditional label/category population, ridge arithmetic, held-out selection,
original-quote regions/returns, statistics and Holm. Compare unchanged references
to the previous saved economic paths. Explicitly report any unverified scope;
an audit passing a narrower contract cannot certify the whole profit objective.
