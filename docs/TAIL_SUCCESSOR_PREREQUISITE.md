# Tail-successor feed-morphology prerequisite — source-only proposal

Status: historical calculation for this response is NOT TESTED. This document
specifies a prerequisite measurement; it does not declare a profit-seeking
round, fit a strategy, or establish the user's PF>=1.5/large-sample objective.
All four flags remain false: LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED,
OPENED_TRADES. No account access or order placement.

## Why measure this particular response

The existing44 closed H4/H1 + M15/M5/M1 inputs, nonlinear interactions,
tick age/previous-event mark, prior adverse variation and one-minute reward
studies did not qualify. Their closed-M5/UTC00-or30 clocks and one-minute
entry delay do not directly characterize the first increment remaining just
after an observed tail event. No inspected result currently establishes
multi-tick continuation or post-event economic advantage.

The public product description calls spikes discontinuities followed by drift;
this is not a published formal transition kernel. A source-level observation
can resolve whether the public feed exhibits the specified successor response,
without assuming that consecutive tail returns belong to one physical jump.
Primary context: [Deriv product explanation](https://experts.deriv.com/insights/boom-and-crash-the-drift-the-spike-and-what-a-spike-does-to-your-indicators).
The question is narrow enough to falsify without changing a learner or exit
until a favorable statistic appears. A positive result permits investigation
only; it is not a net-return, PF, executable-fill or profitable-strategy result.

## Fixed inputs and single primary response

Use exactly the existing24 round7 normalized sources:12 scheduled UTC days
per BOOM600/CRASH600. Preserve all18 missing seconds. Pin round7 sources and
passed audit, round8 detector/calibration/result/audit, and every new helper,
runner, test and this protocol before decoding any historical quote for this
response. No source rewrite, detector refit, extra dates, horizon/latency grid,
frame filter, new learner or payoff engine.

With native side s=+1 for Boom and-1 for Crash and the exact frozen round8
first40% median m, the observed event at t is:

    E_t = 1{s * log(P_t / P_(t-1)) > 10*m}
    Y_t = s * log(P_(t+2) / P_(t+1))

All four quotes must be exact consecutive seconds. P_(t+1) is the first
strict-successor entry proxy; Y is the first increment remaining after that
one-quote latency. Neither the observed event jump nor t-to-(t+1) movement
enters Y. This is a feed response, not measured network/server/order latency.
No horizon other than this one increment enters the primary measurement.

## Chronology, availability and reference

Retain round8 exact observed-row cutoffs. Primary descriptive partition is
40–70%; fixed descriptive repetition is70–100%. Also report the existing
40–50/50–60/60–70 subdivisions. First40% fitted the detector and supplies no
causal-evaluation claim. Every price/payoff date has already been exposed in
prior research; these are not fresh price OOS or prospective observations.

Enumerate every detector-eligible anchor using only current/past quotes.
Require the planned t+2 endpoint strictly inside its source day and partition;
count excluded boundary anchors/events separately. Inside an eligible window,
a missing t+1 or t+2 produces UNKNOWN, never a later substitute or interpolation.
Keep unknown event and reference outcomes in denominators/audit counts. Do not
choose anchors from future-knownness or omit censored responses silently.

Reference: all detector-eligible anchors, INCLUDING events, under the same
endpoint/availability rule. For each observed UTC day d calculate its known
reference mean Ybar_d. Weight daily reference means by that day's eligible
event-anchor count. Report event mean and event-minus-day-matched-reference
separately. If unknown responses prevent a complete cohort, the known-subset
point estimates may be shown explicitly, but a complete-population bounded
conclusion is prohibited. Do not call this a randomized causal effect.

Keep every event, including consecutive events and overlapping windows. Do not
select first/biggest/cleanest run events. Report overlap counts; events and
reference anchors are dependent and cannot count as completed strategy trades.

## Fixed uncertainty and interpretation

Use9999 shared resamples of observed UTC-day units within each partition,
seed20261007. Resample day sums/counts and recompute the day-matched reference
weighting in each draw. Retain observed zero-event days; never add unsampled
calendar dates. Report actual observed-day cluster count, finite/undefined
mean/difference draws and conditional percentile CI95. The five/four observed
clusters in the pooled development/final partitions are sparse; no independent
tick, contiguous-week, global-discovery or physical-event-identity claim.

Conditional evidence against THIS specified positive successor excess requires
at least100 eligible event anchors, complete event/reference responses, all9999
differences defined and the upper descriptive CI95 for excess<=0. Otherwise
report insufficient evidence for that bounded rejection. A positive interval
or point estimate cannot promote a strategy, imply cost coverage or establish
PF. No new latency/horizon/event-run/direction variant may replace a negative
or uncertain response after inspection.

The optional bid/ask in the CURRENT public stream does not prove executable
MT5 CFD pricing and must not be retroactively inserted in these historical
quote files. Historical broker bid/ask, commissions/financing, actual latency,
cash profit and prospective paper performance remain NOT TESTED. Combined
H4/H1/M15/M5/M1 strategy development would require its own subsequent causal
hypothesis and frozen economic protocol if this prerequisite justifies it.

Before any calculation: implement and verify the helper/runner on synthetic
causality, event exclusion, exact-lag, missing/boundary, exposure weighting,
overlap and undefined-resample cases; then exclusively save a declaration and
source hashes. No historical execution is claimed by the presence of this file.
