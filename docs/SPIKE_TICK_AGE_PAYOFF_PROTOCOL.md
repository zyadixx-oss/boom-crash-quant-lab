# Proposed round9: fixed M5-clock economic feasibility pilot

This is adaptive research on the same twelve discontinuous tick days per
BOOM600/CRASH600 from rounds7–8. The modest Crash tail-age association provides
an economic hypothesis; it is not a profitable signal. The previous event mark
has not been tested as a predictor. No new price-OOS, generator-independence,
broker-fill, measured-cost or cash-profit claim is permitted.

Freeze this protocol, the new runner and tests, both pure helper/test pairs,
and inherited dependency/source/result hashes before any new payoff labels,
model fits or comparisons. Declaration is metadata/hash-only. No raw tick,
M1, detector or prior frozen result/source/code may be edited. All four flags
LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES remain false.

## One experiment, shared availability

Retain the exact44 closed-candle H4/H1 + M15/M5/M1 inputs and existing formulas.
Use one fixed decision cadence: every closed UTC M5 candle, minute divisible
by5. This changes the old minute00/30 contract for both models and clock.
Absolute improvement versus round6 is not an isolated tick-feature effect.
No cadence, side, exit, age boundary, detector or probability threshold sweep.

Compare RIDGE44 with RIDGE46 on the same common-available decisions. The latter
adds exactly:

- tail_age_log1p = log1p(pre_event_age_seconds /600)
- prior_tail_mark = directed log return of the most recent tail event known
  through the prior second, in the symbol's spike direction

Join only at signal_time=M5 opening+5minutes. Current event/return/quote does
not enter either added feature. No nearest/forward/interpolated join. Unknown
age/mark or invalid44-frame inputs make the decision unavailable to BOTH models
and the clock. Preserve invalid interior rows; no carry through a missing frame.
Use the round8 first40%-only detector unchanged, including its exact10× scale
and symbol direction. Do not refit it in later folds or replace it after results.

Both SPIKE and DRIFT directions on both symbols remain in the family. The
side is +1/-1 for Boom/Crash SPIKE, negated for DRIFT; age and mark always refer
to the native symbol's tail-event direction. No post-result favorable side or
symbol selection.

## Chronology and labels

Use exact round8 observed-row cutoff timestamps for40/50/60/70/100%; this is
not a split of the intervening calendar. Last30% remains known historical
research, although its NEW five-minute tick-payoff labels are withheld until
selection is frozen. Three expanding fits use first40/50/60% and validate the
next10%; final fit uses first70%. M1 context is clipped at the relevant boundary
before building closed-frame features. The immutable day decoder may parse the whole boundary-day source before
immediate prefix clipping; that does not compute suffix features or payoffs.
Existing M1 source provides earlier causal warm-up; neither future closes nor unobserved tick-day intervals are
fabricated. Detector state continues across a cutoff if quotes are consecutive,
but resets across a missing second or discontinuous sampled day.

Apply frozen round7 primary TickExitConfig: entry nominally signal+1minute,
first observed quote strictly after that time, stop2ATR, hold15minutes anchored
to nominal entry, no profit cap, total modeled cost0.10ATR, one-second gap
allowance and stop successor latency1. A stop crossed at the deadline has
priority; primary fill is the next observed quote. These are indicative
quote paths, not actual broker orders/fills. Preserve missing-entry/censored
states and unknown returns; never encode missing labels as zero.

For TRAINING TARGETS only, replay each available signal individually, so labels
may overlap. Reuse small immutable quote windows from nominal entry through the
nominal deadline+1second, with each signal replayed alone in the unchanged
engine. Apply effective end=min(scheduled day end, current partition end) and
31-minute planned purge; an early observed stop cannot rescue a purged label.
Training labels and their counts are not a one-open strategy ledger, a PF or
the user's1,000-trade evidence. Flag overlapping-label dependence explicitly.

For EVERY model/clock strategy ledger, replay all its selected signals jointly
per symbol/direction/day-partition with the frozen one-open rule. Known exits
release occupancy at actual quote exit; censored/missing-entry paths reserve
planned occupancy. Disjoint days start separate state, and every day receives
the same31-minute purge. Only completed non-overlapping model-ledger paths may
count toward payoff sample gates. Freeze source arrays; never reconstruct fills
from candle extremes or merge individually replayed training labels into a strategy PF.

## Fixed ridge and development selection

Each fit requires at least1,000 completed finite individually replayed training labels.
If unavailable, save NOT TESTED for that fit without a lower floor or another
cadence. Retain ridge mean-SSE penalty0.1, training-only mean/std, intercept,
training q75 score cutoff and positive score floor. Unknown/constant handling
must remain the inherited estimator's behavior. No train/validation tuning.
Hash ordered feature/target/timestamp training matrices and serialized models.
Before final labels, write an exclusive-create selection artifact containing
development eligibility, all serialized44/46 models/scalers/cutoffs and training
hashes; hash it and refuse changed bytes or existing final output at evaluation.

Both44/46 versions use exactly the same individually replayed training rows/labels and opportunity
clock. Primary incremental comparison is46 versus44, not versus mismatched
availability. The clock reference uses that same availability and one-open
execution. Do not pick an estimator based on final results.

Retain inherited development rules: at least500 completed pooled validation
paths,30 active days, no censored/invalid outcomes, positive lower selection
score, and all three folds each at least100 completions with positive meanR
and PF>1. With the current discontinuous sample,30 active validation days are
impossible; therefore all pilot candidates remain development-ineligible even
if later point values look favorable. Still report all fixed diagnostic models
whose training labels were adequate, with selection frozen before final labels.
No insufficient fit or rejected development can be promoted.

## Economic diagnostic and uncertainty

Before model promotion is even considered, report fixed-policy individually replayed
label netR by the existing two age bands(<600,>=600) in pooled40–70 and final30,
for every symbol/direction. Record counts, missing labels and overlap status.
The mean includes non-event drift, costs, stops and overshoot; an increased
jump hazard alone does not imply a positive mean netR.

Use9,999 paired iid draws of the observed scheduled UTC-day units inside each
partition, seed20261005. Keep zero-band-exposure days inside the partition;
never pad with all12source days or unsampled calendar dates. Report observed
cluster count and undefined draws. A band with at least100 known labels, all
9,999 means defined, no missing-entry/censored/invalid payoff among planned
eligible signals in that band, and upper95% mean<=0 can reject positive expected
OVERLAPPING-LABEL payoff under this fixed diagnostic. Count eligible planned
signals before realized outcomes; preserve missing returns and report their
band counts. With any missing payoff, report the known-completed-subset mean
but block rejection of the whole eligible diagnostic. An inconclusive interval
means insufficient evidence, not a strategy pass. This does not reject every
interaction/policy/occupancy rule inside46. No positive-profit discovery claim
is made from these descriptive conditional intervals.

Report model-ledger n, censoring, PF, meanR, day-conditional CI95, active days,
and paired46-minus44/clock mean differences. All means/PF and paired differences
use shared observed-source UTC-day draws and pooled return/count/gain/loss
sums, including days with no available signals inside that partition. Report
undefined mean/PF/difference draws; conditional intervals based on remaining
finite draws cannot be used as complete bounded rejection or promotion when
any relevant draw is undefined. Saved individually replayed labels and
non-overlapping strategy ledgers must be visibly distinct. Sparse discontinuous
days do not support inherited contiguous-week inference; do not pad calendar
weeks to manufacture it. All candidate/live readiness flags remain false.

## Gate for a larger study and the actual goal

The pilot does NOT meet the user target PF>=1.5 with>=1,000 completed held-out
paths per model/symbol,>=60 active held-out days, robust positive day/week
bounds, clock/reference advantage, chronological stability, multiplicity,
drawdown/cost checks, and separately measured execution/prospective evidence.
Do not weaken those inherited gates to pass this twelve-day pilot.

A later immutable full-period data acquisition must have an economic reason
beyond incidence and a feasible sample sized from DEVELOPMENT completion
rates BEFORE held-out outcomes. A180-day70/30 split contains only54 final days
and cannot meet>=60active-day evidence. At least200 usable source days gives
60 calendar held-out days before inactive/censored margin; larger volume may
be required. Cadence and nominal q75 issuance never guarantee1,000 completions.
Do not add dates or change exits/thresholds until a result passes. Any expanded
study needs a separate declaration and honest adaptive-history/prospective
boundary. Actual monetary profit and prospective paper remain NOT TESTED.
