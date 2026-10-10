# Fixed nonlinear learning of conditional filled-region returns

This is one new adaptive development comparison, declared before new scores.
The existing BOOST44 study used timed-entry returns; the existing region-target
study used ridge only. Test the missing nonlinear conditional-region target
combination. Do not change features, labels, clock, regions, source windows,
costs, side, exits, hyperparameters or thresholds after seeing this result.
These prices and prior outcomes were exposed. This is not fresh OOS research.

Always LIVE_TRADING=false, READY_FOR_LIVE=false, LIVE_ALLOWED=false,
OPENED_TRADES=false. No account, credentials, authentication or orders. All work
is offline. CFD mapping/fills, measured historical costs, cash profit and
prospective paper remain NOT TESTED. A favorable historical result cannot
qualify real trading or complete the requested stable-profit objective.

## Exact inherited data and target population

Pin region_reward_20261008 declaration
5a5cd147eefc7d12ac83c8ef839c931292c5fb338e07a1f083001c893c5196e5,
result 6e470b505d4f32ffb28600cd136ae886001f35e34631663553d8807b114d8115,
and passed audit a6194dd9178d2a14a7dee4d317429a1aaaad28d3d3ce4c195d9378f8e588f25b.
Retain the same 722 public daily tick sources, 62,380,130 original quotes and
670 missing seconds. No recollection, interpolation, detector recalculation or
feature-formula revision. Use the exact pinned per-fold RAW44/HYBRID44 caches
with closed H4/H1/M15/M5/M1 inputs. HYBRID44 retains 43 transformed features
and the same raw closed-M5 large-bar age. Require exact saved common validity
and raw execution ATR. Transformed quotes or ATR never supply execution values.

Read the eight existing full CLOCK training label/event tables. Reconstruct
the expected ordered shared clock and label categories from pinned events,
then require exact equality to saved labels, including planned_end. Only
completed, filled, uncensored finite netR labels train either model. Known
no-fills, invalidations and occupancy skips retain NaN reward and distinct
categories; unknown paths remain unknown. Never invent zero-return trades.
Require at least 1,000 shared completed training labels. Preserve the full
label population and hashes; training prefixes overlap and are not independent
held-out trades. No evaluation fill/outcome is an input or selection filter.

## Learner, chronology and references

Use the existing deterministic histogram residual booster unchanged:
100 trees, learning_rate=0.05, max_depth=3, min_leaf=200, n_bins=16,
leaf_regularization=20, quantile=0.75. No scaling, clipping, optimization,
early stopping, new features, random search or cutoff search. Bins and
q75 cutoff use completed training rows only; require score>=threshold and
score>0. Scores are uncalibrated continuous netR estimates. Save every tree,
bin, parameter, cutoff and model/target/matrix hash for both paired learners.

Train independently on first 40/50/60% of the older interval and evaluate
the next 10% each. Fit first 70% once for older last 30% and later 180 days;
no later refit. Older interval is [2025-10-09T11:08Z,2026-04-07T11:08Z),
later is [2026-04-07T11:08Z,2026-10-04T11:08Z). Apply the same 31-minute
planned purge at each boundary and UTC00/30 issuance from closed M5 candles.

Five streams, all using the same geometric application:

- CLOCK: immutable ungated shared-clock reference.
- RAW_REGION and HYBRID_REGION: immutable parent ridge models and results.
- RAW_BOOST_REGION and HYBRID_BOOST_REGION: the two new fixed learners.

Read the exact parent reference rows/artifacts rather than refit or redraw
their statistics. Remove legacy PF1.5 qualification fields from reference
copies, retain source identity and all numerical evidence. Compare
RAW_BOOST_REGION with CLOCK and RAW_REGION. Compare HYBRID_BOOST_REGION
with CLOCK, HYBRID_REGION and RAW_BOOST_REGION. Declare these references
before measurement. This is a learner comparison in conditional behavior-policy
data, not an identified causal policy effect.

## Actual application and inference

Issue at original closed price p and raw ATR A. Boom buy region
[p-0.55A,p-0.45A]; Crash sell region [p+0.45A,p+0.55A]. Invalidate another
0.25A beyond its adverse edge. Monitor immediately, activate after one minute,
require an actually observed quote inside the band strictly after activation
and within 15 minutes; enter on the exact next-second original quote. No
fabricated crossing. Stop 2A from actual entry, hold 15 minutes from touch,
stop priority at expiry then exit on exact next-second quote. Losses uncapped.
Modeled total cost=0.10A; one pending/open exposure per stream; unknowns reserve
full planned occupancy. Replay each new selected stream on original ticks;
filtering a CLOCK ledger is invalid because selection changes later occupancy.

Emit all 60 cells: two symbols x five streams x wf1/wf2/wf3/their union/
final_test/later180. The fold union concatenates separately purged ledgers;
it is not extra data. Preserve all models, selected signals, full dispositions
and ledgers. Report completed/unknown/nonfill/skips, active days, net/gross
meanR, net PF, daily/weekly CI95, reference differences, holding time,
chronological thirds, doubled-cost mean and illustrative closed-trade drawdown.
Use the inherited 9,999 paired daily and circular seven-day bootstrap replicates,
seed 20261008. Undefined values remain undefined. Never pool to meet sample size.

## Revised target and unchanged evidence gates

The user's latest target is positive stable net profit even below PF1.1.
Replace only the prior PF>=1.5 point criterion with finite PF>1 AND finite
mean_net_R>0. Keep all development, no-unknown, sample, uncertainty, stability,
cost and risk gates. No PF1.1 floor is introduced.

Development: each fold>=100 completed, positive mean and PF>1; union>=500
completed, >=30 active days and positive conservative selection score; no
unknown candidate/required-reference outcomes. Held-out: >=1,000 completed
one-open paths and >=60 active days per symbol/model/period; no unknowns;
mean and reference advantage lower CI>0 daily/weekly; PF lower CI>1 daily/weekly;
all three chronological thirds>=200 completed and positive mean; illustrative
closed-trade drawdown<=10%, no illustrative ruin, positive doubled-cost mean.
Conjunction p is max across candidate and all required-reference day/week tests.
Joint Holm family is exactly eight new held-out cells (2x2x2), threshold<0.05.

Known before measurement: CLOCK contains unknown paths for each symbol:
wf2=3, wf3=2, union=5, final_test=3, later180=8. Thus unchanged no-unknown
reference/development gates already preclude historical qualification on these
periods. This does not preclude learning from a fixed estimator comparison;
it must not be concealed or repaired by dropping these opportunities. Always
QUALIFIED=false and fresh_out_of_sample=false.

Freeze protocol, new adapter/runner/gate/tests and passing synthetic-test log,
all reused engine files, prior evidence identities and runtime before any new
historical score. Exclusive outputs prevent overwrites or silent restarts.
Independent review should verify routed-tree training arithmetic, training
cutoff/identity, issuance, original tick paths, inference and revised gates;
state explicitly any omitted optimal-split search, feature reconstruction or
source-acquisition attestation. Audit PASS describes only that scope, never
the requested profitable strategy. No post-result rescue or winners-only report.
