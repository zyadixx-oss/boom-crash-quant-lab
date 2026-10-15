# Fixed pre-issued zone study, 8 October 2026

This experiment directly tests the user's requested regions before native
Boom rises / Crash falls. No new region, feature, touch or economic outcome
has been computed on historical prices before freezing this protocol and code.
Earlier M1 results already exposed the calendar: this is adaptive retrospective
research, not fresh OOS. The separate transformed-price experiment remains
preparatory and is not silently substituted for this zone objective.

## Data and rules

Use the independently audited 722 original public tick CSVs, BOOM600 then
CRASH600, every UTC day 9 October 2025 through 4 October 2026. Audit SHA256
`34471138144552407390938f1ef58d7513d7fe556074265b674cabfb9245116a`
reports 62,380,130 observed quotes and 670 missing seconds. Preserve every gap;
all minute populations and higher-frame contexts require exact completeness.
Raw wire provenance remains available locally; no accounts or authenticated APIs.
Boom500's initial priority and original CRT comparisons were executed in earlier
rounds. This new zone experiment uses the available complete600 tick acquisition;
no claim of a new500 tick replication is made.

The precise immutable geometry and shared H4/H1/M15/M5/M1 context are specified
in DESIGN_REVIEW.md and executable zone_rules.py. Exactly three hypotheses:
CRT_RETEST, FIB_RETRACE, TREND_RETEST. CONTEXT_GEOMETRIC is a same-context fixed
retracement-region control, not unconditional market entry. The common context
may describe past jumps; any claim that it anticipates a further jump requires
the actual future result. All scheduled00/30 rows remain in the candidate table,
including unavailable, disagreeing-context and invalid-geometry dispositions.

## Replay

Region bounds, exterior invalidation and raw M5ATR14 are frozen at issue. Monitor
invalidation immediately after issue. Activate after issue+1minute. An observed
in-zone quote strictly after activation and at/before issue+15minutes triggers
entry on the exact next observed second, unconditionally at its actual price.
Jumping across the band without an in-band observation creates no entry. The
next quote may lie beyond the region; do not cancel or reprice it after the fact.

Stop is2ATR from actual entry; timeout is touch+15minutes. Stop triggers up to
and including expiry take priority; stop exits and timeout both use the next
actual second. No fabricated barrier fill or capped loss. Assumed round-trip
cost0.10ATR, with a0.20ATR sensitivity on the identical paths. These are public
quote proxies, not verified CFD fills, measured costs or cash returns.

One pending region or position per symbol/variant. Unknown required seconds
censor waiting/entry/exit outcomes and reserve maximum planned occupancy. Do
not discard missing pending regions from denominators merely because no trade
was filled. Planned maximum issue+30minutes+1second must lie strictly inside
the partition, and issue+31minutes<=partition end, even if realized exit is early.

## Chronology and comparisons

Full source grid: [2025-10-09T00:00Z,2026-10-05T00:00Z).
Older180days: [2025-10-09T11:08Z,2026-04-07T11:08Z).
Report development first70%, chronological validation40–50/50–60/60–70%, and
held-out last30%. Later180days:
[2026-04-07T11:08Z,2026-10-04T11:08Z). Rules are fixed and no parameters are
fitted; these are expanding-calendar validation windows, not fitted ML folds.
Validation union consists of three separate purged replays; overlapping
development and validation descriptions are never pooled as independent paths.

Report every fixed family/control for both symbols and all partitions. Preserve
scheduled/eligible/issued/purged/occupied/invalidated/expired/touched/unknown/
completed counts and median time to touch. Profit factor, mean netR, active days,
closed-trade drawdown and UTC-day/seven-day CI95 use9999 resamples, seed20261008.
Whole-policy paired comparisons use the geometric control; differing acceptance
and touch rates mean they do not isolate geometry alone. Holm corrects the
12held-out comparisons (three hypotheses×two symbols×two periods), using the
larger day/week p-value for each conjunction. Empty ratios remain unknown.

The present economic endpoint is completed timed quote-path payoff. Precision,
recall, base rate, lift and time-to-spike for additional12event definitions are
NOT TESTED in this new zone experiment; the original requested12definition CRT
study was executed previously. Do not mislabel the region-touch fraction as
spike precision, or a post-entry move as a verified discrete spike. No posthoc
label/horizon/stop/cost/width/direction rescue is allowed in this experiment.

## Qualification

Keep prior gates: each single symbol/model requires netPF>=1.5, >=1000completed
held-out one-open paths, >=60active held-out days; development validation union
>=500/30days, positive lower day-influence selection score, each fold>=100 with
positive mean/PF>1 and complete known outcomes. Require day/week PF lowerCI>1,
meanCI>0, control-advantageCI>0, Holm<.05, each chronological held-out third
>=200with positive mean, no unknown candidate/control required paths, doubled
cost positive mean, no ruin, closed-trade drawdown<=10% at illustrative0.25%R.
Support for populationPF>=1.5 additionally needs both PF lowerCIs>=1.5.

Even passing historical criteria cannot create fresh OOS, prospective paper
evidence or an executable money-profit claim. Qualification remains false until
the entire evidence contract is satisfied and independently audited.
`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false`. Actual CFD fills/costs/cash profit/prospective paper:
NOT TESTED. The user's full goal remains active until genuinely achieved.

Conceptual sources reviewed:
[Deriv Fibonacci guide](https://deriv.com/academy/trading-guides/fibonacci-retracement-levels-in-trading)
and [Deriv support/resistance guide](https://traders-academy.deriv.com/trading-guides/chart-patterns-for-support-and-resistance-trading).
They describe chart construction and possible zones, not evidence that these
fixed Boom/Crash rules have profitable predictive power.
