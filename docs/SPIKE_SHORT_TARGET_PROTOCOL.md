# Round11: fixed short-reward multiframe target ablation

Declared before any round11 historical label, fit, feature or payoff calculation.
The executable metadata declaration records the UTC freeze and all input/science
hashes. Earlier protocols, models, results and audits remain unchanged.
`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false`; public data and offline simulation only.

## Scientific question and prior exposure

The previous native600 combined44 learner was trained on a 15-minute time-exit
reward. Legacy short-horizon grids used different hand-coded signals; delay
sensitivities reused the same 15-minute-trained model. These do not establish
whether fitting the existing 44 causal inputs to a one-minute reward improves
that particular fixed action policy. This experiment changes only the learned
reward horizon, not the algorithm, features, clock, cost, stop, entry delay,
side set or score quantile. It is not a new exit-parameter search.

SHORT44 is trained on SHORT1 (one-minute hold) labels. LONG44 is trained on
LONG15 labels and is a fixed reference only: it can never be selected or
promoted in this round. BOTH models and CLOCK are evaluated under SHORT1.
H1 in feature names still means the one-hour context, not the reward horizon.

Native BOOM600 and CRASH600 are tested separately, each in SPIKE and opposite
DRIFT directions. Price data and the previous long-policy payoffs are already
known from earlier research. All development, secondary and later-period
results are adaptive known-history evidence: not fresh price OOS, a new label
holdout, prospective evidence or a globally adjusted discovery. The unchanged
user target remains net PF>=1.5 with a large sample for the combined frames.

## Sources and chronology

Use exactly the pinned older and later 180-day M1 sources, normalization/page
manifests and ancestor artifacts of round6. Recovered rate limits and source
gaps remain as documented; no recollection, filling or alternative interval.
Source hashing and metadata reading before declaration are allowed; new
indicators, targets and payoffs are not. Paths are canonical repository-relative.

The older source is 2025-10-09 11:08 UTC to 2026-04-07 11:08 UTC exclusive;
the later source immediately follows through 2026-10-04 11:08 UTC exclusive.
Fit older first40/50/60% and validate the next10% each. Final fit uses only
older first70%; freeze its models/scalers/cutoffs and exact training inputs
before calculating older last30% (secondary) and later180 days (primary).
Each training/test endpoint uses the common31-minute planned purge, even for
SHORT1. Every feature is from closed candles available by its issuance; invalid
latest frames remain invalid rather than falling back to old valid context.
No later-period feature, target or model selection during development.

## Fixed inputs, target and execution

- The same ordered 44 inputs from closed H4/H1 + M15/M5/M1, including the
  original19 base inputs, are reused without additions or threshold grids.
- Opportunities are feature-valid UTC M5 closes at minute00/30. Both target
  families and CLOCK share available rows, frozen ATR14 and partitions.
- Entry proxy: exact M1 opening at issuance+1minute, SL2ATR, no TP, primary
  total assumed round-trip cost0.10ATR. Stop gaps and adverse minute extreme,
  timeout, unknown paths and occupancy follow unchanged `payoff_timed.py`.
- Common evaluation hold=1minute. Only LONG15 training labels use hold=15.
  The timeout is the last included minute close, not a broker execution quote.
- Shared training X/timestamps consist of the intersection of individually
  completed, finite SHORT1 and LONG15 labels within the same training purge.
  At this 30-minute clock, the <=16minute target paths cannot overlap, which
  is checked. Repeated target/family records still are not unique strategy
  samples or proof of statistical independence.
- Fit train-only mean/std scaling and unclipped mean-SSE ridge penalty0.1;
  train score q75 with positive floor. Issue only scores>=cutoff AND>0.
  Require1000 shared finite training targets per fit; no smaller fallback,
  alternate fit or replacing invalid rows. Model semantic validation must
  refuse version/schema/scaler/objective/quantile/config tampering even if a
  changed artifact has a newly recomputed checksum.
- Preserve training matrices, both targets, row/byte hashes, models/scalers,
  issue scores, individual labels and one-open strategy ledgers separately.
  Unknown paths reserve planned occupancy and remain unknown, never zero.

## Eligibility, comparisons and rejection

Only SHORT44 can be a candidate. Development uses the inherited learner
requirements: >=500 completed pooled validation paths, >=30 active days,
no censored/invalid outcomes, positive conservative selection score, and
>=100 completed paths with positive mean and PF>1 in EACH of the three
expanding validation folds. LONG44 and CLOCK are reference_only throughout.
No eligible SHORT direction means selected_candidate=null; no fallback.

Report both model families and CLOCK for each symbol/side in all declared
cohorts, including n, active days, skipped/censored/invalid, mean/median R,
PF, win rate, holding time, raw direction move, trade/day tails and closed-trade
risk illustration. Pair comparisons by UTC day over the identical cohort.
Use9999 resamples, seed20261005: day clusters and circular7-day blocks,
including all calendar days. Report PF/mean CI95 and paired SHORT-minus-LONG
and SHORT-minus-CLOCK mean CI95 under both schemes. All planned eligible
payoffs of the models and both references must be known. Undefined draws
block positive inference; they are counted and never silently omitted to pass.

Four primary hypotheses are fixed (2symbols x 2sides), SHORT44 only. Each
hypothesis combines positive absolute net mean AND positive increment over
LONG44 AND CLOCK at both day/week scales, using the maximum centered one-sided
p and Holm correction across these four. Secondary older30% is diagnostic and
cannot select models, parameters or a winning direction.

A conditional modeled-economic pass also needs: eligible development,
>=1000 completed primary later paths and >=60 active days per symbol/model,
PF>=1.5, mean netR>=0.10, PF CI95 lower>1 and all absolute and incremental
mean lower bounds>0 under both schemes, Holm p<.05, no unknown/invalid outcome,
closed-trade illustrative DD<=10% without ruin, and positive mean with
>=200 completions in each predeclared equal-
duration later third. No pooling across symbols/sides, threshold adjustment,
window extension until favorable, or counting labels toward the sample gate.
An additional predeclared cost stress deducts0.20ATR total round-trip cost on
the identical paths without changing signals, entry or exits; its mean must
remain positive. It cannot select or replace the primary0.10ATR result.

Passing these checks sets conditional_economic_gates_pass only. Because these
prices/payoffs were already exposed, historical_candidate and the overall
supports_expected_pf_1_5 remain false. Report observed PF>=1.5/n>=1000 separately,
and whether the conditional day/week PF lower bounds reach1.5; neither supplies
fresh evidence, guaranteed profit or live permission. A positive conditional
pass needs unchanged prospective evidence and tick/cost verification before
a profit conclusion. Frozen selections never change from later outcomes.

## Execution and inference limits

Deriv describes jump timing as independent of time elapsed since the previous
jump in its [primary execution explanation](https://traders-academy.deriv.com/trading-guides/stop-loss-execution-boom-crash-indices).
That product statement and the [drift/spike explanation](https://experts.deriv.com/insights/boom-and-crash-the-drift-the-spike-and-what-a-spike-does-to-your-indicators)
do not give a full conditional return kernel or prove that our sampled
price/log-price process is a martingale. Smooth drift alone is not an edge.

[Trading terms](https://deriv.com/terms-and-conditions/trading-terms),
clauses3.3/3.4, describe next-tick entry/exit after server processing.
This M1 open/adverse-extreme/close proxy does not reproduce that processing,
historical executable bid/ask, contract sizing, financing or measured costs.
Cash profit, actual broker fills and prospective paper performance: NOT TESTED.

The independent stdlib verifier will check captured source/artifact hashes,
saved matrices/scalers/scalar predictions/cutoffs, M1 path replay, point metrics,
common target membership, chronology and policy gates. It will not regenerate
all44 features, refit ridge, regenerate bootstrap draws, prove inferential
validity/independence or verify broker fills/profit. CI software tests alone
cannot validate any historical financial claim. Failed audits must be preserved
before repairing only verifier logic; frozen science/results cannot be revised.
