# Fixed native-300 replication — adaptive research round 4

This is one declared replication prompted by the observed failure of the
500-to-300 feature transfer. It changes only the training symbol and the
chronological training/evaluation periods. There is no feature, penalty,
cutoff, exit, stop, horizon or cost optimization toward the user PF target.
All four flags remain false: LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED and
OPENED_TRADES. No account or broker order is accessed.

## Fixed model and execution assumptions

- Train separately on native BOOM300N and CRASH300N, each in SPIKE and DRIFT
  direction. Exactly four final models. SPIKE means long Boom/short Crash;
  DRIFT reverses that side. No pooling across symbols or directions.
- Preserve the nineteen causal inputs and ordered formulas in the frozen
  `learned_signal.py`. Preserve closed M5 candles, missing/incomplete grid
  rows, unknown large-bar ages, and the all-finite causal-valid mask.
- Opportunities remain exact UTC M5 closes at minute00/30. The clock baseline
  uses every eligible opportunity and the same side, timing, cost and exit.
- Train-only standardization, unclipped continuous net-R label, unpenalized
  intercept, mean-SSE ridge penalty0.1, no interactions or feature selection.
  Refit every scaler/model/cutoff separately within each declared training
  partition. Issue only when score>=max(0, training-prediction q75) and score>0.
  Scores are experimental continuous regressions, not calibrated probabilities.
- Frozen primary: stop2*signal ATR14, no take profit, hold15 M1 minutes, entry
  at M1 open one minute after issuance, adverse-extreme stop stress, modeled
  round-trip cost0.10ATR. Net R subtracts0.05R because stop denominator is2ATR.
  This is a quote-path stress simulation, not a historical broker fill.
- Preserve exact opens, gaps, unknown/censored paths and one-position occupancy
  from `replay_timed`. Apply the common issue+31-minute end purge regardless
  of early realized exit. Training omits unknown labels and records exclusions;
  evaluation preserves issued-path censoring and missing-entry audits.

## Sources and chronology

- Older native300 source: `data/spike_learned_transfer`, requested interval
  [2025-10-09 11:08,2026-04-07 11:08)UTC,259,200 canonical M1 rows per symbol.
  Its complete transfer payoff was already inspected in the previous round.
  All older-period training and diagnostics are therefore reused exploration.
- Newer native300 source: `data/spike_native300_recent`, requested interval
  [2026-04-07 11:08,2026-10-04 11:08)UTC. Each clean source has259,199 rows:
  one declared missing minute and one excluded raw timestamp off the minute
  grid. Keep the raw exclusion and gap manifests. Never shift, fill or
  interpolate the missing quote. The loader restores its unknown grid row.
- The intervals are adjacent and disjoint. Verify exact declared boundaries,
  source/manifest hashes, normalization validity and public-data safety before
  deriving evaluation features. Gap-aware causal eligibility and execution
  audit determine whether missing quotes affect any issued paths.
- Fresh300 payoff/features have not been examined before this declaration and
  final selection freeze. This follows older-to-newer chronology, but historical
  collection and adaptive methodology make it retrospective. It is **not** a
  prospective paper cohort or a wholly independent research universe.

## Development and immutable evaluation release

- Use only first70% of the older source for label fitting/development. Expanding
  fits0–40%,0–50%,0–60% validate40–50%,50–60%,60–70%, each with its own31-minute
  label/evaluation purge. Refitting never uses a validation outcome.
- Direction eligibility is inherited unchanged: all three exact ordered folds,
  >=100 completed trades and positive mean/PF>1 per fold, no censored/invalid
  results, combined>=500,>=30 active days and positive cluster selection score.
  Choose the eligible direction with largest selection score and alphabetic
  tie break. If none qualifies, selected_direction=null. Both fixed rejected
  models remain diagnostic; later results cannot select a replacement.
- Fit both final models on older first70%. Save model/scaler/cutoff/training
  hashes, all inherited source hashes, new runner/protocol/test hashes and all
  old/new price and normalization-manifest hashes. Save declaration.json and
  its hash before old development, then selection.json and selection.sha256
  before any fresh feature preparation or payoff evaluation.
- A separate evaluate stage verifies declaration, selection, code and every
  acquisition/manifest hash before any feature preparation. Refuse overwriting
  frozen selection or final results. Refuse development if results already
  exist. No post-result selection, rescaling, retuning or fallback promotion.
- Report older final30% as `reused_final30` exploration. The complete newer
  period is `fresh_temporal180`, evaluated with the frozen older70% models.

## Inference and fixed evidence gates

- Inherit the prior `evaluate_one` diagnostics and `gate` without modifications:
  net PF, sample size, mean/median R, win rate, holding time, active days,
  clock baseline, closed illustrative equity drawdown, tail concentration,
  chronological thirds and fixed execution/cost sensitivities.
- Fixed9,999 bootstrap replicates, seed20261005. Shared UTC-day model/baseline
  means and seven-day block sensitivity; gain/loss-resampled daily and weekly
  PF95 intervals. Undefined no-loss ratios stay unknown and reject promotion.
- Holm adjustment covers exactly four fresh native hypotheses: two symbols
  times two directions, including development-rejected diagnostics. No pooling
  and no removal of a disappointing hypothesis from this family.
- User point target: net PF>=1.5 with>=1,000 completed fresh trades per
  model/symbol. Historical candidate additionally requires development
  eligibility,>=60 active days, no missing/censored/invalid issued outcomes,
  daily/weekly PF95 lower bounds>1, no undefined PF bootstrap ratios, positive
  mean and excess-CI lower bounds, Holm p<0.05,>=200 positive-mean completed
  trades in each chronological third, no equity ruin,<=10% closed-trade
  illustrative drawdown and positive mean at doubled modeled cost0.20ATR.
- `supports_expected_pf_1_5` additionally requires both PF95 lower bounds>=1.5.
  A point PF>=1.5 alone does not justify that stronger claim.
- Fixed sensitivity costs0,.025,.05,.10,.20,.40ATR, primary adverse/+1, proxy/+1
  and adverse/+0. Sensitivities may disclose fragility; they cannot rescue or
  replace the primary. Old final30 results cannot promote a strategy.

Actual bid/ask, commissions, financing, sizing, broker stop fills, cash profit
and a prospective frozen paper cohort remain NOT TESTED. No retrospective
finding changes any safety flag or proves the user's high-profit objective.
