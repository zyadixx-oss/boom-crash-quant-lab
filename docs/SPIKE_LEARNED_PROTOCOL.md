# Frozen learned-score payoff protocol — adaptive research round 3

This specification is fixed before computing learned-model transfer payoffs.
All four flags remain false: LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED,
OPENED_TRADES. No account, broker order, monetary fill or prospective result is
produced. The user-defined target is modeled net profit factor (PF) >=1.5 with
at least 1,000 completed held-out trades **per model and symbol**, never pooled.

## Fixed model and opportunities

- One standardized linear ridge model for each of SPIKE and DRIFT, separately
  trained on BOOM500 and CRASH500. Exactly four final fitted models; no feature,
  threshold, regularization, horizon, stop or exit search.
- Nineteen causal continuous inputs, in the order and formulas declared in
  `backend/app/research/learned_signal.py`. Only completed M5 candles; issue at
  the candle close. Preserve missing and incomplete candles on the original
  grid. Require the existing causal-valid mask and all nineteen finite inputs.
  Age of a large completed candle is unknown before an observed event and
  after a gap, rather than a fabricated countdown.
- Opportunities are exact UTC M5 closes at minute 00 or 30. The baseline enters
  every eligible opportunity with the same side and exit policy. Feature
  availability defines both cohorts, never future outcome availability.
- Fit the unclipped net-R label with training-only means/std, unpenalized
  intercept, objective mean-SSE + 0.1*sum(coefficient^2). Constant columns use
  scale one. No clipping, interactions, feature selection or target winsorizing.
- Issue when the frozen score is >=max(0, training-prediction q75) and >0.
  Scores are continuous experimental net-R regressions, NOT calibrated
  probabilities. No scaler or cutoff adaptation on held-out data.

## Fixed quote-path label and timing

- Stop 2*signal ATR14, no take-profit, hold at most 15 M1 minutes, entry at the
  M1 open one minute after issuance, modeled round-trip cost 0.10*signal ATR.
  R denominator is 2*ATR, so cost is 0.05R per completed simulation.
- Existing `replay_timed` adverse-extreme stop stress, exact opening gaps,
  missing-path censoring and one-position occupancy. OHLC stress is not a
  measured execution fill. Signal ATR is fixed at issuance.
- Every training/evaluation boundary purges all opportunities whose issuance
  plus 31 minutes exceeds the boundary, independent of realized early exit.
  This exceeds the 16-minute planned entry-plus-hold and is retained from the
  prior experiments. Missing entry/path audits are preserved after issuance.
- Training excludes censored labels and reports exclusions. Evaluation never
  removes an issued trade using knowledge of its future path.

## Development, freezing and replication

- Original BOOM500/CRASH500 180-day data: first70% development, last30% reused
  diagnostic. Expanding fits 0–40%, 0–50%, 0–60%, evaluated on 40–50%, 50–60%,
  60–70% respectively. Refit means/std/coefficients/cutoff from scratch in each
  fold. Pool only these disjoint out-of-fit validation ledgers for selection.
- A direction is development eligible only if its three validation folds have
  >=100 completed trades each, positive mean and PF>1 each, no censored/invalid
  outcomes, combined >=500 trades, >=30 active days and pooled day-cluster
  mean lower approximation >0. Rank eligible directions by that approximation;
  deterministic alphabetical tie break. If none qualifies, selected_direction
  is null; both fixed directions may be reported as rejected diagnostics.
- Fit both final models on first70%, save feature order/scaler/coefficients/
  intercept/cutoff/training intervals and matrix hashes in selection.json.
  Save selection.sha256 and code/data hashes before external evaluation. The
  evaluator rejects changed protocol, sources, models or input prices.
- BOOM300N/CRASH300N: official complete M1 grid
  [2025-10-09 11:08, 2026-04-07 11:08) UTC, 259,200 rows each. Use semantic
  Boom300/Crash300 names with canonical API identifiers retained. Their payoff
  is not read before freezing; acquisition source audit is allowed. Earlier
  conversation contained unverified Boom300 classification claims, so this is
  a new payoff replication, not a wholly untouched research universe.
- Also report the reused original500 final30% and previously examined older500
  180 days as exploratory diagnostics. The 300 period is earlier than training
  dates: cross-symbol retrospective transfer, **not chronological forward
  generalization**. Synthetic RNG independence is not established here.

## Inference and decision gates

- Net PF, n, win rate, mean/median R, holding time, active days, modeled closed
  equity drawdown, tail concentration, baseline and chronological thirds.
- Shared UTC-day bootstrap of model/baseline mean and excess mean; circular
  seven-day blocks as sensitivity; 9,999 repetitions, seed20261005. PF95 uses
  daily gain/loss sums and the same-day indices for numerator and denominator,
  in both day and seven-day resampling. Undefined zero-loss ratios are counted
  and omitted from finite intervals, never reported as infinity or confidence.
- Holm adjust the four 300-transfer tests (two symbols * two directions), using
  the maximum day/week p for positive mean and excess over the clock. Reused
  diagnostics do not enter the transfer discovery family.
- `user_target_observed`: net PF>=1.5 and >=1,000 completed trades per row.
  This alone is a descriptive target, not proof of future profitability.
- `historical_candidate`: observed user target, development eligibility, no
  censored/invalid/missing-entry outcomes, >=60 active days, day/week PF lower
  bounds>1 and no undefined PF resamples, positive day/week mean and excess CI lower bounds, Holm p<0.05,
  all three chronological thirds positive with >=200 completed trades each,
  no equity ruin and <=10% closed-trade illustrative drawdown at 0.25% risk.
  Mean must remain positive at doubled modeled cost (0.20 ATR).
- `supports_expected_pf_1_5`: historical_candidate plus BOTH day/week PF95
  lower bounds>=1.5. Never describe a point PF>=1.5 as this stronger evidence.
- Fixed sensitivity: adverse-extreme +1minute at costs0,.025,.05,.10,.20,.40ATR;
  barrier proxy +1 and adverse-extreme +0 at the same costs. These are
  diagnostics and cannot replace the frozen primary or rescue a failed gate.
  Tail metrics and removal of best five trades/days disclose concentration.

All results are exploratory because prior research informed this model family.
Actual bid/ask, commissions, financing, liquidity, position size, broker stop
fills, cash profit and a frozen prospective paper cohort remain NOT TESTED.
No retrospective result enables execution or makes READY_FOR_LIVE true.
