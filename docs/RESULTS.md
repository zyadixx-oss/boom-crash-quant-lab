# Results

Status: **IMPLEMENTATION VERIFIED; REAL STRATEGY EDGE NOT YET VERIFIED.**

## Implementation verification

GitHub Actions `Arena CI` verifies both application layers on the Arena branch:

- Backend: dependency install + `pytest -q tests ../tests`.
- Frontend: dependency install + production `npm run build`.
- Latest verified workflow for the Arena implementation completed successfully on 2026-09-22.
- The suite covers candidate creation, ten-symbol profiles, backtest metrics, spike detection, qualification, chronological OOS split, leaderboard scoring, safety flags, no-order behavior, API endpoints, and the legacy tests.

## Real Deriv research status

No claim of profitability or predictive edge is made by this implementation. Real historical/OOS/forward observations have not yet been run long enough to justify an edge claim.

Synthetic/generated candles are allowed only as software smoke-test data and must be labeled DEMO when used. They are not evidence of strategy quality.

## UNVERIFIED ASSUMPTIONS

- Pre-spike compression, liquidity behavior, candle structure, S/R, or MTF context contains stable predictive information for Boom/Crash spikes.
- The current ATR-based mathematical definition of a spike maps well to each Deriv symbol after independent calibration.
- The M1/M5/M15 proxy/context implementation retains predictive value on real Deriv data.
- Default qualification thresholds are appropriate; they are starting configuration, not validated truth.
- A 24h forward window contains enough events for useful decisions; sparse symbols may require 3 or 7 days.
- Public Deriv symbol identifiers and feed behavior remain compatible with active-symbol resolution.

## LIKELY FAILURE PATHS

- No stable pre-spike edge exists and historical profit is incidental.
- Rare spikes produce high-variance precision/recall and misleading backtest rankings.
- Regime drift invalidates candidates that appeared stable in an earlier period.
- Data gaps, candle construction differences, or symbol-resolution errors contaminate labels.
- Heuristic ICT concepts may not correspond to a reproducible statistical edge.
- Parameter selection can still overfit despite bounded search and sensitivity tests.
- Free hosting can sleep or lose ephemeral SQLite data, interrupting long forward runs.
- A process restart ends in-memory shadow tasks; persisted run state remains but automatic task resumption is not implemented.

## MINIMAL TEST THAT COULD FAIL THE PLAN

Choose **one MVP symbol** (Crash 500 or Boom 500), collect a clean chronological dataset, freeze one candidate using train + validation only, then run the untouched 20% OOS segment.

The plan should be reconsidered if that candidate has enough trades to be statistically useful but shows non-positive expectancy, poor spike precision/recall versus simple baselines, and instability under ±5% parameter perturbation. If the same failure repeats across several chronological windows, adding more agents or optimization would be complexity without evidence of an edge.
