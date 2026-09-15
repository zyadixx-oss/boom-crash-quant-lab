# Results

Status: **NOT TESTED on real Deriv historical/shadow data yet.**

## Data period
NOT TESTED

## Number of ticks/candles
NOT TESTED

## Symbols tested
No real Deriv dataset has been evaluated yet.

## Strategies tested
Weighted pre-spike signal engine is implemented; real-data validation is pending.

## Best strategy per Symbol
NOT TESTED

## Win Rate / Profit Factor / Expectancy / Max Drawdown
NOT TESTED on real Deriv data. Synthetic smoke-test metrics must not be treated as evidence of an edge.

## Out-of-Sample / Walk Forward
Framework implemented. Real-data evaluation: NOT TESTED.

## Shadow results
NOT TESTED.

## Known limitations
- API symbol identifiers for newly introduced ranges should be resolved with `active_symbols` before collection.
- Tick-derived microstructure features require persistent tick collection; candle-only CSVs cannot reproduce tick frequency exactly.
- ICT concepts are deterministic heuristics here, not assumed alpha.
- Current optimizer is a conservative grid-search baseline; Bayesian optimization is not yet implemented.

## UNVERIFIED ASSUMPTIONS
- Pre-spike volatility compression may carry predictive information for each Boom/Crash range.
- The chosen mathematical spike definition is useful for all symbols after per-symbol calibration.
- M5/M15 confirmation improves out-of-sample performance rather than reducing signal quality.

## LIKELY FAILURE PATHS
- No stable predictive edge exists before spikes.
- Regime drift makes optimized parameters unstable.
- Sparse extreme events create misleading small-sample metrics.
- Tick/candle gaps or symbol mapping errors contaminate labels.
- Feature engineering overfits historical synthetic-index behavior.

## MINIMAL TEST THAT COULD FAIL THE PLAN
Collect a sufficiently large, clean dataset for one symbol, freeze a simple parameter region using train/validation only, then run untouched out-of-sample and walk-forward tests. If expectancy is non-positive and precision does not beat simple baselines with stable confidence across windows, classify **NO RELIABLE EDGE FOUND** and stop adding complexity.

## Implementation verification in this build
- Python unit/integration/acceptance tests: executed locally in the build environment.
- FastAPI `/health` and `/symbols`: executed locally.
- Real Deriv WebSocket connectivity: **NOT TESTED in this build environment** because outbound DNS/network access was unavailable.
- Frontend dependency install/build: **NOT TESTED** because npm registry access was unavailable in the build environment.
