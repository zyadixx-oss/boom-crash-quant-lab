# Acceptance contract — independent single-frame research

Authoritative request: attached Arabic specification dated 11 October 2026.
This campaign supersedes the combined-frame search only for new work. Frozen
prior studies remain immutable and are not recast as this campaign's results.

## Before implementation

Existing repository inspected: public Deriv collectors, frozen research studies,
Python research backend and Arabic React/Vite chart. Existing multiframe engines
and 70/30 results do not satisfy the new single-frame/60/20/20 request.

| Requirement | Evidence required |
| --- | --- |
| Ten Boom/Crash 300/500/600/900/1000 symbols | Saved current public catalogue and per-symbol actual API mapping |
| M1/M2/M3/M5/M10/M15/M30/H1/H2/H4 separately | 100 symbol/frame coverage cells; gap-free complete UTC resampling |
| Multiple named strategy families and independent tuning | Frozen candidate catalogue, settings and validation-only scores per symbol/frame |
| 60/20/20 chronology | Fixed UTC calendar boundaries, planned-horizon purge, no final access in search |
| Walk-forward when enough data | Development-only chronological folds and explicit insufficient states |
| Real history only | Raw public response hashes, source manifests, cache and quality logs |
| Realistic cost and contract evidence | Instrument/contract mapping, historical spread/slippage/fees and execution provenance; otherwise no qualification |
| Tick verification of jump strategies when available | Original-tick replay evidence or explicit unverified execution rejection |
| Fixed stop and target; no trailing stop | Engine tests for next-bar entry, ambiguous path priority, gaps and fixed exits |
| Full virtual ledger and analysis | Per-run trades, equity/drawdown, monthly summaries and references |
| OOS acceptance | n >=100, PF >=1.10, DD <=20%, net outperforms costed buy-and-hold, valid data/causality/reproducibility/execution |
| 32 actual cycles at 15-minute cadence over eight hours | Durable start/completion timestamps and process/automation evidence; never simulated cycle counts |
| Resume after interruption and avoid duplicates | Atomic state, process lock, input+candidate cache keys and recovery tests |
| Arabic dashboard and exports | Actual snapshot rendering, heatmap, ranking, logs, ledger, charts, CSV/JSON |
| All live flags false | Runtime guards, saved state and UI; no account/order methods |

## Unverified assumptions and failure modes

Public option quotes have not been established as executable historical CFD
bid/ask/costs. A displayed point-price backtest with an explicit assumed cost
is exploratory and cannot meet the qualification gate. Missing candles,
conflicting duplicates, unavailable symbols, bar-level SL/TP ambiguity,
insufficient H4 sample, data already exposed by prior research and multiplicity
remain explicit. No zero-cost, perfect-fill or untested live-profit claims.

The current public history's earliest reliable bound is to be measured, not
called the provider's universal maximum. New updates are cached separately;
the campaign's development/final split and initial data snapshot are frozen.
Incoming observations cannot move the holdout boundary during tuning.

## Smallest invalidating tests

1. Append future bars: previously issued single-frame signals must not change.
2. Supply a missing minute: a larger frame spanning it must not be synthesized.
3. Both fixed exits in one bar: never select the profitable ordering.
4. Gap through stop: use uncapped adverse observed-open price.
5. Attempt final-based selection or true live flag: fail immediately.
6. Positive synthetic test metrics with unverified costs: must remain rejected.
7. Resume the same completed candidate/input: no duplicate measurement/cycle.

## Completion audit

Code, tests or a green build alone do not prove the 32 cycles ran or any strategy
qualified. Final evaluation occurs once after development search ends, with
selection frozen before access. No qualifying strategy is a valid research
outcome; each symbol must say so explicitly. Full 32-cycle results and scheduling
completion remain pending until their authoritative records exist.
