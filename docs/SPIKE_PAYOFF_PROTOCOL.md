# Spike Hunter payoff protocol v1

Frozen 2026-10-04 17:21 UTC before reading any new payoff results. This follows
the excursion-classification study. All four trading flags remain false.

## Question and evidence level

Do causal signals create positive normalized quote-path returns after explicit
execution stress? High precision or excursion lift is not profitability.
Public index quotes are not established historical MT5 bid/ask or fills. No
monetary net profit, actual execution, leverage suitability or forward evidence
can be certified from these inputs. A passing result is a candidate for a new
prospective paper study, never a live-trading permission.

## Data and separation

- Reuse the exact old BOOM500/CRASH500 M1 data, 2026-04-07 to 2026-10-04.
- The first 70% is development. Last 30% is `reused_oos_exploratory`: its
  classification outcomes were already inspected, so its new PnL is not a fresh
  confirmatory test.
- Acquire the preceding 180 days ending exclusively at 2026-04-07 11:08 UTC.
  Do not inspect payoff results on this older external data until selection is
  written and hashed. This is retrospective replication in a different period,
  not forward-time validation. Source auditing alone is allowed before freezing.
- Missing minutes stay missing. Signal generation retains the original
  complete-bar, gap, warmup, H1 reference and 30-minute causal cooldown rules.

## Fixed execution model

- Trade in spike direction: long Boom, short Crash. Signal at M5 close, ATR
  frozen from that completed candle. One position per model and symbol.
- Primary entry: exact M1 opening at signal time +1 minute. No shifting to the
  next available row if that timestamp is missing. Delay0 is sensitivity only.
- Brackets are relative to entry quote. Examine opening gaps before H/L. An
  opening beyond TP exits at TP (no beneficial gap credit); an opening beyond
  SL exits at the worse opening quote in the barrier proxy.
- If opening is inside both barriers and H/L touch both, SL comes first.
  Target-first is an explicitly optimistic sensitivity, never the primary.
- Primary `adverse_extreme` execution stresses each SL exit to that minute's
  adverse extreme. Secondary `barrier_proxy` uses SL except adverse opening
  gaps. Neither is a claim of actual broker execution.
- Timeout measured from entry; evaluate exactly H minutes, checking brackets
  before timeout at the last minute close. No minute after timeout is used.
- A missing path before exit is censored with unknown PnL; hold occupancy to
  planned end. Missing data after an already completed exit cannot censor it.
- All experiments use a common31-minute partition-end purge from signal time.
- Gross R is directional quote move / initial stop distance. Net R deducts
  `round_trip_cost_atr / stop_atr` once. Primary modeled cost0.10ATR. Sensitivity
  costs0,0.025,0.05,0.10,0.20,0.40ATR, explicitly hypothetical all-in costs.
- Fixed-risk equity0.25% per initial stop is a normalized illustration only;
  no lot sizing, orders or account access. Closed-trade drawdown is labelled as
  such and cannot substitute for complete executable intratrade risk evidence.

## Candidates, selection and chronology

Reuse all16 predefined signal models. Candidate brackets are the Cartesian
product SL{0.5,1,1.5}ATR × TP{1,2,3}ATR × timeout{5,15,30}minutes:432 per symbol.
Do not expand the grid after viewing external results in this protocol.

The fixed primary hypothesis is SR_ALIGNMENT with SL1ATR/TP2ATR/15minutes.
Also select exactly one bracket/model per symbol from development. Rank by
mean net R minus1.96 times a UTC-day clustered delta-method standard error,
requiring300 completed trades and30 days with trades. Require at least30 trades
and positive mean net R in each of the40–50%,50–60%,60–70% development windows.
If none pass, retain the best eligible or best available model for diagnosis,
label it rejected and do not pretend a valid selection existed.

Expanding walk-forward selection uses only0–40% to choose before40–50%, only
0–50% before50–60%, and only0–60% before60–70%. Record each chosen configuration,
training score and subsequent validation result. Final selection uses0–70%.
No external or reused-OOS result may select a configuration.

Compare every evaluated configuration to a deterministic unconditioned M5
clock baseline, with the same30-minute cooldown, brackets, costs, direction,
entry delay and partition purge. Compare expectancy per completed trade; daily
profit also depends on different turnover and must not imply equal exposure.

## Metrics and gates

Report count, censored/skipped/ambiguous counts, wins, win rate, mean/median R,
sum R, profit factor, average winner/loser, median holding time, trade days,
break-even modeled cost in ATR, closed-trade risk illustration and drawdown.
For locked final hypotheses use paired UTC-day block resampling9999 times,
fixed seed20261004, for mean net R and its difference from baseline. Include
all calendar days, including days without trades. Report pointwise CI95 and
centered one-sided bootstrap p; Holm correction across the up-to-four external
hypotheses (two symbols × fixed primary/selected, deduplicated if identical).

A research candidate requires no censored trades, >=300 completed trades,
>=30 active days, mean net R>=0.10, profit factor>=1.30, mean R CI95 lower>0,
baseline difference CI95 lower>0, Holm p<0.05, and the illustrative closed-trade
drawdown<=10%, plus positive results in all three development validation
windows. These are predeclared research thresholds, not a validated definition
of guaranteed high returns. Actual fills, historical measured costs and a fresh
forward cohort remain required before any real-profit claim.

## Primary sources checked

- https://developers.deriv.com/comparison/ticks-history/
- https://developers.deriv.com/docs/data/
- https://deriv.com/tnc/trading-terms.pdf (3.3,3.4,4.1: next-tick processing/spread)
- https://traders-academy.deriv.com/trading-guides/stop-loss-execution-boom-crash-indices
- https://deriv.com/trading-specifications (current reference, not historical costs)
