# Uncapped time-exit protocol v2

Declared before any v2 historical payoff calculation; the final UTC freeze
timestamp is recorded in selection.json. This is a new
adaptive research round motivated by the failed fixed-target v1 study. It does
not alter the v1 frozen protocol, selection or results. All four trading flags
remain false. High financial profitability is unproven.

## Question and source boundary

Does removing the profit cap preserve enough directional upside to create
positive quote-path expectancy after modeled execution costs? Spike direction
and its opposite are separate predeclared candidates. Opposite direction uses
the same causal observations; it is not renamed as a new ICT structural signal.

The source is official public M1 OHLC, not historical bid/ask or actual fills.
Normalized R, indicative equity and profit factor cannot establish monetary
net profit. No account, order, credential or monetary sizing is used.

## Data and chronology

- Development: original BOOM500/CRASH500 data2026-04-07 to2026-10-04, first70%.
  These development data have already been used; this is an adaptive experiment.
- Expanding selection windows0–40%,0–50%,0–60% precede validation40–50%,50–60%,
  60–70%. Final parameter selection uses only0–70%.
- The recent last30% and older500 period2025-10-09 to2026-04-07 were already
  consumed in v1. They are explicitly `reused_exploratory`, not fresh evidence.
- An attempted preceding180-day500 interval2025-04-12 to2025-10-09 returned a
  retention boundary. Preserve actual availability and do not fabricate a full
  sample or claim sufficient independent temporal confirmation.
- Acquire BOOM1000/CRASH1000 for2025-10-09 to2026-04-07 without reading v2 payoff.
  After final selection freezes, transfer Boom500 rules to Boom1000 and Crash500
  rules to Crash1000 unchanged. This is `cross_symbol_replication`: an unseen
  payoff sample on other symbols, not an independent future time period.
  Independence of the different symbols' underlying random generators is not
  established here; cross-symbol results cannot prove that assumption.
- Earlier conversational Boom1000 classification claims lack executable source
  evidence here and are not treated as validation. This transfer cannot be
  called wholly untouched by every prior human hypothesis. Selection does not
  use those claims, transfer features or transfer outcomes.
- Require at least170 days of authentic transfer data and report gaps. No filling
  or shifting missing timestamps. ATR/features require completed candles.

## Fixed execution

- Reuse the16 causal v1 models and their30-minute causal cooldown. Entry at exact
  M1 open one minute after M5 closed signal, ATR14 frozen at the signal.
- No take-profit barrier. Exit only at SL or exactly H minutes from entry, using
  the last included minute's close. No data after the planned horizon are used.
  The timeout close is a quote-path proxy, not evidence of an executable fill
  at the moment a clock exit is issued. Execution costs/delay remain unmeasured.
- Candidate side: spike direction (longBoom/shortCrash) or drift direction
  (shortBoom/longCrash), with original signal timestamps unchanged.
- SL{1,2,4}ATR × H{1,5,15,30}minutes ×2 sides ×16 models =384 per development symbol.
  Do not expand this grid after reading results in this protocol.
- One concurrent position per candidate/symbol. Common31-minute end purge.
  Max hold30 plus delay1 aligns with the signal cooldown, so no previous trade
  consumes a later eligible entry in development window filtering.
- Opening SL gaps precede intraminute H/L. Primary stop fills at adverse minute
  extreme and exits at that minute's end. Barrier-proxy is sensitivity only:
  worse opening for SL gaps, otherwise exactSL. No TP ambiguity exists.
- Missing entry skips issuance; missing path before exit censors unknownR and
  reserves occupancy to planned end. A completed exit is unaffected by later gaps.
- NetR=grossR−costATR/stopATR. Primary cost0.10ATR, explicitly hypothetical total
  roundtrip. Sensitivity costs0,.025,.05,.10,.20,.40ATR; primary delay1 and delay0
  sensitivity. Cost changes never change signal/exit chronology.
- Illustrative equity risk0.25% per initial stop; closed-trade drawdown only.
  Gap loss may exceed1R, particularly for trades against spikes.

## Selection and fixed primary

Fixed primary: raw CRT in spike direction, SL2ATR, time exit15minutes. Also
choose exactly one configuration per500 symbol from development; transfer its
model/side/SL/H unchanged to the corresponding1000 symbol.

Rank by mean netR−1.96×UTC-day clustered SE, requiring >=300 completed trades,
>=30 active days, no censored or invalid uncensored trades, and >=30 trades with
positive mean in each of the three development validation windows.

If no fully eligible candidate exists, choose the highest score only among
adequately sampled noncensored candidates for diagnosis and set
development_eligible=false. Never choose a one-trade fallback. If no candidate
has adequate data, record no selection and do not replace it with a tiny sample.
Expanding walk-forward selection applies adequate-data rules to its past window
without reading its subsequent validation. Never tune from transfer outcomes.

## Inference, tails and gates

Compare to CLOCK_BASELINE: deterministic feature-valid M5 opportunities every30
minutes, same side, SL, H, delay, modeled cost and purge. Comparisons concern
expectancy per trade with different turnover, not equal-exposure portfolios.

For primary/selected in each transfer symbol (up to4 unique hypotheses), report
paired UTC-day bootstrap9999, seed20261005; pointwise CI95 for meanR and difference
from clock. Also use paired circular moving7-day blocks9999, same seed, including
all calendar days. The combined one-sided centered bootstrap p is the maximum
of positive-mean and positive-difference p under both resampling schemes. Holm
correction covers all unique transfer hypotheses. Undefined empty resamples
count conservatively against significance; CI omits them with count recorded.

Report raw directional quote move, ATR median/IQR, win rate, mean/median R, PF,
count/censor/skip, holding time, closed-trade equity/DD and negative gap tail.
Report positive-winner effective sample size `(sum positiveR)^2/sum positiveR²`,
largest winner/day contribution, top1%/5% trade/day profit shares, expectancy
after deleting best5 trades and best5 days, and worst leave-one-day-out mean.
These tail deletions are sensitivities, not a selection rule. Report all three
predeclared equal-duration transfer thirds without selecting the best third.
Thirds partition the exact source interval [first M1 opening,last+1minute);
apply the common31-minute end purge at each third boundary before summarizing.
Do not interpret ATR-normalized excess as proof of timing knowledge by itself.

A candidate for new prospective paper research requires:

- Final development selection eligible (or fixed primary with adequate data and
  positive results in all three development validation windows).
- >=170 transfer source days, >=300 completed trades, >=30 active days,
  no censored/invalid trades, mean netR>=.10, PF>=1.30.
- Both meanCI lower>0 and excessCI lower>0 under day and7-day resampling,
  Holm p<.05, closed-trade DD<=10%, no illustrative ruin.
- >=30 trades and positive mean in each of the three transfer thirds.
- Worst leave-one-active-day-out expectancy>0, to reject a single-day result.

Passing means a transferable historical research candidate under assumptions.
Measured broker costs, actual fills, a new prospective cohort and monetary
profit remain NOT TESTED; no historical outcome permits live trading.
Holm controls only the declared hypotheses within this round. It does not
correct the entire sequence of adaptive research rounds or establish a global
discovery probability. Any positive result must be locked before a genuinely
prospective paper cohort; repeated historical searching cannot substitute for it.

## Provenance and freeze

Before evaluation save source, protocol and selection SHA256 hashes. Verify
data hashes when entering every cohort. Keep discovery, selection, transfer and
reused exploratory labels in outputs. Preserve failed acquisition provenance.
Changes after transfer results create another adaptive round and consume those
data as exploratory evidence; they cannot repair this round's confirmatory status.

Primary sources: [public history](https://developers.deriv.com/comparison/ticks-history/),
[execution explanation](https://traders-academy.deriv.com/trading-guides/stop-loss-execution-boom-crash-indices),
[drift, spike and indicator interpretation](https://experts.deriv.com/insights/boom-and-crash-the-drift-the-spike-and-what-a-spike-does-to-your-indicators).
