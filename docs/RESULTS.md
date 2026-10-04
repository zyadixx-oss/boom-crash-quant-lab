# Results — actual offline Spike Hunter study

Executed 2026-10-04 using official public Deriv M1 candles. See
[full Arabic report](spike_hunter_20261004/REPORT.ar.md),
[frozen protocol](SPIKE_HUNTER_PROTOCOL.md), and
[all metric rows](spike_hunter_20261004/metrics.csv).

- BOOM500 and CRASH500: 259,199 valid M1 candles each, 2026-04-07 11:08 UTC to
  2026-10-04 11:08 UTC exclusive. One missing minute per symbol remains unfilled.
- Sixteen causal variants, 12 excursion definitions: 1.5/2/3 M5 ATR in 5/10/15/30m.
- Chronological 70/30, three expanding development validation windows, common
  30-minute label purge, paired day-block confidence intervals and Holm correction.
- Primary 2 ATR/15m BOOM500 raw CRT: 59/229 hits, precision25.76%, base22.22%,
  lift1.159, day-block CI95[0.926,1.392]. No confirmed timing advantage.
- Primary CRASH500 resistance alignment:195/639 hits, precision30.52%,
  base21.61%, lift1.412, day-block CI95[1.268,1.551]. Strongest research candidate.
- Post-hoc ATR matching: Crash resistance lift1.196 CI95[1.073,1.312];
  Boom CRT lift1.076 CI95[0.857,1.293]. The diagnostics use the same historical
  period and cannot serve as independent confirmation.
- An independent199,999-resample check supports the raw Crash resistance result;
  it does not test profitability or arrival of independently segmented tick spikes.
- Python validation:46passed. Independent scalar replay matches prices, causal
  features, signal timestamps,24,096labels, all384final comparisons and gates.

## Evidence boundary

These are future directional excursion labels, not trade returns. Opportunity
recall is not unique-event recall. Cost-aware PnL, win rate, profit factor,
expectancy, drawdown, fills, spread, slippage, independently defined tick-spike
recall, and forward/shadow performance remain **NOT TESTED**. The older weighted
backend strategy has not been evaluated by this isolated causal study. No
parameter should be promoted to production/live based on these outputs.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`, and
`OPENED_TRADES=false`. No accounts, credentials or order execution were used.
