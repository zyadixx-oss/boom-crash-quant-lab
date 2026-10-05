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

## Executed quote-path payoff extension

The locked [payoff study](spike_payoff_20261004/REPORT.ar.md) now tests normalized
bracket returns. It does not replace the earlier excursion results.

- 432 configurations per symbol; first70% development, three expanding
  walk-forward selections, final30% explicitly labelled reused/exploratory.
- Additional external period:2025-10-09 11:08 UTC to2026-04-07 11:08 UTC
  exclusive;259,200 genuine M1 candles per symbol, no gaps or overlap.
- No configuration satisfied all development eligibility rules. Diagnostic
  fallbacks remain rejected, regardless of later results.
- External fixed support/resistance model: BOOM500 n=2,216, mean netR=-0.317,
  CI95[-0.371,-0.263], PF0.587; CRASH500 n=2,259, mean netR=-0.304,
  CI95[-0.356,-0.250], PF0.603.
- Crash diagnostic CRT+MSS fallback: n=341, mean netR=-0.105,
  CI95[-0.172,-0.034], PF0.673. Boom compression fallback has only one external
  trade and provides no useful population estimate.
- All six independently selected walk-forward validation folds lost on average.
  All four external gates failed. Tested zero-cost and more favorable execution
  sensitivities did not reverse the locked configurations' negative averages.
- The primary model assumes entry one minute after signal,0.10ATR hypothetical
  total cost, stop-first ambiguity, and adverse minute-extreme stop fills.
  See [interpretation](spike_payoff_20261004/README.md) before using any metric.
- Full Python validation now:121passed. Original frontend code was unchanged
  by this research extension.
- Independent standard-library scalar reconstruction matched all4,817 external
  ledger trades, including causal ATR, execution path and summary arithmetic:
  91,733 checks, zero mismatches. Baseline, bootstrap and sensitivity replay are
  outside that independent ledger audit's scope.

## Evidence boundary

The subsequent [uncapped time-exit round](spike_timed_20261005/REPORT.ar.md)
tested384 configurations per500 symbol without a take-profit cap, in both
spike and opposite direction. No final development selection qualified, and
all six expanding validation folds lost on average. The primary spike-direction
CRT with SL2ATR/time15m transferred unchanged to Boom1000/Crash1000:

| Transfer symbol | n | Mean netR | Day-block CI95 | PF |
|---|---:|---:|---|---:|
| BOOM1000 | 505 | -0.103 | [-0.206,0.006] | 0.807 |
| CRASH1000 | 484 | -0.102 | [-0.212,0.009] | 0.812 |

All four within-round transfer gates failed. The rejected diagnostic
Boom BB-squeeze fallback has a tiny positive gross0.003738R at zero cost but
negative-0.002512R already at hypothetical0.025ATR; its cost-free interval
crosses zero. It does not establish an execution-cost-aware advantage.

This round is adaptive cross-symbol replication, not fresh temporal/forward
confirmation or a project-wide multiple-testing guarantee. The requested
older500 history was only4.28days available and was not used as a sufficiently
powered external test. Authentic1000 transfer history has259,200 M1 rows per
symbol and no gaps. Independent scalar replay verifies all5,191 transfer
trades in98,852 checks with zero mismatches. Full Python tests:184passed.

The first study uses future directional excursion labels; opportunity recall
is not unique-event recall. The extension tests quote-path R, win rate, profit
factor, expectancy and a closed-trade risk illustration under declared assumptions.
Monetary net profit, actual fills, measured historical spread/slippage,
independently defined tick-spike recall, and forward/shadow performance remain
**NOT TESTED**. The older weighted
backend strategy has not been evaluated by this isolated causal study. No
parameter should be promoted to production/live based on these outputs.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`, and
`OPENED_TRADES=false`. No accounts, credentials or order execution were used.
