# Spike Hunter validation protocol v1

Frozen on 2026-10-04 before running the new outcome analysis. This experiment is
offline research only. `LIVE_TRADING`, `READY_FOR_LIVE`, `LIVE_ALLOWED`, and
`OPENED_TRADES` must all be false. No account authorization or order calls exist
in the new study code.

## Data and clock

Start with BOOM500 (upward outcomes) and CRASH500 (downward outcomes). Download
public Deriv M1 OHLC history using a frozen cutoff excluding the current minute;
save API symbol resolution, requests, timestamps, source and SHA256. Validate
numeric finite positive prices, OHLC constraints, unique epoch, UTC minute grid.
Reindex missing minutes to NaN, never fabricate prices. Require every constituent
minute for M5 and H1. Indicators and labels become unavailable around gaps.

Each opportunity is the close of a fully completed M5 candle. All features use
only information available at that close. ATR is the simple average of the last
14 M5 true ranges, including the completed signal candle. The sweep depth uses
the previous completed M5 ATR to avoid changing its threshold with the sweep.
The reference is the immediately preceding complete UTC H1 candle.

## Frozen signals

- CRT: first eligible sweep per reference H1, depth 0.10–0.65 times pre-sweep
  ATR14. Reclaim must close strictly between CRL and CRH on the sweep candle or
  either of the next two M5 candles. Greater sweep depth cancels the setup.
  Failure to reclaim within that window retires the reference. One setup per H1.
- Raw CRT issues at reclaim close. Confirmations can issue on reclaim or the
  next three M5 closes. A close beyond the sweep extreme invalidates outstanding
  confirmations; already issued signals remain in evaluation.
- MSS: directional close beyond the previous five M5 highs/lows.
- Displacement: directional body at least 1.30 times the median of the previous
  20 bodies (current candle excluded).
- FVG: directional three-candle gap at least 0.08 times current completed ATR14.
  Full variant requires MSS, displacement and FVG on the same confirmation close.
- ATR compression: mean TR5 / mean TR30 <=0.80.
- Bollinger squeeze: BB20 width (4 population SD / mean close) <=35th percentile
  of the previous 100 widths, excluding current width.
- Candle compression: mean range of last three / median of preceding 20 ranges
  <=0.65, AND mean body/range of last three <=0.35.
- Candle structure: directional rejection wick >=50% of range, directional close
  position >=65%, range <=1.20 ATR14.
- Support/resistance alignment: close within 0.25 ATR14 of the preceding 12 M5
  lows (Boom) or highs (Crash). Current candle excluded from the level.
- Standalone alternatives issue on eligible closes with a 30-minute cooldown.
  The same cooldown applies independently to each CRT variant. It is causal and
  applied before outcome labels are seen, including across split boundaries.
- Sixteen variants: five CRT ablations, six standalone alternatives, five CRT
  filters. Raw, MSS and displacement confirmations are independent; a failed
  early MSS/disp candle does not block a later qualifying full confirmation.

## Outcomes

The 12 definitions are 1.5/2/3 times frozen signal-time M5 ATR14 within
5/10/15/30 minutes. Primary definition is **2 ATR / 15 minutes**, chosen from the
earlier question, not from the new results. Positive means future directional
high/low excursion relative to the signal close, not a profitable trade and not
necessarily a discrete tick jump. Exclude the confirmation candle from labels.
Require the complete immediate M1 horizon. TTS is the end of the first hit M1
candle, an upper-bound minute estimate; the exact hit time is somewhere within
that minute. No tick precision or execution price is implied.

## Splits and inference

Chronological 70% development / 30% final evaluation based on elapsed UTC time.
Development expanding folds: train 0–40%, validate 40–50%; train 0–50%, validate
50–60%; train 0–60%, validate 60–70%. Parameters are fixed, so no fitting occurs
in any fold. Purge a common 30-minute outcome horizon before every partition
end. Earlier history is allowed for causal features. All definitions and models
share the same eligible opportunity universe requiring complete 30-minute future
coverage and finite past indicators. Final evaluation is reported once. Reading
all its comparisons makes new post-hoc winners exploratory; the next test must
use fresh untouched data.

Precision = hit signals / all eligible signals. Base rate = positive outcomes /
all eligible M5 opportunities in the same partition. Lift = precision/base rate.
Opportunity recall = hit signal opportunities / all positive M5 opportunities;
this is **not recall of unique physical spike events**. Report signal count, hit
count, occupied days, Wilson descriptive precision CI95, paired UTC-day block
bootstrap precision/lift/difference CI95 (9999 resamples, seed 20261004), baseline
CI95 and median TTS. The baseline and signal counts resample the same blocks.
Also report 6-hour blocks for the primary test as a sensitivity check.

Bootstrap one-sided p is a centered approximate test of precision minus base
rate: compare resampled centered differences with the observed difference;
include a plus-one correction and never report p=0. Apply Holm across all final
symbol x variant x definition comparisons, including small/no-signal cases as
p=1. Intervals remain pointwise, not simultaneous. These inference choices are
research approximations, with limited power for sparse setups.

Candidate gate (heuristic, never a profitability gate): >=200 final signals,
>=30 hits, >=20 days with signals, >=20 evaluated days, paired day-block lift
CI95 lower >1, Holm p<0.05, and lift>1 with >=30 signals in each of the three
development validation folds. A positive point estimate without these checks is
exploratory. No strategy is allowed to become live based on this experiment.
PnL, spread, slippage, fills, stop-loss, drawdown and forward shadow performance
are **NOT TESTED**. Keep Boom and Crash results separate.

## Previous experiment limitations corrected

The older SQLite script accepted incomplete H1 bars and one-sided reclaim,
omitted FVG minimum size and purging, and called n>=30 an adequate sample.
The older proxy script requested M5 while treating the response as M1. This
study bypasses the backend's centered rolling swing implementation and does not
reuse its trading results. Historical claims are not accepted as current facts.
