# Round12: frozen SHORT1 issuance, tick-versus-M1 execution fidelity

Declare before decoding any round12 historical CSV, computing a new path or
metric. Metadata/JSON reading and file hashing are allowed before declaration.
The executable declaration records UTC time and pins exact sources, ancestors,
model/issuance identity and science bytes. All existing frozen artifacts stay
unchanged. LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES remain false.
Public data and offline quote arithmetic only; no account, token or order API.

## Objective and exposure

The complete objective remains combined H4/H1 + M15/M5/M1 net PF>=1.5 with a
large sample:>=1000 completed held-out one-open simulations and>=60 active
held-out days per model/symbol, robust uncertainty and measured execution/cost
and prospective evidence. This pilot cannot replace those requirements.

Round11 fitted SHORT44 on SHORT1 and LONG44 on LONG15, then evaluated both on
SHORT1. All development selections and economic gates failed. Its M1 proxy
uses an exact next-minute opening, whole-minute adverse extreme for a stop,
and included-minute close for timeout. This leaves a distinct execution
question: does exact quote ordering materially change that fixed policy?

Keep the exact saved later180-day positive SHORT44/LONG44 issuance and saved
CLOCK issuance of round11. No new features, predictions, fitting, score cutoff,
model choice, indicator, direction set or parameter grid. Prices, one-minute
payoffs and the reused tick ordering are already known from prior research.
New policy replay does not create fresh strategy price OOS, a label holdout,
prospective evidence or an adjusted discovery. It can check an approximation.

## Immutable models, sample and sources

Native BOOM600 and CRASH600 separately, each SPIKE and opposite DRIFT.
SHORT44/LONG44 use the same44 closed H4/H1 + M15/M5/M1 inputs; CLOCK is the
saved common eligible half-hour clock. Model training ended at older70% before
all tick dates. All12 scheduled dates form one post-fit known-history diagnostic
cohort; no40/50/60/70 tick refitting, sample selection or calibration occurs.
The four final-day clusters of rounds9/10 belonged to their tick-feature
training splits and do not define this pilot's unchanged older-M1 model cohort.

Use the exact24 existing round7 tick sources:12 full UTC dates per symbol,
2026-04-11,04-26,05-12,05-28,06-13,06-29,07-15,07-31,08-16,09-01,09-17,10-03.
Use exact round11 later180-day M1 source bytes and saved12 issuance CSV streams.
Before any quote decoding, pin round11 declaration/selection/results/PASS audit,
its science lineage and the signal/source descriptors; also round7 declaration,
tick_sources/results/PASS audit, normalized tick CSVs and raw/page/manifest
provenance. Source normalization is anchored to the passed round7 audit, not
claimed to be fully rerun here. No recollection, interpolated quotes, fill or
favorable source/date substitution. Check all pins before and after execution.

During evaluation compare complete60-second UTC-minute OHLC from ticks with
its exact frozen M1 source, rtol1e-12/atol1e-8; mismatches refuse evaluation.
Incomplete minutes remain unknown; matching complete-minute OHLC does not
establish gap-free quotes. Preserve actual sorted unique whole-second quotes,
source coverage, gaps, provenance and recovered errors. CSV/path corruption,
changed bytes, canonical-path/symlink escape, invalid timestamps/prices/signals
or model-policy semantics must refuse. Every signal retains saved time, frozen
ATR, side, variant and score; no rebuild of feature-valid universe is claimed.

## Exactly fixed policies and paired coverage

Tick primary: SL2 frozen M5ATR14, nominal request at issue+1minute, expiry at
nominal request+1minute, assumed round-trip cost0.10ATR, no take-profit,
max gap1second and stop latency1observed tick. First observed quote STRICTLY
AFTER nominal request determines entry and stop; with the observed1second
cadence it is issue+61seconds. Missing entry reserves nominal occupancy.
Inspect subsequent quotes through nominal expiry inclusive. The first adverse
crossing triggers exit at its next observed quote, without barrier cap.
A crossing at nominal expiry takes priority over timeout. Otherwise exit is
the first quote STRICTLY AFTER expiry; at continuous cadence issue+121seconds.
Nominal expiry never moves to actual entry+hold. Required missing seconds
before a trigger/exit/timeout censor the path with no R; missing quotes after
completed exit are irrelevant. Unknown paths reserve planned occupancy;
known paths use actual exit time. No optimistic trigger-quote sensitivity.

M1 comparison is the unchanged round11 adverse_extreme timed policy:
exact M1 opening issue+1minute, SL2ATR, hold1minute, cost0.10ATR, adverse
whole-minute extreme after a stop trigger or included-minute close timeout.
Do not fabricate barrier fills. Both engines have identical per-observed-day
start/end, saved issuance and planned31-minute purge. A realized early exit
never rescues a planned purge. Tick additionally requires its exit-successor
allowance within the day. No unsampled-day quote bridging or concatenation
creating fictitious continuity. All entry/stop/expiry quote differences change
jointly; tick-minus-M1 is not attributed solely to stop slippage.

Every stream nominally has at most12*48=576 half-hour opportunities; common
31-minute per-day purge gives at most12*47=564 completed paths and12 active
days. This cannot pass1000/60 evidence requirements, irrespective of PF.
No pooling across models, modes, symbols, days, targets or duplicate controls.

## Reporting and uncertainty

Report each of12 observed days and aggregate separately per symbol/mode/model:
issued/filled/completed/censored/missing-entry/invalid/purged/overlap, net/gross
mean/medianR, PF, win rate, active days, holding time and illustrative closed
trade risk0.25%. UnitsR divide by stop distance2ATR; assumed cost0.10ATR=0.05R.
Save exact copied issuance, both ledgers, per-day records, model/source hashes
and metadata. Matched comparisons inner-join only uniquely identified completed
finite paths on saved signal_time and report unmatched/unknown counts. Such
matched-known means are descriptive subsets, not whole-policy profit.

Per-day summaries are point descriptives; no per-day CI is requested.
Aggregate descriptive PF/mean and matched-difference CI95 use9999 paired resamples of the
12 scheduled UTC day units, seed20261005, including scheduled zero-exposure
days. Never represent unsampled days as observed zeros, and do not claim
contiguous-week inference from discontinuous dates. Undefined empty/no-loss/
nonfinite means/ratios/differences remain null, with explicit valid/undefined
replicate counts. Finite-draw intervals do not suppress those exclusions.
Model-minus-CLOCK and SHORT-minus-LONG comparisons are per-completed-path mean
differences under identical scheduled-day resampling, not equal portfolios.
No p-value, discovery correction or superiority claim is made here.

All results remain diagnostic/rejected. Prior round11 development ineligibility
is binding; LONG44/CLOCK are reference_only. historical_candidate,
supports_expected_pf_1_5, live_candidate and goal_achieved remain false even if
a point PF>=1.5 appears. A null PF is unknown, not zero or infinity. This pilot
must not be expanded solely until a favorable sample emerges.

## Execution meaning, verification and failure preservation

[Deriv trading terms](https://deriv.com/terms-and-conditions/trading-terms)
clauses3.3/3.4 describe quotes after server processing for entry/exit.
The [official stop explanation](https://traders-academy.deriv.com/trading-guides/stop-loss-execution-boom-crash-indices)
distinguishes a stop trigger from its executable price. A next-observed-public
quote is an arithmetic proxy, not measured processing, bid/ask, actual latency,
financing, account fills or cash profit. No new market-process independence or
martingale theorem is claimed. Actual money profit, broker fills, measured
costs and prospective paper performance are NOT TESTED.

Freeze source/protocol/tests after meaningful synthetic checks, then declare
metadata once and evaluate once with exclusive stdout and authoritative process
handle. Observation timeout is not terminal and cannot authorize a restart.
Refuse frozen-output overwrites. Failed historical execution/audits must be
preserved with logs/hashes before repairs; no outcome-driven science rewrite.

An independent stdlib verifier will check captured source/artifact hashes,
exact saved signal membership/identity, both scalar quote paths and unknown
occupancy, day bounds/purge, point metrics/matched membership and refusal gates.
It will explicitly exclude full raw-wire renormalization, regeneration of44
features or complete eligible universe, model refitting/predictions, bootstrap
draw/CI regeneration or inferential validity, independence and broker profit.
Synthetic software CI does not establish the financial objective. Any failing
audit must be preserved before changing only verifier logic/tests.
