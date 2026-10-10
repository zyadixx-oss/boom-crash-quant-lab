# Round7: declared tick execution diagnosis

The high-profit objective remains netPF>=1.5 on>=1,000 completed held-out
simulations per symbol/model with H4/H1+M15/M5/M1. This diagnostic does not
replace that objective or promote any development-rejected round6 model.
LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES must remain false.

## Why investigate ticks

Round6's M1 adverse-extreme stop consumes the whole stop-minute and can include
prices after the first crossing. Its barrier alternative fabricates an intrabar
barrier fill across possible jumps. At unchanged modeled cost all12 saved
barrier PFs are below1.5; that shortcut does not satisfy the target.

The official [trading terms](https://docs.deriv.com/tnc/trading-terms.pdf),
sections3.3/3.4, describe the next tick after server processing for Derived CFD
entry and Boom/Crash exit. Sections4.1/8.4.2 describe spread and server delay.
This motivates a strictly-next-observed-quote proxy; it does not measure actual
server processing, bid/ask, liquidity, fills, swaps or cash profit. The
[official stop-loss explanation](https://traders-academy.deriv.com/trading-guides/stop-loss-execution-boom-crash-indices)
also distinguishes trigger levels from executable prices after jumps.

## Predeclared sample and immutable inputs

Both BOOM600 and CRASH600;12 fullUTCdays in2026:
April11/26, May12/28, June13/29, July15/31, August16, September1/17, October3.
These are indices floor(k*175/11),k=0..11, among the176 full days April11 to
October3. Three days after the first full fresh day avoid the existing H4
warmup; spacing is deterministic, never chosen by a price or outcome.
All12 frozen BOOST44/RIDGE44/RIDGE19×SPIKE/DRIFT models and four clock controls
use exactly their saved round6 issuance or unchanged44-feature eligible clock.
Pin model/selection/results/code/M1/issuance bytes before collecting bulk ticks.
No refit, input change, new score cutoff, signal discovery or favorable subset.

At most12×48=576 opportunities per model/symbol. This cannot establish the
1,000-simulation target, regardless of any attractive small-sample result.
Round6 development rejection remains binding. These dates reuse known M1
history in adaptive research; new tick ordering is not a fresh strategy-OOS or
prospective experiment. No full180-day tick coverage claim follows from probes.

## Public acquisition and validation

Canonical endpoint wss://api.derivws.com/trading/v1/options/ws/public.
Tiny engineering probes observed native600 one-second ticks and a1000-row
runtime cap even when requesting5000; documentation has no count maximum.
Legacy endpoints failedHTTP520 before requests; equivalence remains NOT TESTED.
Use count1000, explicitstart=day00 and end=dayend-1 on every request; decrement
end to oldest-1. An end-only old request silently returned current ticks because
start defaulted to the current-day window. Reject every out-of-range response.
No adjust_start_time, authentication, account or order request.
Only active_symbols and ticks_history are permitted. Pace sequential requests.

Preserve raw wire pages, exact requests, returned bounds/counts, hashes,
successful-page chain, errors and same-request retry evidence. Recovered rate
limits/transport failures require an exact bounded successful retry; unmatched
or semantic errors fail acquisition. Resume only matching config/checkpoints.
Sorted whole-second positive quotes; equal duplicate epochs may be removed,
conflicting duplicates refuse. Reconcile each UTC86400-second grid and retain
declared gaps, never interpolate. Checkpoint/raw/clean bytes remain local.

Before any payoff, freeze all24 source manifests and raw/page/clean hashes.
Independently count missing seconds and compare complete60-tick UTC-minute
OHLC to the frozen round6 M1 source with rtol1e-12/atol1e-8. Incomplete minutes
remain unknown; a complete-minute price mismatch refuses paired evaluation.
Matching OHLC alone never establishes complete tick coverage.

## Fixed quote-path policy

Stop2 frozen M5ATR14, nominal request at signal+1minute, nominal deadline at
request+15minutes, modeled round-trip cost0.10ATR, no take-profit and one open
simulation. First observed quote strictly after entry request, within1second,
sets entry/stop. Missing entry reserves planned occupancy.
Inspect chronological quotes after entry through the nominal deadline inclusive.
The first adverse stop crossing triggers an irrevocable exit; primary exit is
the next observed quote, preserving overshoot or recovery without barrier cap.
Otherwise timeout uses the first quote strictly after the nominal deadline.
The tick after a deadline crossing has SL priority. Required quote continuity
is1second; any earlier relevant missing quote or missing successor/timeout is
censored with no R. Ignore missing quotes after an already known exit.
Censored paths reserve occupancy through planned_end. Completed paths use
actual quote exit time. Fixed31minute planned purge plus required successor
inside the partition, independent of realized earlier exits.

One optimistic sensitivity uses the trigger quote itself for stop fill; entry
and timeout remain strictly-next. Costs0/.05/.10/.20ATR reuse fixed paths and
do not select a new primary. Compare frozen OHLC stress and barrier on the
same sampled signals. Full tick proxy changes entry, stop placement and timeout
as well as stop fill; its difference is not attributed solely to stop slippage.

## Descriptive uncertainty and refusal of promotion

Report n/censoring/missing-entry, PF, win rate, mean/medianR, actual holding
time, trigger overshoot, clock comparisons and matched-signal tick-vs-OHLC
differences. Descriptive95% intervals use9999 paired draws of the12 scheduled
day units, seed20261005, including scheduled zero-exposure days. Do not treat
unsampled days as observed zero-return days. Undefined no-loss/empty ratios
stay null with explicit counts. No weekly contiguous-block assertion or
discovery p-value is made for the discontinuous small sample.
Drawdown uses illustrative0.25% risk per closed path and excludes floating
drawdown and contract sizing. Every result is diagnostic/rejected; no promotion,
money profit, measuredfills/costs or prospective paper claim. A full-size,
separately declared validation and cost/fill evidence remain outstanding.
