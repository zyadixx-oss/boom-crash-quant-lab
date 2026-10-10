# Fixed pre-issued region family: source-only design review

Status: no historical region, touch, return or payoff calculation performed by
this reviewer. These are falsifiable geometric hypotheses, not demonstrated
Boom/Crash alpha. Previous negative CRT/learned/tick results remain binding.
Freeze executable science, tests, source hashes and chronology before execution.
All four execution/live flags remain false.

## Shared causal contract

Use original native quotes and exact complete M1 candles. Issue only at UTC00/30;
each issue keeps four audit rows, including every unknown/rejected candidate.
Latest closed H4/H1/M15/M5 rows must be complete and belong to the same exact
one-second observed run as the latest M1 ending at issue. An invalid latest
closed frame cannot fall back to an older valid frame. A true missing native
second restarts all rolling/EMA/context histories; midnight/chunks alone do not.

Let s=+1 for Boom and−1 for Crash, C=latest complete M1 close, A=raw closed-M5
mean TR(14). Require finite A>0, native H4/H1 bodies>=0 and native latest M5
body>0. M1 supplies price freshness only: no native M1-body or recent-spike gate.
These context conditions can describe past spikes; they do not establish advance
information. M15 setup is family-specific below. No pivot requires future bars.

All zones must have positive finite low<high and lie wholly on the adverse side
of C: upper<C for Boom; C<lower for Crash. Otherwise retain a rejected row with
unknown zone boundaries. Never clip, shift, widen or force a region to qualify.
Exterior invalidation is lower−0.25A for Boom, upper+0.25A for Crash. Geometry,
invalidation and raw A are frozen at issue and never follow subsequent prices.

## Exactly three families and one context control

| Variant | Fixed M15 setup and immutable band |
|---|---|
| CRT_RETEST | Latest closed M15 sweeps its reference H1 adverse boundary by0.10–0.65A and closes strictly inside that H1 range. Reference H1 closed **before the M15 opened**: its opening is floor_hour(M15.opening)−1hour. Boom band [CRL,CRL+0.10A]; Crash [CRH−0.10A,CRH]. |
| FIB_RETRACE | Last four closed complete M15 bars, all from the current run. L=min lows/H=max highs; first occurrence breaks ties. Boom requires low-bar index<high-bar index; Crash high-bar index<low-bar index. Same-bar extremes are refused; latest M15 body must point natively. With R=H−L>0, Boom [H−0.786R,H−0.618R]; Crash [L+0.618R,L+0.786R]. |
| TREND_RETEST | Run-reset M15 EMA20 (`span=20,adjust=False,min_periods=20`), requiring20 complete preceding/current M15 closes. Native EMA slope versus preceding M15>0 and native(close−EMA)>0. Band [EMA−0.05A,EMA+0.05A]. |
| CONTEXT_GEOMETRIC | Same closed-frame/native H4/H1/M5 context and fresh M1; M15 completeness required, no family setup. Band endpoints C−s×0.55A and C−s×0.45A, sorted. This is a context-conditioned waiting-region control, not unconditional market entry. |

No MSS/FVG/RSI/volume, alternative ratios, swing confirmation, direction search,
or horizon/width grid is added. Raw prices remain the economic price scale;
the separate tail-removed representation experiment is not mixed into regions.

## Agreed replay semantics

The separate root-owned replay activates at issue+1minute. Monitor invalidation
immediately after issue, including before activation. First observed in-zone
quote strictly after activation and at/before issue+15minutes is the touch.
Entry uses the exact next-second observed market quote unconditionally; a jump
away from the zone cannot cancel a losing/delayed entry retrospectively.
Expiry is touch+15minutes; stops use2A from actual entry, primary next-quote
stop/timeout exit and assumed0.10A round-trip cost. Maximum planned response is
issue+30minutes+1second, with the inherited31-minute partition purge. Missing
required observations remain unknown/censored, never interpolated. One pending
region/order/position per symbol/variant, without replacement or Martingale.
No claim of executable limit fills, measured latency/costs or cash profit.

## Chronology, denominators and gates

Use the full declared calendar interval and fixed70/30 time split, plus fixed
40–50/50–60/60–70 development descriptions where the final protocol requires.
Freeze exact UTC bounds/endpoint purges before reading outcomes. Preserve old
price exposure: reserved calculations on already seen history are retrospective
chronological evaluation, not newly independent OOS or prospective evidence.
No development failure can be rescued by selecting a favorable later partition.

Report separately per family/symbol: all scheduled rows; unavailable/context/
setup/geometry rejections; issued zones; occupied/purged; expired/invalidated;
touches; missing/censored entries/paths; completed trades; PF/net mean/day counts,
CI95 and median issue-to-touch/entry-to-event times. Unknown denominators remain
visible. One observed spike can count at most once per family, with issue and
entry strictly preceding it; events already seen before entry are not catches.
Event-definition labels, primary precision/base-rate/lift/recall denominators
and censoring rules must be fixed in the executable protocol. Do not present a
conditional touch rate or many overlapping event windows as trade precision.

Compare all three region policies against CONTEXT_GEOMETRIC on their fixed
calendar opportunities with shared UTC-day/week resamples. Different candidate
acceptance/touch rates make this a whole-policy comparison, not pure isolation
of geometry. Do not pool variants, sides or repeated controls as independent
trades. Predeclare a twelve-test family (three hypotheses×two symbols×two held-out periods), with
conjunction/inference correction retained from the project contract.

Qualification still requires PF>=1.5, >=1000 completed held-out one-open paths
and >=60 active held-out days per model/symbol, plus all existing development,
uncertainty, costs/drawdown and refusal gates. A small successful region sample
cannot qualify. Actual fills/measured costs/cash profit/prospective paper remain
NOT TESTED. Product articles describe chart concepts, not proof of this edge.
