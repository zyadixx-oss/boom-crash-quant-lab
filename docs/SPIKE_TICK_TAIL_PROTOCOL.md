# Proposed round8: fixed tail-tick mechanism pilot

This protocol is exploratory infrastructure, not an executed result. It must be
hashed together with its runner and helper before any detector calibration or
event/hazard measurement. It reuses round7's deterministic historical sample;
it cannot be called fresh strategy validation. No parameter is promoted.
LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED and OPENED_TRADES remain false.

## Hypothesis and evidence boundary

Previous studies already tested CRT confirmations, compression, both directions,
uncapped exits, 44 closed-frame inputs and nonlinear interactions. Their M5 age
is time since a large-range candle, rather than time since a tick jump. This
pilot tests one previously unmeasured mechanism: whether age since a detected
tail tick changes jump incidence, jump size and non-jump drift. It does not
assume a late jump is due or infer generator independence from a null result.

Use only the 24 immutable BOOM600/CRASH600 day sources in round7 after source
validation. The 12 discontinuous scheduled days remain discontinuous. Retain
gaps, initial unknown age and unobserved calendar intervals; never interpolate
or count those intervals as waiting without an event.
Read each preserved quote string with Python float conversion to IEEE64,
requiring a finite positive result. Do not add decimal rounding or use the
default pandas numeric CSV parser. The original Decimal/raw/CSV integrity was
audited in round7; this fixed conversion is part of the new pilot's calculation.

## Fixed detector and chronological partitions

For each symbol, order observed quote rows by UTC timestamp. Calibration uses
the earliest floor(0.40 * n) rows only. Median absolute log return `m` is computed
from positive finite quote pairs exactly one second apart in that calibration
prefix. It is never recomputed in later folds or held-out rows.

Define `d[t] = side * log(q[t] / q[t-1])`, with side +1 for Boom and -1 for Crash.
A tail event is exactly `d[t] > 10*m`; equality is not an event. The multiplier
10 is a fixed engineering definition, not an optimized profitable setting.
Zero/nonfinite scale or fewer than 100 calibration events makes this detector
inadequate for the pilot; save that finding and leave later segment measurements
NOT TESTED. Do not replace it with a different multiplier,
quantile, nominal event frequency or favorable symbol. Call detected events
tail-tick proxies, not a verified census of physical spikes.

Development ends after floor(0.70 * n) rows. Three chronological validation
segments are [40%,50%), [50%,60%), [60%,70%); the final segment is [70%,100%).
Boundaries depend only on observed row count, never returns/events. This is a
time-ordered observed-second split, not a 70/30 split of the intervening calendar
span. Report exact timestamp bounds, valid increments and scheduled days for
each segment. No refit, tail-threshold sweep or split adjustment is permitted.

For an increment ending at t, pre-event age is t minus the most recent event
timestamp detected through t-1, within the same consecutive observed segment.
The first increment after an event therefore has age1 second. The current event
cannot enter its own age covariate. The first event following unknown age establishes the state
for later increments, but has unknown prior age. A missing second resets age
and the previous event mark to unknown until a new event. Consecutive ticks
across midnight remain consecutive; disjoint sampled dates do not.

## Measurements and a bounded falsification rule

Use exactly two pre-event age bands: age <600 seconds and age >=600 seconds.
The number 600 defines a band boundary, not a calibrated probability or count
target. In each band report observed exposure seconds, detected events, hazard
(events/exposure), event mean d, non-event mean d and unconditional mean d.
The identity `mean d = hazard*event mean + (1-hazard)*non-event mean` must hold
when both groups exist; missing conditional terms stay null.

Also report calibration detector counts and directed-return quantiles, unknown
age exclusions, missing-pair exclusions and positive/negative/zero increments.
Do not convert a conditional log increment into broker profit or a trade entry.

Primary diagnostic comparison is older/younger hazard ratio in the pooled
development validation [40%,70%). Repeat it descriptively in the final30% and
show all three individual validation segments. Intervals use 9,999 paired iid
draws of observed scheduled UTC-day units, seed20261005, retaining zero band
exposure for a day. Report observed cluster count, undefined draws and the
discontinuous-sample limitation. No contiguous-week or independent-tick
inference claim is made.

With at least100 events in each band and all9,999 ratio draws defined,
an upper95% hazard-ratio bound below1.5
rejects the large overdue-event effect in that segment under this detector.
Low event counts or undefined bounds mean insufficient evidence. A ratio near1
does not establish independence, and a large ratio does not establish PF.
Do not select another age boundary, horizon or event definition after results.
These intervals condition on the fixed calibration detector; uncertainty in its
training median is not regenerated. Undefined draws never justify falsification
by discarding them and using only the remaining finite ratios.

## Connection to the combined-timeframe profit objective

No strategy fitting or payoff selection occurs in this pilot. A subsequent
separately declared comparison could retain all44 H4/H1+M15/M5/M1 inputs and
add only causal tick age and the last observed event mark. It would retain the
same availability clock, directions, training discipline, exits and costs.
Pilot evidence must justify that experiment and its required data volume first.

The actual objective remains modeled netPF>=1.5 and >=1,000 completed held-out
quote-path simulations per symbol/model, development eligibility, uncertainty
versus clock/reference and separately measured execution/cost and prospective
paper evidence. Tail-event hazard alone satisfies none of those payoff gates.
These previously read price periods are adaptive research even when a new tick
mechanism is declared before measurement. Monetary profit and real trading
remain NOT TESTED and disabled.
