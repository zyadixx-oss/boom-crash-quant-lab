# Feature-representation ablation: preparation, not an economic declaration

The distinct question is whether replacing previously observed native-tail
increments by zero in a feature-only price chain improves a fixed raw-price
prediction/payoff task compared with the same44 inputs from raw prices. No
historical transformation, feature calculation, fitted model or payoff from
this new study has been measured. The economic runner and its premeasurement
source/data declaration still need to be completed and frozen.

The [Deriv product-team article dated29September2026](https://experts.deriv.com/insights/boom-and-crash-the-drift-the-spike-and-what-a-spike-does-to-your-indicators)
describes how an old spike dominates rolling indicators and can be obscured in
higher frames. It also cautions that filtering those moves hides defining risk.
This motivates a representation comparison; it supplies no predictive-edge
claim. The proposed chain is a specified transformation, not an identified
underlying price without jumps. Removing a detected increment removes its
ordinary component too.

## Fixed data acquisition

`acquisition_declaration.json` was saved before the full acquisition started.
The untouched `scripts/collect_spike_ticks.py` collects every UTC day from
9October2025 through4October2026 inclusive, BOOM600 then CRASH600, in ascending
date order:361days/symbol,722sources,62,380,800expected seconds. It preserves
wire pages, exact requests, retry lineage, normalized quotes and missing seconds.
It does not calculate returns/features/labels. Public-only requests, no accounts.
The initial availability probe proves only four four-second windows.

The intended study interval retains the previous native600 calendar:
`[2025-10-09T11:08Z,2026-04-07T11:08Z)` and the later
`[2026-04-07T11:08Z,2026-10-04T11:08Z)`. Full UTC source days include padding
outside that interval; padding is not extra evaluation exposure. Collection
does not guarantee completeness or enough available model opportunities.
The dates were exposed to previous M1 studies; this new representation remains
adaptive retrospective research, even after a chronological split.

## Causal representation and chronology

For one supplied, frozen detector and each maximal exact1-second run, anchor
`L0=0,Q0=1`. For `r=log_return(original_previous,original_current)`, retain
zero iff `native_side*r > 10*training_median_abs_return`, otherwise retain
the original signed `r`. Accumulate in chronological Neumaier compensated
order and set `Q=exp(L)`. Nonfinite/zero exponentiation is an explicit failure;
no clipping, hidden rescaling, interpolation or return across a gap.

The old round8 detector was calibrated through13June2026. Reusing its scale in
the October–April training period would leak future-estimator information.
It remains immutable and will **not** be reused here. A later economic study
must fit the same median formula only on exact consecutive pairs wholly inside
each declared training interval, then freeze that detector. Each fold must
rebuild its own chain; a chain made under another fitted detector cannot carry
forward. Training preprocessing can use its own training prefix and is not
described as historical live issuance. No evaluation quote may enter the fit.

Raw and transformed M1 bars use the same exact seconds00..59, with identical
past-only completeness masks. Missing/partial minutes stay unknown. The runner
must reindex bars to the frozen full planned UTC grid, including wholly missing
leading/trailing minutes; helper grids bounded by observations cannot erase
that exposure. Every
gap starts fresh rolling histories for **both** representations. Run anchors
cannot become market TR, FVG or sweeps. The original44 H4/H1/M15/M5/M1 inputs
remain unchanged, including `log1p_large_bar_age`. If no qualifying transformed
large bar occurs, that age stays unknown; no zero substitution or feature
deletion is permitted to obtain more signals. A common valid-row intersection
cannot depend on any future target or outcome completeness.

## Numerical validation before historical computation

The six-day synthetic integration checks passed, but an additional180-day
synthetic decay/growth test found a real conditioning defect in the inherited
rolling BB standard deviation. For the Boom-decay fixture, changing only the
transformed price gauge produced42,381 feature-cell differences above1e-6,
with maximum absolute error0.38389408. The initial exit1 output is preserved in
`numerical_failure_initial.log`; no historical feature/payoff had been measured.

The new integration adapter now computes the same
`W=4*std(close20,ddof=0)/mean(close20)` independently within each window, dividing
that window by its own positive maximum and using compensated scalar sums for
centered variance. Windows, prior100median/quantile, feature names and tolerance
are unchanged. Raw and transformed representations both use this arithmetic.
The full original validity formula is reconstructed so the old BB calculation's
finite-state flag cannot conceal repaired rows. Ancestor research code and its
saved outcomes are unchanged.

The same180-day tests now pass at1e-6 without changing validity masks: maximum
all44 gauge differences are6.7363892242156e-12 for Boom and
6.3336003108816e-12 for Crash. Independent50-digit Decimal window calculations
check first/middle/last valid positions; missing/constant windows retain their
specified semantics. Representation plus integration tests pass87cases.
These are implementation checks on synthetic inputs, not profitability evidence
or a proof of arbitrary numerical conditioning for every possible price path.

## Training boundary implemented with synthetic verification

`jump_learning.py` now provides training-only detector calibration, exact shared
common-clock label membership and paired fixed ridge fitting. It requires at
least1,000 completed common training labels, records unknown labels separately,
and enforces issue+31minutes<=training end plus the16-minute planned path and
one-second exit allowance. Both models use the same original-quote targets and
their own training-only standardizers. Native direction is fixed by symbol.
Issuance has no labels or fitting argument and uses raw ATR exclusively.
The pure module passes63 synthetic tests; actual historical fitting is NOT TESTED.

`verify_jump_tick_sources.py` independently reconstructs original wire pages,
same-request retry chains, canonical clean CSV and unknown-second ranges with
stdlib arithmetic, one day at a time. Its77 synthetic tests pass. Complete
mode refuses to decode quotes before all722 declared manifests exist. Metadata
snapshots explicitly have `passed=null` and `completeness_passed=false`.

## Economic boundary still to be implemented and frozen

The representation experiment must retain original quotes/raw M5ATR for labels,
stops, costs and exits. Transformed ATR belongs only to feature calculation.
The proposed primary comparison retains fixed native SPIKE direction,
SL2rawATR/H15/delay1minute/cost0.10rawATR, the existing strict-next-tick replay,
UTC00/30 issuance, one-open exposure, ridge penalty0.1 and trainingquantile0.75.
This is not a horizon, stop, latency or direction rescue of failed models.
The raw44 reference must be rebuilt from these same ticks, not taken directly
from older broker-candle files. Retain a common-availability clock control and
both symbols in one fixed two-candidate multiplicity family. Restricting the
initial question to native SPIKE does not establish superiority over DRIFT.
No changing features or dates after seeing
availability/payoff, and no model promotion by a posthoc fallback.

The integrated economic runner/declaration must still specify older-period70/30, expanding40/50/60%
fits with next10% validations, final70% fit, purge31minutes, paired reference
inference, day/week uncertainty and multiplicity. It must carry forward all
existing development and qualification gates. At minimum PF>=1.5,
>=1,000completed held-out paths/model/symbol,>=60active held-out days are
required without pooling symbols. Missingness, undefined inference or insufficient
availability prevents qualification. Passing on exposed history cannot establish
fresh OOS or prospective evidence. Actual CFD fills/costs and cash profit
remain NOT TESTED.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false`. The full user objective remains active and unmet.
