# Proposed next pilot: native-adverse tick-path variation

Historical calculation, model fitting, strategy comparisons and profit for this
hypothesis are NOT TESTED. Only its pure causal feature and synthetic cases have
been implemented. Do not treat feature tests as predictive evidence.

Round9's fixed tail-age/mark policy has negative final model point means and
no positive incremental bound. That does not justify acquiring a larger sample
solely for that policy. This is a distinct adaptive hypothesis about path risk,
not a retuned age threshold or a proven replacement.

## Single feature, fixed definition

For decision time t, use exactly600 consecutive one-second log-price increments
ending at t-1. This requires601 finite positive quotes at t-601 through t-1:

```
native_adverse_semivariance600(t) =
  mean(min(native_side * log(P_i / P_(i-1)), 0)^2) / frozen_scale^2
```

Native side is Boom+1 and Crash-1 for BOTH SPIKE and DRIFT models. The feature
measures native-adverse variation, not each strategy direction's adverse risk.
The scale is the unchanged first40% median from the independently audited
round8 detector. No recalibration, window grid, clipping or side selection.

The current quote/event/return is excluded and need not exist. A missing second
inside the prior window makes the feature unavailable; recovery requires a new
contiguous601-quote history. No interpolation or stale carry. UTC midnight alone
does not reset a genuinely consecutive history. Descriptor rows are indexed by
last observed quote time+1second; only exact joins at closed M5 decisions count.

Compare fixed RIDGE44 with RIDGE45 on identical common-available rows and clocks,
retaining all44 closed H4/H1 + M15/M5/M1 inputs. Do not add round9 age/mark
features, change the cadence or drop invalid original rows. CLOCK uses the same
availability and one-open replay. Both symbols and both directions stay declared.

## Required declaration before calculation

An executable runner/protocol/helper/tests and all inherited science/source,
round9 result and passed independent-audit hashes must be frozen before new
feature calculations or fitting. No actual historical run is authorized by this
document alone. The task owner must first verify implementation and freeze an
exclusive-create declaration; no inherited files or results may be edited.

If the twelve-day sources are reused, explicitly label the entire pilot adaptive
known-history research. Round9's final payoff labels have already been exposed;
freezing a new selection does not make those labels or prices fresh OOS.
Retain prefix clipping, observed-row40/50/60/70 cutoffs, inherited SL2ATR/H15,
entry+1minute/next quote, modeled cost0.10ATR,31-minute planned purge, ridge0.1,
training-only scaling/q75 positive cutoff and minimum1,000 finite overlapping
training targets. Do not count those targets as strategy completions.

Save matrices/models/cutoffs/common clocks and distinct individually replayed
targets versus jointly replayed one-open ledgers. Unknown payoff stays unknown.
Report absolute net mean/PF, incremental45-minus44/clock, all fixed folds and
conditional uncertainty with actual observed-day denominators and undefined
draws. No profit discovery or week/prospective inference from four final days.

## Decision and the user's target

Positive absolute payoff and positive incremental payoff with stable chronology,
known outcomes and appropriately bounded uncertainty are needed to justify a
separately declared larger study. Negative or inconclusive economics must not
be repaired with favorable windows, exits, symbols, directions or further dates.

The user's target remains observed net PF>=1.5 with>=1,000 completed held-out
one-open paths per model/symbol, plus inherited>=60 active held-out days and
development/reference/stability/multiplicity/cost/drawdown gates. Expected
PF>=1.5 is a stronger claim requiring suitable lower bounds reaching1.5.
Current sparse history cannot meet these requirements. A larger study must size
dates once from development strategy completion rates before held-out outcomes;
200 usable source days are only a calendar minimum, not a sample guarantee.

All four flags LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES remain
false. Account access, order placement, measured broker fills/costs, monetary
profit and prospective paper performance remain outside this pilot and
NOT TESTED. No strategy promotion is permitted from prepared feature code.
