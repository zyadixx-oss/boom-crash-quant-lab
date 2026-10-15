# Stable positive net profit: user objective revision

On 9 October 2026 Riyadh time, the user explicitly replaced the minimum PF1.5
target with: **ربح صافٍ موجب ومستقر، ولو كان PF أقل من 1.1**.

For subsequent research the point condition is now **net PF > 1 and mean net
R > 0**, with no minimum PF1.1 or PF1.2. This changes the requested payoff
magnitude, not the requirements for credible positive evidence. The source
review at commit `9c45632` and all earlier frozen declarations/results retain
their original PF1.5 contract and rejection labels.

Retain the existing combined H4/H1/M15/M5/M1, native Boom-rise/Crash-fall scope,
separate symbols/models/applications/periods, no pooling, and one pending/open
exposure. Existing regions, training rules, quote replay, splits, purges,
unknowns, costs and model parameters are not changed by this assessment.

The unchanged evidence requirements include:

- At least 1,000 completed held-out paths and 60 active held-out days per
  symbol/model/application/period; not 1,000 pooled overlapping records.
- Passing chronological development requirements before held-out promotion.
- Positive daily and weekly net-mean lower bounds, PF lower bounds >1, and
  required reference-advantage lower bounds; existing multiplicity correction.
- Positive chronological thirds with their existing minimum samples, positive
  mean at doubled assumed cost, no illustrative ruin and closed-trade drawdown
  ≤10% under the existing illustrative risk convention.
- No unknown required candidate/reference outcomes or missing required
  comparisons. Closed-trade drawdown is not a measured intratrade account risk.

Legacy `PF_1_5_lower_CI_supported` remains a diagnostic about the old target;
it does not become the new acceptance criterion. PF confidence >1 is the
unchanged positive-edge requirement, distinct from a point-estimate target.

The initial implementation is deliberately a **necessary-condition screen of
saved statistics**, not a full successor gate reimplementation or a new
backtest. It can rule out a row when required point/sample conditions fail;
passing this screen would not establish qualification. The additional frozen
gates and source limitations would still apply. All inspected rows are
reported, including unsuccessful cases, and cost/fill sensitivities remain
separate diagnostics. Lowering an acceptance threshold after observing
outcomes cannot make these dates fresh OOS or establish prospective profit.

A future candidate would require a separately frozen, target-aware protocol
and untouched evaluation after development support. Changes to indicators,
entries, thresholds, fills or costs remain new hypotheses; changing the
acceptance target does not turn such changes into previously validated success.
Source-only quote simulations do not verify
executable CFD pricing, account costs, fills, cash profit or prospective paper.

`fresh_out_of_sample=false`, `QUALIFIED=false` for this retrospective assessment.
`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`, `OPENED_TRADES=false`.
