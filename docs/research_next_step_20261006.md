# Source-only research decision after round10

Reviewed 2026-10-06. This review reads existing aggregate reports and frozen
protocols only. It introduces no model, feature, calculation, data request,
evaluation cohort or change to rounds1–10.

**Decision: reject the tested policies as promotion candidates. No distinct
economic hypothesis is currently supported enough to select for another PF
study.** The user's combined closed H4/H1 + M15/M5/M1 objective remains unmet:
net PF>=1.5, at least1,000 completed held-out paths per model/symbol and at
least60 active held-out days, with development, uncertainty, cost, stability and
reference gates. Failure of these policies does not establish that every
possible Boom/Crash policy is unprofitable.

## Evidence that constrains the next step

| Existing evidence | Consequence |
|---|---|
| [Crash500 resistance excursion lift1.412](RESULTS.md), but the locked external support/resistance payoff PF0.603/2,259 paths and negative mean | Predicting a later price excursion is insufficient evidence of an executable positive-return entry. The excursion association does not justify retuning exits on the same history. |
| [Combined44 native300](spike_multiframe_20261005/REPORT.ar.md): all four later PFs0.737–0.824, two cohorts above1,000 paths | Lack of sample size is not the sole cause of the combined-feature failure. No advantage over the common-clock M5 reference was established. |
| [Nonlinear combined44 native600](spike_nonlinear_20261005/REPORT.ar.md): all four later models above1,000 paths, PF0.827–0.938; no positive nonlinear-minus-reference bound | Merely replacing the learner or adding interactions lacks an observed economic justification. Three of four nonlinear gross means also remained negative. |
| [Crash600 older-age hazard ratio1.226](spike_tick_tail_20261005/REPORT.ar.md), followed by [round9](spike_tick_age_payoff_20261005/REPORT.ar.md) model PFs<1 and no positive age/mark increment | Hazard association is not PF. It does not support another age threshold or more dates for the fixed age/mark policy. |
| [Round10](spike_tick_path_risk_20261006/REPORT.ar.md): RIDGE45 PF0.758/0.761/0.763/1.033 from105/132/121/83 completions respectively; all45-minus44 intervals cross0 | Small sample leaves some effects uncertain, but offers no positive economic/stability evidence for expanding this fixed path-risk policy. Crash DRIFT1.033/n83 cannot be selected after seeing it. |

Passed audits establish the specified arithmetic, inputs and path lineage
within their declared scope. They do not establish bootstrap validity, fresh
OOS, actual execution or profit. Non-overlapping one-open paths also remain
serially dependent; a ledger count is not proof of IID observations.

## Concrete next action

Keep `selected_candidate=null` and publish this rejection and the verified
round10 report. Do not declare round11 or acquire a larger tick sample for a
tested policy merely to reach the requested count.

If further implementation is wanted, the justified preparatory work is a
**prospective engineering-shadow protocol**, using immutable combined44 models
and CLOCK as diagnostic references across both native symbols and directions.
It should freeze model/source hashes, the existing issuance clock and path
policy before starting; log arrival and decision timestamps, closed-frame
availability, exact prior-tick coverage, raw provenance, planned occupancy and
unknown paths. Its acceptance criteria concern causal issuance, reproducibility
and data quality. It does not select a profitable candidate, remedy rejected
development or demonstrate PF>=1.5. No such collection or shadow run is started
by this review. If its payoff outcomes are inspected, disclose that exposure
before reusing the period for a later hypothesis.

A later economic preregistration needs a distinct mechanism or independent
evidence that predicts positive net payoff under the unchanged execution
contract. State its causal inputs, sole change, same-availability combined44
and clock controls, development eligibility, discovery family and rejection
rule before new outcome inspection. A feature, cadence, side, exit or calendar
chosen because a known-history subgroup looked better does not meet that
condition. No such mechanism is selected here.

## Chronology, sizing and cost constraints

- Round10 reused already-known round9 prices and payoffs. Its final-selection
  freeze prevents further fitting on that suffix; it does not make the suffix
  a fresh holdout. Prospective evidence starts strictly after the new freeze.
- The inherited scale was calibrated through June13 in the round8 first40%
  observed-row prefix. Any expanded walk-forward study must have that entire
  calibration available by its earliest fit cutoff. Earlier training rows can
  use a training-fitted transform; earlier validation issuance cannot use a
  scale whose calibration completes later. A nominal longer calendar block
  must not silently place its first fit before calibration completion.
- Size a genuinely justified future study once from development **one-open
  completion and active-day rates**, common-feature availability and gap/censor
  allowance. Two hundred usable source days yield only60 final calendar days
  in a70/30 split; they guarantee neither60 active days nor1,000 completions.
  No symbol/direction pooling, overlapping-target substitution or extension
  until a favorable result. Historical tick retention and future completeness
  require evidence; the sampled12 days do not prove full-period availability.
- Public next-observed quotes and assumed0.10ATR cost remain a proxy. Bid/ask,
  processing delay, contract sizing, financing and executable fills are not
  measured by these studies. Lower hypothetical cost or barrier fills cannot
  replace the locked primary result. Distinguish observed PF>=1.5 from evidence
  for expected PF>=1.5; the latter requires suitable lower bounds reaching1.5.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false`. Broker execution, monetary profit and prospective paper
profit remain NOT TESTED. No live candidate or authorization results from this
review.
