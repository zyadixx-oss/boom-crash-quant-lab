# Execution and review record

The user changed the objective to stable positive net profit, with no PF1.1
floor. `ACCEPTANCE_CHANGE.md` preserves sample, uncertainty, development, cost,
unknown-outcome and risk requirements while retaining all legacy artifacts.

The delegated saved-statistic scanner source was written before the assistant
agent hit a usage limit. Root inspected it and completed execution and repairs.
The second read-only reviewer approved the policy text, then also hit a usage
limit before reviewing the finished code/result. No final independent-agent
arithmetic/completeness review is claimed for this assessment.

Execution chronology, original bytes retained:

1. Initial self-test failed because the new strict PF>1 field used `positive(PF)`
   (PF>0). `failed_source_01.py` SHA256
   `f4ab900b1b3df9fb66594a0d77d050afbcdfe7f8e781e0950bb1a067721a8dca`
   and `self_test.log` preserve this failure. Fixed to `positive(PF) and PF>1`.
   No historical assessment output existed at this point.
2. Eighteen self-tests then passed (`self_test_pass.log`). The first saved-result
   read failed because the prerequisite tail report stores untested PF as JSON
   null, rather than the string NOT TESTED. `failed_source_02.py` SHA256
   `01c215a25ff2b0c31c69dbe5f995315ece6d26aa420145a3ace7a7d1d1f89dcf`
   and `execution.log` preserve this failure. Fixed the adapter to require the
   original null PF and false historical-candidate status; no source changed.
3. The corrected scanner passed (`execution_attempt_02.log`), generating the
   exclusive `assessment.json` with source SHA256
   `c890ff5437882960cc5a4e6f7eaa30c7d8ad0fed75824de0ff5e26a0a8be0923`
   and output SHA256
   `b684b1b7d5af33bd16bc9584225f8991b25ce0432414da03d1962c5b673ab17c`.
4. The final source again passed all18 self-tests (`final_self_test.log`).
   A separate root-written pointer checker, without importing the scanner,
   passes885 checks using Decimal strict comparisons and original saved JSON
   pointer values (`pointer_review.json`). It does not re-audit quote paths or
   implement all qualification requirements.

The screen checks only necessary point/sample conditions. Zero intersections
rule out acceptance under the stronger full conjunction within this scope;
passing this screen would not qualify any policy. Controls and repeated frozen
references are explicit; row counts are not independent strategy counts or
pooled trade samples. There are27 undefined PF cells, never replaced by zero.
All four nonzero-cost positive sensitivities remain separately disclosed.

Source JSON safety flags and runtime gates are false. The scanner reads no price
files, runs no fitting/replay/bootstrap/network/account/order operation and does
not mutate existing artifacts. Final GitHub CI is verified separately from local
execution; its publication receipt stays outside these scientific artifacts.
