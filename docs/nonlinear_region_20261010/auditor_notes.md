# Independent audit execution and oracle repair

The authoritative successful supplement is `independent_audit_final.json`.
The original `independent_audit.json` remains a preserved FAIL and must not be
presented as a successful audit or overwritten. Both attempts used the same
unchanged frozen study declaration and primary results.

## First attempt and demonstrated cause

The first reviewer source was SHA256
`420fbb7e0b925c47ec26842edf2732ebee137827f56462a66213ee461de562d5`.
Its 26 synthetic tests passed. Its actual audit completed 509,240 checks and
reported 93 discrepancies, all confined to chronological-third metrics.
The saved source, tests, failed report and stdout are retained separately.

The initial independent oracle selected a third only by its signal-time bounds.
The frozen production `scripts/run_spike_timed_study.py::thirds` first invokes
`scripts/run_spike_payoff_study.py::window`, whose existing convention is:

    signal_time >= third_start AND signal_time + 31 minutes <= third_end

This is an inner planned-horizon purge for each descriptive third. Equality
at exactly 31 minutes is retained. Actual early exit does not exempt an issue
from that convention. The initial oracle omitted this additional purge.
For example, BOOM600/CLOCK/wf3's first third contained 125 saved trades versus
126 selected by the initial oracle; the extra signal at 2026-01-31 11:00 UTC
was too close to the 11:08 UTC third boundary.

A regression was added before repair. Its first-third boundary is 01:00 UTC:
00:29 plus 31 minutes is retained; 00:40 is excluded even with a 00:50 actual
exit. The 27-test suite produced exactly one expected failing test, saved in
`reviewer_third_boundary_regression_fail.log`. Only the independent reviewer's
third-cohort filtering was then changed to the inherited inclusive convention.
The final 27 tests pass. No learner, gate, runner, protocol, primary result,
reference statistic, historical source or scientific parameter was changed.

## Successful second attempt

The separate final audit completed with exit code 0 and PASS:

- 509,240 checks, zero errors and 994 recorded input/code/artifact pins.
- 16 paired fitted models; 27,456 CLOCK training-label rows and 22,111 completed
  labels across overlapping prefixes. These counts are not independent trades.
- 9,290 selected scores and independently reconstructed original-quote region
  dispositions, preserving sequential pending/open occupancy and missingness.
- 62,380,130 original quote rows and 670 missing seconds.
- 60 result cells, 36 byte-equivalent frozen reference rows after the declared
  judgment-field removal, 120 new paired daily/weekly comparisons and 8 held-out
  qualification conjunctions.

The unchanged declaration SHA256 is
`35756c799689ef6a03f76e3aeda86a4568994a328f79af13685a929f91bee09c`;
the unchanged results SHA256 is
`4664f8d2ad7282acf1336e449efa6376f467452d86df8a3b2e70a83ab04cf4d7`.

The reviewer imports only previously independently authored audit helpers.
Their bytes are checked against the pinned parent audit before quote decoding.
It does not import the production predictor, replay, metrics or gate functions.
One synthetic differential test obtains a model from the production fitter;
its predictions and routed residual arithmetic are checked by the independent
reviewer. That test does not read historical observations.

## Scope limits

Verified here: exact cached feature/label/reference lineage; training target,
matrix and model identity; independently reconstructed training-only bin cuts,
node routing/counts/residual sums/regularized values and gain arithmetic;
training quantile and cutoff; selected issuance; original tick paths and net-R;
point statistics, day/week confidence and comparison p-values; chronological
thirds with their inherited purge; illustrative closed-trade drawdown; Holm and
all stable-positive acceptance conditions, including required-reference unknowns.

Not independently rerun: full optimal split search and split tie optimality;
original 44 feature formulas and intrinsic validity; parent training or frozen
reference tick paths in this run. The latter rely on exact immutable artifacts
and the pinned prior independent path audit. No account execution, measured
historical CFD costs, cash profit or prospective paper test is certified.
PASS establishes this bounded result consistency, not profitable qualification.
History remains adaptive and exposed. All four live/ready/allowed/opened flags
remain false.

## Preserved hashes

| Artifact | SHA256 |
|---|---|
| reviewer_attempt01_source.py | `420fbb7e0b925c47ec26842edf2732ebee137827f56462a66213ee461de562d5` |
| reviewer_attempt01_tests.py | `4e6b8ecd65e86da9c2dff41a8349e3690ab5659fd1c4407c89005ee5925fd4b1` |
| reviewer_self_test.log | `3fe2acbe7e237122460044eb99d9f5115768ddc464431eeb7ec675e9f455bf3d` |
| independent_audit.json | `43e7c94d6a0dab5e97b7ba95be6763424e6a7fcde2497bcaa8034d1b6fe4f297` |
| independent_audit_stdout.log | `3074d78d0bf584ebf963d32ac383a3f6263f182a11ed50672cb1d5e616114a1b` |
| reviewer_third_boundary_regression_fail.log | `89996d4a076d0f21e9b4163f64ea58ce1583479f614aa59fb95804f894cbb193` |
| review_study.py | `7a52a6124910f62421febc702651ef8ffe8b20ccdf7ea5293cc24b248f60a305` |
| test_review_study.py | `d556a267427f37537a1096105601be6f289ee0f5355b094d40393f7a2672bde4` |
| reviewer_self_test_final.log | `1d99794c3eb1dab9dddd1a6c3bfc0db6c6e61d39ae7946f996df638e1abc33e7` |
| independent_audit_final.json | `dcb6ef38f3ee3635f56cd467bf3179fe5f68604a77246d1d7615e0cbf223df4c` |
| independent_audit_final_stdout.log | `1dba624f1d1da5b84157776610424b8ef5dd2ba8491b124a74a6b5d7aaf2879f` |
