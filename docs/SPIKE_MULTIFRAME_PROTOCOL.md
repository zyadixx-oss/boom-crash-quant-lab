# Fixed combined-timeframe native300 study — adaptive research round 5

The user chose one combined process: closed H4/H1 context with M15/M5/M1
inputs. This is not five independently fitted strategies and does not claim
minute-frequency issuance. Native300 M5 reference selection is already frozen;
fresh native/combined-timeframe features and payoffs remain unexamined until
both selections are frozen. All four live/execution flags stay false.

## Sole declared change and fixed execution

- Replace the nineteen M5-only inputs with the same nineteen plus exactly25
  declared continuous H4/H1/M15/M1 inputs, ordered in `multiframe_signal.py`.
  Use that module's exact44 formulas and completeness/as-of validity rules.
  Every joined frame must already be closed at issuance; missing latest
  context invalidates the opportunity rather than carrying older context
  through an unknown candle. M1 timing refers to the last completed M1 candle
  at the same M5 issue time.
- Keep native BOOM300N/CRASH300N, SPIKE/DRIFT, train-only standardization,
  unclipped target/inputs, mean-SSE ridge penalty0.1, free intercept and frozen
  positive-floor training-prediction q75 cutoff. No feature or hyperparameter
  selection, interactions, calendar search, rescaling or probability calibration.
- Keep exact UTC00/30 closed-M5 issuance. Stop2ATR, hold15 minutes, no take
  profit, entry+1 M1 minute, adverse-extreme stress, round-trip cost0.10ATR,
  one position and unknown/censored paths. Keep issue+31-minute end purging
  and every previous fixed execution/cost sensitivity. No exit search.

## Chronology and joint freeze

- Use the exact older native300 interval [2025-10-09 11:08,2026-04-07 11:08)UTC
  from `data/spike_learned_transfer`. Its payoff was already seen: reuse is
  exploratory. Train first70%, independently refit0–40/0–50/0–60 for validation
  40–50/50–60/60–70, and fit both final directions on older first70%.
- Direction eligibility/ranking/null fallback remains exactly native300's
  frozen rule. No retrospective evaluation may choose or retrain a direction.
- Fresh source is [2026-04-07 11:08,2026-10-04 11:08)UTC from
  `data/spike_native300_recent`,259,199 clean rows plus one explicitly unknown
  minute per symbol; raw off-grid exclusions are retained. Never interpolate.
  Validate old/fresh adjacency, all price/manifest hashes and both model
  lineages before any fresh feature preparation.
- Development must verify the native reference selection and SHA, and refuse
  if native evaluation results already exist. Save a predevelopment declaration
  containing all inherited/native/new code hashes, four price/manifest hashes
  and native selection lineage. Freeze combined44 selection and timestamp
  before native or combined fresh evaluation. Refuse overwriting declarations,
  selections or terminal results.
- Store native-reference study paths relative to the repository root and
  resolve them against the current checkout, so reproduction does not depend
  on a particular workstation's absolute directory.
- Evaluation requires the native selection to retain the saved hash and its
  terminal result to refer to that selection, with its evaluation start after
  the combined selection freeze. All code, declarations and acquisition hashes
  are rechecked before deriving any evaluation features.
- Report `reused_final30` as exploration and `fresh_temporal180` as historical
  chronological evaluation. This adaptive methodology and retrospective
  collection are **not** a prospective paper cohort or independent proof.

## M5 comparison and eight-test multiplicity family

- Native M5 predictions use the frozen19-feature native model on the exact
  same combined44-eligible UTCclock timestamps. Intersect feature availability
  only, never the future-knownness of outcomes. Preserve missing entry/path
  audits. The combined mask includes base19 validity, so loss of a combined
  opportunity from the native feature mask is treated as an implementation
  error. Report full native and common-clock opportunity counts and warmup/
  gap exclusions explicitly.
- Default clock comparison remains every combined44-eligible opportunity with
  the same side/config. Additionally compare combined-model vs common-clock
  frozen M5-model expectancy using paired UTC-day and seven-day inference.
  Different model acceptance fractions may produce different trade counts;
  this measures per-completed-trade expectancy, not matched portfolio exposure.
- Use9,999 repetitions and seed20261005. Combined conjunction p is the maximum
  of inherited positive-mean/clock day/week p and positive-mean/M5-reference
  day/week p. Daily/weekly PF uncertainty remains unchanged.
- Joint Holm covers exactly eight fresh tests: four original native19 rows
  and four combined44 conjunction rows. Read native results as immutable
  references; write copied joint-adjusted native inference in the combined
  result. Prior native Holm4 status remains round4 reference-only and cannot
  be presented as a new discovery under the eight-test family. No source
  result or saved selection is modified.

## Evidence gates and claim limits

- Primary PF>=1.5 and>=1,000 completed fresh trades per model/symbol; all prior
  native gates remain mandatory: development eligibility,>=60 active days,
  no censored/invalid/missing-entry outcomes, no undefined PF bootstrap ratios,
  daily/weekly PF lower bounds>1, positive mean and clock-excess CI lower
  bounds, jointHolm8 p<0.05, positive sufficiently sampled chronological
  thirds,<=10% illustrative closed-trade drawdown, no ruin and positive mean
  at doubled cost0.20ATR. No pooling across symbols/directions.
- Combined historical candidate additionally requires both daily and weekly
  excess-CI lower bounds over the common-clock M5 reference to be>0, with no
  censored/invalid/missing-entry reference outcomes. A point PF, sensitivity,
  or apparent timeframe agreement cannot rescue rejected development.
- Expected-PF1.5 support additionally requires both PF95 lower bounds>=1.5.
  Scores never become calibrated probabilities or promised profit.

Actual bid/ask, commissions, financing, executable broker fills, cash profit
and prospective paper validation remain NOT TESTED. H4/H1/M15/M5/M1 inputs
do not by themselves establish profit on every timeframe or enable execution.
