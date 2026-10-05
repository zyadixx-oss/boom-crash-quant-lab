# Round6: fixed nonlinear multiframe payoff hypothesis

This declaration precedes development and evaluation of this round. It does
not change any earlier frozen protocol, model, source or result. All four flags
must remain false: LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES.
Only public OHLC data and offline quote-path simulations are permitted.

## Question and prior evidence

Do nonlinear interactions between the existing closed H4/H1/M15/M5/M1 inputs
improve held-out quote-path expectancy over a same-clock linear44, linear19
and clock benchmark? Earlier CRT, exit and ridge rounds failed profitability
gates. This motivates one fixed algorithmic alternative, not an unrestricted
parameter search until an attractive result appears.

Histogram boosting can model feature interactions via successive regression
trees; training quantile bins reduce the split search. Regularized leaves use
residual sums divided by count plus lambda. See the
[primary technical documentation](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting).
The implementation here is deterministic NumPy code with specified math; it
does not import scikit-learn or claim estimator equivalence.

The price-structure inputs are hypotheses, not evidence of traded order-book
liquidity. Deriv describes these markets as synthetic and its
[FAQ](https://deriv.com/markets/derived-indices/synthetic-indices) says they have
no order book and technical patterns may be coincidental. A positive historical
model would still require independent cost/fill and prospective paper evidence.

## Data and chronology

Native-symbol BOOM600 and CRASH600, exact aliases confirmed by public
active_symbols. Older requested M1 interval [2025-10-09 11:08 UTC,
2026-04-07 11:08 UTC); later [2026-04-07 11:08 UTC,2026-10-04 11:08 UTC).
Acquisition may read raw candles and metadata before freezing, but must not
calculate later-period indicators, labels or payoffs. The unchanged public
collector preserves raw bytes, page boundaries, missing/off-grid rows and
normalization hashes. No missing minute is filled. Both full180-day requested
grids must be available with only declared gaps before fitting. Insufficient
retention is a data failure, not a reason to substitute another symbol/window.
Recovered RateLimit errors may be accepted only when every failed request end
has a matching successful public page in the preserved page audit and the full
requested grid reconciles to clean rows plus declared gaps. Preserve errors,
retry lineage and successful-page hashes. Other/unmatched errors are refused.

The cached conversation contains an unverified older Boom600 excursion claim;
its underlying dates/bytes are unknown. Thus the later data are held out from
this round, not claimed wholly unseen across all prior research. This is an
adaptive historical native-symbol test, not a prospective trading experiment
or proof of independent random generators.

Use the older first70% for development; the last30% and all later180 days are
unread by features/payoff until the entire12-model selection freezes. Three
expanding validation fits: train0–40/50/60%, test40–50/50–60/60–70%.
Every training endpoint and test endpoint uses a fixed31-minute planned purge.
Every fold refits its training estimator and cutoff from its own training
labels. Final models refit only older first70%. Older last30 is a secondary
within-round historical holdout, not a second discovery family or a tuning set.

## Fixed inputs and estimators

Use exactly the earlier frozen44 causal inputs and completed-frame alignment
from multiframe_signal.py. Latest invalid higher-frame rows remain invalid;
never replace them with older valid context. Use UTC minute00/30 opportunities.
M1 contributes timing features; this is not a test of entries every minute.
All models, including19-input reference, use the same44-input-available clock.

For each symbol, fit both SPIKE and DRIFT (opposite) separately:

1. BOOST44: squared-error residual boosting;100 trees; learning rate0.05;
   max_depth3; min_leaf200 training labels; at most16 training bins. For<=16
   distinct training values, use adjacent-value midpoint boundaries so binary
   inputs remain usable; use the upper adjacent value if floating precision
   leaves no representable internal midpoint. For higher cardinality, use
   unique internal training quantiles, excluding observed min/max. Constant
   inputs have no boundaries. Regularized leaf value sum(residual)/(n+20).
   Initial prediction is mean(training netR). Gain is the regularized child
   scores less parent score. Root gain may equal zero to expose a balanced
   pair interaction; deeper gains must be strictly positive, with no numerical
   tolerance accepting negative gains. Balanced higher-order interactions
   requiring another zero-gain split are deliberately outside this learner.
   Deterministic
   tie order: ordered feature then ascending boundary. No random subsampling,
   early stopping, label clipping, feature winsorizing or hyperparameter search.
   Values outside training boundaries route to outer bins; no extrapolated
   linear contribution is introduced. Model integrity protects serialized trees.
2. RIDGE44: same earlier mean-SSE ridge penalty0.1, free intercept and training
   mean/std,44 inputs.
3. RIDGE19: same ridge parameters on the first19 frozen base features, but on
   precisely the same eligible timestamps and labels as the44-input models.

Require at least1,000 completed labels for fitting any estimator. For each
model, cutoff=max(0,training prediction q75); issue only score>=cutoff AND>0.
The score estimates a quote-path netR target and is uncalibrated, never a
probability. Missing/nonfinite inputs/labels remain unknown and cannot fit.
Label target: uncapped15-minute stop-or-time path under the fixed policy below.

No estimator, direction or threshold is selected on held-out outcomes. A
per-symbol selection can only come from development-eligible44-input candidates ranked
by pooled validation selection_score then model ID. None means no selected
model; rejected candidates may still be evaluated diagnostically but cannot
be promoted regardless of an isolated later result.
RIDGE19 remains a measured reference hypothesis in the joint family; it cannot
be selected as satisfying the requested combined H4/H1/M15/M5/M1 strategy.

## Unchanged payoff and evidence gates

Stop2 M5 ATR14, no take-profit cap,15-minute hold, entry one minute after issue,
assumed total cost0.10ATR, adverse minute-extreme stop stress. One non-overlapping
position; fixed UTC30-minute opportunities. Preserve first missing quote,
censoring and planned occupancy. Costs are not measured broker costs.

Development eligibility: exact three ordered folds; each>=100 completed,
PF>1 and positive mean; pooled>=500 completed and>=30 active days; no censored
or invalid labels; pooled mean minus1.96 daily clusterSE>0. Otherwise reject.

Later primary family:12 fixed hypotheses =2 symbols ×2 directions ×3 models.
All must enter joint Holm correction; no removal of weak/empty candidates.
Use9,999 paired calendar-day bootstrap replicates and circular7-day blocks,
seed20261005. Report n, win rate, expectancy, PF, daily/weekly PF95 and mean95,
clock excess, active/calendar days, chronological thirds, holding time, tail
concentration and illustrative closed-trade drawdown. Ratio resamples with no
losses are undefined and prevent promotion; no infinity or fabricated interval.

For BOOST44, additionally compare against both corresponding frozen RIDGE44
and RIDGE19 ledgers on the same input-available clock. Conjunction p is the max
of clock day/week, RIDGE44 day/week and RIDGE19 day/week tests. Both reference
excess confidence lower bounds must be positive at day/week scales and their
paths uncensored/missing-entry-free before claiming nonlinear added value.
These are comparisons of per-completed-trade mean R, not equal-exposure
portfolio returns. Reference predictions/thresholds are never refit on later data.

Observed user target: PF>=1.5 and>=1,000 completed held-out simulations for
each model/symbol, no pooling. Historical candidacy additionally requires
development eligibility,>=60 active days,no unknown/invalid paths,PF95 lower>1,
mean and clock-excess95 lower>0 at both scales,jointHolm p<0.05,
each chronological third>=200 andpositive,closed-risk illustrationDD<=10%
without ruin,andpositive average at doubled assumed cost0.20ATR. Strong
evidence for expectedPF>=1.5 requires both PF95 lower bounds>=1.5.

18 predeclared sensitivities: costs0/.025/.05/.10/.20/.40ATR under primary
stress entry+1,barrier proxy entry+1 andstress entry+0. They do not replace
primary assumptions or rescue a failed primary gate. Risk0.25% per closed
simulated trade is an illustration, excluding floating drawdown/contract sizing.

Actual money profit, measured bid/ask/fills/costs and prospective paper results
remain NOT TESTED. No historical gate permits live execution in this project.
