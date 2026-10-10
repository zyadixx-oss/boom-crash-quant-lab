# Fixed-ledger additive-cost ceiling — independent diagnostic

This verifies ALREADY OBSERVED round6 results and zero-cost sensitivities.
The twelve gross PF values, ledger schema and example rows were inspected
before implementation. It is not a new premeasurement economic experiment,
fresh price/label holdout, model, execution policy or profitable discovery.

For every fixed completed gross return g_i and nonnegative additive cost c_i,
x_i=g_i-c_i<=g_i. Consequently sum(max(x_i,0)) cannot exceed its gross value
and sum(max(-x_i,0)) cannot be smaller. When gross losses are positive,
PF(x)<=PF(g). This includes heterogeneous nonnegative costs and costs that
turn a winning path into a loss. With no gross losses the ratio is undefined;
do not manufacture infinity or a finite ceiling. If the finite gross PF is
below1.5, cost reduction alone cannot attain1.5 on those exact saved paths.

The diagnostic independently reads all twelve round6 later180 primary ledgers,
preserving symbols, BOOST44/RIDGE44/RIDGE19 and SPIKE/DRIFT. It checks exact
frozen result/audit hashes, inherited source/helper hashes, ledger/signal hashes,
complete sample counts, no censored/missing/omitted outcomes, ordered unique
signals, one-open occupancy, positive prices/ATR, gross return arithmetic and
net=gross-0.05R (assumed0.10ATR divided by2ATR stop distance). Standard-library
sums/PF/means and entry-day counts must match the primary summaries, existing
independent audit and the exact same-fill/same-delay zero-cost sensitivities.

The source/test/document hashes are saved before this verification run; all
prior exposure is explicit. No raw price source is parsed, model trained,
signal selected, path re-simulated, uncertainty interval generated or source
artifact rewritten. Reproduction needs the exact local pinned ledgers/signals.
Outputs and failed attempts are exclusive-create and retained. An audit of this
arithmetic is not an audit of inferential validity or actual broker execution.

The ceiling assumes FIXED signals, fills, weights and nonnegative additive
costs. It does not bound alternative execution, price impact, different
selection/sizing, rebates or positive financing. It cannot establish a bound
for every possible future policy. Lower public-feed quoted spreads cannot be
retroactively substituted for historical executable CFD costs. RIDGE19 is a
reference and cannot satisfy the requested combined frames.

The user's PF>=1.5,>=1000 completed held-out paths/model/symbol,>=60 active days
and inherited development/inference/validation gates remain unchanged. Counts
across models and references are never pooled as independent strategy evidence.
Every historical/live candidate and goal-achieved field remains false.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false`. Cash profit, measured executable CFD costs and prospective
paper results remain NOT TESTED.
