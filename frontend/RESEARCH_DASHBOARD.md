# Research dashboard data contract

The default route is `/#research`; the existing public-data CRT chart remains at
`/#chart`. The dashboard reads `public/campaign/latest.json` every 60 seconds and
on manual refresh. No trading or account API is used by the research view.

Before implementation, the acceptance checklist was:

- Preserve the existing chart and all four false live-trading flags.
- Show ten requested symbols by ten independent frames, including missing cells.
- Never substitute validation results for sealed final out-of-sample results.
- Display only explicit qualified selections; preserve all rejection reasons.
- Count completed cycles from unique, dated completion records.
- Support result sorting, filtering, detail charts, full trade ledgers, monthly
  results, references, parameter stability, research logs, and JSON/CSV export.
- Preserve missing metrics as null, and verify RTL on desktop and mobile.

The following were unverified before implementation: whether a current campaign
snapshot exists; whether contract costs are measured; whether any strategy is
qualified; and whether all 32 cycles have actually run. None are assumed true.
The smallest invalidity tests are a sealed final row with profitable validation
metrics, a missing metric, a true live flag, and an unsupported cycle counter.
These must never produce an invented final result or a qualified strategy.

## Snapshot

Required version: `schema_version: 1`. All other fields are optional. The parser
preserves absent metrics as null. Rates and drawdown are fractions, not percent
numbers. Numeric strings and nonfinite values are not treated as measured values.

```text
generated_at: ISO8601
safety: LIVE_TRADING, READY_FOR_LIVE, LIVE_ALLOWED, OPENED_TRADES (all false)
campaign: id, status, started_at, next_cycle_at, current_cycle,
          completed_cycles, target_cycles, interval_minutes
source: provider, period_start, period_end, updated_at,
        cost_provenance, contract_status, data_notes[]
symbols[]: symbol, label, status, reason, best_strategy_id
strategies[]:
  id, symbol, timeframe, name, settings{}, status, reasons[], tested_at,
  source, period_start, period_end, rank_score, details_url
  development:
    training: {metrics} or metrics directly
    validation: {metrics} or metrics directly
    validation_score
    walk_forward[]: label, value, status, reason
  oos: null while sealed, otherwise metrics
cycles[]: number, status, started_at, completed_at, tested_count, notes[]
reproduction: commands[], report_url
```

Accepted symbols: BOOM300/500/600/900/1000 and CRASH300/500/600/900/1000.
Accepted frames: M1, M2, M3, M5, M10, M15, M30, H1, H2, H4.
Final statuses: QUALIFIED, REJECTED, NOT_TESTED, INSUFFICIENT_DATA.
A QUALIFIED label with `oos: null` is downgraded to NOT_TESTED for display.
Completed cycles require `status: COMPLETED`, a valid `completed_at`, and a
unique cycle number. The declared total alone does not count as execution proof.

Metrics fields:

```text
win_rate, profit_factor, completed_trades, net_profit, max_drawdown,
mean_net_R, net_profit_unit, status
equity[]: time (ISO8601), value
drawdown[]: time (ISO8601), value
trades[]: id, entry_time, exit_time, side, entry_price, exit_price, net_profit, status
monthly[]: month, trades, net_profit, profit_factor
references[]: name, net_profit, profit_factor, comparison, status, reason
stability[]: label, value, status, reason
```

The validation/final selector controls the displayed metrics, heatmap, sorting,
detail charts and CSV scope. Validation is the default because the final holdout
must remain sealed during development. Best-strategy cards always require final
QUALIFIED status regardless of the displayed scope. The heatmap selects the recorded risk-adjusted score per cell, displays its PF,
and does not qualify a model based on that point estimate.

## Lazy details

`details_url` may refer to same-origin `.json` only. A detail file can contain
`development.training`, `development.validation`, and/or `oos` with the metric
fields above. Saved scalar metrics are preserved when detail files provide only
arrays. A null final result stays null when a development detail file is loaded.
The main snapshot need not carry thousands of trades. All saved detail trades
are accessible in pages of 50 and in a full CSV download. Curves render recorded
points only. No initial equity point or missing period is fabricated.

The top JSON export downloads the original received snapshot, including fields
the display parser does not consume. Individual analysis export includes all
loaded detail arrays. CSV exports identify the comparison scope and escape
formula-like text without modifying negative numeric returns.

## Validation commands

```sh
npm ci
npm test
npm run typecheck
npm run lint
npm run build
npm run dev -- --host 127.0.0.1
```

Initial implementation checks: 45 frontend tests passed, including nine campaign
integrity tests. Type checking and production build passed. ESLint was previously
unconfigured; the added flat configuration reports zero errors and four existing
`any` warnings in unchanged market transport/test files. Browser QA verified
Arabic/RTL, 100 cells, honest missing-result states, both scope labels, preserved
CRT navigation, and a 390px viewport without document overflow. Integration with
actual campaign results must be checked again once that snapshot is generated.

Actual preparation snapshot integration was checked on 2026-10-11: the received
Deriv catalogue contains ten requested symbol profiles, status DATA_COLLECTION,
zero completed cycles, zero strategy results, null historical period bounds, and
all four false live flags. The parser preserves missing periods and produces zero
qualified selections. The local server serves the file as application/json.
DATA_COLLECTION and NOT_VERIFIED display in Arabic. Follow-up browser verification
for this actual snapshot was unavailable because the IAB browser disconnected;
this does not replace the earlier empty-state desktop/mobile browser checks.

Current product truth fixes (2026-10-12): known assumed-account units and the two
fixed development-cycle notices display in Arabic; unknown strings are retained.
All period/chart dates include the Gregorian year. The summary distinguishes
displayed candidates from the sum of tests in uniquely dated completed cycles.
The default table order uses the saved validation/final risk score; the engine's
-1e12 unrankable marker displays as missing and is placed last. Manual metric
sorting remains available, and CSV includes the selected scope's risk score.
These frontend changes passed 50 tests, type checking, production build, and
lint with zero errors/four existing warnings. No campaign research code changed.
