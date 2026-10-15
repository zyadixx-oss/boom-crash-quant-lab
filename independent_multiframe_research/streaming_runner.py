"""Bounded execution access for supplied signals, not a complete study runner.

This adapter preserves the frozen multiframe runner's consumed-prefix schedule,
reservation, censoring, daily bookkeeping and conditional output fields. It does
not create signals, fit U, select periods, evaluate alpha gates or verify cash
fees/fills. No study protocol is frozen or actual source measured by this file.

The caller declares an aligned stage exactly matching the store's permitted
calendar. Each unblocked exposure obtains a real complete local QuoteTape from
ShardedQuoteStore; sparse issuance indexes never enter execution or trade marking.
Actually executed windows inherit the store's one-hour hold/leg-latency limits.
The fixed v2 fifteen-minute signals fit within that access contract. Unsupported
windows fail instead of silently truncating source or changing horizons.

Only the supplied signal prefix is retained in ledgers; daily output and ledger
memory still grow with days/completed signals. Bounded quote access is not a
whole-process memory bound, source-completeness claim, or cash-profit finding.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math
import statistics
from typing import Iterable

from independent_profit_research.execution import ExecutionConfig
from independent_profit_research.policies import Signal
from independent_profit_research.runner import (
    DAY_MS, EMBARGO_MS, PRICE_SCALE, SAFETY_FIELDS,
    _Drawdown, _Sum, _mark_trade, _pf, _public_outcome,
)
from .runner import _snapshot, _stamp, _positive, _fsum
from .sharded import ShardedQuoteStore


def evaluate_streamed_signals(store: ShardedQuoteStore, signals: Iterable[Signal], stage_start_ms: int,
                     stage_end_ms: int, unit: float, execution_config: ExecutionConfig,
                     *, policy_id: str, baseline: bool = False) -> dict:
    """Evaluate the consumed signal prefix, reserving decision through inclusive exit.

    Validate every consumed signal before embargo or overlap filtering. Signals
    outside [start+30min,end-30min) produce no exposure. UNKNOWN immediately
    closes the iterator, censors all primary stage/day aggregates and leaves
    completed-prefix values explicitly descriptive. SKIP reserves no exposure.
    No assertions are made about an unconsumed signal suffix or actual fills.
    """
    if type(store) is not ShardedQuoteStore:
        raise ValueError('store must be an exact ShardedQuoteStore')
    if not isinstance(policy_id, str) or not policy_id or policy_id != policy_id.strip() or any(
            ord(c) < 32 or ord(c) == 127 for c in policy_id):
        raise ValueError('policy_id must be a nonempty canonical string')
    if type(baseline) is not bool:
        raise ValueError('baseline must be boolean')
    unit = _positive(unit, 'unit')
    if not isinstance(execution_config, ExecutionConfig):
        raise ValueError('execution_config must be ExecutionConfig')
    config = ExecutionConfig(**asdict(execution_config))
    start, end = _stamp(stage_start_ms, 'stage_start_ms'), _stamp(stage_end_ms, 'stage_end_ms')
    if end <= start:
        raise ValueError('Stage end must follow stage start')
    if (start % DAY_MS or end % DAY_MS or store.day_start_ms != start
            or store.day_end_ms != end):
        raise ValueError('Store calendar must exactly match aligned half-open UTC stage boundaries')
    try:
        datetime.fromtimestamp(start/1000, timezone.utc)
        datetime.fromtimestamp((end-1)/1000, timezone.utc)
        iterator = iter(signals)
    except (OverflowError, OSError, TypeError, ValueError) as error:
        raise ValueError('Expected a supported UTC calendar and iterable signals') from error
    eligible_start, eligible_end = start+EMBARGO_MS, end-EMBARGO_MS
    daily = {}
    for day in range((start//DAY_MS)*DAY_MS, ((end-1)//DAY_MS+1)*DAY_MS, DAY_MS):
        daily[day] = {'day_start_ms': day,
                      'date_utc': datetime.fromtimestamp(day/1000, timezone.utc).date().isoformat(),
                      'completed': 0, 'unknown': 0, 'cash_net': None,
                      'safety': {key: False for key in SAFETY_FIELDS}}
    ledger, completed_rows = [], []
    running_r, tracker = _Sum(), _Drawdown(start)
    reserved = previous = unknown_from = None
    unknown = skipped = blocked = excluded = excluded_after = consumed = 0
    stopped_early = False
    try:
        for raw in iterator:
            signal = _snapshot(raw, policy_id, previous, end, config)
            previous, consumed = signal.decision_ms, consumed+1
            if signal.decision_ms >= eligible_end:
                excluded_after += 1
                stopped_early = True
                break
            if signal.decision_ms < eligible_start:
                excluded += 1
                continue
            if reserved is not None and signal.decision_ms <= reserved:
                blocked += 1
                continue
            trade_tape = store.trade_tape(signal.decision_ms, signal.horizon_end_ms, config)
            outcome = trade_tape.simulate(signal.decision_ms, signal.direction, signal.horizon_end_ms,
                                          signal.stop_distance, signal.target_distance, config)
            row = _public_outcome(outcome, unit)
            row.update(policy_id=policy_id, baseline=baseline)
            ledger.append(row)
            if outcome.status == 'UNKNOWN':
                unknown, unknown_from, stopped_early = 1, signal.decision_ms, True
                daily[(unknown_from//DAY_MS)*DAY_MS]['unknown'] += 1
                break
            if outcome.status == 'SKIP':
                skipped += 1
                continue
            row['marked_quote_rows'] = _mark_trade(trade_tape, outcome, running_r.value, tracker)
            running_r.add(row['R'])
            row['conditional_equity_index_after_exit'] = 10000+25*running_r.value
            reserved = outcome.exit_time_msc
            tracker.mark(row['conditional_equity_index_after_exit'], reserved)
            daily[(reserved//DAY_MS)*DAY_MS]['completed'] += 1
            completed_rows.append(row)
    finally:
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()
    censored = bool(unknown)
    field_functions = {
        'observed_sum_R': lambda row: row['R'],
        'observed_sum_quote': lambda row: row['signed_move_after_spread_and_slippage'],
        'observed_gain_R': lambda row: max(0.0, row['R']),
        'observed_loss_R': lambda row: max(0.0, -row['R']),
    }
    by_day = {day: [] for day in daily}
    for row in completed_rows:
        by_day[(row['exit_time_msc']//DAY_MS)*DAY_MS].append(row)
    totals = {name: _fsum(function(row) for row in completed_rows)
              for name, function in field_functions.items()}
    daily_exact = True
    for day, result in daily.items():
        for name, function in field_functions.items():
            result[name] = _fsum(function(row) for row in by_day[day])
            daily_exact &= result[name] == math.fsum(function(row) for row in by_day[day])
        result['sum_R'] = None if censored else result['observed_sum_R']
        result['sum_quote'] = None if censored else result['observed_sum_quote']
        result['primary_measurement_censored'] = censored
    pf, pf_reason = _pf(totals['observed_gain_R'], totals['observed_loss_R'])
    ending = 10000+25*totals['observed_sum_R']
    if not math.isfinite(ending):
        raise ValueError('Conditional equity index must remain finite')
    summary = {
        **totals, 'sum_R': None if censored else totals['observed_sum_R'],
        'sum_quote': None if censored else totals['observed_sum_quote'],
        'pf': None if censored else pf, 'observed_pf': pf,
        'pf_reason': 'UNKNOWN_AND_REMAINING_CENSORED' if censored else pf_reason,
        'completed': len(completed_rows), 'unknown': unknown, 'skipped': skipped,
        'blocked_overlap': blocked, 'excluded_before_embargo': excluded,
        'excluded_after_embargo': excluded_after, 'input_signals_consumed': consumed,
        'remaining_censored': censored, 'unknown_and_remaining_censored': censored,
        'remaining_opportunities_count': None if censored else 0,
        'censored_from_ms': unknown_from,
        'censored_until_exclusive_ms': eligible_end if censored else None,
        'marked_drawdown_pct': None if censored else 100*tracker.maximum,
        'observed_completed_prefix_marked_drawdown_pct': 100*tracker.maximum,
        'marked_quote_rows': tracker.marked_rows,
        'drawdown_peak_time_ms': None if censored else tracker.maximum_peak_time_ms,
        'drawdown_trough_time_ms': None if censored else tracker.trough_time_ms,
        'ending_equity_index': None if censored else ending,
        'observed_completed_prefix_ending_equity_index': ending,
        'primary_measurement_usable': not censored, 'qualified_for_profit_claim': False,
        'cash_net': None,
        'median_trade_R_observed': statistics.median([row['R'] for row in completed_rows]) if completed_rows else None,
        'median_holding_ms_observed': statistics.median([
            row['exit_time_msc']-row['entry_time_msc'] for row in completed_rows]) if completed_rows else None,
        'median_holding_scope': 'COMPLETED_OBSERVED_PREFIX_ONLY_NOT_TIME_TO_SPIKE',
    }
    reconciliation = {
        'summary_to_ledger_fsum_exact': all(totals[name] == math.fsum(function(row) for row in completed_rows)
                                          for name, function in field_functions.items()),
        'each_day_to_exit_day_ledger_fsum_exact': bool(daily_exact),
        'daily_to_stage_fsum_residual': {name: _fsum(day[name] for day in daily.values())-totals[name]
                                       for name in field_functions},
        'scope': 'FLOAT64_FSUM_OF_LEDGER_ROWS_ROUNDED_DAILY_TOTALS_CAN_DIFFER_FROM_WHOLE_LEDGER',
    }
    return {
        'status': ('UNKNOWN_AND_REMAINING_CENSORED' if censored else
                   'NO_ELIGIBLE_DECISION_WINDOW' if eligible_end <= eligible_start else
                   'CONDITIONAL_QUOTE_MEASUREMENT_ONLY'),
        'policy_id': policy_id, 'baseline': baseline,
        'stage_start_ms': start, 'stage_end_exclusive_ms': end,
        'eligible_decision_start_ms': eligible_start, 'eligible_decision_end_exclusive_ms': eligible_end,
        'summary': summary, 'daily': list(daily.values()), 'ledger': ledger,
        'accounting_reconciliation': reconciliation,
        'unconsumed_signal_suffix_not_validated': stopped_early,
        'signal_issuance_causality_owned_by_caller': True,
        'equity_model': 'CONDITIONAL_INDEX_10000_PLUS_25_TIMES_CUMULATIVE_R_NOT_CASH',
        'drawdown_model': 'OPPOSITE_SIDE_PLUS_EXIT_SLIPPAGE_SAME_MS_FAVOURABLE_THEN_ADVERSE_ENVELOPE',
        'same_millisecond_order_known': False,
        'model_label': 'CONDITIONAL_EVENT_TIME_NOT_REAL_ARRIVAL_OR_FILL',
        'price_scale_divisor': PRICE_SCALE, 'cost_verified': False, 'execution_verified': False,
        'cash_net': None, 'full_tick_coverage_verified': False, 'cash_profit_tested': False,
        'conditional_quote_measurement_performed': True,
        'safety': {key: False for key in SAFETY_FIELDS},
    }
