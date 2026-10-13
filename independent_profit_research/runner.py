"""Causal stage measurement on an already supplied, validated artificial/quote tape.

This module never loads evidence, fits parameters, sends orders or estimates cash
profit. Prices supplied to the kernel are integer 1e-8 price units. Public price
fields are divided by 1e8; R and the conditional risk index remain unitless.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict
from datetime import datetime, timezone
import math
import statistics
from numbers import Real

from .execution import ExecutionConfig, TradeOutcome
from .policies import POLICY_IDS, Signal, generate_signals
from .tape import QuoteTape


PRICE_SCALE = 10**8
DAY_MS = 86400000
EMBARGO_MS = 1800000
CLOCK_MS = 1800000  # UTC minute 00/30 of each hour: 48 schedules per day.
SAFETY_FIELDS = ('LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES',
                 'live_trading', 'ready_for_live', 'live_allowed', 'opened_trades')


class _Sum:
    """Deterministic compensated accumulation without repeatedly rescanning history."""
    def __init__(self):
        self.total = self.correction = 0.0

    def add(self, value):
        total = self.total + value
        self.correction += ((self.total-total)+value if abs(self.total) >= abs(value)
                            else (value-total)+self.total)
        self.total = total
        if not math.isfinite(self.value):
            raise ValueError('Conditional arithmetic overflow')

    @property
    def value(self):
        return self.total + self.correction


class _Drawdown:
    def __init__(self, stamp):
        self.peak = 10000.0
        self.peak_time_ms = stamp
        self.maximum = 0.0
        self.maximum_peak_time_ms = self.trough_time_ms = None
        self.marked_rows = 0

    def mark(self, value, stamp):
        if not math.isfinite(value):
            raise ValueError('Conditional equity index must remain finite')
        if value > self.peak:
            self.peak, self.peak_time_ms = value, stamp
        drawdown = (self.peak-value)/self.peak
        if drawdown > self.maximum:
            self.maximum = drawdown
            self.maximum_peak_time_ms = self.peak_time_ms
            self.trough_time_ms = stamp


def _baseline_signals(tape, policy_id, native_direction, unit, start, end, clocks):
    """Use only the first observed row at the fixed schedule; never search for a better one."""
    clock = ((start+CLOCK_MS-1)//CLOCK_MS)*CLOCK_MS
    while clock < end:
        due = clock + (2000 if policy_id == 'POST_JUMP_CARRY_30S' else 0)
        index = bisect_left(tape.timestamps, due)
        clocks['scheduled'] += 1
        if index == len(tape):
            clocks['missing_observation'] += 1
        elif tape.timestamps[index]-due > 2000:
            clocks['stale_first_observation'] += 1
        else:
            decision = tape.timestamps[index]
            carry = policy_id == 'POST_JUMP_CARRY_30S'
            horizon = clock+30000 if carry else decision+(5000 if policy_id == 'SIDE_LEAD' else 30000)
            if decision < horizon:
                yield Signal(policy_id, -native_direction if carry else native_direction,
                             decision, horizon, (4 if carry else 2)*unit,
                             None if carry else 2*unit)
        clock += CLOCK_MS


def _public_outcome(outcome, unit):
    row = asdict(outcome)
    for field in ('entry_quote_price', 'entry_price', 'exit_quote_price', 'exit_price',
                  'stop_distance', 'target_distance', 'signed_move_after_spread_and_slippage'):
        if row[field] is not None:
            row[field] /= PRICE_SCALE
    row['config']['adverse_slippage_per_leg'] /= PRICE_SCALE
    for field in ('entry_bundle', 'exit_bundle'):
        row[field] = [dict(time_msc=q.time_msc, bid=q.bid/PRICE_SCALE, ask=q.ask/PRICE_SCALE)
                      for q in getattr(outcome, field)]
    row['R'] = (None if outcome.status != 'COMPLETE' else
                outcome.signed_move_after_spread_and_slippage/outcome.stop_distance)
    if row['R'] is not None and not math.isfinite(row['R']):
        raise ValueError('Conditional R must remain finite')
    row['move_units'] = 'QUANTIZED_QUOTE_PRICE_UNITS_NOT_CASH'
    row['public_price_scale_divisor'] = PRICE_SCALE
    row['training_unit_quote'] = unit/PRICE_SCALE
    row['cash_net'] = None
    return row


def _mark_trade(tape, outcome, prior_r, tracker):
    """Conservative same-ms envelope; no within-bundle market order is asserted.

    Every raw row contributes its opposite-side liquidation mark, including
    pending exit. Process each bundle's favourable mark before its adverse mark
    to retain the conservative drawdown envelope under unknown same-ms order.
    """
    first = bisect_left(tape.timestamps, outcome.entry_time_msc)
    end = bisect_right(tape.timestamps, outcome.exit_time_msc)
    index = first
    base = 10000+25*prior_r
    slip = outcome.config.adverse_slippage_per_leg
    while index < end:
        stamp = tape.timestamps[index]
        stop = bisect_right(tape.timestamps, stamp, lo=index, hi=end)
        marks = []
        for quote in tape.quotes[index:stop]:
            side = quote.bid if outcome.direction == 1 else quote.ask
            assumed_exit = side-outcome.direction*slip
            if assumed_exit <= 0 or not math.isfinite(assumed_exit):
                raise ValueError('Assumed marked exit price must be finite and positive')
            move = outcome.direction*(assumed_exit-outcome.entry_price)
            marks.append(base+25*(move/outcome.stop_distance))
        tracker.mark(max(marks), stamp)
        tracker.mark(min(marks), stamp)
        tracker.marked_rows += stop-index
        index = stop
    return end-first


def _pf(gain, loss):
    if loss == 0:
        return None, 'NO_OBSERVED_LOSSES' if gain > 0 else 'NO_NONZERO_COMPLETED_MOVES'
    value = gain/loss
    if not math.isfinite(value):
        raise ValueError('Conditional profit factor must remain finite')
    return value, 'CONDITIONAL_QUOTE_R_COMPONENTS_ONLY'


def evaluate_tape(tape: QuoteTape, policy_id: str, native_direction: int, unit: float,
                  stage_start_ms: int, stage_end_ms: int,
                  execution_config: ExecutionConfig, baseline: bool = False) -> dict:
    """Measure one frozen policy/stage; UNKNOWN censors all primary stage metrics.

    Decisions lie in [stage_start+30min, stage_end-30min). Features and execution
    use only this stage's quotes. A COMPLETE exposure reserves its decision,
    entry latency, hold and exit latency through its inclusive exit timestamp.
    SKIP reserves no exposure. At UNKNOWN the generator is closed immediately;
    unmeasured later opportunities are unknown, never counted as zero returns.
    """
    if not isinstance(tape, QuoteTape):
        raise ValueError('tape must be a validated QuoteTape')
    if policy_id not in POLICY_IDS:
        raise ValueError('Unknown frozen policy_id')
    if type(native_direction) is not int or native_direction not in (-1, 1):
        raise ValueError('native_direction must be integer -1 or 1')
    if (isinstance(unit, bool) or not isinstance(unit, Real) or not math.isfinite(unit)
            or unit <= 0 or not math.isfinite(10*unit)):
        raise ValueError('unit must be a finite positive frozen training scale')
    if (type(stage_start_ms) is not int or type(stage_end_ms) is not int
            or stage_start_ms < 0 or stage_end_ms <= stage_start_ms):
        raise ValueError('Expected a nonnegative half-open stage window')
    if type(baseline) is not bool:
        raise ValueError('baseline must be boolean')
    if not isinstance(execution_config, ExecutionConfig):
        raise ValueError('execution_config must be ExecutionConfig')
    # Reconstruct to validate even a frozen dataclass deliberately mutated by a caller.
    execution_config = ExecutionConfig(**asdict(execution_config))
    unit = float(unit)
    first, last = (bisect_left(tape.timestamps, stage_start_ms),
                   bisect_left(tape.timestamps, stage_end_ms))
    stage_tape = tape if first == 0 and last == len(tape) else QuoteTape(tape.quotes[first:last])
    eligible_start, eligible_end = stage_start_ms+EMBARGO_MS, stage_end_ms-EMBARGO_MS
    daily, daily_sums = {}, {}
    for day_start in range((stage_start_ms//DAY_MS)*DAY_MS,
                           ((stage_end_ms-1)//DAY_MS+1)*DAY_MS, DAY_MS):
        daily[day_start] = {'day_start_ms': day_start,
                            'date_utc': datetime.fromtimestamp(day_start/1000, timezone.utc).date().isoformat(),
                            'observed_sum_R': 0.0, 'observed_sum_quote': 0.0,
                            'observed_gain_R': 0.0, 'observed_loss_R': 0.0,
                            'completed': 0, 'unknown': 0, 'cash_net': None,
                            'safety': {key: False for key in SAFETY_FIELDS}}
        daily_sums[day_start] = {key: _Sum() for key in
                                 ('observed_sum_R', 'observed_sum_quote', 'observed_gain_R', 'observed_loss_R')}
    clock_stats = dict(scheduled=0, missing_observation=0, stale_first_observation=0)
    signals = (_baseline_signals(stage_tape, policy_id, native_direction, unit,
                                 stage_start_ms, stage_end_ms, clock_stats) if baseline else
               generate_signals(stage_tape, policy_id=policy_id,
                                native_direction=native_direction, spread_unit=unit))
    ledger, observed_r, observed_quote, gain_r, loss_r = [], _Sum(), _Sum(), _Sum(), _Sum()
    tracker = _Drawdown(stage_start_ms)
    reserved_until, unknown_from = None, None
    completed = unknown = skipped = blocked = excluded = excluded_after = 0
    try:
        for signal in signals:
            if signal.decision_ms >= eligible_end:
                excluded_after += 1
                break
            if signal.decision_ms < eligible_start:
                excluded += 1
                continue
            if reserved_until is not None and signal.decision_ms <= reserved_until:
                blocked += 1
                continue
            outcome = stage_tape.simulate(signal.decision_ms, signal.direction,
                                           signal.horizon_end_ms, signal.stop_distance,
                                           signal.target_distance, execution_config)
            row = _public_outcome(outcome, unit)
            row.update(policy_id=policy_id, baseline=baseline)
            ledger.append(row)
            if outcome.status == 'UNKNOWN':
                unknown, unknown_from = 1, signal.decision_ms
                daily[(unknown_from//DAY_MS)*DAY_MS]['unknown'] += 1
                break
            if outcome.status == 'SKIP':
                skipped += 1
                continue
            row['marked_quote_rows'] = _mark_trade(stage_tape, outcome, observed_r.value, tracker)
            value = row['R']
            observed_r.add(value)
            observed_quote.add(outcome.signed_move_after_spread_and_slippage/PRICE_SCALE)
            (gain_r if value > 0 else loss_r).add(abs(value))
            completed += 1
            reserved_until = outcome.exit_time_msc
            row['conditional_equity_index_after_exit'] = 10000+25*observed_r.value
            tracker.mark(row['conditional_equity_index_after_exit'], reserved_until)
            day_start = (reserved_until//DAY_MS)*DAY_MS
            day = daily[day_start]
            daily_sums[day_start]['observed_sum_R'].add(value)
            daily_sums[day_start]['observed_sum_quote'].add(outcome.signed_move_after_spread_and_slippage/PRICE_SCALE)
            daily_sums[day_start]['observed_gain_R'].add(max(0.0, value))
            daily_sums[day_start]['observed_loss_R'].add(max(0.0, -value))
            day['completed'] += 1
    finally:
        signals.close()
    censored = unknown != 0
    observed_pf, pf_reason = _pf(gain_r.value, loss_r.value)
    for day_start, day in daily.items():
        for key, accumulator in daily_sums[day_start].items():
            day[key] = accumulator.value
        day['sum_R'] = None if censored else day['observed_sum_R']
        day['sum_quote'] = None if censored else day['observed_sum_quote']
        day['primary_measurement_censored'] = censored
    summary = {
        'sum_R': None if censored else observed_r.value, 'observed_sum_R': observed_r.value,
        'sum_quote': None if censored else observed_quote.value,
        'observed_sum_quote': observed_quote.value,
        'observed_gain_R': gain_r.value, 'observed_loss_R': loss_r.value,
        'pf': None if censored else observed_pf, 'observed_pf': observed_pf,
        'pf_reason': 'UNKNOWN_AND_REMAINING_CENSORED' if censored else pf_reason,
        'completed': completed, 'unknown': unknown, 'skipped': skipped,
        'blocked_overlap': blocked, 'excluded_before_embargo': excluded,
        'excluded_after_embargo': excluded_after,
        'remaining_censored': censored, 'unknown_and_remaining_censored': censored,
        'remaining_opportunities_count': None if censored else 0,
        'censored_from_ms': unknown_from,
        'censored_until_exclusive_ms': eligible_end if censored else None,
        'marked_drawdown_pct': None if censored else 100*tracker.maximum,
        'observed_completed_prefix_marked_drawdown_pct': 100*tracker.maximum,
        'marked_quote_rows': tracker.marked_rows,
        'drawdown_peak_time_ms': None if censored else tracker.maximum_peak_time_ms,
        'drawdown_trough_time_ms': None if censored else tracker.trough_time_ms,
        'ending_equity_index': None if censored else 10000+25*observed_r.value,
        'observed_completed_prefix_ending_equity_index': 10000+25*observed_r.value,
        'primary_measurement_usable': not censored,
        'qualified_for_profit_claim': False, 'cash_net': None,
        'median_trade_R_observed': statistics.median([r['R'] for r in ledger if r['R'] is not None]) if completed else None,
        'median_holding_ms_observed': statistics.median([
            r['exit_time_msc']-r['entry_time_msc'] for r in ledger if r['status']=='COMPLETE']) if completed else None,
        'median_holding_scope': 'COMPLETED_OBSERVED_PREFIX_ONLY_NOT_TIME_TO_SPIKE',
    }
    return {
        'status': ('UNKNOWN_AND_REMAINING_CENSORED' if censored else
                   'NO_ELIGIBLE_DECISION_WINDOW' if eligible_end <= eligible_start else
                   'CONDITIONAL_QUOTE_MEASUREMENT_ONLY'),
        'policy_id': policy_id, 'baseline': baseline, 'native_direction': native_direction,
        'stage_start_ms': stage_start_ms, 'stage_end_exclusive_ms': stage_end_ms,
        'eligible_decision_start_ms': eligible_start,
        'eligible_decision_end_exclusive_ms': eligible_end,
        'summary': summary, 'daily': list(daily.values()), 'ledger': ledger,
        'baseline_clock': {'period_ms': CLOCK_MS, 'clock': 'UTC_MINUTE_00_AND_30',
                           'freshness_limit_ms': 2000, **clock_stats} if baseline else None,
        'equity_model': 'CONDITIONAL_INDEX_10000_PLUS_25_TIMES_CUMULATIVE_R_NOT_CASH',
        'drawdown_model': 'OPPOSITE_SIDE_PLUS_EXIT_SLIPPAGE_SAME_MS_FAVOURABLE_THEN_ADVERSE_ENVELOPE',
        'same_millisecond_order_known': False,
        'model_label': 'CONDITIONAL_EVENT_TIME_NOT_REAL_ARRIVAL_OR_FILL',
        'price_scale_divisor': PRICE_SCALE, 'cost_verified': False,
        'execution_verified': False, 'cash_net': None,
        'full_tick_coverage_verified': False, 'cash_profit_tested': False,
        'conditional_quote_measurement_performed': True,
        'safety': {key: False for key in SAFETY_FIELDS},
    }
