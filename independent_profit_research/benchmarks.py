"""Pure descriptive endpoint references; no position, cash, fee or path claim.

The input uses the study's frozen 1e-8 integer price grid. Every equal-ms row
remains available when choosing the adverse observed side. A price difference
between two retained endpoints is still reported across a source gap, explicitly
as ENDPOINT_ONLY_NOT_PATH_OR_CASH; it never becomes a COMPLETE simulated trade.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict
import math

from .data import MAX_SCALED_PRICE, PRICE_SCALE, SAFETY_FIELDS
from .execution import ExecutionConfig
from .tape import QuoteTape

EMBARGO_MS = 30*60*1000
SOURCE_GAP_MS = 5000
SCOPE = 'ENDPOINT_ONLY_NOT_PATH_OR_CASH'


def endpoint_reference(tape: QuoteTape, start_ms: int, end_ms: int,
                       native_direction: int,
                       execution_config: ExecutionConfig) -> dict:
    """Describe BUY and native-direction price endpoints in this stage only.

    Entry: first bundle >= start+30min+entry latency. Exit: first bundle >=
    end-30min+exit latency. Both must lie in [start,end), with exit after entry.
    No barrier is monitored. Slippage is the supplied adverse paper assumption,
    not measured execution. Missing/invalid endpoints retain null price moves.
    """
    if not isinstance(tape, QuoteTape):
        raise ValueError('tape must be a validated QuoteTape')
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms < 0 or end_ms <= start_ms):
        raise ValueError('Expected nonnegative half-open stage bounds')
    if type(native_direction) is not int or native_direction not in (-1, 1):
        raise ValueError('native_direction must be integer +1 or -1')
    if not isinstance(execution_config, ExecutionConfig):
        raise ValueError('execution_config must be ExecutionConfig')
    config = ExecutionConfig(**asdict(execution_config))
    if config.max_gap_ms != SOURCE_GAP_MS:
        raise ValueError('Endpoint reference uses the fixed 5000ms source-gap threshold')
    left = bisect_left(tape.timestamps, start_ms)
    right = bisect_left(tape.timestamps, end_ms)
    # Reconstruct snapshots to reject mutated caller-owned frozen objects while
    # retaining every observed row, its duplicate ordinal and original order.
    stage = QuoteTape(tape.quotes[left:right])
    for quote in stage:
        if not start_ms <= quote.time_msc < end_ms:
            raise ValueError('Quote snapshot no longer matches the half-open stage bounds')
        if any(not price.is_integer() or not 0 < price < MAX_SCALED_PRICE
               for price in (quote.bid, quote.ask)):
            raise ValueError('Expected positive integer-scaled prices below 2**45')
    stamps = stage.timestamps
    entry_due = start_ms+EMBARGO_MS+config.entry_latency_ms
    exit_due = end_ms-EMBARGO_MS+config.exit_latency_ms
    entry_index = bisect_left(stamps, entry_due)
    exit_index = bisect_left(stamps, exit_due)

    def bundle(index):
        if index == len(stage):
            return ()
        stop = bisect_right(stamps, stamps[index], lo=index)
        return stage.quotes[index:stop]

    entry, exit_ = bundle(entry_index), bundle(exit_index)
    first = stamps[0] if stamps else None
    last = stamps[-1] if stamps else None
    leading = first-start_ms if first is not None else end_ms-start_ms
    trailing = end_ms-1-last if last is not None else end_ms-start_ms
    stage_gaps = [(a,b) for a,b in zip(stamps,stamps[1:]) if b-a>SOURCE_GAP_MS]
    entry_time = entry[0].time_msc if entry else None
    exit_time = exit_[0].time_msc if exit_ else None
    ordered = bool(entry and exit_ and entry_time < exit_time)
    missing = ([] if entry else ['ENTRY'])+([] if exit_ else ['EXIT'])
    reasons = ['MISSING_'+name+'_ENDPOINT' for name in missing]
    if entry and exit_ and not ordered:
        reasons.append('ENDPOINTS_NOT_ORDERED')
    holding_gaps = ([(a,b) for a,b in stage_gaps if entry_time<=a<b<=exit_time]
                    if ordered else None)
    if holding_gaps:
        reasons.append('ANOMALOUS_GAP_UNRESOLVED')
    entry_delay = entry_time-entry_due if entry else None
    exit_delay = exit_time-exit_due if exit_ else None
    if entry_delay is not None and entry_delay>SOURCE_GAP_MS:
        reasons.append('ENTRY_TARGET_UNOBSERVED_OVER_5000MS')
    if exit_delay is not None and exit_delay>SOURCE_GAP_MS:
        reasons.append('EXIT_TARGET_UNOBSERVED_OVER_5000MS')
    path_quality = 'UNKNOWN' if reasons else 'OBSERVED_QUOTES_ONLY_COVERAGE_NOT_PROVEN'
    common = {
        'scope': SCOPE, 'status': SCOPE,
        'endpoint_status': 'AVAILABLE' if ordered else ('MISSING_ENDPOINTS' if missing else 'ENDPOINTS_NOT_ORDERED'),
        'entry_eligible_time_msc': entry_due, 'exit_eligible_time_msc': exit_due,
        'entry_time_msc': entry_time, 'exit_time_msc': exit_time,
        'entry_delay_from_target_ms': entry_delay, 'exit_delay_from_target_ms': exit_delay,
        'held_duration_ms': exit_time-entry_time if ordered else None,
        'held_duration_scope': 'OBSERVED_ENDPOINT_DISTANCE_NOT_BROKER_HOLDING',
        'entry_bundle_rows_preserved': len(entry), 'exit_bundle_rows_preserved': len(exit_),
        'endpoint_bundle_completeness': 'NOT_PROVEN',
        'missing_endpoints': missing, 'path_quality': path_quality,
        'path_unknown_reasons': reasons, 'source_gap_count': len(holding_gaps) if ordered else None,
        'source_gap_bounds_msc': [{'left_msc':a,'right_msc':b,'delta_ms':b-a}
                                  for a,b in holding_gaps or ()],
        'stage_source_gap_count': len(stage_gaps),
        'leading_unobserved_ms': leading, 'trailing_unobserved_ms': trailing,
        'boundary_intervals_are_additive': bool(stage),
        'dd': None, 'dd_scope': 'NOT_COMPUTED_ENDPOINT_REFERENCE',
        'cash_pnl': None, 'net_profit': None, 'fees': None, 'financing': None,
        'cost_verified': False, 'execution_verified': False, 'profit_qualified': False,
        'profit_tested': False, 'full_tick_coverage_verified': False,
        'stop_loss': None, 'take_profit': None,
        'safety': {name:False for name in SAFETY_FIELDS},
    }

    def reference(direction):
        out = dict(common, direction=direction,
                   entry_quote_price_scaled=None, exit_quote_price_scaled=None,
                   entry_effective_price_scaled=None, exit_effective_price_scaled=None,
                   endpoint_price_move_scaled=None, endpoint_price_move_quote_units=None)
        if ordered:
            # Opposite-side exit quotes include spread once. There is no stop,
            # target, financing assumption, contract multiplier or cash sizing.
            entry_quote = max(q.ask for q in entry) if direction==1 else min(q.bid for q in entry)
            exit_quote = min(q.bid for q in exit_) if direction==1 else max(q.ask for q in exit_)
            effective_entry = entry_quote+direction*config.adverse_slippage_per_leg
            effective_exit = exit_quote-direction*config.adverse_slippage_per_leg
            move = direction*(effective_exit-effective_entry)
            if not all(math.isfinite(v) for v in (effective_entry,effective_exit,move)):
                raise ValueError('Nonfinite endpoint arithmetic')
            out.update(entry_quote_price_scaled=entry_quote, exit_quote_price_scaled=exit_quote,
                       entry_effective_price_scaled=effective_entry,
                       exit_effective_price_scaled=effective_exit,
                       endpoint_price_move_scaled=move,
                       endpoint_price_move_quote_units=move/PRICE_SCALE)
        return out

    buy, native = reference(1), reference(native_direction)
    native['same_exposure_as_BUY'] = native_direction==1
    return {
        'schema_version': 1, 'scope': SCOPE,
        'stage_start_msc': start_ms, 'stage_end_exclusive_msc': end_ms,
        'embargo_ms': EMBARGO_MS, 'source_gap_threshold_ms': SOURCE_GAP_MS,
        'price_scale': PRICE_SCALE, 'input_price_units': 'INTEGER_1E_MINUS8_PRICE_UNITS',
        'observed_stage_rows': len(stage), 'first_observed_stage_msc': first,
        'last_observed_stage_msc': last, 'stage_source_gap_count': len(stage_gaps),
        'leading_unobserved_ms': leading, 'trailing_unobserved_ms': trailing,
        'execution_assumptions': asdict(config),
        'references': {'BUY': buy, 'NATIVE': native},
        'direction_duplicates_are_independent_samples': False,
        'cash_pnl': None, 'net_profit': None, 'fees': None, 'financing': None,
        'cost_verified': False, 'execution_verified': False, 'profit_tested': False,
        'full_tick_coverage_verified': False, 'used_for_acceptance_gate': False,
        'safety': {name:False for name in SAFETY_FIELDS},
    }
