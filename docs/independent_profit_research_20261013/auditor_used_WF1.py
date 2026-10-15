"""Independent WF1 saved-ledger arithmetic audit, never a strategy/backtest runner.

Only declared development shards are readable. Raw timestamps, entry/exit bundle
rows and HALF_EVEN quantized prices are checked independently with Decimal. All
completed trades receive arithmetic/side/latency/gap checks; at most 64 evenly
spaced completed paths per case receive first-barrier chronology checks. Issuance
features and full marked drawdown are explicitly not reproduced.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
STUDY = 'work/independent_profit_study_20261013'
DATA = 'data/independent_profit_ticks_20261013'
OUTPUT = 'work/independent_quote_audit_20261013'
SCALE = Decimal(10**8)
DAY = 86400000
START = 1735776000000
WF_START, WF_END = START+30*DAY, START+50*DAY
FLAGS = ('LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES',
         'live_trading', 'ready_for_live', 'live_allowed', 'opened_trades')
POLICIES = ('POST_JUMP_CARRY_30S', 'QUOTE_CLOCK_ACCELERATION', 'SIDE_LEAD')
SYMBOLS = {'BOOM500': ('Boom 500 Index', 1), 'CRASH500': ('Crash 500 Index', -1)}
RAW_FIELDS = ('time', 'time_msc', 'bid', 'ask', 'last', 'volume', 'flags', 'volume_real')
HASH = re.compile(r'[0-9a-f]{64}\Z')
MAX_JSON = 256*1024*1024


class AuditError(ValueError):
    pass


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise AuditError('Duplicate JSON key: '+key)
        value[key] = item
    return value


def _json(data):
    try:
        return json.loads(data, parse_float=Decimal, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(AuditError('Nonfinite JSON')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuditError('Invalid strict JSON') from error


def _path(root, relative):
    relative = Path(relative)
    if (relative.is_absolute() or '..' in relative.parts or
            any('sealed_final' in p or 'private_history' in p for p in relative.parts)):
        raise AuditError('Forbidden or noncanonical source path')
    path = root/relative
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise AuditError('Expected regular file without symlink ancestors')
    return path


def _bytes(root, relative, expected, maximum=MAX_JSON):
    if not isinstance(expected, str) or not HASH.fullmatch(expected):
        raise AuditError('Invalid SHA256 pin')
    path = _path(root, relative)
    before = path.stat()
    if before.st_size > maximum:
        raise AuditError('Bounded input exceeded')
    data = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise AuditError('Input changed while reading')
    if hashlib.sha256(data).hexdigest() != expected:
        raise AuditError('SHA256 mismatch: '+str(relative))
    return data


def _sha(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest


def _int(value, label):
    if type(value) is not int or not 0 <= value <= 2**63-1:
        raise AuditError('Invalid integer: '+label)
    return value


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise AuditError('Expected finite numeric value')
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise AuditError('Invalid decimal') from error
    if not number.is_finite() or abs(number) > Decimal('1.7976931348623157e308'):
        raise AuditError('Nonfinite or out-of-domain decimal')
    return number


def _scaled(value):
    number = _decimal(value)
    if number <= 0:
        raise AuditError('Nonpositive raw quote')
    with localcontext() as context:
        context.prec = 160
        scaled = (number*SCALE).to_integral_value(rounding=ROUND_HALF_EVEN)
    if not 0 < scaled < 2**45:
        raise AuditError('Raw quote outside declared quantization domain')
    return scaled


def _false_flags(value):
    return isinstance(value, dict) and set(value) == set(FLAGS) and all(value[k] is False for k in FLAGS)


def sample_indices(ledger, maximum=64):
    indices = [i for i, row in enumerate(ledger) if row.get('status') == 'COMPLETE']
    if len(indices) <= maximum:
        return set(indices)
    return {indices[i*(len(indices)-1)//(maximum-1)] for i in range(maximum)}


def scan_raw(root, shards, needed_times, queries, intervals):
    """One streaming pass; retain only requested bundles or selected complete paths."""
    bounds = []
    for left, right in sorted(intervals):
        if bounds and left <= bounds[-1][1]:
            bounds[-1] = (bounds[-1][0], max(right, bounds[-1][1]))
        else:
            bounds.append((left, right))
    wanted = sorted(set(queries))
    cursor = interval = total = 0
    previous = None
    groups, successors, prefixes, gaps = {}, {}, {}, []
    raw_rows = 0
    for shard in shards:
        path = _path(root, shard['path'])
        before = path.stat()
        digest = hashlib.sha256()
        count = 0
        def lines():
            with path.open('rb') as stream:
                for ordinal, line in enumerate(stream):
                    if len(line) > 8192:
                        raise AuditError('Unbounded raw row')
                    digest.update(line)
                    yield line.decode('utf-8-sig' if ordinal == 0 else 'utf-8')
        reader = csv.reader(lines(), delimiter='\t', strict=True)
        if next(reader, None) != list(RAW_FIELDS):
            raise AuditError('Unexpected strict raw TSV header')
        for row in reader:
            count += 1
            if len(row) != 8 or count > 1000000:
                raise AuditError('Malformed or unbounded raw shard')
            if any(any(ord(c) < 32 or ord(c) == 127 for c in cell) for cell in row):
                raise AuditError('Control character in raw field')
            if any(not re.fullmatch(r'(?:0|[1-9][0-9]*)', row[i]) for i in (0, 1, 5, 6)):
                raise AuditError('Noncanonical raw integer')
            stamp = int(row[1])
            if (int(row[0]) != stamp//1000 or not shard['start_ms'] <= stamp < shard['end_ms']
                    or previous is not None and stamp < previous):
                raise AuditError('Raw source time/bounds/order mismatch')
            if int(row[5]) > 2**64-1 or int(row[6]) > 2**32-1:
                raise AuditError('Raw integer domain exceeded')
            raw_bid, raw_ask = _decimal(row[2]), _decimal(row[3])
            if raw_bid <= 0 or raw_bid > raw_ask or _decimal(row[4]) < 0 or _decimal(row[7]) < 0:
                raise AuditError('Invalid/crossed raw quote or negative auxiliary field')
            bid, ask = _scaled(row[2]), _scaled(row[3])
            if previous != stamp:
                if previous is not None and stamp-previous > 5000:
                    gaps.append((previous, stamp))
                while cursor < len(wanted) and wanted[cursor] <= stamp:
                    successors[wanted[cursor]] = stamp
                    prefixes[wanted[cursor]] = total
                    cursor += 1
            while interval < len(bounds) and stamp > bounds[interval][1]:
                interval += 1
            in_sample = interval < len(bounds) and bounds[interval][0] <= stamp <= bounds[interval][1]
            if stamp in needed_times or in_sample:
                groups.setdefault(stamp, []).append((bid, ask))
            total += 1
            previous = stamp
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise AuditError('Raw source changed during scan')
        if digest.hexdigest() != shard['sha256']:
            raise AuditError('Raw source SHA256 mismatch')
        if count != shard['rows'] or count == 0:
            raise AuditError('Raw source row count mismatch')
        raw_rows += count
    while cursor < len(wanted):
        successors[wanted[cursor]], prefixes[wanted[cursor]] = None, total
        cursor += 1
    return {'groups': groups, 'group_times': tuple(sorted(groups)), 'successors': successors, 'prefixes': prefixes,
            'gaps': gaps, 'gap_rights': [b for _, b in gaps], 'rows': raw_rows}


def _gap(scan, left, right):
    index = bisect_left(scan['gap_rights'], left+1)
    return index < len(scan['gaps']) and scan['gaps'][index][0] < right


def audit_case(case, ledger, scan, sampled):
    """Arithmetic checks against independently parsed raw; errors are retained, never repaired."""
    errors = []
    def check(ok, field, ordinal=None):
        if not ok:
            errors.append({'field': field, 'ledger_ordinal': ordinal})
    def equal(actual, expected, field, ordinal=None, price=False):
        if expected is None:
            check(actual is None, field, ordinal)
            return
        try:
            if isinstance(actual, bool) or not isinstance(actual, (int, float, Decimal)):
                raise AuditError('Saved numeric fields must be JSON numbers')
            actual = _decimal(actual)
            expected = _decimal(expected)
            tolerance = Decimal('1e-9') if price else max(Decimal('1e-10'), abs(expected)*Decimal('1e-10'))
            check(abs(actual-expected) <= tolerance, field, ordinal)
        except AuditError:
            check(False, field, ordinal)
    unit = _decimal(case['U_scaled'])
    if unit <= 0 or unit*2 != (unit*2).to_integral_value() or unit >= 2**45:
        raise AuditError('Invalid frozen training U')
    policy, symbol, baseline = case['policy'], case['symbol'], case['baseline']
    native = SYMBOLS[symbol][1]
    direction = -native if policy == 'POST_JUMP_CARRY_30S' else native
    slip = unit*(Decimal('.25') if case['scenario'] == 'BASE' else Decimal('.5'))
    latency = 250 if case['scenario'] == 'BASE' else 2000
    stop = unit*(4 if policy == 'POST_JUMP_CARRY_30S' else 2)
    target = None if policy == 'POST_JUMP_CARRY_30S' else 2*unit
    daily = {day: {'R': Decimal(0), 'quote': Decimal(0), 'gain': Decimal(0),
                   'loss': Decimal(0), 'completed': 0, 'unknown': 0}
             for day in range(WF_START, WF_END, DAY)}
    total = gain = loss = quote_sum = Decimal(0)
    completed = unknown = skipped = matched = checked_paths = 0
    reserved = previous_decision = None
    observed_returns, holdings = [], []
    for ordinal, row in enumerate(ledger):
        status = row['status']
        if status not in ('COMPLETE', 'UNKNOWN', 'SKIP'):
            raise AuditError('Unknown ledger status')
        decision, horizon = _int(row['decision_ms'], 'decision'), _int(row['horizon_end_ms'], 'horizon')
        check(WF_START+1800000 <= decision < WF_END-1800000, 'decision_embargo', ordinal)
        check(previous_decision is None or decision > previous_decision, 'decision_order', ordinal)
        check(reserved is None or decision > reserved, 'overlapping_pending_exposure', ordinal)
        check(scan['successors'].get(decision) == decision, 'decision_observed_clock', ordinal)
        check(row['direction'] == direction and type(row['direction']) is int, 'direction', ordinal)
        check(row['policy_id'] == policy and row['baseline'] is baseline, 'ledger_policy_identity', ordinal)
        check(row['config']['entry_latency_ms'] == latency and row['config']['exit_latency_ms'] == latency
              and row['config']['max_gap_ms'] == 5000, 'latency_gap_config', ordinal)
        equal(row['config']['adverse_slippage_per_leg'], slip/SCALE, 'slippage', ordinal, True)
        equal(row['training_unit_quote'], unit/SCALE, 'training_U', ordinal, True)
        equal(row['stop_distance'], stop/SCALE, 'stop_distance', ordinal, True)
        equal(row['target_distance'], None if target is None else target/SCALE, 'target_distance', ordinal, True)
        check(row['public_price_scale_divisor'] == 10**8, 'public_price_scale', ordinal)
        check(row['cash_pnl'] is None and row['cash_net'] is None and row['costs_verified'] is False
              and row['execution_verified'] is False, 'unverified_cash_costs_execution', ordinal)
        if policy != 'POST_JUMP_CARRY_30S':
            check(horizon == decision+(5000 if policy == 'SIDE_LEAD' else 30000), 'fixed_horizon', ordinal)
        else:
            due = horizon-28000
            check(scan['successors'].get(due) == decision, 'carry_first_due_observation', ordinal)
            if baseline:
                check((horizon-30000) % 1800000 == 0, 'carry_baseline_clock', ordinal)
        previous_decision = decision
        entry, exit_ = row['entry_time_msc'], row['exit_time_msc']
        first_entry = scan['successors'].get(decision+latency)
        effective_entry = None
        if entry is not None:
            entry = _int(entry, 'entry')
            check(entry == first_entry and entry < horizon, 'first_eligible_entry', ordinal)
            bundle = scan['groups'].get(entry, [])
            if not bundle:
                check(False, 'entry_raw_bundle_missing', ordinal)
            else:
                raw_entry = max(ask for _, ask in bundle) if direction == 1 else min(bid for bid, _ in bundle)
                effective_entry = raw_entry+direction*slip
                equal(row['entry_quote_price'], raw_entry/SCALE, 'entry_worst_side', ordinal, True)
                equal(row['entry_price'], effective_entry/SCALE, 'effective_entry', ordinal, True)
                _bundle_check(row['entry_bundle'], entry, bundle, equal, check, 'entry_bundle', ordinal)
        else:
            check(row['entry_price'] is None and row['entry_quote_price'] is None and row['entry_bundle'] == [],
                  'missing_entry_null_fields', ordinal)
        if status != 'COMPLETE':
            for field in ('R', 'signed_move_after_spread_and_slippage', 'exit_time_msc', 'exit_price', 'exit_quote_price'):
                check(row[field] is None, 'unknown_or_skip_null_'+field, ordinal)
            check(row['exit_bundle'] == [], 'unknown_or_skip_empty_exit_bundle', ordinal)
            if status == 'UNKNOWN':
                unknown += 1
                daily[(decision//DAY)*DAY]['unknown'] += 1
                check(ordinal == len(ledger)-1, 'unknown_censors_remaining_ledger', ordinal)
                reason = row['reason']
                if reason == 'MISSING_ENTRY_QUOTE':
                    check(first_entry is None, 'missing_entry_source_evidence', ordinal)
                elif reason.startswith('MISSING_'):
                    trigger = row['trigger_time_msc']
                    check(trigger is not None and scan['successors'].get(trigger+latency) is None,
                          'missing_exit_source_evidence', ordinal)
                elif reason.startswith('GAP_'):
                    endpoint = scan['successors'].get((row['trigger_time_msc'] or horizon)+latency)
                    check(_gap(scan, decision, endpoint if endpoint is not None else WF_END),
                          'unknown_gap_source_evidence', ordinal)
                else:
                    check(False, 'unsupported_unknown_reason', ordinal)
            else:
                skipped += 1
                check(entry is None and row['entry_price'] is None and row['entry_bundle'] == [], 'skip_no_exposure', ordinal)
                check(decision+latency >= horizon or first_entry is not None and first_entry >= horizon,
                      'skip_entry_timing_evidence', ordinal)
            continue
        completed += 1
        exit_ = _int(exit_, 'exit')
        trigger = _int(row['trigger_time_msc'], 'trigger')
        check(entry is not None and entry <= trigger <= exit_, 'entry_trigger_exit_order', ordinal)
        check(scan['successors'].get(trigger+latency) == exit_, 'first_eligible_exit', ordinal)
        check(not _gap(scan, decision, exit_), 'complete_path_gap_above_5000ms', ordinal)
        if row['trigger_reason'] == 'HORIZON':
            check(trigger == horizon and row['trigger_source_time_msc'] is None
                  and row['both_barriers_in_bundle'] is False, 'horizon_trigger', ordinal)
        else:
            source = row['trigger_source_time_msc']
            check(type(source) is int and entry <= source < trigger < horizon
                  and scan['successors'].get(source+1) == trigger, 'barrier_confirmation_not_backdated', ordinal)
        bundle = scan['groups'].get(exit_, [])
        if not bundle or effective_entry is None:
            check(False, 'complete_raw_bundle_missing', ordinal)
            continue
        raw_exit = min(bid for bid, _ in bundle) if direction == 1 else max(ask for _, ask in bundle)
        effective_exit = raw_exit-direction*slip
        move = direction*(effective_exit-effective_entry)
        value = move/stop
        equal(row['exit_quote_price'], raw_exit/SCALE, 'exit_worst_side', ordinal, True)
        equal(row['exit_price'], effective_exit/SCALE, 'effective_exit', ordinal, True)
        equal(row['signed_move_after_spread_and_slippage'], move/SCALE, 'signed_move', ordinal, True)
        equal(row['R'], value, 'R', ordinal)
        _bundle_check(row['exit_bundle'], exit_, bundle, equal, check, 'exit_bundle', ordinal)
        count = scan['prefixes'][exit_+1]-scan['prefixes'][entry]
        check(row['marked_quote_rows'] == count, 'marked_quote_row_count', ordinal)
        total += value
        quote_sum += move/SCALE
        gain += max(Decimal(0), value)
        loss += max(Decimal(0), -value)
        equal(row['conditional_equity_index_after_exit'], Decimal(10000)+25*total, 'conditional_equity_index', ordinal)
        observed_returns.append(value); holdings.append(exit_-entry)
        day = daily[(exit_//DAY)*DAY]
        day['R'] += value; day['quote'] += move/SCALE
        day['gain'] += max(Decimal(0), value); day['loss'] += max(Decimal(0), -value)
        day['completed'] += 1
        reserved = exit_
        matched += 1
        if ordinal in sampled:
            _path_check(row, scan, effective_entry, stop, target, check, ordinal)
            checked_paths += 1
    summary = case['summary']; censored = unknown != 0
    for key, expected in (('completed', completed), ('unknown', unknown), ('skipped', skipped)):
        check(type(summary[key]) is int and summary[key] == expected, 'summary_'+key)
    for key in ('remaining_censored', 'unknown_and_remaining_censored'):
        check(summary[key] is censored, 'summary_'+key)
    check(summary['primary_measurement_usable'] is (not censored), 'primary_measurement_usable')
    for key, expected in (('observed_sum_R', total), ('observed_sum_quote', quote_sum),
                          ('observed_gain_R', gain), ('observed_loss_R', loss),
                          ('observed_pf', gain/loss if loss else None),
                          ('observed_completed_prefix_ending_equity_index', Decimal(10000)+25*total)):
        equal(summary[key], expected, 'summary_'+key)
    for key, expected in (('sum_R', total), ('sum_quote', quote_sum),
                          ('pf', gain/loss if loss else None), ('ending_equity_index', Decimal(10000)+25*total)):
        equal(summary[key], None if censored else expected, 'summary_'+key)
    if censored:
        check(summary['marked_drawdown_pct'] is None and summary['drawdown_peak_time_ms'] is None
              and summary['drawdown_trough_time_ms'] is None, 'unknown_primary_drawdown_null')
        check(summary['censored_from_ms'] == ledger[-1]['decision_ms']
              and summary['censored_until_exclusive_ms'] == WF_END-1800000
              and summary['remaining_opportunities_count'] is None, 'unknown_censor_bounds')
    def median(values):
        ordered = sorted(values)
        if not ordered:
            return None
        middle = len(ordered)//2
        return _decimal(ordered[middle]) if len(ordered)%2 else (_decimal(ordered[middle-1])+_decimal(ordered[middle]))/2
    equal(summary['median_trade_R_observed'], median(observed_returns), 'median_trade_R_observed')
    equal(summary['median_holding_ms_observed'], median(holdings), 'median_holding_ms_observed')
    check(summary['cash_net'] is None and summary['qualified_for_profit_claim'] is False, 'summary_not_cash_profit')
    check(len(case['daily']) == 20 and [d['day_start_ms'] for d in case['daily']] == list(daily), 'daily_full_UTC_calendar')
    for day in case['daily']:
        if day['day_start_ms'] not in daily:
            continue
        expected = daily[day['day_start_ms']]
        for key, target_key in (('observed_sum_R', 'R'), ('observed_sum_quote', 'quote'),
                                ('observed_gain_R', 'gain'), ('observed_loss_R', 'loss')):
            equal(day[key], expected[target_key], 'daily_'+key, day['day_start_ms'])
        for key in ('completed', 'unknown'):
            check(type(day[key]) is int and day[key] == expected[key], 'daily_'+key, day['day_start_ms'])
        equal(day['sum_R'], None if censored else expected['R'], 'daily_primary_R', day['day_start_ms'])
        equal(day['sum_quote'], None if censored else expected['quote'], 'daily_primary_quote', day['day_start_ms'])
        check(day['primary_measurement_censored'] is censored and day['cash_net'] is None
              and _false_flags(day['safety']), 'daily_safety_and_censor', day['day_start_ms'])
    return {'errors': errors, 'ledger_rows': len(ledger), 'completed': completed, 'unknown': unknown,
            'skipped': skipped, 'complete_entry_exit_bundles_checked': matched,
            'sampled_complete_paths_checked': checked_paths, 'arithmetic_matches': not errors}


def _bundle_check(saved, stamp, raw, equal, check, label, ordinal):
    check(isinstance(saved, list) and len(saved) == len(raw), label+'_multiplicity', ordinal)
    for actual, (bid, ask) in zip(saved, raw):
        check(set(actual) == {'time_msc', 'bid', 'ask'} and actual['time_msc'] == stamp, label+'_timestamp_schema', ordinal)
        equal(actual['bid'], bid/SCALE, label+'_ordered_bid', ordinal, True)
        equal(actual['ask'], ask/SCALE, label+'_ordered_ask', ordinal, True)


def _path_check(row, scan, entry, stop, target, check, ordinal):
    direction, horizon = row['direction'], row['horizon_end_ms']
    all_times = scan['group_times']
    times = all_times[bisect_left(all_times, row['entry_time_msc']):bisect_left(all_times, row['exit_time_msc']+1)]
    expected = (None, horizon, 'HORIZON', False)
    for source, confirmation in zip(times, times[1:]):
        if confirmation >= horizon:
            break
        sides = [q[0 if direction == 1 else 1] for q in scan['groups'][source]]
        adverse = min(sides) if direction == 1 else max(sides)
        favourable = max(sides) if direction == 1 else min(sides)
        stopped = direction*(adverse-entry) <= -stop
        targeted = target is not None and direction*(favourable-entry) >= target
        if stopped or targeted:
            expected = source, confirmation, 'STOP' if stopped else 'TARGET', bool(stopped and targeted)
            break
    actual = (row['trigger_source_time_msc'], row['trigger_time_msc'], row['trigger_reason'], row['both_barriers_in_bundle'])
    check(actual == expected, 'sample_first_barrier_horizon_stop_priority', ordinal)
    check(row['reason'] == ('STOP_PRIORITY_SAME_MILLISECOND' if expected[3] else expected[2]), 'sample_exit_reason', ordinal)


def audit(root, declaration_sha):
    """Read exactly the existing 24 WF1 cases; never open any final/raw private path."""
    root = Path(root).absolute()
    declaration_bytes = _bytes(root, STUDY+'/declaration.json', declaration_sha, 1048576)
    declaration = _json(declaration_bytes)
    if not all(declaration.get(k) is False for k in FLAGS):
        raise AuditError('Unsafe measurement declaration flags')
    for path, pinned in declaration['source_pins'].items():
        _bytes(root, path, pinned, 4*1024*1024)
    for name in ('protocol', 'premeasurement_validation'):
        _bytes(root, declaration[name+'_path'], declaration[name+'_sha256'], 4*1024*1024)
    acquisition = _json(_bytes(root, DATA+'/acquisition_declaration.json', declaration['acquisition_declaration_sha256'], 4*1024*1024))
    if not all(acquisition.get(k) is False for k in FLAGS):
        raise AuditError('Unsafe acquisition declaration flags')
    index_path = _path(root, STUDY+'/artifact_index.json')
    index = _json(index_path.read_bytes())
    if index.get('declaration.json') != declaration_sha:
        raise AuditError('Declaration not registered in artifact index')
    cases = []; sources = {}
    for symbol in SYMBOLS:
        requests = [r for r in acquisition['requests'] if r['symbol'] == symbol and 30 <= r['day_index'] < 50]
        if len(requests) != 20 or {r['day_index'] for r in requests} != set(range(30, 50)):
            raise AuditError('Expected exact WF1 source calendar')
        expected = {}
        for r in requests:
            if (r['partition'] != 'development' or r['start_ms'] != START+r['day_index']*DAY
                    or r['end_exclusive_ms'] != r['start_ms']+DAY
                    or not re.fullmatch('[A-Za-z0-9_-]{1,64}', r['prefix'])):
                raise AuditError('Non-WF1 development source request')
            folder = DATA+'/development/'+r['prefix']+'/'
            expected.update({folder+r['prefix']+'_receipt.tsv': r, folder+r['prefix']+'_ticks.tsv': r,
                             folder+'independent_verification.json': r})
        for policy in POLICIES:
            for scenario in ('BASE', 'STRESS'):
                for baseline in (False, True):
                    stem = f'{symbol}_WF1_{policy}_{scenario}'+('_CLOCK' if baseline else '')
                    relative = STUDY+'/cases/'+stem+'.json'
                    case = _json(_bytes(root, relative, index.get('cases/'+stem+'.json'), 16*1024*1024))
                    if (case['declaration_sha256'] != declaration_sha or case['symbol'] != symbol or case['policy'] != policy
                            or case['policy_id'] != policy or case['scenario'] != scenario or case['baseline'] is not baseline
                            or case['stage'] != 'WF1' or case['stage_start_ms'] != WF_START or case['stage_end_exclusive_ms'] != WF_END
                            or case['native_direction'] != SYMBOLS[symbol][1] or case['price_scale_divisor'] != 10**8
                            or case['eligible_decision_start_ms'] != WF_START+1800000
                            or case['eligible_decision_end_exclusive_ms'] != WF_END-1800000
                            or set(case['source_files']) != set(expected) or not _false_flags(case['safety'])
                            or any(case.get(k) is not False for k in ('cost_verified', 'costs_verified', 'execution_verified', 'profit_qualified'))
                            or case['cash_net'] is not None):
                        raise AuditError('Case identity/window/safety/source mismatch')
                    if symbol in sources and case['source_files'] != sources[symbol]['pins']:
                        raise AuditError('Cases use different source pins for the same WF1 symbol')
                    sources[symbol] = {'pins': case['source_files'], 'expected': expected}
                    ledger_relative = STUDY+'/ledgers/'+stem+'.json.gz'
                    if case['ledger_path'] != ledger_relative:
                        raise AuditError('Unexpected saved ledger path')
                    compressed = _bytes(root, ledger_relative, case['ledger_sha256'], 64*1024*1024)
                    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
                        payload = stream.read(MAX_JSON+1)
                    if len(payload) > MAX_JSON:
                        raise AuditError('Unbounded saved ledger')
                    ledger = _json(payload)
                    if not isinstance(ledger, list) or len(ledger) > 1000000:
                        raise AuditError('Invalid saved ledger array')
                    cases.append((stem, case, ledger, sample_indices(ledger)))
    reports = {}; source_rows = {}
    with localcontext() as context:
        context.prec = 80
        for symbol, source in sources.items():
            needed, queries, intervals, shards = set(), set(), [], []
            selected = [item for item in cases if item[1]['symbol'] == symbol]
            for _, _, ledger, sampled in selected:
                for i, row in enumerate(ledger):
                    decision, horizon = _int(row['decision_ms'], 'decision'), _int(row['horizon_end_ms'], 'horizon')
                    latency = _int(row['config']['entry_latency_ms'], 'entry latency')
                    queries.update((decision, decision+latency, horizon-28000))
                    for key in ('entry_time_msc', 'exit_time_msc'):
                        if row[key] is not None:
                            stamp = _int(row[key], key); needed.add(stamp); queries.update((stamp, stamp+1))
                    if row['trigger_time_msc'] is not None:
                        queries.add(row['trigger_time_msc']+row['config']['exit_latency_ms'])
                    if row['trigger_source_time_msc'] is not None:
                        queries.add(row['trigger_source_time_msc']+1)
                    if i in sampled:
                        intervals.append((decision, row['exit_time_msc']))
            for relative, request in sorted(source['expected'].items(), key=lambda item: (item[1]['day_index'], item[0])):
                if relative.endswith('_ticks.tsv'):
                    continue
                payload = _bytes(root, relative, source['pins'][relative], 4*1024*1024)
                if relative.endswith('independent_verification.json'):
                    verified = _json(payload)
                    receipt = relative.rsplit('/', 1)[0]+'/'+request['prefix']+'_receipt.tsv'
                    raw = relative.rsplit('/', 1)[0]+'/'+request['prefix']+'_ticks.tsv'
                    if (verified['status'] != 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN' or verified['errors']
                            or verified['raw_sha256'] != source['pins'][raw]
                            or verified['receipt_sha256'] != source['pins'][receipt]
                            or verified['symbol'] != SYMBOLS[symbol][0]
                            or verified['requested_start_msc'] != request['start_ms']
                            or verified['requested_end_exclusive_msc'] != request['end_exclusive_ms']
                            or any(verified.get(k) is not False for k in
                                   ('cost_verified', 'execution_verified', 'full_tick_coverage_verified', 'profit_tested'))):
                        raise AuditError('Pinned source verification receipt mismatch')
                    shards.append({'path': raw, 'sha256': source['pins'][raw],
                                   'start_ms': request['start_ms'], 'end_ms': request['end_exclusive_ms'],
                                   'rows': _int(verified['rows'], 'verified rows')})
                else:
                    rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig')), delimiter='\t', strict=True))
                    if not rows or rows[0] != ['field', 'value'] or any(len(row) != 2 for row in rows[1:]):
                        raise AuditError('Invalid source metadata TSV')
                    metadata = _pairs(rows[1:])
                    if (metadata['symbol'] != SYMBOLS[symbol][0] or metadata['raw_file'] != request['prefix']+'_ticks.tsv'
                            or metadata['terminal_trade_allowed_observed'] != 'false'
                            or metadata['mql_trade_allowed_observed'] != 'false'
                            or metadata['account_identity_captured'] != 'false'
                            or any(metadata.get(k) != 'false' for k in
                                   ('live_trading', 'ready_for_live', 'live_allowed', 'opened_trades',
                                    'safety_upper_live_trading', 'safety_upper_ready_for_live', 'safety_upper_live_allowed', 'safety_upper_opened_trades'))):
                        raise AuditError('Unsafe source receipt context/flags')
            scan = scan_raw(root, sorted(shards, key=lambda item: item['start_ms']), needed, queries, intervals)
            source_rows[symbol] = scan['rows']
            for stem, case, ledger, sampled in selected:
                reports[stem] = audit_case(case, ledger, scan, sampled)
    failures = sum(len(report['errors']) for report in reports.values())
    return {'schema_version': 1, 'status': 'PASS' if failures == 0 else 'MISMATCH',
            'declaration_sha256': declaration_sha, 'stage': 'WF1', 'cases': reports,
            'source_rows_streamed_once_per_symbol': source_rows, 'mismatches': failures,
            'all_complete_trade_arithmetic_checked': True, 'all_saved_entry_exit_bundles_checked': True,
            'all_complete_path_timestamp_gap_checks': True, 'barrier_path_scope': 'DETERMINISTIC_AT_MOST_64_COMPLETE_PATHS_PER_CASE',
            'full_barrier_path_reproduction': False, 'signal_issuance_features_reproduced': False,
            'full_marked_drawdown_reproduced': False, 'cash_net': None, 'cost_verified': False,
            'execution_verified': False, 'profit_qualified': False, 'safety': {key: False for key in FLAGS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--expected-declaration-sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    output = (args.output or root/OUTPUT/'WF1.json').absolute()
    allowed = root/OUTPUT
    try:
        if not output.is_relative_to(allowed) or any(p.is_symlink() for p in (output, *output.parents)):
            raise AuditError('Output must be a new audit artifact under the fixed work directory')
        if output.exists():
            raise AuditError('Audit output already exists')
        report = audit(root, args.expected_declaration_sha256)
        report['auditor_sha256'] = _sha(Path(__file__))
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        print('AUDIT_PASS' if report['status'] == 'PASS' else 'AUDIT_MISMATCH')
        return 0 if report['status'] == 'PASS' else 1
    except (AuditError, OSError, KeyError, TypeError, ValueError, csv.Error):
        print('AUDIT_FAILED')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
