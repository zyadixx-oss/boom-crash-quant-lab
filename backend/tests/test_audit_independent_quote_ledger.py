"""Independent fake raw files/receipts only; never open actual research evidence."""
from copy import deepcopy
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_independent_quote_ledger as a


D = a.WF_START+1800000
E, H, X = D+250, D+5000, D+5250


def write(root, relative, data):
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def raw_text(points):
    return '\t'.join(a.RAW_FIELDS)+'\n'+''.join(
        f'{t//1000}\t{t}\t{bid}\t{ask}\t0\t0\t6\t0\n' for t, bid, ask in points)


def trade(decision=D, *, losing=False):
    entry, horizon, exit_ = decision+250, decision+5000, decision+5250
    last_bid = 100 if losing else 103
    middle = 100 if losing else 102
    points = [(decision, 100, 101), (entry, 100, 101), (decision+500, middle, middle+1),
              (horizon, middle, middle+1), (exit_, last_bid, last_bid+1)]
    row = dict(status='COMPLETE', reason='HORIZON', decision_ms=decision, direction=1,
               horizon_end_ms=horizon, config=dict(entry_latency_ms=250, exit_latency_ms=250,
               max_gap_ms=5000, adverse_slippage_per_leg=.25), stop_distance=2, target_distance=2,
               entry_time_msc=entry, entry_quote_price=101, entry_price=101.25,
               trigger_source_time_msc=None, trigger_time_msc=horizon, trigger_reason='HORIZON',
               exit_time_msc=exit_, exit_quote_price=last_bid, exit_price=last_bid-.25,
               signed_move_after_spread_and_slippage=-1.5 if losing else 1.5,
               both_barriers_in_bundle=False, entry_bundle=[dict(time_msc=entry, bid=100, ask=101)],
               exit_bundle=[dict(time_msc=exit_, bid=last_bid, ask=last_bid+1)],
               R=-.75 if losing else .75, policy_id='SIDE_LEAD', baseline=False,
               public_price_scale_divisor=10**8, training_unit_quote=1,
               cash_pnl=None, cash_net=None, costs_verified=False, execution_verified=False,
               marked_quote_rows=4, conditional_equity_index_after_exit=9981.25 if losing else 10018.75)
    return row, points


def case_for(ledger):
    complete = [r for r in ledger if r['status'] == 'COMPLETE']
    unknown = [r for r in ledger if r['status'] == 'UNKNOWN']
    total = sum(r['R'] for r in complete)
    quote = sum(r['signed_move_after_spread_and_slippage'] for r in complete)
    gain = sum(max(0, r['R']) for r in complete)
    loss = sum(max(0, -r['R']) for r in complete)
    pf = gain/loss if loss else None
    censored = bool(unknown)
    summary = dict(completed=len(complete), unknown=len(unknown), skipped=0,
                   remaining_censored=censored, unknown_and_remaining_censored=censored,
                   primary_measurement_usable=not censored, observed_sum_R=total,
                   observed_sum_quote=quote, observed_gain_R=gain, observed_loss_R=loss,
                   observed_pf=pf, observed_completed_prefix_ending_equity_index=10000+25*total,
                   sum_R=None if censored else total, sum_quote=None if censored else quote,
                   pf=None if censored else pf, ending_equity_index=None if censored else 10000+25*total,
                   marked_drawdown_pct=None if censored else .1,
                   drawdown_peak_time_ms=None, drawdown_trough_time_ms=None,
                   censored_from_ms=unknown[0]['decision_ms'] if censored else None,
                   censored_until_exclusive_ms=a.WF_END-1800000 if censored else None,
                   remaining_opportunities_count=None if censored else 0,
                   cash_net=None, qualified_for_profit_claim=False,
                   median_trade_R_observed=(sum(sorted(r['R'] for r in complete)[(len(complete)-1)//2:len(complete)//2+1])/
                                            (2 if len(complete)%2 == 0 else 1)) if complete else None,
                   median_holding_ms_observed=5000 if complete else None)
    daily = []
    for day in range(a.WF_START, a.WF_END, a.DAY):
        rows = [r for r in complete if day <= r['exit_time_msc'] < day+a.DAY]
        daily.append(dict(day_start_ms=day, observed_sum_R=sum(r['R'] for r in rows),
                          observed_sum_quote=sum(r['signed_move_after_spread_and_slippage'] for r in rows),
                          observed_gain_R=sum(max(0, r['R']) for r in rows),
                          observed_loss_R=sum(max(0, -r['R']) for r in rows), completed=len(rows),
                          unknown=sum(day <= r['decision_ms'] < day+a.DAY for r in unknown),
                          sum_R=None if censored else sum(r['R'] for r in rows),
                          sum_quote=None if censored else sum(r['signed_move_after_spread_and_slippage'] for r in rows),
                          primary_measurement_censored=censored, cash_net=None,
                          safety={key: False for key in a.FLAGS}))
    return dict(U_scaled=10**8, policy='SIDE_LEAD', symbol='BOOM500', baseline=False,
                scenario='BASE', summary=summary, daily=daily)


def scan_for(tmp_path, ledger, points):
    relative = 'data/independent_profit_ticks_20261013/development/fake_ticks.tsv'
    pin = write(tmp_path, relative, raw_text(points))
    queries, needed, intervals = set(), set(), []
    for row in ledger:
        queries.update((row['decision_ms'], row['decision_ms']+250, row['horizon_end_ms']-28000))
        for key in ('entry_time_msc', 'exit_time_msc'):
            if row[key] is not None:
                needed.add(row[key]); queries.update((row[key], row[key]+1))
        if row['trigger_time_msc'] is not None:
            queries.add(row['trigger_time_msc']+250)
        if row['trigger_source_time_msc'] is not None:
            queries.add(row['trigger_source_time_msc']+1)
        if row['status'] == 'COMPLETE':
            intervals.append((row['decision_ms'], row['exit_time_msc']))
    shard = dict(path=relative, sha256=pin, start_ms=a.WF_START, end_ms=a.WF_START+a.DAY, rows=len(points))
    return a.scan_raw(tmp_path, [shard], needed, queries, intervals)


def test_valid_worst_side_arithmetic_daily_and_partial_path_audit(tmp_path):
    row, points = trade()
    result = a.audit_case(case_for([row]), [row], scan_for(tmp_path, [row], points), {0})
    assert result['errors'] == []
    assert result['complete_entry_exit_bundles_checked'] == 1
    assert result['sampled_complete_paths_checked'] == 1


@pytest.mark.parametrize('field,value,expected_error', [
    ('R', .76, 'R'), ('entry_price', 101.5, 'effective_entry'),
    ('entry_quote_price', 100, 'entry_worst_side'), ('exit_quote_price', 104, 'exit_worst_side'),
    ('exit_price', 102.5, 'effective_exit'), ('signed_move_after_spread_and_slippage', 2, 'signed_move'),
    ('direction', -1, 'direction'), ('stop_distance', 3, 'stop_distance'),
    ('target_distance', None, 'target_distance'), ('training_unit_quote', 2, 'training_U'),
])
def test_corrupted_trade_fields_are_detected(tmp_path, field, value, expected_error):
    row, points = trade(); case = case_for([row]); scan = scan_for(tmp_path, [row], points)
    row[field] = value
    errors = a.audit_case(case, [row], scan, {0})['errors']
    assert expected_error in {e['field'] for e in errors}


def test_corrupted_slippage_configuration_is_detected(tmp_path):
    row, points = trade(); case = case_for([row]); scan = scan_for(tmp_path, [row], points)
    row['config']['adverse_slippage_per_leg'] = 0
    assert 'slippage' in {e['field'] for e in a.audit_case(case, [row], scan, {0})['errors']}


def test_pf_and_daily_totals_cannot_be_tampered(tmp_path):
    first, points = trade(); second, tail = trade(X+1000, losing=True)
    second['conditional_equity_index_after_exit'] = 10000
    rows = [first, second]; case = case_for(rows); scan = scan_for(tmp_path, rows, points+tail)
    assert a.audit_case(case, rows, scan, {0, 1})['errors'] == []
    case['summary']['pf'] = 2
    case['daily'][0]['observed_sum_R'] = 9
    names = {e['field'] for e in a.audit_case(case, rows, scan, {0, 1})['errors']}
    assert {'summary_pf', 'daily_observed_sum_R'} <= names


def test_unknown_keeps_descriptive_prefix_and_null_primary(tmp_path):
    first, points = trade(); unknown = deepcopy(first)
    unknown.update(status='UNKNOWN', reason='MISSING_ENTRY_QUOTE', decision_ms=X+1000,
                   horizon_end_ms=X+6000, entry_time_msc=None, entry_price=None,
                   entry_quote_price=None, entry_bundle=[], trigger_time_msc=None,
                   exit_time_msc=None, exit_quote_price=None, exit_price=None, exit_bundle=[],
                   signed_move_after_spread_and_slippage=None, R=None)
    points.append((unknown['decision_ms'], 100, 101))
    rows = [first, unknown]; case = case_for(rows); scan = scan_for(tmp_path, rows, points)
    assert a.audit_case(case, rows, scan, {0})['errors'] == []
    case['summary']['sum_R'] = .75
    case['daily'][0]['sum_R'] = 0
    names = {e['field'] for e in a.audit_case(case, rows, scan, {0})['errors']}
    assert {'summary_sum_R', 'daily_primary_R'} <= names


def test_same_ms_bundle_multiplicity_and_order_are_checked(tmp_path):
    row, points = trade(); scan = scan_for(tmp_path, [row], points)
    row['entry_bundle'].append(deepcopy(row['entry_bundle'][0]))
    assert 'entry_bundle_multiplicity' in {e['field'] for e in a.audit_case(case_for([row]), [row], scan, {0})['errors']}


def test_sample_barrier_confirmation_and_stop_priority_are_independent(tmp_path):
    row, points = trade()
    points[2:3] = [(D+500, 98, 99), (D+500, 104, 105), (D+750, 102, 103)]
    row.update(trigger_source_time_msc=D+500, trigger_time_msc=D+750, trigger_reason='STOP',
               both_barriers_in_bundle=True, reason='STOP_PRIORITY_SAME_MILLISECOND',
               exit_time_msc=D+1000, exit_quote_price=103, exit_price=102.75,
               exit_bundle=[dict(time_msc=D+1000, bid=103, ask=104)], marked_quote_rows=5)
    points = [p for p in points if p[0] < D+1000]+[(D+1000, 103, 104)]
    case = case_for([row]); case['summary']['median_holding_ms_observed'] = 750
    scan = scan_for(tmp_path, [row], points)
    assert a.audit_case(case, [row], scan, {0})['errors'] == []
    row['trigger_time_msc'] = D+500
    errors = {e['field'] for e in a.audit_case(case, [row], scan, {0})['errors']}
    assert 'barrier_confirmation_not_backdated' in errors
    assert 'sample_first_barrier_horizon_stop_priority' in errors


def test_quantization_is_half_even_before_side_selection():
    assert a._scaled('1.000000005') == 100000000
    assert a._scaled('1.000000015') == 100000002
    with pytest.raises(a.AuditError):
        a._scaled('0.000000005')


@pytest.mark.parametrize('bad', [
    [(D, 102, 101)], [(D, 0, 1)], [(D, 'NaN', 101)],
    [(D+1, 100, 101), (D, 100, 101)], [(a.WF_END, 100, 101)],
])
def test_raw_crossed_nonfinite_order_and_bounds_rejected(tmp_path, bad):
    with pytest.raises(a.AuditError):
        scan_for(tmp_path, [], bad)


def test_source_hash_mismatch_is_fatal(tmp_path):
    relative = 'data/independent_profit_ticks_20261013/development/fake_ticks.tsv'
    write(tmp_path, relative, raw_text([(D, 100, 101)]))
    shard = dict(path=relative, sha256='0'*64, start_ms=a.WF_START, end_ms=a.WF_START+a.DAY, rows=1)
    with pytest.raises(a.AuditError, match='SHA256'):
        a.scan_raw(tmp_path, [shard], set(), set(), [])


def test_forbidden_final_private_symlink_and_duplicate_json_rejected(tmp_path):
    for name in ('data/sealed_final/source.tsv', 'work/private_history_availability_20261013/source.tsv'):
        write(tmp_path, name, 'x')
        with pytest.raises(a.AuditError):
            a._path(tmp_path, name)
    write(tmp_path, 'regular.txt', 'x'); (tmp_path/'linked.txt').symlink_to(tmp_path/'regular.txt')
    with pytest.raises(a.AuditError):
        a._path(tmp_path, 'linked.txt')
    with pytest.raises(a.AuditError):
        a._json('{"R":1,"R":2}')
    with pytest.raises(a.AuditError):
        a._json('{"R":NaN}')


def test_numbers_and_counts_cannot_be_replaced_with_strings_or_booleans(tmp_path):
    row, points = trade(); case = case_for([row]); scan = scan_for(tmp_path, [row], points)
    row['R'] = '.75'; case['summary']['completed'] = True
    names = {e['field'] for e in a.audit_case(case, [row], scan, {0})['errors']}
    assert {'R', 'summary_completed'} <= names


def test_sample_is_deterministic_bounded_and_spans_whole_ledger():
    rows = [dict(status='COMPLETE') for _ in range(1000)]
    result = a.sample_indices(rows)
    assert len(result) == 64 and min(result) == 0 and max(result) == 999
    assert result == a.sample_indices(rows)


def fake_repo(root):
    flags = {key: False for key in a.FLAGS}
    requests, source_pins = [], {}
    for symbol, (name, _) in a.SYMBOLS.items():
        pins = {}
        for day in range(30, 50):
            prefix = f'FAKE_{symbol}_{day}'
            start = a.START+day*a.DAY
            requests.append(dict(symbol=symbol, day_index=day, prefix=prefix, partition='development',
                                 start_ms=start, end_exclusive_ms=start+a.DAY))
            folder = a.DATA+'/development/'+prefix+'/'
            raw = folder+prefix+'_ticks.tsv'; receipt = folder+prefix+'_receipt.tsv'
            raw_pin = write(root, raw, raw_text([(start, 100, 101)]))
            metadata = {'symbol': name, 'raw_file': prefix+'_ticks.tsv',
                        'terminal_trade_allowed_observed': 'false', 'mql_trade_allowed_observed': 'false',
                        'account_identity_captured': 'false', **{k: 'false' for k in
                        ('live_trading', 'ready_for_live', 'live_allowed', 'opened_trades',
                         'safety_upper_live_trading', 'safety_upper_ready_for_live',
                         'safety_upper_live_allowed', 'safety_upper_opened_trades')}}
            receipt_pin = write(root, receipt, 'field\tvalue\n'+''.join(f'{k}\t{v}\n' for k,v in metadata.items()))
            verified = dict(status='QUOTES_VALIDATED_COVERAGE_NOT_PROVEN', errors=[], raw_sha256=raw_pin,
                            receipt_sha256=receipt_pin, symbol=name, requested_start_msc=start,
                            requested_end_exclusive_msc=start+a.DAY, rows=1, cost_verified=False,
                            execution_verified=False, full_tick_coverage_verified=False, profit_tested=False)
            verification = folder+'independent_verification.json'
            pins.update({raw: raw_pin, receipt: receipt_pin, verification: write(root, verification, encoded(verified))})
        source_pins[symbol] = pins
    acquisition_pin = write(root, a.DATA+'/acquisition_declaration.json', encoded(dict(requests=requests, **flags)))
    protocol_pin = write(root, 'docs/frozen.md', 'fake frozen protocol')
    validation_pin = write(root, 'docs/validation.json', '{}')
    code_pin = write(root, 'independent_profit_research/sentinel.py', 'raise RuntimeError("NEVER IMPORT")\n')
    declaration = dict(source_pins={'independent_profit_research/sentinel.py': code_pin},
                       acquisition_declaration_sha256=acquisition_pin, protocol_path='docs/frozen.md',
                       protocol_sha256=protocol_pin, premeasurement_validation_path='docs/validation.json',
                       premeasurement_validation_sha256=validation_pin, **flags)
    declaration_pin = write(root, a.STUDY+'/declaration.json', encoded(declaration))
    index = {'declaration.json': declaration_pin}
    for symbol, (_, native) in a.SYMBOLS.items():
        for policy in a.POLICIES:
            for scenario in ('BASE', 'STRESS'):
                for baseline in (False, True):
                    stem = f'{symbol}_WF1_{policy}_{scenario}'+('_CLOCK' if baseline else '')
                    ledger_relative = a.STUDY+'/ledgers/'+stem+'.json.gz'
                    ledger_pin = write(root, ledger_relative, gzip.compress(b'[]', mtime=0))
                    case = case_for([])
                    case.update(symbol=symbol, policy=policy, policy_id=policy, scenario=scenario,
                                baseline=baseline, native_direction=native, price_scale_divisor=10**8,
                                declaration_sha256=declaration_pin, stage='WF1', stage_start_ms=a.WF_START,
                                stage_end_exclusive_ms=a.WF_END, eligible_decision_start_ms=a.WF_START+1800000,
                                eligible_decision_end_exclusive_ms=a.WF_END-1800000,
                                source_files=source_pins[symbol], safety=flags, cost_verified=False,
                                costs_verified=False, execution_verified=False, profit_qualified=False,
                                cash_net=None, ledger_path=ledger_relative, ledger_sha256=ledger_pin)
                    index['cases/'+stem+'.json'] = write(root, a.STUDY+'/cases/'+stem+'.json', encoded(case))
    write(root, a.STUDY+'/artifact_index.json', encoded(index))
    return declaration_pin


def test_whole_cli_strict_pins_24_cases_scope_and_exclusive_output(tmp_path, capsys):
    pin = fake_repo(tmp_path)
    args = ['--root', str(tmp_path), '--expected-declaration-sha256', pin]
    assert a.main(args) == 0
    assert capsys.readouterr().out == 'AUDIT_PASS\n'
    output = tmp_path/a.OUTPUT/'WF1.json'
    report = json.loads(output.read_text())
    assert len(report['cases']) == 24
    assert report['full_barrier_path_reproduction'] is False
    assert report['signal_issuance_features_reproduced'] is False
    assert report['full_marked_drawdown_reproduced'] is False
    assert output.stat().st_mode & 0o777 == 0o600
    assert a.main(args) == 2
    assert capsys.readouterr().out == 'AUDIT_FAILED\n'


def test_whole_audit_rejects_changed_source_and_wrong_decl_pin(tmp_path):
    pin = fake_repo(tmp_path)
    with pytest.raises(a.AuditError, match='SHA256'):
        a.audit(tmp_path, '0'*64)
    file = tmp_path/a.DATA/'development/FAKE_BOOM500_30/FAKE_BOOM500_30_ticks.tsv'
    file.write_text(file.read_text().replace('100\t101', '100\t102'))
    with pytest.raises(a.AuditError, match='SHA256'):
        a.audit(tmp_path, pin)


def test_cli_rejects_output_outside_fixed_audit_directory(tmp_path, capsys):
    assert a.main(['--root', str(tmp_path), '--expected-declaration-sha256', '0'*64,
                   '--output', str(tmp_path/'docs/public.json')]) == 2
    assert capsys.readouterr().out == 'AUDIT_FAILED\n'
    assert not (tmp_path/'docs/public.json').exists()


def test_cli_parent_traversal_cannot_escape_with_valid_inputs(tmp_path, capsys):
    pin = fake_repo(tmp_path)
    output = tmp_path/a.OUTPUT/'../../docs/escaped.json'
    assert a.main(['--root', str(tmp_path), '--expected-declaration-sha256', pin,
                   '--output', str(output)]) == 2
    assert capsys.readouterr().out == 'AUDIT_FAILED\n'
    assert not (tmp_path/'docs/escaped.json').exists()
