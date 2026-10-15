"""Offline fixtures for the independent retained-MQL5-TSV checker."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'verify_mt5_quote_export.py'
spec = importlib.util.spec_from_file_location('verify_mt5_quote_export', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
START, END = 1761955200000, 1761955260000
NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)


def write_tsv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\r\n')
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def evidence(tmp_path):
    prefix = 'OfflineFixture'
    rows = [
        [str(START//1000), str(START), '100', '100.2', '0', '0', '6', '0'],
        [str(START//1000), str(START), '100.1', '100.3', '0', '0', '6', '0'],
        [str((START+2000)//1000), str(START+2000), '101', '101.2', '0', '4', '14', '4'],
    ]
    meta = dict.fromkeys(checker.FALSE_FIELDS, 'false')
    meta.update({
        'schema_version': '1', 'export_status': 'SAVED_NOT_PROVEN', 'failure_reason': '',
        'symbol': 'Boom 500 Index', 'server': 'DerivSVG-Server', 'company': 'Deriv (SVG) LLC',
        'account_mode': 'LIVE', 'symbol_custom_observed': 'false',
        'capture_time_gmt': '2026.10.12 21:47:29 UTC',
        'request_from_gmt': '2025.11.01 00:00:00 UTC',
        'request_end_exclusive_gmt': '2025.11.01 00:01:00 UTC',
        'request_from_msc': str(START), 'request_end_exclusive_msc': str(END),
        'copy_to_inclusive_msc': str(END-1), 'window_end_exclusive': 'true',
        'max_window_seconds': '86400', 'copy_mode': 'COPY_TICKS_ALL',
        'raw_file': prefix+'_ticks.tsv', 'raw_partial_file': '',
        'row_count': '3', 'raw_rows_written': '3',
        'first_time_msc': str(START), 'last_time_msc': str(START+2000),
        'last_error': '0', 'guard_error': '0', 'write_error': '0',
        'attempts': '1', 'retries': '0', 'selected_attempt': '1',
        'raw_retention_scope': 'SELECTED_ATTEMPT_ONLY', 'copy_returned_count': '3',
        'api_returned_without_error': 'true', 'connection_wait_limit_seconds': '30',
        'connection_wait_elapsed_ms': '100', 'terminal_trade_allowed_observed': 'false',
        'mql_trade_allowed_observed': 'false', 'out_of_window_rows': '0',
        'backward_time_rows': '0', 'crossed_quote_rows': '0', 'invalid_quote_rows': '0',
        'same_millisecond_rows_preserved': 'true', 'row_order_preserved': 'true',
        'synchronization_status': 'NOT_PROVEN', 'historical_tick_completeness': 'NOT_PROVEN',
        'limitation': 'Current symbol specifications and retained quotes do not prove historical costs or fills.',
        'attempt_1_returned_count': '3', 'attempt_1_array_rows': '3',
        'attempt_1_last_error': '0', 'attempt_1_elapsed_ms': '10',
        'spec_snapshot_status': 'CURRENT_ONLY_NOT_HISTORICAL',
        'spec_snapshot_capture_time_gmt': '2026.10.12 21:47:29 UTC',
    })
    values = ('0.001', '3', '1', '0.001', '0.001', '0.001', '0.001', '0.2',
              '190', '0.01', '5', '-23', '-23', '7', '4')
    for key, value in zip(checker.SPEC_FIELDS, values):
        meta[key], meta[key+'_error_code'] = value, '0'
    receipt = tmp_path/(prefix+'_receipt.tsv')
    raw = tmp_path/(prefix+'_ticks.tsv')

    def save():
        write_tsv(receipt, ['field', 'value'], meta.items())
        write_tsv(raw, checker.FIELDS, rows)
        return receipt

    save()
    return meta, rows, receipt, raw, save


def verify(evidence):
    return checker.verify_export(evidence[4](), now=NOW)


def test_valid_same_millisecond_rows_are_retained(evidence):
    before = evidence[3].read_bytes()
    result = verify(evidence)
    assert result['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
    assert result['rows'] == 3 and result['same_millisecond_adjacent_pairs'] == 1
    assert result['flag_counts'] == {'6': 2, '14': 1}
    assert result['spread_price_median'] == '0.2'
    assert result['raw_sha256'] == hashlib.sha256(before).hexdigest()
    assert evidence[3].read_bytes() == before
    assert not any(result[key] for key in ('cost_verified', 'execution_verified', 'profit_tested',
                                           'full_tick_coverage_verified', 'broker_origin_independently_authenticated',
                                           'spec_historical', 'same_account_identity_verified',
                                           'all_attempt_raw_arrays_retained'))
    assert result['spec_scope'] == 'CURRENT_ONLY_NOT_HISTORICAL'
    assert result['retention_scope'] == 'SELECTED_ATTEMPT_ONLY'


@pytest.mark.parametrize('field', checker.FALSE_FIELDS)
@pytest.mark.parametrize('value', ['true', '', 'False', '0'])
def test_every_safety_and_evidence_false_is_explicit(evidence, field, value):
    evidence[0][field] = value
    with pytest.raises(ValueError, match='explicitly false'):
        verify(evidence)


@pytest.mark.parametrize('field,value,error', [
    ('bid', '0', 'INVALID_QUOTE_ROWS'), ('ask', '-1', 'INVALID_QUOTE_ROWS'),
    ('ask', '99', 'CROSSED_QUOTE_ROWS'), ('last', '-1', 'INVALID_QUOTE_ROWS'),
    ('volume_real', '-1', 'INVALID_QUOTE_ROWS'), ('bid', 'nan', 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('ask', 'Infinity', 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('volume_real', '1e9999', 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('bid', ' 100', 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('bid', '1_00', 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('volume', '-1', 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('flags', str(2**32), 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('volume', str(2**64), 'NONNUMERIC_OR_NONFINITE_RAW_ROWS'),
    ('time', str(START//1000+1), 'INVALID_OR_INCONSISTENT_INTEGER_FIELDS'),
    ('time_msc', str(END), 'OUT_OF_WINDOW_ROWS'),
    ('time_msc', str(START-1), 'OUT_OF_WINDOW_ROWS'),
])
def test_invalid_rows_fail_without_summarizing_subset(evidence, field, value, error):
    evidence[1][1][checker.FIELDS.index(field)] = value
    result = verify(evidence)
    assert result['status'] == 'FAIL_PRESERVED'
    assert error in result['errors']
    assert result['rows'] == 3 and result['raw_sha256']
    assert result['spread_price_min'] is None and result['flag_counts'] == {}


def test_backward_rows_preserved_and_rejected(evidence):
    evidence[1][0], evidence[1][2] = evidence[1][2], evidence[1][0]
    evidence[0].update(first_time_msc=str(START+2000), last_time_msc=str(START), backward_time_rows='1')
    result = verify(evidence)
    assert 'BACKWARD_TIME_ROWS' in result['errors']
    assert 'RECEIPT_BACKWARD_TIME_ROWS_MISMATCH' not in result['errors']


@pytest.mark.parametrize('field,value,error', [
    ('server', '', 'UNEXPECTED_SERVER'), ('server', 'Derivatives-Server', 'UNEXPECTED_SERVER'),
    ('server', 'DerivNotRecognized', 'UNEXPECTED_SERVER'), ('company', 'Other Broker', 'UNEXPECTED_COMPANY'),
    ('symbol', 'EURUSD', 'UNSUPPORTED_BOOM_CRASH_SYMBOL'),
    ('symbol_custom_observed', 'true', 'CUSTOM_SYMBOL_NOT_EXCLUDED'),
    ('symbol_custom_observed', 'NOT_AVAILABLE', 'CUSTOM_SYMBOL_NOT_EXCLUDED'),
    ('terminal_trade_allowed_observed', 'true', 'TRADING_PROTECTION_NOT_VERIFIED'),
    ('mql_trade_allowed_observed', 'true', 'TRADING_PROTECTION_NOT_VERIFIED'),
    ('guard_error', '4001', 'TRADING_PROTECTION_NOT_VERIFIED'),
    ('account_mode', 'CONTEST', 'UNKNOWN_ACCOUNT_MODE'),
    ('export_status', 'PARTIAL', 'PARTIAL_OR_FAILED_API_RETURN'),
    ('failure_reason', 'partial', 'PARTIAL_OR_FAILED_API_RETURN'),
    ('api_returned_without_error', 'false', 'PARTIAL_OR_FAILED_API_RETURN'),
    ('write_error', '5001', 'PARTIAL_OR_FAILED_API_RETURN'),
    ('historical_tick_completeness', 'VERIFIED', 'UNSUPPORTED_COMPLETENESS_CLAIM'),
    ('synchronization_status', 'VERIFIED', 'UNSUPPORTED_COMPLETENESS_CLAIM'),
    ('raw_retention_scope', 'ALL_ATTEMPTS', 'UNSUPPORTED_RAW_RETENTION_CLAIM'),
    ('same_millisecond_rows_preserved', 'false', 'UNSUPPORTED_RAW_RETENTION_CLAIM'),
    ('row_order_preserved', 'false', 'UNSUPPORTED_RAW_RETENTION_CLAIM'),
    ('window_end_exclusive', 'false', 'REQUEST_BOUNDARY_MISMATCH'),
    ('copy_to_inclusive_msc', str(END), 'REQUEST_BOUNDARY_MISMATCH'),
    ('copy_mode', 'COPY_TICKS_INFO', 'UNEXPECTED_COPY_CONTRACT'),
    ('capture_time_gmt', '2030.01.01 00:00:00 UTC', 'FUTURE_REQUEST_OR_CAPTURE_TIME'),
    ('spec_snapshot_capture_time_gmt', '2026.10.13 00:00:00 UTC', 'SPEC_CAPTURE_TIME_AFTER_RECEIPT'),
])
def test_claim_and_context_guards(evidence, field, value, error):
    evidence[0][field] = value
    result = verify(evidence)
    assert result['status'] == 'FAIL_PRESERVED' and error in result['errors']


@pytest.mark.parametrize('symbol', ['Boom 500 Index', 'Crash 500 Index', 'Boom 300 Index', 'Crash 1000 Index'])
@pytest.mark.parametrize('mode', ['LIVE', 'DEMO'])
def test_existing_live_or_demo_quote_context_allowed(evidence, symbol, mode):
    evidence[0].update(symbol=symbol, account_mode=mode)
    assert verify(evidence)['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'


def test_unknown_or_missing_metadata_rejected(evidence):
    evidence[0]['account_login'] = 'secret'
    with pytest.raises(ValueError, match='schema'):
        verify(evidence)
    del evidence[0]['account_login'], evidence[0]['symbol_custom_observed']
    with pytest.raises(ValueError, match='schema'):
        verify(evidence)


def test_duplicate_metadata_rejected(evidence):
    with evidence[2].open('a') as stream: stream.write('symbol\tBoom 500 Index\n')
    with pytest.raises(ValueError, match='Duplicate'):
        checker.verify_export(evidence[2], now=NOW)


@pytest.mark.parametrize('content', ['wrong\theader\n', 'field\tvalue\nonly_one\n',
                                      'field\tvalue\nfield\tvalue\textra\n',
                                      'field\tvalue\nfield\t"unterminated\n',
                                      'field\tvalue\nfield\t"line\nbreak"\n',
                                      'field\tvalue\nfield\tbad\x00value\n'])
def test_malformed_receipt_tsv(evidence, content):
    evidence[2].write_text(content)
    with pytest.raises((ValueError, csv.Error)):
        checker.verify_export(evidence[2], now=NOW)


@pytest.mark.parametrize('name', ['../outside.tsv', '/tmp/outside.tsv', 'OfflineFixture_ticks.tsv/evil',
                                  'Other_ticks.tsv', 'OfflineFixture_ticks.tsv\\evil'])
def test_path_escape_or_nonmatching_name_rejected(evidence, name):
    evidence[0]['raw_file'] = name
    with pytest.raises(ValueError, match='basename'):
        verify(evidence)


def test_receipt_symlink_rejected(evidence, tmp_path):
    link = tmp_path/'Link_receipt.tsv'
    link.symlink_to(evidence[2])
    with pytest.raises(ValueError, match='symlink'):
        checker.verify_export(link, now=NOW)


def test_raw_symlink_rejected(evidence, tmp_path):
    stored = tmp_path/'stored.tsv'
    evidence[3].rename(stored)
    evidence[3].symlink_to(stored)
    with pytest.raises(ValueError, match='symlink'):
        checker.verify_export(evidence[2], now=NOW)


def test_symlink_ancestor_rejected(evidence, tmp_path):
    folder = tmp_path/'linked'
    folder.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        checker.verify_export(folder/evidence[2].name, now=NOW)


def set_attempts(meta, attempts):
    for key in list(meta):
        if key.startswith('attempt_'): del meta[key]
    meta.update(attempts=str(len(attempts)), retries=str(max(len(attempts)-1, 0)))
    for i, (returned, count, error) in enumerate(attempts, 1):
        meta.update({f'attempt_{i}_returned_count': str(returned), f'attempt_{i}_array_rows': str(count),
                     f'attempt_{i}_last_error': str(error), f'attempt_{i}_elapsed_ms': '10'})
    meta['copy_returned_count'], meta['last_error'] = (str(v) for v in attempts[-1][::2]) if attempts else ('-1', '0')


def test_partial_api_retains_largest_earlier_array(evidence):
    meta = evidence[0]
    set_attempts(meta, [(3, 3, 4403), (2, 2, 4403), (-1, 0, 4403)])
    meta.update(export_status='PARTIAL', api_returned_without_error='false', failure_reason='timeout')
    result = verify(evidence)
    assert result['selected_attempt'] == 1 and result['rows'] == 3
    assert 'SELECTED_ATTEMPT_RETENTION_MISMATCH' not in result['errors']
    assert 'PARTIAL_OR_FAILED_API_RETURN' in result['errors']
    assert result['attempts'][2]['returned_count'] == -1


def test_clean_later_array_is_selected_even_if_smaller(evidence):
    set_attempts(evidence[0], [(5, 5, 4403), (3, 3, 0)])
    evidence[0]['selected_attempt'] = '2'
    result = verify(evidence)
    assert result['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
    assert result['all_attempt_raw_arrays_retained'] is False


@pytest.mark.parametrize('field,value,error', [
    ('selected_attempt', '0', 'SELECTED_ATTEMPT_RETENTION_MISMATCH'),
    ('copy_returned_count', '2', 'LAST_ATTEMPT_RETURN_COUNT_MISMATCH'),
    ('last_error', '4403', 'LAST_ATTEMPT_ERROR_MISMATCH'),
    ('retries', '1', 'ATTEMPT_RETRY_MISMATCH'),
    ('row_count', '2', 'RECEIPT_ROW_COUNT_MISMATCH'),
    ('raw_rows_written', '2', 'RECEIPT_ROW_COUNT_MISMATCH'),
    ('out_of_window_rows', '1', 'RECEIPT_OUT_OF_WINDOW_ROWS_MISMATCH'),
])
def test_attempt_and_count_mismatch(evidence, field, value, error):
    evidence[0][field] = value
    assert error in verify(evidence)['errors']


def test_attempt_after_clean_response_rejected(evidence):
    set_attempts(evidence[0], [(3, 3, 0), (3, 3, 0)])
    evidence[0]['selected_attempt'] = '2'
    assert 'ATTEMPTS_CONTINUED_AFTER_CLEAN_RETURN' in verify(evidence)['errors']


def test_empty_data_is_not_success(evidence, tmp_path, capsys):
    evidence[1].clear()
    evidence[0].update(raw_file='', row_count='0', raw_rows_written='0', first_time_msc='NONE',
                       last_time_msc='NONE', selected_attempt='0', export_status='EMPTY',
                       failure_reason='no rows', api_returned_without_error='false')
    set_attempts(evidence[0], [(0, 0, 0), (0, 0, 0), (0, 0, 0)])
    result = verify(evidence)
    assert result['status'] == 'EMPTY_OR_FAILED_NOT_VERIFIED' and result['rows'] == 0
    output = tmp_path/'empty.json'
    assert checker.main(['--receipt', str(evidence[2]), '--output', str(output)]) == 2
    assert json.loads(output.read_text())['full_tick_coverage_verified'] is False


def test_partial_staging_is_preserved_but_rejected(evidence):
    evidence[4]()
    partial = evidence[3].with_name('OfflineFixture_ticks.part.tsv')
    evidence[3].rename(partial)
    evidence[0].update(raw_file='', raw_partial_file=partial.name)
    write_tsv(evidence[2], ['field', 'value'], evidence[0].items())
    result = checker.verify_export(evidence[2], now=NOW)
    assert 'ONLY_PARTIAL_RAW_FILE_AVAILABLE' in result['errors'] and result['raw_sha256']


def test_current_spec_unavailable_is_disclosed_not_historical(evidence):
    evidence[0].update(symbol_swap_long='NOT_AVAILABLE', symbol_swap_long_error_code='4303')
    result = verify(evidence)
    assert result['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
    assert result['current_spec_unavailable_fields'] == ['symbol_swap_long']
    assert result['spec_historical'] is False and result['cost_verified'] is False


@pytest.mark.parametrize('value', ['nan', '1e9999', '-1', '0.1_0'])
def test_malformed_spec_values_not_accepted(evidence, value):
    evidence[0]['symbol_point'] = value
    assert 'INVALID_CURRENT_SPEC_NUMERIC_VALUE' in verify(evidence)['errors']


def test_spec_api_error_not_claimed_available(evidence):
    evidence[0]['symbol_point_error_code'] = '4303'
    assert 'SPEC_VALUE_WITH_API_ERROR' in verify(evidence)['errors']


def test_no_guarded_current_spec_does_not_validate_quotes(evidence):
    evidence[0].update(spec_snapshot_status='UNSTABLE_NOT_CAPTURED', spec_snapshot_capture_time_gmt='NOT_CAPTURED')
    for key in checker.SPEC_FIELDS:
        del evidence[0][key], evidence[0][key+'_error_code']
    assert 'QUOTE_CONTEXT_WITHOUT_GUARDED_CURRENT_SPEC' in verify(evidence)['errors']


@pytest.mark.parametrize('field,value', [('attempts', '4'), ('attempts', '1.0'), ('row_count', 'true'),
                                       ('row_count', '+3'), ('row_count', '03'), ('row_count', '-1'),
                                       ('connection_wait_limit_seconds', '31'), ('guard_error', '-1')])
def test_strict_integer_metadata(evidence, field, value):
    evidence[0][field] = value
    with pytest.raises(ValueError): verify(evidence)


def test_input_tamper_between_read_and_publish_rejected(evidence, monkeypatch):
    original = checker.read_tsv
    def read_then_tamper(path, fields):
        result = original(path, fields)
        if fields == checker.FIELDS:
            with Path(path).open('a') as stream: stream.write('\n')
        return result
    monkeypatch.setattr(checker, 'read_tsv', read_then_tamper)
    with pytest.raises(ValueError, match='changed'):
        checker.verify_export(evidence[2], now=NOW)


def test_cli_no_overwrite_or_symlink_output(evidence, tmp_path, capsys):
    output = tmp_path/'output.json'
    output.write_text('existing')
    assert checker.main(['--receipt', str(evidence[2]), '--output', str(output)]) == 2
    assert output.read_text() == 'existing'
    link = tmp_path/'link.json'
    link.symlink_to(output)
    assert checker.main(['--receipt', str(evidence[2]), '--output', str(link)]) == 2
    assert output.read_text() == 'existing'


def test_cli_valid_success_and_sanitized_failure(evidence, tmp_path, capsys):
    output = tmp_path/'valid.json'
    assert checker.main(['--receipt', str(evidence[2]), '--output', str(output)]) == 0
    assert json.loads(output.read_text())['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
    evidence[0]['account_password'] = 'DO_NOT_ECHO'
    evidence[4]()
    bad = tmp_path/'bad.json'
    assert checker.main(['--receipt', str(evidence[2]), '--output', str(bad)]) == 2
    assert not bad.exists() and 'DO_NOT_ECHO' not in capsys.readouterr().err
