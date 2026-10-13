"""Independently inspect retained MQL5 quote TSVs; never run a trading API."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

FIELDS = ['time', 'time_msc', 'bid', 'ask', 'last', 'volume', 'flags', 'volume_real']
FALSE_FIELDS = ('live_trading', 'ready_for_live', 'live_allowed', 'opened_trades',
                'safety_upper_live_trading', 'safety_upper_ready_for_live',
                'safety_upper_live_allowed', 'safety_upper_opened_trades',
                'costs_verified', 'execution_verified', 'historical_costs_verified',
                'cfd_fills_verified', 'cash_contract_mapping_verified',
                'quote_export_is_fill_evidence', 'account_identity_captured',
                'all_attempt_raw_arrays_retained', 'deduplication_applied', 'spec_snapshot_historical')
BASE_FIELDS = set(FALSE_FIELDS) | set('''schema_version export_status failure_reason symbol server company account_mode
symbol_custom_observed capture_time_gmt request_from_gmt request_end_exclusive_gmt request_from_msc
request_end_exclusive_msc copy_to_inclusive_msc window_end_exclusive max_window_seconds copy_mode
raw_file raw_partial_file row_count raw_rows_written first_time_msc last_time_msc last_error guard_error
write_error attempts retries selected_attempt raw_retention_scope copy_returned_count api_returned_without_error
connection_wait_limit_seconds connection_wait_elapsed_ms terminal_trade_allowed_observed mql_trade_allowed_observed
out_of_window_rows backward_time_rows crossed_quote_rows invalid_quote_rows same_millisecond_rows_preserved
row_order_preserved synchronization_status historical_tick_completeness limitation spec_snapshot_status
spec_snapshot_capture_time_gmt'''.split())
SPEC_FIELDS = ("symbol_point", "symbol_digits", "symbol_trade_contract_size", "symbol_trade_tick_size",
               "symbol_trade_tick_value", "symbol_trade_tick_value_profit", "symbol_trade_tick_value_loss",
               "symbol_volume_min", "symbol_volume_max", "symbol_volume_step", "symbol_swap_mode",
               "symbol_swap_long", "symbol_swap_short", "symbol_swap_rollover3days", "symbol_trade_calc_mode")
STATUSES = {'SAVED_NOT_PROVEN', 'PARTIAL', 'EMPTY', 'FAILED_UNSAFE', 'FAILED_DISCONNECTED',
            'FAILED_ACCOUNT_CONTEXT', 'FAILED_INPUT', 'FAILED_SYMBOL', 'FAILED_FILE_WRITE', 'FAILED_NO_OVERWRITE'}
QUOTE_FIELDS = ('bid', 'ask', 'last', 'volume_real')
MAX_DOUBLE = Decimal('1.7976931348623157e308')


def real(value):
    # Bound both parsing and arithmetic to finite values representable by MQL doubles.
    if len(value) > 128 or not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,4})?', value):
        raise ValueError('Invalid real field')
    number = Decimal(value)
    if not number.is_finite() or abs(number) > MAX_DOUBLE:
        raise ValueError('Nonfinite or out-of-range real field')
    return number


def integer(value, label, minimum=None, maximum=None):
    if not isinstance(value, str) or len(value) > 21 or not re.fullmatch(r'-?(?:0|[1-9][0-9]*)', value):
        raise ValueError('Invalid integer field: ' + label)
    result = int(value)
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise ValueError('Out-of-range integer field: ' + label)
    return result


def gmt(value):
    if not re.fullmatch(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2} UTC', value):
        raise ValueError('Malformed GMT date')
    return datetime.strptime(value, '%Y.%m.%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in [path, *path.parents]) or not path.is_file():
        raise ValueError('Expected a regular retained TSV without symlink ancestors')
    return path


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read_tsv(path, fields):
    path = regular(path)
    if path.stat().st_size > (1048576 if fields == ['field', 'value'] else 256*1048576):
        raise ValueError('Unbounded TSV file size')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t', strict=True)
        if next(reader, None) != fields: raise ValueError('Unexpected TSV header')
        result = []
        for values in reader:
            if len(values) != len(fields) or any(any(ord(c)<32 or ord(c)==127 for c in value) for value in values):
                raise ValueError('Malformed TSV row')
            result.append(dict(zip(fields, values)))
            if len(result) > (200 if fields == ['field', 'value'] else 1000000):
                raise ValueError('Unbounded TSV row count')
    return result


def verify_export(receipt_path, *, now=None):
    receipt_path = regular(receipt_path)
    match = re.fullmatch(r'([A-Za-z0-9_-]{1,64})_receipt\.tsv', receipt_path.name)
    if not match or re.fullmatch(r'(?:con|prn|aux|nul|com[1-9]|lpt[1-9])', match[1], re.IGNORECASE):
        raise ValueError('Invalid MQL5 receipt prefix')
    prefix = match[1]
    receipt_hash, tool_hash = sha(receipt_path), sha(Path(__file__))
    pairs = read_tsv(receipt_path, ['field', 'value'])
    meta = {r['field']: r['value'] for r in pairs}
    if len(meta) != len(pairs) or any(not re.fullmatch(r'[a-z][a-z0-9_]*', key) for key in meta):
        raise ValueError('Duplicate receipt field')
    if meta.get('schema_version') != '1' or not BASE_FIELDS <= set(meta):
        raise ValueError('Unknown receipt schema')
    if any(meta.get(key) != 'false' for key in FALSE_FIELDS):
        raise ValueError('Safety or evidence claim is not explicitly false')
    attempts = integer(meta['attempts'], 'attempts', 0, 3)
    expected = BASE_FIELDS | {f'attempt_{i}_{suffix}' for i in range(1, attempts+1)
                              for suffix in ('returned_count', 'array_rows', 'last_error', 'elapsed_ms')}
    if meta['spec_snapshot_status'] == 'CURRENT_ONLY_NOT_HISTORICAL':
        expected |= set(SPEC_FIELDS) | {key+'_error_code' for key in SPEC_FIELDS}
    elif meta['spec_snapshot_status'] not in {'NOT_CAPTURED', 'UNSTABLE_NOT_CAPTURED', 'NOT_CAPTURED_BUFFER_FAILURE'}:
        raise ValueError('Unsupported specification scope')
    if set(meta) != expected or meta['export_status'] not in STATUSES:
        raise ValueError('Unexpected or missing schema/status/attempt/specification fields')
    for key in ('api_returned_without_error', 'terminal_trade_allowed_observed', 'mql_trade_allowed_observed'):
        if meta[key] not in {'true', 'false'}: raise ValueError('Invalid boolean field')
    count = integer(meta['row_count'], 'row_count', 0, 1000000)
    if meta['raw_file'] not in {'', prefix+'_ticks.tsv'} or meta['raw_partial_file'] not in {'', prefix+'_ticks.part.tsv'}:
        raise ValueError('Raw file must be an adjacent matching safe basename')
    if meta['raw_file'] and meta['raw_partial_file']: raise ValueError('Two raw destinations declared')
    raw_name = meta['raw_file'] or meta['raw_partial_file']
    raw_hash = sha(regular(receipt_path.parent/raw_name)) if raw_name else None
    rows = read_tsv(receipt_path.parent / raw_name, FIELDS) if raw_name else []
    errors = []
    if len(rows) != count or integer(meta['raw_rows_written'], 'raw_rows_written', 0, 1000000) != len(rows):
        errors.append('RECEIPT_ROW_COUNT_MISMATCH')
    if bool(raw_name) != bool(rows): errors.append('RAW_FILE_DECLARATION_MISMATCH')
    if meta['raw_partial_file']: errors.append('ONLY_PARTIAL_RAW_FILE_AVAILABLE')
    start, end = (integer(meta[key], key, 0, 2**63-1) for key in ('request_from_msc', 'request_end_exclusive_msc'))
    if not start > 0 or not 0 < end - start <= 86400000:
        errors.append('INVALID_REQUESTED_WINDOW')
    if meta['window_end_exclusive'] != 'true' or meta['copy_to_inclusive_msc'] != str(end - 1):
        errors.append('REQUEST_BOUNDARY_MISMATCH')
    if meta['max_window_seconds'] != '86400' or meta['copy_mode'] != 'COPY_TICKS_ALL': errors.append('UNEXPECTED_COPY_CONTRACT')
    capture, requested_start, requested_end = (gmt(meta[key]) for key in ('capture_time_gmt', 'request_from_gmt', 'request_end_exclusive_gmt'))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    ms = lambda d: ((d-epoch).days*86400+(d-epoch).seconds)*1000
    if ms(requested_start) != start or ms(requested_end) != end: errors.append('REQUEST_DATE_MISMATCH')
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None or now.utcoffset().total_seconds() != 0: raise ValueError('UTC-aware verifier clock required')
    if not requested_end <= capture <= now: errors.append('FUTURE_REQUEST_OR_CAPTURE_TIME')
    if (meta['raw_retention_scope'] != 'SELECTED_ATTEMPT_ONLY' or meta['same_millisecond_rows_preserved'] != 'true'
            or meta['row_order_preserved'] != 'true'): errors.append('UNSUPPORTED_RAW_RETENTION_CLAIM')
    if integer(meta['retries'], 'retries', 0, 2) != max(0, attempts-1): errors.append('ATTEMPT_RETRY_MISMATCH')
    selected = integer(meta['selected_attempt'], 'selected_attempt', 0, attempts)
    retained_count = retained_attempt = 0
    attempt_records = []
    for i in range(1, attempts+1):
        record = {suffix: integer(meta[f'attempt_{i}_{suffix}'], suffix, -1 if suffix=='returned_count' else 0,
                                  1000000 if suffix in {'returned_count','array_rows'} else 2**64-1)
                  for suffix in ('returned_count', 'array_rows', 'last_error', 'elapsed_ms')}
        clean = record['returned_count']>0 and record['last_error']==0 and record['returned_count']==record['array_rows']
        if record['array_rows'] > retained_count or clean: retained_count, retained_attempt = record['array_rows'], i
        if clean and i != attempts: errors.append('ATTEMPTS_CONTINUED_AFTER_CLEAN_RETURN')
        attempt_records.append(record)
    if retained_count != count or retained_attempt != selected: errors.append('SELECTED_ATTEMPT_RETENTION_MISMATCH')
    if integer(meta['copy_returned_count'], 'copy_returned_count', -1, 1000000) != (attempt_records[-1]['returned_count'] if attempts else -1):
        errors.append('LAST_ATTEMPT_RETURN_COUNT_MISMATCH')
    if integer(meta['last_error'], 'last_error', 0, 2**31-1) != (attempt_records[-1]['last_error'] if attempts else 0):
        errors.append('LAST_ATTEMPT_ERROR_MISMATCH')
    integer(meta['connection_wait_limit_seconds'], 'connection_wait_limit_seconds', 0, 30)
    integer(meta['connection_wait_elapsed_ms'], 'connection_wait_elapsed_ms', 0, 2**64-1)
    unavailable_spec = []
    if meta['spec_snapshot_status'] == 'CURRENT_ONLY_NOT_HISTORICAL':
        if gmt(meta['spec_snapshot_capture_time_gmt']) > capture: errors.append('SPEC_CAPTURE_TIME_AFTER_RECEIPT')
        for key in SPEC_FIELDS:
            error = integer(meta[key+'_error_code'], key+'_error_code', 0, 2**31-1)
            if meta[key] == 'NOT_AVAILABLE':
                unavailable_spec.append(key)
                continue
            try:
                number = real(meta[key])
                if key not in {'symbol_swap_long', 'symbol_swap_short'} and number < 0: raise ValueError
                if key in {'symbol_digits','symbol_swap_mode','symbol_swap_rollover3days','symbol_trade_calc_mode'} and number != number.to_integral_value(): raise ValueError
                if error: errors.append('SPEC_VALUE_WITH_API_ERROR')
            except (ValueError, InvalidOperation): errors.append('INVALID_CURRENT_SPEC_NUMERIC_VALUE')
    else:
        unavailable_spec = list(SPEC_FIELDS)
        if meta['spec_snapshot_capture_time_gmt'] != 'NOT_CAPTURED': errors.append('UNCAPTURED_SPEC_HAS_CAPTURE_TIME')
        if count: errors.append('QUOTE_CONTEXT_WITHOUT_GUARDED_CURRENT_SPEC')
    numeric, invalid, bad_quotes = [], 0, 0
    parsed_times, crossed, outside = [], 0, 0
    for raw in rows:
        row = {}
        quote_invalid = False
        for key in QUOTE_FIELDS:
            try:
                row[key] = real(raw[key])
            except (ValueError, InvalidOperation):
                quote_invalid = True
        if not quote_invalid:
            quote_invalid = row['bid'] <= 0 or row['ask'] <= 0 or row['last'] < 0 or row['volume_real'] < 0
        bad_quotes += bool(quote_invalid)
        if 'ask' in row and 'bid' in row: crossed += row['ask'] < row['bid']
        integer_invalid = False
        for key, maximum in (('time', 2**63-1), ('time_msc', 2**63-1), ('volume', 2**64-1), ('flags', 2**32-1)):
            try:
                row[key] = integer(raw[key], key, 0, maximum)
            except ValueError:
                integer_invalid = True
        parsed_times.append(row.get('time_msc'))
        if 'time_msc' in row: outside += not start <= row['time_msc'] < end
        if integer_invalid or len(row) != len(FIELDS):
            invalid += 1
        else:
            numeric.append(row)
    if invalid:
        errors.append('NONNUMERIC_OR_NONFINITE_RAW_ROWS')
    backwards = sum(b < a for a, b in zip(parsed_times, parsed_times[1:]) if a is not None and b is not None)
    inconsistent_time = sum(r['time'] != r['time_msc'] // 1000 for r in numeric)
    invalid_integer = sum(r['time_msc'] <= 0 or r['volume'] < 0 or not 0 <= r['flags'] <= 2**32 - 1 for r in numeric)
    for field, value in (('out_of_window_rows', outside), ('backward_time_rows', backwards),
                         ('crossed_quote_rows', crossed), ('invalid_quote_rows', bad_quotes)):
        if integer(meta[field], field, 0, 1000000) != value:
            errors.append('RECEIPT_' + field.upper() + '_MISMATCH')
        if value:
            errors.append(field.upper())
    if inconsistent_time or invalid_integer:
        errors.append('INVALID_OR_INCONSISTENT_INTEGER_FIELDS')
    if rows:
        if (integer(meta['first_time_msc'], 'first_time_msc', 0, 2**63-1) != parsed_times[0]
                or integer(meta['last_time_msc'], 'last_time_msc', 0, 2**63-1) != parsed_times[-1]):
            errors.append('RECEIPT_TIME_BOUND_MISMATCH')
    elif meta['first_time_msc'] != 'NONE' or meta['last_time_msc'] != 'NONE':
        errors.append('EMPTY_BOUND_MISMATCH')
    server_ok = len(meta['server']) <= 128 and bool(re.fullmatch(
        r'Deriv(?:SVG|BVI|FX|V|EU|Labuan|Mauritius|Cayman)?(?:[._ -][A-Za-z0-9._() -]+)?', meta['server'], re.IGNORECASE))
    company_ok = len(meta['company']) <= 256 and bool(re.fullmatch(r'Deriv(?:[ (][A-Za-z0-9 .(),-]*)?', meta['company'], re.IGNORECASE))
    symbol_ok = bool(re.fullmatch(r'(?:Boom|Crash) (?:300|500|600|900|1000) Index', meta['symbol']))
    if not server_ok:
        errors.append('UNEXPECTED_SERVER')
    if not company_ok: errors.append('UNEXPECTED_COMPANY')
    if not symbol_ok: errors.append('UNSUPPORTED_BOOM_CRASH_SYMBOL')
    if meta['symbol_custom_observed'] != 'false': errors.append('CUSTOM_SYMBOL_NOT_EXCLUDED')
    if (meta['terminal_trade_allowed_observed'] != 'false'
                    or meta['mql_trade_allowed_observed'] != 'false'
                    or integer(meta['guard_error'], 'guard_error', 0, 2**31-1) != 0):
        errors.append('TRADING_PROTECTION_NOT_VERIFIED')
    if meta['account_mode'] not in ('LIVE', 'DEMO'):
        errors.append('UNKNOWN_ACCOUNT_MODE')
    clean_last = bool(attempt_records and attempt_records[-1]['returned_count'] > 0
                      and attempt_records[-1]['returned_count'] == attempt_records[-1]['array_rows']
                      and attempt_records[-1]['last_error'] == 0)
    if rows and (meta['export_status'] != 'SAVED_NOT_PROVEN'
                    or meta['api_returned_without_error'] != 'true'
                    or meta['failure_reason'] != '' or not clean_last
                    or integer(meta['last_error'], 'last_error', 0, 2**31-1) != 0
                    or integer(meta['write_error'], 'write_error', 0, 2**31-1) != 0):
        errors.append('PARTIAL_OR_FAILED_API_RETURN')
    if not rows and (meta['export_status'] != 'EMPTY' or meta['api_returned_without_error'] != 'false'):
        errors.append('EMPTY_OR_FAILED_API_RETURN')
    if meta['historical_tick_completeness'] != 'NOT_PROVEN' or meta['synchronization_status'] != 'NOT_PROVEN':
        errors.append('UNSUPPORTED_COMPLETENESS_CLAIM')
    # Do not silently summarize a valid subset of a malformed retained array.
    summary_rows = numeric if not (invalid or outside or backwards or crossed or bad_quotes or inconsistent_time or invalid_integer) else []
    deltas = [b['time_msc'] - a['time_msc'] for a, b in zip(summary_rows, summary_rows[1:])]
    spreads = [r['ask'] - r['bid'] for r in summary_rows]
    if (sha(regular(receipt_path)) != receipt_hash or sha(regular(Path(__file__))) != tool_hash
            or raw_name and sha(regular(receipt_path.parent/raw_name)) != raw_hash):
        raise ValueError('Input or verifier changed during verification')
    return {
        'schema_version': 1,
        'status': 'FAIL_PRESERVED' if errors else ('QUOTES_VALIDATED_COVERAGE_NOT_PROVEN' if numeric else 'EMPTY_OR_FAILED_NOT_VERIFIED'),
        'scope': 'INDEPENDENT_RETAINED_MQL5_QUOTE_CHECK_ONLY',
        'receipt_sha256': receipt_hash, 'verifier_sha256': tool_hash,
        'raw_sha256': raw_hash,
        'symbol': meta['symbol'] if symbol_ok else None,
        'server': meta['server'] if server_ok else None, 'company': meta['company'] if company_ok else None,
        'reported_deriv_context_validated': server_ok and company_ok and symbol_ok and meta['symbol_custom_observed']=='false',
        'broker_origin_independently_authenticated': False,
        'account_mode': meta['account_mode'], 'requested_start_msc': start,
        'requested_end_exclusive_msc': end, 'rows': len(rows),
        'first_time_msc': summary_rows[0]['time_msc'] if summary_rows else None,
        'last_time_msc': summary_rows[-1]['time_msc'] if summary_rows else None,
        'same_millisecond_adjacent_pairs': sum(d == 0 for d in deltas),
        'inter_tick_gap_above_5000ms': sum(d > 5000 for d in deltas),
        'max_inter_tick_ms': max(deltas, default=None),
        'leading_unobserved_ms': summary_rows[0]['time_msc'] - start if summary_rows else None,
        'trailing_unobserved_ms': end - 1 - summary_rows[-1]['time_msc'] if summary_rows else None,
        'spread_price_min': str(min(spreads)) if spreads else None,
        'spread_price_median': str(statistics.median(spreads)) if spreads else None,
        'spread_price_max': str(max(spreads)) if spreads else None,
        'flag_counts': dict(Counter(str(r['flags']) for r in summary_rows)),
        'errors': list(dict.fromkeys(errors)), 'cost_verified': False, 'execution_verified': False,
        'profit_tested': False, 'full_tick_coverage_verified': False,
        'spec_scope': meta['spec_snapshot_status'],
        'spec_historical': False, 'current_spec_unavailable_fields': unavailable_spec,
        'retention_scope': meta['raw_retention_scope'], 'selected_attempt': selected,
        'attempts': attempt_records, 'all_attempt_raw_arrays_retained': False,
        'same_account_identity_verified': False,
        'limitations': ['A retained terminal quote array does not prove full server history or executable fills.',
                        'Only the selected attempt is retained; other returned arrays cannot be reconstructed.',
                        'The specification snapshot is current at capture, not historical.',
                        'Declared server and company are checked locally, not authenticated to the broker.'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        if any(p.is_symlink() for p in [output, *output.parents]) or not output.parent.is_dir() or output.exists():
            raise ValueError('Output must be new and have no symlink ancestors')
        report = verify_export(args.receipt)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError, UnicodeError, csv.Error, InvalidOperation):
        print('QUOTE_VERIFICATION_FAILED: malformed, unsafe, changed, or unavailable local evidence/output.', file=sys.stderr)
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0 if report['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN' else 2


if __name__ == '__main__':
    raise SystemExit(main())
