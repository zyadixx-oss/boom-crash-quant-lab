"""Synthetic pinned TSVs only: no platform, market data, or sealed final access."""
import csv
from decimal import Decimal, localcontext
import hashlib

import pytest

from independent_profit_research import data
from scripts import verify_mt5_quote_export as checker

START, END = 1735776000000, 1735776060000  # Artificial Jan2 2025 one-minute shard.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\r\n')
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    for key in data.SAFETY_FIELDS:
        monkeypatch.delenv(key, raising=False)
    prefix = 'ArtificialDay'
    rows = [[str((START+offset)//1000), str(START+offset), bid, ask, '0', '0', flags, '0']
            for offset, bid, ask, flags in [
                (1000, '100.000000005', '100.20000000000000284', '6'),
                (1000, '100.000000015', '100.20000000000000284', '2'),
                (6000, '100.1', '100.3', '6'),
                (12001, '100.2', '100.4', '6'),
            ]]
    meta = dict.fromkeys(checker.FALSE_FIELDS, 'false')
    meta.update({
        'schema_version': '1', 'export_status': 'SAVED_NOT_PROVEN', 'failure_reason': '',
        'symbol': 'Boom 500 Index', 'server': 'DerivSVG-Server', 'company': 'Deriv (SVG) LLC',
        'account_mode': 'DEMO', 'symbol_custom_observed': 'false',
        'capture_time_gmt': '2026.10.12 21:47:29 UTC',
        'request_from_gmt': '2025.01.02 00:00:00 UTC',
        'request_end_exclusive_gmt': '2025.01.02 00:01:00 UTC',
        'request_from_msc': str(START), 'request_end_exclusive_msc': str(END),
        'copy_to_inclusive_msc': str(END-1), 'window_end_exclusive': 'true',
        'max_window_seconds': '86400', 'copy_mode': 'COPY_TICKS_ALL',
        'raw_file': prefix+'_ticks.tsv', 'raw_partial_file': '',
        'row_count': '4', 'raw_rows_written': '4',
        'first_time_msc': str(START+1000), 'last_time_msc': str(START+12001),
        'last_error': '0', 'guard_error': '0', 'write_error': '0',
        'attempts': '1', 'retries': '0', 'selected_attempt': '1',
        'raw_retention_scope': 'SELECTED_ATTEMPT_ONLY', 'copy_returned_count': '4',
        'api_returned_without_error': 'true', 'connection_wait_limit_seconds': '30',
        'connection_wait_elapsed_ms': '100', 'terminal_trade_allowed_observed': 'false',
        'mql_trade_allowed_observed': 'false', 'out_of_window_rows': '0',
        'backward_time_rows': '0', 'crossed_quote_rows': '0', 'invalid_quote_rows': '0',
        'same_millisecond_rows_preserved': 'true', 'row_order_preserved': 'true',
        'synchronization_status': 'NOT_PROVEN', 'historical_tick_completeness': 'NOT_PROVEN',
        'limitation': 'Artificial fixture; no historical cost/fill evidence.',
        'attempt_1_returned_count': '4', 'attempt_1_array_rows': '4',
        'attempt_1_last_error': '0', 'attempt_1_elapsed_ms': '10',
        'spec_snapshot_status': 'CURRENT_ONLY_NOT_HISTORICAL',
        'spec_snapshot_capture_time_gmt': '2026.10.12 21:47:29 UTC',
    })
    values = ('0.001', '3', '1', '0.001', '0.001', '0.001', '0.001', '0.2',
              '190', '0.01', '5', '-23', '-23', '7', '4')
    for key, value in zip(checker.SPEC_FIELDS, values):
        meta[key], meta[key+'_error_code'] = value, '0'
    receipt, raw = tmp_path/(prefix+'_receipt.tsv'), tmp_path/(prefix+'_ticks.tsv')

    def save():
        write_tsv(receipt, ['field', 'value'], meta.items())
        write_tsv(raw, data.FIELDS, rows)
        return receipt

    save()
    return meta, rows, receipt, raw, save


def load(evidence, **overrides):
    arguments = {'receipt_path': evidence[2], 'expected_receipt_sha': digest(evidence[2]),
                 'expected_raw_sha': digest(evidence[3]), 'expected_symbol': 'BOOM500',
                 'start_ms': START, 'end_ms': END}
    arguments.update(overrides)
    return data.load_day(**arguments)


def stub_original_report(evidence, monkeypatch):
    report = checker.verify_export(evidence[2])
    assert report['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'

    def report_after_repin(_):
        return {**report, 'receipt_sha256': digest(evidence[2]), 'raw_sha256': digest(evidence[3])}

    monkeypatch.setattr(data, 'verify_export', report_after_repin)


def test_real_independent_verifier_and_preserved_quantized_tape(evidence):
    before = evidence[2].read_bytes(), evidence[3].read_bytes()
    result = load(evidence)
    assert len(result.tape) == 4
    assert result.tape.timestamps == (START+1000, START+1000, START+6000, START+12001)
    assert [q.bid for q in result.tape] == [10000000000, 10000000002, 10010000000, 10020000000]
    assert result.tape.quotes[0].ask == 10020000000
    assert result.price_scale == 10**8
    assert result.qualityfields is result.quality
    assert result.quality['quantization_mode'] == 'QUANTIZED_1E_MINUS8_NOT_RAW_EXACT'
    assert Decimal(result.quality['max_abs_representation_error']) == Decimal('0.000000005')
    assert before == (evidence[2].read_bytes(), evidence[3].read_bytes())
    assert result.quality['same_millisecond_adjacent_pairs'] == 1
    assert result.quality['safety'] == dict.fromkeys(data.SAFETY_FIELDS, False)
    assert not any(result.quality[key] for key in ('cost_verified', 'execution_verified', 'profit_tested',
                                                'full_tick_coverage_verified', 'deduplication_applied',
                                                'flags_used_as_features', 'spec_historical'))


def test_gap_boundary_and_head_tail_unknown_masks(evidence):
    quality = load(evidence).quality
    assert quality['inter_tick_gap_above_5000ms'] == 1  # Exactly5000 is not >5000.
    assert quality['source_gaps'] == [{'left_time_msc': START+6000,
        'right_time_msc': START+12001, 'delta_ms': 6001,
        'reason': 'ANOMALOUS_GAP_UNRESOLVED', 'warmup_reset_at_msc': START+12001}]
    assert quality['unknown_intervals'] == [
        {'start_ms': START, 'end_exclusive_ms': START+1000, 'reason': 'UNOBSERVED_HEAD'},
        {'start_ms': START+6001, 'end_exclusive_ms': START+12001, 'reason': 'ANOMALOUS_GAP_UNRESOLVED'},
        {'start_ms': START+12002, 'end_exclusive_ms': END, 'reason': 'UNOBSERVED_TAIL'},
    ]
    assert quality['last_bundle_unconfirmed'] and quality['unknown_path_handling_required']
    assert not quality['request_is_full_utc_day']


@pytest.mark.parametrize('symbol', ['BOOM500', 'Boom 500 Index', 'CRASH500', 'Crash 500 Index'])
def test_explicit_two_symbol_allowlist(evidence, symbol):
    evidence[0]['symbol'] = data.SYMBOLS.get(symbol, symbol)
    evidence[4]()
    assert load(evidence, expected_symbol=symbol).quality['symbol'] == evidence[0]['symbol']


@pytest.mark.parametrize('key', data.SAFETY_FIELDS)
@pytest.mark.parametrize('value', ['true', '1', 'unknown', ''])
def test_each_unsafe_environment_flag_fails_closed(evidence, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='Unsafe research'):
        load(evidence)


@pytest.mark.parametrize('argument,value', [
    ('expected_receipt_sha', '0'*64), ('expected_raw_sha', '0'*64),
    ('expected_receipt_sha', 'A'*64), ('expected_raw_sha', None),
    ('expected_symbol', 'BOOM300'), ('expected_symbol', 'EURUSD'),
    ('expected_symbol', 'CRASH500'), ('expected_symbol', None),
    ('start_ms', START+1000), ('end_ms', END+1000),
    ('start_ms', True), ('end_ms', float(END)), ('start_ms', 0),
    ('end_ms', START), ('end_ms', START+86400001),
])
def test_pins_symbol_and_window_required(evidence, argument, value):
    with pytest.raises(ValueError):
        load(evidence, **{argument: value})


def test_raw_tamper_rejected_without_rewriting(evidence):
    pin = digest(evidence[3])
    evidence[3].write_bytes(evidence[3].read_bytes().replace(b'100.1\t', b'100.9\t'))
    before = evidence[3].read_bytes()
    with pytest.raises(ValueError, match='Raw SHA256'):
        load(evidence, expected_raw_sha=pin)
    assert evidence[3].read_bytes() == before


@pytest.mark.parametrize('field,value', [('terminal_trade_allowed_observed', 'true'),
    ('safety_upper_live_trading', 'true'), ('costs_verified', 'true'),
    ('export_status', 'PARTIAL'), ('historical_tick_completeness', 'PROVEN')])
def test_verifier_receipt_safety_and_status_guards(evidence, field, value):
    evidence[0][field] = value
    evidence[4]()
    with pytest.raises(ValueError):
        load(evidence)


@pytest.mark.parametrize('field,value', [
    ('time', str(START//1000)), ('time_msc', str(END)),
    ('time_msc', str(START-1)), ('time', '1.0'), ('volume', '-1'),
    ('flags', str(2**32)), ('bid', 'NaN'), ('ask', 'Infinity'),
    ('last', '-1'), ('volume_real', '-1'), ('bid', ' 100'), ('ask', '99'),
    ('bid', '0'), ('bid', '1e309'), ('volume_real', '1_00'),
])
def test_independent_parser_rejects_invalid_rows_even_after_stubbed_verification(evidence, monkeypatch, field, value):
    stub_original_report(evidence, monkeypatch)
    evidence[1][0][data.FIELDS.index(field)] = value
    evidence[4]()
    with pytest.raises(ValueError):
        load(evidence)


def test_backward_order_rejected_by_own_parser(evidence, monkeypatch):
    stub_original_report(evidence, monkeypatch)
    evidence[1][1], evidence[1][3] = evidence[1][3], evidence[1][1]
    evidence[4]()
    with pytest.raises(ValueError, match='Nondecreasing'):
        load(evidence)


@pytest.mark.parametrize('header', [list(reversed(data.FIELDS)), list(data.FIELDS[:-1]), list(data.FIELDS)+['extra']])
def test_exact_eight_column_schema(evidence, monkeypatch, header):
    stub_original_report(evidence, monkeypatch)
    write_tsv(evidence[3], header, evidence[1])
    with pytest.raises(ValueError, match='header'):
        load(evidence)


@pytest.mark.parametrize('row', [[], ['0']*7, ['0']*9, ['0\x00']*8])
def test_row_shape_and_controls_rejected(evidence, monkeypatch, row):
    stub_original_report(evidence, monkeypatch)
    evidence[1][0] = row
    evidence[4]()
    with pytest.raises(ValueError):
        load(evidence)


@pytest.mark.parametrize('value', ['0.0000000049', '0.000000005', '351843.72088832', '351843.72088833'])
def test_zero_or_upper_bound_quantization_rejected(evidence, monkeypatch, value):
    stub_original_report(evidence, monkeypatch)
    evidence[1][0][2:4] = [value, value]
    evidence[4]()
    with pytest.raises(ValueError, match='Quantized price'):
        load(evidence)


@pytest.mark.parametrize('value,expected', [('0.0000000051', 1), ('0.000000015', 2),
    ('100.000000005', 10000000000), ('100.000000015', 10000000002),
    ('351843.72088831', 2**45-1)])
def test_ties_even_and_exact_kernel_range_independent_of_decimal_context(value, expected):
    with localcontext() as context:
        context.prec = 3
        scaled, error = data._scaled_price(Decimal(value))
    assert scaled == expected and 0 < scaled < 2**45
    assert Decimal(0) <= error <= Decimal('0.000000005')
    assert float(scaled).as_integer_ratio() == (scaled, 1)


def test_flags_only_changes_do_not_change_prices_or_time(evidence):
    before = load(evidence).tape.quotes
    for row in evidence[1]:
        row[6] = '14'
    evidence[4]()
    assert load(evidence).tape.quotes == before


def test_report_cannot_claim_profit_or_complete_coverage(evidence, monkeypatch):
    report = checker.verify_export(evidence[2])
    report['full_tick_coverage_verified'] = True
    monkeypatch.setattr(data, 'verify_export', lambda _: report)
    with pytest.raises(ValueError, match='evidence claim'):
        load(evidence)


@pytest.mark.parametrize('target', ['receipt', 'raw'])
def test_valid_change_during_verification_caught_by_final_hash(evidence, monkeypatch, target):
    real_verify = data.verify_export

    def mutate_after_verify(path):
        report = real_verify(path)
        if target == 'raw':
            evidence[3].write_bytes(evidence[3].read_bytes().replace(b'100.1\t', b'100.15\t'))
        else:
            evidence[2].write_bytes(evidence[2].read_bytes().replace(b'Artificial fixture;', b'Changed fixture;'))
        return report

    monkeypatch.setattr(data, 'verify_export', mutate_after_verify)
    with pytest.raises(ValueError, match='Pinned input changed during loading'):
        load(evidence)


def test_full_utc_day_request_does_not_claim_complete_history(evidence):
    end = START+data.DAY_MS
    evidence[0].update(request_end_exclusive_gmt='2025.01.03 00:00:00 UTC',
                       request_end_exclusive_msc=str(end), copy_to_inclusive_msc=str(end-1))
    evidence[4]()
    quality = load(evidence, end_ms=end).quality
    assert quality['request_is_full_utc_day']
    assert quality['max_scaled_price_exclusive'] == 2**45
    assert not quality['full_tick_coverage_verified']
    assert quality['trailing_unobserved_ms'] > 0


def test_bom_and_unchanged_duplicate_rows_are_preserved(evidence):
    evidence[1][1] = evidence[1][0].copy()
    evidence[4]()
    evidence[3].write_bytes(b'\xef\xbb\xbf'+evidence[3].read_bytes())
    before = evidence[3].read_bytes()
    loaded = load(evidence)
    assert loaded.tape.quotes[0] == loaded.tape.quotes[1]
    assert len(loaded.tape) == 4 and loaded.quality['same_millisecond_adjacent_pairs'] == 1
    assert evidence[3].read_bytes() == before


def test_symlink_input_rejected(evidence, tmp_path):
    linked = tmp_path/'Other_receipt.tsv'
    linked.symlink_to(evidence[2])
    with pytest.raises(ValueError, match='regular evidence'):
        load(evidence, receipt_path=linked)


def test_no_auto_discovery_of_sibling_sealed_files(evidence):
    sealed = evidence[2].parent/'SEALED_FINAL_ticks.tsv'
    sealed.write_text('not a TSV; must never be opened')
    assert len(load(evidence).tape) == 4
    assert sealed.read_text() == 'not a TSV; must never be opened'
