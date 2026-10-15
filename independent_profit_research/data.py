"""Read one explicitly selected, hash-pinned MT5 shard without altering evidence.

This module neither discovers manifests/final shards nor fits a scale or computes
returns. The caller must select the permitted training/validation/final path.
Prices are ROUND_HALF_EVEN quantized to 1e-8 price units before entering the
float execution kernel; prices below 2**45 leave headroom for dyadic arithmetic.
Raw text, duplicate rows and their original order are never rewritten.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from scripts.verify_mt5_quote_export import verify_export

from .execution import Quote
from .tape import QuoteTape

PRICE_SCALE = 10**8
MAX_SCALED_PRICE = 2**45
MAX_GAP_MS = 5000
DAY_MS = 86400000
FIELDS = ('time', 'time_msc', 'bid', 'ask', 'last', 'volume', 'flags', 'volume_real')
SAFETY_FIELDS = ('LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES',
                 'live_trading', 'ready_for_live', 'live_allowed', 'opened_trades')
SYMBOLS = {'BOOM500': 'Boom 500 Index', 'CRASH500': 'Crash 500 Index'}
_INTEGER = re.compile(r'(?:0|[1-9][0-9]*)\Z')
_REAL = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,4})?\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_MAX_DOUBLE = Decimal('1.7976931348623157e308')


@dataclass(frozen=True)
class LoadedDay:
    tape: QuoteTape
    quality: dict[str, Any]
    receipt_path: Path
    raw_path: Path
    price_scale: int = PRICE_SCALE

    @property
    def qualityfields(self) -> dict[str, Any]:
        """Compatibility name for callers requesting the shard's quality fields."""
        return self.quality


def _safe_environment() -> None:
    for key in SAFETY_FIELDS:
        if os.environ.get(key, 'false').strip().lower() != 'false':
            raise ValueError('Unsafe research environment: ' + key)


def _regular(path: Path | str, max_bytes: int) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Expected a regular evidence file without symlink ancestors')
    if path.stat().st_size > max_bytes:
        raise ValueError('Evidence file exceeds bounded size')
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: str, field: str, maximum: int) -> int:
    if not isinstance(value, str) or len(value) > 21 or not _INTEGER.fullmatch(value):
        raise ValueError('Invalid integer field: ' + field)
    result = int(value)
    if result > maximum:
        raise ValueError('Out-of-range integer field: ' + field)
    return result


def _decimal(value: str, field: str) -> Decimal:
    if len(value) > 128 or not _REAL.fullmatch(value):
        raise ValueError('Invalid real field: ' + field)
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError('Invalid real field: ' + field) from None
    if not number.is_finite() or abs(number) > _MAX_DOUBLE:
        raise ValueError('Nonfinite or out-of-range real field: ' + field)
    return number


def _scaled_price(number: Decimal) -> tuple[int, Decimal]:
    if number <= 0:
        raise ValueError('Bid/Ask must be positive before quantization')
    # Integer-ratio arithmetic avoids ambient Decimal precision rounding before
    # the declared HALF_EVEN operation, including exact halfway cases.
    numerator, denominator = number.as_integer_ratio()
    scaled, remainder = divmod(numerator * PRICE_SCALE, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and scaled % 2):
        scaled += 1
    if not 0 < scaled < MAX_SCALED_PRICE:
        raise ValueError('Quantized price must be positive and below 2**45')
    with localcontext() as context:
        context.prec = max(160, len(number.as_tuple().digits) + abs(number.as_tuple().exponent) + 16)
        error = abs(number - Decimal(scaled).scaleb(-8))
    return scaled, error


def _rows(path: Path, fields: tuple[str, ...], limit: int):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t', strict=True)
        if next(reader, None) != list(fields):
            raise ValueError('Unexpected strict TSV header')
        for ordinal, row in enumerate(reader):
            if ordinal >= limit or len(row) != len(fields):
                raise ValueError('Malformed or unbounded TSV row')
            if any(any(ord(c) < 32 or ord(c) == 127 for c in value) for value in row):
                raise ValueError('Control character in TSV field')
            yield dict(zip(fields, row))


def load_day(receipt_path: Path | str, expected_receipt_sha: str,
             expected_raw_sha: str, expected_symbol: str,
             start_ms: int, end_ms: int) -> LoadedDay:
    """Load one nonempty verified shard of at most a day, never scan a directory.

    Expected pins and bounds must come from the caller's frozen manifest. Both
    files are checked before and after verification/parsing. Gaps and uncovered
    boundaries remain explicit UNKNOWN regions; an accepted shard is not proof
    of complete broker history, historical costs, fills or profit.
    """
    _safe_environment()
    for value in (expected_receipt_sha, expected_raw_sha):
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError('Expected canonical lowercase SHA256 pins')
    if (type(start_ms) is not int or type(end_ms) is not int or start_ms <= 0
            or end_ms > 2**63-1 or not 0 < end_ms-start_ms <= DAY_MS):
        raise ValueError('Expected positive bounded integer window')
    if not isinstance(expected_symbol, str):
        raise ValueError('Unsupported expected symbol')
    symbol = SYMBOLS.get(expected_symbol, expected_symbol)
    if symbol not in SYMBOLS.values():
        raise ValueError('Unsupported expected symbol')

    receipt = _regular(receipt_path, 1048576)
    if _sha(receipt) != expected_receipt_sha:
        raise ValueError('Receipt SHA256 pin mismatch')
    match = re.fullmatch(r'([A-Za-z0-9_-]{1,64})_receipt\.tsv', receipt.name)
    if not match:
        raise ValueError('Invalid receipt basename')
    pairs = list(_rows(receipt, ('field', 'value'), 200))
    meta = {row['field']: row['value'] for row in pairs}
    if len(meta) != len(pairs):
        raise ValueError('Duplicate receipt field')
    raw_name = match[1] + '_ticks.tsv'
    if meta.get('raw_file') != raw_name or meta.get('raw_partial_file') != '':
        raise ValueError('Expected adjacent complete retained raw TSV')
    raw = _regular(receipt.parent / raw_name, 256*1048576)
    if _sha(raw) != expected_raw_sha:
        raise ValueError('Raw SHA256 pin mismatch')

    report = verify_export(receipt)
    if report.get('status') != 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN' or report.get('errors'):
        raise ValueError('Independent MT5 verification did not accept the shard')
    if (report.get('receipt_sha256') != expected_receipt_sha
            or report.get('raw_sha256') != expected_raw_sha):
        raise ValueError('Independent verification pins do not match')
    if (report.get('symbol') != symbol or report.get('requested_start_msc') != start_ms
            or report.get('requested_end_exclusive_msc') != end_ms):
        raise ValueError('Expected symbol or requested window mismatch')
    if any(report.get(key) is not False for key in
           ('cost_verified', 'execution_verified', 'profit_tested', 'full_tick_coverage_verified')):
        raise ValueError('Unsupported independent evidence claim')

    quotes, gaps = [], []
    previous = None
    same_ms = 0
    max_error = Decimal(0)
    for row in _rows(raw, FIELDS, 1000000):
        stamp = _integer(row['time_msc'], 'time_msc', 2**63-1)
        seconds = _integer(row['time'], 'time', 2**63-1)
        _integer(row['volume'], 'volume', 2**64-1)
        _integer(row['flags'], 'flags', 2**32-1)
        if seconds != stamp//1000 or not start_ms <= stamp < end_ms:
            raise ValueError('Inconsistent time or row outside requested window')
        bid, ask = _decimal(row['bid'], 'bid'), _decimal(row['ask'], 'ask')
        if not 0 < bid <= ask:
            raise ValueError('Invalid or crossed Bid/Ask before quantization')
        if _decimal(row['last'], 'last') < 0 or _decimal(row['volume_real'], 'volume_real') < 0:
            raise ValueError('Negative last or volume_real')
        scaled_bid, error_bid = _scaled_price(bid)
        scaled_ask, error_ask = _scaled_price(ask)
        max_error = max(max_error, error_bid, error_ask)
        if previous is not None:
            if stamp < previous:
                raise ValueError('Nondecreasing row order required; no sorting')
            delta = stamp-previous
            same_ms += delta == 0
            if delta > MAX_GAP_MS:
                gaps.append({'left_time_msc': previous, 'right_time_msc': stamp,
                             'delta_ms': delta, 'reason': 'ANOMALOUS_GAP_UNRESOLVED',
                             'warmup_reset_at_msc': stamp})
        quotes.append(Quote(stamp, scaled_bid, scaled_ask))
        previous = stamp
    if not quotes or len(quotes) != report.get('rows'):
        raise ValueError('Nonempty independently verified row count required')
    first, last = quotes[0].time_msc, quotes[-1].time_msc
    if (first != report.get('first_time_msc') or last != report.get('last_time_msc')
            or same_ms != report.get('same_millisecond_adjacent_pairs')
            or len(gaps) != report.get('inter_tick_gap_above_5000ms')):
        raise ValueError('Independent row/time/gap accounting mismatch')

    unknown = []
    if first > start_ms:
        unknown.append({'start_ms': start_ms, 'end_exclusive_ms': first,
                        'reason': 'UNOBSERVED_HEAD'})
    for gap in gaps:
        unknown.append({'start_ms': gap['left_time_msc']+1,
                        'end_exclusive_ms': gap['right_time_msc'],
                        'reason': 'ANOMALOUS_GAP_UNRESOLVED'})
    if last < end_ms-1:
        unknown.append({'start_ms': last+1, 'end_exclusive_ms': end_ms,
                        'reason': 'UNOBSERVED_TAIL'})
    tape = QuoteTape(quotes)
    if (_sha(_regular(receipt, 1048576)) != expected_receipt_sha
            or _sha(_regular(raw, 256*1048576)) != expected_raw_sha):
        raise ValueError('Pinned input changed during loading')
    _safe_environment()
    quality = {
        'status': 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN',
        'receipt_sha256': expected_receipt_sha, 'raw_sha256': expected_raw_sha,
        'verifier_sha256': report.get('verifier_sha256'), 'symbol': symbol,
        'server': report.get('server'), 'company': report.get('company'),
        'account_mode': report.get('account_mode'),
        'requested_start_msc': start_ms, 'requested_end_exclusive_msc': end_ms,
        'request_is_full_utc_day': end_ms-start_ms == DAY_MS and start_ms % DAY_MS == 0,
        'rows': len(tape), 'first_time_msc': first, 'last_time_msc': last,
        'same_millisecond_adjacent_pairs': same_ms,
        'inter_tick_gap_above_5000ms': len(gaps), 'source_gaps': gaps,
        'leading_unobserved_ms': first-start_ms,
        'trailing_unobserved_ms': end_ms-1-last,
        'unknown_intervals': unknown, 'unknown_interval_bounds': 'INTEGER_MILLISECONDS_HALF_OPEN',
        'last_bundle_unconfirmed': True, 'last_bundle_time_msc': last,
        'unknown_path_handling_required': True, 'gap_threshold_ms': MAX_GAP_MS,
        'price_scale': PRICE_SCALE, 'price_units': 'INTEGER_1E_MINUS8_PRICE_UNITS',
        'max_scaled_price_exclusive': MAX_SCALED_PRICE,
        'quantization_mode': 'QUANTIZED_1E_MINUS8_NOT_RAW_EXACT',
        'quantization_rounding': 'ROUND_HALF_EVEN',
        'max_abs_representation_error': str(max_error),
        'max_abs_representation_error_bound': '0.000000005',
        'raw_unchanged': True, 'row_order_preserved': True, 'deduplication_applied': False,
        'flags_used_as_features': False, 'retention_scope': 'SELECTED_ATTEMPT_ONLY',
        'all_attempt_raw_arrays_retained': False, 'spec_historical': False,
        'cost_verified': False, 'execution_verified': False, 'profit_tested': False,
        'full_tick_coverage_verified': False,
        'broker_origin_independently_authenticated': False,
        'same_account_identity_verified': False,
        'safety': {key: False for key in SAFETY_FIELDS},
    }
    return LoadedDay(tape, quality, receipt, raw)
