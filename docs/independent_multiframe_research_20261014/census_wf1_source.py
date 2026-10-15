"""Source-quality census only: WF1 development, no strategy/final calculations."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import run_independent_profit_study as original
from independent_multiframe_research.bars import FRAME_MINUTES, MINUTE_MS, build_frames
from independent_profit_research.data import SAFETY_FIELDS
from independent_profit_research.tape import QuoteTape

OUT = Path(__file__).resolve().parent
DECL_SHA = 'a3f9b73838b63abfcb0e85cc0fce076c5e5ae7c31fd8545ad05038f208dbe0e6'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def source_counts(tape):
    distinct = ambiguous = exact_duplicates = bundles_multiple_rows = 0
    observed_minutes, ambiguous_minutes, gap_minutes = set(), set(), set()
    gaps, current, pairs, rows_in_bundle = [], None, set(), 0

    def finish():
        nonlocal distinct, ambiguous, exact_duplicates, bundles_multiple_rows
        if current is None:
            return
        distinct += 1
        observed_minutes.add(current // MINUTE_MS * MINUTE_MS)
        exact_duplicates += rows_in_bundle - len(pairs)
        bundles_multiple_rows += rows_in_bundle > 1
        if len(pairs) > 1:
            ambiguous += 1
            ambiguous_minutes.add(current // MINUTE_MS * MINUTE_MS)

    for quote in tape:
        stamp = quote.time_msc
        if current is not None and stamp != current:
            finish()
            if stamp - current > 5000:
                gaps.append({'previous_ms': current, 'next_ms': stamp,
                             'interval_ms': stamp-current,
                             'crosses_utc_day': current//original.DAY_MS != stamp//original.DAY_MS})
                gap_minutes.update(range(current//MINUTE_MS*MINUTE_MS,
                                         stamp//MINUTE_MS*MINUTE_MS+MINUTE_MS,
                                         MINUTE_MS))
            pairs, rows_in_bundle = set(), 0
        current = stamp
        pairs.add((quote.bid, quote.ask))
        rows_in_bundle += 1
    finish()
    return {'raw_rows': len(tape), 'distinct_timestamps': distinct,
            'extra_rows_at_repeated_timestamps': len(tape)-distinct,
            'multiple_row_timestamp_bundles': bundles_multiple_rows,
            'exact_duplicate_price_pair_rows_within_timestamp': exact_duplicates,
            'ambiguous_same_ms_price_bundles': ambiguous,
            'ambiguous_affected_minutes': len(ambiguous_minutes),
            'observed_source_minutes': len(observed_minutes),
            'gaps_above_5000ms': len(gaps),
            'gaps_crossing_utc_day': sum(g['crosses_utc_day'] for g in gaps),
            'largest_gap_ms': max((g['interval_ms'] for g in gaps), default=None),
            'gap_endpoint_and_crossed_minutes': len(gap_minutes),
            'ambiguous_or_gap_affected_minute_union': len(ambiguous_minutes | gap_minutes),
            'gaps': gaps,
            '_observed_minutes': observed_minutes}


def frame_counts(frames, start, end):
    result = {}
    for size in FRAME_MINUTES:
        width = size * MINUTE_MS
        first = (start+width-1)//width*width
        expected = max(0, (end-first)//width)
        # At an aligned half-open end, the final nominal candle cannot borrow
        # the quote at end to confirm itself; this is a structural upper bound.
        tail_unconfirmable = int(expected > 0 and end % width == 0)
        rows = frames[size]
        by_run = Counter(bar.run_id for bar in rows)
        result[str(size)] = {'nominal_complete_utc_intervals': expected,
            'structurally_unconfirmable_tail_intervals': tail_unconfirmable,
            'maximum_confirmable_without_outside_quote': expected-tail_unconfirmable,
            'valid_bars': len(rows), 'invalid_or_unconfirmed_nominal_intervals': expected-len(rows),
            'valid_fraction_of_nominal': len(rows)/expected if expected else None,
            'valid_run_ids': sorted(by_run),
            'valid_bars_per_run': {str(k): v for k, v in sorted(by_run.items())},
            'first_valid_open_ms': rows[0].open_ms if rows else None,
            'last_valid_end_ms': rows[-1].end_ms if rows else None}
    return result


def context_availability(frames):
    """Count data availability only; do not compute indicator values or signals."""
    by_end = {size: {bar.end_ms: bar for bar in rows} for size, rows in frames.items()}

    def at(size, end, now, run):
        bar = by_end[size].get(end)
        return bar is not None and bar.confirmed_ms <= now and bar.run_id == run

    def history(size, end, count, now, run):
        return all(at(size, end-i*size*MINUTE_MS, now, run) for i in range(count))

    counts = Counter()
    for bar in frames[15]:
        time, now, run = bar.end_ms, bar.confirmed_ms, bar.run_id
        h4 = at(240, time//(240*MINUTE_MS)*(240*MINUTE_MS), now, run)
        h1 = at(60, time//(60*MINUTE_MS)*(60*MINUTE_MS), now, run)
        atr_history = history(5, time, 15, now, run)
        ema_history = history(15, time, 20, now, run)
        compression_history = history(15, time, 31, now, run)
        counts['valid_m15_close_times'] += 1
        counts['exact_same_run_h4_available'] += h4
        counts['exact_same_run_h1_available'] += h1
        counts['exact_same_run_h4_and_h1_available'] += h4 and h1
        counts['m5_atr14_required_15_bar_history_available'] += atr_history
        counts['m15_ema20_required_20_bar_history_available'] += ema_history
        counts['m15_compression_required_31_bar_history_available'] += compression_history
        counts['h4_h1_and_atr_history_available'] += h4 and h1 and atr_history
        counts['h4_h1_atr_and_ema_history_available'] += h4 and h1 and atr_history and ema_history
        counts['h4_h1_atr_and_compression_history_available'] += h4 and h1 and atr_history and compression_history
    return {'counts': dict(counts),
            'scope': 'STRUCTURAL_SOURCE_AVAILABILITY_ONLY_NOT_POLICY_ELIGIBILITY',
            'native_candle_body_filter_computed': False, 'atr_or_ema_values_computed': False,
            'sweep_breakout_or_pullback_conditions_computed': False}


def main():
    original.safe()
    declaration = OUT/'declaration.json'
    if sha(declaration) != DECL_SHA:
        raise ValueError('Unexpected frozen multiframe declaration')
    spec = read(declaration)
    if spec['stages']['WF1'] != [30, 50, 30] or any(spec[k] is not False for k in SAFETY_FIELDS):
        raise ValueError('Unexpected calendar/safety declaration')
    v1 = original.OUT/'declaration.json'
    if sha(v1) != spec['original_declaration_sha256']:
        raise ValueError('Original frozen declaration changed')
    dependency_pins = dict(read(v1)['source_pins']) | dict(spec['source_pins'])
    for name, digest in dependency_pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen diagnostic dependency changed: '+name)
    start, end = original.START_MS+30*original.DAY_MS, original.START_MS+50*original.DAY_MS
    report = {'schema_version': 1, 'stage': 'WF1', 'development_day_range': [30, 50],
        'start_ms': start, 'end_exclusive_ms': end,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'declaration_sha256': DECL_SHA, 'original_declaration_sha256': sha(v1),
        'helper_path': str(Path(__file__).resolve().relative_to(ROOT)),
        'helper_sha256': sha(Path(__file__).resolve()),
        'dependency_pins': dependency_pins, 'symbols': {},
        'market_strategy_computations': 0, 'strategy_signals_computed': 0,
        'returns_or_selection_recomputed': False, 'sealed_final_raw_reads': 0,
        'private_history_reads': 0, 'mt5_queries': 0,
        'source_coverage_verified': False, 'cash_net': None,
        'costs_verified': False, 'execution_verified': False,
        'safety': {key: False for key in SAFETY_FIELDS}}
    for symbol in ('BOOM500', 'CRASH500'):
        records = original.records(symbol, 30, 50)
        if any(row['partition'] != 'development' or not 30 <= row['day_index'] < 50 for row in records):
            raise ValueError('Census attempted an unauthorized source partition')
        pins = {name: digest for record in records for name, digest in record['files'].items()}
        tape = QuoteTape(q for record in records for q in original.load(record).tape)
        print('LOADED', symbol, len(tape), flush=True)
        counts = source_counts(tape)
        observed_minutes = counts.pop('_observed_minutes')
        counts['missing_source_minutes'] = (end-start)//MINUTE_MS-len(observed_minutes)
        counts['first_timestamp_ms'] = tape.timestamps[0] if len(tape) else None
        counts['last_timestamp_ms'] = tape.timestamps[-1] if len(tape) else None
        frames = build_frames(tape, start, end)
        report['symbols'][symbol] = {'shards': len(records), 'source_files': pins,
            'source_file_pins_rechecked_before_load': len(pins), 'source_counts': counts,
            'frames': frame_counts(frames, start, end), 'context_availability': context_availability(frames)}
        for name, digest in pins.items():
            if sha(ROOT/name) != digest:
                raise ValueError('Source changed during census: '+name)
        print('CENSUS_DONE', symbol, 'M1', len(frames[1]), 'H4', len(frames[240]), flush=True)
        del frames, tape, records, observed_minutes
        gc.collect()
    for name, digest in dependency_pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen dependency changed during census: '+name)
    report['status'] = 'SOURCE_CENSUS_COMPLETE_COVERAGE_NOT_PROVEN'
    report['source_unchanged_after_census'] = True
    report['dependency_pins_unchanged_after_census'] = True
    with (OUT/'WF1_source_census.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('SAVED', str(OUT/'WF1_source_census.json'), flush=True)


if __name__ == '__main__':
    main()
