"""Explicit, immutable orchestration for a separately reviewed replication.

Importing this module performs no acquisition, source discovery, freeze or study.
Source dates never change on price/payoff/coverage. Missing or failed declared
days block the entire requested stage before any raw quote is opened. The old
studies remain untouched. A statistical pass never establishes cash profit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from independent_profit_research.data import SAFETY_FIELDS, load_day
from independent_profit_research.execution import ExecutionConfig
from independent_profit_research.tape import QuoteTape
from independent_multiframe_research.sharded import DAY_MS, ShardedQuoteStore
from independent_multiframe_research.spread_fit import fit_positive_spread_prefixes
from independent_multiframe_research.streaming import build_streamed_frames
from independent_multiframe_research.streaming_policies import generate_signals, clock_signals
from independent_multiframe_research.replication_control import (
    POLICY_ID as SPARSE_ID, generate_sparse_control, sparse_schedule,
)
from independent_multiframe_research.streaming_runner import evaluate_streamed_signals

START = int(datetime(2023, 1, 3, tzinfo=timezone.utc).timestamp() * 1000)
END = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
PILOTS = ('2024-01-02', '2024-04-02', '2024-07-02', '2024-10-02', '2024-10-03')
STAGES = {'WF1': (90, 180, 90), 'WF2': (180, 270, 180), 'WF3': (270, 360, 270),
          'VALIDATION': (360, 510, 360), 'FINAL': (510, 729, 510)}
DEVELOPMENT = ('WF1', 'WF2', 'WF3', 'VALIDATION')
POLICIES = ('CRT_SWEEP_RECLAIM', 'SQUEEZE_BREAKOUT', 'PULLBACK_CONTINUATION')
CONTROLS = ('SPARSE_CLOCK', 'DENSE_CLOCK')
SCENARIOS = ('BASE', 'STRESS')
STATE_NAMES = ('state.json', 'batch2_state.json', 'batch3_state.json', 'batch4_state.json')
PRODUCER_NAMES = ('calendar_declaration.json', 'calendar_declaration_batch2.json',
                  'calendar_declaration_batch3.json', 'calendar_declaration_batch4.json')
BATCH_RANGES = ((0, 30), (30, 180), (180, 508), (508, 724))
ACQUISITION_CODE = ('scripts/mt5/ExportResearchQuotes.mq5',
                    'scripts/verify_mt5_quote_export.py',
                    'work/independent_profit_pilot_20261013/run_dataset_probe.py')
OLD_DECLARATIONS = ('work/independent_profit_study_20261013/declaration.json',
                    'work/independent_multiframe_study_20261014/declaration.json')
NEW_CODE = tuple('independent_multiframe_research/' + name + '.py' for name in (
    'sharded', 'streaming', 'streaming_policies', 'streaming_runner', 'spread_fit',
    'replication_control', 'replication_statistics')) + ('scripts/run_multiframe_replication_study.py',)
SAFETY = {key: False for key in SAFETY_FIELDS}
HASH = re.compile(r'[0-9a-f]{64}\Z')
META = ROOT / 'work/multiframe_replication_source_20261014'
OUT = ROOT / 'work/multiframe_replication_study_20261015'
CALENDAR_SHA256 = '917b08901b48073ad66e79aa1c3743078e21e2bbb5f9fefe35b4a142f470f84d'
SOURCE_ROOT = 'data/independent_profit_ticks_20261013/source_diagnostics/replication_source_20261014'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def safe():
    if any(os.environ.get(key, 'false').strip().lower() != 'false' for key in SAFETY_FIELDS):
        raise ValueError('All eight live-trading environment flags must remain false')


def utc_day(stamp):
    return datetime.fromtimestamp(stamp / 1000, timezone.utc).date().isoformat()


def mask_for(start, end):
    return [utc_day(day) not in PILOTS for day in range(start, end, DAY_MS)]


def segments(start, end):
    output, first = [], None
    for day in range(start, end, DAY_MS):
        if utc_day(day) in PILOTS:
            if first is not None:
                output.append((first, day))
                first = None
        elif first is None:
            first = day
    if first is not None:
        output.append((first, end))
    return tuple(output)


def embargoed_signals(signals, start, end, exit_latency_ms):
    """Pure calendar filter, never coverage/payoff filtering or sparse redrawing."""
    intervals = segments(start, end)
    iterator = iter(signals)
    try:
        for signal in iterator:
            decision = signal.decision_ms
            if decision >= end - 1_800_000:
                break
            if decision < start + 1_800_000:
                continue
            for left, right in intervals:
                if left + 1_800_000 <= decision < right - 1_800_000:
                    if signal.horizon_end_ms + exit_latency_ms < right:
                        yield signal
                    break
    finally:
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()


def template():
    return {'id': 'crash500-multiframe-replication-20261015', 'schema_version': 1,
            'calendar_start_ms': START, 'calendar_end_exclusive_ms': END,
            'calendar_slots': 729, 'development_slots': 510, 'final_slots': 219,
            'excluded_pilot_days': list(PILOTS), 'stages': {k: list(v) for k, v in STAGES.items()},
            'symbol': 'CRASH500', 'native_direction': -1, 'policies': list(POLICIES),
            'controls': list(CONTROLS), 'scenarios': list(SCENARIOS), 'gap_ms': 5000,
            'BASE': {'latency_ms': 250, 'slippage_U_per_leg': .25},
            'STRESS': {'latency_ms': 2000, 'slippage_U_per_leg': .5},
            'bootstrap_draws': 99999, 'bootstrap_block_days': [3, 7], 'holm_family_size': 33,
            'development_gate': {'both_scenarios_required': True, 'candidate_completed': 100,
                'candidate_validation_completed': 20, 'sparse_completed': 100,
                'sparse_validation_completed': 30, 'positive_validation': True,
                'positive_WF_folds': 2, 'positive_excess_sparse_validation': True,
                'positive_excess_sparse_WF_folds': 2, 'unknown_allowed': 0},
            'final_sample_gate': {'candidate_completed': 100, 'sparse_completed': 80,
                'candidate_active_days': 40, 'sparse_active_days': 60,
                'candidate_per_sixth': 5, 'sparse_per_sixth': 5, 'candidate_per_third': 20},
            'marked_drawdown_max_pct': 20., 'PF_floor_required': False,
            'old_studies_reinterpreted': False, 'global_unseen_history_proven': False,
            'cash_net': None, 'costs_verified': False, 'execution_verified': False,
            'profit_qualified': False, 'alpha_protocol_frozen': False, **SAFETY}


class Controller:
    def __init__(self, root=ROOT, study_dir=OUT, source_meta_dir=META, calendar_path=None):
        self.root = Path(root).absolute()
        self.out = Path(study_dir).absolute()
        self.meta = Path(source_meta_dir).absolute()
        self.calendar_path = self.meta / 'calendar_declaration.json' if calendar_path is None else Path(calendar_path).absolute()
        self._final_authorization = None

    def _guard_final(self, permit_final, declaration_sha=None):
        if not permit_final or self._final_authorization is None:
            raise ValueError('Final source/cache access requires current rederived selection')
        declared, selected = self._final_authorization
        if (declaration_sha is not None and declaration_sha != declared
                or sha(self._output('declaration.json')) != declared
                or fingerprint(self._read('selection.json')) != selected):
            raise ValueError('Final authorization differs from current declaration/selection')

    def _path(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Evidence paths must be explicit repository-relative paths')
        path = self.root / path
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlink evidence paths are not permitted')
        return path

    def _output(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid receipt path')
        path = self.out / path
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlink study paths are not permitted')
        return path

    @contextmanager
    def locked(self):
        safe()
        self._output('.study.lock')
        self.out.mkdir(parents=True, exist_ok=True)
        lock = self.out / '.study.lock'
        if lock.is_symlink():
            raise ValueError('Lock may not be a symlink')
        with lock.open('a+') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another controller owns the study lock') from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _write(self, relative, value):
        path = self._output(relative)
        index_path = self._output('artifact_index.json')
        index = read_json(index_path) if index_path.exists() else {}
        if relative in index:
            raise ValueError('Receipt identity already registered')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        index[relative] = sha(path)
        temporary = self.out / ('artifact_index.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x') as stream:
            stream.write(json.dumps(index, sort_keys=True, allow_nan=False) + '\n')
        temporary.replace(index_path)
        return value

    def _read(self, relative):
        path = self._output(relative)
        index = read_json(self._output('artifact_index.json'))
        if index.get(relative) != sha(path):
            raise ValueError('Changed or unregistered receipt: ' + relative)
        return read_json(path)

    def _calendar(self):
        self._path(str(self.calendar_path.relative_to(self.root)))
        if self.root == ROOT and sha(self.calendar_path) != CALENDAR_SHA256:
            raise ValueError('Production acquisition declaration must retain its original hash')
        calendar = read_json(self.calendar_path)
        if (calendar.get('calendar_start_inclusive') != '2023-01-03'
                or calendar.get('calendar_end_exclusive') != '2025-01-01'
                or calendar.get('fixed_calendar_days') != 729
                or calendar.get('only_symbol') != 'CRASH500'
                or calendar.get('output_root') != SOURCE_ROOT
                or calendar.get('calendar_split_70_30') != {
                    'development_calendar_slots': 510, 'final_calendar_slots': 219,
                    'split_utc': '2024-05-27', 'quality_exclusions_do_not_compress_calendar': True}
                or calendar.get('excluded_whole_pilot_days_both_symbols') != list(PILOTS)
                or any(calendar.get('safety', {}).get(k) is not False for k in SAFETY_FIELDS)):
            raise ValueError('Acquisition calendar differs from the fixed replication')
        expected, ordinal = [], 0
        for day_index, day in enumerate(range(START, END, DAY_MS)):
            if utc_day(day) in PILOTS:
                continue
            prefix = 'MFR23_CRASH500_' + utc_day(day).replace('-', '')
            expected.append({'ordinal': ordinal, 'calendar_day_index': day_index,
                'start_utc': utc_day(day) + 'T00:00:00Z', 'start_ms': day,
                'end_exclusive_ms': day + DAY_MS, 'seconds': 86400, 'symbol': 'CRASH500',
                'prefix': prefix, 'partition': 'development_source' if day_index < 510 else 'sealed_final_source'})
            ordinal += 1
        if calendar.get('requests') != expected:
            raise ValueError('Source dates, prefixes or fixed partitions changed')
        source_root = self._path(calendar['output_root'])
        pins = calendar.get('source_pins')
        if not isinstance(pins, dict) or not all(name in pins for name in ACQUISITION_CODE):
            raise ValueError('Mandatory audited acquisition code pins are absent')
        for name, pinned in pins.items():
            if not isinstance(pinned, str) or not HASH.fullmatch(pinned) or sha(self._path(name)) != pinned:
                raise ValueError('Frozen acquisition dependency changed: ' + name)
        return calendar, source_root

    def _fixed_pins(self, calendar):
        pins = dict(calendar['source_pins'])
        for relative in OLD_DECLARATIONS:
            declaration = read_json(self._path(relative))
            pins[relative] = sha(self._path(relative))
            pins.update(declaration['source_pins'])
            pins[declaration['protocol_path']] = declaration['protocol_sha256']
        for relative in NEW_CODE:
            pins[relative] = sha(self._path(relative))
        for relative, pinned in pins.items():
            if sha(self._path(relative)) != pinned:
                raise ValueError('Old or new declared dependency changed: ' + relative)
        return pins

    def freeze(self, protocol_path, validation_path):
        with self.locked():
            calendar, _ = self._calendar()
            pins = self._fixed_pins(calendar)
            protocol = self._path(str(Path(protocol_path).absolute().relative_to(self.root)))
            validation = self._path(str(Path(validation_path).absolute().relative_to(self.root)))
            report = read_json(validation)
            if (report.get('status') != 'PASS' or report.get('approved_for_protocol_freeze') is not True
                    or report.get('actual_strategy_measurements') != 0 or report.get('code_pins') != pins
                    or report.get('protocol_sha256') != sha(protocol)
                    or report.get('calendar_declaration_sha256') != sha(self.calendar_path)):
                raise ValueError('Reviewed passing premeasurement evidence is required before freeze')
            spec = template() | {'alpha_protocol_frozen': True,
                'frozen_utc': datetime.now(timezone.utc).isoformat(), 'code_pins': pins,
                'calendar_path': str(self.calendar_path.relative_to(self.root)),
                'calendar_sha256': sha(self.calendar_path),
                'protocol_path': str(protocol.relative_to(self.root)), 'protocol_sha256': sha(protocol),
                'validation_path': str(validation.relative_to(self.root)), 'validation_sha256': sha(validation),
                'python': platform.python_version(), 'numpy': np.__version__}
            return self._write('declaration.json', spec)

    def declaration(self):
        safe()
        spec = self._read('declaration.json')
        expected = template()
        for key, value in expected.items():
            if key != 'alpha_protocol_frozen' and spec.get(key) != value:
                raise ValueError('Frozen rules/calendar/gates differ: ' + key)
        if spec.get('alpha_protocol_frozen') is not True:
            raise ValueError('Protocol has not been frozen')
        if spec['python'] != platform.python_version() or spec['numpy'] != np.__version__:
            raise ValueError('Frozen runtime changed')
        for key in ('calendar', 'protocol', 'validation'):
            path = self._path(spec[key + '_path'])
            if sha(path) != spec[key + '_sha256']:
                raise ValueError('Frozen ' + key + ' evidence changed')
        if self.calendar_path != self._path(spec['calendar_path']):
            raise ValueError('Controller calendar differs from frozen source calendar')
        for name, pinned in spec['code_pins'].items():
            if sha(self._path(name)) != pinned:
                raise ValueError('Frozen code dependency changed: ' + name)
        self._calendar()
        return spec, sha(self.out / 'declaration.json')

    def _read_state(self, batch):
        path = self._path(str((self.meta / STATE_NAMES[batch]).relative_to(self.root)))
        return read_json(path)

    def source_records(self, start_index, end_index, *, permit_final=False, hash_files=True):
        """Metadata completeness first; no raw hashing until every day is present."""
        if (type(start_index) is not int or type(end_index) is not int
                or not 0 <= start_index < end_index <= 729 or end_index > 510 and not permit_final):
            raise ValueError('Final source access requires rederived selection')
        if end_index > 510:
            self._guard_final(permit_final)
        calendar, source_root = self._calendar()
        wanted = [q for q in calendar['requests'] if start_index <= q['calendar_day_index'] < end_index]
        batches = sorted({0 if q['ordinal'] < 30 else 1 if q['ordinal'] < 180
                          else 2 if q['ordinal'] < 508 else 3 for q in wanted})
        records, producers = {}, {}
        for batch in batches:
            if batch == 3 and not permit_final:
                raise ValueError('Final metadata access prohibited')
            producer_path = self._path(str((self.meta / PRODUCER_NAMES[batch]).relative_to(self.root)))
            state = self._read_state(batch)
            producer = read_json(producer_path)
            producer_sha = sha(producer_path)
            if state.get('declaration_sha256') != producer_sha:
                raise ValueError('Source producer declaration differs from saved state')
            for key in ('calendar_start_inclusive', 'calendar_end_exclusive', 'fixed_calendar_days',
                        'excluded_whole_pilot_days_both_symbols', 'calendar_split_70_30',
                        'only_symbol', 'requests', 'output_root', 'safety'):
                if producer.get(key) != calendar[key]:
                    raise ValueError('Source producer calendar/requests/safety contract changed')
            left, right = BATCH_RANGES[batch]
            if producer.get('permitted_batch_ordinals') != list(range(left, right)):
                raise ValueError('Producer did not declare its exact permitted batch')
            if (any(state.get('safety', {}).get(k) is not False for k in SAFETY_FIELDS)
                    or state.get('strategy_measurements') != 0
                    or state.get('final_strategy_metrics_opened') is not False
                    or state.get('orders') is not False):
                raise ValueError('Source state must be source-only with all safety flags false')
            producer_pins = producer.get('source_pins')
            if not isinstance(producer_pins, dict):
                raise ValueError('Source producer requires explicit audited source pins')
            mandatory = {name: pinned for name, pinned in calendar['source_pins'].items()
                         if Path(name).suffix in ('.py', '.mq5')}
            if any(producer_pins.get(name) != pinned for name, pinned in mandatory.items()):
                raise ValueError('Producer acquisition code differs from the canonical calendar')
            # Producer declaration preserves preparation snapshots. A live state
            # SHA is never a dependency: selected record fingerprints bind data.
            producer_code = {name: pinned for name, pinned in producer_pins.items()
                             if Path(name).suffix in ('.py', '.mq5')}
            for name, pinned in producer_pins.items():
                if not isinstance(pinned, str) or not HASH.fullmatch(pinned):
                    raise ValueError('Malformed source producer pin')
                self._path(name)
            for name, pinned in producer_code.items():
                if sha(self._path(name)) != pinned:
                    raise ValueError('Source producer dependency changed')
            producers[STATE_NAMES[batch]] = {'path': str(producer_path.relative_to(self.root)),
                'sha256': producer_sha, 'source_code_pins': producer_code}
            for record in state.get('records', []):
                ordinal = record.get('ordinal')
                if type(ordinal) is not int or ordinal not in range(left, right):
                    raise ValueError('Source state record lies outside producer permitted batch')
                if ordinal in records:
                    raise ValueError('Duplicate source ordinal')
                if any(q['ordinal'] == ordinal for q in wanted):
                    records[ordinal] = record
        bindings, file_paths = [], set()
        for request in wanted:
            record = records.get(request['ordinal'])
            if record is None or any(record.get(k) != v for k, v in request.items()):
                raise ValueError('Required source day missing or date/prefix/partition changed')
            if (record.get('verification_status') != 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
                    or type(record.get('rows')) is not int or record['rows'] <= 0):
                raise ValueError('Required source day failed or is empty; no substitution')
            folder = source_root / request['partition'] / request['prefix']
            paths = {kind: str((folder / (request['prefix'] + suffix)).relative_to(self.root))
                     for kind, suffix in [('receipt', '_receipt.tsv'), ('raw', '_ticks.tsv')]}
            selected_pins = {}
            for kind, relative in paths.items():
                pinned = record.get('files', {}).get(relative)
                if not isinstance(pinned, str) or not HASH.fullmatch(pinned) or relative in file_paths:
                    raise ValueError('Missing/invalid/colliding source file pin')
                file_paths.add(relative)
                selected_pins[relative] = pinned
            batch = 0 if request['ordinal'] < 30 else 1 if request['ordinal'] < 180 else 2 if request['ordinal'] < 508 else 3
            bindings.append({'request': request, 'selected_record_sha256': fingerprint(record),
                             'producer': producers[STATE_NAMES[batch]], 'paths': paths, 'source_files': selected_pins})
        # Failure/missing any declared day above prevents even this raw-byte pass.
        if hash_files:
            for binding in bindings:
                for relative, pinned in binding['source_files'].items():
                    if sha(self._path(relative)) != pinned:
                        raise ValueError('Selected source file changed')
        return bindings

    def preflight(self, stage):
        """Read-only development metadata availability; no freeze, raw or alpha."""
        safe()
        if stage not in DEVELOPMENT:
            raise ValueError('Read-only preflight is restricted to development stages')
        try:
            records = self.source_records(0, STAGES[stage][1], hash_files=False)
            return {'stage': stage, 'source_metadata_available': True,
                    'selected_records': len(records), 'records_sha256': fingerprint(records),
                    'raw_files_verified': False, 'actual_strategy_measurements': 0, **SAFETY}
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {'stage': stage, 'source_metadata_available': False, 'error': str(error),
                    'raw_files_verified': False, 'actual_strategy_measurements': 0, **SAFETY}

    def _source_binding(self, stage, records):
        value = {'stage': stage, 'records': records, 'records_sha256': fingerprint(records),
                 'state_file_sha_dependency': False, 'producers_pinned_before_measurement': True}
        name = 'source_bindings/' + stage + '.json'
        if (self.out / name).exists():
            if self._read(name) != value:
                raise ValueError('Immutable stage source binding changed')
        else:
            self._write(name, value)
        return value

    def _loader(self, records):
        by_day = {r['request']['start_ms']: r for r in records}
        if len(by_day) != len(records):
            raise ValueError('Duplicate selected source date')

        def selected(day):
            if utc_day(day) in PILOTS:
                return QuoteTape([])
            if day not in by_day:
                raise ValueError('Day outside explicitly bound source prefix')
            row = by_day[day]
            request = row['request']
            loaded = load_day(self._path(row['paths']['receipt']),
                row['source_files'][row['paths']['receipt']], row['source_files'][row['paths']['raw']],
                'CRASH500', request['start_ms'], request['end_exclusive_ms'])
            return loaded.tape
        return selected

    def _fit(self, train_end_index, records, declaration_sha):
        rows = [r for r in records if r['request']['calendar_day_index'] < train_end_index]
        binding = {'declaration_sha256': declaration_sha, 'train_start_ms': START,
                   'train_end_exclusive_ms': START + train_end_index * DAY_MS,
                   'records_sha256': fingerprint(rows), 'records': rows}
        name = f'fits/TRAIN0_{train_end_index}.json'
        if (self.out / name).exists():
            cached = self._read(name)
            if cached['sqlite_path'] != f'fits/TRAIN0_{train_end_index}.sqlite':
                raise ValueError('Cached TRAIN histogram path differs from its exact prefix')
            if cached['binding'] != binding or sha(self._output(cached['sqlite_path'])) != cached['sqlite_sha256']:
                raise ValueError('Cached TRAIN fit differs from exact source prefix')
            return cached
        database = f'fits/TRAIN0_{train_end_index}.sqlite'
        (self.out / 'fits').mkdir(exist_ok=True)
        fitted = fit_positive_spread_prefixes(START, binding['train_end_exclusive_ms'],
            [binding['train_end_exclusive_ms']], self._loader(rows), self._output(database))
        return self._write(name, {'binding': binding, 'fit': asdict(fitted[0]),
            'sqlite_path': database, 'sqlite_sha256': sha(self.out / database),
            'uses_strategy_payoff': False, 'uses_validation_or_final': False, **SAFETY})

    def _case_name(self, stage, policy, scenario):
        return f'cases/{stage}_{policy}_{scenario}.json'

    def _case_context(self, stage, declaration_sha, records=None, *, permit_final=False):
        if stage == 'FINAL':
            self._guard_final(permit_final, declaration_sha)
        lo, hi, train = STAGES[stage]
        if records is None:
            records = self.source_records(0, hi, permit_final=permit_final)
        source_binding = self._read(f'source_bindings/{stage}.json')
        expected_source = {'stage': stage, 'records': records, 'records_sha256': fingerprint(records),
            'state_file_sha_dependency': False, 'producers_pinned_before_measurement': True}
        if source_binding != expected_source:
            raise ValueError('Current selected source record lineage changed')
        fit_path = f'fits/TRAIN0_{train}.json'
        fit = self._read(fit_path)
        fit_records = [r for r in records if r['request']['calendar_day_index'] < train]
        expected_fit = {'declaration_sha256': declaration_sha, 'train_start_ms': START,
            'train_end_exclusive_ms': START + train * DAY_MS, 'records_sha256': fingerprint(fit_records),
            'records': fit_records}
        if fit['binding'] != expected_fit or fit['sqlite_path'] != f'fits/TRAIN0_{train}.sqlite':
            raise ValueError('Cached fit does not bind the exact declared TRAIN prefix')
        scalar = fit['fit']
        if (scalar['train_start_ms'] != START or scalar['train_end_exclusive_ms'] != START + train * DAY_MS
                or not math.isfinite(scalar['U_scaled']) or scalar['U_scaled'] <= 0):
            raise ValueError('Cached spread fit scalar/bounds are invalid')
        if sha(self._output(fit['sqlite_path'])) != fit['sqlite_sha256']:
            raise ValueError('Cached fit histogram changed')
        start, end = START + lo * DAY_MS, START + hi * DAY_MS
        schedule_value = {'declaration_sha256': declaration_sha, 'stage': stage,
            'slots': [asdict(s) for s in sparse_schedule(start, end, declaration_sha)]}
        if self._read(f'schedules/{stage}.json') != schedule_value:
            raise ValueError('Cached fixed sparse schedule changed')
        return {'declaration_sha256': declaration_sha, 'stage': stage, 'symbol': 'CRASH500',
            'U_scaled': scalar['U_scaled'], 'fit_path': fit_path, 'fit_sha256': sha(self._output(fit_path)),
            'schedule_sha256': fingerprint(schedule_value),
            'records': [r for r in records if lo <= r['request']['calendar_day_index'] < hi],
            'stage_start_ms': start, 'stage_end_exclusive_ms': end, 'valid_day_mask': mask_for(start, end)}

    def _cached_case(self, stage, policy, scenario, binding=None, *, context=None, permit_final=False):
        if stage not in STAGES or policy not in (*POLICIES, *CONTROLS) or scenario not in SCENARIOS:
            raise ValueError('Invalid cached case identity')
        if stage == 'FINAL':
            self._guard_final(permit_final)
        name = self._case_name(stage, policy, scenario)
        value = self._read(name)
        expected_fit_path = f'fits/TRAIN0_{STAGES[stage][2]}.json'
        expected_ledger = f'ledgers/{stage}_{policy}_{scenario}.json.gz'
        # Validate fixed names before opening even a reindexed forged cache path.
        if value.get('ledger_path') != expected_ledger or value.get('binding', {}).get('fit_path') != expected_fit_path:
            raise ValueError('Cached evidence path differs from its exact case identity')
        if (value['binding']['stage'], value['binding']['policy'], value['binding']['scenario']) != (stage, policy, scenario):
            raise ValueError('Cached case identity changed')
        if context is None:
            _, declaration_sha = self.declaration()
            context = self._case_context(stage, declaration_sha, permit_final=permit_final)
        expected_binding = context | {'policy': policy, 'scenario': scenario}
        if value['binding'] != expected_binding:
            raise ValueError('Cached case source/fit/calendar differs from current exact lineage')
        if binding is not None and value['binding'] != binding:
            raise ValueError('Cached case source/fit/calendar differs')
        if sha(self._output(value['ledger_path'])) != value['ledger_sha256']:
            raise ValueError('Cached ledger changed')
        return value

    def _calculate_stage(self, stage, policies, spec, declaration_sha, *, permit_final=False):
        if stage == 'FINAL':
            self._guard_final(permit_final, declaration_sha)
        lo, hi, train = STAGES[stage]
        all_rows = self.source_records(0, hi, permit_final=permit_final)
        self._source_binding(stage, all_rows)
        fit = self._fit(train, all_rows, declaration_sha)
        fit_path = f'fits/TRAIN0_{train}.json'
        stage_rows = [r for r in all_rows if lo <= r['request']['calendar_day_index'] < hi]
        start, end = START + lo * DAY_MS, START + hi * DAY_MS
        schedule = [asdict(s) for s in sparse_schedule(start, end, declaration_sha)]
        schedule_path = f'schedules/{stage}.json'
        schedule_value = {'declaration_sha256': declaration_sha, 'stage': stage, 'slots': schedule}
        if (self.out / schedule_path).exists():
            if self._read(schedule_path) != schedule_value:
                raise ValueError('Fixed sparse schedule changed')
        else:
            self._write(schedule_path, schedule_value)
        context = self._case_context(stage, declaration_sha, all_rows, permit_final=permit_final)
        unit = fit['fit']['U_scaled']
        loader = self._loader(stage_rows)
        prepared = None
        store = ShardedQuoteStore(start, end, loader)
        calculated = reused = 0
        results = {}
        for policy in (*policies, *CONTROLS):
            for scenario in SCENARIOS:
                binding = {'declaration_sha256': declaration_sha, 'stage': stage, 'policy': policy,
                    'scenario': scenario, 'symbol': 'CRASH500', 'U_scaled': unit,
                    'fit_path': fit_path, 'fit_sha256': sha(self.out / fit_path),
                    'schedule_sha256': fingerprint(schedule_value), 'records': stage_rows,
                    'stage_start_ms': start, 'stage_end_exclusive_ms': end,
                    'valid_day_mask': mask_for(start, end)}
                name = self._case_name(stage, policy, scenario)
                if (self.out / name).exists():
                    results[(policy, scenario)] = self._cached_case(stage, policy, scenario, binding,
                        context=context, permit_final=permit_final)
                    reused += 1
                    continue
                if prepared is None:
                    def source():
                        for day in range(start, end, DAY_MS):
                            yield from loader(day).quotes
                    prepared = build_streamed_frames(source(), start, end)
                latency, slip = (250, .25) if scenario == 'BASE' else (2000, .5)
                config = ExecutionConfig(latency, latency, 5000, slip * unit)
                frames, index = prepared.frames, prepared.issuance_index
                if policy == 'SPARSE_CLOCK':
                    signals = generate_sparse_control(frames, index, unit, -1, start, end, declaration_sha)
                    identity = SPARSE_ID
                elif policy == 'DENSE_CLOCK':
                    signals = clock_signals(frames, index, unit, -1, start, end)
                    identity = 'CLOCK_00_30'
                else:
                    signals = generate_signals(frames, policy_id=policy, native_direction=-1,
                        spread_unit=unit, issuance_index=index)
                    identity = policy
                result = evaluate_streamed_signals(store,
                    embargoed_signals(signals, start, end, latency), start, end, unit, config,
                    policy_id=identity, baseline=policy in CONTROLS)
                ledger = result.pop('ledger')
                ledger_path = f'ledgers/{stage}_{policy}_{scenario}.json.gz'
                (self.out / 'ledgers').mkdir(exist_ok=True)
                with self._output(ledger_path).open('xb') as raw:
                    with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
                        zipped.write(canonical(ledger))
                for day in result['daily']:
                    if day['date_utc'] in PILOTS:
                        if day['completed'] or day['unknown']:
                            raise ValueError('Exposure or UNKNOWN was attributed to an excluded pilot')
                        for key in ('sum_R', 'sum_quote', 'observed_sum_R', 'observed_sum_quote',
                                    'observed_gain_R', 'observed_loss_R'):
                            day[key] = None
                        day['excluded_pilot_day'] = True
                value = result | {'binding': binding, 'ledger_path': ledger_path,
                    'ledger_sha256': sha(self.out / ledger_path),
                    'declared_calendar_source_available': True,
                    'availability_scope': 'ALL_DECLARED_NONPILOT_SHARDS_VALIDATED_NOT_BROKER_COMPLETENESS',
                    'cash_net': None, 'profit_qualified': False, **SAFETY}
                results[(policy, scenario)] = self._write(name, value)
                calculated += 1
        return {'stage': stage, 'calculated_cases': calculated, 'reused_cases': reused,
                'case_count': len(results), 'source_binding_sha256': fingerprint(all_rows)}

    def stage(self, stage):
        if stage not in DEVELOPMENT:
            raise ValueError('Use final() for guarded final access')
        with self.locked():
            spec, declared = self.declaration()
            return self._calculate_stage(stage, POLICIES, spec, declared)

    def derive_selection(self, declaration_sha):
        rows = self.source_records(0, 510)
        contexts = {stage: self._case_context(stage, declaration_sha,
            [r for r in rows if r['request']['calendar_day_index'] < STAGES[stage][1]])
            for stage in DEVELOPMENT}
        gates, evidence = {}, {}
        for policy in POLICIES:
            both = {}
            for scenario in SCENARIOS:
                candidate, sparse = [], []
                for stage in DEVELOPMENT:
                    c = self._cached_case(stage, policy, scenario, context=contexts[stage])
                    s = self._cached_case(stage, 'SPARSE_CLOCK', scenario, context=contexts[stage])
                    if c['binding']['declaration_sha256'] != declaration_sha or s['binding']['declaration_sha256'] != declaration_sha:
                        raise ValueError('Development case belongs to a different declaration')
                    candidate.append(c['summary'])
                    sparse.append(s['summary'])
                    evidence[self._case_name(stage, policy, scenario)] = sha(self.out / self._case_name(stage, policy, scenario))
                    evidence[self._case_name(stage, 'SPARSE_CLOCK', scenario)] = sha(self.out / self._case_name(stage, 'SPARSE_CLOCK', scenario))
                known = all(v['sum_R'] is not None and v['unknown'] == 0 for v in (*candidate, *sparse))
                excess = [c['sum_R'] - s['sum_R'] for c, s in zip(candidate, sparse)] if known else [None] * 4
                both[scenario] = {
                    'known_no_UNKNOWN': known,
                    'candidate_completed_at_least_100': sum(c['completed'] for c in candidate) >= 100,
                    'candidate_VALIDATION_completed_at_least_20': candidate[-1]['completed'] >= 20,
                    'sparse_completed_at_least_100': sum(s['completed'] for s in sparse) >= 100,
                    'sparse_VALIDATION_completed_at_least_30': sparse[-1]['completed'] >= 30,
                    'candidate_VALIDATION_positive': known and candidate[-1]['sum_R'] > 0,
                    'candidate_two_of_three_WF_positive': known and sum(c['sum_R'] > 0 for c in candidate[:3]) >= 2,
                    'excess_VALIDATION_positive': known and excess[-1] > 0,
                    'excess_two_of_three_WF_positive': known and sum(v > 0 for v in excess[:3]) >= 2,
                }
            gates[policy] = {'scenarios': both, 'eligible': all(all(v.values()) for v in both.values())}
        return {'declaration_sha256': declaration_sha, 'gates': gates,
                'eligible': [p for p in POLICIES if gates[p]['eligible']], 'development_evidence': evidence,
                'dense_not_primary_selection_gate': True, 'cash_net': None, 'profit_qualified': False, **SAFETY}

    def select(self):
        with self.locked():
            _, declared = self.declaration()
            result = self.derive_selection(declared)
            if (self.out / 'selection.json').exists():
                if self._read('selection.json') != result:
                    raise ValueError('Frozen selection differs from current development evidence')
                return result
            return self._write('selection.json', result)

    def _final_selection(self, declaration_sha):
        saved = self._read('selection.json')
        recomputed = self.derive_selection(declaration_sha)
        if saved != recomputed:
            raise ValueError('Selection must match rederived development before final access')
        return saved

    def _statistics(self):
        from independent_multiframe_research import replication_statistics
        return replication_statistics

    def _daily_case(self, statistics, case):
        rows = case['daily']
        return statistics.DailyCase(
            [d['sum_R'] for d in rows], [d['completed'] for d in rows], [d['unknown'] for d in rows],
            case['declared_calendar_source_available'], case['summary']['primary_measurement_usable'],
            [d['observed_gain_R'] for d in rows], [d['observed_loss_R'] for d in rows])

    def final(self):
        with self.locked():
            spec, declared = self.declaration()
            selected = self._final_selection(declared)  # BEFORE batch4 lookup or final cache access.
            if selected['eligible']:
                self._final_authorization = (declared, fingerprint(selected))
            try:
                return self._finish_final(spec, declared, selected)
            finally:
                self._final_authorization = None

    def _finish_final(self, spec, declared, selected):
        statistics = self._statistics()
        analyses = {policy: None for policy in POLICIES}
        case_count = 0
        if selected['eligible']:
            measured = self._calculate_stage('FINAL', tuple(selected['eligible']), spec, declared, permit_final=True)
            case_count = measured['case_count']
            context = self._case_context('FINAL', declared, permit_final=True)
            lo, hi, _ = STAGES['FINAL']
            start, end = START + lo * DAY_MS, START + hi * DAY_MS
            calendar = statistics.Calendar([utc_day(day) for day in range(start, end, DAY_MS)], mask_for(start, end))
            for policy in selected['eligible']:
                scenarios = {scenario: statistics.ScenarioRecords(*[
                    self._daily_case(statistics, self._cached_case('FINAL', role, scenario,
                        context=context, permit_final=True))
                    for role in (policy, 'SPARSE_CLOCK', 'DENSE_CLOCK')]) for scenario in SCENARIOS}
                analyses[policy] = statistics.analyze_policy(calendar, scenarios, policy_id=policy,
                    declaration_sha256=declared, draws=spec['bootstrap_draws'])
        holm = statistics.holm33(statistics.claims_from_analyses(analyses))
        gates = {}
        for policy, analysis in analyses.items():
            if analysis is None:
                continue
            gate = statistics.conditional_statistical_gate(analysis, holm)
            dd = all(self._cached_case('FINAL', policy, scenario, context=context, permit_final=True)['summary']['marked_drawdown_pct'] is not None
                     and self._cached_case('FINAL', policy, scenario, context=context, permit_final=True)['summary']['marked_drawdown_pct'] <= spec['marked_drawdown_max_pct']
                     for scenario in SCENARIOS)
            gates[policy] = {'statistics': gate, 'both_marked_drawdowns_pass': dd,
                'conditional_partial_pass': gate['conditional_statistical_pass'] and dd,
                'source_reproduction_audit_complete': False, 'cash_qualified': False}
        result = {'declaration_sha256': declared, 'selection_sha256': sha(self.out / 'selection.json'),
            'eligible': selected['eligible'], 'final_cases': case_count, 'analyses': analyses, 'holm33': holm,
            'actual_analyze_policy_calls': sum(v is not None for v in analyses.values()),
            'zero_eligible_means_no_final_source_metadata_or_raw_access': not selected['eligible'],
            'holm_placeholders_are_not_actual_tests': True, 'gates': gates,
            'cash_net': None, 'costs_verified': False, 'execution_verified': False,
            'profit_qualified': False, 'goal_profit_achieved': False, **SAFETY}
        if (self.out / 'final_summary.json').exists():
            if self._read('final_summary.json') != result:
                raise ValueError('Immutable final summary differs; no reinterpretation')
            return result
        return self._write('final_summary.json', result)

    def summary(self):
        with self.locked():
            _, declared = self.declaration()
            self._final_selection(declared)
            final = self._read('final_summary.json')
            if final['declaration_sha256'] != declared:
                raise ValueError('Final summary declaration differs')
            return {'status': 'COMPLETED_CONDITIONAL_STUDY_ONLY', 'final': final,
                'cash_net': None, 'profit_qualified': False, 'goal_profit_achieved': False, **SAFETY}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('template', 'preflight', 'freeze', *DEVELOPMENT, 'select', 'final', 'summary'))
    parser.add_argument('--study-dir', type=Path, default=OUT)
    parser.add_argument('--source-meta-dir', type=Path, default=META)
    parser.add_argument('--calendar', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--stage', choices=DEVELOPMENT, default='WF1')
    args = parser.parse_args(argv)
    controller = Controller(study_dir=args.study_dir, source_meta_dir=args.source_meta_dir, calendar_path=args.calendar)
    if args.command == 'template':
        safe()
        value = template()
    elif args.command == 'preflight':
        value = controller.preflight(args.stage)
    elif args.command == 'freeze':
        if args.protocol is None or args.validation is None:
            parser.error('freeze requires explicit --protocol and --validation')
        value = controller.freeze(args.protocol, args.validation)
    elif args.command in DEVELOPMENT:
        value = controller.stage(args.command)
    else:
        value = getattr(controller, args.command)()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
