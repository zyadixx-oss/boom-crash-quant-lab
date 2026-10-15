"""Guarded storage plan for our pinned first30 source RAW duplicates only.

The default command only hashes/stats files. Future ``apply`` replaces one
FILE_COMMON directory entry atomically with a DATA hardlink. Both names and
bytes remain, but independent copies and independent file attributes are lost.
``restore`` atomically makes a separate byte-for-byte copy at FILE_COMMON.
Neither operation permits symlinks, other prefixes, changed evidence or a live
collector/terminal. No trading, compression, cache purge or RAW unlink exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import uuid

ROOT = Path(__file__).absolute().parents[1]
DATA_REL = 'data/independent_profit_ticks_20261013/source_diagnostics/replication_source_20261014'
DECL_REL = 'work/multiframe_replication_source_20261014/calendar_declaration.json'
STATE_REL = 'work/multiframe_replication_source_20261014/state.json'
WORK_REL = 'work/source_storage_guard_20261014'
DECL_SHA = '917b08901b48073ad66e79aa1c3743078e21e2bbb5f9fefe35b4a142f470f84d'
STATE_SHA = 'fc4633400dbc333e68ca0a5bbfb9aa0f7e486e4c9aac67c02732fa344ed9bb38'
PREFIXES = tuple('MFR23_CRASH500_'+(date(2023, 1, 3)+timedelta(days=i)).strftime('%Y%m%d') for i in range(30))
SAFETY = ('LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES',
          'live_trading', 'ready_for_live', 'live_allowed', 'opened_trades')
WARNINGS = [
    'Both filenames/content SHA are preserved; FILE_COMMON and DATA will share one inode.',
    'This removes independent-copy protection: later writes, chmod and corruption affect both names.',
    'Future apply protects the shared inode with mode0444; an owner/root can still chmod it, so this is not immutable storage.',
    'Reversal restores separate content copies, not the original FILE_COMMON inode/ctime.',
    'Allocated-block reclaim is a potential upper bound; APFS shared extents, snapshots and open descriptors can reduce it.',
    'Namespace replacement is atomic; storage persistence across power loss is not guaranteed.',
]


class GuardError(ValueError):
    pass


@dataclass(frozen=True)
class Contract:
    root: Path
    common_root: Path
    declaration_sha256: str = DECL_SHA
    state_sha256: str = STATE_SHA


def _check(value, message):
    if not value:
        raise GuardError(message)


def _hash_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(fd, 1024*1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def _snapshot(fd):
    s = os.fstat(fd)
    _check(stat.S_ISREG(s.st_mode), 'Expected regular file')
    return {'dev': s.st_dev, 'ino': s.st_ino, 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'mode': stat.S_IMODE(s.st_mode), 'uid': s.st_uid, 'gid': s.st_gid,
            'nlink': s.st_nlink, 'allocated_bytes': s.st_blocks*512}


@contextmanager
def _directory(path):
    """Resolve every ancestor with dir_fd + O_NOFOLLOW, without symlink traversal."""
    path = Path(path)
    _check(path.is_absolute() and '..' not in path.parts, 'Absolute canonical directory required')
    _check(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_DIRECTORY'), 'Safe dir_fd platform required')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for part in path.parts[1:]:
            new = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = new
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _file(path):
    path = Path(path)
    with _directory(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            _snapshot(fd)
            yield fd, parent, path.name
        finally:
            os.close(fd)


def _pinned_bytes(path, expected, *, maximum=2**22):
    _check(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'Canonical SHA256 required')
    with _file(path) as (fd, _, __):
        before = _snapshot(fd)
        _check(before['size'] <= maximum, 'Metadata/code file exceeds bound')
        _check(_hash_fd(fd) == expected, 'Pinned metadata/code SHA changed')
        os.lseek(fd, 0, os.SEEK_SET)
        value = os.read(fd, maximum+1)
        _check(_snapshot(fd) == before, 'Metadata/code changed while reading')
        return value


def runtime_snapshot(owner_pid):
    """Inspect executable names first, never expose process arguments or identities."""
    native, writers, owner_alive = [], [], False
    rows = subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True).splitlines()
    for row in rows:
        pid, executable = row.strip().split(None, 1)
        pid, name = int(pid), Path(executable).name
        owner_alive |= pid == owner_pid
        if name not in ('wine64-preloader', 'wine-preloader', 'wine64', 'wine', 'terminal64.exe') and not name.lower().startswith('python'):
            continue
        args = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True).stdout
        if name in ('wine64-preloader', 'wine-preloader', 'wine64', 'wine', 'terminal64.exe') and 'terminal64.exe' in args:
            native.append(pid)
        if any(fragment in args for fragment in (
                'work/independent_profit_pilot_20261013/run_dataset_probe.py',
                'work/multiframe_replication_source_20261014/collect_first_batch.py')):
            writers.append(pid)
    return {'native_terminal_pids': native, 'source_writer_pids': writers,
            'completed_batch_owner_pid_alive': owner_alive}


def _idle(owner_pid):
    observed = runtime_snapshot(owner_pid)
    _check(not any(observed.values()), 'Actual terminal/source writer/batch owner remains alive')
    return observed


def _manifest(contract):
    _check(contract.root.is_absolute() and contract.common_root.is_absolute(), 'Absolute trusted roots required')
    for key in SAFETY:
        _check(os.environ.get(key, 'false').strip().lower() == 'false', 'Unsafe environment flag: '+key)
    declaration = json.loads(_pinned_bytes(contract.root/DECL_REL, contract.declaration_sha256))
    state = json.loads(_pinned_bytes(contract.root/STATE_REL, contract.state_sha256))
    _check(declaration['only_symbol'] == 'CRASH500' and declaration['output_root'] == DATA_REL,
           'Unexpected source symbol/root')
    _check(declaration['permitted_batch_ordinals'] == list(range(30)) and len(declaration['requests']) == 724,
           'Only the exact first30 declared batch is allowed')
    for obj in (declaration, state):
        _check(set(obj['safety']) == set(SAFETY) and all(obj['safety'][key] is False for key in SAFETY),
               'All eight false flags required')
    _check(state['declaration_sha256'] == contract.declaration_sha256
           and state['status'] == 'COMPLETED_FIRST_30_SOURCE_ONLY'
           and len(state['records']) == 30 and state['strategy_measurements'] == 0
           and state['final_strategy_metrics_opened'] is False and state['orders'] is False
           and state.get('pending') is None and state.get('actual_terminal_pids') == [],
           'Pinned source batch is not completed/idle/source-only')
    for relative, expected in declaration['source_pins'].items():
        _check(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
               'Invalid source-code pin path')
        _pinned_bytes(contract.root/relative, expected)
    pairs = []
    for ordinal, (request, record, prefix) in enumerate(zip(declaration['requests'], state['records'], PREFIXES)):
        _check(request['ordinal'] == record['ordinal'] == ordinal and request['prefix'] == record['prefix'] == prefix,
               'Unexpected ordinal or prefix')
        _check(all(record.get(k) == v for k, v in request.items()), 'Record differs from exact declared request')
        _check(request['symbol'] == 'CRASH500' and request['partition'] == 'development_source'
               and request['seconds'] == 86400 and record['verification_status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
               and record['actual_terminal_pids_after_capture'] == [] and record['observer_exit'] == 0,
               'Source capture was not verified/completed')
        folder = contract.root/DATA_REL/'development_source'/prefix
        raw = folder/(prefix+'_ticks.tsv')
        receipt = folder/(prefix+'_receipt.tsv')
        verified = folder/'source_verification.json'
        for path in (raw, receipt, verified):
            _check(str(path.relative_to(contract.root)) in record['files'], 'Exact source file pin missing')
        raw_sha = record['files'][str(raw.relative_to(contract.root))]
        receipt_sha = record['files'][str(receipt.relative_to(contract.root))]
        _pinned_bytes(receipt, receipt_sha, maximum=65536)
        verification = json.loads(_pinned_bytes(verified, record['files'][str(verified.relative_to(contract.root))]))
        _check(verification['status'] == 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN'
               and verification['raw_sha256'] == raw_sha and verification['receipt_sha256'] == receipt_sha
               and verification['rows'] == record['rows'] and not verification['errors'],
               'Pinned RAW completeness verification differs')
        pairs.append({'prefix': prefix, 'data': raw, 'common': contract.common_root/raw.name,
                      'sha256': raw_sha, 'bytes': record['raw_bytes']})
    return state, pairs


@contextmanager
def guarded(contract):
    state, pairs = _manifest(contract)
    observed = _idle(state['owner_pid'])
    work = contract.root/WORK_REL
    work.mkdir(parents=True, exist_ok=True)
    # Existing collector lock is opened read-only; neither its content nor the
    # acquisition state is modified. The new guard lock is ours alone.
    with _file(contract.root/DATA_REL/'.collection.lock') as (source_lock, _, __):
        with _directory(work) as work_fd:
            guard_fd = os.open('.storage_guard.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=work_fd)
        with os.fdopen(guard_fd, 'r+') as guard_lock:
            _snapshot(guard_lock.fileno())
            try:
                fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(guard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise GuardError('Source/guard lock is held') from error
            _idle(state['owner_pid'])
            yield state, pairs, observed
            _manifest(contract)
            _idle(state['owner_pid'])


def _same_device(left, right):
    _check(left['dev'] == right['dev'], 'Cross-filesystem hardlinks are forbidden')


def _pair_info(pair):
    with _file(pair['data']) as (data, data_dir, __), _file(pair['common']) as (common, common_dir, ____):
        before_data, before_common = _snapshot(data), _snapshot(common)
        _same_device(before_data, before_common)
        _check(os.fstat(data_dir).st_dev == os.fstat(common_dir).st_dev == before_data['dev'],
               'File and parent directories must be on the same filesystem')
        _check(before_data['size'] == before_common['size'] == pair['bytes'], 'RAW size differs')
        _check(before_data['uid'] == before_common['uid'] == os.getuid()
               and before_data['gid'] == before_common['gid'] and before_data['mode'] == before_common['mode'],
               'Owner/group/mode differ; shared attributes would change access')
        _check(_hash_fd(data) == _hash_fd(common) == pair['sha256'], 'RAW SHA differs')
        _check(_snapshot(data) == before_data and _snapshot(common) == before_common, 'RAW changed during hashing')
        shared = (before_data['dev'], before_data['ino']) == (before_common['dev'], before_common['ino'])
        _check(shared or before_data['nlink'] == before_common['nlink'] == 1, 'Unexpected other hardlinks')
        _check(not shared or before_data['mode'] == before_common['mode'] == 0o444,
               'Already shared RAW inode is not protected read-only')
        return {**{k: str(v) if isinstance(v, Path) else v for k, v in pair.items()},
                'data_stat': before_data, 'common_stat': before_common,
                'already_shared_inode': shared,
                'potential_reclaim_bytes_upper_bound': 0 if shared else before_common['allocated_bytes']}


def dry_run(contract):
    with guarded(contract) as (state, pairs, observed):
        entries = [_pair_info(pair) for pair in pairs]
        return {'schema_version': 1, 'status': 'DRY_RUN_ONLY_NO_DATA_OR_COMMON_MUTATIONS',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'declaration_sha256': contract.declaration_sha256, 'state_sha256': contract.state_sha256,
                'guard_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'common_root': str(contract.common_root), 'entries': entries,
                'allowed_prefixes': list(PREFIXES), 'runtime_observation': observed,
                'potential_reclaim_bytes_upper_bound': sum(e['potential_reclaim_bytes_upper_bound'] for e in entries),
                'logical_duplicate_raw_bytes': sum(e['bytes'] for e in entries if not e['already_shared_inode']),
                'actual_reclaim_measured': False, 'physical_copy_count_after_future_apply': 1,
                'independent_copy_protection_after_future_apply': False,
                'warnings': WARNINGS, 'safety': {key: False for key in SAFETY}}


def _unchanged_name(parent, name, snapshot):
    s = os.stat(name, dir_fd=parent, follow_symlinks=False)
    _check(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
           == tuple(snapshot[k] for k in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')),
           'Directory entry changed before atomic replacement')


def _rewrite(pair, expected, *, restore=False, observe=lambda: None, journal=lambda phase: None, preflight_only=False):
    """Future mutation primitive, exercised on toy files only during this task."""
    with _file(pair['data']) as (data, data_dir, data_name), _file(pair['common']) as (common, common_dir, common_name):
        ds, cs = _snapshot(data), _snapshot(common)
        _same_device(ds, cs)
        _check(os.fstat(data_dir).st_dev == os.fstat(common_dir).st_dev == ds['dev'],
               'File and parent directories must be on the same filesystem')
        _check(_hash_fd(data) == _hash_fd(common) == pair['sha256'] and ds['size'] == cs['size'] == pair['bytes'],
               'RAW changed before replacement')
        shared = (ds['dev'], ds['ino']) == (cs['dev'], cs['ino'])
        _check((ds['dev'], ds['ino']) == (expected['data_stat']['dev'], expected['data_stat']['ino']),
               'Original DATA inode changed')
        if restore:
            for current, original in ((ds, expected['data_stat']), (cs, expected['common_stat'])):
                _check(current['uid'] == original['uid'] and current['gid'] == original['gid']
                       and current['mode'] in (0o444, original['mode']), 'Owner/group/mode changed outside the reviewed operation')
            if not shared and ds['mode'] == expected['data_stat']['mode'] and cs['mode'] == expected['common_stat']['mode']:
                return {'prefix': pair['prefix'], 'status': 'ALREADY_INDEPENDENT_NO_CHANGE'}
        else:
            _check(ds == expected['data_stat'] and cs == expected['common_stat'], 'Reviewed plan stat changed')
            if shared:
                return {'prefix': pair['prefix'], 'status': 'ALREADY_SHARED_NO_CHANGE'}
        _unchanged_name(data_dir, data_name, ds)
        _unchanged_name(common_dir, common_name, cs)
        if preflight_only:
            return {'prefix': pair['prefix'], 'status': 'PREFLIGHT_ONLY_NO_CHANGE'}
        temporary = '.source_storage_guard_'+pair['prefix']+'_'+uuid.uuid4().hex+'.tmp'
        committed = created = False
        try:
            observe()
            journal('PREPARE_VERIFIED')
            if restore:
                fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, cs['mode'], dir_fd=common_dir)
                created = True
                try:
                    os.lseek(data, 0, os.SEEK_SET)
                    for chunk in iter(lambda: os.read(data, 1024*1024), b''):
                        view = memoryview(chunk)
                        while view:
                            written = os.write(fd, view)
                            _check(written > 0, 'Short copy write')
                            view = view[written:]
                    os.fchmod(fd, expected['common_stat']['mode'])
                    _check(_hash_fd(fd) == pair['sha256'], 'Restored copy SHA differs')
                    os.fsync(fd)
                finally:
                    os.close(fd)
                journal('INDEPENDENT_COPY_READY')
            else:
                os.fchmod(data, 0o444)
                journal('DATA_READONLY_PREPARED')
                os.link(data_name, temporary, src_dir_fd=data_dir, dst_dir_fd=common_dir, follow_symlinks=False)
                created = True
                linked = os.stat(temporary, dir_fd=common_dir, follow_symlinks=False)
                _check((linked.st_dev, linked.st_ino) == (ds['dev'], ds['ino']), 'Linked a swapped source entry')
                _check(stat.S_IMODE(linked.st_mode) == 0o444, 'Prepared shared inode must be read-only')
                journal('READONLY_HARDLINK_READY')
            # Our hardlink changes DATA ctime/nlink; content, inode and mtime
            # must still agree. FILE_COMMON is unchanged until os.replace.
            current_data = os.stat(data_name, dir_fd=data_dir, follow_symlinks=False)
            _check(stat.S_ISREG(current_data.st_mode)
                   and (current_data.st_dev, current_data.st_ino, current_data.st_size, current_data.st_mtime_ns)
                   == tuple(ds[k] for k in ('dev', 'ino', 'size', 'mtime_ns'))
                   and _hash_fd(data) == pair['sha256'], 'Original DATA changed during preparation')
            _unchanged_name(common_dir, common_name, cs)
            observe()
            os.replace(temporary, common_name, src_dir_fd=common_dir, dst_dir_fd=common_dir)
            committed = True
            journal('COMMON_ATOMICALLY_REPLACED')
            with _file(pair['common']) as (after, _, __):
                after_stat = _snapshot(after)
                _check(_hash_fd(after) == _hash_fd(data) == pair['sha256'], 'Post-operation SHA differs; inspect, do not retry')
                _check((after_stat['dev'], after_stat['ino']) != (ds['dev'], ds['ino']) if restore
                       else (after_stat['dev'], after_stat['ino']) == (ds['dev'], ds['ino']), 'Unexpected post-operation inode')
                if restore:
                    # Only after the Common link is broken may DATA become
                    # writable again. Its inode and bytes remain unchanged.
                    os.fchmod(data, expected['data_stat']['mode'])
                    _check(_snapshot(after)['mode'] == expected['common_stat']['mode'], 'Restored Common mode differs')
                else:
                    _check(after_stat['mode'] == _snapshot(data)['mode'] == 0o444, 'Shared inode lost read-only protection')
                _check(_hash_fd(after) == _hash_fd(data) == pair['sha256'], 'Bytes changed during mode restoration')
            with _file(pair['data']) as (final_data, _, __):
                final_stat = _snapshot(final_data)
                _check((final_stat['dev'], final_stat['ino']) == (ds['dev'], ds['ino'])
                       and _hash_fd(final_data) == pair['sha256'], 'Original DATA name/inode/content changed; inspect')
            observe()
            journal('VERIFIED_INDEPENDENT_MODES_RESTORED' if restore else 'VERIFIED_SHARED_READONLY')
            return {'prefix': pair['prefix'], 'status': 'RESTORED_INDEPENDENT_COPY' if restore else 'ATOMIC_HARDLINK_REPLACEMENT',
                    'data_inode_preserved': True, 'content_sha256': pair['sha256'],
                    'independent_copies': restore, 'old_common_inode_restored': False}
        finally:
            if created and not committed:
                # Only our generated temporary directory entry is removed;
                # neither original RAW filename is ever unlinked.
                try:
                    os.unlink(temporary, dir_fd=common_dir)
                except FileNotFoundError:
                    pass


def _atomic_work_json(path, value):
    """Only our own journal directory entries may be replaced."""
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    temporary = '.source_storage_guard_journal_'+uuid.uuid4().hex+'.tmp'
    with _directory(path.parent) as directory:
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            _check(stat.S_ISREG(existing.st_mode), 'Journal target is not a regular own file')
        except FileNotFoundError:
            pass
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                _check(written > 0, 'Short journal write')
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)


def execute_reviewed_plan(contract, plan, *, restore=False, journal_path=None, resume_journal_sha256=None):
    _check(plan['declaration_sha256'] == contract.declaration_sha256 and plan['state_sha256'] == contract.state_sha256,
           'Plan is not bound to current frozen source evidence')
    _check(plan['guard_code_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'Guard code changed since review')
    _check(plan['common_root'] == str(contract.common_root) and plan['allowed_prefixes'] == list(PREFIXES), 'Plan root/allowlist differs')
    with guarded(contract) as (state, pairs, _):
        _check(len(plan['entries']) == len(pairs) == 30, 'Exactly thirty reviewed entries required')
        for pair, expected in zip(pairs, plan['entries']):
            _check(all(expected.get(k) == (str(v) if isinstance(v, Path) else v) for k, v in pair.items()),
                   'Reviewed pair identity differs')
        journal_path = journal_path or contract.root/WORK_REL/('restore_journal.json' if restore else 'apply_journal.json')
        _check(journal_path.parent == contract.root/WORK_REL, 'Journal must remain under guard work')
        plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if journal_path.exists():
            _check(resume_journal_sha256 is not None, 'Existing journal requires explicit reviewed resume SHA')
            saved = json.loads(_pinned_bytes(journal_path, resume_journal_sha256))
            _check(saved['plan_canonical_sha256'] == plan_digest and saved['restore'] is restore, 'Journal scope differs')
            _check(all(entry['phase'] in ('VERIFIED_SHARED_READONLY', 'VERIFIED_INDEPENDENT_MODES_RESTORED')
                       for entry in saved['pairs'].values()), 'Partial operation requires inspection/explicit restore, never automatic repair')
        else:
            _check(resume_journal_sha256 is None, 'Resume journal does not exist')
            saved = {'schema_version': 1, 'plan_canonical_sha256': plan_digest, 'restore': restore,
                     'declaration_sha256': contract.declaration_sha256, 'state_sha256': contract.state_sha256,
                     'pairs': {}, 'status': 'RUNNING_EXPLICIT_REVIEWED_OPERATION'}
            _atomic_work_json(journal_path, saved)
        # Preflight the entire reviewed batch before touching any RAW pair.
        # Per-pair checks are still repeated immediately before its commit.
        for pair, expected in zip(pairs, plan['entries']):
            if pair['prefix'] in saved['pairs']:
                current = _pair_info(pair)
                _check(current['already_shared_inode'] is not restore
                       and (current['data_stat']['dev'], current['data_stat']['ino'])
                       == (expected['data_stat']['dev'], expected['data_stat']['ino']),
                       'Verified journal DATA inode/independence changed')
            else:
                _rewrite(pair, expected, restore=restore, preflight_only=True)
        results = []
        for pair, expected in zip(pairs, plan['entries']):
            _idle(state['owner_pid'])
            _manifest(contract)
            if pair['prefix'] in saved['pairs']:
                current = _pair_info(pair)
                _check(current['already_shared_inode'] is not restore, 'Completed journal pair has unexpected independence state')
                results.append({'prefix': pair['prefix'], 'status': 'REUSED_VERIFIED_JOURNAL_BOUNDARY_NOT_NEW_ACTION'})
                continue

            def phase(value):
                saved['pairs'][pair['prefix']] = {'phase': value, 'updated_utc': datetime.now(timezone.utc).isoformat()}
                _atomic_work_json(journal_path, saved)

            try:
                result = _rewrite(pair, expected, restore=restore,
                                  observe=lambda: _idle(state['owner_pid']), journal=phase)
                if result['status'] in ('ALREADY_SHARED_NO_CHANGE', 'ALREADY_INDEPENDENT_NO_CHANGE'):
                    phase('VERIFIED_INDEPENDENT_MODES_RESTORED' if restore else 'VERIFIED_SHARED_READONLY')
                results.append(result)
            except BaseException as error:
                saved['status'] = 'INSPECT_PARTIAL_ACTION_NO_AUTOMATIC_REPAIR_OR_RESTART'
                saved['error_type'] = type(error).__name__
                _atomic_work_json(journal_path, saved)
                raise
            _idle(state['owner_pid'])
        saved['status'] = 'RESTORE_COMPLETE' if restore else 'APPLY_COMPLETE'
        _atomic_work_json(journal_path, saved)
        return {'status': 'RESTORE_COMPLETE' if restore else 'APPLY_COMPLETE', 'results': results,
                'journal_sha256': hashlib.sha256(journal_path.read_bytes()).hexdigest(),
                'warnings': WARNINGS, 'safety': {key: False for key in SAFETY}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='dry-run', choices=('dry-run', 'apply', 'restore'))
    parser.add_argument('--common-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviewed-plan', type=Path)
    parser.add_argument('--reviewed-plan-sha256')
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--resume-journal-sha256')
    args = parser.parse_args()
    common = args.common_root.absolute()
    expected_users = Path.home()/'Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users'
    try:
        tail = common.relative_to(expected_users).parts
    except ValueError:
        raise GuardError('FILE_COMMON root is outside the known existing Wine prefix') from None
    _check(len(tail) == 7 and tail[1:] == ('AppData', 'Roaming', 'MetaQuotes', 'Terminal', 'Common', 'Files'), 'Exact FILE_COMMON root required')
    contract = Contract(ROOT, common)
    output = args.output.absolute()
    _check(output.parent == ROOT/WORK_REL and not output.exists(), 'New output under guard work only')
    if args.command == 'dry-run':
        report = dry_run(contract)
    else:
        _check(args.reviewed_plan is not None and args.reviewed_plan_sha256 is not None, 'Explicit reviewed plan/SHA required')
        _check(args.reviewed_plan.absolute().parent == ROOT/WORK_REL, 'Plan must be our guard work artifact')
        plan = json.loads(_pinned_bytes(args.reviewed_plan.absolute(), args.reviewed_plan_sha256))
        report = execute_reviewed_plan(contract, plan, restore=args.command == 'restore',
                                       journal_path=args.journal.absolute() if args.journal else None,
                                       resume_journal_sha256=args.resume_journal_sha256)
    with _directory(output.parent) as parent:
        fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    with os.fdopen(fd, 'w') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
