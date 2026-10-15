"""Toy files only; no actual DATA, FILE_COMMON, terminal or account operations."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import fcntl
from types import SimpleNamespace

import pytest

from scripts import source_storage_guard as guard


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def toy(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    common = root/'common'
    common.mkdir()
    data = root/guard.DATA_REL
    data.mkdir(parents=True)
    (data/'.collection.lock').write_text('99999999\n')
    requests, records = [], []
    for ordinal, prefix in enumerate(guard.PREFIXES):
        folder = data/'development_source'/prefix
        folder.mkdir(parents=True)
        raw = folder/(prefix+'_ticks.tsv')
        raw.write_bytes((prefix+'\ttoy-only\n').encode())
        counterpart = common/raw.name
        counterpart.write_bytes(raw.read_bytes())
        for path in (raw, counterpart):
            path.chmod(0o600)
        receipt = folder/(prefix+'_receipt.tsv')
        receipt.write_bytes(b'toy receipt only\n')
        verified = folder/'source_verification.json'
        put(verified, {'status': 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN',
                       'raw_sha256': sha(raw), 'receipt_sha256': sha(receipt), 'rows': 3, 'errors': []})
        request = {'ordinal': ordinal, 'prefix': prefix, 'symbol': 'CRASH500',
                   'partition': 'development_source', 'seconds': 86400}
        requests.append(request)
        records.append({**request, 'verification_status': 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN',
                        'actual_terminal_pids_after_capture': [], 'observer_exit': 0,
                        'rows': 3, 'raw_bytes': raw.stat().st_size,
                        'files': {str(p.relative_to(root)): sha(p) for p in (raw, receipt, verified)}})
    declaration = root/guard.DECL_REL
    put(declaration, {'only_symbol': 'CRASH500', 'output_root': guard.DATA_REL,
                      'permitted_batch_ordinals': list(range(30)), 'requests': requests+[{}]*694,
                      'source_pins': {}, 'safety': {k: False for k in guard.SAFETY}})
    state = root/guard.STATE_REL
    put(state, {'declaration_sha256': sha(declaration), 'status': 'COMPLETED_FIRST_30_SOURCE_ONLY',
                'records': records, 'strategy_measurements': 0, 'final_strategy_metrics_opened': False,
                'orders': False, 'pending': None, 'actual_terminal_pids': [], 'owner_pid': 99999999,
                'safety': {k: False for k in guard.SAFETY}})
    for key in guard.SAFETY:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(guard, 'runtime_snapshot', lambda pid: {
        'native_terminal_pids': [], 'source_writer_pids': [], 'completed_batch_owner_pid_alive': False})
    return guard.Contract(root, common, sha(declaration), sha(state))


def first_paths(contract):
    prefix = guard.PREFIXES[0]
    raw = contract.root/guard.DATA_REL/'development_source'/prefix/(prefix+'_ticks.tsv')
    return raw, contract.common_root/raw.name


def test_dry_run_changes_no_source_names_bytes_modes_or_inodes(toy):
    raw, common = first_paths(toy)
    before = [(p.read_bytes(), p.stat().st_ino, stat.S_IMODE(p.stat().st_mode)) for p in (raw, common)]
    plan = guard.dry_run(toy)
    assert len(plan['entries']) == 30 and plan['logical_duplicate_raw_bytes'] > 0
    assert plan['independent_copy_protection_after_future_apply'] is False
    assert not any(plan['safety'].values())
    assert [(p.read_bytes(), p.stat().st_ino, stat.S_IMODE(p.stat().st_mode)) for p in (raw, common)] == before


def test_atomic_apply_and_explicit_copy_restore_preserve_original_data_bytes_and_inode(toy, monkeypatch):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    content, inode = raw.read_bytes(), raw.stat().st_ino
    original_replace = guard.os.replace
    observed = []

    def inspect(source, target, **kwargs):
        if target.endswith('_ticks.tsv'):
            observed.append((source, target))
            assert os.stat(source, dir_fd=kwargs['src_dir_fd']).st_size == len(content)
        return original_replace(source, target, **kwargs)

    monkeypatch.setattr(guard.os, 'replace', inspect)
    applied = guard.execute_reviewed_plan(toy, plan)
    assert applied['status'] == 'APPLY_COMPLETE' and len(observed) == 30
    assert raw.stat().st_ino == common.stat().st_ino == inode
    assert raw.read_bytes() == common.read_bytes() == content
    assert stat.S_IMODE(raw.stat().st_mode) == stat.S_IMODE(common.stat().st_mode) == 0o444
    restored = guard.execute_reviewed_plan(toy, plan, restore=True)
    assert restored['status'] == 'RESTORE_COMPLETE'
    assert raw.stat().st_ino == inode and common.stat().st_ino != inode
    assert raw.read_bytes() == common.read_bytes() == content
    assert stat.S_IMODE(raw.stat().st_mode) == stat.S_IMODE(common.stat().st_mode) == 0o600
    assert raw.stat().st_nlink == common.stat().st_nlink == 1


def test_changed_bytes_after_plan_reject_without_replacement(toy):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    inode = common.stat().st_ino
    common.write_bytes(b'changed')
    with pytest.raises(guard.GuardError, match='RAW changed'):
        guard.execute_reviewed_plan(toy, plan)
    assert common.stat().st_ino == inode and common.read_bytes() == b'changed'
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600


def test_same_bytes_new_inode_after_plan_reject(toy):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    temporary = common.with_suffix('.replacement')
    temporary.write_bytes(common.read_bytes());temporary.chmod(0o600)
    temporary.replace(common)
    with pytest.raises(guard.GuardError, match='Reviewed plan stat changed'):
        guard.execute_reviewed_plan(toy, plan)
    assert raw.stat().st_ino != common.stat().st_ino


@pytest.mark.parametrize('field', ['declaration_sha256', 'state_sha256'])
def test_wrong_metadata_sha_fails_closed(toy, field):
    with pytest.raises(guard.GuardError, match='SHA changed'):
        guard.dry_run(replace(toy, **{field: 'a'*64}))


def test_wrong_prefix_rejected_even_if_new_fixture_pin_is_trusted(toy):
    state = toy.root/guard.STATE_REL
    value = json.loads(state.read_text());value['records'][0]['prefix'] = 'Pilot24_CRASH500_20240102'
    put(state, value)
    with pytest.raises(guard.GuardError, match='prefix'):
        guard.dry_run(replace(toy, state_sha256=sha(state)))


def test_nonallowed_raw_path_rejected_without_opening_it(toy):
    state = toy.root/guard.STATE_REL
    value = json.loads(state.read_text());raw, _ = first_paths(toy)
    key = str(raw.relative_to(toy.root));pin = value['records'][0]['files'].pop(key)
    value['records'][0]['files']['private/'+raw.name] = pin
    put(state, value)
    with pytest.raises(guard.GuardError, match='Exact source file pin missing'):
        guard.dry_run(replace(toy, state_sha256=sha(state)))


@pytest.mark.parametrize('which', ['data', 'common', 'common_ancestor'])
def test_symlink_files_or_ancestors_rejected(toy, which):
    raw, common = first_paths(toy)
    if which == 'common_ancestor':
        alias = toy.root/'alias';alias.symlink_to(toy.common_root, target_is_directory=True)
        contract = replace(toy, common_root=alias)
    else:
        target = raw if which == 'data' else common
        real = target.with_suffix('.real');target.rename(real);target.symlink_to(real)
        contract = toy
    with pytest.raises((guard.GuardError, OSError)):
        guard.dry_run(contract)


def test_cross_filesystem_rejected_before_any_hardlink(toy, monkeypatch):
    original = guard._same_device
    monkeypatch.setattr(guard, '_same_device', lambda left, right: original(left, {**right, 'dev': left['dev']+1}))
    with pytest.raises(guard.GuardError, match='Cross-filesystem'):
        guard.dry_run(toy)


@pytest.mark.parametrize('key,value', [('native_terminal_pids', [123]),
    ('source_writer_pids', [124]), ('completed_batch_owner_pid_alive', True)])
def test_active_process_rejected(toy, monkeypatch, key, value):
    monkeypatch.setattr(guard, 'runtime_snapshot', lambda pid: {
        'native_terminal_pids': value if key == 'native_terminal_pids' else [],
        'source_writer_pids': value if key == 'source_writer_pids' else [],
        'completed_batch_owner_pid_alive': value if key == 'completed_batch_owner_pid_alive' else False})
    with pytest.raises(guard.GuardError, match='remains alive'):
        guard.dry_run(toy)


@pytest.mark.parametrize('key', guard.SAFETY)
def test_all_unsafe_flags_rejected(toy, monkeypatch, key):
    monkeypatch.setenv(key, 'true')
    with pytest.raises(guard.GuardError, match='Unsafe environment'):
        guard.dry_run(toy)


def test_failed_atomic_replace_preserves_both_raw_bytes_and_records_partial_state(toy, monkeypatch):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    original_replace = guard.os.replace
    original_content, original_common_inode = raw.read_bytes(), common.stat().st_ino

    def fail(source, target, **kwargs):
        if target.endswith('_ticks.tsv'):
            raise OSError('synthetic rename failure')
        return original_replace(source, target, **kwargs)

    monkeypatch.setattr(guard.os, 'replace', fail)
    with pytest.raises(OSError, match='synthetic'):
        guard.execute_reviewed_plan(toy, plan)
    assert raw.read_bytes() == common.read_bytes() == original_content
    assert common.stat().st_ino == original_common_inode
    assert stat.S_IMODE(raw.stat().st_mode) == 0o444  # Deliberate safe partial preparation.
    assert not list(toy.common_root.glob('.source_storage_guard_*'))
    journal = toy.root/guard.WORK_REL/'apply_journal.json'
    saved = json.loads(journal.read_text())
    assert saved['status'] == 'INSPECT_PARTIAL_ACTION_NO_AUTOMATIC_REPAIR_OR_RESTART'
    with pytest.raises(guard.GuardError, match='Partial operation'):
        guard.execute_reviewed_plan(toy, plan, resume_journal_sha256=sha(journal))
    monkeypatch.setattr(guard.os, 'replace', original_replace)
    guard.execute_reviewed_plan(toy, plan, restore=True)
    assert raw.read_bytes() == common.read_bytes() == original_content
    assert raw.stat().st_ino != common.stat().st_ino
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600


def test_verified_completed_journal_resume_requires_sha_and_is_no_new_action(toy):
    plan = guard.dry_run(toy)
    guard.execute_reviewed_plan(toy, plan)
    with pytest.raises(guard.GuardError, match='explicit reviewed resume SHA'):
        guard.execute_reviewed_plan(toy, plan)
    journal = toy.root/guard.WORK_REL/'apply_journal.json'
    resumed = guard.execute_reviewed_plan(toy, plan, resume_journal_sha256=sha(journal))
    assert all(row['status'] == 'REUSED_VERIFIED_JOURNAL_BOUNDARY_NOT_NEW_ACTION' for row in resumed['results'])


def test_postplan_runtime_recheck_prevents_mutation(toy, monkeypatch):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    monkeypatch.setattr(guard, '_idle', lambda pid: (_ for _ in ()).throw(guard.GuardError('now active')))
    with pytest.raises(guard.GuardError, match='now active'):
        guard.execute_reviewed_plan(toy, plan)
    assert raw.stat().st_ino != common.stat().st_ino and stat.S_IMODE(raw.stat().st_mode) == 0o600


def test_held_collection_lock_rejects(toy):
    lockpath = toy.root/guard.DATA_REL/'.collection.lock'
    with lockpath.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(guard.GuardError, match='lock is held'):
            guard.dry_run(toy)


def test_bad_later_plan_identity_rejects_before_first_raw_changes(toy):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    plan['entries'][5]['common'] = str(toy.root/'not-allowed.tsv')
    with pytest.raises(guard.GuardError, match='identity differs'):
        guard.execute_reviewed_plan(toy, plan)
    assert raw.stat().st_ino != common.stat().st_ino and stat.S_IMODE(raw.stat().st_mode) == 0o600


def test_changed_later_raw_rejects_before_first_raw_changes(toy):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    Path(plan['entries'][9]['common']).write_bytes(b'changed later')
    with pytest.raises(guard.GuardError, match='RAW changed'):
        guard.execute_reviewed_plan(toy, plan)
    assert raw.stat().st_ino != common.stat().st_ino and stat.S_IMODE(raw.stat().st_mode) == 0o600


def test_wrong_guard_code_pin_rejects_before_actions(toy):
    plan = guard.dry_run(toy)
    plan['guard_code_sha256'] = 'b'*64
    with pytest.raises(guard.GuardError, match='Guard code changed'):
        guard.execute_reviewed_plan(toy, plan)


def test_temp_name_collision_is_never_unlinked_as_our_created_file(toy, monkeypatch):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    original_link = guard.os.link
    created = []

    def collide(source, target, **kwargs):
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=kwargs['dst_dir_fd'])
        os.write(fd, b'foreign collision');os.close(fd)
        created.append(target)
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(guard.os, 'link', collide)
    with pytest.raises(FileExistsError):
        guard.execute_reviewed_plan(toy, plan)
    assert (toy.common_root/created[0]).read_bytes() == b'foreign collision'
    assert raw.stat().st_ino != common.stat().st_ino


def test_native_becoming_active_immediately_before_commit_blocks_rename(toy, monkeypatch):
    plan = guard.dry_run(toy)
    raw, common = first_paths(toy)
    original_inode, content = common.stat().st_ino, common.read_bytes()
    calls = []

    def observe(pid):
        calls.append(pid)
        if len(calls) >= 2:
            with (toy.root/guard.DATA_REL/'.collection.lock').open('r') as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if len(calls) == 5:
            raise guard.GuardError('native became active before commit')
        return {'native_terminal_pids': [], 'source_writer_pids': [], 'completed_batch_owner_pid_alive': False}

    monkeypatch.setattr(guard, '_idle', observe)
    with pytest.raises(guard.GuardError, match='native became active'):
        guard.execute_reviewed_plan(toy, plan)
    assert common.stat().st_ino == original_inode and common.read_bytes() == raw.read_bytes() == content
    assert stat.S_IMODE(raw.stat().st_mode) == 0o444
    assert not list(toy.common_root.glob('.source_storage_guard_*'))


def test_runtime_probe_uses_executable_names_and_recognizes_versioned_python(monkeypatch):
    monkeypatch.setattr(guard.subprocess, 'check_output', lambda *args, **kwargs:
        '91 /usr/bin/python3.12\n92 /bin/zsh\n93 /Applications/wine64-preloader\n')
    calls = []

    def args(command, **kwargs):
        pid = command[2];calls.append(pid)
        return SimpleNamespace(stdout=('python work/independent_profit_pilot_20261013/run_dataset_probe.py'
                                       if pid == '91' else 'wine terminal64.exe'))

    monkeypatch.setattr(guard.subprocess, 'run', args)
    result = guard.runtime_snapshot(99999999)
    assert result['native_terminal_pids'] == [93] and result['source_writer_pids'] == [91]
    assert calls == ['91', '93']  # Shell text cannot become a false terminal match.
