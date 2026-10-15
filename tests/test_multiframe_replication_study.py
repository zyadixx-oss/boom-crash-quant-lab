"""Temporary synthetic repositories only; never production quotes or alpha."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from independent_profit_research.execution import Quote
from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/run_multiframe_replication_study.py'
spec = importlib.util.spec_from_file_location('replication_controller_under_test', MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Full fixed calendar, tiny fake quote files, no native/network machinery."""
    root = tmp_path / 'repo'
    meta = root / 'work/source'
    root.mkdir()
    pins = {}
    for relative in (*m.ACQUISITION_CODE, 'synthetic_dependency.py'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic audited machinery placeholder\n')
        pins[relative] = m.sha(path)
    requests = []
    for day_index, day in enumerate(range(m.START, m.END, m.DAY_MS)):
        if m.utc_day(day) in m.PILOTS:
            continue
        requests.append({'ordinal': len(requests), 'calendar_day_index': day_index,
            'start_utc': m.utc_day(day) + 'T00:00:00Z', 'start_ms': day,
            'end_exclusive_ms': day + m.DAY_MS, 'seconds': 86400, 'symbol': 'CRASH500',
            'prefix': 'MFR23_CRASH500_' + m.utc_day(day).replace('-', ''),
            'partition': 'development_source' if day_index < 510 else 'sealed_final_source'})
    calendar = {'calendar_start_inclusive': '2023-01-03', 'calendar_end_exclusive': '2025-01-01',
        'fixed_calendar_days': 729, 'only_symbol': 'CRASH500', 'output_root': m.SOURCE_ROOT,
        'calendar_split_70_30': {'development_calendar_slots': 510, 'final_calendar_slots': 219,
            'split_utc': '2024-05-27', 'quality_exclusions_do_not_compress_calendar': True},
        'excluded_whole_pilot_days_both_symbols': list(m.PILOTS), 'requests': requests,
        'source_pins': pins, 'safety': m.SAFETY, 'permitted_batch_ordinals': list(range(30))}
    write(meta / m.PRODUCER_NAMES[0], calendar)
    states = []
    for batch, (left, right) in enumerate(m.BATCH_RANGES[:3]):
        producer = deepcopy(calendar)
        producer['permitted_batch_ordinals'] = list(range(left, right))
        write(meta / m.PRODUCER_NAMES[batch], producer)
        state = {'declaration_sha256': m.sha(meta / m.PRODUCER_NAMES[batch]),
            'safety': m.SAFETY, 'orders': False, 'strategy_measurements': 0,
            'final_strategy_metrics_opened': False, 'status': 'RUNNING', 'records': []}
        for request in requests[left:right]:
            # No FINAL files/state/declaration are ever created by this fixture.
            files = {}
            folder = Path(m.SOURCE_ROOT) / request['partition'] / request['prefix']
            for suffix in ('_ticks.tsv', '_receipt.tsv'):
                relative = str(folder / (request['prefix'] + suffix))
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('SYNTHETIC ' + request['start_utc'] + suffix)
                files[relative] = m.sha(path)
            state['records'].append(request | {'files': files, 'rows': 3,
                'verification_status': 'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN', 'test_metadata': 'original'})
        write(meta / m.STATE_NAMES[batch], state)
        states.append(state)
    controller = m.Controller(root, root / 'work/study', meta)
    monkeypatch.setattr(controller, '_fixed_pins', lambda calendar: dict(pins))
    loads = []

    def fake_load(receipt, receipt_sha, raw_sha, symbol, start, end):
        assert symbol == 'CRASH500' and end == start + m.DAY_MS
        assert m.utc_day(start) not in m.PILOTS
        loads.append(start)
        return SimpleNamespace(tape=QuoteTape([
            Quote(start + 1000, 10000, 10010), Quote(start + 2000, 10000, 10010),
            Quote(start + 3000, 10000, 10010)]))

    monkeypatch.setattr(m, 'load_day', fake_load)
    protocol = root / 'protocol.json'
    protocol.write_text('synthetic reviewed protocol')
    validation = root / 'validation.json'
    write(validation, {'status': 'PASS', 'approved_for_protocol_freeze': True,
        'actual_strategy_measurements': 0, 'code_pins': pins, 'protocol_sha256': m.sha(protocol),
        'calendar_declaration_sha256': m.sha(meta / m.PRODUCER_NAMES[0])})
    return SimpleNamespace(c=controller, root=root, meta=meta, calendar=calendar,
        states=states, loads=loads, protocol=protocol, validation=validation)


def frozen(lab):
    lab.c.freeze(lab.protocol, lab.validation)
    return lab.c


def development(lab):
    c = frozen(lab)
    for stage in m.DEVELOPMENT:
        assert c.stage(stage)['calculated_cases'] == 10
    return c


def reindex(c, relative, mutate):
    value = m.read_json(c.out / relative)
    mutate(value)
    write(c.out / relative, value)
    index = m.read_json(c.out / 'artifact_index.json')
    index[relative] = m.sha(c.out / relative)
    write(c.out / 'artifact_index.json', index)


def rewrite_state(lab, batch=1):
    write(lab.meta / m.STATE_NAMES[batch], lab.states[batch])


def rewrite_producer(lab, mutate, batch=1):
    path = lab.meta / m.PRODUCER_NAMES[batch]
    value = m.read_json(path)
    mutate(value)
    write(path, value)
    lab.states[batch]['declaration_sha256'] = m.sha(path)
    rewrite_state(lab, batch)


def no_raw_hash(monkeypatch):
    original = m.sha
    calls = []

    def checked(path):
        calls.append(str(path))
        assert not str(path).endswith(('_ticks.tsv', '_receipt.tsv'))
        return original(path)

    monkeypatch.setattr(m, 'sha', checked)
    return calls


def test_exact_fixed_calendars_masks_and_template():
    assert (m.END - m.START) // m.DAY_MS == 729
    assert sum(m.mask_for(m.START, m.END)) == 724
    assert sum(m.mask_for(m.START, m.START + 510 * m.DAY_MS)) == 508
    for stage, (lo, hi, train) in m.STAGES.items():
        assert train == lo
        assert len(m.mask_for(m.START + lo * m.DAY_MS, m.START + hi * m.DAY_MS)) == hi - lo
    assert m.template()['development_gate']['unknown_allowed'] == 0
    assert m.template()['cash_net'] is None
    assert all(m.template()[key] is False for key in m.SAFETY)


def test_freeze_requires_reviewed_no_measurement_report(lab):
    review = m.read_json(lab.validation)
    review['actual_strategy_measurements'] = 1
    write(lab.validation, review)
    with pytest.raises(ValueError, match='premeasurement'):
        lab.c.freeze(lab.protocol, lab.validation)
    assert not (lab.c.out / 'declaration.json').exists()


@pytest.mark.parametrize('key', tuple(m.SAFETY))
def test_all_eight_environment_flags_fail_closed(lab, monkeypatch, key):
    monkeypatch.setenv(key, 'true')
    with pytest.raises(ValueError, match='eight'):
        lab.c.preflight('WF1')
    assert not lab.c.out.exists()


def test_exclusive_lock_and_output_symlink_rejected(lab):
    with lab.c.locked():
        with pytest.raises(ValueError, match='owns'):
            with lab.c.locked():
                pass
    target = lab.root / 'other'
    target.mkdir()
    (lab.c.out / 'linked').symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        lab.c._write('linked/test.json', {})


def test_registered_receipt_cannot_be_recreated_after_deletion(lab):
    c = frozen(lab)
    path = c.out / 'declaration.json'
    path.unlink()
    with pytest.raises(ValueError, match='registered'):
        c._write('declaration.json', {})
    assert not path.exists()


@pytest.mark.parametrize('kind', ['missing', 'failed', 'wrong_date', 'duplicate'])
def test_all_source_metadata_required_before_any_raw_or_load(lab, monkeypatch, kind):
    row = lab.states[1]['records'][0]
    if kind == 'missing':
        lab.states[1]['records'].pop(0)
    elif kind == 'failed':
        row['rows'] = 0
        row['verification_status'] = 'EMPTY_OR_FAILED_NOT_VERIFIED'
    elif kind == 'wrong_date':
        row['start_ms'] += m.DAY_MS
    else:
        lab.states[1]['records'].append(deepcopy(row))
    rewrite_state(lab)
    no_raw_hash(monkeypatch)
    with pytest.raises(ValueError):
        lab.c.source_records(0, 180)
    assert lab.loads == []


@pytest.mark.parametrize('kind', ['missing_pins', 'empty_pins', 'required_pin', 'requests',
                                  'calendar', 'safety', 'permitted', 'output_root'])
def test_producer_provenance_is_required_before_raw(lab, monkeypatch, kind):
    def change(p):
        if kind == 'missing_pins':
            p.pop('source_pins')
        elif kind == 'empty_pins':
            p['source_pins'] = {}
        elif kind == 'required_pin':
            p['source_pins'][m.ACQUISITION_CODE[0]] = '0' * 64
        elif kind == 'requests':
            p['requests'][0]['start_ms'] += 1
        elif kind == 'calendar':
            p['fixed_calendar_days'] = 728
        elif kind == 'safety':
            p['safety']['LIVE_TRADING'] = True
        elif kind == 'permitted':
            p['permitted_batch_ordinals'].pop()
        else:
            p['output_root'] += '_other'
    rewrite_producer(lab, change)
    no_raw_hash(monkeypatch)
    with pytest.raises(ValueError):
        lab.c.source_records(0, 180)


@pytest.mark.parametrize('kind', ['safety', 'orders', 'measurements', 'final_opened'])
def test_producer_state_source_only_contract(lab, monkeypatch, kind):
    state = lab.states[1]
    if kind == 'safety':
        state['safety'] = m.SAFETY | {'ready_for_live': True}
    elif kind == 'orders':
        state['orders'] = True
    elif kind == 'measurements':
        state['strategy_measurements'] = 1
    else:
        state['final_strategy_metrics_opened'] = True
    rewrite_state(lab)
    no_raw_hash(monkeypatch)
    with pytest.raises(ValueError, match='source-only'):
        lab.c.source_records(0, 180)


def test_readonly_preflight_retains_failed_day_no_freeze_or_alpha(lab, monkeypatch):
    row = lab.states[1]['records'][0]
    row['rows'] = 0
    row['verification_status'] = 'EMPTY_OR_FAILED_NOT_VERIFIED'
    rewrite_state(lab)
    no_raw_hash(monkeypatch)
    result = lab.c.preflight('WF1')
    assert result['source_metadata_available'] is False
    assert 'failed' in result['error']
    assert result['actual_strategy_measurements'] == 0
    assert lab.loads == [] and not lab.c.out.exists()
    with pytest.raises(ValueError, match='development'):
        lab.c.preflight('FINAL')


def test_preflight_success_is_not_raw_coverage_claim(lab, monkeypatch):
    no_raw_hash(monkeypatch)
    result = lab.c.preflight('WF1')
    assert result['source_metadata_available'] and result['selected_records'] == 180
    assert not result['raw_files_verified']
    assert not lab.c.out.exists() and not lab.loads


@pytest.mark.parametrize('access', ['source', 'calculate', 'cache'])
def test_direct_final_access_rejects_bare_boolean_before_any_index(lab, monkeypatch, access):
    monkeypatch.setattr(lab.c, '_read_state', lambda *a: pytest.fail('source index opened'))
    monkeypatch.setattr(lab.c, '_read', lambda *a: pytest.fail('cached receipt opened'))
    with pytest.raises(ValueError, match='rederived'):
        if access == 'source':
            lab.c.source_records(0, 729, permit_final=True)
        elif access == 'calculate':
            lab.c._calculate_stage('FINAL', m.POLICIES, {}, 'a' * 64, permit_final=True)
        else:
            lab.c._cached_case('FINAL', m.POLICIES[0], 'BASE', permit_final=True)


def test_actual_synthetic_stage_has_ten_cases_then_ten_reuses(lab, monkeypatch):
    c = frozen(lab)
    evaluations = []
    original = m.evaluate_streamed_signals

    def counted(*args, **kwargs):
        evaluations.append((kwargs['policy_id'], kwargs['baseline']))
        return original(*args, **kwargs)

    monkeypatch.setattr(m, 'evaluate_streamed_signals', counted)
    first = c.stage('WF1')
    assert first['case_count'] == first['calculated_cases'] == len(evaluations) == 10
    assert sum(baseline for _, baseline in evaluations) == 4
    assert len({policy for policy, _ in evaluations}) == 5
    loads_before = len(lab.loads)
    second = c.stage('WF1')
    assert second['reused_cases'] == 10 and second['calculated_cases'] == 0
    assert len(evaluations) == 10 and len(lab.loads) == loads_before


def test_unrelated_state_append_and_status_change_preserve_prior_case(lab):
    c = frozen(lab)
    # WF1 requires all batch2 records; WF2's selected prefix ends at ordinal269.
    c.stage('WF2')
    state = lab.states[2]
    state['records'] = state['records'][:90]
    rewrite_state(lab, 2)
    before = c.stage('WF2')
    state['records'].append(deepcopy(lab.calendar['requests'][270]) | {
        'rows': 0, 'verification_status': 'EMPTY_OR_FAILED_NOT_VERIFIED', 'files': {}})
    state['updated_utc'] = 'unrelated future update'
    state['status'] = 'FAILED'
    rewrite_state(lab, 2)
    after = c.stage('WF2')
    assert before == after and after['reused_cases'] == 10


def test_selected_metadata_mutation_rejects_cached_case(lab):
    c = frozen(lab)
    c.stage('WF1')
    lab.states[1]['records'][0]['test_metadata'] = 'changed selected record'
    rewrite_state(lab)
    with pytest.raises(ValueError, match='source binding changed'):
        c.stage('WF1')
    with pytest.raises(ValueError, match='lineage changed'):
        c._cached_case('WF1', m.POLICIES[0], 'BASE')


@pytest.mark.parametrize('kind', ['calendar', 'code', 'raw', 'ledger', 'case', 'sqlite'])
def test_dependency_and_cache_drift_rejected(lab, kind):
    c = frozen(lab)
    c.stage('WF1')
    case = c._read(c._case_name('WF1', m.POLICIES[0], 'BASE'))
    if kind == 'calendar':
        value = deepcopy(lab.calendar)
        value['only_symbol'] = 'BOOM500'
        write(c.calendar_path, value)
    elif kind == 'code':
        (lab.root / m.ACQUISITION_CODE[0]).write_text('drift')
    elif kind == 'raw':
        relative = next(p for p in lab.states[1]['records'][0]['files'] if p.endswith('_ticks.tsv'))
        (lab.root / relative).write_text('drift')
    elif kind == 'ledger':
        (c.out / case['ledger_path']).write_bytes(b'drift')
    elif kind == 'case':
        (c.out / c._case_name('WF1', m.POLICIES[0], 'BASE')).write_text('{}')
    else:
        (c.out / 'fits/TRAIN0_90.sqlite').write_bytes(b'drift')
    with pytest.raises(ValueError):
        c.stage('WF1')


@pytest.mark.parametrize('field,value', [
    ('ledger_path', 'ledgers/FINAL_CRT_SWEEP_RECLAIM_BASE.json.gz'),
    ('ledger_path', '/tmp/unpermitted'), ('ledger_path', '../unpermitted'),
    ('fit_path', 'fits/TRAIN0_510.json'), ('fit_path', '/tmp/unpermitted'),
])
def test_forged_reindexed_paths_fail_before_target_open(lab, monkeypatch, field, value):
    c = frozen(lab)
    c.stage('WF1')
    relative = c._case_name('WF1', m.POLICIES[0], 'BASE')
    def change(row):
        if field == 'ledger_path':
            row[field] = value
        else:
            row['binding'][field] = value
    reindex(c, relative, change)
    original = m.sha
    def checked(path):
        assert str(path) != str(c.out / value)
        return original(path)
    monkeypatch.setattr(m, 'sha', checked)
    with pytest.raises(ValueError, match='exact case identity'):
        c._cached_case('WF1', m.POLICIES[0], 'BASE')


@pytest.mark.parametrize('kind', ['train_end', 'train_decl', 'unit', 'sqlite_path', 'case_mask', 'case_calendar'])
def test_reindexed_fit_and_case_contract_mutations_rejected(lab, kind):
    c = frozen(lab)
    c.stage('WF1')
    if kind.startswith('case'):
        relative = c._case_name('WF1', m.POLICIES[0], 'BASE')
        def change(row):
            if kind == 'case_mask':
                row['binding']['valid_day_mask'][0] = False
            else:
                row['binding']['stage_start_ms'] += m.DAY_MS
    else:
        relative = 'fits/TRAIN0_90.json'
        def change(row):
            if kind == 'train_end':
                row['binding']['train_end_exclusive_ms'] += m.DAY_MS
            elif kind == 'train_decl':
                row['binding']['declaration_sha256'] = '0' * 64
            elif kind == 'unit':
                row['fit']['U_scaled'] *= 2
            else:
                row['sqlite_path'] = 'fits/TRAIN0_510.sqlite'
    reindex(c, relative, change)
    with pytest.raises(ValueError):
        c._cached_case('WF1', m.POLICIES[0], 'BASE')


def test_zero_eligible_no_final_index_cases_or_bootstrap(lab, monkeypatch):
    c = development(lab)
    selected = c.select()
    assert selected['eligible'] == []
    original = c._read_state
    def guarded(batch):
        assert batch != 3, 'FINAL source index opened'
        return original(batch)
    monkeypatch.setattr(c, '_read_state', guarded)
    monkeypatch.setattr(c, '_calculate_stage', lambda *a, **k: pytest.fail('FINAL case calculation'))
    statistics = c._statistics()
    monkeypatch.setattr(statistics, 'analyze_policy', lambda *a, **k: pytest.fail('bootstrap called'))
    result = c.final()
    assert result['final_cases'] == result['actual_analyze_policy_calls'] == 0
    assert all(v is None for v in result['analyses'].values())
    assert result['holm_placeholders_are_not_actual_tests']
    assert result['profit_qualified'] is result['goal_profit_achieved'] is False
    assert c._final_authorization is None
    assert c.final() == result
    assert not list((c.out / 'cases').glob('FINAL*'))


def test_changed_selection_rejected_before_statistics_final_metadata(lab, monkeypatch):
    c = development(lab)
    c.select()
    reindex(c, 'selection.json', lambda s: s.update(eligible=[m.POLICIES[0]]))
    monkeypatch.setattr(c, '_statistics', lambda: pytest.fail('statistics reached before selection'))
    with pytest.raises(ValueError, match='rederived'):
        c.final()
    assert c._final_authorization is None


@pytest.mark.parametrize('kind', ['candidate_count', 'validation_count', 'sparse_count',
                                  'sparse_val', 'unknown', 'val_sign', 'WF_sign', 'excess'])
def test_development_gates_both_configs_strict(lab, monkeypatch, kind):
    c = lab.c
    monkeypatch.setattr(c, 'source_records', lambda *a, **k: [])
    monkeypatch.setattr(c, '_case_context', lambda *a, **k: {})
    c.out.mkdir(parents=True)
    for stage in m.DEVELOPMENT:
        for role in (*m.POLICIES, 'SPARSE_CLOCK'):
            for scenario in m.SCENARIOS:
                p = c.out / c._case_name(stage, role, scenario)
                p.parent.mkdir(exist_ok=True)
                p.write_text('synthetic evidence')
    def cached(stage, role, scenario, **kwargs):
        sparse = role == 'SPARSE_CLOCK'
        row = {'completed': 30, 'sum_R': 1. if sparse else 3., 'unknown': 0}
        if scenario == 'STRESS':
            if kind == 'candidate_count' and not sparse:
                row['completed'] = 24
            elif kind == 'validation_count' and not sparse and stage == 'VALIDATION':
                row['completed'] = 19
            elif kind == 'sparse_count' and sparse:
                row['completed'] = 24
            elif kind == 'sparse_val' and sparse and stage == 'VALIDATION':
                row['completed'] = 29
            elif kind == 'unknown' and not sparse and stage == 'WF1':
                row.update(unknown=1, sum_R=None)
            elif kind == 'val_sign' and not sparse and stage == 'VALIDATION':
                row['sum_R'] = 0
            elif kind == 'WF_sign' and not sparse and stage in ('WF1', 'WF2'):
                row['sum_R'] = -1
            elif kind == 'excess' and not sparse:
                row['sum_R'] = 1.
        return {'binding': {'declaration_sha256': 'a' * 64}, 'summary': row}
    monkeypatch.setattr(c, '_cached_case', cached)
    result = c.derive_selection('a' * 64)
    assert result['eligible'] == []
    for gate in result['gates'].values():
        assert all(gate['scenarios']['BASE'].values())
        assert not all(gate['scenarios']['STRESS'].values())


def test_calendar_embargo_does_not_bridge_pilot_or_redraw_sparse():
    pilot = m.START + 364 * m.DAY_MS  # 2024-01-02, within VALIDATION.
    assert m.utc_day(pilot) == '2024-01-02'
    start, end = pilot - m.DAY_MS, pilot + 2 * m.DAY_MS
    decisions = [start, start + 1_800_000, pilot - 1_800_000 - 1,
                 pilot - 1_800_000, pilot, pilot + m.DAY_MS + 1_800_000,
                 end - 1_800_000]
    signals = [Signal('SYNTHETIC', -1, d, d + 1000, 10., None) for d in decisions]
    out = list(m.embargoed_signals(signals, start, end, 250))
    assert [s.decision_ms for s in out] == [decisions[1], decisions[2], decisions[5]]
    crossing = Signal('SYNTHETIC', -1, decisions[2], pilot - 250, 10., None)
    assert list(m.embargoed_signals([crossing], start, end, 250)) == []


def test_pilot_loader_is_empty_and_never_discovers_paths(lab, monkeypatch):
    loader = lab.c._loader([])
    monkeypatch.setattr(lab.c, '_path', lambda *a: pytest.fail('pilot path lookup'))
    for date in m.PILOTS:
        stamp = int(m.datetime.fromisoformat(date).replace(tzinfo=m.timezone.utc).timestamp() * 1000)
        assert not loader(stamp).quotes
    with pytest.raises(ValueError, match='outside'):
        loader(m.START)


def test_strict_json_duplicate_and_nonfinite_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            m.read_json(path)


@pytest.mark.parametrize('kind', ['state', 'producer', 'outside_meta'])
def test_metadata_paths_rejected_before_unapproved_file_open(lab, monkeypatch, kind):
    if kind == 'outside_meta':
        forbidden = lab.root.parent / 'outside' / m.STATE_NAMES[0]
        forbidden.parent.mkdir()
        forbidden.write_text('{}')
        lab.c.meta = forbidden.parent
    else:
        name = m.STATE_NAMES[1] if kind == 'state' else m.PRODUCER_NAMES[1]
        forbidden = lab.root / ('forbidden_' + name)
        path = lab.meta / name
        forbidden.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(forbidden)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != forbidden, 'unapproved metadata file opened'
        assert not path.is_symlink(), 'symlink metadata file opened'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(ValueError):
        lab.c.source_records(0, 180)


def test_eligible_final_calls_real_statistics_contract_without_source(lab, monkeypatch):
    c = frozen(lab)
    _, declared = c.declaration()
    selected = {'eligible': [m.POLICIES[0]], 'declaration_sha256': declared}
    c._write('selection.json', selected)
    monkeypatch.setattr(c, '_final_selection', lambda sha: selected)
    monkeypatch.setattr(c, '_read_state', lambda *a: pytest.fail('source metadata read'))
    calculated = []
    def calculate(stage, policies, spec, sha, **kwargs):
        assert stage == 'FINAL' and policies == (m.POLICIES[0],)
        assert kwargs['permit_final'] and c._final_authorization == (declared, m.fingerprint(selected))
        calculated.append(stage)
        return {'case_count': 6}
    monkeypatch.setattr(c, '_calculate_stage', calculate)
    monkeypatch.setattr(c, '_case_context', lambda *a, **k: {})
    lo, hi, _ = m.STAGES['FINAL']
    dates = [m.utc_day(m.START + i * m.DAY_MS) for i in range(lo, hi)]
    def cached(stage, role, scenario, **kwargs):
        assert stage == 'FINAL' and kwargs['permit_final']
        return {'daily': [{'date_utc': date, 'sum_R': None if date in m.PILOTS else 0.,
            'completed': 0, 'unknown': 0, 'observed_gain_R': None if date in m.PILOTS else 0.,
            'observed_loss_R': None if date in m.PILOTS else 0.} for date in dates],
            'summary': {'primary_measurement_usable': True, 'marked_drawdown_pct': 0.},
            'declared_calendar_source_available': True}
    monkeypatch.setattr(c, '_cached_case', cached)
    statistics = c._statistics()
    original = statistics.analyze_policy
    calls = []
    def counted(calendar, scenarios, **kwargs):
        assert isinstance(calendar, statistics.Calendar) and len(calendar.utc_days) == 219
        assert sum(calendar.valid_day_mask) == 216
        assert set(scenarios) == {'BASE', 'STRESS'}
        assert all(isinstance(v, statistics.ScenarioRecords) for v in scenarios.values())
        assert kwargs['draws'] == 99999 and kwargs['declaration_sha256'] == declared
        calls.append(kwargs['policy_id'])
        # Zero synthetic trades intentionally prevent any bootstrap computation.
        return original(calendar, scenarios, **kwargs)
    monkeypatch.setattr(statistics, 'analyze_policy', counted)
    result = c.final()
    assert calculated == ['FINAL'] and calls == [m.POLICIES[0]]
    assert result['final_cases'] == 6 and result['actual_analyze_policy_calls'] == 1
    assert result['analyses'][m.POLICIES[0]]['requested_draws'] == 99999
    assert result['analyses'][m.POLICIES[1]] is None
    assert not result['gates'][m.POLICIES[0]]['conditional_partial_pass']
    assert not result['profit_qualified'] and c._final_authorization is None


def test_selected_final_exception_revokes_authorization(lab, monkeypatch):
    c = frozen(lab)
    _, declared = c.declaration()
    selected = {'eligible': [m.POLICIES[0]], 'declaration_sha256': declared}
    c._write('selection.json', selected)
    monkeypatch.setattr(c, '_final_selection', lambda sha: selected)
    def interrupted(*args, **kwargs):
        c._guard_final(True, declared)
        raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(c, '_calculate_stage', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        c.final()
    assert c._final_authorization is None and not (c.out / 'final_summary.json').exists()


def test_final_authorization_binds_declaration_and_saved_selection(lab):
    c = frozen(lab)
    _, declared = c.declaration()
    selected = {'eligible': [m.POLICIES[0]], 'declaration_sha256': declared}
    c._write('selection.json', selected)
    c._final_authorization = (declared, m.fingerprint(selected))
    c._guard_final(True, declared)
    with pytest.raises(ValueError, match='authorization'):
        c._guard_final(True, '0' * 64)
    reindex(c, 'selection.json', lambda row: row.update(eligible=[]))
    with pytest.raises(ValueError, match='authorization'):
        c._guard_final(True, declared)


@pytest.mark.parametrize('artifact', ['sqlite', 'ledger'])
def test_orphan_artifact_never_overwritten_or_silently_reused(lab, artifact):
    c = frozen(lab)
    name = 'fits/TRAIN0_90.sqlite' if artifact == 'sqlite' else 'ledgers/WF1_CRT_SWEEP_RECLAIM_BASE.json.gz'
    path = c.out / name
    path.parent.mkdir()
    path.write_bytes(b'preserved interruption evidence')
    with pytest.raises((ValueError, FileExistsError)):
        c.stage('WF1')
    assert path.read_bytes() == b'preserved interruption evidence'
