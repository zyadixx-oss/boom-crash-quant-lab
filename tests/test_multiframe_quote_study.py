"""Synthetic orchestration tests; market/private files are never opened."""
import gzip
import json
from pathlib import Path

import pytest

from independent_profit_research.statistics import Claim
from independent_profit_research.execution import Quote
from independent_profit_research.policies import Signal
from independent_profit_research.tape import QuoteTape
from scripts import run_multiframe_quote_study as study

DECL = 'd'*64


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True)+'\n')


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'OUT', tmp_path/'v2')
    monkeypatch.setattr(study, 'DOCS', tmp_path/'docs')
    monkeypatch.setattr(study.original, 'OUT', tmp_path/'v1')
    monkeypatch.setattr(study.original, 'DATA', tmp_path/'data')
    monkeypatch.setattr(study.original, 'declaration', lambda: ({}, 'a'*64))
    monkeypatch.setattr(study.original, 'check_receipt', lambda p: None)
    monkeypatch.setattr(study.original, 'records', lambda *args: [])
    for key in study.SAFETY_FIELDS:
        monkeypatch.delenv(key, raising=False)
    study.DOCS.mkdir()
    (study.DOCS/'PROTOCOL_DRAFT.ar.md').write_text('DRAFT / NOT FROZEN / NOT TESTED')
    put(study.original.OUT/'declaration.json', {'artificial_fixture': True})
    put(study.original.OUT/'final_summary.json', {'results': {}})
    put(study.original.DATA/'acquisition_declaration.json', {'artificial_fixture': True})
    put(study.original.DATA/'acquisition_state.json', {'status': 'COMPLETED'})
    for symbol in study.SYMBOLS:
        for end in (30, 50, 70, 90):
            put(study.original.OUT/'fits'/f'{symbol}_TRAIN0_{end}.json', {
                'declaration_sha256': 'a'*64, 'symbol': symbol, 'train_start_day': 0,
                'train_end_day_exclusive': end, 'uses_validation_or_final': False,
                'uses_strategy_payoff': False, 'source_files': {}, 'U_scaled': 1000})
    monkeypatch.setattr(study, 'SOURCE_FILES', ['engine.py'])
    (tmp_path/'engine.py').write_text('# artificial frozen implementation\n')
    validation = tmp_path/'validation.json'
    put(validation, {'status': 'PASS', 'market_strategy_computations': 0})
    return validation


def cases(monkeypatch, passing=None):
    monkeypatch.setattr(study, 'declaration', lambda: ({}, DECL))
    for symbol in study.SYMBOLS:
        for policy in study.POLICY_IDS:
            key = symbol+'/'+policy
            for stage in ('WF1', 'WF2', 'WF3', 'VALIDATION'):
                path = study.case_path(symbol, stage, policy, 'BASE')
                ledger = study.OUT/'ledgers'/(path.stem+'.json.gz')
                ledger.parent.mkdir(parents=True, exist_ok=True)
                ledger.write_bytes(gzip.compress(b'[]', mtime=0))
                study.create(path, {'declaration_sha256': DECL, 'source_files': {},
                    'symbol': symbol, 'stage': stage, 'policy': policy, 'scenario': 'BASE',
                    'stage_start_ms': study.original.START_MS+study.STAGES[stage][0]*study.original.DAY_MS,
                    'stage_end_exclusive_ms': study.original.START_MS+study.STAGES[stage][1]*study.original.DAY_MS,
                    'ledger_path': str(ledger.relative_to(study.ROOT)), 'ledger_sha256': study.sha(ledger),
                    'summary': {'sum_R': 2 if key == passing else -1,
                                'completed': 25, 'unknown': 0, 'remaining_censored': False}})


def test_freeze_and_runtime_pins(isolated):
    study.freeze(isolated)
    value, digest = study.declaration()
    assert len(digest) == 64
    assert value['development_previously_exposed'] is True
    assert value['final_strategy_measured'] is False
    assert len(value['claims']) == 12 and value['holm_family_size'] == 24
    assert all(value[k] is False for k in study.SAFETY_FIELDS)
    assert len(value['reused_training_fit_pins']) == 8
    with pytest.raises(ValueError):
        study.freeze(isolated)


@pytest.mark.parametrize('mutated', ['engine.py', 'validation.json', 'v1/declaration.json',
    'data/acquisition_declaration.json', 'v1/fits/BOOM500_TRAIN0_30.json', 'docs/PROTOCOL_FROZEN.ar.md',
    'v1/final_summary.json', 'data/acquisition_state.json'])
def test_frozen_dependency_change_rejected(isolated, mutated):
    study.freeze(isolated)
    (study.ROOT/mutated).write_text('{}\n')
    with pytest.raises(ValueError):
        study.declaration()


@pytest.mark.parametrize('report', [{'status': 'FAIL', 'market_strategy_computations': 0},
    {'status': 'PASS', 'market_strategy_computations': 1}, {'status': 'PASS'}])
def test_unverified_freeze_rejected(isolated, report):
    put(isolated, report)
    with pytest.raises(ValueError):
        study.freeze(isolated)
    assert not (study.OUT/'declaration.json').exists()


def test_previously_opened_final_rejected(isolated):
    put(study.original.OUT/'final_summary.json', {'results': {'ANY': {}}})
    with pytest.raises(ValueError, match='holdout'):
        study.freeze(isolated)


@pytest.mark.parametrize('field,bad', [('symbol', 'CRASH500'), ('train_start_day', 1),
    ('train_end_day_exclusive', 90), ('uses_strategy_payoff', True),
    ('uses_validation_or_final', True), ('declaration_sha256', 'b'*64),
    ('source_files', {'unexpected-source': 'b'*64})])
def test_fit_lineage_must_be_valid_before_freeze(isolated, field, bad):
    path = study.original.OUT/'fits/BOOM500_TRAIN0_30.json'
    fit = study.read(path)
    fit[field] = bad
    put(path, fit)
    with pytest.raises(ValueError, match='different training calendar'):
        study.freeze(isolated)


@pytest.mark.parametrize('key', study.SAFETY_FIELDS)
def test_unsafe_environment_rejected(isolated, monkeypatch, key):
    monkeypatch.setenv(key, 'true')
    with pytest.raises(ValueError, match='Unsafe'):
        study.freeze(isolated)


def test_derived_receipt_tamper_rejected(isolated):
    path = study.OUT/'test.json'
    study.create(path, {'safe': True})
    study.check(path)
    put(path, {'safe': False})
    with pytest.raises(ValueError):
        study.check(path)


def test_ledger_tamper_rejected(isolated, monkeypatch):
    cases(monkeypatch)
    path = study.case_path('BOOM500', 'WF1', sorted(study.POLICY_IDS)[0], 'BASE')
    saved = study.validated_case(path, DECL)
    (study.ROOT/saved['ledger_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='ledger'):
        study.validated_case(path, DECL)


@pytest.mark.parametrize('field,bad', [('symbol', 'CRASH500'), ('policy', 'ANOTHER'),
    ('stage', 'WF2'), ('scenario', 'STRESS'), ('stage_start_ms', 0),
    ('stage_end_exclusive_ms', 1), ('ledger_path', 'other.json.gz')])
def test_case_identity_cannot_be_swapped_even_with_registry_update(isolated, monkeypatch, field, bad):
    cases(monkeypatch)
    path = study.case_path('BOOM500', 'WF1', sorted(study.POLICY_IDS)[0], 'BASE')
    value = study.read(path)
    value[field] = bad
    put(path, value)
    index = study.read(study.OUT/'artifact_index.json')
    index[str(path.relative_to(study.OUT))] = study.sha(path)
    put(study.OUT/'artifact_index.json', index)
    with pytest.raises(ValueError):
        study.validated_case(path, DECL)


def test_zero_eligible_never_opens_final_or_computes_inference(isolated, monkeypatch):
    cases(monkeypatch)
    study.select()
    selected = study.read(study.OUT/'selection.json')
    assert selected['conditionally_eligible'] == []
    def forbidden(*args, **kwargs):
        pytest.fail('Failed policy must not open source, fit, build frames or compute final returns')
    for name in ('records', 'load'):
        monkeypatch.setattr(study.original, name, forbidden)
    for name in ('build_frames', 'calculate', 'analyze_daily'):
        monkeypatch.setattr(study, name, forbidden)
    study.run_stage('FINAL')
    study.summarize()
    report = study.read(study.OUT/'final_summary.json')
    assert report['results'] == {} and len(report['unmeasured_final_claims']) == 6
    assert report['qualified_profit_strategies'] == 0
    assert all(v['effective_p_value'] == v['holm_p_value'] == 1 for v in report['holm']['claims'].values())


def test_exactly_one_development_candidate(isolated, monkeypatch):
    key = 'BOOM500/'+sorted(study.POLICY_IDS)[0]
    cases(monkeypatch, passing=key)
    study.select()
    selected = study.read(study.OUT/'selection.json')
    assert selected['conditionally_eligible'] == [key]
    assert len(selected['development_receipts']) == 24


def test_selection_cannot_be_reinterpreted_after_development(isolated, monkeypatch):
    cases(monkeypatch)
    study.select()
    selected = study.read(study.OUT/'selection.json')
    selected['conditionally_eligible'] = ['BOOM500/'+sorted(study.POLICY_IDS)[0]]
    put(study.OUT/'selection.json', selected)
    # Even updating the registry cannot bypass the independently reproduced gates.
    index = study.read(study.OUT/'artifact_index.json')
    index['selection.json'] = study.sha(study.OUT/'selection.json')
    put(study.OUT/'artifact_index.json', index)
    with pytest.raises(ValueError, match='reproduced'):
        study.run_stage('FINAL')


def test_holm_keeps_both_generations_and_missing_claims(isolated):
    values = {key: Claim() for key in study.claim_ids()}
    key = study.claim_ids()[0]
    values[key] = Claim(.001, True)
    report = study.holm24(values)
    assert report['family_size'] == len(report['claims']) == 24
    assert report['claims']['V2/'+key]['holm_p_value'] == pytest.approx(.024)
    assert all(v['effective_p_value'] == 1 for k, v in report['claims'].items() if k.startswith('V1/'))
    values.pop(key)
    with pytest.raises(ValueError):
        study.holm24(values)


def test_holm_ties_monotonic_and_non_testable_p_not_promoted(isolated):
    values = {key: Claim(0, False) for key in study.claim_ids()}
    assert all(x['holm_p_value'] == 1 for x in study.holm24(values)['claims'].values())
    values[study.claim_ids()[0]] = object()
    with pytest.raises(ValueError):
        study.holm24(values)


def test_actual_clock_generator_evaluator_and_saved_case_integrate(isolated, monkeypatch):
    monkeypatch.setitem(study.STAGES, 'SYNTHETIC', (0, 1, 0))
    start = study.original.START_MS
    tape = QuoteTape(Quote(start+i*1000, 10_000_000_000+i*10000,
                          10_000_001_000+i*10000) for i in range(7201))
    frames = study.build_frames(tape, start, start+study.original.DAY_MS)
    result = study.calculate(tape, frames, 'BOOM500', 'SYNTHETIC', 'CLOCK', 'BASE', 1000, DECL, {})
    assert result['policy_id'] == study.CLOCK_POLICY_ID
    assert result['summary']['completed'] > 0
    assert result['cash_net'] is None and result['profit_qualified'] is False
    assert result['accounting_reconciliation']['summary_to_ledger_fsum_exact']
    assert study.validated_case(study.case_path('BOOM500', 'SYNTHETIC', 'CLOCK', 'BASE'), DECL) == result


def test_bounded_signal_stream_closes_without_consuming_future_suffix():
    seen, closed = [], []
    def source():
        try:
            for decision in (10, 20, 30, 40):
                seen.append(decision)
                yield Signal('A', 1, decision, decision+1, 1, 2)
        finally:
            closed.append(True)
    assert [s.decision_ms for s in study.bounded_signals(source(), 20, 30)] == [20]
    assert seen == [10, 20, 30] and closed == [True]
