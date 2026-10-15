"""Staged study orchestration on isolated synthetic files; never open market data."""
import gzip
import json
from types import SimpleNamespace

import pytest

from independent_profit_research.execution import Quote
from independent_profit_research.tape import QuoteTape
from scripts import run_independent_profit_study as study

DECL_SHA = 'd'*64


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for key in study.SAFETY_FIELDS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'DATA', tmp_path/'data')
    monkeypatch.setattr(study, 'OUT', tmp_path/'output')
    monkeypatch.setattr(study, 'DOCS', tmp_path/'docs')
    monkeypatch.setattr(study, 'SOURCE_FILES', ['source/engine.py'])
    for folder in (study.DATA, study.OUT, study.DOCS, tmp_path/'source'):
        folder.mkdir()
    (tmp_path/'source/engine.py').write_text('# Synthetic source pin only.\n')
    (study.DOCS/'PROTOCOL_DRAFT.ar.md').write_text('DRAFT / NOT FROZEN / NOT TESTED\n')
    acquisition = {'split_day_indices': [0,90,120,150], 'requests': [], 'source_pins': {}}
    put(study.DATA/'acquisition_declaration.json', acquisition)
    state = {'status': 'RUNNING', 'records': [], 'strategy_measurements': 0,
             'final_metrics_opened': False,
             'declaration_sha256': study.sha(study.DATA/'acquisition_declaration.json')}
    put(study.DATA/'acquisition_state.json', state)
    validation = tmp_path/'premeasurement.json'
    put(validation, {'status': 'PASS', 'market_strategy_computations': 0})
    return SimpleNamespace(root=tmp_path, validation=validation,
                           acquisition=acquisition, state=state)


def add_source(workspace, day, symbol='BOOM500'):
    prefix = f'Artificial_{symbol}_{day:03d}'
    start = study.START_MS+day*study.DAY_MS
    end = start+study.DAY_MS
    request = {'ordinal': day*2+(symbol=='CRASH500'), 'symbol': symbol,
               'day_index': day, 'partition': 'development' if day<120 else 'sealed_final',
               'prefix': prefix, 'start_ms': start, 'end_exclusive_ms': end,
               'seconds': 86400,
               'start_utc': study.datetime.datetime.fromtimestamp(start/1000,study.datetime.UTC).isoformat()}
    folder=study.DATA/request['partition']/prefix
    folder.mkdir(parents=True)
    receipt=folder/(prefix+'_receipt.tsv')
    raw=folder/(prefix+'_ticks.tsv')
    verification=folder/'independent_verification.json'
    receipt.write_text('field\tvalue\nsymbol\tArtificial fixture, never broker evidence\n')
    raw.write_text('time\ttime_msc\tbid\task\tlast\tvolume\tflags\tvolume_real\n'
                   f'{start//1000}\t{start}\t100\t101\t0\t0\t6\t0\n')
    put(verification, {'status':'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN',
        'receipt_sha256':study.sha(receipt),'raw_sha256':study.sha(raw),
        'requested_start_msc':start,'requested_end_exclusive_msc':end,
        'symbol':'Boom 500 Index' if symbol=='BOOM500' else 'Crash 500 Index',
        'rows':1,'errors':[],'cost_verified':False,'execution_verified':False,
        'profit_tested':False,'full_tick_coverage_verified':False})
    record={**request,'verification_status':'QUOTES_VALIDATED_COVERAGE_NOT_PROVEN',
            'files':{str(path.relative_to(study.ROOT)):study.sha(path)
                     for path in (receipt,raw,verification)}}
    workspace.acquisition['requests'].append(request)
    workspace.state['records'].append(record)
    save_sources(workspace)
    return record


def save_sources(workspace):
    put(study.DATA/'acquisition_declaration.json',workspace.acquisition)
    workspace.state['declaration_sha256']=study.sha(study.DATA/'acquisition_declaration.json')
    put(study.DATA/'acquisition_state.json',workspace.state)


def synthetic_cases(workspace, monkeypatch, passing=None):
    monkeypatch.setattr(study,'declaration',lambda: ({},DECL_SHA))
    for symbol in study.SYMBOLS:
        for policy in study.POLICY_IDS:
            key=symbol+'/'+policy
            for stage in ('WF1','WF2','WF3','VALIDATION'):
                path,ledger=study.case_paths(symbol,stage,policy,'BASE',False)
                ledger.parent.mkdir(parents=True,exist_ok=True)
                ledger.write_bytes(gzip.compress(b'[]',mtime=0))
                study.create(path,{'declaration_sha256':DECL_SHA,'symbol':symbol,'policy':policy,
                    'stage':stage,'scenario':'BASE','baseline':False,'U_scaled':2,
                    'source_files':{},'ledger_path':str(ledger.relative_to(study.ROOT)),
                    'ledger_sha256':study.sha(ledger),
                    'summary':{'sum_R':1 if key==passing else -1,'completed':25,'unknown':0,
                               'remaining_censored':False},
                    'cash_net':None,'costs_verified':False,'execution_verified':False,
                    'profit_qualified':False})
    workspace.state['status']='COMPLETED'
    save_sources(workspace)


def test_failed_development_policies_never_open_final_sources(workspace,monkeypatch):
    synthetic_cases(workspace,monkeypatch)
    study.select()
    selection=study.read(study.OUT/'selection.json')
    assert selection['conditionally_eligible']==[]
    def forbidden(*args,**kwargs):
        pytest.fail('Failed policy must not fit, read final source, or evaluate final signals')
    for name in ('records','load','unit_fit','evaluate_case'):
        monkeypatch.setattr(study,name,forbidden)
    study.run_stage('FINAL')


def test_only_eligible_policy_is_measured_in_final(workspace,monkeypatch):
    policy=sorted(study.POLICY_IDS)[0]
    synthetic_cases(workspace,monkeypatch,'BOOM500/'+policy)
    study.select()
    calls=[]
    monkeypatch.setattr(study,'unit_fit',lambda symbol,end,decl:
        calls.append(('fit',symbol,end)) or {'U_scaled':2})
    def records(symbol,start,end):
        calls.append(('records',symbol,start,end))
        return [{'symbol':symbol,'day_index':120,'files':{}}]
    monkeypatch.setattr(study,'records',records)
    monkeypatch.setattr(study,'load',lambda record: SimpleNamespace(tape=QuoteTape([
        Quote(study.START_MS+120*study.DAY_MS,100,102)])))
    monkeypatch.setattr(study,'evaluate_case',lambda tape,symbol,stage,policy,scenario,baseline,*args:
        calls.append(('case',symbol,stage,policy,scenario,baseline)))
    study.run_stage('FINAL')
    assert ('fit','BOOM500',90) in calls
    assert [c for c in calls if c[0]=='records']==[('records','BOOM500',120,150)]
    cases=[c for c in calls if c[0]=='case']
    assert len(cases)==4 and all(c[1:4]==('BOOM500','FINAL',policy) for c in cases)


def test_final_waits_for_fixed_acquisition_completion(workspace,monkeypatch):
    synthetic_cases(workspace,monkeypatch,'BOOM500/'+sorted(study.POLICY_IDS)[0])
    study.select()
    workspace.state['status']='RUNNING'
    save_sources(workspace)
    monkeypatch.setattr(study,'records',lambda *args:pytest.fail('Final accessed before source completion'))
    with pytest.raises(ValueError,match='acquisition completes'):
        study.run_stage('FINAL')


def test_changed_development_receipt_blocks_final_before_read(workspace,monkeypatch):
    policy=sorted(study.POLICY_IDS)[0]
    synthetic_cases(workspace,monkeypatch,'BOOM500/'+policy)
    study.select()
    path,_=study.case_paths('BOOM500','WF1',policy,'BASE',False)
    value=study.read(path);value['summary']['sum_R']=999
    put(path,value)
    monkeypatch.setattr(study,'records',lambda *args:pytest.fail('Final opened despite changed selection evidence'))
    with pytest.raises(ValueError):study.run_stage('FINAL')


def test_registered_selection_cannot_override_failed_development_gates(workspace,monkeypatch):
    synthetic_cases(workspace,monkeypatch)
    receipts={}
    gates={}
    for symbol in study.SYMBOLS:
        for policy in study.POLICY_IDS:
            for stage in ('WF1','WF2','WF3','VALIDATION'):
                path,_=study.case_paths(symbol,stage,policy,'BASE',False)
                receipts[str(path.relative_to(study.ROOT))]=study.sha(path)
            gates[symbol+'/'+policy]=study.development_gate(validation_sum_R=-1,
                walk_forward_sum_R=[-1,-1,-1],completed=100,unknown=0)
    study.create(study.OUT/'selection.json',{'declaration_sha256':DECL_SHA,
        'conditionally_eligible':['BOOM500/'+sorted(study.POLICY_IDS)[0]],
        'gates':gates,'development_receipts':receipts,
        'final_strategy_measurements_before_selection':0,'profit_qualified':False,
        'cash_net':None,'costs_verified':False})
    monkeypatch.setattr(study,'records',lambda *args:pytest.fail('Forged eligibility opened final'))
    with pytest.raises(ValueError):study.run_stage('FINAL')


def test_records_require_every_requested_day_and_verified_pins(workspace):
    record=add_source(workspace,0)
    assert study.records('BOOM500',0,1)==[record]
    with pytest.raises(ValueError,match='not yet available'):study.records('BOOM500',0,2)
    raw=study.ROOT/next(path for path in record['files'] if path.endswith('_ticks.tsv'))
    raw.write_text(raw.read_text()+'tampered\n')
    with pytest.raises(ValueError,match='changed'):study.records('BOOM500',0,1)


@pytest.mark.parametrize('mutation', ['duplicate','wrong_calendar','wrong_symbol','wrong_path','empty_pins','failed_quality'])
def test_records_reject_misbound_or_ambiguous_inputs(workspace,mutation):
    record=add_source(workspace,0)
    if mutation=='duplicate':workspace.state['records'].append(record.copy())
    elif mutation=='wrong_calendar':
        record['start_ms']=study.START_MS+120*study.DAY_MS
        record['end_exclusive_ms']=record['start_ms']+study.DAY_MS
    elif mutation=='wrong_symbol':record['symbol']='CRASH500'
    elif mutation=='wrong_path':
        value=next(iter(record['files'].values()))
        record['files']={'unrelated.tsv':value}
    elif mutation=='empty_pins':record['files']={}
    else:record['verification_status']='FAIL_PRESERVED'
    save_sources(workspace)
    with pytest.raises(ValueError):study.records('BOOM500',0,1)


def test_declared_wrong_calendar_also_rejected(workspace):
    record=add_source(workspace,0)
    for item in (workspace.acquisition['requests'][0],record):
        item['start_ms']=study.START_MS+120*study.DAY_MS
        item['end_exclusive_ms']=item['start_ms']+study.DAY_MS
    save_sources(workspace)
    with pytest.raises(ValueError,match='calendar'):study.records('BOOM500',0,1)


def test_training_prefix_fit_never_loads_validation_or_final(workspace,monkeypatch):
    for day in (0,1,2,95,130):add_source(workspace,day)
    loaded=[]
    def load(record):
        day=record['day_index'];loaded.append(day)
        assert day<3,'Validation/final prices leaked into training U'
        spreads={0:[2,2],1:[4],2:[8]}[day]
        return SimpleNamespace(tape=QuoteTape(Quote(study.START_MS+day*study.DAY_MS+i,
            100,100+spread) for i,spread in enumerate(spreads)))
    monkeypatch.setattr(study,'load',load)
    first=study.unit_fit('BOOM500',2,DECL_SHA)
    assert loaded==[0,1] and first['U_scaled']==2
    loaded.clear()
    extended=study.unit_fit('BOOM500',3,DECL_SHA)
    assert loaded==[2] and extended['reused_training_days']==2
    assert extended['U_scaled']==3 and extended['positive_spread_rows']==4
    assert not extended['uses_validation_or_final'] and not extended['uses_strategy_payoff']
    assert len(extended['source_files'])==9


@pytest.mark.parametrize('field,value', [('symbol','CRASH500'),('train_end_day_exclusive',150),
    ('uses_validation_or_final',True),('source_files',{}),('U_scaled',777),
    ('spread_histogram_scaled',[[777,1]])])
def test_changed_cached_fit_fails_instead_of_reuse(workspace,monkeypatch,field,value):
    add_source(workspace,0)
    monkeypatch.setattr(study,'load',lambda record:SimpleNamespace(tape=QuoteTape([Quote(0,100,102)])))
    study.unit_fit('BOOM500',1,DECL_SHA)
    path=study.OUT/'fits'/'BOOM500_TRAIN0_1.json'
    changed=study.read(path);changed[field]=value;put(path,changed)
    monkeypatch.setattr(study,'load',lambda record:pytest.fail('Changed cache must not silently regenerate'))
    with pytest.raises(ValueError):study.unit_fit('BOOM500',1,DECL_SHA)


def test_no_positive_training_spread_has_no_fallback(workspace,monkeypatch):
    add_source(workspace,0)
    monkeypatch.setattr(study,'load',lambda record:SimpleNamespace(tape=QuoteTape([Quote(0,100,100)])))
    with pytest.raises(ValueError,match='No positive training spread'):
        study.unit_fit('BOOM500',1,DECL_SHA)
    assert not (study.OUT/'fits'/'BOOM500_TRAIN0_1.json').exists()


@pytest.mark.parametrize('target', ['source','acquisition','protocol','validation'])
def test_all_frozen_input_pins_are_enforced(workspace,target):
    study.freeze(workspace.validation)
    _,before=study.declaration()
    assert len(before)==64
    paths={'source':workspace.root/'source/engine.py',
           'acquisition':study.DATA/'acquisition_declaration.json',
           'protocol':study.DOCS/'PROTOCOL_FROZEN.ar.md',
           'validation':workspace.validation}
    path=paths[target];path.write_text(path.read_text()+'\nchanged\n')
    with pytest.raises(ValueError):study.declaration()


@pytest.mark.parametrize('evidence', [{'status':'FAIL','market_strategy_computations':0},
    {'status':'PASS','market_strategy_computations':1}, {'status':'PASS'}])
def test_freeze_requires_actual_passing_preverification(workspace,evidence):
    put(workspace.validation,evidence)
    with pytest.raises(ValueError):study.freeze(workspace.validation)
    assert not (study.DOCS/'PROTOCOL_FROZEN.ar.md').exists()
    assert not (study.OUT/'declaration.json').exists()


def test_failed_calendar_precheck_leaves_no_frozen_snapshot(workspace):
    workspace.acquisition['split_day_indices']=[0,70,100,150]
    save_sources(workspace)
    with pytest.raises(ValueError):study.freeze(workspace.validation)
    assert not (study.DOCS/'PROTOCOL_FROZEN.ar.md').exists()


def test_changed_cached_ledger_fails_without_regeneration(workspace,monkeypatch):
    policy=sorted(study.POLICY_IDS)[0]
    calls=[]
    monkeypatch.setattr(study,'evaluate_tape',lambda *args,**kwargs:
        calls.append('evaluate') or {'ledger':[],'summary':{'completed':0}})
    args=(QuoteTape([]),'BOOM500','WF1',policy,'BASE',False,2,DECL_SHA,{})
    created=study.evaluate_case(*args)
    ledger=study.ROOT/created['ledger_path']
    ledger.write_bytes(ledger.read_bytes()+b'changed')
    with pytest.raises(ValueError):study.evaluate_case(*args)
    assert calls==['evaluate']


def test_changed_development_ledger_blocks_selection(workspace,monkeypatch):
    synthetic_cases(workspace,monkeypatch)
    path,ledger=study.case_paths('BOOM500','WF1',sorted(study.POLICY_IDS)[0],'BASE',False)
    ledger.write_bytes(ledger.read_bytes()+b'changed')
    with pytest.raises(ValueError):study.select()
    assert not (study.OUT/'selection.json').exists()
