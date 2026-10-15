"""Synthetic-only checks of the independent stdlib tick auditor."""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_spike_tick_execution as verify


def signal(time=0, side=1, atr=1., variant="SYNTHETIC"):
    return {"signal_time":time,"side":side,"atr":atr,"variant":variant}


def flat_ticks():
    return dict.fromkeys(range(4001),100.)


@pytest.mark.parametrize("side",[1,-1])
def test_entry_and_timeout_use_strict_next_second(side):
    ticks = flat_ticks();ticks[60]=150.;ticks[61]=100.;ticks[960]=100.+side;ticks[961]=100.+2*side
    rows, accounting = verify.scalar_replay(ticks,[signal(side=side)],0,4000)
    row = rows[0]
    assert (row['entry_time'],row['entry'],row['exit_time'],row['exit'],row['reason']) == (61,100.,961,100.+2*side,'time')
    assert row['gross_R'] == 1. and row['net_R'] == .95
    assert row['holding_minutes'] == 15. and accounting['completed'] == 1


@pytest.mark.parametrize("side",[1,-1])
@pytest.mark.parametrize("trigger,successor",[(96.,95.),(96.,101.),(98.,100.)])
def test_stop_trigger_irrevocable_next_quote_uncapped_or_recovered(side,trigger,successor):
    ticks=flat_ticks();ticks[70]=100.+side*(trigger-100);ticks[71]=100.+side*(successor-100)
    rows,_=verify.scalar_replay(ticks,[signal(side=side)],0,4000)
    row=rows[0]
    assert row['reason']=='sl' and row['trigger_time']==70 and row['exit_time']==71
    assert row['exit']==ticks[71] and row['gross_R']==(successor-100)/2


@pytest.mark.parametrize("missing",[62,500,960,961])
def test_first_required_gap_censors_without_return(missing):
    ticks=flat_ticks();del ticks[missing]
    rows,accounting=verify.scalar_replay(ticks,[signal()],0,4000)
    assert rows[0]['censored'] and rows[0]['missing_time']==missing
    assert rows[0]['exit_time']==960 and rows[0]['gross_R'] is None and rows[0]['net_R'] is None
    assert accounting['censored']==accounting['missing_path']==1


def test_missing_entry_has_no_row_and_reserves_planned_occupancy():
    ticks=flat_ticks();del ticks[61]
    rows,accounting=verify.scalar_replay(ticks,[signal(),signal(60),signal(900)],0,4000)
    assert accounting['missing_entry']==1 and accounting['overlap_skipped']==1
    assert len(rows)==1 and rows[0]['signal_time']==900  # nominal960 equals planned busy960


def test_censored_stop_keeps_trigger_and_reserves_planned_occupancy():
    ticks=flat_ticks();ticks[70]=97.;del ticks[71]
    rows,accounting=verify.scalar_replay(ticks,[signal(),signal(60)],0,4000)
    assert rows[0]['trigger_time']==70 and rows[0]['trigger_quote']==97.
    assert rows[0]['missing_time']==71 and rows[0]['exit_time']==960
    assert accounting['overlap_skipped']==1


def test_missing_quote_after_known_exit_is_irrelevant_and_actual_exit_frees_occupancy():
    ticks=flat_ticks();ticks[70]=97.;del ticks[80]
    rows,accounting=verify.scalar_replay(ticks,[signal(),signal(60)],0,4000)
    assert len(rows)==2 and not rows[0]['censored'] and not rows[1]['censored']
    assert accounting['overlap_skipped']==0


def test_deadline_trigger_has_priority_over_timeout():
    ticks=flat_ticks();ticks[960]=97.;ticks[961]=101.
    row=verify.scalar_replay(ticks,[signal()],0,4000)[0][0]
    assert row['reason']=='sl' and row['trigger_time']==960 and row['exit_time']==961


def test_planned_purge_not_rescued_by_early_exit():
    ticks=flat_ticks();ticks[70]=97.
    rows,accounting=verify.scalar_replay(ticks,[signal()],0,1800)
    assert not rows and accounting['purged']==1
    rows,accounting=verify.scalar_replay(ticks,[signal()],0,1860)
    assert len(rows)==1 and accounting['purged']==0


def test_outside_partition_counted_before_purge_or_entry():
    rows,accounting=verify.scalar_replay({},[signal(-60),signal(4000)],0,4000)
    assert not rows and accounting['outside_partition']==2


def test_same_time_signals_preserve_order_and_one_position():
    ticks=flat_ticks()
    rows,accounting=verify.scalar_replay(ticks,[signal(variant='FIRST'),signal(variant='SECOND')],0,4000)
    assert rows[0]['variant']=='FIRST' and len(rows)==1 and accounting['overlap_skipped']==1


def test_summary_pf_null_empty_and_no_loss_censor_not_zero():
    assert verify.summary([])['profit_factor'] is None and verify.summary([])['mean_net_R'] is None
    ticks=flat_ticks();ticks[961]=104.
    rows,_=verify.scalar_replay(ticks,[signal()],0,4000)
    censored=dict(rows[0],censored=True,net_R=None,gross_R=None)
    summary=verify.summary(rows+[censored])
    assert summary['trades']==2 and summary['completed']==1 and summary['censored']==1
    assert summary['profit_factor'] is None and summary['mean_net_R']==1.95
    assert summary['calendar_days']==12 and summary['span_calendar_days']==176


def test_stop_overshoot_and_successor_move_are_separate():
    ticks=flat_ticks();ticks[70]=96.;ticks[71]=101.
    rows,_=verify.scalar_replay(ticks,[signal()],0,4000)
    assert verify.stop_summary(rows)['mean_trigger_overshoot_R']==1.
    assert verify.stop_summary(rows)['mean_successor_move_R']==2.5


def test_closed_risk_ruin_cannot_recover_on_later_win():
    ticks=flat_ticks();ticks[70]=.1;ticks[71]=.1
    rows,_=verify.scalar_replay(ticks,[signal(atr=.1)],0,4000)
    later=dict(rows[0],net_R=100.,gross_R=100.05,exit_time=1000,entry_time=990)
    result=verify.summary(rows+[later])
    assert result['equity_ruin'] and result['closed_trade_return']==-1. and result['closed_trade_max_drawdown']==1.


def test_causal_atr_uses_only_closed_m5_bars():
    minutes={t:(100.,101.,99.,100.) for t in range(-5000//60*60,61,60)}
    assert verify.causal_atr(minutes,0)==2.
    minutes[0]=(100.,9999.,.01,100.)
    assert verify.causal_atr(minutes,0)==2.


def source_fixture(tmp_path,monkeypatch,retry=False):
    monkeypatch.setattr(verify,'ROOT',tmp_path)
    date='2026-04-11';start,end=verify.date_bounds(date)
    request={'ticks_history':'BOOM600','start':start,'end':end-1,'style':'ticks','count':1000,'req_id':1000}
    key=verify.request_key(request)
    wire=json.dumps({'req_id':1000,'msg_type':'history','history':{'times':[start,start,start+2],'prices':['100.00','100','101.00']}})
    sha=hashlib.sha256(wire.encode()).hexdigest()
    record={'endpoint':verify.ENDPOINT,'request':request,'request_key':key,'attempt':2 if retry else 1,
            'received_at_utc':'2026-10-05T15:01:00+00:00','response_wire':wire,'wire_sha256':sha}
    page={'endpoint':verify.ENDPOINT,'request':request,'request_key':key,'attempt':record['attempt'],'accepted':True,
          'wire_sha256':sha,'rows':3,'equal_duplicates_removed':1,'oldest_epoch':start,'newest_epoch':start+2}
    raws,pages,errors,lineage=[record],[page],[],[]
    if retry:
        failure_wire=json.dumps({'req_id':1000,'msg_type':'ticks_history','error':{'code':'RateLimit','message':'synthetic'}})
        failure_sha=hashlib.sha256(failure_wire.encode()).hexdigest()
        error={'endpoint':verify.ENDPOINT,'request':request,'request_key':key,'attempt':1,'error_type':'RateLimitError',
               'error':'synthetic retained error','wire_sha256':failure_sha}
        raws.insert(0,dict(record,attempt=1,response_wire=failure_wire,wire_sha256=failure_sha,error=error))
        pages.insert(0,{'endpoint':verify.ENDPOINT,'request':request,'request_key':key,'attempt':1,'accepted':False,
                       'recoverable':True,'wire_sha256':failure_sha,'error':error})
        errors=[error]
        lineage=[{'error_index':0,'successful_page_audit_index':1,'request_key':key,'successful_wire_sha256':sha,
                  'endpoint':verify.ENDPOINT,'start_epoch':start,'end_epoch':end-1}]
    rawfile=tmp_path/'boom600_2026-04-11_raw_pages.jsonl';auditfile=tmp_path/'boom600_2026-04-11_page_audit.jsonl'
    rawfile.write_text(''.join(json.dumps(r)+'\n' for r in raws));auditfile.write_text(''.join(json.dumps(p)+'\n' for p in pages))
    clean=tmp_path/'boom600_2026-04-11_ticks_clean.csv'
    with clean.open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['epoch','quote']);writer.writerows([(start,'100.00'),(start+2,'101.00')])
    value={'symbol':'BOOM600','date':date,'start_epoch':start,'end_exclusive_epoch':end,'endpoint':verify.ENDPOINT,
           'cadence_seconds':1,'expected_grid_rows':86400,'normalization_valid':True,'fills_or_interpolations':False,
           'authentication_used':False,'features_labels_or_tick_outcomes_computed':False,
           'safety':dict.fromkeys(verify.FLAGS,False),'config_sha256':'0'*64,
           'clean_file':clean.name,'clean_sha256':verify.digest(clean),'raw_pages_file':rawfile.name,'raw_pages_sha256':verify.digest(rawfile),
           'page_audit_file':auditfile.name,'page_audit_sha256':verify.digest(auditfile),
           'completed_at_utc':'2026-10-05T15:02:00+00:00','request_errors':errors,'recovered_retry_lineage':lineage,
           'rows':2,'first_epoch':start,'last_epoch':start+2,'missing_seconds':86398,
           'gaps':verify.gaps([start,start+2],start,end),'gap_free':False,'equal_duplicates_removed':1,'successful_page_count':1}
    checkpoint={'config_sha256':value['config_sha256'],'symbol':'BOOM600','date':date,'start_epoch':start,'end_exclusive_epoch':end,
                'endpoint':verify.ENDPOINT,'next_end':start-1,'page_size':1000,'success_pages':1,'request_errors':errors,
                'semantic_failure':False,'raw_sha256':value['raw_pages_sha256'],'audit_sha256':value['page_audit_sha256']}
    (tmp_path/'boom600_2026-04-11_checkpoint.json').write_text(json.dumps(checkpoint))
    declaration={'page_size':1000,'run_utc':'2026-10-05T15:00:00+00:00'}
    return value,declaration,raws,pages


@pytest.mark.parametrize('retry',[False,True])
def test_source_raw_decimal_normalization_gaps_retry_and_checkpoint(tmp_path,monkeypatch,retry):
    value,declaration,_,_=source_fixture(tmp_path,monkeypatch,retry)
    audit=verify.Audit();ticks,report=verify.audit_source(value,declaration,audit)
    assert len(ticks)==2 and report['missing_seconds']==86398 and report['equal_duplicates_removed']==1
    assert report['recovered_errors']==int(retry) and not audit.failures


@pytest.mark.parametrize('field,bad',[('missing_seconds',0),('gap_free',True),('equal_duplicates_removed',0),
                                    ('first_epoch',0),('successful_page_count',2),('gaps',[])])
def test_source_metadata_cannot_overstate_coverage(tmp_path,monkeypatch,field,bad):
    value,declaration,_,_=source_fixture(tmp_path,monkeypatch)
    value[field]=bad
    with pytest.raises(ValueError):verify.audit_source(value,declaration,verify.Audit())


def test_failure_omission_or_wrong_recovery_is_rejected(tmp_path,monkeypatch):
    value,declaration,_,_=source_fixture(tmp_path,monkeypatch,True)
    value['recovered_retry_lineage'][0]['end_epoch']-=1
    with pytest.raises(ValueError):verify.audit_source(value,declaration,verify.Audit())


@pytest.mark.parametrize('mutation',['reqid','unordered','fractional','bool_epoch','bool_quote','negative','unequal_duplicate','array_length','msgtype','server_error','out_of_bounds'])
def test_frozen_looking_wire_and_audit_hashes_cannot_hide_invalid_ticks(tmp_path,monkeypatch,mutation):
    value,declaration,raws,pages=source_fixture(tmp_path,monkeypatch)
    response=json.loads(raws[0]['response_wire']);start=value['start_epoch']
    if mutation=='reqid':response['req_id']=99
    if mutation=='unordered':response['history']['times']=[start+2,start,start]
    if mutation=='fractional':response['history']['times'][1]=start+.5
    if mutation=='bool_epoch':response['history']['times'][0]=True
    if mutation=='bool_quote':response['history']['prices'][0]=True
    if mutation=='negative':response['history']['prices'][0]=-1
    if mutation=='unequal_duplicate':response['history']['prices'][1]='99'
    if mutation=='array_length':response['history']['prices'].pop()
    if mutation=='msgtype':response['msg_type']='ticks_history'
    if mutation=='server_error':response['error']={'code':'InputValidationFailed'}
    if mutation=='out_of_bounds':response['history']['times'][2]=value['end_exclusive_epoch']
    raws[0]['response_wire']=json.dumps(response)
    fingerprint=hashlib.sha256(raws[0]['response_wire'].encode()).hexdigest()
    raws[0]['wire_sha256']=pages[0]['wire_sha256']=fingerprint
    for field,records,hashfield in [('raw_pages_file',raws,'raw_pages_sha256'),('page_audit_file',pages,'page_audit_sha256')]:
        path=tmp_path/value[field];path.write_text(''.join(json.dumps(row)+'\n' for row in records));value[hashfield]=verify.digest(path)
    with pytest.raises(ValueError):verify.audit_source(value,declaration,verify.Audit())


def test_final_checkpoint_hash_and_error_history_are_required(tmp_path,monkeypatch):
    value,declaration,_,_=source_fixture(tmp_path,monkeypatch,True)
    path=tmp_path/'boom600_2026-04-11_checkpoint.json';checkpoint=json.loads(path.read_text())
    checkpoint['request_errors']=[];path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):verify.audit_source(value,declaration,verify.Audit())


def test_request_attempt_time_is_not_allowed_before_declaration(tmp_path,monkeypatch):
    value,declaration,_,_=source_fixture(tmp_path,monkeypatch)
    declaration['run_utc']='2026-10-05T15:01:01+00:00'
    with pytest.raises(ValueError):verify.audit_source(value,declaration,verify.Audit())


def test_canonical_decimal_string_is_verified_not_only_numeric_equality(tmp_path,monkeypatch):
    value,declaration,_,_=source_fixture(tmp_path,monkeypatch)
    path=tmp_path/value['clean_file'];path.write_text(path.read_text().replace('100.00','100'))
    value['clean_sha256']=verify.digest(path)
    with pytest.raises(ValueError):verify.audit_source(value,declaration,verify.Audit())


@pytest.mark.parametrize('wire',['{"a":1,"a":2}','{"quote":NaN}','{"quote":Infinity}'])
def test_strict_original_wire_json(wire):
    with pytest.raises(ValueError):verify.strict_wire(wire)


def test_second_gap_ranges_include_boundaries():
    assert verify.gaps([2,4],0,7)==[{'start_epoch':0,'end_exclusive_epoch':2,'missing_seconds':2},
                                  {'start_epoch':3,'end_exclusive_epoch':4,'missing_seconds':1},
                                  {'start_epoch':5,'end_exclusive_epoch':7,'missing_seconds':2}]


def test_nonportable_paths_and_unsafe_flags_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(verify,'ROOT',tmp_path)
    with pytest.raises(ValueError):verify.source_path('../bad.csv')
    with pytest.raises(ValueError):verify.source_path('/tmp/bad.csv')
    with pytest.raises(ValueError):verify.Audit().safety({'LIVE_TRADING':True},'synthetic')


@pytest.mark.parametrize('minute',[1,15,29,31,59])
def test_saved_issuance_rejects_minute_outside_frozen_utc00_30_clock(tmp_path,minute):
    path=tmp_path/'synthetic_signals.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['signal_time','atr','side','variant','signal_close'])
        writer.writeheader();writer.writerow({'signal_time':f'2026-04-11T00:{minute:02d}:00+00:00','atr':1,
                                              'side':1,'variant':'SYNTHETIC','signal_close':100})
    with pytest.raises(ValueError,match='UTC00/30'):verify.read_signals(path)


@pytest.mark.parametrize('minute',[0,30])
def test_saved_issuance_accepts_declared_utc00_30_clock(tmp_path,minute):
    path=tmp_path/'synthetic_signals.csv'
    path.write_text(f'signal_time,atr,side,variant,signal_close\n2026-04-11T00:{minute:02d}:00+00:00,1,1,SYNTHETIC,100\n')
    assert len(verify.read_signals(path))==1


@pytest.mark.parametrize('quote',['1e400','1e-400','NaN','Infinity'])
def test_normalized_decimal_cannot_overflow_or_underflow_float_quote(quote):
    with pytest.raises(ValueError):verify.float_quotes({0:Decimal(quote)})


def test_normalized_decimal_float_conversion_preserves_normal_prices():
    assert verify.float_quotes({0:Decimal('123.45')})=={0:123.45}


def test_complete_minute_reconciliation_does_not_hide_second_gaps():
    date='2026-04-11';start,_=verify.date_bounds(date)
    ticks={start+t:100. for t in range(120)};minutes={start:(100.,100.,100.,100.),start+60:(100.,100.,100.,100.)}
    del ticks[start+30]
    report=verify.reconcile(ticks,minutes,date,verify.Audit())
    assert report['complete_minutes']==1 and report['unknown_minutes']==1439
    ticks[start+30]=101.
    with pytest.raises(ValueError):verify.reconcile(ticks,minutes,date,verify.Audit())


def test_verifier_uses_no_research_engine_or_scientific_imports():
    import ast
    tree=ast.parse(Path(verify.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
    assert not any(name and name.startswith(('app.','scripts.','numpy','pandas','scipy','sklearn')) for name in imports)
