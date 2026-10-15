"""Synthetic boundary/source/statistics refusal cases; no historical outcomes."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_spike_tick_execution as study


def ledger(values, days=None):
    days = days or [study.DATES[0]]*len(values)
    times = [pd.Timestamp(day,tz='UTC')+pd.Timedelta(minutes=30*i) for i,day in enumerate(days)]
    return pd.DataFrame({'signal_time':times,'entry_time':times,'exit_time':times,
                         'gross_R':np.array(values)+.05,'net_R':values,
                         'reason':['time']*len(values),'ambiguous':[False]*len(values),
                         'holding_minutes':[15.]*len(values),'censored':[False]*len(values)})


def test_declared_dates_are_fixed_strata_not_favorable_signal_days():
    assert study.DATES == ('2026-04-11','2026-04-26','2026-05-12','2026-05-28',
                           '2026-06-13','2026-06-29','2026-07-15','2026-07-31',
                           '2026-08-16','2026-09-01','2026-09-17','2026-10-03')
    assert asdict(study.CONFIG)['stop_latency_ticks'] == 1
    assert study.CONFIG.max_gap_seconds == 1


def test_sample_bootstrap_uses12_scheduled_days_not_unsampled_calendar_zeros():
    values = [2.,-1.]*12
    days = [d for d in study.DATES for _ in range(2)]
    result=study.sampled_summary(ledger(values,days))
    assert result['completed']==24
    assert result['calendar_days']==12
    assert result['span_calendar_days']==176
    assert result['profit_factor']==2
    assert result['profit_factor_ci95']==[2,2]
    assert result['mean_net_R_ci95']==[.5,.5]
    assert result['cluster_se']==0
    assert result['selection_score'] is None
    assert result['inference_kind'].startswith('descriptive')


@pytest.mark.parametrize('values',[[],[1.],[0.]])
def test_empty_or_no_loss_ratio_stays_unknown(values):
    result=study.sampled_summary(ledger(values))
    assert result['profit_factor'] is None
    assert result['profit_factor_ci95']==[None,None]
    assert result['pf_undefined_replicates']==9999


def test_censored_outcomes_are_excluded_and_counted_never_zero_r():
    frame=ledger([1.,-2.])
    frame.loc[1,'censored']=True
    frame.loc[1,'net_R']=np.nan
    result=study.sampled_summary(frame)
    assert result['completed']==1 and result['censored']==1
    assert result['mean_net_R']==1


def test_unscheduled_payoff_is_refused():
    with pytest.raises(ValueError,match='Unscheduled'):
        study.sampled_summary(ledger([1.],['2026-04-12']))


def test_matched_difference_uses_same_completed_signal_not_full_cohort_means():
    a=ledger([3.,1.,20.])
    b=ledger([2.,-1.])
    result=study.matched_difference(a,b)
    assert result['matched_completed']==2
    assert result['tick_minus_coarse_mean_R']==1.5
    assert result['difference_ci95']==[1.5,1.5]
    assert result['unmatched_tick_completed']==1
    assert result['unmatched_coarse_completed']==0


def test_duplicate_issuance_refuses_many_to_many_matching():
    a=ledger([1.,2.])
    a.loc[1,'signal_time']=a.loc[0,'signal_time']
    with pytest.raises(pd.errors.MergeError):
        study.matched_difference(a,ledger([1.]))


def test_clock_comparison_is_day_paired_but_not_equal_portfolio_exposure():
    a=ledger([2.,-1.]*12,[d for d in study.DATES for _ in range(2)])
    b=ledger([0.]*12,list(study.DATES))
    result=study.sampled_clock_comparison(a,b)
    assert result['mean_R_difference']==.5
    assert result['difference_ci95']==[.5,.5]
    assert result['descriptive_only'] is True
    assert 'different issuance exposure' in result['comparison']


def test_stop_diagnostics_keep_gap_overshoot_and_successor_recovery_separate():
    frame=ledger([-1.])
    frame['reason']='sl';frame['entry']=100.;frame['side']=1
    frame['atr']=2.;frame['trigger_quote']=94.;frame['exit']=95.
    result=study.stop_diagnostics(frame)
    assert result['mean_trigger_overshoot_R']==.5
    assert result['mean_successor_move_R']==.25
    assert result['completed_stops']==1


def test_reconciliation_requires_full60tick_minute_and_price_identity():
    date=study.DATES[0]
    index=pd.date_range(date,periods=60,freq='s',tz='UTC')
    ticks=pd.DataFrame({'quote':np.arange(60)+100.},index=index)
    m1=pd.DataFrame({'open':[100.],'high':[159.],'low':[100.],'close':[159.]},index=index[:1])
    value=study.reconcile(ticks,m1,date)
    assert value['complete_minutes']==1 and value['unknown_minutes']==1439
    assert value['max_abs_difference']==0
    damaged=ticks.drop(index[30])
    value=study.reconcile(damaged,m1,date)
    assert value['complete_minutes']==0 and value['unknown_minutes']==1440
    m1.loc[index[0],'close']=158
    with pytest.raises(ValueError,match='price mismatch'):
        study.reconcile(ticks,m1,date)


def test_complete_tick_minute_missing_from_frozen_m1_is_not_replaced():
    index=pd.date_range(study.DATES[0],periods=60,freq='s',tz='UTC')
    ticks=pd.DataFrame({'quote':100.},index=index)
    m1=pd.DataFrame(columns=['open','high','low','close'],index=pd.DatetimeIndex([],tz='UTC'))
    with pytest.raises(ValueError,match='missing from frozen'):
        study.reconcile(ticks,m1,study.DATES[0])


def source_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    start=int(pd.Timestamp(study.DATES[0],tz='UTC').timestamp())
    csv=tmp_path/'quotes.csv'
    pd.DataFrame({'epoch':[start,start+1],'quote':[100.,101.]}).to_csv(csv,index=False)
    raw=tmp_path/'raw.jsonl';raw.write_text('{}\n')
    pages=tmp_path/'pages.json';pages.write_text('[]\n')
    return {'clean_file':'quotes.csv','clean_sha256':study.digest(csv),
            'raw_pages_file':'raw.jsonl','raw_pages_sha256':study.digest(raw),
            'page_audit_file':'pages.json','page_audit_sha256':study.digest(pages),
            'start_epoch':start,'end_exclusive_epoch':start+86400,'rows':2,
            'expected_grid_rows':86400,'missing_seconds':86398,'gap_free':False}


@pytest.mark.parametrize('field',['clean_file','raw_pages_file','page_audit_file'])
def test_acquisition_hash_layers_are_all_immutable(tmp_path,monkeypatch,field):
    value=source_fixture(tmp_path,monkeypatch)
    study.load_ticks(value)
    (tmp_path/value[field]).write_text('tampered\n')
    with pytest.raises(ValueError,match='bytes changed'):
        study.load_ticks(value)


@pytest.mark.parametrize('change',[
    {'missing_seconds':0},{'gap_free':True},{'rows':1},{'expected_grid_rows':2},
    {'start_epoch':0},{'end_exclusive_epoch':0}])
def test_source_grid_and_gap_metadata_cannot_overstate_coverage(tmp_path,monkeypatch,change):
    value=source_fixture(tmp_path,monkeypatch);value.update(change)
    with pytest.raises(ValueError):
        study.load_ticks(value)


@pytest.mark.parametrize('epochs,quotes',[
    ([1,1],[100,100]),([2,1],[100,100]),([1.5,2],[100,100]),
    ([1,2],[100,np.inf]),([1,2],[100,-1])])
def test_noncanonical_tick_arrays_are_refused(tmp_path,monkeypatch,epochs,quotes):
    value=source_fixture(tmp_path,monkeypatch)
    pd.DataFrame({'epoch':epochs,'quote':quotes}).to_csv(tmp_path/'quotes.csv',index=False)
    value['clean_sha256']=study.digest(tmp_path/'quotes.csv')
    with pytest.raises(ValueError):
        study.load_ticks(value)


def page_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    value={'symbol':'BOOM600','start_epoch':100,'end_exclusive_epoch':110,
           'page_audit_file':'pages.jsonl','raw_pages_file':'raw.jsonl','clean_file':'quotes.csv',
           'request_errors':[],'recovered_retry_lineage':[]}
    endpoint='wss://api.derivws.com/trading/v1/options/ws/public'
    request={'ticks_history':'BOOM600','start':100,'end':109,'style':'ticks','count':1000,'req_id':1}
    wire=json.dumps({'msg_type':'history','req_id':1,'history':{'times':list(range(100,110)),'prices':[100]*10}})
    key=hashlib.sha256(json.dumps({'endpoint':endpoint,'request':request},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    sha=hashlib.sha256(wire.encode()).hexdigest()
    pages=[{'request':request,'endpoint':endpoint,'request_key':key,'accepted':True,
            'rows':10,'oldest_epoch':100,'newest_epoch':109,'wire_sha256':sha}]
    raw=[{'request':request,'endpoint':endpoint,'request_key':key,'response_wire':wire,'wire_sha256':sha}]
    write_pages(tmp_path,pages,raw)
    pd.DataFrame({'epoch':range(100,110),'quote':[100]*10}).to_csv(tmp_path/'quotes.csv',index=False)
    return value,pages,raw,{'page_size':1000,'endpoint':endpoint}


def write_pages(path,pages,raw):
    (path/'pages.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pages))
    (path/'raw.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in raw))


@pytest.mark.parametrize('mutation',[
    ('start',101),('end',200),('ticks_history','CRASH600'),('count',5000),
    ('style','candles'),('authorize','secret')])
def test_page_bounds_start_and_public_whitelist_are_enforced(tmp_path,monkeypatch,mutation):
    value,pages,raw,declaration=page_fixture(tmp_path,monkeypatch)
    study.validate_pages(value,declaration)
    pages[0]['request'][mutation[0]]=mutation[1]
    write_pages(tmp_path,pages,raw)
    with pytest.raises(ValueError,match='chain/request'):
        study.validate_pages(value,declaration)


def test_silent_default_start_fallback_to_current_ticks_is_rejected(tmp_path,monkeypatch):
    value,pages,raw,declaration=page_fixture(tmp_path,monkeypatch)
    pages[0].update(oldest_epoch=10000,newest_epoch=10009)
    response=json.loads(raw[0]['response_wire']);response['history']['times']=list(range(10000,10010))
    raw[0]['response_wire']=json.dumps(response)
    raw[0]['wire_sha256']=pages[0]['wire_sha256']=hashlib.sha256(raw[0]['response_wire'].encode()).hexdigest()
    write_pages(tmp_path,pages,raw)
    with pytest.raises(ValueError,match='silently adjusted'):
        study.validate_pages(value,declaration)


def test_only_exact_successfully_recovered_error_requests_allowed(tmp_path,monkeypatch):
    value,pages,raw,declaration=page_fixture(tmp_path,monkeypatch)
    error={'request':pages[0]['request'],'endpoint':pages[0]['endpoint'],'request_key':pages[0]['request_key'],
           'error_type':'TimeoutError','error':'timeout','wire_sha256':None}
    failed={'request':pages[0]['request'],'endpoint':pages[0]['endpoint'],'request_key':pages[0]['request_key'],
            'accepted':False,'recoverable':True,'wire_sha256':None,'error':error}
    failed_raw={'request':pages[0]['request'],'endpoint':pages[0]['endpoint'],'request_key':pages[0]['request_key'],
                'response_wire':None,'wire_sha256':None,'error':error}
    pages.insert(0,failed);raw.insert(0,failed_raw)
    value['request_errors']=[error]
    value['recovered_retry_lineage']=[{'error_index':0,'successful_page_audit_index':1,'request_key':error['request_key'],
                                      'successful_wire_sha256':pages[1]['wire_sha256']}]
    write_pages(tmp_path,pages,raw)
    study.validate_pages(value,declaration)
    value['recovered_retry_lineage'][0]['successful_page_audit_index']=0
    with pytest.raises(ValueError,match='Unmatched'):
        study.validate_pages(value,declaration)


def test_raw_wire_cannot_be_replaced_by_consistent_looking_page_metadata(tmp_path,monkeypatch):
    value,pages,raw,declaration=page_fixture(tmp_path,monkeypatch)
    raw[0]['response_wire']=raw[0]['response_wire'].replace('100, 100','99, 100',1)
    write_pages(tmp_path,pages,raw)
    with pytest.raises(ValueError,match='raw response hash'):
        study.validate_pages(value,declaration)


def test_clean_quotes_must_reconstruct_exactly_from_raw_prices(tmp_path,monkeypatch):
    value,_,_,declaration=page_fixture(tmp_path,monkeypatch)
    csv=pd.read_csv(tmp_path/'quotes.csv');csv.loc[1,'quote']=99
    csv.to_csv(tmp_path/'quotes.csv',index=False)
    with pytest.raises(ValueError,match='differ from preserved raw'):
        study.validate_pages(value,declaration)


def test_post_result_evaluation_guard_runs_before_reading_sources(tmp_path,monkeypatch):
    (tmp_path/'results.json').write_text('{}')
    monkeypatch.setattr(study,'frozen_sources',lambda _:pytest.fail('must not read after existing result'))
    with pytest.raises(ValueError,match='post-result overwrite'):
        study.evaluate(tmp_path)


def test_existing_declaration_guard_runs_before_original_model_access(tmp_path,monkeypatch):
    (tmp_path/'declaration.json').write_text('{}')
    monkeypatch.setattr(study,'frozen6',lambda _:pytest.fail('must not touch frozen model'))
    with pytest.raises(ValueError,match='overwrite a declaration'):
        study.declare(tmp_path)


@pytest.mark.parametrize('change',[
    ('config','page_size',5000),('config','symbols',['BOOM600']),
    ('config','collector_sha256','changed'),('summary','status','FAILED_PRESERVED'),
    ('summary','authentication_used',True),('summary','account_or_order_requests',True)])
def test_incomplete_or_changed_acquisition_is_refused_before_prices(tmp_path,monkeypatch,change):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'DATA',tmp_path)
    endpoint='wss://api.derivws.com/trading/v1/options/ws/public'
    config={'symbols':list(study.SYMBOLS),'dates':list(study.DATES),'endpoint':endpoint,
            'page_size':1000,'cadence_seconds':1,'collector_sha256':'fixed',
            'safety':dict.fromkeys(('LIVE_TRADING','READY_FOR_LIVE','LIVE_ALLOWED','OPENED_TRADES'),False)}
    d={'endpoint':endpoint,'page_size':1000,'code_hashes':{'scripts/collect_spike_ticks.py':'fixed'}}
    summary={'config':config,'config_sha256':None,'status':'COMPLETE','authentication_used':False,
             'account_or_order_requests':False,'safety':config['safety']}
    target,key,value=change
    (config if target=='config' else summary)[key]=value
    (tmp_path/'acquisition_config.json').write_text(json.dumps(config))
    summary['config_sha256']=study.digest(tmp_path/'acquisition_config.json')
    (tmp_path/'acquisition_summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(study,'frozen_declaration',lambda _:d)
    monkeypatch.setattr(study,'frozen6',lambda _:({},'unused'))
    monkeypatch.setattr(study,'load_m1',lambda _:pytest.fail('must reject before reading prices'))
    with pytest.raises(ValueError,match='configuration/status'):
        study.freeze_data(tmp_path)


def test_frozen_acquisition_summary_bytes_cannot_change_after_source_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    (tmp_path/'declaration.json').write_text('{}')
    (tmp_path/'acquisition_summary.json').write_text('{"status":"COMPLETE"}')
    data={'declaration_sha256':study.digest(tmp_path/'declaration.json'),'sources':{},
          'acquisition_files':{'acquisition_summary.json':study.digest(tmp_path/'acquisition_summary.json')}}
    (tmp_path/'tick_sources.json').write_text(json.dumps(data))
    (tmp_path/'tick_sources.sha256').write_text(study.digest(tmp_path/'tick_sources.json'))
    monkeypatch.setattr(study,'frozen_declaration',lambda _:{})
    study.frozen_sources(tmp_path)
    (tmp_path/'acquisition_summary.json').write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError,match='configuration or summary changed'):
        study.frozen_sources(tmp_path)
