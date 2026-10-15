"""Synthetic-only checks for the independent tail-tick verifier."""
import ast
import math
from pathlib import Path

import pytest

from scripts import verify_spike_tick_tail as verify


def path(returns,side=1,times=None):
    quotes=[100.]
    for value in returns:quotes.append(quotes[-1]*math.exp(side*value))
    return times or list(range(len(quotes))),quotes


def test_integer_observed_row_cuts():
    assert verify.cutoffs(103)=={'40':41,'50':51,'60':61,'70':72,'100':103}
    assert verify.cutoffs(100)=={'40':40,'50':50,'60':60,'70':70,'100':100}


def test_calibration_uses_prefix_and_only_consecutive_pairs():
    times,quotes=path([.001,-.002,.003,-.004,.2],times=[0,1,2,100,101,102])
    scale,count=verify.fit_scale(times,quotes,5)
    assert count==3 and scale==pytest.approx(.002)
    quotes[-1]=1e200
    assert verify.fit_scale(times,quotes,5)==(scale,count)


@pytest.mark.parametrize('times,quotes,cut',[([0,1],[100.,100.],2),([0,2],[100.,101.],2),([0],[100.],1)])
def test_invalid_or_zero_calibration_refused_without_numerical_floor(times,quotes,cut):
    with pytest.raises(ValueError):verify.fit_scale(times,quotes,cut)


def test_even_median_averages_both_middle_values():
    times,quotes=path([.001,-.002,.003,-.004])
    assert verify.fit_scale(times,quotes,5)[0]==pytest.approx(.0025)


def test_current_event_has_strictly_prior_age_and_mark():
    times,quotes=path([.02,-.001,.03,-.001])
    rows=verify.decode(times,quotes,.001,1)
    assert rows[1].event and rows[1].age is None and rows[1].prior_mark is None
    assert rows[2].age==1 and rows[2].prior_mark==pytest.approx(.02)
    assert rows[3].event and rows[3].age==2 and rows[3].prior_mark==pytest.approx(.02)
    assert rows[4].age==1 and rows[4].prior_mark==pytest.approx(.03)


def test_threshold_equality_is_not_an_event():
    times,quotes=[0,1],[1.,2.]
    directed=verify.log_increment(quotes[0],quotes[1])
    scale=directed/10
    assert 10*scale==directed
    assert verify.decode(times,quotes,scale,1)[1].event is False


@pytest.mark.parametrize('side',[1,-1])
def test_600_age_boundary_and_direction(side):
    times,quotes=path([.02]+[-.00001]*600,side=side)
    rows=verify.decode(times,quotes,.001,side)
    assert rows[600].age==599 and rows[600].band=='lt_N'
    assert rows[601].age==600 and rows[601].band=='ge_N'


@pytest.mark.parametrize('gap',[4,86400])
def test_missing_pair_resets_age_and_mark_until_new_event(gap):
    times,quotes=path([.02,-.001,-.001,-.001,.02,-.001],times=[0,1,2,gap,gap+1,gap+2,gap+3])
    rows=verify.decode(times,quotes,.001,1)
    assert not rows[3].known and rows[3].event is None and rows[3].directed is None
    assert rows[4].known and rows[4].age is None and rows[4].prior_mark is None
    assert rows[5].event and rows[5].age is None and rows[6].age==1


def test_continuous_midnight_keeps_state():
    times,quotes=path([.02,-.001,-.001],times=[86398,86399,86400,86401])
    rows=verify.decode(times,quotes,.001,1)
    assert rows[2].age==1 and rows[3].age==2 and rows[2].known


def test_row_split_does_not_reset_state_or_drop_boundary_increment():
    times,quotes=path([.02]+[-.001]*99)
    rows=verify.decode(times,quotes,.001,1)
    result=verify.segment_summary(rows,40,50)
    assert result['valid_increments']==10 and result['unknown_age_exclusions']==0
    assert rows[40].age==39 and result['start_row']==40 and result['end_row_exclusive']==50


def test_band_mean_identity_and_null_conditional_groups():
    rows=[verify.Row(0,True,.02,True,100,.01,'lt_N'),verify.Row(1,True,-.001,False,101,.02,'lt_N')]
    result=verify.band_summary(rows,'lt_N')
    assert result['hazard']==.5 and result['events']==1 and result['non_event_count']==1
    assert result['unconditional_mean_d']==pytest.approx(result['mixture_identity_mean_d'])
    assert verify.band_summary(rows[:1],'lt_N')['non_event_mean_d'] is None
    assert verify.band_summary(rows[1:],'lt_N')['event_mean_d'] is None
    assert verify.band_summary([],'lt_N')['hazard'] is None


def test_unknown_age_exclusions_do_not_turn_return_or_event_into_zero():
    times,quotes=path([.02,0.,-.001])
    rows=verify.decode(times,quotes,.001,1)
    result=verify.segment_summary(rows,0,4)
    assert result['valid_increments']==3 and result['missing_pair_exclusions']==1
    assert result['unknown_age_exclusions']==1 and result['known_age_exposure_seconds']==2
    assert result['positive_increments']==1 and result['negative_increments']==1 and result['zero_increments']==1
    assert result['detected_events']==1 and result['bands']['lt_N']['events']==0


def test_segment_daily_units_only_observed_dates_keep_zero_band_exposure():
    rows=[verify.Row(0,False,None,None,None,None,None),verify.Row(1,True,.02,True,None,None,None),
          verify.Row(86400,False,None,None,None,None,None),verify.Row(86401,True,-.001,False,None,None,None)]
    result=verify.segment_summary(rows,0,4)
    assert result['observed_clusters']==2 and len(result['daily_band_sufficient_statistics'])==2
    assert all(item['ge_N']['exposure_seconds']==0 for item in result['daily_band_sufficient_statistics'].values())
    assert result['hazard_ratio_ge_N_over_lt_N'] is None


def test_falsification_requires_every_declared_draw_defined():
    assert verify.falsification(100,100,[.8,1.49],9999,True)=='large_overdue_effect_rejected'
    assert verify.falsification(100,100,[.8,1.49],9998,True)=='insufficient_evidence'
    assert verify.falsification(100,99,[.8,1.49],9999,True)=='insufficient_evidence'
    assert verify.falsification(100,100,[.8,1.5],9999,True)=='large_overdue_effect_not_rejected'
    assert verify.falsification(100,100,[None,None],9999,True)=='insufficient_evidence'
    assert verify.falsification(1000,1000,[.8,1.49],9999,False)=='detector_inadequate'


@pytest.mark.parametrize('side',[True,0,2,1.0,'1'])
def test_invalid_direction_refused(side):
    with pytest.raises(ValueError):verify.decode([0,1],[100.,101.],.001,side)


@pytest.mark.parametrize('quotes',[[100.,True],[100.,0.],[100.,float('inf')],[100.,float('nan')]])
def test_invalid_quote_refused(quotes):
    with pytest.raises(ValueError):verify.decode([0,1],quotes,.001,1)


@pytest.mark.parametrize('times',[[0,0],[1,0],[0,.5],[0,True]])
def test_invalid_second_grid_refused(times):
    with pytest.raises(ValueError):verify.decode(times,[100.,101.],.001,1)


def test_log_arithmetic_handles_extreme_positive_ratios():
    assert verify.log_increment(1e-300,1e300)==pytest.approx(math.log(1e300)-math.log(1e-300))
    assert verify.log_increment(1e300,1e-300)==pytest.approx(math.log(1e-300)-math.log(1e300))


def test_type7_quantiles_use_linear_interpolation_and_null_empty():
    assert verify.quantile([1.,2.,3.,4.],.5)==2.5
    assert verify.quantile([1.,2.,3.,4.],.25)==1.75
    assert verify.quantile([],.5) is None


def test_source_csv_and_repository_path_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(verify,'ROOT',tmp_path)
    path=tmp_path/'synthetic.csv';path.write_text('epoch,quote\n0,100\n1,101\n')
    assert verify.read_quotes(path)==([0,1],[100.,101.])
    path.write_text('epoch,quote\n1,100\n1,101\n')
    with pytest.raises(ValueError):verify.read_quotes(path)
    with pytest.raises(ValueError):verify.source_path('../bad.csv')


def test_all_false_and_no_research_engine_imports():
    audit=verify.Audit();audit.safety(dict.fromkeys(verify.FLAGS,False),'synthetic')
    assert audit.safety_values_checked==4 and not audit.failures
    with pytest.raises(ValueError):verify.Audit().safety({'LIVE_ALLOWED':True},'synthetic')
    tree=ast.parse(Path(verify.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
    assert not any(name and name.startswith(('app.','scripts.','numpy','pandas','scipy','sklearn')) for name in imports)


@pytest.mark.parametrize('actual,expected',[(True,1),(False,0),(True,1.),(0,False),(1,True)])
def test_boolean_cannot_masquerade_as_saved_numeric_counter(actual,expected):
    audit=verify.Audit();audit.equal('synthetic/counter',actual,expected)
    assert len(audit.failures)==1


def test_boolean_flags_require_identical_boolean_type():
    audit=verify.Audit();audit.equal('flag',False,False)
    assert not audit.failures


def test_recursive_point_comparison_preserves_float_tolerance_and_nulls():
    audit=verify.Audit()
    verify.compare_points(audit,'synthetic',{'bands':{'lt_N':{'hazard':.1+1e-15,'event_mean':None}},'interval':[.9,1.1]},
                          {'bands':{'lt_N':{'hazard':.1,'event_mean':None}},'interval':[.9,1.1]})
    assert not audit.failures
    verify.compare_points(audit,'synthetic/bad',{'counter':True},{'counter':1})
    assert len(audit.failures)==1


def literal_band(exposure,events,event_sum,non_event_sum):
    non_event=exposure-events
    hazard=events/exposure if exposure else None
    event_mean=event_sum/events if events else None
    non_event_mean=non_event_sum/non_event if non_event else None
    mean=(event_sum+non_event_sum)/exposure if exposure else None
    identity=hazard*event_mean+(1-hazard)*non_event_mean if events and non_event else None
    return {'exposure_seconds':exposure,'events':events,'hazard':hazard,'event_return_sum':event_sum,
            'event_mean_d':event_mean,'non_event_seconds':non_event,'non_event_return_sum':non_event_sum,
            'non_event_mean_d':non_event_mean,'return_sum':event_sum+non_event_sum,'unconditional_mean_d':mean,
            'decomposition_identity_available':identity is not None,'decomposition_reconstructed_mean_d':identity,
            'decomposition_identity_abs_error':abs(mean-identity) if identity is not None else None}


def literal_segment(start,lo,hi,calibration=False):
    # This fixture has119 event increments, then only -0.001 drift, no600s exposure.
    count=hi-lo
    events=119 if calibration else 0
    pairs=count-1 if calibration else count
    band=literal_band(238,118,2.36,-.12) if calibration else literal_band(count,0,0.,-.001*count)
    empty=literal_band(0,0,0.,0.)
    names=('p00','p01','p05','p25','p50','p75','p95','p99','p100')
    qvalues=[-.001]*5+[.02]*4 if calibration else [-.001]*9
    day='2026-04-11'
    return {'status':'MEASURED_EXPLORATORY','row_start_inclusive':lo,'row_end_exclusive':hi,'observed_rows':count,
            'first_observed_utc':verify.iso(start+lo),'last_observed_utc':verify.iso(start+hi-1),
            'observed_scheduled_days':[day],'observed_cluster_count':1,'valid_increments':pairs,
            'initial_or_gap_pair_exclusions':int(calibration),'unknown_age_exclusions':int(calibration),
            'known_age_increments':pairs-int(calibration),'detected_events':events,
            'events_with_unknown_prior_age':int(calibration),'directed_sign_counts':{'positive':events,'negative':pairs-events,'zero':0},
            'directed_return_quantiles':dict(zip(names,qvalues)),'bands':{'lt_N':band,'ge_N':empty},
            'daily_bands':[{'date':day,'observed_rows':count,'bands':{'lt_N':band,'ge_N':empty}}]}


def literal_undefined_comparison():
    return {'status':'DESCRIPTIVE_EXPLORATORY','older_over_younger_hazard_ratio':None,'hazard_ratio_ci95':[None,None],
            'bootstrap_repeats':9999,'bootstrap_seed':20261005,'valid_ratio_replicates':0,'undefined_ratio_replicates':9999,
            'paired_scheduled_day_resampling':True,'same_day_draws_for_both_bands':True,'observed_cluster_count':1,
            'observed_scheduled_days':['2026-04-11'],'zero_band_exposure_days':{'lt_N':0,'ge_N':1},
            'interval_omits_undefined_draws':True,'undefined_draws_block_falsification':True,
            'inference_kind':'conditional_on_fixed_detector_descriptive_iid_observed_scheduled_days',
            'training_median_refits_in_bootstrap':False,'no_contiguous_weekly_claim':True,
            'no_independent_tick_inference_claim':True,'no_strategy_or_profitability_claim':True,
            'falsification':{'status':'INSUFFICIENT_EVIDENCE','large_overdue_effect_falsified':False,
                             'reasons':['fewer_than100_events_in_at_least_one_band','not_all9999_bootstrap_ratios_defined','undefined_or_invalid_ratio_or_interval'],
                             'minimum_events_per_band':100,'hazard_ratio_upper_threshold':1.5,'all9999_ratios_defined':False,
                             'does_not_establish_independence':True,'does_not_test_profitability':True,'strategy_eligible':False}}


def measured_fixture(side=1):
    start=int(verify.timestamp('2026-04-11T00:00:00+00:00'))
    returns=[.02 if i<=237 and i%2 else -.001 for i in range(1,600)]
    _,quotes=path(returns,side=side);times=list(range(start,start+600))
    cuts={'40':240,'50':300,'60':360,'70':420,'100':600}
    saved={'side':side,'observed_rows':600,'row_cutoffs':cuts,'detector_fit_attempts':1,'no_detector_refit':True,
           'physical_spike_census':False,'strategy_eligible':False,'profit_factor':'NOT TESTED','actual_money_profit':'NOT TESTED',
           'detector':{'adequate':True,'median_abs_log_return':.001,'threshold':.01,'threshold_multiplier':10.,
                       'calibration_events':119,'calibration_source':'earliest40_percent_observed_rows_only'},
           'calibration40':literal_segment(start,0,240,True),
           'segments':{name:literal_segment(start,lo,hi) for name,(lo,hi) in
                       {'wf1':(240,300),'wf2':(300,360),'wf3':(360,420),'dev_validation':(240,420),'final30':(420,600)}.items()},
           'comparisons':{'dev_validation':literal_undefined_comparison(),'final30':literal_undefined_comparison()}}
    return times,quotes,saved


@pytest.mark.parametrize('symbol,side',[('BOOM600',1),('CRASH600',-1)])
def test_full_independent_symbol_verification_against_literal_fixture(symbol,side):
    times,quotes,saved=measured_fixture(side)
    audit=verify.Audit();report=verify.audit_symbol(times,quotes,saved,audit,symbol)
    assert report['detector_adequate'] and report['calibration_events']==119 and report['measured_later_segments']==5
    assert not audit.failures


def test_independent_symbol_verifier_detects_wrong_age_band_or_sign_count():
    times,quotes,saved=measured_fixture()
    saved['segments']['wf1']['directed_sign_counts']['positive']=1
    audit=verify.Audit();verify.audit_symbol(times,quotes,saved,audit,'BOOM600')
    assert any('directed_sign_counts/positive' in failure['check'] for failure in audit.failures)


def test_independent_verifier_blocks_falsification_on_one_undefined_saved_draw():
    segment=literal_segment(int(verify.timestamp('2026-04-11T00:00:00+00:00')),240,420)
    saved=literal_undefined_comparison();saved['valid_ratio_replicates']=9998;saved['undefined_ratio_replicates']=1
    saved['hazard_ratio_ci95']=[.9,1.2];saved['falsification']['large_overdue_effect_falsified']=True
    audit=verify.Audit();verify.comparison_policy(saved,segment,audit,'synthetic')
    assert any('large_overdue_effect_falsified' in failure['check'] for failure in audit.failures)


@pytest.mark.parametrize('bad',[-1,True,9998])
def test_bad_saved_replicate_counts_rejected(bad):
    segment=literal_segment(int(verify.timestamp('2026-04-11T00:00:00+00:00')),240,420)
    saved=literal_undefined_comparison();saved['undefined_ratio_replicates']=bad
    with pytest.raises(ValueError):verify.comparison_policy(saved,segment,verify.Audit(),'synthetic')


def test_zero_scale_inadequacy_keeps_every_later_segment_not_tested(monkeypatch):
    start=int(verify.timestamp('2026-04-11T00:00:00+00:00'));times=list(range(start,start+100));quotes=[100.]*100
    reason='zero_or_invalid_calibration_scale_or_no_consecutive_increment'
    cuts={'40':40,'50':50,'60':60,'70':70,'100':100}
    saved={'side':1,'observed_rows':100,'row_cutoffs':cuts,'detector_fit_attempts':1,'no_detector_refit':True,
           'physical_spike_census':False,'strategy_eligible':False,'profit_factor':'NOT TESTED','actual_money_profit':'NOT TESTED',
           'detector':{'adequate':False,'median_abs_log_return':None,'threshold':None,'threshold_multiplier':10.,'reason':reason,'calibration_events':None},
           'calibration40':{'status':'NOT TESTED','reason':reason,'row_start_inclusive':0,'row_end_exclusive':40},
           'segments':{name:{'status':'NOT TESTED','reason':reason,'row_start_inclusive':lo,'row_end_exclusive':hi} for name,(lo,hi) in
                       {'wf1':(40,50),'wf2':(50,60),'wf3':(60,70),'dev_validation':(40,70),'final30':(70,100)}.items()},
           'comparisons':{'status':'NOT TESTED','reason':reason}}
    def forbidden(*args):raise AssertionError('Inadequate detector must not decode later rows')
    monkeypatch.setattr(verify,'decode',forbidden)
    audit=verify.Audit();report=verify.audit_symbol(times,quotes,saved,audit,'BOOM600')
    assert not audit.failures and not report['detector_adequate'] and report['measured_later_segments']==0
    saved['segments']['wf1']['valid_increments']=10
    with pytest.raises(ValueError):verify.audit_symbol(times,quotes,saved,verify.Audit(),'BOOM600')


def test_fewer_than100_calibration_events_only_decodes_prefix(monkeypatch):
    start=int(verify.timestamp('2026-04-11T00:00:00+00:00'))
    _,quotes=path([.02]+[-.001]*598);times=list(range(start,start+600))
    _,_,saved=measured_fixture();reason='fewer_than100_calibration_tail_events'
    calibration=saved['calibration40'];calibration['detected_events']=1
    calibration['directed_sign_counts']={'positive':1,'negative':238,'zero':0}
    calibration['directed_return_quantiles']={key:(.02 if key=='p100' else -.001) for key in calibration['directed_return_quantiles']}
    young=literal_band(238,0,0.,-.238)
    calibration['bands']['lt_N']=young;calibration['daily_bands'][0]['bands']['lt_N']=young
    saved['detector'].update(adequate=False,calibration_events=1,reason=reason)
    saved['segments']={name:{'status':'NOT TESTED','reason':reason,'row_start_inclusive':lo,'row_end_exclusive':hi} for name,(lo,hi) in
                       {'wf1':(240,300),'wf2':(300,360),'wf3':(360,420),'dev_validation':(240,420),'final30':(420,600)}.items()}
    saved['comparisons']={'status':'NOT TESTED','reason':reason}
    original=verify.decode;calls=[]
    def prefix_only(ts,qs,scale,side):
        calls.append(len(ts));assert len(ts)==240
        return original(ts,qs,scale,side)
    monkeypatch.setattr(verify,'decode',prefix_only)
    audit=verify.Audit();report=verify.audit_symbol(times,quotes,saved,audit,'BOOM600')
    assert calls==[240] and not report['detector_adequate'] and not audit.failures


@pytest.mark.parametrize('bounds',[[1.2,.9],[-.1,1.2],[True,1.2]])
def test_falsification_cannot_use_invalid_interval(bounds):
    assert verify.falsification(100,100,bounds,9999,True)=='insufficient_evidence'


def test_tiny_calibration_prefix_is_explicitly_inadequate():
    start=int(verify.timestamp('2026-04-11T00:00:00+00:00'));cuts={'40':1,'50':1,'60':1,'70':2,'100':3}
    reason='fewer_than_two_calibration_quotes'
    saved={'side':1,'observed_rows':3,'row_cutoffs':cuts,'detector_fit_attempts':1,'no_detector_refit':True,
           'physical_spike_census':False,'strategy_eligible':False,'profit_factor':'NOT TESTED','actual_money_profit':'NOT TESTED',
           'detector':{'adequate':False,'median_abs_log_return':None,'threshold':None,'threshold_multiplier':10.,'reason':reason,'calibration_events':None},
           'calibration40':{'status':'NOT TESTED','reason':reason,'row_start_inclusive':0,'row_end_exclusive':1},
           'segments':{name:{'status':'NOT TESTED','reason':reason,'row_start_inclusive':lo,'row_end_exclusive':hi} for name,(lo,hi) in
                       {'wf1':(1,1),'wf2':(1,1),'wf3':(1,2),'dev_validation':(1,2),'final30':(2,3)}.items()},
           'comparisons':{'status':'NOT TESTED','reason':reason}}
    audit=verify.Audit();verify.audit_symbol([start,start+1,start+2],[100.,101.,102.],saved,audit,'BOOM600')
    assert not audit.failures
