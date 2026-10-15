"""Artificial endpoint references only; no source history or cash calculations."""
from dataclasses import asdict

import pytest

from independent_profit_research.benchmarks import EMBARGO_MS as E, endpoint_reference
from independent_profit_research.data import MAX_SCALED_PRICE, PRICE_SCALE, SAFETY_FIELDS
from independent_profit_research.execution import ExecutionConfig as C, Quote as Q
from independent_profit_research.tape import QuoteTape

START, END = 2_000_000, 2_000_000+4*E


def basic(config=C(), dense=False):
    entry=START+E+config.entry_latency_ms
    exit_=END-E+config.exit_latency_ms
    rows=[Q(START-1,200,201),Q(START,90,91),
          Q(entry-1,1000,1001),Q(entry,100,102),Q(entry,99,105)]
    if dense:
        rows += [Q(t,103,104) for t in range(entry+5000,exit_,5000)]
    rows += [Q(exit_,110,112),Q(exit_,108,115),Q(END-1,108,109),Q(END,200,300)]
    return QuoteTape(rows)


def test_buy_and_native_sell_use_adverse_same_ms_sides_and_slippage():
    tape=basic(C(adverse_slippage_per_leg=.5))
    before=tuple(asdict(q) for q in tape)
    out=endpoint_reference(tape,START,END,-1,C(adverse_slippage_per_leg=.5))
    buy,native=out['references']['BUY'],out['references']['NATIVE']
    assert buy['entry_quote_price_scaled']==105 and buy['exit_quote_price_scaled']==108
    assert buy['entry_effective_price_scaled']==105.5 and buy['exit_effective_price_scaled']==107.5
    assert buy['endpoint_price_move_scaled']==2 and buy['endpoint_price_move_quote_units']==2/PRICE_SCALE
    assert native['entry_quote_price_scaled']==99 and native['exit_quote_price_scaled']==115
    assert native['endpoint_price_move_scaled']==-17
    assert not native['same_exposure_as_BUY']
    assert buy['entry_bundle_rows_preserved']==buy['exit_bundle_rows_preserved']==2
    assert buy['held_duration_ms']==2*E
    assert tuple(asdict(q) for q in tape)==before


def test_native_buy_is_same_exposure_without_double_sample_claim():
    out=endpoint_reference(basic(),START,END,1,C())
    native=dict(out['references']['NATIVE'])
    assert native.pop('same_exposure_as_BUY')
    assert native==out['references']['BUY']
    assert not out['direction_duplicates_are_independent_samples']


def test_source_gap_preserves_endpoint_description_but_unknown_path_and_null_dd():
    out=endpoint_reference(basic(),START,END,-1,C())
    for reference in out['references'].values():
        assert reference['endpoint_status']=='AVAILABLE'
        assert reference['scope']==reference['status']=='ENDPOINT_ONLY_NOT_PATH_OR_CASH'
        assert reference['endpoint_price_move_scaled'] is not None
        assert reference['path_quality']=='UNKNOWN' and reference['source_gap_count']==1
        assert reference['dd'] is None
        assert 'ANOMALOUS_GAP_UNRESOLVED' in reference['path_unknown_reasons']
        assert reference['source_gap_bounds_msc'][0]['delta_ms']==2*E


def test_dense_observed_path_still_does_not_prove_history_or_cash():
    out=endpoint_reference(basic(dense=True),START,END,-1,C())
    for reference in out['references'].values():
        assert reference['source_gap_count']==0
        assert reference['path_quality']=='OBSERVED_QUOTES_ONLY_COVERAGE_NOT_PROVEN'
        assert reference['dd'] is None and reference['cash_pnl'] is None
        assert reference['net_profit'] is None and reference['fees'] is None and reference['financing'] is None
        assert not any(reference[key] for key in ('cost_verified','execution_verified',
                        'profit_qualified','profit_tested','full_tick_coverage_verified'))
        assert reference['stop_loss'] is reference['take_profit'] is None
        assert reference['safety']==dict.fromkeys(SAFETY_FIELDS,False)
        assert 'R' not in reference and 'capital' not in reference
    assert out['leading_unobserved_ms']==out['trailing_unobserved_ms']==0
    assert not out['used_for_acceptance_gate']


def test_missing_exit_does_not_reuse_last_quote_as_exit():
    entry=START+E+250
    tape=QuoteTape([Q(entry,100,102),Q(entry+5000,110,112)])
    out=endpoint_reference(tape,START,END,-1,C())
    for reference in out['references'].values():
        assert reference['missing_endpoints']==['EXIT']
        assert reference['endpoint_status']=='MISSING_ENDPOINTS'
        assert reference['exit_time_msc'] is None and reference['endpoint_price_move_scaled'] is None
        assert reference['held_duration_ms'] is None and reference['path_quality']=='UNKNOWN'
    assert out['leading_unobserved_ms']==E+250
    assert out['trailing_unobserved_ms']==END-1-(entry+5000)


def test_empty_stage_keeps_both_endpoints_null():
    out=endpoint_reference(QuoteTape([]),START,END,1,C())
    assert out['observed_stage_rows']==0
    assert out['leading_unobserved_ms']==out['trailing_unobserved_ms']==END-START
    for reference in out['references'].values():
        assert reference['missing_endpoints']==['ENTRY','EXIT']
        assert reference['endpoint_price_move_quote_units'] is None
        assert reference['source_gap_count'] is None


def test_short_stage_does_not_make_reversed_endpoints_a_price_trade():
    end=START+E
    out=endpoint_reference(QuoteTape([Q(START+250,100,101),Q(end-1,110,111)]),START,end,-1,C())
    for reference in out['references'].values():
        assert reference['endpoint_price_move_scaled'] is None
        assert reference['path_quality']=='UNKNOWN'


def test_same_bundle_for_both_endpoints_is_not_a_hold():
    end=START+2*E
    target=START+E+250
    out=endpoint_reference(QuoteTape([Q(target,100,101)]),START,end,-1,C())
    for reference in out['references'].values():
        assert reference['endpoint_status']=='ENDPOINTS_NOT_ORDERED'
        assert reference['endpoint_price_move_scaled'] is None
        assert reference['held_duration_ms'] is None


def test_stage_end_exclusive_and_outside_quotes_never_fill_endpoint():
    entry=START+E+250
    tape=QuoteTape([Q(START-1,1,2),Q(entry,100,102),Q(END,110,112)])
    out=endpoint_reference(tape,START,END,-1,C())
    assert out['observed_stage_rows']==1
    assert out['references']['BUY']['missing_endpoints']==['EXIT']
    assert out['references']['BUY']['endpoint_price_move_scaled'] is None


def test_entry_and_exit_latency_each_select_first_eligible_bundle():
    config=C(entry_latency_ms=2000,exit_latency_ms=1500,adverse_slippage_per_leg=.125)
    out=endpoint_reference(basic(config),START,END,-1,config)
    buy=out['references']['BUY']
    assert buy['entry_time_msc']==START+E+2000
    assert buy['exit_time_msc']==END-E+1500
    assert buy['entry_delay_from_target_ms']==buy['exit_delay_from_target_ms']==0
    assert out['execution_assumptions']==asdict(config)


@pytest.mark.parametrize('gap,unknown', [(5000,False),(5001,True)])
def test_fixed_source_gap_boundary_and_duplicate_preservation(gap,unknown):
    entry=START+E+250
    end=entry+gap+E-250
    tape=QuoteTape([Q(entry,100,101),Q(entry,100,101),Q(entry+gap,105,106)])
    out=endpoint_reference(tape,START,end,-1,C())
    reference=out['references']['BUY']
    assert reference['source_gap_count']==int(unknown)
    assert (reference['path_quality']=='UNKNOWN') is unknown
    assert reference['entry_bundle_rows_preserved']==2


def test_late_first_observation_is_unknown_without_hiding_endpoint_difference():
    entry=START+E+250+5001
    exit_=END-E+250
    out=endpoint_reference(QuoteTape([Q(entry,100,102),Q(exit_,110,112)]),START,END,-1,C())
    buy=out['references']['BUY']
    assert buy['endpoint_price_move_scaled']==8
    assert buy['entry_delay_from_target_ms']==5001
    assert buy['path_quality']=='UNKNOWN'
    assert 'ENTRY_TARGET_UNOBSERVED_OVER_5000MS' in buy['path_unknown_reasons']


def test_large_intermediate_moves_are_not_stop_or_target_exits():
    entry,exit_=START+E+250,END-E+250
    tape=QuoteTape([Q(entry,100,102),Q(entry+1000,1,2),Q(entry+2000,5000,5001),Q(exit_,110,112)])
    buy=endpoint_reference(tape,START,END,-1,C())['references']['BUY']
    assert buy['exit_time_msc']==exit_ and buy['endpoint_price_move_scaled']==8
    assert buy['stop_loss'] is buy['take_profit'] is None


@pytest.mark.parametrize('argument,value', [('tape',[]),('start_ms',True),('end_ms',float(END)),
    ('start_ms',-1),('end_ms',START),('native_direction',True),('native_direction',0),
    ('native_direction',2),('execution_config',{}),('execution_config',C(max_gap_ms=5001))])
def test_invalid_reference_contract_is_rejected(argument,value):
    args={'tape':basic(),'start_ms':START,'end_ms':END,'native_direction':-1,'execution_config':C()}
    args[argument]=value
    with pytest.raises(ValueError):endpoint_reference(**args)


@pytest.mark.parametrize('price', [100.5,MAX_SCALED_PRICE,MAX_SCALED_PRICE+1])
def test_unscaled_or_out_of_headroom_price_rejected(price):
    tape=QuoteTape([Q(START+E+250,price,price),Q(END-E+250,price,price)])
    with pytest.raises(ValueError,match='integer-scaled'):
        endpoint_reference(tape,START,END,-1,C())


def test_mutated_config_is_revalidated():
    config=C()
    object.__setattr__(config,'entry_latency_ms',True)
    with pytest.raises(ValueError):endpoint_reference(basic(),START,END,-1,config)


@pytest.mark.parametrize('edge', ['first','last'])
def test_mutated_indexed_quote_cannot_escape_half_open_stage(edge):
    tape=basic()
    quote=tape.quotes[1] if edge=='first' else tape.quotes[-2]
    object.__setattr__(quote,'time_msc',START-1 if edge=='first' else END+1)
    with pytest.raises(ValueError,match='stage bounds'):
        endpoint_reference(tape,START,END,-1,C())


def test_reference_is_deterministic_and_json_safe():
    import json
    tape=basic(dense=True)
    first=endpoint_reference(tape,START,END,-1,C())
    second=endpoint_reference(tape,START,END,-1,C())
    assert first==second
    assert json.loads(json.dumps(first,allow_nan=False))==first
