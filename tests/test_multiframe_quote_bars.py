"""Synthetic causal midpoint bars; no actual tick data or platform calls."""
from dataclasses import asdict, FrozenInstanceError
import math

import pytest

from independent_multiframe_research.bars import Bar, build_frames, FRAME_MINUTES, MINUTE_MS as M
from independent_profit_research.data import MAX_SCALED_PRICE
from independent_profit_research.execution import Quote as Q
from independent_profit_research.tape import QuoteTape


def tape(start=0, end=6*M, step=1000, price=100):
    return QuoteTape(Q(t,price,price+1) for t in range(start,end+1,step))


def test_m1_confirmation_is_observed_next_minute_and_last_unconfirmed_not_emitted():
    rows=[Q(t,100,102) for t in range(0,60000,5000)]+[Q(56000,100,102),Q(60017,110,112)]
    frames=build_frames(QuoteTape(rows),0,120000)
    assert frames[1]==[Bar(0,60000,60017,101,101,101,101,0)]
    assert all(not frames[size] for size in FRAME_MINUTES if size!=1)


def test_confirmation_delay_is_part_of_the_5000ms_boundary_gap():
    rows=[Q(t,100,102) for t in range(0,60000,5000)]+[Q(60017,110,112)]
    assert build_frames(QuoteTape(rows),0,120000)[1]==[]  # 55000 ->60017 is5017ms.


def test_end_exclusive_is_not_an_implicit_confirmation_or_future_quote_access():
    rows=[Q(t,100,102) for t in range(0,60001,5000)]
    assert build_frames(QuoteTape(rows),0,60000)[1]==[]
    assert len(build_frames(QuoteTape(rows),0,60001)[1])==1
    assert build_frames(QuoteTape(rows+[Q(60000,200,201),Q(70000,300,301)]),0,60000)[1]==[]


def test_exact_utc_alignment_and_larger_frame_confirmation():
    frames=build_frames(tape(end=241*M),0,241*M)
    assert tuple(frames)==FRAME_MINUTES
    assert len(frames[1])==240
    assert len(frames[5])==48 and len(frames[15])==16 and len(frames[60])==4 and len(frames[240])==1
    for size,bars in frames.items():
        for bar in bars:
            assert bar.open_ms%(size*M)==0 and bar.end_ms-bar.open_ms==size*M
            assert bar.confirmed_ms==bar.end_ms
            assert (bar.open,bar.high,bar.low,bar.close)==(100.5,)*4


def test_ohlc_uses_only_quantized_midpoint_and_closed_source_bundles():
    rows=[]
    for t in range(0,61000,1000):
        mid=101 if t<10000 else 106 if t<30000 else 98 if t<55000 else 103
        rows.append(Q(t,mid-1,mid+1))
    rows.append(Q(61000,10000,10002))
    bar=build_frames(QuoteTape(rows),0,120000)[1][0]
    assert (bar.open,bar.high,bar.low,bar.close)==(101,106,98,103)
    assert bar.confirmed_ms==60000


def test_identical_same_ms_duplicates_do_not_change_bars_or_mutate_tape():
    original=tape(end=6*M)
    duplicate=QuoteTape(q for quote in original for q in (quote,quote,quote))
    before=tuple(asdict(q) for q in duplicate)
    assert build_frames(original,0,6*M)==build_frames(duplicate,0,6*M)
    assert tuple(asdict(q) for q in duplicate)==before


@pytest.mark.parametrize('ambiguous_stamp',[0,30000,59000])
def test_same_ms_different_bid_ask_invalidates_minute_even_with_same_mid(ambiguous_stamp):
    rows=[]
    for t in range(0,3*M+1,1000):
        rows.append(Q(t,100,102))
        if t==ambiguous_stamp:rows.append(Q(t,99,103))
    bars=build_frames(QuoteTape(rows),0,3*M)[1]
    assert [bar.open_ms for bar in bars]==[M]
    assert bars[0].run_id>0


def test_ambiguity_in_confirming_minute_never_rewrites_previous_completed_bar():
    prefix=[Q(t,100,102) for t in range(0,M+1,1000)]
    first=build_frames(QuoteTape(prefix),0,3*M)[1]
    suffix=[Q(M,99,103)]+[Q(t,100,102) for t in range(M+1000,3*M+1,1000)]
    completed=build_frames(QuoteTape(prefix+suffix),0,3*M)[1]
    assert first[0] in completed and first[0]==completed[0]
    assert all(bar.open_ms!=M for bar in completed)


def test_internal_gap_over5s_invalidates_its_minute_and_restarts_run():
    rows=[Q(t,100,102) for t in range(0,4*M+1,1000) if not M+10000<=t<M+16000]
    bars=build_frames(QuoteTape(rows),0,4*M)[1]
    assert [bar.open_ms for bar in bars]==[0,2*M]
    assert bars[1].run_id>bars[0].run_id


def test_cross_minute_gap_invalidates_both_endpoint_minutes():
    rows=[Q(t,100,102) for t in range(0,4*M+1,1000) if not 58000<=t<64000]
    bars=build_frames(QuoteTape(rows),0,4*M)[1]
    assert [bar.open_ms for bar in bars]==[2*M]
    assert bars[0].run_id>0


def test_long_gap_never_fills_all_crossed_minutes():
    rows=[Q(t,100,102) for t in range(0,8*M+1,1000) if not M-2000<=t<5*M+2000]
    frames=build_frames(QuoteTape(rows),0,8*M)
    assert [bar.open_ms for bar in frames[1]]==[6*M]
    assert frames[5]==[]


def test_exactly5000ms_internal_and_neighbor_gaps_are_permitted():
    frames=build_frames(tape(end=3*M,step=5000),0,3*M)
    assert [bar.open_ms for bar in frames[1]]==[0,M]
    assert {bar.run_id for bar in frames[1]}=={0}


def test_head_coverage5000ms_inclusive_and5001ms_invalid():
    accepted=[Q(5000,100,102)]+[Q(t,100,102) for t in range(10000,M+1,5000)]
    rejected=[Q(5001,100,102)]+[Q(t,100,102) for t in range(10000,M+1,5000)]
    assert len(build_frames(QuoteTape(accepted),0,2*M)[1])==1
    assert build_frames(QuoteTape(rejected),0,2*M)[1]==[]


def test_partial_head_and_unfinished_tail_are_not_manufactured():
    frames=build_frames(tape(end=4*M),1234,3*M+25000)
    assert [bar.open_ms for bar in frames[1]]==[M,2*M]
    assert all(bar.open_ms>=1234 and bar.end_ms<=3*M+25000 for bar in frames[1])


def test_larger_frame_cannot_compress_missing_minute_or_span_run_reset():
    rows=[Q(t,100,102) for t in range(0,11*M+1,1000) if not 2*M+10000<=t<2*M+17000]
    frames=build_frames(QuoteTape(rows),0,11*M)
    assert [bar.open_ms for bar in frames[5]]==[5*M]
    assert frames[5][0].run_id>0 and frames[15]==[]
    parts=[bar for bar in frames[1] if 5*M<=bar.open_ms<10*M]
    assert frames[5][0].confirmed_ms==max(bar.confirmed_ms for bar in parts)


def test_daily_file_boundary_has_no_special_reset_or_candle_alignment_effect():
    start=24*60*M-5*M
    rows=tape(start=start,end=start+11*M)
    frames=build_frames(rows,start,start+11*M)
    assert [bar.open_ms for bar in frames[5]]==[start,start+5*M]
    assert {bar.run_id for bar in frames[1]}=={0}


@pytest.mark.parametrize('cut',[M,2*M+1234,5*M,6*M-1])
def test_already_emitted_bars_are_invariant_under_future_suffix(cut):
    rows=list(tape(end=10*M))
    prefix=QuoteTape(q for q in rows if q.time_msc<=cut)
    before=build_frames(prefix,0,10*M)
    after=build_frames(QuoteTape(rows),0,10*M)
    for size in FRAME_MINUTES:
        assert all(bar in after[size] for bar in before[size])
        assert after[size][:len(before[size])]==before[size]


def test_fixed_source_window_is_invariant_to_valid_outside_suffix():
    original=list(tape(end=7*M))
    assert build_frames(QuoteTape(original),0,6*M)==build_frames(QuoteTape(original+[Q(8*M,9999,10000)]),0,6*M)


@pytest.mark.parametrize('argument,value',[('tape',[]),('start_ms',True),('start_ms',-1),
    ('end_ms',False),('end_ms',0),('end_ms',1.2),('end_ms',2**63)])
def test_invalid_build_contract_rejected(argument,value):
    values={'tape':tape(),'start_ms':0,'end_ms':6*M}
    values[argument]=value
    with pytest.raises(ValueError):build_frames(**values)


@pytest.mark.parametrize('mutation',['backward','time_bool','nan','crossed','fractional','overbound'])
def test_mutated_quote_input_revalidated_without_sorting(mutation):
    rows=tape()
    quote=rows.quotes[3]
    if mutation=='backward':object.__setattr__(quote,'time_msc',1000)
    elif mutation=='time_bool':object.__setattr__(quote,'time_msc',True)
    elif mutation=='nan':object.__setattr__(quote,'bid',math.nan)
    elif mutation=='crossed':object.__setattr__(quote,'bid',quote.ask+1)
    elif mutation=='fractional':object.__setattr__(quote,'bid',100.25)
    else:object.__setattr__(quote,'ask',MAX_SCALED_PRICE)
    with pytest.raises(ValueError):build_frames(rows,0,6*M)


def test_empty_tape_returns_all_five_empty_frame_lists():
    assert build_frames(QuoteTape([]),0,M)=={size:[] for size in FRAME_MINUTES}


def test_bar_snapshot_is_frozen_and_checks_ohlc_confirmation():
    bar=Bar(0,M,M+1,100,101,99,100.5,0)
    with pytest.raises(FrozenInstanceError):bar.close=1
    for values in [(0,M,M-1,100,101,99,100,0),(0,M,M,100,99,99,100,0),
                   (0,M,M,100,101,99,math.nan,0),(0,M,M,100,101,99,100,-1)]:
        with pytest.raises(ValueError):Bar(*values)
