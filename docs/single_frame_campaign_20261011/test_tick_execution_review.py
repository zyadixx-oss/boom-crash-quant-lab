"""Synthetic unit fixtures only; no fabricated historical performance."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research_campaign.engine import CostModel, SplitPlan
from research_campaign.state import atomic_json
from research_campaign.strategies import candidates_for_round, prepare_context

path=Path(__file__).with_name("tick_execution_review.py")
spec=importlib.util.spec_from_file_location("tick_execution_review",path)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def request():
    return {"ticks_history":"B500","style":"ticks","count":4,"start":100,"end":103,"req_id":1}


def response():
    r=request()
    return {"req_id":1,"echo_req":r,"history":{"times":[100,101,102,103],"prices":[100,100,100,100]}}


@pytest.mark.parametrize("change",[{"authorize":"unit"},{"buy":"unit"},{"proposal":1},{"subscribe":1},
    {"style":"candles"},{"count":5},{"count":True},{"end":104},{"ticks_history":"BTC"}])
def test_unapproved_request_cannot_be_sent(change):
    with pytest.raises(ValueError):t.validate_request({**request(),**change},request())


def test_exact_page_and_partial_remain_distinct():
    t.validate_request(request(),request())
    exact=t.parse_page(response(),request())
    assert exact["status"]=="EXACT_1S_WINDOW" and len(exact["ticks"])==4
    assert exact["historical_bid_ask_present"] is False
    raw=response();raw["history"]["times"].pop();raw["history"]["prices"].pop()
    partial=t.parse_page(raw,request())
    assert partial["status"]=="PARTIAL_WINDOW" and partial["missing_seconds"]==1


@pytest.mark.parametrize("times,prices",[
    ([100,101,101,103],[1,1,1,1]),([99,100,101,102],[1,1,1,1]),
    ([100,102,101,103],[1,1,1,1]),([100.,101,102,103],[1,1,1,1]),
    ([100,101,102,103],[1,1,float("nan"),1]),([100,101,102,103],[1,1,-1,1]),
    ([100,101,102,103],[1,1,True,1]),([100,101,102,103],[1,1,1])])
def test_invalid_actual_wire_is_not_repaired(times,prices):
    raw=response();raw["history"]={"times":times,"prices":prices}
    with pytest.raises(ValueError):t.parse_page(raw,request())


def test_api_error_preserves_unknown_window():
    raw={"req_id":1,"echo_req":request(),"error":{"code":"RateLimit","message":"unit"}}
    result=t.parse_page(raw,request())
    assert result["status"]=="API_ERROR" and result["ticks"]==[] and result["missing_seconds"]==4


def test_requests_are_fixed_merged_paginated_nonoverlapping():
    rows=[{"symbol":"BOOM500","signals":[{"entry_epoch":100,"planned_end_epoch":340},
        {"entry_epoch":300,"planned_end_epoch":1540}]}]
    plan=t.request_plan(rows,{"BOOM500":"B500","CRASH500":"C500"})
    assert [(p["request"]["start"],p["request"]["end"]) for p in plan]==[(100,1099),(1100,1539)]
    assert sum(p["request"]["count"] for p in plan)==1440
    assert t.merge_windows([(2,4),(4,8),(1,3),(20,21)])==[[1,8],[20,21]]


def context(n=300):
    bars=pd.DataFrame({"epoch":np.arange(n)*60,"open":100.,"high":100.1,"low":99.9,"close":100.})
    ctx=prepare_context(bars,"M1");ctx.arrays["atr"]=np.ones(n)
    return ctx


def test_first50_selection_is_chronological_before_trade_outcomes():
    ctx=context();signals=np.ones(300,dtype=np.int8);s=candidates_for_round(0)[7]
    first=t.select_indices(ctx,signals,s,120*60,300*60)
    assert first==list(range(119,169))
    # Price changes in later holding bars do not select winning/losing paths.
    ctx.arrays["high"]=np.full(300,999.);ctx.arrays["low"]=np.full(300,1.)
    assert t.select_indices(ctx,signals,s,120*60,300*60)==first


def test_selection_purges_full_horizon_and_missing_next_bar():
    ctx=context();signals=np.zeros(300,dtype=np.int8);signals[295:]=1;s=candidates_for_round(0)[7]
    assert t.select_indices(ctx,signals,s,120*60,300*60)==[295]
    ctx.bars.loc[296,"epoch"]+=1
    assert t.select_indices(ctx,signals,s,120*60,300*60)==[]


def test_diagnostic_runs_same_engine_preserves_unknown_and_never_qualifies(tmp_path):
    ctx=context();s=candidates_for_round(0)[7];signals=np.zeros(300,dtype=np.int8);signals[[130,140,145]]=1
    split=SplitPlan(0,120*60,300*60,400*60)
    indices=t.select_indices(ctx,signals,s,split.train_end,split.validation_end)
    selected={"symbol":"BOOM500","spec":s.to_dict(),"signals":[
        {"signal_index":i,"entry_epoch":int(ctx.bars.epoch.iloc[i]+60),"planned_end_epoch":int(ctx.bars.epoch.iloc[i]+300)} for i in indices],
        "full_prefix_signal_sha256":t.sha256(signals.tobytes()).hexdigest()}
    declaration={"datasets":{"BOOM500":{"split":split.to_dict(),"requested_sha256":"synthetic unit only",
        "source_audit":{"internal_missing_minutes":0}}},"costs":CostModel().to_dict()}
    selection={"policies":[selected]}
    for name in ("protocol.json","selection.json","collection_result.json"):atomic_json(tmp_path/name,{"synthetic_unit_test_only":True})
    quotes=pd.DataFrame({"epoch":np.arange(7860,8100),"quote":100.})
    with patch.object(t,"_verify",return_value=(declaration,selection)),patch.object(t,"_contexts",return_value={"BOOM500":ctx}),\
         patch.object(t,"_observed_ticks",return_value=({"BOOM500":quotes,"CRASH500":pd.DataFrame(columns=["epoch","quote"])},[])),\
         patch.object(t,"generate_signals",return_value=signals):
        result=t.evaluate(tmp_path)
    p=result["policies"][0]
    assert p["candle_metrics"]["completed_trades"]==3
    assert p["tick_metrics"]["completed_trades"]==1
    assert p["tick_unknown_paths"]==2 and p["paired_completed"]==1
    assert p["paired_mean_delta_net_R"]==0
    assert result["qualified_policies"]==0 and result["costs_verified"] is False
    assert result["final_holdout_accessed"] is False and all(v is False for v in result["safety"].values())
    assert (tmp_path/"REPORT.ar.md").exists()
