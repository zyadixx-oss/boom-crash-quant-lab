#!/usr/bin/env python3
"""Independent stdlib verification of the fixed tail-tick descriptive pilot.

No helper/runner/metrics imports, fitting of strategies, collection or orders.
Calibration and historical execution require explicit results-ready authority.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("LIVE_TRADING","READY_FOR_LIVE","LIVE_ALLOWED","OPENED_TRADES")
SYMBOLS = ("BOOM600","CRASH600")
MULTIPLIER = 10.
BAND_BOUNDARY = 600
REPEATS = 9999
SEED = 20261005
DATES = ("2026-04-11","2026-04-26","2026-05-12","2026-05-28","2026-06-13","2026-06-29",
         "2026-07-15","2026-07-31","2026-08-16","2026-09-01","2026-09-17","2026-10-03")
CONFIG = {"calibration_fraction":.4,"development_fraction":.7,"threshold_multiplier":10.,
          "nominal_age_boundary_seconds":600,"minimum_calibration_events":100,"minimum_events_per_band":100,
          "bootstrap_repeats":9999,"bootstrap_seed":20261005,"large_overdue_ratio":1.5,
          "undefined_bootstrap_draws_block_falsification":True}
SCIENCE = ("docs/SPIKE_TICK_TAIL_PROTOCOL.md","backend/app/research/tick_tail.py","backend/tests/test_tick_tail.py",
           "scripts/run_spike_tick_tail.py","backend/tests/test_spike_tick_tail_study.py")
Row = namedtuple("Row", "epoch known directed event age prior_mark band")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    stamp = datetime.fromisoformat(value.replace("Z","+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return stamp.timestamp()


def iso(stamp):
    return datetime.fromtimestamp(stamp,timezone.utc).isoformat()


def date_of(stamp):
    return datetime.fromtimestamp(stamp,timezone.utc).strftime("%Y-%m-%d")


def source_path(value):
    path = (ROOT/value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Repository-relative path required")
    return path


def positive(value):
    if isinstance(value,bool):
        raise ValueError("Boolean quote/scale refused")
    result=float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Finite positive quote/scale required")
    return result


class Audit:
    def __init__(self):
        self.checks=0
        self.failures=[]
        self.inputs={}
        self.safety_values_checked=0
        self.max_numeric_error={}

    def equal(self,label,actual,expected):
        self.checks+=1
        if isinstance(actual,bool) or isinstance(expected,bool):
            ok=type(actual) is type(expected) is bool and actual is expected
        elif isinstance(actual,float) and isinstance(expected,(int,float)):
            error=abs(actual-expected)
            self.max_numeric_error[label.rsplit("/",1)[-1]]=max(self.max_numeric_error.get(label.rsplit("/",1)[-1],0),error)
            ok=math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-13)
        else:
            ok=actual==expected
        if not ok:self.failures.append({"check":label,"actual":actual,"expected":expected})

    def require(self,label,condition):
        self.equal(label,condition is True,True)
        if condition is not True:raise ValueError(label)

    def pin(self,path,expected=None):
        value=digest(path);self.inputs[str(Path(path).resolve())]=value
        if expected is not None:self.require("hash/"+str(path),value==expected)
        return value

    def safety(self,payload,label):
        if isinstance(payload,dict):
            for key,item in payload.items():
                if key in FLAGS:
                    self.safety_values_checked+=1
                    self.require(label+"/"+key,item is False)
                self.safety(item,label+"/"+key)
        elif isinstance(payload,list):
            for index,item in enumerate(payload):self.safety(item,label+f"/{index}")


def compare_points(audit,label,observed,expected):
    """Compare a reconstructed subset recursively with numeric tolerance."""
    if isinstance(expected,dict):
        audit.require(label+"/mapping",isinstance(observed,dict))
        for key,value in expected.items():
            audit.require(label+"/field/"+key,key in observed)
            compare_points(audit,label+"/"+key,observed[key],value)
    elif isinstance(expected,list):
        audit.require(label+"/list",isinstance(observed,list) and len(observed)==len(expected))
        for index,(a,b) in enumerate(zip(observed,expected,strict=True)):
            compare_points(audit,label+f"/{index}",a,b)
    else:audit.equal(label,observed,expected)


def read_quotes(path):
    times,quotes=[],[]
    with Path(path).open(newline="") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames != ["epoch","quote"]:raise ValueError("Exact tick CSV schema required")
        for row in reader:
            stamp=int(row["epoch"])
            if str(stamp)!=row["epoch"] or times and stamp<=times[-1]:raise ValueError("Sorted unique whole seconds required")
            times.append(stamp);quotes.append(positive(row["quote"]))
    if not times:raise ValueError("Nonempty quote stream required")
    return times,quotes


def log_increment(previous,current):
    # Scalar computation protects quantized tiny changes and extreme ratios.
    previous,current=positive(previous),positive(current)
    relative=(current-previous)/previous
    if math.isfinite(relative) and relative>-1:
        return math.log1p(relative)
    return math.log(current)-math.log(previous)


def cutoffs(n):
    if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError("At least one observed row required")
    return {**{str(percent):n*percent//100 for percent in (40,50,60,70)},"100":n}


def fit_scale(times,quotes,cut):
    if len(times)!=len(quotes) or not 2<=cut<=len(times):raise ValueError("Invalid fixed calibration prefix")
    values=[abs(log_increment(quotes[i-1],quotes[i])) for i in range(1,cut) if times[i]-times[i-1]==1]
    if not values:raise ValueError("Calibration lacks consecutive increments")
    scale=positive(statistics.median(values))
    if not math.isfinite(MULTIPLIER*scale):raise ValueError("Finite fixed tail threshold required")
    return scale,len(values)


def decode(times,quotes,scale,side):
    """One chronological stream; the current event only changes future age."""
    scale=positive(scale)
    if isinstance(side,bool) or not isinstance(side,int) or side not in (-1,1):raise ValueError("Direction must be integer -1 or +1")
    if len(times)!=len(quotes) or not times:raise ValueError("Nonempty matching quote arrays required")
    if any(isinstance(t,bool) or not isinstance(t,int) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError("Strictly increasing integer seconds required")
    for quote in quotes:positive(quote)
    rows=[];last_event=None;last_mark=None
    for i,stamp in enumerate(times):
        if not i or stamp-times[i-1]!=1:
            last_event=None;last_mark=None
            rows.append(Row(stamp,False,None,None,None,None,None))
            continue
        age=stamp-last_event if last_event is not None else None
        band=("lt_N" if age<BAND_BOUNDARY else "ge_N") if age is not None else None
        directed=side*log_increment(quotes[i-1],quotes[i])
        event=directed>MULTIPLIER*scale
        rows.append(Row(stamp,True,directed,event,age,last_mark,band))
        if event:last_event,last_mark=stamp,directed
    return rows


def quantile(values,q):
    if not values:return None
    if not 0<=q<=1:raise ValueError("Quantile must lie in [0,1]")
    values=sorted(values);index=(len(values)-1)*q
    low=math.floor(index);high=math.ceil(index)
    return values[low]+(values[high]-values[low])*(index-low)


def band_summary(rows,band):
    selected=[r for r in rows if r.known and r.band==band]
    events=[r.directed for r in selected if r.event]
    non_events=[r.directed for r in selected if not r.event]
    values=[r.directed for r in selected]
    n=len(values);hazard=len(events)/n if n else None
    event_mean=statistics.fmean(events) if events else None
    non_event_mean=statistics.fmean(non_events) if non_events else None
    mean=statistics.fmean(values) if values else None
    identity=(hazard*event_mean+(1-hazard)*non_event_mean
              if events and non_events else None)
    return {"exposure_seconds":n,"events":len(events),"event_return_sum":math.fsum(events),
            "non_event_count":len(non_events),"non_event_return_sum":math.fsum(non_events),"return_sum":math.fsum(values),
            "hazard":hazard,"event_mean_d":event_mean,"non_event_mean_d":non_event_mean,"unconditional_mean_d":mean,
            "mixture_identity_mean_d":identity,"mixture_identity_abs_error":abs(mean-identity) if identity is not None else None}


def segment_summary(rows,lo,hi,quantiles=(.01,.05,.25,.5,.75,.95,.99)):
    if not 0<=lo<hi<=len(rows):raise ValueError("Nonempty chronological segment required")
    part=rows[lo:hi];known=[r for r in part if r.known];values=[r.directed for r in known]
    days=sorted({date_of(r.epoch) for r in part})
    bands={band:band_summary(part,band) for band in ("lt_N","ge_N")}
    daily={}
    for day in days:
        selected=[r for r in part if date_of(r.epoch)==day]
        daily[day]={band:{key:value for key,value in band_summary(selected,band).items() if key in
                         ("exposure_seconds","events","event_return_sum","non_event_count","non_event_return_sum","return_sum")}
                    for band in ("lt_N","ge_N")}
    young,old=bands["lt_N"],bands["ge_N"]
    hr=(old["hazard"]/young["hazard"] if young["hazard"] is not None and young["hazard"]>0 and old["hazard"] is not None else None)
    return {"start_row":lo,"end_row_exclusive":hi,"rows":hi-lo,"first_epoch":part[0].epoch,"last_epoch":part[-1].epoch,
            "from_utc":iso(part[0].epoch),"to_exclusive_utc":iso(part[-1].epoch+1),"observed_days":days,"observed_clusters":len(days),
            "valid_increments":len(known),"missing_pair_exclusions":len(part)-len(known),
            "unknown_age_exclusions":sum(r.age is None for r in known),
            "known_age_exposure_seconds":sum(r.age is not None for r in known),
            "positive_increments":sum(v>0 for v in values),"negative_increments":sum(v<0 for v in values),
            "zero_increments":sum(v==0 for v in values),"detected_events":sum(bool(r.event) for r in known),
            "directed_return_quantiles":{str(q):quantile(values,q) for q in quantiles},"bands":bands,
            "daily_band_sufficient_statistics":daily,"hazard_ratio_ge_N_over_lt_N":hr}


def falsification(events_young,events_old,interval,valid,adequate):
    enough=adequate and events_young>=100 and events_old>=100
    bounded=(isinstance(interval,list) and len(interval)==2 and
             all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in interval) and interval[0]<=interval[1])
    if not adequate:return "detector_inadequate"
    if not enough or not bounded or valid!=REPEATS:return "insufficient_evidence"
    return "large_overdue_effect_rejected" if interval[1]<1.5 else "large_overdue_effect_not_rejected"


def segment_slices(cuts):
    return {"calibration40":(0,cuts["40"]),"wf1":(cuts["40"],cuts["50"]),
            "wf2":(cuts["50"],cuts["60"]),"wf3":(cuts["60"],cuts["70"]),
            "dev_validation":(cuts["40"],cuts["70"]),"final30":(cuts["70"],cuts["100"])}


def saved_band(band):
    return {"exposure_seconds":band["exposure_seconds"],"events":band["events"],"hazard":band["hazard"],
            "event_return_sum":band["event_return_sum"],"event_mean_d":band["event_mean_d"],
            "non_event_seconds":band["non_event_count"],"non_event_return_sum":band["non_event_return_sum"],
            "non_event_mean_d":band["non_event_mean_d"],"return_sum":band["return_sum"],
            "unconditional_mean_d":band["unconditional_mean_d"],
            "decomposition_identity_available":band["mixture_identity_mean_d"] is not None,
            "decomposition_reconstructed_mean_d":band["mixture_identity_mean_d"],
            "decomposition_identity_abs_error":band["mixture_identity_abs_error"]}


def saved_segment(rows,lo,hi):
    point=segment_summary(rows,lo,hi,quantiles=(0.,.01,.05,.25,.5,.75,.95,.99,1.))
    part=rows[lo:hi]
    bands={name:saved_band(value) for name,value in point["bands"].items()}
    daily=[]
    for day in point["observed_days"]:
        day_rows=[r for r in part if date_of(r.epoch)==day]
        daily.append({"date":day,"observed_rows":len(day_rows),"bands":{band:saved_band(band_summary(day_rows,band)) for band in ("lt_N","ge_N")}})
    quantile_names=("p00","p01","p05","p25","p50","p75","p95","p99","p100")
    return {"status":"MEASURED_EXPLORATORY","row_start_inclusive":lo,"row_end_exclusive":hi,
            "observed_rows":hi-lo,"first_observed_utc":iso(part[0].epoch),"last_observed_utc":iso(part[-1].epoch),
            "observed_scheduled_days":point["observed_days"],"observed_cluster_count":point["observed_clusters"],
            "valid_increments":point["valid_increments"],"initial_or_gap_pair_exclusions":point["missing_pair_exclusions"],
            "unknown_age_exclusions":point["unknown_age_exclusions"],"known_age_increments":point["known_age_exposure_seconds"],
            "detected_events":point["detected_events"],"events_with_unknown_prior_age":sum(r.known and r.event and r.age is None for r in part),
            "directed_sign_counts":{"positive":point["positive_increments"],"negative":point["negative_increments"],"zero":point["zero_increments"]},
            "directed_return_quantiles":dict(zip(quantile_names,point["directed_return_quantiles"].values(),strict=True)),
            "bands":bands,"daily_bands":daily}


def metadata(study,audit):
    declaration,result=(read_json(study/name) for name in ("declaration.json","results.json"))
    declaration_sha=audit.pin(study/"declaration.json",(study/"declaration.sha256").read_text().strip())
    audit.pin(study/"results.json",(study/"results.sha256").read_text().strip())
    audit.require("results/declaration_link",result["declaration_sha256"]==declaration_sha)
    for name,payload in (("declaration",declaration),("result",result)):
        audit.safety(payload,name)
        compare_points(audit,name+"/safety",payload["safety"],dict.fromkeys(FLAGS,False))
        audit.require(name+"/exact_config_fields",set(payload["config"])==set(CONFIG))
        compare_points(audit,name+"/config",payload["config"],CONFIG)
        audit.require(name+"/dates",payload["dates"]==list(DATES))
        audit.require(name+"/study_kind",payload["study_kind"]=="adaptive_known_history_tail_tick_mechanism_pilot_not_strategy_validation")
    audit.require("declaration/stage",declaration["stage"]=="tail_tick_precalibration_declaration")
    audit.require("results/stage",result["stage"]=="exploratory_tail_tick_mechanism_results")
    audit.require("declaration/symbols",declaration["symbols"]==list(SYMBOLS))
    audit.require("results/symbols",set(result["symbols"])==set(SYMBOLS))
    for field in ("quotes_parsed","detector_calibrated","events_or_hazards_evaluated"):
        audit.require("declaration/"+field,declaration[field] is False)
    audit.require("declaration/partition_basis",declaration["partition_basis"]=="time_ordered_observed_quote_rows_not_intervening_calendar_span")
    for field in ("goal_achieved","historical_strategy_candidate","physical_spike_census"):
        audit.require("results/"+field,result[field] is False)
    for field in ("actual_money_profit","prospective_paper"):
        audit.require("results/"+field,result[field]=="NOT TESTED")
    audit.require("chronology/declaration_before_measurement",timestamp(declaration["run_utc"])<=timestamp(result["run_utc"])<=timestamp(result["completed_utc"]))
    audit.require("science/exact_files",set(declaration["science_code_hashes"])==set(SCIENCE))
    for name,expected in declaration["science_code_hashes"].items():audit.pin(source_path(name),expected)
    lineage=declaration["lineage"]
    for name,expected in lineage["inherited_files"].items():audit.pin(source_path(name),expected)
    old=ROOT/"docs/spike_tick_execution_20261005"
    old_declaration,old_sources,old_audit,old_result=(read_json(old/name) for name in
                                                   ("declaration.json","tick_sources.json","independent_audit.json","results.json"))
    for name,field in (("declaration.json","round7_declaration_sha256"),("tick_sources.json","round7_tick_sources_sha256"),
                       ("results.json","round7_results_sha256"),("independent_audit.json","round7_audit_sha256")):
        audit.pin(old/name,lineage[field])
    audit.require("lineage/round6_selection",old_declaration["round6_selection_sha256"]==lineage["round6_selection_sha256"])
    audit.pin(ROOT/"docs/spike_nonlinear_20261005/selection.json",lineage["round6_selection_sha256"])
    audit.require("lineage/round7_source_link",old_sources["declaration_sha256"]==digest(old/"declaration.json") and
                  old_result["tick_sources_sha256"]==digest(old/"tick_sources.json"))
    audit.require("lineage/source_snapshot",lineage["sources"]==old_sources["sources"])
    audit.require("lineage/passed_prior_audit",old_audit["passed"] is True and not old_audit["errors"] and
                  old_audit["stage"]=="independent_primary_tick_execution_audit" and len(old_audit["sources"])==24 and len(old_audit["groups"])==16)
    audit.pin(source_path(old_audit["verifier_file"]),old_audit["verifier_sha256"])
    audit.require("chronology/round7_first",timestamp(old_sources["frozen_utc"])<=timestamp(old_result["run_utc"])<=timestamp(old_audit["run_utc"])<=timestamp(declaration["run_utc"]))
    aliases=[]
    for symbol in SYMBOLS:
        values=lineage["sources"][symbol]
        audit.require(symbol+"/twelve_sources",set(values)==set(DATES))
        total=0
        for date in DATES:
            value=values[date];audit.safety(value,f"source/{symbol}/{date}")
            audit.require(f"source/{symbol}/{date}/identity",value["symbol"]==symbol and value["date"]==date)
            for field in ("rows","missing_seconds"):
                audit.require(f"source/{symbol}/{date}/integer_{field}",type(value[field]) is int and value[field]>=0)
            audit.require(f"source/{symbol}/{date}/cadence",value["rows"]+value["missing_seconds"]==86400 and value["cadence_seconds"]==1 and
                          value["gap_free"] is (value["missing_seconds"]==0))
            for filekey,shakey in (("manifest","manifest_sha256"),("clean_file","clean_sha256"),("raw_pages_file","raw_pages_sha256"),("page_audit_file","page_audit_sha256")):
                path=source_path(value[filekey]);audit.pin(path,value[shakey]);aliases.append(value[filekey])
                matches=[sha for name,sha in old_audit["input_sha256"].items() if name==value[filekey] or name.endswith("/"+value[filekey])]
                audit.require(f"source/{symbol}/{date}/prior_pin/{filekey}",matches==[value[shakey]])
            manifest=read_json(source_path(value["manifest"]))
            audit.require(f"source/{symbol}/{date}/manifest_snapshot",all(value[key]==item for key,item in manifest.items()))
            total+=value["rows"]
        audit.equal(symbol+"/declared_rows",declaration["observed_rows"][symbol],total)
        compare_points(audit,symbol+"/declared_cuts",declaration["row_cutoffs"][symbol],cutoffs(total))
    audit.require("source/no_aliases",len(aliases)==len(set(aliases))==96)
    return declaration,result


def load_symbol(declaration,symbol,audit):
    times,quotes=[],[];reports=[]
    for date in DATES:
        value=declaration["lineage"]["sources"][symbol][date]
        day_times,day_quotes=read_quotes(source_path(value["clean_file"]))
        start=int(timestamp(date+"T00:00:00+00:00"));end=start+86400
        label=symbol+"/source/"+date
        audit.require(label+"/bounds",start<=day_times[0]<=day_times[-1]<end)
        audit.equal(label+"/rows",len(day_times),value["rows"])
        audit.equal(label+"/first",day_times[0],value["first_epoch"])
        audit.equal(label+"/last",day_times[-1],value["last_epoch"])
        actual_gaps=[];cursor=start
        for stamp in day_times:
            if stamp>cursor:actual_gaps.append({"start_epoch":cursor,"end_exclusive_epoch":stamp,"missing_seconds":stamp-cursor})
            cursor=stamp+1
        if cursor<end:actual_gaps.append({"start_epoch":cursor,"end_exclusive_epoch":end,"missing_seconds":end-cursor})
        compare_points(audit,label+"/gaps",value["gaps"],actual_gaps)
        audit.equal(label+"/missing_seconds",value["missing_seconds"],sum(g["missing_seconds"] for g in actual_gaps))
        audit.require(label+"/chronological_order",not times or times[-1]<day_times[0])
        times.extend(day_times);quotes.extend(day_quotes)
        reports.append({"symbol":symbol,"date":date,"rows":len(day_times),"missing_seconds":value["missing_seconds"],"clean_sha256":value["clean_sha256"]})
    return times,quotes,reports


def comparison_policy(saved,segment,audit,label):
    young,old=segment["bands"]["lt_N"],segment["bands"]["ge_N"]
    ratio=(old["hazard"]/young["hazard"] if old["hazard"] is not None and young["hazard"] is not None and young["hazard"]>0 else None)
    audit.equal(label+"/point_ratio",saved["older_over_younger_hazard_ratio"],ratio)
    compare_points(audit,label+"/policy",saved,{"status":"DESCRIPTIVE_EXPLORATORY","bootstrap_repeats":REPEATS,"bootstrap_seed":SEED,
                                              "paired_scheduled_day_resampling":True,"same_day_draws_for_both_bands":True,
                                              "observed_cluster_count":segment["observed_cluster_count"],
                                              "observed_scheduled_days":segment["observed_scheduled_days"],
                                              "interval_omits_undefined_draws":True,"undefined_draws_block_falsification":True,
                                              "inference_kind":"conditional_on_fixed_detector_descriptive_iid_observed_scheduled_days",
                                              "training_median_refits_in_bootstrap":False,"no_contiguous_weekly_claim":True,
                                              "no_independent_tick_inference_claim":True,"no_strategy_or_profitability_claim":True})
    zero_days={band:sum(day["bands"][band]["exposure_seconds"]==0 for day in segment["daily_bands"]) for band in ("lt_N","ge_N")}
    compare_points(audit,label+"/zero_days",saved["zero_band_exposure_days"],zero_days)
    valid,unknown=saved["valid_ratio_replicates"],saved["undefined_ratio_replicates"]
    audit.require(label+"/replicate_counts",type(valid) is int and type(unknown) is int and 0<=valid<=REPEATS and valid+unknown==REPEATS)
    interval=saved["hazard_ratio_ci95"]
    audit.require(label+"/interval_shape",isinstance(interval,list) and len(interval)==2)
    if not valid:audit.require(label+"/null_interval",interval==[None,None])
    else:audit.require(label+"/finite_interval",all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in interval) and interval[0]<=interval[1])
    if not young["events"] or not young["exposure_seconds"] or not old["exposure_seconds"]:
        audit.equal(label+"/necessarily_undefined",valid,0)
    if all(day["bands"]["lt_N"]["events"]>0 and day["bands"]["ge_N"]["exposure_seconds"]>0 for day in segment["daily_bands"]):
        audit.equal(label+"/necessarily_all_defined",valid,REPEATS)
    reasons=[]
    counts_ok=young["events"]>=100 and old["events"]>=100
    all_defined=valid==REPEATS and unknown==0
    if not counts_ok:reasons.append("fewer_than100_events_in_at_least_one_band")
    if not all_defined:reasons.append("not_all9999_bootstrap_ratios_defined")
    if (ratio is None or not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (ratio,*interval)) or
            interval[0]>interval[1]):reasons.append("undefined_or_invalid_ratio_or_interval")
    insufficient=bool(reasons)
    rejected=not insufficient and interval[1]<1.5
    if not insufficient and not rejected:reasons.append("upper_bound_not_below1.5")
    gate={"status":"INSUFFICIENT_EVIDENCE" if insufficient else ("REJECT_LARGE_OVERDUE_EFFECT" if rejected else "LARGE_OVERDUE_EFFECT_NOT_REJECTED"),
          "large_overdue_effect_falsified":rejected,"reasons":reasons,"minimum_events_per_band":100,
          "hazard_ratio_upper_threshold":1.5,"all9999_ratios_defined":all_defined,"does_not_establish_independence":True,
          "does_not_test_profitability":True,"strategy_eligible":False}
    compare_points(audit,label+"/falsification",saved["falsification"],gate)


def audit_symbol(times,quotes,saved,audit,label):
    side=1 if label=="BOOM600" else -1
    cuts=cutoffs(len(times));slices=segment_slices(cuts)
    compare_points(audit,label+"/immutable_policy",saved,{"side":side,"observed_rows":len(times),"row_cutoffs":cuts,
                                                       "detector_fit_attempts":1,"no_detector_refit":True,"physical_spike_census":False,
                                                       "strategy_eligible":False,"profit_factor":"NOT TESTED","actual_money_profit":"NOT TESTED"})
    detector=saved["detector"]
    try:scale,calibration_pairs=fit_scale(times,quotes,cuts["40"])
    except ValueError:
        reason=("fewer_than_two_calibration_quotes" if cuts["40"]<2 else
                "zero_or_invalid_calibration_scale_or_no_consecutive_increment")
        compare_points(audit,label+"/inadequate_scale",detector,{"adequate":False,"median_abs_log_return":None,"threshold":None,
                                                               "threshold_multiplier":10.,"reason":reason,"calibration_events":None})
        compare_points(audit,label+"/calibration_not_tested",saved["calibration40"],{"status":"NOT TESTED","reason":reason,
                                                                                 "row_start_inclusive":0,"row_end_exclusive":cuts["40"]})
        calibration=None;adequate=False;events=None
    else:
        prefix=decode(times[:cuts["40"]],quotes[:cuts["40"]],scale,side)
        calibration=saved_segment(prefix,0,len(prefix))
        compare_points(audit,label+"/calibration",saved["calibration40"],calibration)
        audit.equal(label+"/calibration_valid_pairs",calibration["valid_increments"],calibration_pairs)
        events=calibration["detected_events"];adequate=events>=100
        compare_points(audit,label+"/fixed_detector",detector,{"adequate":adequate,"median_abs_log_return":scale,"threshold":10*scale,
                                                             "threshold_multiplier":10.,"calibration_events":events,
                                                             "calibration_source":"earliest40_percent_observed_rows_only"})
        reason="fewer_than100_calibration_tail_events"
        if not adequate:audit.equal(label+"/inadequate_event_reason",detector["reason"],reason)
    audit.require(label+"/exact_segments",set(saved["segments"])==set(slices)-{"calibration40"})
    measured={}
    if not adequate:
        for name,(lo,hi) in slices.items():
            if name=="calibration40":continue
            expected={"status":"NOT TESTED","reason":reason,"row_start_inclusive":lo,"row_end_exclusive":hi}
            audit.require(label+"/"+name+"/no_later_measurement",set(saved["segments"][name])==set(expected))
            compare_points(audit,label+"/"+name,saved["segments"][name],expected)
        audit.require(label+"/no_later_comparisons",set(saved["comparisons"])=={"status","reason"})
        compare_points(audit,label+"/comparison_not_tested",saved["comparisons"],{"status":"NOT TESTED","reason":reason})
    else:
        rows=decode(times,quotes,scale,side)
        for name,(lo,hi) in slices.items():
            if name=="calibration40":continue
            expected=saved_segment(rows,lo,hi);measured[name]=expected
            compare_points(audit,label+"/"+name,saved["segments"][name],expected)
        audit.require(label+"/exact_comparisons",set(saved["comparisons"])=={"dev_validation","final30"})
        for name in ("dev_validation","final30"):
            comparison_policy(saved["comparisons"][name],measured[name],audit,label+"/comparison/"+name)
    return {"symbol":label,"observed_rows":len(times),"median_abs_log_return":scale if calibration is not None else None,
            "calibration_events":events,"detector_adequate":adequate,"measured_later_segments":len(measured),
            "calibration":calibration,"segments":measured}


def verify(study,audit):
    declaration,result=metadata(study,audit)
    symbols,sources=[],[]
    for symbol in SYMBOLS:
        times,quotes,source_reports=load_symbol(declaration,symbol,audit);sources.extend(source_reports)
        audit.equal(symbol+"/loaded_rows",len(times),declaration["observed_rows"][symbol])
        before=len(audit.failures)
        report=audit_symbol(times,quotes,result["symbols"][symbol],audit,symbol)
        report["mismatches"]=len(audit.failures)-before;symbols.append(report)
    table_path=study/"metrics.csv";audit.pin(table_path)
    expected_table=[]
    for symbol in symbols:
        if not symbol["detector_adequate"]:continue
        for name,segment in {"calibration40":symbol["calibration"],**symbol["segments"]}.items():
            for band,point in segment["bands"].items():
                expected_table.append({"symbol":symbol["symbol"],"segment":name,"band":band,**point})
    with table_path.open(newline="") as stream:table=list(csv.DictReader(stream))
    audit.require("table/row_count",len(table)==len(expected_table))
    for i,(observed,expected) in enumerate(zip(table,expected_table,strict=True)):
        audit.require(f"table/{i}/schema",set(observed)==set(expected))
        parsed={}
        for key,value in expected.items():
            text=observed[key]
            if key in ("symbol","segment","band"):parsed[key]=text
            elif isinstance(value,bool):
                audit.require(f"table/{i}/{key}/boolean",text in ("True","False"));parsed[key]=text=="True"
            elif value is None:
                audit.require(f"table/{i}/{key}/null",text=="");parsed[key]=None
            elif isinstance(value,int):
                parsed[key]=int(text);audit.require(f"table/{i}/{key}/integer",str(parsed[key])==text)
            else:
                parsed[key]=float(text);audit.require(f"table/{i}/{key}/finite",math.isfinite(parsed[key]))
        compare_points(audit,f"table/{i}",parsed,expected)
    for path,expected in tuple(audit.inputs.items()):audit.require("unchanged_during_audit/"+path,digest(path)==expected)
    return symbols,sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/spike_tick_tail_20261005")
    args=parser.parse_args();audit=Audit();symbols=[];sources=[]
    for flag in FLAGS:audit.require("environment/"+flag,os.environ.get(flag,"false").lower()=="false")
    target=args.output/"independent_audit.json"
    if target.exists():raise ValueError("Refusing audit overwrite; preserve previous attempt explicitly")
    try:symbols,sources=verify(args.output.resolve(),audit)
    except Exception as exc:audit.failures.append({"check":"fatal_exception","type":type(exc).__name__,"message":str(exc)})
    result={"stage":"independent_fixed_tail_tick_pilot_audit","run_utc":datetime.now(timezone.utc).isoformat(),
            "passed":not audit.failures,"checks":audit.checks,"errors":audit.failures,"safety":dict.fromkeys(FLAGS,False),
            "saved_false_values_checked":audit.safety_values_checked,"symbols":symbols,"sources":sources,
            "input_sha256":audit.inputs,"max_numeric_error":audit.max_numeric_error,
            "verifier_file":"scripts/verify_spike_tick_tail.py","verifier_sha256":digest(Path(__file__)),
            "scope":{"implementation":"Independent stdlib prefix median, scalar tail/age/mark decoder and segment/day arithmetic",
                     "included":["all24 reused source byte hashes and independent clean grid/gap counts",
                                 "round7 passed raw provenance audit linkage and frozen declaration/code/result/source lineage",
                                 "fixedfirst40-percent median and strict10x event counts; inadequate detector blocks later measurements",
                                 "causal prior age/gap resets/two600s bands, all chronological segments/daily exposures/signs/quantiles",
                                 "event/non-event drift and unconditional mixture identities; pooled/final point hazard ratios",
                                 "saved bootstrap labels/count/null/power/falsification/nonpromotion policy arithmetic"],
                     "excluded":["bootstrap draw regeneration, percentile inference validity and training-median uncertainty",
                                 "rechecking raw quote normalization already anchored by the passed round7 audit",
                                 "runtime prior-mark feature-table comparison; no such table is saved",
                                 "model fitting, strategy performance, generator independence, broker execution or real profit"]}}
    with target.open("x") as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
    print(json.dumps({"passed":result["passed"],"checks":audit.checks,"sources":len(sources),"symbols":len(symbols),
                      "errors":len(audit.failures),"audit":str(target)}),flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()
