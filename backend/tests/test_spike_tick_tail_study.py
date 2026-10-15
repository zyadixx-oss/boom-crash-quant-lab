"""Synthetic-only cutoff, decomposition, uncertainty and source-boundary checks."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.tick_tail import FixedTailDetector, describe_tick_tail
from scripts import run_spike_tick_tail as study


def ticks(n=3000, event_every=8, side=1):
    increments = np.full(n-1,-.001)
    if event_every:
        increments[np.arange(1,n) % event_every == 0] = .03
    quotes = 100*np.exp(np.r_[0,np.cumsum(side*increments)])
    return pd.DataFrame({"quote":quotes},index=pd.date_range(study.DATES[0],periods=n,freq="s",tz="UTC"))


def manual_description(days):
    """Each tuple: day, band, increments, event flags; unknown age is band None."""
    frames = []
    cursor = {}
    for day,band,returns,events in days:
        offset = cursor.get(day,0)
        index = pd.date_range(day,periods=len(returns),freq="s",tz="UTC")+pd.Timedelta(seconds=offset)
        cursor[day] = offset+len(returns)
        frames.append(pd.DataFrame({"quote":100.,"increment_known":True,"directed_log_return":returns,
                                    "tail_event":pd.array(events,dtype="boolean"),
                                    "pre_event_age_seconds":np.nan if band is None else (1 if band=="lt_N" else 600),
                                    "prior_tail_mark":np.nan if band is None else .03,
                                    "age_band":pd.array([band]*len(returns),dtype="string")},index=index))
    return pd.concat(frames).sort_index()


def synthetic_segment(days):
    frame = manual_description(days)
    return study.segment_summary(frame,0,len(frame))


def basic_segment():
    return synthetic_segment([(study.DATES[0],"lt_N",[.02,-.001,-.001],[True,False,False]),
                              (study.DATES[0],"ge_N",[.03,-.002],[True,False]),
                              (study.DATES[1],None,[.001],[False])])


def test_cutoffs_are_exact_rational_floors_and_slices_partition_rows():
    cuts = study.cutoffs(103)
    assert cuts == {"40":41,"50":51,"60":61,"70":72,"100":103}
    slices = study.segment_slices(cuts)
    assert slices["calibration40"] == (0,41) and slices["dev_validation"] == (41,72)
    assert slices["wf1"] == (41,51) and slices["wf2"] == (51,61) and slices["wf3"] == (61,72)
    assert slices["final30"] == (72,103)


@pytest.mark.parametrize("n",[True,0,-1,3.5,"100"])
def test_cutoffs_reject_invalid_counts(n):
    with pytest.raises(ValueError):study.cutoffs(n)


def test_calibration_is_fitted_once_and_excludes_increment_ending_at_cut(monkeypatch):
    frame = ticks()
    cuts = study.cutoffs(len(frame))
    original = study.fit_fixed_tail_detector
    calls = []
    def spy(prefix,side):
        calls.append((len(prefix),prefix.index[-1],side))
        return original(prefix,side)
    monkeypatch.setattr(study,"fit_fixed_tail_detector",spy)
    result = study.evaluate_symbol(frame,"BOOM600",cuts)
    assert calls == [(cuts["40"],frame.index[cuts["40"]-1],1)]
    assert result["detector"]["adequate"] and result["detector_fit_attempts"] == 1
    assert result["calibration40"]["detected_events"] == (cuts["40"]-1)//8
    # Row1200 is an event; it belongs to wf1, retaining age8 from the earlier stream.
    assert result["segments"]["wf1"]["events_with_unknown_prior_age"] == 0
    assert result["calibration40"]["events_with_unknown_prior_age"] == 1


def test_future_perturbation_cannot_change_scale_or_calibration_summary():
    frame = ticks()
    cuts = study.cutoffs(len(frame))
    altered = frame.copy()
    altered.iloc[cuts["40"]:,0] = 1e100
    a = study.evaluate_symbol(frame,"BOOM600",cuts)
    b = study.evaluate_symbol(altered,"BOOM600",cuts)
    assert a["detector"] == b["detector"] and a["calibration40"] == b["calibration40"]


def test_low_calibration_events_never_describe_later_rows_or_bootstrap(monkeypatch):
    frame = ticks(n=1000,event_every=50)
    original = study.describe_tick_tail
    lengths = []
    def spy(prefix,*args):
        lengths.append(len(prefix));return original(prefix,*args)
    def forbidden(*args,**kwargs):
        raise AssertionError("An inadequate detector cannot measure later segments")
    monkeypatch.setattr(study,"describe_tick_tail",spy)
    monkeypatch.setattr(study,"paired_hazard_comparison",forbidden)
    result = study.evaluate_symbol(frame,"BOOM600",study.cutoffs(len(frame)))
    assert lengths == [400] and result["detector"]["adequate"] is False
    assert result["detector"]["calibration_events"] < 100
    assert all(segment["status"] == "NOT TESTED" for segment in result["segments"].values())
    assert result["comparisons"]["status"] == "NOT TESTED"


def test_zero_scale_has_no_description_or_later_measurement(monkeypatch):
    frame = ticks(n=1000);frame["quote"] = 100.
    def forbidden(*args,**kwargs):raise AssertionError("No fallback detector allowed")
    monkeypatch.setattr(study,"describe_tick_tail",forbidden)
    result = study.evaluate_symbol(frame,"BOOM600",study.cutoffs(len(frame)))
    assert result["detector"]["adequate"] is False and result["detector"]["threshold"] is None
    assert result["calibration40"]["status"] == "NOT TESTED"
    assert result["comparisons"]["status"] == "NOT TESTED"


def test_no_consecutive_calibration_quotes_is_inadequate_without_filling():
    frame = ticks(n=20)
    frame.index = frame.index[0]+pd.to_timedelta(np.arange(20)*2,unit="s")
    result = study.evaluate_symbol(frame,"BOOM600",study.cutoffs(len(frame)))
    assert result["detector"]["adequate"] is False and result["detector"]["threshold"] is None


def test_tiny_calibration_prefix_is_saved_inadequate():
    result = study.evaluate_symbol(ticks(n=1),"BOOM600",study.cutoffs(1))
    assert result["detector"]["adequate"] is False
    assert result["detector"]["reason"] == "fewer_than_two_calibration_quotes"


def test_mirrored_symbol_has_same_detector_and_event_counts():
    a = study.evaluate_symbol(ticks(),"BOOM600",study.cutoffs(3000))
    b = study.evaluate_symbol(ticks(side=-1),"CRASH600",study.cutoffs(3000))
    assert b["side"] == -1
    assert a["detector"]["median_abs_log_return"] == pytest.approx(b["detector"]["median_abs_log_return"])
    assert a["calibration40"]["detected_events"] == b["calibration40"]["detected_events"]


def test_band_means_decompose_and_empty_conditional_terms_stay_null():
    summary = study.band_summary([.03,-.002,-.001],[True,False,False])
    assert summary["exposure_seconds"] == 3 and summary["events"] == 1
    assert summary["hazard"] == pytest.approx(1/3)
    assert summary["event_mean_d"] == .03 and summary["non_event_mean_d"] == -.0015
    assert summary["unconditional_mean_d"] == pytest.approx(.009)
    assert summary["decomposition_reconstructed_mean_d"] == pytest.approx(.009)
    assert summary["decomposition_identity_available"]
    assert study.band_summary([],[])["hazard"] is None
    assert study.band_summary([.03],[True])["non_event_mean_d"] is None
    assert study.band_summary([-.001],[False])["event_mean_d"] is None
    assert study.band_summary([.03],[True])["decomposition_reconstructed_mean_d"] is None


def test_segment_counts_signs_nulls_and_observed_day_zero_exposure_are_preserved():
    segment = basic_segment()
    assert segment["observed_rows"] == segment["valid_increments"] == 6
    assert segment["unknown_age_exclusions"] == 1 and segment["known_age_increments"] == 5
    assert segment["detected_events"] == 2
    assert segment["directed_sign_counts"] == {"positive":3,"negative":3,"zero":0}
    assert segment["observed_scheduled_days"] == list(study.DATES[:2])
    assert len(segment["daily_bands"]) == 2  # not176calendar days or all12source days
    assert segment["daily_bands"][1]["bands"]["lt_N"]["exposure_seconds"] == 0
    assert segment["daily_bands"][1]["bands"]["ge_N"]["exposure_seconds"] == 0
    assert segment["bands"]["lt_N"]["exposure_seconds"] == 3
    assert segment["bands"]["ge_N"]["exposure_seconds"] == 2


def test_empty_segment_has_null_bounds_means_and_no_padded_days():
    description = describe_tick_tail(ticks(n=2),FixedTailDetector(1,.001),600)
    segment = study.segment_summary(description,0,0)
    assert segment["first_observed_utc"] is None and segment["observed_cluster_count"] == 0
    assert segment["directed_return_quantiles"]["p50"] is None
    comparison = study.paired_hazard_comparison(segment)
    assert comparison["hazard_ratio_ci95"] == [None,None]
    assert comparison["valid_ratio_replicates"] == 0 and comparison["undefined_ratio_replicates"] == 9999


def test_missing_pair_and_first_event_unknown_are_separate_exclusions():
    frame = ticks(n=6,event_every=None)
    frame["quote"] = [100.,110.,109.,108.,120.,119.]
    frame.index = frame.index[0]+pd.to_timedelta([0,1,2,10,11,12],unit="s")
    description = describe_tick_tail(frame,FixedTailDetector(1,.001),600)
    segment = study.segment_summary(description,0,6)
    assert segment["initial_or_gap_pair_exclusions"] == 2
    assert segment["unknown_age_exclusions"] == 2
    assert segment["events_with_unknown_prior_age"] == 2
    assert segment["known_age_increments"] == 2


def test_bootstrap_reproduces_one_shared_seeded_draw_matrix():
    segment = synthetic_segment([(study.DATES[0],"lt_N",[.03,-.001],[True,False]),
                                 (study.DATES[0],"ge_N",[.03,-.001,-.001],[True,False,False]),
                                 (study.DATES[1],"lt_N",[.03,.03,-.001],[True,True,False]),
                                 (study.DATES[1],"ge_N",[.03,.03,-.001,-.001],[True,True,False,False])])
    comparison = study.paired_hazard_comparison(segment,101,123)
    weights = np.random.default_rng(123).multinomial(2,[.5,.5],size=101)
    young = (weights@np.array([1,2]))/(weights@np.array([2,3]))
    old = (weights@np.array([1,2]))/(weights@np.array([3,4]))
    np.testing.assert_allclose(comparison["hazard_ratio_ci95"],np.quantile(old/young,[.025,.975]))
    assert comparison == study.paired_hazard_comparison(segment,101,123)
    assert comparison["same_day_draws_for_both_bands"] and comparison["valid_ratio_replicates"] == 101


def test_high_event_counts_with_undefined_draws_cannot_falsify():
    segment = synthetic_segment([(study.DATES[0],"ge_N",[.03]*100+[-.001]*100,[True]*100+[False]*100),
                                 (study.DATES[1],"lt_N",[.03]*100+[-.001]*100,[True]*100+[False]*100)])
    comparison = study.paired_hazard_comparison(segment)
    assert comparison["hazard_ratio_ci95"] == [1.,1.] and comparison["undefined_ratio_replicates"] > 0
    gate = study.falsification_gate(segment,comparison)
    assert gate["status"] == "INSUFFICIENT_EVIDENCE" and not gate["large_overdue_effect_falsified"]
    assert "not_all9999_bootstrap_ratios_defined" in gate["reasons"]


def fully_observed_comparison():
    segment = synthetic_segment([(study.DATES[0],"lt_N",[.03]*200+[-.001]*200,[True]*200+[False]*200),
                                 (study.DATES[0],"ge_N",[.03]*100+[-.001]*300,[True]*100+[False]*300)])
    return segment,study.paired_hazard_comparison(segment)


def test_all9999_defined_bounded_ratio_falsifies_only_the_declared_large_effect():
    segment,comparison = fully_observed_comparison()
    gate = study.falsification_gate(segment,comparison)
    assert comparison["hazard_ratio_ci95"] == [.5,.5]
    assert gate["large_overdue_effect_falsified"] and gate["status"] == "REJECT_LARGE_OVERDUE_EFFECT"
    assert gate["does_not_establish_independence"] and gate["does_not_test_profitability"]
    assert gate["strategy_eligible"] is False


@pytest.mark.parametrize("point,interval",[(np.nan,[.5,.5]),(np.inf,[.5,.5]),(.5,[.5,np.nan]),
                                          (.5,[.5,np.inf]),(.5,[None,.5]),(.5,[1.,.5]),(.5,[-.1,.5])])
def test_nonfinite_or_invalid_ratio_interval_cannot_create_bounded_falsification(point,interval):
    segment,comparison = fully_observed_comparison()
    comparison.update(older_over_younger_hazard_ratio=point,hazard_ratio_ci95=interval)
    gate = study.falsification_gate(segment,comparison)
    assert gate["status"] == "INSUFFICIENT_EVIDENCE" and not gate["large_overdue_effect_falsified"]


def test_threshold_equality_and_small_sample_do_not_falsify():
    segment,comparison = fully_observed_comparison()
    comparison["hazard_ratio_ci95"] = [.5,1.5]
    assert study.falsification_gate(segment,comparison)["status"] == "LARGE_OVERDUE_EFFECT_NOT_REJECTED"
    segment["bands"]["ge_N"]["events"] = 99
    assert study.falsification_gate(segment,comparison)["status"] == "INSUFFICIENT_EVIDENCE"


def test_daily_bootstrap_stats_and_units_must_match_segment():
    segment = basic_segment()
    segment["daily_bands"][0]["bands"]["lt_N"]["events"] += 1
    with pytest.raises(ValueError):study.paired_hazard_comparison(segment)
    segment = basic_segment();segment["daily_bands"][0]["date"] = "2026-04-12"
    with pytest.raises(ValueError):study.paired_hazard_comparison(segment)


def test_unscheduled_quote_days_are_rejected():
    frame = describe_tick_tail(ticks(n=2),FixedTailDetector(1,.001),600)
    frame.index = frame.index+pd.Timedelta(days=1)
    with pytest.raises(ValueError):study.segment_summary(frame,0,2)


def metadata_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"ROOT",tmp_path)
    olddir = tmp_path/"docs/spike_tick_execution_20261005";olddir.mkdir(parents=True)
    monkeypatch.setattr(study,"ROUND7",olddir)
    def write(label,text):
        path = tmp_path/label;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return study.digest(path)
    collector = "scripts/collect_spike_ticks.py"
    collector_sha = write(collector,"synthetic collector source\n")
    old = {"safety":study.offline(),"dates":list(study.DATES),"symbols":list(study.SYMBOLS),
           "code_hashes":{collector:collector_sha},"round6_source_files":{},"round6_selection_sha256":"6"*64}
    monkeypatch.setattr(study,"frozen7_declaration",lambda path:old)
    write("docs/spike_tick_execution_20261005/declaration.json",json.dumps(old))
    write("docs/spike_tick_execution_20261005/declaration.sha256",study.digest(olddir/"declaration.json"))
    inputs = {}
    sources = {"stage":"tick_sources_frozen_before_payoff","safety":study.offline(),"tick_outcomes_evaluated":False,
               "declaration_sha256":study.digest(olddir/"declaration.json"),"sources":{},"acquisition_files":{}}
    for symbol in study.SYMBOLS:
        sources["sources"][symbol] = {}
        for date in study.DATES:
            start = int(pd.Timestamp(date,tz="UTC").timestamp())
            value = {"symbol":symbol,"date":date,"rows":86400,"missing_seconds":0,"gap_free":True,
                     "start_epoch":start,"end_exclusive_epoch":start+86400,"expected_grid_rows":86400,
                     "cadence_seconds":1,"endpoint":study.ENDPOINT,"normalization_valid":True,
                     "authentication_used":False,"fills_or_interpolations":False,
                     "features_labels_or_tick_outcomes_computed":False,"collector_sha256":collector_sha,
                     "safety":study.offline(),"reconciliation":{"price_match":True,"complete_minutes":1440,"unknown_minutes":0}}
            for filekey,shakey,suffix in (("clean_file","clean_sha256","clean.csv"),("raw_pages_file","raw_pages_sha256","raw.jsonl"),
                                         ("page_audit_file","page_audit_sha256","pages.jsonl")):
                label = f"data/{symbol}_{date}_{suffix}"
                value[filekey],value[shakey] = label,write(label,"SYNTHETIC BYTES: MUST NOT PARSE DURING DECLARATION\n")
                inputs[str(tmp_path/label)] = value[shakey]
            label = f"data/{symbol}_{date}_manifest.json"
            fingerprint = write(label,json.dumps(value))
            value["manifest"],value["manifest_sha256"] = label,fingerprint
            inputs[str(tmp_path/label)] = fingerprint
            sources["sources"][symbol][date] = value
    sources_path = olddir/"tick_sources.json"
    sources_path.write_text(json.dumps(sources));(olddir/"tick_sources.sha256").write_text(study.digest(sources_path))
    results = olddir/"results.json";results.write_text("SYNTHETIC RESULT BYTES: MUST NOT PARSE\n")
    for path in (olddir/"declaration.json",sources_path,results):inputs[str(path)] = study.digest(path)
    verifier = "scripts/verify_spike_tick_execution.py";verifier_sha = write(verifier,"synthetic verifier\n")
    audit = {"stage":"independent_primary_tick_execution_audit","safety":study.offline(),"passed":True,"errors":[],
             "sources":[{"symbol":symbol,"date":date} for symbol in study.SYMBOLS for date in study.DATES],
             "groups":[{} for _ in range(16)],"input_sha256":inputs,"verifier_file":verifier,"verifier_sha256":verifier_sha}
    (olddir/"independent_audit.json").write_text(json.dumps(audit))
    return old,sources,audit


def test_metadata_only_guard_anchors_sources_without_parsing_quotes_or_results(tmp_path,monkeypatch):
    metadata_fixture(tmp_path,monkeypatch)
    original = study.read_json
    def guarded(path):
        assert Path(path).name != "results.json"
        return original(path)
    def forbidden(*args,**kwargs):raise AssertionError("Declaration cannot parse quotes or fit")
    monkeypatch.setattr(study,"read_json",guarded)
    monkeypatch.setattr(study,"load_tick_day",forbidden)
    monkeypatch.setattr(study,"fit_fixed_tail_detector",forbidden)
    value = study.verified_round7_metadata()
    assert len(value["sources"]) == 2 and all(len(rows)==12 for rows in value["sources"].values())


@pytest.mark.parametrize("mutation",["audit_failed","audit_source_count","wrong_dates","unsafe","coverage","alias","verifier","clean_bytes","result_bytes"])
def test_prior_audit_identity_hash_coverage_and_safety_guards(tmp_path,monkeypatch,mutation):
    old,sources,audit = metadata_fixture(tmp_path,monkeypatch)
    value = sources["sources"]["BOOM600"][study.DATES[0]]
    if mutation == "audit_failed":audit["passed"] = False
    elif mutation == "audit_source_count":audit["sources"].pop()
    elif mutation == "wrong_dates":old["dates"] = old["dates"][:-1]
    elif mutation == "unsafe":value["safety"]["LIVE_ALLOWED"] = True
    elif mutation == "coverage":value["missing_seconds"] = 1
    elif mutation == "alias":
        other = sources["sources"]["BOOM600"][study.DATES[1]]
        value["clean_file"],value["clean_sha256"] = other["clean_file"],other["clean_sha256"]
    elif mutation == "verifier":(tmp_path/audit["verifier_file"]).write_text("changed")
    elif mutation == "clean_bytes":(tmp_path/value["clean_file"]).write_text("changed")
    elif mutation == "result_bytes":(study.ROUND7/"results.json").write_text("changed")
    (study.ROUND7/"independent_audit.json").write_text(json.dumps(audit))
    if mutation in {"unsafe","coverage","alias"}:
        (study.ROUND7/"tick_sources.json").write_text(json.dumps(sources))
        (study.ROUND7/"tick_sources.sha256").write_text(study.digest(study.ROUND7/"tick_sources.json"))
    with pytest.raises(ValueError):study.verified_round7_metadata()


def test_declare_and_frozen_are_metadata_only_and_refuse_overwrite(tmp_path,monkeypatch):
    metadata_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(study,"science_hashes",lambda:{"science":"a"*64})
    def forbidden(*args,**kwargs):raise AssertionError("No quote fitting before evaluate")
    monkeypatch.setattr(study,"load_tick_day",forbidden);monkeypatch.setattr(study,"fit_fixed_tail_detector",forbidden)
    output = tmp_path/"docs/pilot"
    document = study.declare(output)
    assert document["quotes_parsed"] is document["detector_calibrated"] is document["events_or_hazards_evaluated"] is False
    assert document["row_cutoffs"]["BOOM600"] == study.cutoffs(12*86400)
    assert study.frozen(output) == document
    with pytest.raises(ValueError):study.declare(output)
    monkeypatch.setattr(study,"science_hashes",lambda:{"science":"b"*64})
    with pytest.raises(ValueError):study.frozen(output)


def test_post_result_and_unsafe_environment_refuse_before_source_loading(tmp_path,monkeypatch):
    (tmp_path/"results.json").write_text("preserved")
    def forbidden(*args,**kwargs):raise AssertionError("Must refuse first")
    monkeypatch.setattr(study,"frozen",forbidden)
    with pytest.raises(ValueError):study.evaluate(tmp_path)
    monkeypatch.setenv("OPENED_TRADES","true")
    with pytest.raises(ValueError):study.declare(tmp_path)


def test_exact_csv_quote_text_decoding_and_missing_day_grid(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"ROOT",tmp_path)
    start = int(pd.Timestamp(study.DATES[0],tz="UTC").timestamp())
    quote = "100.00000000000003"
    clean = tmp_path/"clean.csv"
    with clean.open("w",newline="") as stream:
        writer = csv.writer(stream);writer.writerow(["epoch","quote"]);writer.writerows([(start,"100"),(start+1,quote)])
    raw,audit = tmp_path/"raw.jsonl",tmp_path/"audit.jsonl";raw.write_text("synthetic");audit.write_text("synthetic")
    value = {"clean_file":clean.name,"clean_sha256":study.digest(clean),"raw_pages_file":raw.name,"raw_pages_sha256":study.digest(raw),
             "page_audit_file":audit.name,"page_audit_sha256":study.digest(audit),"start_epoch":start,"end_exclusive_epoch":start+86400,
             "rows":2,"expected_grid_rows":86400,"missing_seconds":86398,"gap_free":False}
    loaded = study.load_tick_day(value)
    assert loaded.quote.iloc[1] == float(quote) and len(loaded) == 2
    value["missing_seconds"] = 0
    with pytest.raises(ValueError):study.load_tick_day(value)


def test_point_results_serialize_without_nan_and_never_become_strategy():
    result = study.evaluate_symbol(ticks(),"BOOM600",study.cutoffs(3000))
    json.dumps(result,allow_nan=False)
    assert result["profit_factor"] == result["actual_money_profit"] == "NOT TESTED"
    assert result["strategy_eligible"] is result["physical_spike_census"] is False
