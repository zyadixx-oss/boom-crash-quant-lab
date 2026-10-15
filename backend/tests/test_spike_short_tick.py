"""Synthetic-only frozen issuance fidelity, exact quote order and day inference."""
from copy import deepcopy
from dataclasses import asdict
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("short_tick_runner_tests", ROOT / "scripts/run_spike_short_tick.py")
R = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(R)
FLAGS = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "ROUND11", tmp_path / "docs/spike_short_target_20261006")
    return tmp_path / "docs/spike_short_tick_20261006"


def metadata_fixture(workspace, monkeypatch):
    root, old = R.ROOT, R.ROUND11
    old.mkdir(parents=True)
    def bytes_file(label, content=b"synthetic non-price bytes"):
        path = root/label; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        return R.digest(path)
    source_science = {"scripts/synthetic_science.py": bytes_file("scripts/synthetic_science.py")}
    lineage = {"files": {}, "sources": {}, "round6_selection_sha256": "ancestor6"}
    selection = {"safety": FLAGS, "science_code_hashes": source_science, "lineage": lineage,
                 "config": R.round11.CONFIG, "symbols": {}}
    result = {"stage": "short_target_known_history_evaluation", "adaptive_round": 11, "safety": FLAGS,
              "science_code_hashes": source_science, "lineage": lineage, "config": R.round11.CONFIG,
              "history": R.round11.HISTORY, "goal_achieved": False, "historical_strategy_candidate": False,
              "live_candidate": False, "symbols": {}}
    for symbol in R.SYMBOLS:
        source = {}
        for key, shakey in (("path", "sha256"), ("manifest", "manifest_sha256"),
                           ("raw_file", "raw_sha256"), ("page_audit", "page_audit_sha256")):
            source[key] = f"data/{symbol}_{key}.synthetic"
            source[shakey] = bytes_file(source[key])
        lineage["sources"][symbol] = {"fresh": source}
        selection["symbols"][symbol] = {"models": {}}
        models = {}
        for family in R.FAMILIES:
            for mode in R.MODES:
                key = f"{family}_{mode}"
                model = {"training_end": "2026-04-01T00:00:00+00:00", "estimator": {"threshold": .1}} if family != "CLOCK" else None
                if model:
                    selection["symbols"][symbol]["models"][key] = {"final_model": model, "development_eligible": False}
                label = f"docs/spike_short_target_20261006/signals/{symbol}_{key}.csv"
                descriptor = {"path": label, "sha256": bytes_file(label, (",".join(R.SIGNAL_COLUMNS)+"\n").encode()),
                              "rows": 0, "columns": R.SIGNAL_COLUMNS, "local_ignored_artifact": True}
                models[key] = {"family": family, "mode": mode, "config": asdict(R.COARSE),
                    "model_sha256": R.json_hash(model) if model else None, "development_eligible": False,
                    "historical_candidate": False, "reference_only": family != "SHORT44", "selected_model": False,
                    "artifacts": {"signals": descriptor}}
        result["symbols"][symbol] = {"cohorts": {R.COHORT: {"source": source, "models": models}}}
    declaration = {"safety": FLAGS}
    R.save_frozen(old/"declaration.json", declaration)
    R.save_frozen(old/"selection.json", selection)
    result.update(declaration_sha256=R.digest(old/"declaration.json"), selection_sha256=R.digest(old/"selection.json"))
    R.save_frozen(old/"results.json", result)
    scriptsha = bytes_file("scripts/verify_spike_short_target.py", b"synthetic independent verifier bytes")
    audit = {"safety": FLAGS, "status": "PASS", "pass": True, "error_count": 0, "errors": [],
        "declaration_sha256": R.digest(old/"declaration.json"), "selection_sha256": R.digest(old/"selection.json"),
        "results_sha256": R.digest(old/"results.json"), "script_sha256": scriptsha, "helper_sha256": {}}
    R.save_frozen(old/"independent_audit.json", audit)
    tick_lineage = {"round6_selection_sha256": "ancestor6", "inherited_files": {}, "sources": {}}
    monkeypatch.setattr(R.round11, "frozen_selection", lambda *a: (deepcopy(selection), R.digest(old/"selection.json")))
    monkeypatch.setattr(R.round8, "verified_round7_metadata", lambda: deepcopy(tick_lineage))
    return selection, result, audit, tick_lineage


def frozen_fixture(workspace, monkeypatch):
    monkeypatch.setattr(R, "hashes", lambda: {"synthetic": "a"*64})
    monkeypatch.setattr(R, "metadata_lineage", lambda: {"files": {}, "synthetic": True})
    return R.declare(workspace)


def rewrite(path, value):
    path.write_text(json.dumps(R.canonical(value)))
    path.with_suffix(".sha256").write_text(R.digest(path))


@pytest.mark.parametrize("flag", FLAGS)
def test_all_false_flags_required_before_metadata_or_quotes(flag, workspace, monkeypatch):
    R.assert_offline(); monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(R, "metadata_lineage", lambda: pytest.fail("unsafe metadata read"))
    for action in (R.declare, R.evaluate):
        with pytest.raises(RuntimeError, match="flags"):
            action(workspace)


@pytest.mark.parametrize("stage,name", [("declare", "declaration.sha256"), ("declare", "results.json"),
    ("evaluate", "evaluation_state.json"), ("evaluate", "results.sha256"), ("evaluate", "metrics.csv")])
def test_authoritative_started_state_or_frozen_bytes_never_restarted(stage, name, workspace, monkeypatch):
    workspace.mkdir(parents=True); (workspace/name).write_text("immutable STARTED or final artifact")
    monkeypatch.setattr(R, "metadata_lineage", lambda: pytest.fail("overwrite source read"))
    with pytest.raises(ValueError, match="overwrite"):
        {"declare": R.declare, "evaluate": R.evaluate}[stage](workspace)


def test_metadata_declaration_never_decodes_any_signal_or_price_csv(workspace, monkeypatch):
    metadata_fixture(workspace, monkeypatch)
    monkeypatch.setattr(R, "hashes", lambda: {"synthetic": "a"*64})
    for name in ("load_m1", "load_signals", "replay_ticks", "replay_timed"):
        monkeypatch.setattr(R, name, lambda *a, **k: pytest.fail("predeclaration historical calculation"))
    monkeypatch.setattr(R.round8, "load_tick_day", lambda *a: pytest.fail("predeclaration tick decode"))
    monkeypatch.setattr(R.round7, "reconcile", lambda *a: pytest.fail("predeclaration M1 calculation"))
    doc = R.declare(workspace)
    assert len(doc["lineage"]["sampled_signals"]) == 2
    assert sum(len(value) for value in doc["lineage"]["sampled_signals"].values()) == 12
    assert doc["history"]["all_twelve_tick_days_postdate_frozen_older70_fit"]
    assert not doc["prices_or_signal_csv_decoded"] and not doc["new_model_or_scores_computed"]
    assert not doc["goal_achieved"] and not doc["study_can_meet_sample_gate"]
    assert doc["config"]["maximum_common_purged_opportunities_per_symbol_model"] == 564


@pytest.mark.parametrize("problem", ["audit_fail", "audit_hash", "verifier", "result_hash", "signal_hash", "source_hash", "future_fit", "family_alias", "different_ancestor"])
def test_exact_pass11_model_issuance_sources_and_ancestry_required(problem, workspace, monkeypatch):
    selection, result, audit, tick = metadata_fixture(workspace, monkeypatch)
    if problem == "audit_fail":
        audit.update(status="FAIL", **{"pass": False}); rewrite(R.ROUND11/"independent_audit.json", audit)
    elif problem == "audit_hash": (R.ROUND11/"independent_audit.sha256").write_text("0"*64)
    elif problem == "verifier": (R.ROOT/"scripts/verify_spike_short_target.py").write_text("changed verifier")
    elif problem == "result_hash": (R.ROUND11/"results.json").write_text("changed results")
    elif problem == "signal_hash":
        desc = result["symbols"]["BOOM600"]["cohorts"][R.COHORT]["models"]["SHORT44_SPIKE"]["artifacts"]["signals"]
        R.path_for(desc["path"]).write_text("changed saved issuance")
    elif problem == "source_hash": R.path_for(selection["lineage"]["sources"]["BOOM600"]["fresh"]["path"]).write_text("changed M1")
    elif problem == "future_fit":
        selection["symbols"]["BOOM600"]["models"]["SHORT44_SPIKE"]["final_model"]["training_end"] = R.DATES[0]+"T00:00:00Z"
    elif problem == "family_alias":
        result["symbols"]["BOOM600"]["cohorts"][R.COHORT]["models"]["SHORT44_SPIKE"]["reference_only"] = 0
        rewrite(R.ROUND11/"results.json", result)
        audit["results_sha256"] = R.digest(R.ROUND11/"results.json"); rewrite(R.ROUND11/"independent_audit.json", audit)
    else: tick["round6_selection_sha256"] = "different history"
    with pytest.raises(ValueError):
        R.metadata_lineage()


@pytest.mark.parametrize("field", ["hold_bool", "history_int", "purge_float", "sample_float", "lineage_int"])
def test_frozen_decl_semantics_refuse_rehashed_numeric_boolean_aliases(field, workspace, monkeypatch):
    doc = frozen_fixture(workspace, monkeypatch)
    if field == "hold_bool": doc["config"]["tick"]["max_hold_minutes"] = True
    elif field == "history_int": doc["history"]["prospective"] = 0
    elif field == "purge_float": doc["config"]["purge_minutes"] = 31.
    elif field == "sample_float": doc["user_target"]["completed_per_symbol_model"] = 1000.
    else: doc["lineage"]["synthetic"] = 1
    rewrite(workspace/"declaration.json", doc)
    with pytest.raises(ValueError, match="semantics"):
        R.frozen(workspace)


@pytest.mark.parametrize("label", ["../outside", "/absolute", "docs//ambiguous", "docs/../source", "docs\\bad", ".", ""])
def test_path_traversal_and_ambiguous_identity_are_refused(label, workspace):
    with pytest.raises(ValueError):
        R.path_for(label)


def test_internal_and_external_symlink_aliases_are_refused(workspace):
    target = R.ROOT/"data/original"; target.parent.mkdir(); target.write_text("pinned bytes")
    (R.ROOT/"data/alias").symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        R.path_for("data/alias")
    with pytest.raises(ValueError, match="Symlink"):
        R.output_path(R.ROOT/"data/alias")


def test_output_never_aliases_prior_frozen_state(workspace):
    R.ROUND11.mkdir(parents=True)
    with pytest.raises(ValueError, match="ancestor"):
        R.output_path(R.ROUND11/"substudy")
    assert R.output_path("docs/spike_short_tick_20261006") == workspace


def ticks(n=3600, quote=100., date=None):
    return pd.DataFrame({"quote": np.full(n, quote)}, index=pd.date_range(date or R.DATES[0], periods=n, freq="s", tz="UTC"))


def minutes(frame):
    return frame.quote.resample("1min").ohlc()


def signal(time=None, side=1):
    return {"signal_time": R.stamp(time if time is not None else R.DATES[0]+"T00:00:00Z"), "atr": 1., "side": side,
            "signal_close": 100., "variant": "SHORT44_SPIKE", "score": .2}


def test_primary_policy_has_no_reward_refit_or_execution_sensitivity():
    assert asdict(R.TICK) == {"stop_atr": 2., "max_hold_minutes": 1, "entry_delay_minutes": 1,
        "round_trip_cost_atr": .1, "max_gap_seconds": 1, "stop_latency_ticks": 1}
    assert R.COARSE == R.round11.CFG and R.PURGE == 31
    assert R.CONFIG["weekly_resampling"] is False and R.CONFIG["new_scores_features_fits_or_thresholds"] is False


def test_strict_nextquote_entry_and_nominal_expiry_do_not_move_to_actual_entry():
    frame = ticks(); frame.iloc[60, 0] = 500.; frame.iloc[61, 0] = 101.; frame.iloc[120, 0] = 102.; frame.iloc[121, 0] = 103.
    tick, audit, coarse, _ = R.replay_day(frame, minutes(frame), [signal()], R.DATES[0])
    start = frame.index[0]
    assert tick.entry_time.iloc[0] == start+pd.Timedelta(seconds=61) and tick.entry.iloc[0] == 101.
    assert tick.planned_end.iloc[0] == start+pd.Timedelta(seconds=120)
    assert tick.exit_time.iloc[0] == start+pd.Timedelta(seconds=121) and tick.exit.iloc[0] == 103.
    assert tick.net_R.iloc[0] == pytest.approx(.95)
    assert coarse.entry.iloc[0] == 500. and coarse.reason.iloc[0] == "sl"
    assert audit["expiry_anchor"] == "nominal_entry_time_not_actual_entry_time"


@pytest.mark.parametrize("side", [1, -1])
def test_first_stop_crossing_uses_successor_quote_including_jump_and_recovery(side):
    frame = ticks(); frame.iloc[70, 0] = 100.-side*3.; frame.iloc[71, 0] = 100.-side*1.
    record = signal(side=side)
    primary, audit, coarse, _ = R.replay_day(frame, minutes(frame), [record], R.DATES[0])
    assert primary.trigger_time.iloc[0] == frame.index[70]
    assert primary.exit_time.iloc[0] == frame.index[71] and primary.exit.iloc[0] == 100.-side
    assert primary.net_R.iloc[0] == pytest.approx(-.55)
    assert coarse.net_R.iloc[0] == pytest.approx(-1.55)
    assert audit["stop_rule"] == "next_observed_quote_strictly_after_trigger"


def test_stop_trigger_at_nominal_deadline_takes_priority_over_timeout():
    frame = ticks(); frame.iloc[120, 0] = 97.; frame.iloc[121, 0] = 101.
    primary, audit, _, _ = R.replay_day(frame, minutes(frame), [signal()], R.DATES[0])
    assert primary.reason.iloc[0] == "sl" and primary.trigger_time.iloc[0] == frame.index[120]
    assert primary.exit.iloc[0] == 101. and primary.net_R.iloc[0] == pytest.approx(.45)
    assert audit["deadline_priority"] == "stop_trigger_at_expiry_precedes_timeout"


@pytest.mark.parametrize("gap", [61, 70, 121, 200])
def test_required_tick_gaps_preserve_unknowns_and_postexit_gap_is_irrelevant(gap):
    whole = ticks(); frame = whole.drop(whole.index[gap])
    primary, audit, _, _ = R.replay_day(frame, minutes(whole), [signal()], R.DATES[0])
    if gap == 61:
        assert audit["missing_entry"] == 1 and primary.empty
    elif gap < 122:
        assert audit["censored"] == 1 and primary.censored.iloc[0] and pd.isna(primary.net_R.iloc[0])
    else:
        assert audit["completed"] == 1 and primary.net_R.iloc[0] == pytest.approx(-.05)


def test_day_purge_is_shared_and_last_halfhour_cannot_be_rescued_by_short_exit():
    frame = ticks(n=86400)
    records = [signal(pd.Timestamp(R.DATES[0], tz="UTC")+pd.Timedelta(minutes=30*i)) for i in range(48)]
    primary, ta, coarse, ca = R.replay_day(frame, minutes(frame), records, R.DATES[0])
    assert ta["completed"] == ca["completed"] == len(primary) == len(coarse) == 47
    assert ta["purged"] == ca["purged"] == 1
    assert ta["issued"] == ca["issued"] == 48


def test_ohlc_reconciliation_never_implies_missing_second_is_filled():
    whole = ticks(n=120); frame = whole.drop(whole.index[30]); m1 = minutes(whole)
    report = R.round7.reconcile(frame, m1, R.DATES[0])
    assert report["complete_minutes"] == 1 and report["unknown_minutes"] == 1439
    changed = m1.copy(); changed.loc[changed.index[1], "close"] += 1.
    with pytest.raises(ValueError, match="mismatch"):
        R.round7.reconcile(frame, changed, R.DATES[0])


def ledger(values, days=None):
    days = list(days) if days is not None else [R.DATES[0]]*len(values)
    counts, times = {}, []
    for day in days:
        ordinal = counts.get(day, 0); counts[day] = ordinal+1
        times.append(pd.Timestamp(day, tz="UTC")+pd.Timedelta(minutes=30*ordinal))
    return pd.DataFrame({"signal_time": pd.to_datetime(times, utc=True),
        "entry_time": pd.to_datetime(times, utc=True)+pd.Timedelta(seconds=61),
        "exit_time": pd.to_datetime(times, utc=True)+pd.Timedelta(seconds=121),
        "gross_R": np.asarray(values, float)+.05, "net_R": np.asarray(values, float),
        "holding_minutes": np.ones(len(values)), "reason": ["time"]*len(values),
        "censored": [False]*len(values), "ambiguous": [False]*len(values)})


def test_descriptive_resampling_uses_twelve_observed_days_including_zeroexposure_not176calendar_days():
    data = ledger([2., -1.]*12, [day for day in R.DATES for _ in range(2)])
    result = R.summary(data)
    assert result["completed"] == 24 and result["calendar_days"] == 12 and result["span_calendar_days"] == 176
    assert result["profit_factor"] == 2. and result["mean_net_R_ci95"] == [.5, .5]
    assert result["profit_factor_ci95"] == [2., 2.] and result["cluster_se"] == 0.
    assert result["pf_undefined_replicates"] == result["mean_undefined_replicates"] == 0
    assert result["selection_score"] is None and result["no_contiguous_weekly_claim"]
    assert "p" not in result and "holm_p" not in result


@pytest.mark.parametrize("values", [[], [1.], [0.]])
def test_empty_and_noloss_pf_is_null_and_undefined_draws_remain_explicit(values):
    result = R.summary(ledger(values))
    assert result["profit_factor"] is None and result["profit_factor_ci95"] == [None, None]
    assert result["pf_undefined_replicates"] == R.BOOTSTRAP
    assert result["pf_valid_replicates"] == 0
    if not values:
        assert result["mean_undefined_replicates"] == R.BOOTSTRAP and result["mean_net_R_ci95"] == [None, None]
    else:
        assert result["mean_undefined_replicates"] > 0  # Observed empty dates remain in12-day resampling.


def test_unknown_outcome_excluded_from_mean_not_replaced_by_zero():
    frame = ledger([1., -20.]); frame.loc[1, "censored"] = True; frame.loc[1, "net_R"] = np.nan
    result = R.summary(frame)
    assert result["completed"] == 1 and result["censored"] == 1 and result["mean_net_R"] == 1.


def test_unscheduled_day_or_duplicate_paths_are_refused_before_statistics():
    with pytest.raises(ValueError, match="Unscheduled"):
        R.summary(ledger([1.], ["2026-04-12"]))
    frame = ledger([1., 2.]); frame.loc[1, "signal_time"] = frame.signal_time.iloc[0]
    with pytest.raises(ValueError, match="Unique"):
        R.matched_known(frame, ledger([1.]))


def test_matched_comparison_is_unique_inner_join_of_known_paths_and_reports_unknown_policy():
    tick, coarse = ledger([3., 1., 20.]), ledger([2., -1.])
    tick.loc[2, "censored"] = True; tick.loc[2, "net_R"] = np.nan
    matched, result = R.matched_known(tick, coarse, {"missing_entry": 1}, {"missing_entry": 0})
    assert len(matched) == result["matched_completed"] == 2
    assert result["tick_minus_coarse_mean_R"] == 1.5
    assert result["difference_ci95"] == [1.5, 1.5]
    assert result["policy_unknowns_present"] and result["tick_unknown_counts"] == {"missing_entry": 1, "censored": 1, "invalid_uncensored": 0}
    assert result["undefined_replicates"] > 0
    assert "stop-only" in result["interpretation"]


def test_clock_or_long_reference_difference_is_descriptive_unequal_exposure_and_unknowns_retained():
    model = ledger([2., -1.]*12, [day for day in R.DATES for _ in range(2)])
    clock = ledger([0.]*12, R.DATES)
    result = R.clock_comparison(model, clock, {"missing_entry": 0}, {"missing_entry": 1})
    assert result["mean_R_difference"] == .5 and result["difference_ci95"] == [.5, .5]
    assert result["policy_unknowns_present"] and result["descriptive_only"]
    assert "different saved issuance exposure" in result["comparison"]
    assert result["valid_replicates"] == R.BOOTSTRAP and result["undefined_replicates"] == 0
    assert "p" not in result


def saved_stream(workspace, rows, family="SHORT44", mode="SPIKE"):
    path = R.ROOT/"fixtures/saved.csv"; path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=R.SIGNAL_COLUMNS); writer.writeheader(); writer.writerows(rows)
    descriptor = {"path": str(path.relative_to(R.ROOT)), "sha256": R.digest(path), "rows": len(rows),
                  "columns": R.SIGNAL_COLUMNS, "local_ignored_artifact": True}
    return {"signals": descriptor, "family": family, "mode": mode,
            "training_end": "2026-04-01T00:00:00Z", "frozen_score_threshold": .1}


def saved_row(**overrides):
    return {"signal_time": R.DATES[0]+"T00:00:00+00:00", "atr": 1., "side": 1,
            "signal_close": 100., "variant": "SHORT44_SPIKE", "score": .2, **overrides}


def test_saved_csv_is_consumed_without_recomputing_features_or_scores(workspace, monkeypatch):
    value = saved_stream(workspace, [saved_row()])
    for name in ("fit_ridge", "predict_ridge", "prepared_frame", "issued"):
        monkeypatch.setattr(R.round11, name, lambda *a, **k: pytest.fail("New learner calculation forbidden"))
    records = R.load_signals(value, "BOOM600")
    assert len(records) == 1 and records[0]["score"] == .2 and records[0]["side"] == 1
    assert R.for_day(records, R.DATES[0]) == records and R.for_day(records, R.DATES[1]) == []


@pytest.mark.parametrize("problem", ["score_zero", "score_below", "nan_atr", "wrong_side", "clock", "variant", "duplicate", "before_fit"])
def test_saved_signal_semantics_refuse_corruption_even_after_descriptor_rehash(problem, workspace):
    row = saved_row()
    if problem == "score_zero": row["score"] = 0.
    elif problem == "score_below": row["score"] = .099
    elif problem == "nan_atr": row["atr"] = "NaN"
    elif problem == "wrong_side": row["side"] = -1
    elif problem == "clock": row["signal_time"] = R.DATES[0]+"T00:01:00Z"
    elif problem == "variant": row["variant"] = "OTHER"
    elif problem == "before_fit": row["signal_time"] = "2026-03-01T00:00:00Z"
    value = saved_stream(workspace, [row, row] if problem == "duplicate" else [row])
    with pytest.raises(ValueError):
        R.load_signals(value, "BOOM600")


def test_saved_clock_stream_requires_no_generated_or_fabricated_score(workspace):
    value = saved_stream(workspace, [saved_row(variant="CLOCK_SPIKE", score="")], family="CLOCK")
    assert "score" not in R.load_signals(value, "BOOM600")[0]
    value = saved_stream(workspace, [saved_row(variant="CLOCK_SPIKE", score=.2)], family="CLOCK")
    with pytest.raises(ValueError, match="unscored"):
        R.load_signals(value, "BOOM600")


def test_copied_output_artifacts_are_exclusive_and_hash_anchored(workspace):
    frame = pd.DataFrame([saved_row()], columns=R.SIGNAL_COLUMNS)
    descriptor = R.dump_frame(workspace, "signals", "sample", frame)
    assert descriptor["sha256"] == R.digest(R.path_for(descriptor["path"])) and descriptor["local_ignored_artifact"]
    before = descriptor["sha256"]
    with pytest.raises(FileExistsError):
        R.dump_frame(workspace, "signals", "sample", frame)
    assert R.digest(R.path_for(descriptor["path"])) == before


def test_perday_matching_is_point_summary_without_implicit_bootstrap():
    frame = ledger([1., -1.])
    _, result = R.matched_known(frame, frame, {"missing_entry": 0}, {"missing_entry": 0}, bootstrap=False)
    assert result["tick_minus_coarse_mean_R"] == 0.
    assert result["difference_ci95"] == [None, None]
    assert result["bootstrap_repeats"] == result["valid_replicates"] == result["undefined_replicates"] == 0
    assert result["bootstrap_seed"] is None and result["inference_status"] == "NOT_REQUESTED_PER_DAY_POINT_SUMMARY"


def test_full_synthetic_evaluate_keeps_empty_models_blank_clocks_gaps_and_purge(workspace, monkeypatch, capsys):
    """Exercise the declared orchestration and real engines, without historical files."""
    sampled, tick_sources, tick_frames, m1_frames = {}, {}, {}, {}
    for symbol in R.SYMBOLS:
        sampled[symbol], tick_sources[symbol], tick_frames[symbol] = {}, {}, {}
        days = []
        for ordinal, date in enumerate(R.DATES):
            whole = ticks(n=3600, date=date)
            if ordinal == 2:
                whole.iloc[70, 0], whole.iloc[71, 0] = 97., 99.
            days.append(minutes(whole))
            tick_frames[symbol][date] = whole.drop(whole.index[61 if ordinal == 0 else 70]) if ordinal < 2 else whole
            tick_sources[symbol][date] = {"symbol": symbol, "date": date, "synthetic": True}
        m1_frames[symbol] = pd.concat(days)
        for family in R.FAMILIES:
            for mode in R.MODES:
                key, rows = f"{family}_{mode}", []
                side = (1 if symbol == "BOOM600" else -1) * (1 if mode == "SPIKE" else -1)
                if key != "SHORT44_SPIKE":
                    for date in R.DATES:
                        issue_minutes = [0] if family == "SHORT44" else [0, 30]
                        if family == "CLOCK": issue_minutes.append(23*60+30)
                        for minute in issue_minutes:
                            rows.append(saved_row(signal_time=R.stamp(date+"T00:00:00Z")+pd.Timedelta(minutes=minute),
                                side=side, variant=key, score="" if family == "CLOCK" else .2))
                path = R.ROOT/f"fixtures/{symbol}_{key}.csv"; path.parent.mkdir(exist_ok=True)
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=R.SIGNAL_COLUMNS)
                    writer.writeheader(); writer.writerows(rows)
                sampled[symbol][key] = {"signals": {"path": str(path.relative_to(R.ROOT)), "sha256": R.digest(path),
                    "rows": len(rows), "columns": R.SIGNAL_COLUMNS, "local_ignored_artifact": True},
                    "family": family, "mode": mode, "model_sha256": "synthetic" if family != "CLOCK" else None,
                    "development_eligible": False, "reference_only": family != "SHORT44", "round11_selected_model": False,
                    "training_end": "2026-04-01T00:00:00Z" if family != "CLOCK" else None,
                    "frozen_score_threshold": .1 if family != "CLOCK" else None}
    lineage = {"files": {}, "sampled_signals": sampled,
        "m1_sources": {symbol: {"path": f"fixtures/{symbol}_M1.synthetic", "symbol": symbol} for symbol in R.SYMBOLS},
        "tick_lineage": {"sources": tick_sources}}
    monkeypatch.setattr(R, "hashes", lambda: {"synthetic": "a"*64})
    monkeypatch.setattr(R, "metadata_lineage", lambda: deepcopy(lineage))
    monkeypatch.setattr(R, "load_m1", lambda path: (m1_frames[path.name.split("_M1")[0]].copy(), {"synthetic": True}))
    monkeypatch.setattr(R, "check_prepared", lambda *a: None)
    monkeypatch.setattr(R.round8, "load_tick_day", lambda source: tick_frames[source["symbol"]][source["date"]].copy())
    for name in ("fit_ridge", "predict_ridge", "prepared_frame", "issued"):
        monkeypatch.setattr(R.round11, name, lambda *a, **k: pytest.fail("New learner computation is forbidden"))
    declared = R.declare(workspace)
    result = R.evaluate(workspace)
    assert "SHORT_TICK_COMPLETE" in capsys.readouterr().out
    assert result["declaration_sha256"] == R.digest(workspace/"declaration.json")
    assert R.digest(workspace/"results.json") == (workspace/"results.sha256").read_text().strip()
    assert result["safety"] == FLAGS and result["history"] == declared["history"]
    assert not result["goal_achieved"] and not result["live_candidate"] and not result["historical_strategy_candidate"]
    for symbol in R.SYMBOLS:
        assert len(result["symbols"][symbol]["inputs"]) == 12
        models = result["symbols"][symbol]["models"]
        empty = models["SHORT44_SPIKE"]
        assert empty["tick"]["metrics"]["completed"] == empty["coarse"]["metrics"]["completed"] == 0
        assert empty["tick"]["metrics"]["profit_factor"] is None
        assert empty["tick"]["metrics"]["pf_undefined_replicates"] == R.BOOTSTRAP
        assert empty["matched_known"]["metrics"]["undefined_replicates"] == R.BOOTSTRAP
        assert empty["clock_comparisons"]["tick"]["undefined_replicates"] == R.BOOTSTRAP
        for family in R.FAMILIES:
            for mode in R.MODES:
                row = models[f"{family}_{mode}"]
                assert len(row["days"]) == 12
                assert row["selected_model"] is row["historical_candidate"] is row["supports_expected_pf_1_5"] is False
                assert row["user_target_observed"] is row["conditional_economic_gates_pass"] is False
                for day in row["days"]:
                    match = day["matched_known"]["metrics"]
                    assert match["bootstrap_repeats"] == 0 and match["inference_status"] == "NOT_REQUESTED_PER_DAY_POINT_SUMMARY"
                if family == "CLOCK":
                    assert row["tick"]["audit_totals"]["issued"] == 36
                    assert row["tick"]["audit_totals"]["purged"] == row["coarse"]["audit_totals"]["purged"] == 12
                    assert row["tick"]["audit_totals"]["missing_entry"] == 1
                    assert row["tick"]["audit_totals"]["censored"] == 1
                    assert row["tick"]["metrics"]["completed"] == 22
                    assert row["coarse"]["metrics"]["completed"] == 24
                    matched = row["matched_known"]["metrics"]
                    assert matched["matched_completed"] == 22 and matched["unmatched_coarse_completed"] == 2
                    assert matched["policy_unknowns_present"]
                    copied = R.path_for(row["days"][0]["signals"]["path"])
                    with copied.open(newline="") as stream:
                        assert all(record["score"] == "" for record in csv.DictReader(stream))
    assert len(pd.read_csv(workspace/"metrics.csv")) == 24
    preserved = R.digest(workspace/"results.json")
    with pytest.raises(ValueError, match="overwrite"):
        R.evaluate(workspace)
    assert R.digest(workspace/"results.json") == preserved
