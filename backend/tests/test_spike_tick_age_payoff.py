"""Round9 release, causal-prefix and economic inference tests; synthetic only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.research.multiframe_signal import FEATURE_NAMES
from app.research.payoff_ticks import replay_ticks

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tick_age_payoff_tests", ROOT / "scripts/run_spike_tick_age_payoff.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
N44 = list(FEATURE_NAMES)
N46 = [*N44, "tail_age_log1p", "prior_tail_mark"]
FLAGS = dict.fromkeys(R.round8.FLAGS, False)


def ticks(day="2026-04-11", minutes=90, slope=0.):
    index = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=minutes * 60, freq="s").as_unit("ns")
    return pd.DataFrame({"quote": 100 + np.arange(len(index)) * slope}, index=index)


def signals(day="2026-04-11", count=3, age=20):
    index = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=count, freq="5min")
    return [{"signal_time": t, "atr": 1., "side": 1, "variant": "test", "prior_age_seconds": age} for t in index]


def simple_ledger(dates, values, *, censored=None):
    return pd.DataFrame({"signal_time": pd.to_datetime(dates, utc=True), "net_R": values,
                         "censored": censored if censored is not None else np.zeros(len(values), bool)})


def model_stub(family="RIDGE44", status="TESTED"):
    names = N44 if family == "RIDGE44" else N46
    return {"family": family, "status": status, "feature_names": names,
            "estimator": {"threshold": .25}, "serialized_estimator_sha256": "synthetic"}


def training_fixture(n=1000):
    # 250 five-minute opportunities per declared source day, with the daily
    # planned31-minute guard intact. These targets are deliberately overlapping.
    chunks = []
    for day in R.DATES[:4]:
        chunks.extend(pd.date_range(pd.Timestamp(day, tz="UTC"), periods=250, freq="5min"))
    index = pd.DatetimeIndex(chunks[:n]).as_unit("ns")
    q = np.arange(n, dtype=float)
    values = np.column_stack([np.sin(q / (j + 2)) for j in range(46)])
    rows = pd.DataFrame(values, index=index, columns=N46)
    rows["atr"], rows["close"], rows["prior_age_seconds"] = 1., 100., 20.
    labels = pd.DataFrame({"signal_time": index, "net_R": .3 + values[:, 0] * .1,
                           "censored": False, "planned_end": index + pd.Timedelta(minutes=16)})
    return rows, labels, R.START, pd.Timestamp(R.DATES[3], tz="UTC") + pd.Timedelta(days=1)


def empty_context():
    rows = pd.DataFrame(columns=[*N46, "atr", "close", "prior_age_seconds"],
                        index=pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
    return {"rows": rows, "ticks": ticks().iloc[:0], "names44": N44, "names46": N46, "inputs": rows.reset_index(drop=True), "audit": {"synthetic": True}}


def declaration_fixture():
    bounds = {"0": R.START.isoformat(), "40": "2026-06-13T19:11:56+00:00",
              "50": "2026-06-29T23:59:58+00:00", "60": "2026-07-31T04:47:57+00:00",
              "70": "2026-08-16T09:36:02+00:00", "100": R.END.isoformat()}
    lineage = {"boundaries": {symbol: deepcopy(bounds) for symbol in R.SYMBOLS},
               "detectors": {symbol: {"adequate": True} for symbol in R.SYMBOLS}}
    return {"stage": "tick_age_payoff_premeasurement_declaration", "run_utc": "synthetic",
            "safety": deepcopy(FLAGS), "config": R.CONFIG, "dates": list(R.DATES), "symbols": list(R.SYMBOLS),
            "families": list(R.FAMILIES), "modes": list(R.MODES), "science_code_hashes": {"fake": "hash"},
            "lineage": lineage, "quotes_parsed": False, "new_features_labels_models_computed": False,
            "goal_achieved": False, "historical_strategy_candidate": False}


def freeze_fixture(tmp_path, monkeypatch):
    declaration = declaration_fixture()
    R.save(tmp_path / "declaration.json", declaration)
    (tmp_path / "declaration.sha256").write_text(R.digest(tmp_path / "declaration.json") + "\n")
    monkeypatch.setattr(R, "hashes", lambda: declaration["science_code_hashes"])
    monkeypatch.setattr(R, "verified_round8_metadata", lambda: declaration["lineage"])
    return declaration


@pytest.mark.parametrize("flag", R.round8.FLAGS)
def test_each_live_flag_blocks_every_stage_before_source_read(flag, tmp_path, monkeypatch):
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(R, "verified_round8_metadata", lambda: pytest.fail("No metadata read when unsafe"))
    for action in (R.declare, R.develop, R.evaluate):
        with pytest.raises(ValueError, match="flags"):
            action(tmp_path)


@pytest.mark.parametrize("stage,name", [("declare", "declaration.json"), ("declare", "selection.sha256"),
                                         ("develop", "selection.json"), ("develop", "results.sha256"),
                                         ("evaluate", "metrics.csv"), ("evaluate", "results.json")])
def test_stage_no_overwrite_before_any_source_read(stage, name, tmp_path, monkeypatch):
    (tmp_path / name).write_text("frozen")
    monkeypatch.setattr(R, "verified_round8_metadata", lambda: pytest.fail("No metadata read on overwrite"))
    with pytest.raises(ValueError, match="overwrite"):
        {"declare": R.declare, "develop": R.develop, "evaluate": R.evaluate}[stage](tmp_path)


def test_declaration_only_hashes_metadata_and_performs_no_price_or_label_calculation(tmp_path, monkeypatch):
    lineage = {"synthetic": "metadata"}
    monkeypatch.setattr(R, "hashes", lambda: {"fake": "sha"})
    monkeypatch.setattr(R, "verified_round8_metadata", lambda: lineage)
    for name in ("load_m1", "describe_tick_tail", "causal_multiframe_inputs", "independent_tick_labels", "fit_ridge"):
        monkeypatch.setattr(R, name, lambda *a, **k: pytest.fail("Declaration must never measure"))
    monkeypatch.setattr(R.round8, "load_tick_day", lambda *a: pytest.fail("No quote decode during declaration"))
    value = R.declare(tmp_path)
    assert value["quotes_parsed"] is False and value["new_features_labels_models_computed"] is False
    assert value["user_target"]["pilot_can_meet_target"] is False
    assert value["safety"] == FLAGS
    assert R.digest(tmp_path / "declaration.json") == (tmp_path / "declaration.sha256").read_text().strip()


@pytest.mark.parametrize("kind", ("source", "science", "bytes", "safety"))
def test_frozen_declaration_drift_blocks_before_any_features(kind, tmp_path, monkeypatch):
    declaration = freeze_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: pytest.fail("Cannot measure changed declaration"))
    if kind == "source":
        monkeypatch.setattr(R, "verified_round8_metadata", lambda: {"changed": True})
    elif kind == "science":
        monkeypatch.setattr(R, "hashes", lambda: {"changed": "sha"})
    elif kind == "bytes":
        (tmp_path / "declaration.json").write_text("{}")
    else:
        declaration["safety"]["LIVE_ALLOWED"] = True
        (tmp_path / "declaration.json").write_text(json.dumps(declaration))
        (tmp_path / "declaration.sha256").write_text(R.digest(tmp_path / "declaration.json"))
    with pytest.raises(ValueError):
        R.develop(tmp_path)


def test_inside_minute_cutoff_withholds_unclosed_m1_and_suffix_tail_rows(monkeypatch):
    end = pd.Timestamp("2026-04-11T00:17:30Z")
    m1_index = pd.date_range("2026-04-10T23:50Z", "2026-04-11T00:20Z", freq="min").as_unit("ns")
    m1 = pd.DataFrame(dict(open=100., high=101., low=99., close=100.), index=m1_index)
    source = ticks(minutes=25)
    calls = []
    monkeypatch.setattr(R, "DATES", ("2026-04-11", "2026-04-12"))
    monkeypatch.setattr(R, "load_m1", lambda *a: (m1, {}))
    monkeypatch.setattr(R, "check_prepared", lambda *a: None)
    def loader(value):
        calls.append(value["date"])
        assert value["date"] == "2026-04-11"
        return source
    monkeypatch.setattr(R.round8, "load_tick_day", loader)
    def descriptor(frame, detector, nominal):
        assert frame.index.max() < end and len(frame) == 17 * 60 + 30
        assert detector.median_abs_log_return == 1e-6 and nominal == 600
        return pd.DataFrame({"increment_known": True, "pre_event_age_seconds": 10., "prior_tail_mark": .01}, index=frame.index)
    monkeypatch.setattr(R, "describe_tick_tail", descriptor)
    def features(frame, direction):
        assert direction == "boom"
        assert (frame.index + pd.Timedelta(minutes=1) <= end).all()
        assert frame.index.max() == pd.Timestamp("2026-04-11T00:16Z")
        opening = pd.date_range("2026-04-11T00:00Z", periods=4, freq="5min").as_unit("ns")
        table = pd.DataFrame(1., index=opening, columns=N44)
        table["feature_valid"] = [True, False, True, False]
        table["atr"], table["close"] = 1., 100.
        return table, N44
    monkeypatch.setattr(R, "causal_multiframe_inputs", features)
    declaration = {"lineage": {"m1_sources": {"BOOM600": {"path": "data/synthetic.csv"}},
        "tick_sources": {"BOOM600": {"2026-04-11": {"date": "2026-04-11"}}},
        "detectors": {"BOOM600": {"adequate": True, "median_abs_log_return": 1e-6, "threshold": 10. * 1e-6}}}}
    context = R.prepare_prefix(declaration, "BOOM600", end)
    assert calls == ["2026-04-11"]
    assert list(context["rows"].index) == [pd.Timestamp("2026-04-11T00:05Z"), pd.Timestamp("2026-04-11T00:15Z")]
    assert context["names46"] == N46 and context["names44"] == N44
    assert context["rows"].prior_age_seconds.eq(10.).all()
    assert context["audit"]["suffix_calculations"] is False


def test_exact_row_age_joins_never_fill_a_missing_tick_decision(monkeypatch):
    end = pd.Timestamp("2026-04-11T00:20Z")
    source = ticks(minutes=20).drop(pd.Timestamp("2026-04-11T00:10Z"))
    monkeypatch.setattr(R, "DATES", ("2026-04-11",))
    m1 = pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
                      index=pd.date_range("2026-04-10T23:00Z", periods=81, freq="min").as_unit("ns"))
    monkeypatch.setattr(R, "load_m1", lambda *a: (m1, {}))
    monkeypatch.setattr(R, "check_prepared", lambda *a: None)
    monkeypatch.setattr(R.round8, "load_tick_day", lambda *a: source)
    monkeypatch.setattr(R, "describe_tick_tail", lambda f, *a: pd.DataFrame(
        {"increment_known": True, "pre_event_age_seconds": 30., "prior_tail_mark": .01}, index=f.index))
    table = pd.DataFrame(1., index=pd.date_range("2026-04-11T00:00Z", periods=3, freq="5min").as_unit("ns"), columns=N44)
    table["feature_valid"], table["atr"], table["close"] = True, 1., 100.
    monkeypatch.setattr(R, "causal_multiframe_inputs", lambda *a: (table, N44))
    d = {"lineage": {"m1_sources": {"BOOM600": {"path": "data/synthetic.csv"}},
        "tick_sources": {"BOOM600": {"2026-04-11": {}}},
        "detectors": {"BOOM600": {"adequate": True, "median_abs_log_return": 1e-6, "threshold": 10. * 1e-6}}}}
    context = R.prepare_prefix(d, "BOOM600", end)
    assert pd.Timestamp("2026-04-11T00:10Z") not in context["rows"].index
    for model in (None, model_stub("RIDGE44", "NOT TESTED"), model_stub("RIDGE46", "NOT TESTED")):
        issued = R.issue(context["rows"], "BOOM600", "SPIKE", model)
        assert all(r["signal_time"] != pd.Timestamp("2026-04-11T00:10Z") for r in issued)


def test_individually_replayed_targets_overlap_but_strategy_uses_one_open():
    source = ticks(minutes=90)
    records = signals(count=4)
    labels, independent = R.replay_partition(source, records, R.START, R.START + pd.Timedelta(minutes=90), overlapping=True)
    strategy, joint = R.replay_partition(source, records, R.START, R.START + pd.Timedelta(minutes=90))
    assert len(labels) == 4 and len(strategy) == 1
    assert independent["overlap_skipped"] == 0 and joint["overlap_skipped"] == 3
    assert independent["counts_toward_profit_sample_target"] is False
    assert independent["statistically_independent_labels"] is False
    assert joint["strategy_returns"] is True


def test_effective_cutoff_purge_is_planned_and_never_rescued_by_early_stop():
    source = ticks(minutes=70)
    source.loc[source.index >= R.START + pd.Timedelta(minutes=32), "quote"] = 90.
    records = [{**signals(count=1)[0], "signal_time": R.START + pd.Timedelta(minutes=30)}]
    cutoff = R.START + pd.Timedelta(minutes=60)
    assert R.planned_eligible(records, R.START, cutoff) == []
    ledger, audit = R.replay_partition(source, records, R.START, cutoff, overlapping=True)
    assert ledger.empty and audit["purged"] == 1
    later, _ = R.replay_partition(source, records, R.START, cutoff + pd.Timedelta(minutes=1), overlapping=True)
    assert len(later) == 1 and later.reason.iloc[0] == "sl"


def test_missing_entry_is_unknown_and_reserves_joint_occupancy():
    source = ticks(minutes=90).drop(R.START + pd.Timedelta(minutes=1, seconds=1))
    records = signals(count=3)
    ledger, audit = R.replay_partition(source, records, R.START, R.START + pd.Timedelta(minutes=90))
    assert ledger.empty and audit["missing_entry"] == 1 and audit["overlap_skipped"] == 2
    labels, independent = R.replay_partition(source, records, R.START, R.START + pd.Timedelta(minutes=90), overlapping=True)
    assert len(labels) == 2 and independent["missing_entry"] == 1


def test_44_and46_fit_exactly_same_completed_rows_targets_and_training_only_cutoff():
    rows, labels, start, end = training_fixture()
    pair = R.training_pair(rows, labels.sample(frac=1, random_state=4), N44, N46, start, end)
    assert pair["RIDGE44"]["status"] == pair["RIDGE46"]["status"] == "TESTED"
    assert pair["RIDGE44"]["common_timestamp_target_sha256"] == pair["RIDGE46"]["common_timestamp_target_sha256"]
    assert pair["RIDGE44"]["training_completed_labels"] == 1000
    for family, names in (("RIDGE44", N44), ("RIDGE46", N46)):
        model = pair[family]
        assert model["estimator"]["means"] == pytest.approx(rows[names].mean().to_numpy())
        assert model["serialized_estimator_sha256"] == R.json_hash(model["estimator"])
    outside = rows.iloc[:1].copy()
    outside.index = pd.DatetimeIndex([end + pd.Timedelta(days=5)])
    outside[N46] = 1e12
    heldout = pd.DataFrame({"signal_time": outside.index, "net_R": [1e12], "censored": [False]})
    changed = R.training_pair(pd.concat([rows, outside]), pd.concat([labels, heldout]), N44, N46, start, end)
    assert changed == pair


@pytest.mark.parametrize("failure", ("999", "censored", "invalid", "planned_purge", "unsampled_day"))
def test_1000_completed_label_floor_is_not_lowered(failure, monkeypatch):
    rows, labels, start, end = training_fixture()
    if failure == "999":
        labels = labels.iloc[:-1]
    elif failure == "censored":
        labels.loc[0, "censored"] = True
    elif failure == "invalid":
        labels.loc[0, "net_R"] = np.nan
    elif failure == "planned_purge":
        t = end - pd.Timedelta(minutes=30)
        rows.loc[t] = rows.iloc[0]
        labels.loc[0, "signal_time"] = t
    else:
        t = start + pd.Timedelta(days=1)
        rows.loc[t] = rows.iloc[0]
        labels.loc[0, "signal_time"] = t
    monkeypatch.setattr(R, "fit_ridge", lambda *a, **k: pytest.fail("Never lower the floor"))
    pair = R.training_pair(rows, labels, N44, N46, start, end)
    assert all(m["status"] == "NOT TESTED" and m["estimator"] is None for m in pair.values())


@pytest.mark.parametrize("problem", ("duplicate_labels", "duplicate_clocks", "missing_row", "nonfinite_feature", "wrong_order"))
def test_alignment_and_feature_schema_guards_reject_silent_changes(problem):
    rows, labels, start, end = training_fixture()
    names = N44
    if problem == "duplicate_labels":
        labels = pd.concat([labels, labels.iloc[:1]])
    elif problem == "duplicate_clocks":
        rows = pd.concat([rows, rows.iloc[:1]])
    elif problem == "missing_row":
        rows = rows.iloc[1:]
    elif problem == "nonfinite_feature":
        rows.iloc[0, 0] = np.nan
    else:
        names = N44[::-1]
    with pytest.raises(ValueError):
        R.training_pair(rows, labels, names, N46, start, end)


def test_negative_scores_or_insufficient_fit_never_force_a_signal(monkeypatch):
    rows, _, _, _ = training_fixture(3)
    monkeypatch.setattr(R, "predict_ridge", lambda *a, **k: np.array([-.5, 0., .2]))
    model = model_stub()
    assert R.issue(rows, "BOOM600", "SPIKE", model) == []
    assert R.issue(rows, "BOOM600", "SPIKE", model_stub(status="NOT TESTED")) == []
    monkeypatch.setattr(R, "predict_ridge", lambda *a, **k: np.array([.3, .2, -.1]))
    issued = R.issue(rows, "CRASH600", "DRIFT", model)
    assert len(issued) == 1 and issued[0]["side"] == 1 and issued[0]["score"] == .3
    assert model["estimator"]["threshold"] == .25
    assert len(R.issue(rows, "BOOM600", "SPIKE")) == 3


def test_observed_day_units_are_only_inside_exact_partition_with_zero_exposure_retained():
    start, end = pd.Timestamp("2026-06-13T19:11:56Z"), pd.Timestamp("2026-08-16T09:36:02Z")
    days, draws = R.weights(start, end)
    assert days == ["2026-06-13", "2026-06-29", "2026-07-15", "2026-07-31", "2026-08-16"]
    assert draws.shape == (9999, 5) and (draws.sum(axis=1) == 5).all()
    final, _ = R.weights(end, R.END)
    assert final == ["2026-08-16", "2026-09-01", "2026-09-17", "2026-10-03"]
    assert "2026-04-11" not in final
    ledger = simple_ledger(["2026-06-13T20:00Z"], [-1.])
    m = R.metric(ledger, {}, days, draws)
    assert m["observed_day_clusters"] == 5 and m["active_days"] == 1
    assert m["mean_inference"]["undefined_draws"] > 0
    assert m["selection_score"] is None and m["bounded_inference_allowed"] is False


def test_bootstrap_means_use_pooled_returns_counts_instead_of_mean_of_day_means():
    days = ["2026-04-11", "2026-04-26"]
    ledger = simple_ledger(["2026-04-11T00:00Z", "2026-04-26T00:00Z", "2026-04-26T00:05Z"], [-2., 1., 1.])
    sums = R.day_sums(ledger, days)
    draws = np.array([[1, 1], [2, 0], [0, 2]])
    means, pf = R.draw_means(sums, draws)
    assert means == pytest.approx([0., -2., 1.])
    assert pf[:2] == pytest.approx([1., 0.]) and np.isnan(pf[2])
    assert R.metric(ledger, {}, days, draws)["mean_net_R"] == 0.


def test_paired_bootstrap_uses_shared_weights_and_marks_undefined_pairs():
    days = ["2026-04-11", "2026-04-26"]
    left = simple_ledger(["2026-04-11T00:00Z", "2026-04-26T00:00Z"], [2., -2.])
    right = simple_ledger(["2026-04-11T00:00Z", "2026-04-26T00:00Z"], [1., -1.])
    draws = np.array([[1, 1], [2, 0], [0, 2]])
    paired = R.paired_difference(left, right, days, draws)
    assert paired["mean_difference_R"] == 0.
    assert paired["ci95"] == pytest.approx([-.95, .95])
    assert paired["undefined_draws"] == 0
    assert R.paired_difference(left, left, days, draws)["ci95"] == [0., 0.]
    empty = right.iloc[:0]
    undefined = R.paired_difference(left, empty, days, draws)
    assert undefined["ci95"] is None and undefined["undefined_draws"] == 3
    assert undefined["all_draws_defined"] is False


@pytest.mark.parametrize("unknown", ("none", "missing_entry", "censored", "invalid"))
def test_band_rejection_counts_unknowns_before_realized_outcome_filter(unknown):
    start, end = R.START, pd.Timestamp("2026-04-27T00:00Z")
    days, draws = R.weights(start, end)
    planned = signals(count=60) + signals("2026-04-26", count=60)
    ledger = simple_ledger([r["signal_time"] for r in planned], [-.2] * 120)
    if unknown == "missing_entry":
        ledger = ledger.iloc[1:].copy()
    elif unknown == "censored":
        ledger.loc[0, "censored"], ledger.loc[0, "net_R"] = True, np.nan
    elif unknown == "invalid":
        ledger.loc[0, "net_R"] = np.nan
    band = R.age_diagnostics(ledger, planned, start, end, days, draws)["lt600"]
    assert band["planned_eligible_signals"] == 120
    assert band["metrics"]["day_mean_ci95"] == pytest.approx([-.2, -.2])
    if unknown == "none":
        assert band["status"] == "REJECT_POSITIVE_FIXED_OVERLAPPING_LABEL_MEAN"
        assert band["known_subset_only"] is False
    else:
        assert band["status"] == "INSUFFICIENT_EVIDENCE"
        assert band["known_subset_only"] is True
        assert band["whole_eligible_policy_rejection_allowed"] is False
    assert "profit_factor" not in band["metrics"] and "pf_inference" not in band["metrics"]
    assert band["strategy_pf_evidence"] is False and band["statistically_independent_labels"] is False


def test_ge600_is_exact_boundary_and_zero_band_days_block_bounded_rejection():
    start, end = R.START, pd.Timestamp("2026-04-27T00:00Z")
    days, draws = R.weights(start, end)
    planned = signals(count=120, age=600)
    ledger = simple_ledger([r["signal_time"] for r in planned], [-1.] * 120)
    report = R.age_diagnostics(ledger, planned, start, end, days, draws)
    assert report["lt600"]["planned_eligible_signals"] == 0
    assert report["ge600"]["planned_eligible_signals"] == 120
    assert report["ge600"]["metrics"]["mean_inference"]["undefined_draws"] > 0
    assert report["ge600"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_shared_common_clock_is_used_for_both_models_and_clock_reference(tmp_path, monkeypatch):
    rows, _, start, _ = training_fixture(4)
    context = {"rows": rows, "ticks": ticks(minutes=90)}
    pair = {family: model_stub(family) for family in R.FAMILIES}
    calls = []
    def predict(frame, *a, **k):
        calls.append(frame.index.copy())
        return np.ones(len(frame))
    monkeypatch.setattr(R, "predict_ridge", predict)
    monkeypatch.setattr(R, "dump_frame", lambda *a: {"synthetic": True})
    reports, comparisons, ledgers = R.evaluate_partition(context, pair, "BOOM600", "SPIKE", start,
                                                       start + pd.Timedelta(minutes=90), tmp_path, "fake")
    assert len(calls) == 2 and calls[0].equals(calls[1]) and calls[0].equals(rows.index)
    assert all(report["common_available_rows"] == 4 for report in reports.values())
    assert comparisons["RIDGE46_minus_RIDGE44"]["mean_difference_R"] == 0.
    assert all(len(ledger) == 1 for ledger in ledgers.values())


def test_develop_only_uses40_50_60_70_prefixes_and_freezes_before_final30(tmp_path, monkeypatch):
    declaration = freeze_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "SYMBOLS", ("BOOM600",))
    # Frozen declaration identity was created with two symbols; keep its guard
    # mocked here so the test focuses on development's calculation boundaries.
    monkeypatch.setattr(R, "frozen_declaration", lambda *a: declaration)
    calls = []
    def prepare(document, symbol, end):
        calls.append(end)
        assert end <= pd.Timestamp(declaration["lineage"]["boundaries"][symbol]["70"])
        return empty_context()
    monkeypatch.setattr(R, "prepare_prefix", prepare)
    monkeypatch.setattr(R, "dump_frame", lambda *a: {"path": "synthetic", "sha256": "synthetic", "rows": len(a[-1])})
    result = R.develop(tmp_path)
    assert calls == [pd.Timestamp(declaration["lineage"]["boundaries"]["BOOM600"][key]) for key in ("40", "50", "60", "70")]
    assert (tmp_path / "selection.json").exists() and (tmp_path / "selection.sha256").exists()
    assert not (tmp_path / "results.json").exists()
    assert result["final30_features_or_payoffs_evaluated"] is False
    saved = result["symbols"]["BOOM600"]
    assert saved["selected_candidate"] is None
    assert all(not c["development_eligible"] for c in saved["candidates"].values())
    for candidate in saved["candidates"].values():
        assert [(f["training"], f["test"]) for f in candidate["walk_forward"]] == [("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")]
        assert candidate["validation"]["observed_day_clusters"] == 5
    assert all(m["status"] == "NOT TESTED" for m in saved["final_models"].values())


def test_changed_selection_blocks_final_features_before_release(tmp_path, monkeypatch):
    declaration = freeze_fixture(tmp_path, monkeypatch)
    selection = {"stage": "frozen_tick_age_payoff_development", "declaration_sha256": R.digest(tmp_path / "declaration.json"),
                 "safety": deepcopy(FLAGS), "science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"],
                 "config": R.CONFIG, "final30_features_or_payoffs_evaluated": False,
                 "goal_achieved": False, "historical_strategy_candidate": False,
                 "symbols": {s: {"selected_candidate": None, "final_models": {}} for s in R.SYMBOLS}}
    R.save(tmp_path / "selection.json", selection)
    (tmp_path / "selection.sha256").write_text(R.digest(tmp_path / "selection.json"))
    # External reformatting still changes frozen bytes, even if semantic values
    # coincide. The selection hash, rather than permissive JSON equality, binds it.
    (tmp_path / "selection.json").write_text(json.dumps(selection, separators=(",", ":")))
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: pytest.fail("No final feature/label read after tamper"))
    with pytest.raises(ValueError, match="selection changed"):
        R.evaluate(tmp_path)


def test_evaluation_only_releases_final30_and_never_refits_even_with_good_points(tmp_path, monkeypatch):
    declaration = declaration_fixture()
    context = empty_context()
    pair = {mode + "_" + family: model_stub(family, "NOT TESTED") for mode in R.MODES for family in R.FAMILIES}
    selection = {"declaration_sha256": "saved", "safety": deepcopy(FLAGS), "science_code_hashes": {},
                 "lineage": declaration["lineage"], "config": R.CONFIG,
                 "symbols": {s: {"status": "TESTED", "final_models": deepcopy(pair),
                       "candidates": {key: {"development_eligible": False} for key in pair}} for s in R.SYMBOLS}}
    monkeypatch.setattr(R, "frozen_selection", lambda *a: (selection, "frozen"))
    monkeypatch.setattr(R, "fit_ridge", lambda *a, **k: pytest.fail("No fitting on final"))
    monkeypatch.setattr(R, "training_pair", lambda *a: pytest.fail("No training on final"))
    calls = []
    def prepare(value, symbol, end):
        assert end == R.END
        calls.append(symbol)
        return context
    monkeypatch.setattr(R, "prepare_prefix", prepare)
    monkeypatch.setattr(R, "dump_frame", lambda *a: {"synthetic": True})
    result = R.evaluate(tmp_path)
    assert calls == list(R.SYMBOLS)
    assert result["goal_achieved"] is False and result["historical_strategy_candidate"] is False
    assert result["actual_money_profit"] == result["prospective_paper"] == "NOT TESTED"
    for symbol in R.SYMBOLS:
        for mode in R.MODES:
            reports = result["symbols"][symbol]["models"][mode]["reports"]
            assert all(not reports[f]["historical_gate"] for f in R.FAMILIES)
            assert reports["CLOCK"]["metrics"]["observed_day_clusters"] == 4


@pytest.mark.parametrize("value", ("2026-04-11T00:00:00", "2026-04-11T03:00:00+03:00", "2026-04-10T19:00:00-05:00"))
def test_boundaries_require_explicit_zero_utc_offset(value):
    with pytest.raises(ValueError, match="UTC"):
        R.stamp(value)
    assert R.stamp("2026-04-11T00:00:00+00:00") == R.START


def full_selection_fixture(tmp_path, monkeypatch):
    declaration = freeze_fixture(tmp_path, monkeypatch)
    models = {}
    for mode in R.MODES:
        for family in R.FAMILIES:
            model = model_stub(family, "NOT TESTED")
            model.update(estimator=None, serialized_estimator_sha256=None, training_completed_labels=12,
                         training_start=R.START.isoformat(), training_end=declaration["lineage"]["boundaries"]["BOOM600"]["70"],
                         common_timestamp_target_sha256="shared_targets", training_clock_issuance_sha256="shared_clock")
            models[mode + "_" + family] = model
    selection = {"stage": "frozen_tick_age_payoff_development", "declaration_sha256": R.digest(tmp_path / "declaration.json"),
                 "safety": deepcopy(FLAGS), "science_code_hashes": declaration["science_code_hashes"], "lineage": declaration["lineage"],
                 "config": R.CONFIG, "final30_features_or_payoffs_evaluated": False,
                 "goal_achieved": False, "historical_strategy_candidate": False,
                 "symbols": {s: {"status": "TESTED", "selected_candidate": None, "final_models": deepcopy(models),
                     "candidates": {key: {"development_eligible": False} for key in models}} for s in R.SYMBOLS}}
    return selection


def save_selection(tmp_path, selection):
    R.save(tmp_path / "selection.json", selection)
    (tmp_path / "selection.sha256").write_text(R.digest(tmp_path / "selection.json"))


@pytest.mark.parametrize("kind", ("inputs", "signals", "training_labels", "diagnostic_labels", "ledgers"))
def test_every_local_frozen_evidence_byte_layer_is_verified_before_final_release(kind, tmp_path, monkeypatch):
    selection = full_selection_fixture(tmp_path, monkeypatch)
    artifact = tmp_path / (kind + ".csv")
    artifact.write_text("signal_time,value\n2026-04-11T00:00Z,1\n")
    monkeypatch.setattr(R.round8, "path_for", lambda value: tmp_path / value)
    selection["symbols"]["BOOM600"]["saved_" + kind] = {"path": artifact.name, "sha256": R.digest(artifact),
                                                           "local_ignored_artifact": True}
    save_selection(tmp_path, selection)
    # Verify the positive path first, then corrupt the same pinned local layer.
    R.frozen_selection(tmp_path)
    artifact.write_text(artifact.read_text() + "changed bytes\n")
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: pytest.fail("No held-out feature/label read after artifact drift"))
    with pytest.raises(ValueError, match="local inputs/signals/labels/ledger bytes"):
        R.evaluate(tmp_path)


@pytest.mark.parametrize("tamper", ("missing_family", "wrong_status", "wrong_features", "wrong_targets",
                                     "wrong_clock", "changed_cutoff_time", "eligible", "selected_candidate"))
def test_frozen_final_model_pairs_cannot_change_schema_common_targets_or_selection(tamper, tmp_path, monkeypatch):
    selection = full_selection_fixture(tmp_path, monkeypatch)
    symbol = selection["symbols"]["BOOM600"]
    model = symbol["final_models"]["SPIKE_RIDGE46"]
    if tamper == "missing_family":
        symbol["final_models"].pop("DRIFT_RIDGE44")
    elif tamper == "wrong_status":
        model["status"] = "REFIT"
    elif tamper == "wrong_features":
        model["feature_names"] = N46[::-1]
    elif tamper == "wrong_targets":
        model["common_timestamp_target_sha256"] = "different_completed_labels"
    elif tamper == "wrong_clock":
        model["training_clock_issuance_sha256"] = "different_rows"
    elif tamper == "changed_cutoff_time":
        model["training_end"] = R.END.isoformat()
    elif tamper == "eligible":
        symbol["candidates"]["SPIKE_RIDGE44"]["development_eligible"] = True
    else:
        symbol["selected_candidate"] = "SPIKE_RIDGE46"
    save_selection(tmp_path, selection)
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: pytest.fail("No final features before model-pair validation"))
    with pytest.raises(ValueError):
        R.evaluate(tmp_path)


def test_validated_estimator_requires_1000_and_exact_frozen_scaler_cutoff_checksum():
    rows, labels, start, end = training_fixture()
    model = R.training_pair(rows, labels, N44, N46, start, end)["RIDGE44"]
    R.validate_model(model)
    changed = deepcopy(model)
    changed["estimator"]["threshold"] += .1
    with pytest.raises(ValueError, match="Saved model"):
        R.validate_model(changed)
    changed = deepcopy(model)
    changed["training_completed_labels"] = 999
    with pytest.raises(ValueError, match="Saved model"):
        R.validate_model(changed)


def test_local_input_file_keeps_invalid_opportunities_and_exact46_order(tmp_path, monkeypatch):
    context = empty_context()
    context["inputs"] = pd.DataFrame({"m5_open_time": pd.to_datetime(["2026-04-11T00:00Z", "2026-04-11T00:05Z"]),
                                       "signal_time": pd.to_datetime(["2026-04-11T00:05Z", "2026-04-11T00:10Z"]),
                                       **{name: [1., np.nan] for name in N46},
                                       "original_feature_valid": [True, False], "original44_all_finite": [True, False],
                                       "common_available": [True, False], "feature_valid": [True, False]})
    monkeypatch.setattr(R.round8, "relative", lambda path: str(Path(path).relative_to(tmp_path)))
    artifact = R.persist_context(context, tmp_path, "BOOM600", "prefix40")
    stored = pd.read_csv(tmp_path / artifact["path"])
    assert len(stored) == 2 and stored.feature_valid.tolist() == [True, False]
    assert stored.columns[2:48].tolist() == N46
    assert artifact["sha256"] == R.digest(tmp_path / artifact["path"])
    assert context["audit"]["inputs"] == artifact


def test_partial_fit_pooled_comparisons_are_not_tested_against_all_fold_clock(tmp_path, monkeypatch):
    declaration = freeze_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "SYMBOLS", ("BOOM600",))
    monkeypatch.setattr(R, "frozen_declaration", lambda *a: declaration)
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: empty_context())
    monkeypatch.setattr(R, "dump_frame", lambda *a: {"synthetic": True})
    saved = R.develop(tmp_path)["symbols"]["BOOM600"]
    for mode in R.MODES:
        validation = saved["validation"][mode]
        assert all(validation[f]["status"] == "NOT TESTED" for f in R.FAMILIES)
        assert all(i["status"] == "NOT TESTED" for i in validation["comparisons"].values())
        assert all("unmatched_fit_availability" in i["reason"] for i in validation["comparisons"].values())
