"""Synthetic-only round10 lineage,45-dimension and known-payoff chronology tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tick_path_study_tests", ROOT / "scripts/run_spike_tick_path_risk.py")
R = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(R)
N44, N45 = list(R.NAMES44), list(R.NAMES45)
FLAGS = dict.fromkeys(R.round9.round8.FLAGS, False)


def model(family="RIDGE45", status="TESTED"):
    names = N44 if family == "RIDGE44" else N45
    est = {"version": 1, "feature_names": names, "means": [0.] * len(names), "std": [1.] * len(names),
           "coefs": [0.] * len(names), "intercept": .3, "threshold": .25, "penalty": .1, "quantile": .75,
           "fit_rows": 1000, "score_kind": "continuous_uncalibrated_score", "scaler_fitted_on": "training_rows_only",
           "target_clipped": False, "features_clipped": False,
           "objective": "mean_squared_error_plus_penalty_times_squared_coefficients"} if status == "TESTED" else None
    return {"family": family, "status": status, "feature_names": names, "training_completed_labels": 1000 if est else 0,
            "estimator": est, "serialized_estimator_sha256": R.json_hash(est) if est else None}


def fixture(n=1000):
    stamps = []
    for day in R.DATES[:4]:
        stamps.extend(pd.date_range(pd.Timestamp(day, tz="UTC"), periods=250, freq="5min"))
    index = pd.DatetimeIndex(stamps[:n]).as_unit("ns")
    q = np.arange(n, dtype=float)
    rows = pd.DataFrame(np.column_stack([np.sin(q / (j + 2)) for j in range(45)]), index=index, columns=N45)
    rows[R.PATH_FEATURE_NAME] = rows[R.PATH_FEATURE_NAME].abs()
    rows["atr"], rows["close"] = 1., 100.
    labels = pd.DataFrame({"signal_time": index, "net_R": .2 + rows[N44[0]].to_numpy() * .1, "censored": False})
    return rows, labels, R.START, pd.Timestamp(R.DATES[3], tz="UTC") + pd.Timedelta(days=1)


def empty_context():
    rows = pd.DataFrame(columns=[*N45, "atr", "close"], index=pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
    ticks = pd.DataFrame({"quote": []}, index=pd.DatetimeIndex([], tz="UTC").as_unit("ns"))
    return {"rows": rows, "ticks": ticks, "inputs": rows.reset_index(), "names44": N44, "names45": N45, "audit": {}}


def declaration():
    bounds = {"0": R.START.isoformat(), "40": "2026-06-13T19:11:56+00:00", "50": "2026-06-29T23:59:58+00:00",
              "60": "2026-07-31T04:47:57+00:00", "70": "2026-08-16T09:36:02+00:00", "100": R.END.isoformat()}
    return {"stage": "tick_path_risk_premeasurement_declaration", "safety": deepcopy(FLAGS), "config": R.CONFIG,
        "history": R.HISTORY, "dates": list(R.DATES), "symbols": list(R.SYMBOLS), "families": list(R.FAMILIES),
        "modes": list(R.MODES), "feature_names45": N45, "new_feature_formulas": R.FEATURE_FORMULAS,
        "science_code_hashes": {"synthetic": "sha"}, "lineage": {"boundaries": {s: deepcopy(bounds) for s in R.SYMBOLS},
        "detectors": {s: {"adequate": True} for s in R.SYMBOLS}}, "quotes_parsed": False,
        "new_risk_features_models_or_ledgers_computed": False, "goal_achieved": False,
        "historical_strategy_candidate": False, "live_candidate": False}


def freeze(tmp_path, monkeypatch):
    doc = declaration(); R.save(tmp_path / "declaration.json", doc)
    (tmp_path / "declaration.sha256").write_text(R.digest(tmp_path / "declaration.json"))
    monkeypatch.setattr(R, "hashes", lambda: doc["science_code_hashes"])
    monkeypatch.setattr(R, "verified_round9_metadata", lambda: doc["lineage"])
    return doc


def synthetic_artifacts(monkeypatch):
    monkeypatch.setattr(R, "dump_frame", lambda *a: {"path": "synthetic", "sha256": "synthetic", "rows": len(a[-1])})
    def persist(context, output, symbol, prefix):
        value = {"path": "synthetic_inputs", "sha256": "synthetic", "rows": len(context["inputs"])}
        context["audit"]["inputs"] = value
        return value
    monkeypatch.setattr(R, "persist_context", persist)


@pytest.mark.parametrize("flag", FLAGS)
def test_all_four_false_flags_block_each_stage_before_any_metadata_or_price_read(flag, tmp_path, monkeypatch):
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(R, "verified_round9_metadata", lambda: pytest.fail("Unsafe metadata read"))
    for action in (R.declare, R.develop, R.evaluate):
        with pytest.raises(ValueError, match="flags"):
            action(tmp_path)


@pytest.mark.parametrize("stage,name", [("declare", "declaration.sha256"), ("declare", "selection.json"),
                                         ("develop", "selection.sha256"), ("develop", "results.json"),
                                         ("evaluate", "metrics.csv"), ("evaluate", "results.sha256")])
def test_stage_overwrite_refusal_precedes_all_calculation(stage, name, tmp_path, monkeypatch):
    (tmp_path / name).write_text("frozen")
    monkeypatch.setattr(R, "verified_round9_metadata", lambda: pytest.fail("No source access after overwrite"))
    with pytest.raises(ValueError, match="overwrite"):
        {"declare": R.declare, "develop": R.develop, "evaluate": R.evaluate}[stage](tmp_path)


def test_declaration_is_only_metadata_and_does_not_turn_known_payoffs_into_fresh_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "hashes", lambda: {"synthetic": "sha"})
    monkeypatch.setattr(R, "verified_round9_metadata", lambda: {"synthetic": "lineage"})
    for name in ("load_m1", "load_tick_day", "describe_tick_path_risk", "causal_multiframe_inputs", "fit_ridge", "replay_partition"):
        monkeypatch.setattr(R, name, lambda *a, **k: pytest.fail("Declaration must not calculate"))
    doc = R.declare(tmp_path)
    assert doc["history"]["known_prices"] and doc["history"]["known_payoffs"]
    assert not doc["history"]["price_oos"] and not doc["history"]["new_label_holdout"]
    assert doc["quotes_parsed"] is False and doc["new_risk_features_models_or_ledgers_computed"] is False
    assert doc["feature_names45"] == N45 and doc["user_target"]["pilot_can_meet_target"] is False


def metadata_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    study = tmp_path / "docs/spike_tick_age_payoff_20261005"; study.mkdir(parents=True)
    monkeypatch.setattr(R, "ROUND9", study)
    monkeypatch.setattr(R, "path_for", lambda label: tmp_path / label)
    lineage = {"files": {}, "boundaries": {}, "tick_sources": {}, "m1_sources": {}, "detectors": {}}
    for name in ("declaration.json", "declaration.sha256", "selection.json", "selection.sha256", "development_metrics.csv", "metrics.csv"):
        (study / name).write_text("synthetic_" + name)
    select = {"lineage": lineage, "science_code_hashes": {}}
    monkeypatch.setattr(R, "frozen_round9_selection", lambda *a: (select, R.digest(study / "selection.json")))
    result = {"stage": "fixed_tick_age_payoff_final30_diagnostics", "selection_sha256": R.digest(study / "selection.json"),
              "declaration_sha256": R.digest(study / "declaration.json"), "lineage": lineage,
              "config": R.round9.CONFIG, "safety": FLAGS, "goal_achieved": False,
              "historical_strategy_candidate": False, "live_candidate": False, "price_oos": False}
    R.save(study / "results.json", result); (study / "results.sha256").write_text(R.digest(study / "results.json"))
    verifier = tmp_path / "scripts/independent9.py"; verifier.parent.mkdir(); verifier.write_text("synthetic verifier")
    oldroot = "/old-host/boom-crash-quant-lab"
    files = {oldroot + "/" + str(path.relative_to(tmp_path)): R.digest(path)
             for path in (study / "declaration.json", study / "selection.json", study / "results.json")}
    audit = {"stage": "independent_tick_age_payoff_audit", "passed": True, "errors": [], "safety": FLAGS,
             "input_sha256": files, "verifier_file": "scripts/independent9.py", "verifier_sha256": R.digest(verifier)}
    R.save(study / "independent_audit.json", audit)
    (study / "independent_audit.sha256").write_text(R.digest(study / "independent_audit.json"))
    return study, audit


@pytest.mark.parametrize("tamper", ("missing_audit", "failed_audit", "audit_hash", "verifier", "result", "audited_input", "outside_path", "ambiguous_anchor"))
def test_pass9_audit_and_every_immutable_byte_layer_are_required_before_prices(tamper, tmp_path, monkeypatch):
    study, audit = metadata_fixture(tmp_path, monkeypatch)
    if tamper == "missing_audit":
        (study / "independent_audit.json").unlink()
    elif tamper == "failed_audit":
        audit["passed"] = False; audit["errors"] = ["unresolved"]
    elif tamper == "audit_hash":
        (study / "independent_audit.sha256").write_text("wrong")
    elif tamper == "verifier":
        (tmp_path / "scripts/independent9.py").write_text("changed")
    elif tamper == "result":
        (study / "results.json").write_text("{}")
    elif tamper == "audited_input":
        (study / "selection.json").write_text("changed")
    elif tamper == "outside_path":
        audit["input_sha256"]["/outside/secret.json"] = "wrong"
    else:
        key = next(k for k in audit["input_sha256"] if k.endswith("/declaration.json"))
        audit["input_sha256"]["/second-host" + key] = audit["input_sha256"][key]
    if tamper in ("failed_audit", "outside_path", "ambiguous_anchor"):
        (study / "independent_audit.json").write_text(json.dumps(audit))
        (study / "independent_audit.sha256").write_text(R.digest(study / "independent_audit.json"))
    monkeypatch.setattr(R, "load_tick_day", lambda *a: pytest.fail("No actual quote decode during lineage gate"))
    with pytest.raises(ValueError):
        R.verified_round9_metadata()


def test_audited_old_host_prefix_is_unique_hash_anchored_and_saved_as_relative_pins(tmp_path, monkeypatch):
    _, audit = metadata_fixture(tmp_path, monkeypatch)
    result = R.verified_round9_metadata()
    assert result["round9_audit_sha256"]
    assert all(not Path(label).is_absolute() for label in result["files"])
    assert "docs/spike_tick_age_payoff_20261005/results.json" in result["files"]
    prefix = R.audited_repository_prefix(audit)
    assert prefix == "/old-host/boom-crash-quant-lab"
    with pytest.raises(ValueError):
        R.audited_relative_path(prefix + "/docs/../outside", prefix)
    with pytest.raises(ValueError):
        R.audited_relative_path("/other-root/docs/source.csv", prefix)


@pytest.mark.parametrize("kind", ("science", "lineage", "history", "dimension"))
def test_declaration_drift_prevents_risk_calculation(kind, tmp_path, monkeypatch):
    doc = freeze(tmp_path, monkeypatch)
    if kind == "science":
        monkeypatch.setattr(R, "hashes", lambda: {"changed": "sha"})
    elif kind == "lineage":
        monkeypatch.setattr(R, "verified_round9_metadata", lambda: {"changed": True})
    else:
        doc = deepcopy(doc)
        if kind == "history":
            doc["history"]["new_label_holdout"] = True
        else:
            doc["feature_names45"].append("tail_age_log1p")
        (tmp_path / "declaration.json").write_text(json.dumps(doc))
        (tmp_path / "declaration.sha256").write_text(R.digest(tmp_path / "declaration.json"))
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: pytest.fail("No changed risk calculation"))
    with pytest.raises(ValueError):
        R.develop(tmp_path)


@pytest.mark.parametrize("symbol,side", (("BOOM600", 1), ("CRASH600", -1)))
def test_prepare_clips_closed_m1_and_tick_prefix_before_risk_without_any_age_mask(symbol, side, monkeypatch):
    end = pd.Timestamp("2026-04-11T00:17:30Z")
    m1 = pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
        index=pd.date_range("2026-04-10T23:50Z", "2026-04-11T00:20Z", freq="min").as_unit("ns"))
    m1.loc[pd.Timestamp("2026-04-11T00:05Z")] = np.nan
    tick = pd.DataFrame({"quote": 100.}, index=pd.date_range("2026-04-11T00:00Z", periods=1500, freq="s").as_unit("ns"))
    monkeypatch.setattr(R, "load_m1", lambda *a: (m1, {})); monkeypatch.setattr(R, "check_prepared", lambda *a: None)
    monkeypatch.setattr(R, "DATES", ("2026-04-11", "2026-04-12"))
    monkeypatch.setattr(R, "load_tick_day", lambda value: tick if value["date"] == "2026-04-11" else pytest.fail("No future day read"))
    original = R.describe_tick_path_risk
    def describe(frame, native_side, frozen_scale):
        assert frame.index.max() < end and len(frame) == 1050
        assert native_side == side and frozen_scale == .001
        return original(frame, native_side, frozen_scale)
    monkeypatch.setattr(R, "describe_tick_path_risk", describe)
    def features(frame, direction):
        assert direction == ("boom" if side == 1 else "crash")
        assert (frame.index + pd.Timedelta(minutes=1) <= end).all()
        assert frame.close.isna().sum() == 1
        table = pd.DataFrame(1., index=pd.date_range("2026-04-11T00:00Z", periods=4, freq="5min").as_unit("ns"), columns=N44)
        table["feature_valid"] = [True, False, True, False]; table["atr"], table["close"] = 1., 100.
        return table, N44
    monkeypatch.setattr(R, "causal_multiframe_inputs", features)
    d = {"lineage": {"m1_sources": {symbol: {"path": "synthetic.csv"}}, "tick_sources": {symbol: {"2026-04-11": {"date": "2026-04-11"}}},
                     "detectors": {symbol: {"adequate": True, "median_abs_log_return": .001}}}}
    context = R.prepare_prefix(d, symbol, end)
    assert context["rows"].index.tolist() == [pd.Timestamp("2026-04-11T00:15Z")]
    assert len(context["inputs"]) == 3 and context["inputs"].feature_valid.tolist() == [False, False, True]
    assert "tail_age_log1p" not in context["inputs"] and "prior_tail_mark" not in context["inputs"]
    assert context["audit"]["m1_unknown_closed_rows_retained"] == 1
    assert context["audit"]["m1_available_closed_rows_used"] + 1 == context["audit"]["m1_closed_rows_used"]
    assert context["audit"]["age_or_mark_availability_used"] is False


def test_training_matrices44_45_share_exact_rows_targets_and_frozen_train_only_scalers(tmp_path, monkeypatch):
    rows, labels, start, end = fixture()
    saved = []
    def dump(output, category, prefix, frame):
        assert category == "matrices"; saved.append(frame.copy())
        return {"path": "synthetic_matrix", "sha256": R.json_hash(frame.signal_time.to_list()), "rows": len(frame)}
    monkeypatch.setattr(R, "dump_frame", dump)
    pair = R.training_pair(rows, labels.sample(frac=1, random_state=2), N44, N45, start, end, tmp_path, "train")
    assert saved[0].columns.tolist() == ["signal_time", *N45, "net_R"] and len(saved[0]) == 1000
    assert pair["RIDGE44"]["training_matrix"] == pair["RIDGE45"]["training_matrix"]
    assert pair["RIDGE44"]["common_timestamp_target_sha256"] == pair["RIDGE45"]["common_timestamp_target_sha256"]
    for family, names in (("RIDGE44", N44), ("RIDGE45", N45)):
        assert pair[family]["status"] == "TESTED"
        assert pair[family]["estimator"]["means"] == pytest.approx(rows[names].mean().to_numpy())
    outside = rows.iloc[:1].copy(); outside.index = pd.DatetimeIndex([end + pd.Timedelta(days=3)])
    outside[N45] = 1e12
    extra = pd.DataFrame({"signal_time": outside.index, "net_R": [1e12], "censored": [False]})
    same = R.training_pair(pd.concat([rows, outside]), pd.concat([labels, extra]), N44, N45, start, end, tmp_path, "again")
    assert same == pair


@pytest.mark.parametrize("fault", ("999", "unknown", "censored", "planned_purge", "unsampled_day", "schema46", "alignment"))
def test_training_floor_and_shared45_schema_never_relax_to_rescue_an_unavailable_fit(fault, tmp_path, monkeypatch):
    rows, labels, start, end = fixture(); names = N45
    synthetic_artifacts(monkeypatch)
    if fault == "999": labels = labels.iloc[:-1]
    elif fault == "unknown": labels.loc[0, "net_R"] = np.nan
    elif fault == "censored": labels.loc[0, "censored"] = True
    elif fault in ("planned_purge", "unsampled_day"):
        time = end - pd.Timedelta(minutes=30) if fault == "planned_purge" else start + pd.Timedelta(days=1)
        rows.loc[time] = rows.iloc[0]; labels.loc[0, "signal_time"] = time
    elif fault == "schema46": names = [*N45, "prior_tail_mark"]
    else: rows = rows.iloc[1:]
    monkeypatch.setattr(R, "fit_ridge", lambda *a, **k: pytest.fail("No smaller/unavailable fit"))
    if fault in ("schema46", "alignment"):
        with pytest.raises(ValueError): R.training_pair(rows, labels, N44, names, start, end, tmp_path, "train")
    else:
        result = R.training_pair(rows, labels, N44, names, start, end, tmp_path, "train")
        assert all(m["status"] == "NOT TESTED" and m["estimator"] is None for m in result.values())


def test_signal_cutoff_and_native_risk_values_are_identical_for_both_strategy_directions(monkeypatch):
    rows, _, _, _ = fixture(3)
    monkeypatch.setattr(R, "predict_ridge", lambda *a, **k: np.array([-.1, 0., .3]))
    frozen = model()
    for mode, side in (("SPIKE", -1), ("DRIFT", 1)):
        issued = R.issue(rows, "CRASH600", mode, frozen)
        assert len(issued) == 1 and issued[0]["side"] == side
        assert issued[0][R.PATH_FEATURE_NAME] == rows[R.PATH_FEATURE_NAME].iloc[-1]
        assert "prior_age_seconds" not in issued[0]
    assert frozen["estimator"]["threshold"] == .25
    assert R.issue(rows, "BOOM600", "SPIKE", model(status="NOT TESTED")) == []


def test_shared_risk_clock_uses_one_open_strategy_and_separate_overlapping_diagnostic(tmp_path, monkeypatch):
    synthetic_artifacts(monkeypatch)
    rows, _, start, _ = fixture(4)
    tick = pd.DataFrame({"quote": 100.}, index=pd.date_range(start, periods=90 * 60, freq="s").as_unit("ns"))
    context = {"rows": rows, "ticks": tick}
    pair = {family: model(family) for family in R.FAMILIES}
    reports, comparisons, ledgers = R.evaluate_partition(context, pair, "BOOM600", "SPIKE", start,
        start + pd.Timedelta(minutes=90), tmp_path, "test")
    assert all(report["common_available_rows"] == 4 for report in reports.values())
    assert all(len(ledger) == 1 for ledger in ledgers.values())
    assert comparisons["RIDGE45_minus_RIDGE44"]["mean_difference_R"] == 0.
    diagnostics = R.opportunity_diagnostics(context, "BOOM600", "SPIKE", start, start + pd.Timedelta(minutes=90), tmp_path, "test")
    assert diagnostics["metrics"]["completed"] == 4 and diagnostics["strategy_pf_evidence"] is False
    assert "profit_factor" not in diagnostics["metrics"] and "pf_inference" not in diagnostics["metrics"]


def test_partial_fits_do_not_compare_partial_model_sample_with_all_fold_clock():
    ledger = pd.DataFrame({"signal_time": pd.to_datetime(["2026-04-11T00:00Z"], utc=True), "net_R": [-1.], "censored": [False]})
    ledgers = {family: ledger for family in ("CLOCK", *R.FAMILIES)}
    reports = {family: {"status": "TESTED"} for family in ledgers}; reports["RIDGE45"]["status"] = "PARTIALLY TESTED"
    result = R.comparison_reports(ledgers, reports, ["2026-04-11"], np.ones((3, 1), dtype=int))
    assert result["RIDGE45_minus_RIDGE44"]["status"] == result["RIDGE45_minus_CLOCK"]["status"] == "NOT TESTED"
    assert result["RIDGE44_minus_CLOCK"]["status"] == "TESTED"


def test_development_builds_only40_50_60_70_and_freezes_all_fixed_models_with_known_history(tmp_path, monkeypatch):
    doc = freeze(tmp_path, monkeypatch); synthetic_artifacts(monkeypatch)
    monkeypatch.setattr(R, "SYMBOLS", ("BOOM600",)); monkeypatch.setattr(R, "frozen_declaration", lambda *a: doc)
    cuts = []
    def prepare(value, symbol, end):
        assert end <= pd.Timestamp(doc["lineage"]["boundaries"][symbol]["70"])
        cuts.append(end); return empty_context()
    monkeypatch.setattr(R, "prepare_prefix", prepare)
    output = R.develop(tmp_path)
    assert cuts == [pd.Timestamp(doc["lineage"]["boundaries"]["BOOM600"][k]) for k in ("40", "50", "60", "70")]
    saved = output["symbols"]["BOOM600"]
    assert len(saved["final_models"]) == 4 and all(m["status"] == "NOT TESTED" for m in saved["final_models"].values())
    assert output["final45_features_scores_or_ledgers_computed"] is False and output["history"]["known_payoffs"] is True
    assert saved["selected_candidate"] is None
    assert all(not c["development_eligible"] for c in saved["candidates"].values())
    assert all(c["validation"]["observed_day_clusters"] == 5 for c in saved["candidates"].values())
    for candidate in saved["candidates"].values():
        assert [(fold["training"], fold["test"]) for fold in candidate["walk_forward"]] == [("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3")]


@pytest.mark.parametrize("fault", ("selection_bytes", "matrix_bytes", "dimension46", "different_target", "eligible", "fresh_history"))
def test_final_release_validates_selection_matrices_pair_identity_and_adaptive_history_before_risk(fault, tmp_path, monkeypatch):
    doc = freeze(tmp_path, monkeypatch)
    models = {}
    for mode in R.MODES:
        for family in R.FAMILIES:
            m = model(family, "NOT TESTED")
            m.update(training_start=R.START.isoformat(), training_end=doc["lineage"]["boundaries"]["BOOM600"]["70"],
                common_timestamp_target_sha256="same", training_clock_issuance_sha256="same", training_matrix={"path": "matrix.csv", "sha256": "s"})
            models[mode + "_" + family] = m
    selection = {"stage": "frozen_tick_path_risk_development", "safety": deepcopy(FLAGS), "config": R.CONFIG, "history": deepcopy(R.HISTORY),
        "declaration_sha256": R.digest(tmp_path / "declaration.json"), "science_code_hashes": doc["science_code_hashes"], "lineage": doc["lineage"],
        "final45_features_scores_or_ledgers_computed": False, "goal_achieved": False, "historical_strategy_candidate": False, "live_candidate": False,
        "symbols": {s: {"status": "TESTED", "final_models": deepcopy(models), "selected_candidate": None,
                      "candidates": {key: {"development_eligible": False} for key in models}} for s in R.SYMBOLS}}
    target = selection["symbols"]["BOOM600"]["final_models"]["SPIKE_RIDGE45"]
    if fault == "matrix_bytes":
        file = tmp_path / "matrix.csv"; file.write_text("original")
        artifact = {"path": file.name, "sha256": R.digest(file), "local_ignored_artifact": True}
        target["training_matrix"] = artifact; file.write_text("changed")
        monkeypatch.setattr(R, "path_for", lambda label: tmp_path / label)
    elif fault == "dimension46": target["feature_names"].append("prior_tail_mark")
    elif fault == "different_target": target["common_timestamp_target_sha256"] = "different"
    elif fault == "eligible": selection["symbols"]["BOOM600"]["candidates"]["SPIKE_RIDGE45"]["development_eligible"] = True
    elif fault == "fresh_history": selection["history"]["new_label_holdout"] = True
    R.save(tmp_path / "selection.json", selection); (tmp_path / "selection.sha256").write_text(R.digest(tmp_path / "selection.json"))
    if fault == "selection_bytes": (tmp_path / "selection.json").write_text(json.dumps(selection))
    monkeypatch.setattr(R, "prepare_prefix", lambda *a: pytest.fail("No final risk calculation after freeze violation"))
    with pytest.raises(ValueError): R.evaluate(tmp_path)


def test_final_known_history_recalculation_does_not_fit_or_claim_a_new_label_holdout(tmp_path, monkeypatch):
    doc = declaration(); synthetic_artifacts(monkeypatch)
    pairs = {mode + "_" + family: model(family, "NOT TESTED") for mode in R.MODES for family in R.FAMILIES}
    selection = {"declaration_sha256": "synthetic", "science_code_hashes": {}, "lineage": doc["lineage"],
        "symbols": {s: {"status": "TESTED", "final_models": deepcopy(pairs),
                   "candidates": {key: {"development_eligible": False} for key in pairs}} for s in R.SYMBOLS}}
    monkeypatch.setattr(R, "frozen_selection", lambda *a: (selection, "frozen"))
    monkeypatch.setattr(R, "fit_ridge", lambda *a, **k: pytest.fail("No fit on final known history"))
    cuts = []
    def prepare(value, symbol, end):
        assert end == R.END; cuts.append(symbol); return empty_context()
    monkeypatch.setattr(R, "prepare_prefix", prepare)
    result = R.evaluate(tmp_path)
    assert cuts == list(R.SYMBOLS) and result["history"]["known_payoffs"] is True
    assert result["history"]["new_label_holdout"] is False and result["goal_achieved"] is False
    assert result["actual_money_profit"] == result["prospective_paper"] == "NOT TESTED"
    for symbol in R.SYMBOLS:
        for mode in R.MODES:
            reports = result["symbols"][symbol]["models"][mode]["reports"]
            assert reports["CLOCK"]["metrics"]["observed_day_clusters"] == 4
            assert all(reports[f]["historical_gate"] is False for f in R.FAMILIES)


@pytest.mark.parametrize("fault", ("negative_absolute", "negative_increment", "undefined", "missing", "partial", "reference_unknown", "none"))
def test_expansion_requires_positive_absolute_and_incremental_bounds_but_never_authorizes_expansion(fault):
    reports = {"RIDGE45": {"status": "TESTED", "metrics": {"day_mean_ci95": [.1, .4], "mean_inference": {"all_draws_defined": True},
                       "censored": 0, "missing_entry": 0, "invalid_uncensored": 0}}}
    reports.update({family: deepcopy(reports["RIDGE45"]) for family in ("RIDGE44", "CLOCK")})
    comparisons = {key: {"status": "TESTED", "all_draws_defined": True, "ci95": [.01, .2]}
                   for key in ("RIDGE45_minus_RIDGE44", "RIDGE45_minus_CLOCK")}
    if fault == "negative_absolute": reports["RIDGE45"]["metrics"]["day_mean_ci95"] = [-.3, -.1]
    elif fault == "negative_increment": comparisons["RIDGE45_minus_CLOCK"]["ci95"] = [-.1, .3]
    elif fault == "undefined": comparisons["RIDGE45_minus_RIDGE44"]["all_draws_defined"] = False
    elif fault == "missing": reports["RIDGE45"]["metrics"]["missing_entry"] = 1
    elif fault == "partial": comparisons["RIDGE45_minus_RIDGE44"]["status"] = "NOT TESTED"
    elif fault == "reference_unknown": reports["CLOCK"]["metrics"]["censored"] = 1
    result = R.expansion_diagnostic(reports, comparisons)
    assert result["positive_absolute_and_incremental_day_bounds"] is (fault == "none")
    assert result["expansion_authorized"] is False and result["historical_gate"] is False and result["live_candidate"] is False


@pytest.mark.parametrize("field,value", (("coefs", [0.] * 44), ("std", [0.] * 45), ("means", [float("inf")] * 45)))
def test45_dimension_and_finite_scaler_guards_run_before_scalar_prediction(field, value):
    saved = model()
    saved["estimator"][field] = value
    if field != "means":
        saved["serialized_estimator_sha256"] = R.json_hash(saved["estimator"])
    with pytest.raises(ValueError, match="dimensions or finite"):
        R.validate_model(saved)


@pytest.mark.parametrize("field,value", (("version", 2), ("version", True), ("scaler_fitted_on", "all_rows"),
                                         ("target_clipped", True), ("features_clipped", True), ("objective", "sum_SSE")))
def test_rehashed_estimator_cannot_change_declared_training_only_unclipped_ridge_semantics(field, value):
    saved = model()
    saved["estimator"][field] = value
    saved["serialized_estimator_sha256"] = R.json_hash(saved["estimator"])
    # Checksum agreement does not make an incompatible estimator acceptable.
    assert saved["serialized_estimator_sha256"] == R.json_hash(saved["estimator"])
    with pytest.raises(ValueError, match="semantic fields"):
        R.validate_model(saved)
