"""Synthetic-only target-horizon ablation, causal prefixes and freeze guards."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("short_target_runner_tests", ROOT / "scripts/run_spike_short_target.py")
R = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(R)
FLAGS = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)
NAMES = list(R.FEATURE_NAMES)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "ROUND6", tmp_path / "docs/ancestor")
    return tmp_path / "docs/synthetic11"


def synthetic_training(n=1000):
    index = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC").as_unit("ns")
    q = np.arange(n, dtype=float)
    X = pd.DataFrame(np.column_stack([np.sin(q / (j+2)) for j in range(44)]), index=index, columns=NAMES)
    X["atr"], X["close"] = 1., 100.
    labels = {}
    for family, minutes in R.TARGET_MINUTES.items():
        labels[family] = pd.DataFrame({"signal_time": index, "entry_time": index+pd.Timedelta(minutes=1),
            "planned_end": index+pd.Timedelta(minutes=minutes+1), "censored": False,
            "net_R": .1 + (1 if minutes == 1 else -1)*X[NAMES[0]].to_numpy()*.2})
    return X, labels, index[0], index[-1]+pd.Timedelta(minutes=31)


@pytest.fixture
def fitted(workspace):
    rows, targets, start, end = synthetic_training()
    models, _ = R.fit_pair(rows, targets, start, end, workspace.with_name("synthetic_fit_fixtures"), "synthetic")
    return models


def minimal_model(family="SHORT44", n=1000):
    rows, targets, start, end = synthetic_training(n)
    est = R.fit_ridge(rows[NAMES], targets[family].net_R.to_numpy(), NAMES) if n >= R.MIN_TRAIN else None
    return {"family": family, "status": "TESTED" if est else "NOT TESTED",
        "target_name": R.CONFIG["target_names"][family],
        "target_horizon_minutes": R.TARGET_MINUTES[family], "execution_horizon_minutes": 1,
        "feature_names": NAMES, "training_start": start, "training_end": end,
        "training_completed_labels": n, "label_purge_minutes": 31,
        "training_identity": {"common_training_rows": n, "training_latest_issue": rows.index[-1],
                              "training_latest_long_planned_end": targets["LONG44"].planned_end.max()},
        "estimator": est, "serialized_estimator_sha256": R.json_hash(est) if est else None,
        "shared_scaler_sha256": R.json_hash({"means": est["means"], "std": est["std"]}) if est else None}


def refresh(model):
    est = model["estimator"]
    model["serialized_estimator_sha256"] = R.json_hash(est)
    model["shared_scaler_sha256"] = R.json_hash({"means": est["means"], "std": est["std"]})


def synthetic_lineage():
    return {"sources": {}, "files": {}, "synthetic": True}


def freeze_declaration(workspace, monkeypatch):
    monkeypatch.setattr(R, "hashes", lambda: {"synthetic_source": "0"*64})
    monkeypatch.setattr(R, "metadata_lineage", synthetic_lineage)
    return R.declare(workspace)


@pytest.mark.parametrize("flag", FLAGS)
def test_unsafe_flag_refuses_all_stages_before_reading_any_source(flag, workspace, monkeypatch):
    R.assert_offline()  # Load safe application config before exercising env refusal.
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(R, "metadata_lineage", lambda: pytest.fail("Unsafe metadata read"))
    for action in (R.declare, R.develop, R.evaluate):
        with pytest.raises(RuntimeError, match="flags"):
            action(workspace)


@pytest.mark.parametrize("stage,name", [("declare", "declaration.sha256"), ("declare", "selection.json"),
    ("develop", "selection.sha256"), ("develop", "development_state.json"),
    ("evaluate", "results.sha256"), ("evaluate", "evaluation_state.json"), ("evaluate", "metrics.csv")])
def test_state_handle_never_restarts_or_overwrites_after_observation_timeout(stage, name, workspace, monkeypatch):
    workspace.mkdir(parents=True); (workspace/name).write_text("immutable or authoritative STARTED state")
    monkeypatch.setattr(R, "metadata_lineage", lambda: pytest.fail("Overwritten state source read"))
    with pytest.raises(ValueError, match="overwrite"):
        {"declare": R.declare, "develop": R.develop, "evaluate": R.evaluate}[stage](workspace)


def test_declaration_only_metadata_preserves_known_history(workspace, monkeypatch):
    monkeypatch.setattr(R, "hashes", lambda: {"synthetic": "a"*64})
    monkeypatch.setattr(R, "metadata_lineage", synthetic_lineage)
    for name in ("load_m1", "prepare", "prepared_frame", "fit_pair", "fit_ridge", "replay_timed", "causal_multiframe_inputs"):
        monkeypatch.setattr(R, name, lambda *a, **k: pytest.fail("Declaration calculated historical science"))
    doc = R.declare(workspace)
    assert doc["history"]["known_prices"] and doc["history"]["known_fifteen_minute_payoffs"]
    assert not doc["history"]["price_oos"] and not doc["history"]["new_label_holdout"]
    assert not doc["prices_decoded"] and not doc["new_features_labels_models_or_ledgers_computed"]
    assert doc["config"]["primary_holm_family"] == 4


def test_declaration_refuses_existing_computed_artifact_state_before_any_source_read(workspace, monkeypatch):
    path = workspace/"matrices"/"already_computed.csv"; path.parent.mkdir(parents=True); path.write_text("known prices computed early")
    monkeypatch.setattr(R, "metadata_lineage", lambda: pytest.fail("A declaration cannot bless earlier computation"))
    with pytest.raises(ValueError, match="predeclaration"):
        R.declare(workspace)


@pytest.mark.parametrize("key,value", [("price_oos", True), ("prospective", True), ("known_prices", False)])
def test_rehashed_declaration_history_semantic_tamper_is_refused(key, value, workspace, monkeypatch):
    doc = freeze_declaration(workspace, monkeypatch); doc["history"][key] = value
    path = workspace/"declaration.json"; path.write_text(json.dumps(doc))
    path.with_suffix(".sha256").write_text(R.digest(path))
    with pytest.raises(ValueError, match="semantics"):
        R.frozen_declaration(workspace)


@pytest.mark.parametrize("field", ["hold_bool", "purge_float", "history_int", "symbol_source_alias", "target_sample_float"])
def test_rehashed_declaration_numeric_boolean_type_aliases_are_refused(field, workspace, monkeypatch):
    doc = freeze_declaration(workspace, monkeypatch)
    if field == "hold_bool": doc["config"]["execution"]["max_hold_minutes"] = True
    elif field == "purge_float": doc["config"]["purge_minutes"] = 31.
    elif field == "history_int": doc["history"]["price_oos"] = 0
    elif field == "symbol_source_alias": doc["lineage"]["synthetic"] = 1
    else: doc["user_target"]["completed_per_symbol_model"] = 1000.
    path = workspace/"declaration.json"; path.write_text(json.dumps(doc))
    path.with_suffix(".sha256").write_text(R.digest(path))
    with pytest.raises(ValueError, match="semantics"):
        R.frozen_declaration(workspace)


@pytest.mark.parametrize("label", ["../outside", "/absolute/path", "docs/../state", "docs//state", "docs\\state", ""])
def test_repository_path_traversal_and_ambiguous_alias_are_refused(label, workspace):
    with pytest.raises(ValueError, match="path|paths|alias|relative"):
        R.path_for(label)


@pytest.mark.parametrize("directory", ["data", "scripts", "backend", "frontend", "docs/ancestor"])
def test_state_cannot_alias_sources_or_ancestor_science(directory, workspace):
    with pytest.raises(ValueError, match="alias"):
        R.output_path(R.ROOT/directory)


def test_only_reward_horizon_differs_between_training_configs():
    short, long = (asdict(R.TARGET_CONFIGS[family]) for family in R.FAMILIES)
    assert short.pop("max_hold_minutes") == 1 and long.pop("max_hold_minutes") == 15
    assert short == long == {"stop_atr": 2., "entry_delay_minutes": 1, "round_trip_cost_atr": .1, "fill_mode": "adverse_extreme"}
    assert R.CFG == R.TARGET_CONFIGS["SHORT44"] and R.PURGE == 31


def test_training_intersection_excludes_long_unknown_and_common_cutoff_crossing():
    rows, targets, start, end = synthetic_training(5)
    targets["LONG44"].loc[1, ["censored", "net_R"]] = [True, np.nan]
    X, pair = R.common_training(rows, targets, start, end-pd.Timedelta(minutes=1))
    assert X.index.tolist() == rows.index[[0, 2, 3]].tolist()
    assert pair.index.equals(X.index)
    assert pair.loc[rows.index[2], "short_net_R"] != pair.loc[rows.index[2], "long_net_R"]


@pytest.mark.parametrize("problem", ["duplicate", "missing_row", "nonfinite_feature", "crossing_long_end"])
def test_training_intersection_refuses_invalid_identity(problem):
    rows, targets, start, end = synthetic_training(5)
    if problem == "duplicate":
        targets["SHORT44"] = pd.concat([targets["SHORT44"], targets["SHORT44"].iloc[:1]], ignore_index=True)
    elif problem == "missing_row":
        rows = rows.iloc[1:]
    elif problem == "nonfinite_feature":
        rows.loc[rows.index[0], NAMES[0]] = np.nan
    else:
        targets["LONG44"].loc[0, "planned_end"] = end+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="Duplicate|exact|unknown|cutoff"):
        R.common_training(rows, targets, start, end)


def test_two_targets_have_identical_X_scaler_but_distinct_unclipped_y(fitted):
    short, long = fitted["SHORT44"], fitted["LONG44"]
    assert short["training_identity"] == long["training_identity"]
    assert short["shared_scaler_sha256"] == long["shared_scaler_sha256"]
    assert short["estimator"]["means"] == long["estimator"]["means"]
    assert short["estimator"]["std"] == long["estimator"]["std"]
    assert short["estimator"]["coefs"] != long["estimator"]["coefs"]
    assert short["training_completed_labels"] == long["training_completed_labels"] == 1000


def test_too_few_common_targets_returns_nonfitted_state_without_rescuing_sample(workspace):
    rows, targets, start, end = synthetic_training(999)
    models, identity = R.fit_pair(rows, targets, start, end, workspace, "insufficient")
    assert identity["common_training_rows"] == 999
    for model in models.values():
        assert model["status"] == "NOT TESTED" and model["estimator"] is None
        assert model["serialized_estimator_sha256"] is None
        assert R.issued(rows, "BOOM600", "SPIKE", model) == []


@pytest.mark.parametrize("key,value", [("version", True), ("version", 2), ("scaler_fitted_on", "all_rows"),
    ("objective", "sum_sse"), ("target_clipped", True), ("features_clipped", True),
    ("penalty", .01), ("quantile", .5), ("score_kind", "calibrated_probability"), ("fit_rows", True)])
def test_semantic_model_tamper_refused_even_after_hashes_are_recomputed(key, value):
    model = minimal_model(); model["estimator"][key] = value; refresh(model)
    with pytest.raises(ValueError, match="semantics"):
        R.validate_model(model)


@pytest.mark.parametrize("problem", ["wrong_dimension", "nonfinite", "zero_std", "negative_cutoff", "schema", "late_training"])
def test_model_geometry_and_temporal_tamper_not_rescued_by_integrity_hash(problem):
    model = minimal_model(); est = model["estimator"]
    if problem == "wrong_dimension":
        est["coefs"].pop()
    elif problem == "nonfinite":
        est["intercept"] = float("inf")
        with pytest.raises(ValueError):
            R.validate_model(model)
        return
    elif problem == "zero_std":
        est["std"][0] = 0.
    elif problem == "negative_cutoff":
        est["threshold"] = -.1
    elif problem == "schema":
        est["feature_names"] = est["feature_names"][::-1]
    else:
        model["training_end"] = model["training_identity"]["training_latest_issue"]
    refresh(model)
    with pytest.raises(ValueError):
        R.validate_model(model)


def test_signal_strict_positive_cutoff_and_native_mode_side():
    model = minimal_model(); est = model["estimator"]
    est.update(coefs=[0.]*44, intercept=0., threshold=0.); refresh(model)
    rows, _, _, _ = synthetic_training(2)
    assert R.issued(rows, "BOOM600", "SPIKE", model) == []
    est.update(intercept=.2, threshold=.2); refresh(model)
    assert len(R.issued(rows, "BOOM600", "SPIKE", model)) == 2
    for symbol, mode, side in [("BOOM600", "SPIKE", 1), ("BOOM600", "DRIFT", -1),
                              ("CRASH600", "SPIKE", -1), ("CRASH600", "DRIFT", 1)]:
        assert {s["side"] for s in R.issued(rows, symbol, mode, model)} == {side}


def test_dense_nonclock_opportunities_rejected_before_individual_label_claim():
    rows, _, _, _ = synthetic_training(2)
    rows.index = rows.index+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="clock"):
        R.issued(rows, "BOOM600", "SPIKE")


@pytest.fixture(scope="module")
def minute_history():
    n = 14*1440
    index = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    phase = np.arange(n)
    close = 1000+3*np.sin(phase/67)+.2*np.sin(phase/11)+phase/5000
    opening = np.r_[close[0]-.05, close[:-1]]
    frame = pd.DataFrame({"open": opening, "high": np.maximum(opening, close)+.15,
                          "low": np.minimum(opening, close)-.15, "close": close}, index=index)
    for i in range(150, n, 1440):
        frame.loc[index[i], "high"] += 12
    return frame


@pytest.mark.parametrize("symbol", R.SYMBOLS)
def test_future_suffix_mutation_cannot_change_fit_prefix_closed44_inputs(minute_history, symbol):
    cutoff = minute_history.index[10*1440]
    suffix = minute_history.copy(); suffix.loc[suffix.index >= cutoff, :] *= 50
    before = R.prepared_frame(minute_history, symbol, cutoff)
    after = R.prepared_frame(suffix, symbol, cutoff)
    pd.testing.assert_frame_equal(before[0], after[0])
    pd.testing.assert_frame_equal(before[1], after[1])
    assert not before[1].empty and before[2]["last_completed_m1_close"] <= cutoff


def test_missing_latest_frame_stays_invalid_instead_of_reusing_earlier_context(minute_history):
    cutoff = minute_history.index[10*1440]
    broken = minute_history.copy(); broken.loc[cutoff-pd.Timedelta(minutes=60), :] = np.nan
    _, intact, _ = R.prepared_frame(minute_history, "BOOM600", cutoff)
    _, rows, _ = R.prepared_frame(broken, "BOOM600", cutoff)
    assert cutoff in intact.index and cutoff not in rows.index


def minute_path(unknown=False):
    index = pd.date_range("2026-01-01", periods=90, freq="min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100.}, index=index)
    frame.loc[index[1], ["high", "close"]] = [102., 101.]
    frame.loc[index[2:16], ["open", "high", "low", "close"]] = [101., 101.1, 98., 99.]
    if unknown:
        frame.loc[index[2], :] = np.nan
    rows = pd.DataFrame({"atr": [1., 1.], "close": [100., 100.]}, index=index[[0, 30]])
    return frame, rows, index[0], index[-1]+pd.Timedelta(minutes=1)


def test_one_and_fifteen_minute_labels_use_same_delayed_open_cost_stop_and_purge():
    m1, rows, start, end = minute_path()
    targets, audits = R.target_pair(m1, rows, "BOOM600", "SPIKE", start, end)
    short, long = targets["SHORT44"], targets["LONG44"]
    assert short.entry.iloc[0] == long.entry.iloc[0] == 100.
    assert short.entry_time.iloc[0] == long.entry_time.iloc[0] == start+pd.Timedelta(minutes=1)
    assert short.net_R.iloc[0] == pytest.approx(.45) and long.net_R.iloc[0] == pytest.approx(-1.05)
    assert short.exit_time.iloc[0] == start+pd.Timedelta(minutes=2)
    assert long.exit_time.iloc[0] == start+pd.Timedelta(minutes=3)
    assert all(a["purge_minutes"] == 31 and a["overlap_skipped"] == 0 for a in audits.values())


def test_unknown_long_path_removed_from_both_models_common_training_without_zero_payoff():
    m1, rows, start, end = minute_path(unknown=True)
    targets, audits = R.target_pair(m1, rows, "BOOM600", "SPIKE", start, end)
    assert not targets["SHORT44"].censored.iloc[0]
    assert targets["LONG44"].censored.iloc[0] and pd.isna(targets["LONG44"].net_R.iloc[0])
    assert audits["LONG44"]["censored"] == 1
    matrix = rows.copy()
    for name in NAMES:
        matrix[name] = 1.
    X, pair = R.common_training(matrix, targets, start, end)
    assert X.index.tolist() == [rows.index[1]] and len(pair) == 1


def inference_fixtures(monkeypatch, undefined=False):
    day = {"bootstrap_mean_valid_replicates": R.BOOTSTRAP-(1 if undefined else 0),
           "bootstrap_difference_valid_replicates": R.BOOTSTRAP, "calendar_days": 30, "p": .001,
           "mean_net_R_ci95": [.1, .3], "baseline_difference_ci95": [.05, .2]}
    week = {"weekly_valid_mean_replicates": R.BOOTSTRAP, "weekly_valid_difference_replicates": R.BOOTSTRAP,
            "weekly_p": .002, "weekly_mean_net_R_ci95": [.1, .3], "weekly_difference_ci95": [.05, .2]}
    monkeypatch.setattr(R, "paired_inference", lambda *a, **k: deepcopy(day))
    monkeypatch.setattr(R, "weekly_inference", lambda *a, **k: deepcopy(week))


@pytest.mark.parametrize("unknown", ["model_censor", "reference_censor", "reference_entry", "undefined_draw", "none"])
def test_unknown_model_reference_or_undefined_mean_draw_forbids_positive_policy_inference(unknown, monkeypatch):
    inference_fixtures(monkeypatch, unknown == "undefined_draw")
    m = {"censored": int(unknown == "model_censor"), "invalid_uncensored": 0}
    b = {"censored": int(unknown == "reference_censor"), "invalid_uncensored": 0}
    a, ba = {"missing_entry": 0}, {"missing_entry": int(unknown == "reference_entry")}
    result = R.inference(None, None, None, None, m, a, b, ba)
    assert result["policy_inference_available"] is (unknown == "none")
    assert result["p"] == (.002 if unknown == "none" else 1.)


def eligible_candidate(family="SHORT44"):
    m = {"completed": 500, "active_days": 30, "censored": 0, "invalid_uncensored": 0,
         "selection_score": .1, "mean_net_R": .2, "profit_factor": 1.8}
    return {"family": family, "mode": "SPIKE", "validation": m,
            "walk_forward": [{"training": tr, "test": te, "metrics": {**m, "completed": 100},
                              "audit": {"missing_entry": 0}} for tr, te in R.WALK_FORWARD]}


def test_long_never_selected_even_if_its_development_metrics_are_better():
    short, long = eligible_candidate(), eligible_candidate("LONG44")
    short["development_eligible"], long["development_eligible"] = R.eligible(short), R.eligible(long)
    long["validation"]["selection_score"] = 100.
    assert short["development_eligible"] and not long["development_eligible"]
    assert R.choose({"SHORT44_SPIKE": short, "LONG44_SPIKE": long}) == "SHORT44_SPIKE"
    short["development_eligible"] = False
    assert R.choose({"SHORT44_SPIKE": short, "LONG44_SPIKE": long}) is None


@pytest.mark.parametrize("failed", ["pooled_sample", "active_days", "fold_sample", "fold_mean", "fold_pf", "unknown_entry", "censor"])
def test_inherited_development_requirements_cannot_be_softened(failed):
    candidate = eligible_candidate()
    if failed == "pooled_sample": candidate["validation"]["completed"] = 499
    elif failed == "active_days": candidate["validation"]["active_days"] = 29
    elif failed == "fold_sample": candidate["walk_forward"][0]["metrics"]["completed"] = 99
    elif failed == "fold_mean": candidate["walk_forward"][0]["metrics"]["mean_net_R"] = 0.
    elif failed == "fold_pf": candidate["walk_forward"][0]["metrics"]["profit_factor"] = 1.
    elif failed == "unknown_entry": candidate["walk_forward"][0]["audit"]["missing_entry"] = 1
    else: candidate["validation"]["censored"] = 1
    assert not R.eligible(candidate)


def test_artifact_checks_refuse_training_matrix_tamper_after_freeze(fitted):
    model = fitted["SHORT44"]
    R.verify_artifacts(model)
    label = model["artifacts"]["matrix_and_both_targets"]["path"]
    with R.path_for(label).open("a") as stream:
        stream.write("tamper\n")
    with pytest.raises(ValueError, match="changed"):
        R.verify_artifacts(model)


def test_repeat_model_write_is_refused_without_mutating_first_matrix(workspace):
    rows, targets, start, end = synthetic_training()
    models, _ = R.fit_pair(rows, targets, start, end, workspace, "frozen")
    label = models["SHORT44"]["artifacts"]["matrix_and_both_targets"]["path"]
    before = R.digest(R.path_for(label))
    with pytest.raises(FileExistsError):
        R.fit_pair(rows, targets, start, end, workspace, "frozen")
    assert R.digest(R.path_for(label)) == before


@pytest.mark.parametrize("key,value", [("target_horizon_minutes", True), ("target_horizon_minutes", 1.),
    ("execution_horizon_minutes", True), ("execution_horizon_minutes", 1.),
    ("label_purge_minutes", 31.), ("training_completed_labels", 1000.)])
def test_temporal_model_metadata_requires_exact_integer_types(key, value):
    model = minimal_model(); model[key] = value
    with pytest.raises(ValueError, match="Frozen family"):
        R.validate_model(model)


def freeze_selection(workspace, monkeypatch, fitted):
    declaration = freeze_declaration(workspace, monkeypatch)
    selection = {**declaration, "stage": "frozen_short_target_development", "symbols": {},
                 "declaration_sha256": R.digest(workspace/"declaration.json")}
    for symbol in R.SYMBOLS:
        candidates = {}
        for family in R.FAMILIES:
            for mode in R.MODES:
                candidate = eligible_candidate(family)
                candidate.update(mode=mode, config=asdict(R.CFG), reference_only=family == "LONG44",
                                 final_model=deepcopy(fitted[family]))
                candidate["development_eligible"] = R.eligible(candidate)
                candidates[f"{family}_{mode}"] = candidate
        selection["symbols"][symbol] = {"models": candidates, "selected_model": R.choose(candidates)}
    R.save_frozen(workspace/"selection.json", selection)
    return selection


def rewrite_selection(workspace, selection):
    path = workspace/"selection.json"; path.write_text(json.dumps(R.canonical(selection)))
    path.with_suffix(".sha256").write_text(R.digest(path))


def test_frozen_selection_accepts_two_target_models_and_only_short_development_choice(workspace, monkeypatch, fitted):
    selection = freeze_selection(workspace, monkeypatch, fitted)
    checked, fingerprint = R.frozen_selection(workspace)
    assert fingerprint == R.digest(workspace/"selection.json")
    assert checked["symbols"]["BOOM600"]["selected_model"].startswith("SHORT44_")
    assert checked["history"]["price_oos"] is False


@pytest.mark.parametrize("problem", ["choose_long", "rehashed_semantic_model", "scaler_diverged", "common_rows_diverged",
    "eligibility_rescue", "history_rescue", "source_drift"])
def test_frozen_selection_refuses_semantic_or_lineage_drift_even_with_new_checksum(problem, workspace, monkeypatch, fitted):
    selection = freeze_selection(workspace, monkeypatch, fitted)
    payload = selection["symbols"]["BOOM600"]
    if problem == "choose_long": payload["selected_model"] = "LONG44_SPIKE"
    elif problem == "rehashed_semantic_model":
        model = payload["models"]["SHORT44_SPIKE"]["final_model"]
        model["estimator"]["quantile"] = .5; refresh(model)
    elif problem == "scaler_diverged":
        model = payload["models"]["LONG44_SPIKE"]["final_model"]
        model["estimator"]["means"][0] += 1.; refresh(model)
    elif problem == "common_rows_diverged":
        payload["models"]["LONG44_SPIKE"]["final_model"]["training_identity"]["matrix_native_sha256"] = "changed"
    elif problem == "eligibility_rescue": payload["models"]["LONG44_SPIKE"]["development_eligible"] = True
    elif problem == "history_rescue": selection["history"]["price_oos"] = True
    else: monkeypatch.setattr(R, "hashes", lambda: {"synthetic_source": "1"*64})
    rewrite_selection(workspace, selection)
    with pytest.raises(ValueError):
        R.frozen_selection(workspace)


@pytest.mark.parametrize("field", ["reference_int", "eligibility_int", "hold_bool", "history_int", "training_identity_float"])
def test_rehashed_selection_rejects_numeric_boolean_aliases(field, workspace, monkeypatch, fitted):
    selection = freeze_selection(workspace, monkeypatch, fitted)
    candidate = selection["symbols"]["BOOM600"]["models"]["SHORT44_SPIKE"]
    if field == "reference_int": candidate["reference_only"] = 0
    elif field == "eligibility_int": candidate["development_eligible"] = 1
    elif field == "hold_bool": candidate["config"]["max_hold_minutes"] = True
    elif field == "history_int": selection["history"]["price_oos"] = 0
    else: candidate["final_model"]["training_identity"]["common_training_rows"] = 1000.
    rewrite_selection(workspace, selection)
    with pytest.raises(ValueError):
        R.frozen_selection(workspace)


def economic_row(family="SHORT44"):
    m = {"completed": 1500, "active_days": 90, "censored": 0, "invalid_uncensored": 0,
        "profit_factor": 1.8, "mean_net_R": .2, "day_undefined_replicates": 0, "weekly_undefined_replicates": 0,
        "day_profit_factor_ci95": [1.55, 2.2], "weekly_profit_factor_ci95": [1.5, 2.3], "holm_p": .01,
        "equity_ruin": False, "closed_trade_max_drawdown": .05}
    comparison = {"policy_inference_available": True,
        "day": {"mean_net_R_ci95": [.1, .3], "baseline_difference_ci95": [.01, .2]},
        "weekly": {"weekly_mean_net_R_ci95": [.08, .3], "weekly_difference_ci95": [.02, .2]}}
    return {"family": family, "metrics": m, "audit": {"missing_entry": 0}, "development_eligible": True,
        "comparisons": {"LONG44": deepcopy(comparison), "CLOCK": deepcopy(comparison)},
        "thirds": [{"metrics": {"completed": 300, "mean_net_R": .1}} for _ in range(3)],
        "doubled_cost": {"mean_net_R": .15}}


@pytest.mark.parametrize("failure", ["mean", "pf", "sample", "days", "pf_day_ci", "pf_week_ci", "undefined_pf",
    "reference_unknown", "reference_difference", "third_sample", "double_cost", "dd", "none"])
def test_conditional_economic_gate_keeps_full_requested_criteria_and_never_claims_fresh_profit(failure):
    row = economic_row(); m = row["metrics"]
    if failure == "mean": m["mean_net_R"] = .099
    elif failure == "pf": m["profit_factor"] = 1.49
    elif failure == "sample": m["completed"] = 999
    elif failure == "days": m["active_days"] = 59
    elif failure == "pf_day_ci": m["day_profit_factor_ci95"][0] = 1.
    elif failure == "pf_week_ci": m["weekly_profit_factor_ci95"][0] = 1.
    elif failure == "undefined_pf": m["day_undefined_replicates"] = 1
    elif failure == "reference_unknown": row["comparisons"]["LONG44"]["policy_inference_available"] = False
    elif failure == "reference_difference": row["comparisons"]["CLOCK"]["weekly"]["weekly_difference_ci95"][0] = 0.
    elif failure == "third_sample": row["thirds"][0]["metrics"]["completed"] = 199
    elif failure == "double_cost": row["doubled_cost"]["mean_net_R"] = 0.
    elif failure == "dd": m["closed_trade_max_drawdown"] = .101
    R.gate(row)
    assert row["conditional_economic_gates_pass"] is (failure == "none")
    assert row["historical_candidate"] is False and row["supports_expected_pf_1_5"] is False
    assert "adaptive_known_history_requires_new_evidence" in row["rejection_reasons"]


@pytest.mark.parametrize("family", ["LONG44", "CLOCK"])
def test_diagnostic_family_never_passes_or_supports_requested_profit(family):
    row = economic_row(family); R.gate(row)
    assert row["reference_only"] and not row["historical_candidate"] and not row["supports_expected_pf_1_5"]
    assert row["rejection_reasons"] == ["fixed_diagnostic_reference_only"]
