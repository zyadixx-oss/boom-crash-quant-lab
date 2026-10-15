"""Combined-frame staging/lineage/common-clock inference, synthetic fixtures."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("multiframe_study_tests", ROOT / "scripts/run_spike_multiframe_study.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def sources():
    intervals = {"old": ("2025-10-09T11:08Z", "2026-04-07T11:08Z"),
                 "fresh": ("2026-04-07T11:08Z", "2026-10-04T11:08Z")}
    return {symbol: {role: {
        "path": f"data/{'spike_learned_transfer' if role == 'old' else 'spike_native300_recent'}/{symbol.lower()}_m1_180d_clean.csv",
        "manifest": f"fixture/{symbol}_{role}.json", "sha256": f"{symbol}_{role}",
        "manifest_sha256": "fixturemanifest", "first_epoch": pd.Timestamp(start).value // 1_000_000_000,
        "last_epoch": pd.Timestamp(end).value // 1_000_000_000 - 60,
        "declared_missing_minutes": 0 if role == "old" else 1,
    } for role, (start, end) in intervals.items()} for symbol in RUNNER.SYMBOLS}


def splits(value):
    start = pd.Timestamp(value["first_epoch"], unit="s", tz="UTC")
    end = pd.Timestamp(value["last_epoch"] + 60, unit="s", tz="UTC")
    span = end - start
    b = {q: start + q * span for q in (.4, .5, .6, .7)}
    return {"development": (start, b[.7]), "final_test": (b[.7], end),
            "train40": (start, b[.4]), "train50": (start, b[.5]), "train60": (start, b[.6]),
            "wf1": (b[.4], b[.5]), "wf2": (b[.5], b[.6]), "wf3": (b[.6], b[.7])}


def metrics():
    return {"completed": 1250, "active_days": 90, "censored": 0, "invalid_uncensored": 0,
            "selection_score": .1, "mean_net_R": .2, "profit_factor": 1.8,
            "p": .001, "holm_p": .004, "day_undefined_replicates": 0, "weekly_undefined_replicates": 0,
            "day_profit_factor_ci95": [1.6, 2.], "weekly_profit_factor_ci95": [1.5, 2.],
            "mean_net_R_ci95": [.1, .3], "weekly_mean_net_R_ci95": [.05, .35],
            "baseline_difference_ci95": [.02, .2], "weekly_difference_ci95": [.01, .2],
            "closed_trade_max_drawdown": .05, "equity_ruin": False}


def row(mode="SPIKE"):
    return {"mode": mode, "metrics": metrics(), "audit": {"missing_entry": 0},
            "config": asdict(RUNNER.CFG), "baseline": metrics(), "baseline_audit": {},
            "tail": {}, "thirds": [{"metrics": metrics()} for _ in range(3)],
            "sensitivities": [{"fill_mode": "adverse_extreme", "entry_delay_minutes": 1,
                               "round_trip_cost_atr": .2, "mean_net_R": .1}],
            "development_eligible": False}


def candidate(mode, names):
    return {"mode": mode, "development_eligible": False,
            "final_model": {"feature_names": names, "fixture_mode": mode}}


def add_reference(target, lower_day=.02, lower_week=.01):
    target["m5_reference"] = {"metrics": metrics(), "audit": {"missing_entry": 0},
                              "day_inference": {"baseline_difference_ci95": [lower_day, .2]},
                              "weekly_inference": {"weekly_difference_ci95": [lower_week, .2]}}
    return target


def native_selection(values):
    return {"declaration_sha256": "native_declaration", "code_hashes": {"native": "frozen"},
            "sources": values, "symbols": {symbol: {
                "models": {mode: candidate(mode, [f"f{i}" for i in range(19)]) for mode in RUNNER.MODES},
                "selected_direction": None} for symbol in RUNNER.SYMBOLS}}


def test_prepare_is_closed_m5_utc_clock_without_replacement_or_finer_issuance(monkeypatch):
    m1 = pd.DataFrame(index=pd.date_range("2026-01-01", periods=120, freq="min", tz="UTC"))
    m5 = pd.DataFrame(index=pd.date_range(m1.index[0], periods=24, freq="5min"))
    m5["feature_valid"] = True
    m5.loc[m5.index[5], "feature_valid"] = False
    names = [f"f{i}" for i in range(44)]
    monkeypatch.setattr(RUNNER, "load_m1", lambda path: (m1, {}))
    monkeypatch.setattr(RUNNER, "causal_multiframe_inputs", lambda frame, direction: (m5, names))
    _, opportunities, result_names, audit, _ = RUNNER.prepare(Path("unread_fixture"), "BOOM300N")
    assert opportunities.index.tolist() == [pd.Timestamp("2026-01-01T01:00Z"),
                                            pd.Timestamp("2026-01-01T01:30Z"),
                                            pd.Timestamp("2026-01-01T02:00Z")]
    assert result_names == names and audit["feature_count"] == 44
    assert audit["frames"] == ["H4", "H1", "M15", "M5", "M1"]


def test_native_freeze_is_required_before_combined_development(monkeypatch, tmp_path):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    def missing(path):
        raise FileNotFoundError("native selection is not frozen")
    monkeypatch.setattr(RUNNER, "native_frozen_selection", missing)
    def forbidden(*args):
        raise AssertionError("No native freeze: no combined features")
    monkeypatch.setattr(RUNNER, "prepare", forbidden)
    with pytest.raises(FileNotFoundError):
        RUNNER.develop(SimpleNamespace(output=tmp_path, native_study=tmp_path / "native"))


def test_native_results_before_joint_freeze_block_new_development(monkeypatch, tmp_path):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    native_path = tmp_path / "native"
    native_path.mkdir()
    (native_path / "results.json").write_text('{}')
    monkeypatch.setattr(RUNNER, "native_frozen_selection", lambda path: (native_selection(sources()), "native_sha"))
    def forbidden(*args):
        raise AssertionError("Fresh native results cannot precede combined selection")
    monkeypatch.setattr(RUNNER, "prepare", forbidden)
    with pytest.raises(ValueError, match="preceded"):
        RUNNER.develop(SimpleNamespace(output=tmp_path, native_study=native_path))


def test_development_only_prepares_old_context_and_purged_training_folds(monkeypatch, tmp_path):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    values = sources()
    native = native_selection(values)
    prepared, fitted, replayed = [], [], []
    monkeypatch.setattr(RUNNER, "native_frozen_selection", lambda path: (native, "native_sha"))
    monkeypatch.setattr(RUNNER, "source", lambda folder, symbol: deepcopy(values[symbol][
        "old" if folder.endswith("spike_learned_transfer") else "fresh"]))
    monkeypatch.setattr(RUNNER, "hashes", lambda: {"combined": "frozen"})
    monkeypatch.setattr(RUNNER, "verify_prices", lambda values: None)

    def prepare(path, symbol):
        assert path.parent.name == "spike_learned_transfer"
        assert (tmp_path / "declaration.json").exists()
        assert not (tmp_path / "selection.json").exists()
        prepared.append(path)
        value = values[symbol]["old"]
        split = splits(value)
        start, end = split["development"][0], pd.Timestamp(value["last_epoch"] + 60, unit="s", tz="UTC")
        m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
        return m1, pd.DataFrame(), [f"f{i}" for i in range(44)], {
            "sha256": value["sha256"], "missing_minutes": 0}, split

    def replay(m1, signals, cfg, start, end, purge_minutes):
        assert cfg == RUNNER.CFG and purge_minutes == 31
        assert end <= splits(values["BOOM300N"]["old"])["development"][1]
        replayed.append((start, end))
        return pd.DataFrame(columns=["signal_time"]), {}

    def fit(rows, labels, names, start, end):
        assert len(names) == 44
        fitted.append((start, end))
        return {"feature_names": names, "training_end": end}

    monkeypatch.setattr(RUNNER, "prepare", prepare)
    monkeypatch.setattr(RUNNER, "issued", lambda *args: [])
    monkeypatch.setattr(RUNNER, "replay_timed", replay)
    monkeypatch.setattr(RUNNER, "fit_at", fit)
    monkeypatch.setattr(RUNNER, "summarize", lambda *args, **kwargs: metrics())
    RUNNER.develop(SimpleNamespace(output=tmp_path, native_study=tmp_path / "native"))
    assert len(prepared) == 2 and len(fitted) == len(replayed) == 16
    assert set(fitted) == {splits(values["BOOM300N"]["old"])[part] for part in (
        "train40", "train50", "train60", "development")}
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["native_reference"]["selection_sha256"] == "native_sha"
    assert selection["joint_fresh_hypotheses"] == 8
    assert selection["fresh_features_evaluated"] is selection["fresh_payoffs_evaluated"] is False
    assert selection["frozen_utc"]
    assert (tmp_path / "selection.sha256").read_text().strip() == RUNNER.digest(tmp_path / "selection.json")


def test_common_clock_uses_feature_timestamps_and_never_outcome_knownness():
    index = pd.date_range("2026-01-01", periods=5, freq="30min", tz="UTC")
    native = pd.DataFrame({"future_net_R": [1., np.nan, -999, 999, np.inf]}, index=index)
    combined = pd.DataFrame({"future_net_R": [np.nan, np.inf, -999]}, index=index[[1, 2, 4]])
    a, b = RUNNER.common_clock(combined, native)
    assert a.index.equals(combined.index) and b.index.equals(combined.index)
    assert np.isnan(b.iloc[0].future_net_R)
    with pytest.raises(ValueError, match="base19"):
        RUNNER.common_clock(combined, native.drop(index[2]))


def test_m5_reference_conjunction_p_and_execution_keep_frozen_parameters(monkeypatch):
    index = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    shared = pd.DataFrame({"first": [1., 2., 3.]}, index=index)
    target = row()
    target["metrics"]["p"] = .03
    frozen = {"fixture": "frozen_native"}
    captured = []
    def issued(rows, symbol, mode, model):
        assert rows.index.equals(shared.index) and model is frozen
        return []
    def replay(m1, signals, cfg, start, end, purge_minutes):
        captured.append((cfg, purge_minutes))
        return pd.DataFrame(), {"missing_entry": 0}
    monkeypatch.setattr(RUNNER, "issued", issued)
    monkeypatch.setattr(RUNNER, "replay_timed", replay)
    monkeypatch.setattr(RUNNER, "paired_inference", lambda *args, **kwargs: {
        "p": .02, "baseline_difference_ci95": [.01, .2]})
    monkeypatch.setattr(RUNNER, "weekly_inference", lambda *args, **kwargs: {
        "weekly_p": .04, "weekly_difference_ci95": [.01, .2]})
    monkeypatch.setattr(RUNNER, "summarize", lambda *args, **kwargs: metrics())
    RUNNER.compare_m5_reference(target, pd.DataFrame(), pd.DataFrame(), shared, "BOOM300N", "SPIKE",
                               frozen, index[0], index[-1] + pd.Timedelta(minutes=30), 9999)
    assert captured == [(RUNNER.CFG, 31)]
    assert target["metrics"]["p"] == .04
    assert target["metrics"]["clock_conjunction_p"] == .03
    assert target["metrics"]["m5_reference_day_p"] == .02
    assert target["metrics"]["m5_reference_weekly_p"] == .04


@pytest.mark.parametrize("day, week", [(None, .1), (0., .1), (.1, None), (.1, 0.)])
def test_added_value_needs_positive_excess_in_both_day_and_week(day, week):
    target = add_reference(row(), day, week)
    RUNNER.combined_gate(target, eligible=True)
    assert target["user_target_observed"] and not target["historical_candidate"]
    assert not target["supports_expected_pf_1_5"] and not target["adds_value_over_m5_reference"]
    assert "no_positive_day_week_excess_over_common_clock_m5" in target["rejection_reasons"]


def test_reference_censoring_and_failed_development_cannot_be_rescued_by_pf():
    target = add_reference(row())
    target["m5_reference"]["metrics"]["censored"] = 1
    RUNNER.combined_gate(target, eligible=True)
    assert "missing_or_censored_m5_reference" in target["rejection_reasons"]
    target = add_reference(row())
    RUNNER.combined_gate(target, eligible=False)
    assert "development_rejected" in target["rejection_reasons"]
    assert not target["historical_candidate"]


def test_native_results_must_match_lineage_and_follow_combined_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    reference = {"study_dir": ".", "selection_sha256": "native_sha",
                 "code_hashes": {"native": "frozen"}}
    selection = {"native_reference": reference, "frozen_utc": "2026-10-05T04:00:00Z"}
    result = {"selection_sha256": "native_sha", "code_hashes": reference["code_hashes"],
              "run_utc": "2026-10-05T04:00:01Z"}
    RUNNER.save(tmp_path / "results.json", result)
    loaded, fingerprint = RUNNER.native_result(selection)
    assert loaded == result and fingerprint == RUNNER.digest(tmp_path / "results.json")
    result["run_utc"] = "2026-10-05T03:59:59Z"
    RUNNER.save(tmp_path / "results.json", result)
    with pytest.raises(ValueError, match="before"):
        RUNNER.native_result(selection)
    result["run_utc"] = "2026-10-05T04:00:01Z"
    result["selection_sha256"] = "different_native"
    RUNNER.save(tmp_path / "results.json", result)
    with pytest.raises(ValueError, match="selection"):
        RUNNER.native_result(selection)


def test_native_lineage_paths_are_portable_and_cannot_escape_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "native_frozen_selection", lambda path: (native_selection(sources()), "native_sha"))
    _, lineage = RUNNER.native_lineage(tmp_path / "docs/native")
    assert lineage["study_dir"] == "docs/native"
    assert RUNNER.native_study_path(lineage["study_dir"]) == tmp_path / "docs/native"
    with pytest.raises(ValueError, match="inside the repository"):
        RUNNER.native_lineage(tmp_path.parent / "outside-native-study")


def test_joint_evaluation_keeps_native_results_immutable_and_adjusts_all8(monkeypatch, tmp_path):
    values = sources()
    native = native_selection(values)
    combined_names = [f"f{i}" for i in range(44)]
    native_names = combined_names[:19]
    selection = {"declaration_sha256": "combined_decl", "code_hashes": {"combined": "frozen"},
                 "native_reference": {"selection_sha256": "native_sha"}, "sources": values,
                 "symbols": {symbol: {"selected_direction": None,
                     "models": {mode: candidate(mode, combined_names) for mode in RUNNER.MODES}}
                     for symbol in RUNNER.SYMBOLS}}
    native_results = {"symbols": {symbol: {"cohorts": {"fresh_temporal180": {
        "models": {mode: row(mode) for mode in RUNNER.MODES}}}} for symbol in RUNNER.SYMBOLS}}
    untouched_native = deepcopy(native_results)
    prepare_calls, family_sizes = [], []
    monkeypatch.setattr(RUNNER, "frozen_selection", lambda output: (selection, "combined_sha", native))
    monkeypatch.setattr(RUNNER, "native_result", lambda frozen: (native_results, "native_result_sha"))

    def prepare(path, symbol, names):
        role = "old" if path.parent.name == "spike_learned_transfer" else "fresh"
        value = values[symbol][role]
        split = splits(value)
        start = split["development"][0]
        end = pd.Timestamp(value["last_epoch"] + 60, unit="s", tz="UTC")
        first_issue = split["final_test"][0].ceil("30min") if role == "old" else start.ceil("30min")
        index = pd.date_range(first_issue, periods=5, freq="30min")
        if len(names) == 44:
            index = index[1:]
        rows = pd.DataFrame(1., index=index, columns=names)
        prepare_calls.append((symbol, role, len(names)))
        m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
        return m1, rows, names, {"sha256": value["sha256"],
                                "missing_minutes": value["declared_missing_minutes"]}, split

    def evaluate_one(m1, rows, symbol, mode, model, start, end, repeats):
        assert len(rows) == 4 and model["feature_names"] == combined_names
        return row(mode), pd.DataFrame(), pd.DataFrame()

    def reference(target, trades, m1, rows, symbol, mode, model, start, end, repeats):
        assert len(rows) == 4 and model["feature_names"] == native_names
        add_reference(target)
        target["metrics"]["p"] = .01
        return pd.DataFrame(), pd.DataFrame()

    original_holm = RUNNER.holm
    def holm(family):
        family_sizes.append(len(family))
        original_holm(family)
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation cannot fit or reselect")
    monkeypatch.setattr(RUNNER, "prepare", lambda path, symbol: prepare(path, symbol, combined_names))
    monkeypatch.setattr(RUNNER, "native_prepare", lambda path, symbol: prepare(path, symbol, native_names))
    monkeypatch.setattr(RUNNER, "evaluate_one", evaluate_one)
    monkeypatch.setattr(RUNNER, "compare_m5_reference", reference)
    monkeypatch.setattr(RUNNER, "holm", holm)
    monkeypatch.setattr(RUNNER, "fit_at", forbidden)
    monkeypatch.setattr(RUNNER, "choose_direction", forbidden)
    RUNNER.evaluate(SimpleNamespace(output=tmp_path, bootstrap=9999))
    assert native_results == untouched_native and family_sizes == [8]
    assert len(prepare_calls) == 8
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["joint_fresh_hypotheses"] == 8
    for symbol in RUNNER.SYMBOLS:
        data = result["symbols"][symbol]["cohorts"]["fresh_temporal180"]
        assert data["comparison_clock"]["common_clock_rows"] == 4
        assert data["comparison_clock"]["full_native_eligible_clock_rows"] == 5
        assert data["comparison_clock"]["intersection_uses_future_outcomes"] is False
        for mode in RUNNER.MODES:
            fresh = data["models"][mode]
            assert fresh["metrics"]["holm_p"] == pytest.approx(.04)
            assert not fresh["historical_candidate"] and not fresh["selected_direction"]
            copied_native = result["native_reference_joint_inference"]["symbols"][symbol]["models"][mode]
            assert copied_native["original_native_holm_p"] == .004
            assert copied_native["metrics"]["holm_p"] == pytest.approx(.008)
