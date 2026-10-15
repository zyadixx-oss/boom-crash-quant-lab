"""Nonlinear chronological-release tests using synthetic fixtures and mocks."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.research.multiframe_signal import FEATURE_NAMES
from app.research.nonlinear_signal import fit_histogram_boost, predict_histogram_boost


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("nonlinear_study_tests", ROOT / "scripts/run_spike_nonlinear_study.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
FLAGS = {name: False for name in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")}
INTERVALS = {"old": (pd.Timestamp("2025-10-09T11:08Z"), pd.Timestamp("2026-04-07T11:08Z")),
             "fresh": (pd.Timestamp("2026-04-07T11:08Z"), pd.Timestamp("2026-10-04T11:08Z"))}


def metric():
    return {"completed": 1250, "active_days": 90, "censored": 0, "invalid_uncensored": 0,
            "selection_score": .1, "mean_net_R": .2, "profit_factor": 1.8,
            "p": .001, "holm_p": .012, "day_undefined_replicates": 0, "weekly_undefined_replicates": 0,
            "day_profit_factor_ci95": [1.6, 2.], "weekly_profit_factor_ci95": [1.5, 2.],
            "mean_net_R_ci95": [.1, .3], "weekly_mean_net_R_ci95": [.05, .35],
            "baseline_difference_ci95": [.02, .2], "weekly_difference_ci95": [.01, .2],
            "closed_trade_max_drawdown": .05, "equity_ruin": False}


def result_row(family="BOOST44", mode="SPIKE"):
    return {"family": family, "mode": mode, "config": asdict(RUNNER.CFG), "metrics": metric(),
            "audit": {"missing_entry": 0}, "baseline": metric(), "baseline_audit": {},
            "tail": {}, "thirds": [{"metrics": metric()} for _ in range(3)],
            "sensitivities": [{"fill_mode": "adverse_extreme", "entry_delay_minutes": 1,
                               "round_trip_cost_atr": .2, "mean_net_R": .1}],
            "development_eligible": True}


def reference(day=.02, week=.01):
    return {"metrics": metric(), "audit": {"missing_entry": 0},
            "day_inference": {"baseline_difference_ci95": [day, .2], "p": .001},
            "weekly_inference": {"weekly_difference_ci95": [week, .2], "weekly_p": .002}}


def source_value(symbol, role):
    start, end = INTERVALS[role]
    return {"path": f"data/fixture_{role}/{symbol}.csv", "manifest": f"data/fixture_{role}/{symbol}.json",
            "raw_file": f"data/fixture_{role}/{symbol}_raw.csv",
            "page_audit": f"data/fixture_{role}/{symbol}_pages.json",
            "first_epoch": start.value // 10 ** 9, "last_epoch": end.value // 10 ** 9 - 60,
            "rows": 259200, "expected_grid_rows": 259200, "declared_missing_minutes": 0,
            "sha256": "fixture_price", "manifest_sha256": "fixture_manifest",
            "raw_sha256": "fixture_raw", "page_audit_sha256": "fixture_pages"}


def sources():
    return {symbol: {role: source_value(symbol, role) for role in INTERVALS} for symbol in RUNNER.SYMBOLS}


def split_for(role):
    start, end = INTERVALS[role]
    duration = end - start
    boundary = {q: start + duration * q for q in (.4, .5, .6, .7)}
    return {"development": (start, boundary[.7]), "final_test": (boundary[.7], end),
            "train40": (start, boundary[.4]), "train50": (start, boundary[.5]),
            "train60": (start, boundary[.6]), "wf1": (boundary[.4], boundary[.5]),
            "wf2": (boundary[.5], boundary[.6]), "wf3": (boundary[.6], boundary[.7])}


def candidate(family, mode, eligible=True):
    names = list(RUNNER.BASE_NAMES) if family == "RIDGE19" else list(FEATURE_NAMES)
    return {"family": family, "mode": mode, "development_eligible": eligible,
            "validation": {"selection_score": .1}, "final_model": {
                "family": family, "feature_names": names,
                "estimator": {"fixture": f"{family}_{mode}", "threshold": 0.}}}


def source_fixture(tmp_path, monkeypatch, recovered=True, missing=0):
    """Metadata and arbitrary byte fixtures; no price decoding or OHLC features."""
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: FLAGS.copy())
    symbol = "BOOM600"
    folder = tmp_path / "fixture"
    folder.mkdir()
    first = pd.Timestamp("2026-01-01T00:00Z").value // 10 ** 9
    last = first + 1199 * 60
    raw = folder / "raw.csv"
    raw.write_bytes(b"synthetic-byte-fixture-not-price-data\n")
    request = lambda end, req_id: {"ticks_history": symbol, "count": 1000, "end": end,
                                   "style": "candles", "granularity": 60, "req_id": req_id}
    pages = [{"request": request(last + 59, 1), "returned_rows": 1000,
              "oldest_epoch": first + 200 * 60, "newest_epoch": last, "response_sha256": "a" * 64},
             {"request": request(first + 200 * 60 - 1, 2), "returned_rows": 200,
              "oldest_epoch": first, "newest_epoch": first + 199 * 60, "response_sha256": "b" * 64}]
    errors = [{"end": last + 59, "error": 'RuntimeError: {"code":"RateLimit","message":"retry"}'}] if recovered else []
    document = {"symbol": symbol, "normalization_valid": True, "finite_positive_ohlc": True,
                "canonical_utc_minute_grid": True, "authentication_used": False,
                "fills_or_interpolations": False, "safety": FLAGS.copy(), "page_count": 2,
                "request_errors": errors, "requested_start_epoch": first,
                "requested_cutoff_exclusive_epoch": last + 60,
                "gaps": ([{"missing_bars": missing}] if missing else []),
                "raw_file": "fixture/raw.csv", "raw_sha256": RUNNER.digest(raw), "excluded_rows": []}
    value = {"path": "fixture/clean.csv", "sha256": "unread_normalized_fixture",
             "manifest": "fixture/BOOM600_normalization_manifest.json", "rows": 1200 - missing,
             "first_epoch": first, "last_epoch": last}
    manifest = tmp_path / value["manifest"]
    page_path = folder / f"{symbol}_page_audit.json"

    def write():
        RUNNER.save(manifest, document)
        RUNNER.save(page_path, pages)

    monkeypatch.setattr(RUNNER, "audited_source", lambda name, actual_symbol: deepcopy(value))
    write()
    return value, document, pages, raw, write


def freeze_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: FLAGS.copy())
    monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture_source": "unchanged"})
    values = sources()
    for symbol, pair in values.items():
        for role, value in pair.items():
            for path_key, hash_key in (("path", "sha256"), ("manifest", "manifest_sha256"),
                                       ("raw_file", "raw_sha256"), ("page_audit", "page_audit_sha256")):
                path = tmp_path / value[path_key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"unread-byte-fixture,{symbol},{role},{path_key}\n")
                value[path_key] = str(path)  # Imported frozen verifier accepts fixture absolute paths.
                value[hash_key] = RUNNER.digest(path)
    output = tmp_path / "study"
    output.mkdir()
    declaration = {"stage": "nonlinear_predevelopment_declaration", "sources": values,
                   "code_hashes": RUNNER.hashes(), "config": asdict(RUNNER.CFG),
                   "boost_parameters": RUNNER.BOOST_PARAMS, "families": list(RUNNER.FAMILIES),
                   "joint_fresh_hypotheses": 12}
    RUNNER.save(output / "declaration.json", declaration)
    declared_hash = RUNNER.digest(output / "declaration.json")
    (output / "declaration.sha256").write_text(declared_hash + "\n")
    selection = {**declaration, "stage": "frozen_nonlinear_development",
                 "declaration_sha256": declared_hash, "symbols": {
        symbol: {"selected_model": None, "models": {
            f"{family}_{mode}": candidate(family, mode) for family in RUNNER.FAMILIES for mode in RUNNER.MODES
        }} for symbol in RUNNER.SYMBOLS}}
    RUNNER.save(output / "selection.json", selection)
    (output / "selection.sha256").write_text(RUNNER.digest(output / "selection.json") + "\n")
    return output, values


def test_recovered_rate_limit_requires_same_end_successful_public_page(tmp_path, monkeypatch):
    value, document, pages, _, _ = source_fixture(tmp_path, monkeypatch)
    result = RUNNER.source("fixture", "BOOM600")
    assert result["recovered_request_errors"] == document["request_errors"]
    assert document["request_errors"][0]["end"] == pages[0]["request"]["end"]
    assert result["expected_grid_rows"] == result["rows"] == 1200
    assert result["declared_missing_minutes"] == 0
    assert result["page_audit_sha256"] == RUNNER.digest(tmp_path / result["page_audit"])
    assert result["manifest_sha256"] == RUNNER.digest(tmp_path / value["manifest"])


@pytest.mark.parametrize("error", [
    {"end": 123, "error": 'RuntimeError: {"code":"RateLimit"}'},
    {"end": "SAME", "error": 'RuntimeError: {"code":"InvalidToken"}'},
    {"end": "SAME", "error": "unknown error text"},
    {"end": "SAME", "error": "RuntimeError: []"},
])
def test_unknown_or_unmatched_request_errors_are_refused(error, tmp_path, monkeypatch):
    _, document, pages, _, write = source_fixture(tmp_path, monkeypatch)
    error = deepcopy(error)
    if error["end"] == "SAME":
        error["end"] = pages[0]["request"]["end"]
    document["request_errors"] = [error]
    write()
    with pytest.raises(ValueError, match="request error"):
        RUNNER.source("fixture", "BOOM600")


@pytest.mark.parametrize("tamper", ["duplicate_end", "unknown_request_key", "wrong_count", "nonhex_sha",
                                    "first_end", "broken_cursor", "no_start", "page_count"])
def test_successful_page_chain_must_be_unique_public_complete_and_consistent(tamper, tmp_path, monkeypatch):
    _, document, pages, _, write = source_fixture(tmp_path, monkeypatch)
    if tamper == "duplicate_end":
        pages[1]["request"]["end"] = pages[0]["request"]["end"]
    elif tamper == "unknown_request_key":
        pages[0]["request"]["authorize"] = "unusable-fixture"
    elif tamper == "wrong_count":
        pages[0]["request"]["count"] = 999
    elif tamper == "nonhex_sha":
        pages[0]["response_sha256"] = "x" * 64
    elif tamper == "first_end":
        pages[0]["request"]["end"] -= 1
    elif tamper == "broken_cursor":
        pages[1]["request"]["end"] += 60
    elif tamper == "no_start":
        pages[1]["oldest_epoch"] += 60
    else:
        document["page_count"] += 1
    write()
    with pytest.raises(ValueError, match="page|pagination"):
        RUNNER.source("fixture", "BOOM600")


@pytest.mark.parametrize("tamper", ["authentication", "interpolation", "live_true", "live_missing", "normalization",
                                    "wrong_symbol", "missing_endpoint", "gap_count", "raw_bytes"])
def test_unsafe_metadata_missing_grid_and_raw_byte_drift_are_refused(tamper, tmp_path, monkeypatch):
    _, document, _, raw, write = source_fixture(tmp_path, monkeypatch)
    if tamper == "authentication":
        document["authentication_used"] = True
    elif tamper == "interpolation":
        document["fills_or_interpolations"] = True
    elif tamper == "live_true":
        document["safety"]["LIVE_ALLOWED"] = True
    elif tamper == "live_missing":
        document["safety"].pop("OPENED_TRADES")
    elif tamper == "normalization":
        document["canonical_utc_minute_grid"] = False
    elif tamper == "wrong_symbol":
        document["symbol"] = "CRASH600"
    elif tamper == "missing_endpoint":
        document["requested_cutoff_exclusive_epoch"] -= 60
    elif tamper == "gap_count":
        document["gaps"] = [{"missing_bars": 1}]
    else:
        raw.write_bytes(b"changed-unread-fixture\n")
    write()
    with pytest.raises(ValueError):
        RUNNER.source("fixture", "BOOM600")


def test_declared_real_gap_is_preserved_unknown_without_interpolation(tmp_path, monkeypatch):
    _, _, _, _, _ = source_fixture(tmp_path, monkeypatch, missing=1)
    result = RUNNER.source("fixture", "BOOM600")
    assert result["rows"] == 1199 and result["declared_missing_minutes"] == 1
    assert result["expected_grid_rows"] == 1200


@pytest.mark.parametrize("symbol", ["BOOM500", "CRASH300N", "BOOM600N"])
def test_undeclared_symbol_refused_before_reading_a_source(symbol, monkeypatch):
    monkeypatch.setattr(RUNNER, "audited_source", lambda *args: pytest.fail("Undeclared source must not be read"))
    with pytest.raises(ValueError, match="declared600"):
        RUNNER.source("unread", symbol)


@pytest.mark.parametrize("kind", ["path", "manifest", "raw_file", "page_audit"])
def test_all_frozen_source_byte_layers_are_verified_before_feature_prepare(kind, tmp_path, monkeypatch):
    output, values = freeze_fixture(tmp_path, monkeypatch)
    Path(values["CRASH600"]["fresh"][kind]).write_text("changed-unread-fixture\n")
    monkeypatch.setattr(RUNNER, "prepare", lambda *args, **kwargs: pytest.fail("Drift must block features"))
    with pytest.raises(ValueError, match="changed"):
        RUNNER.evaluate(SimpleNamespace(output=output, bootstrap=9999))


@pytest.mark.parametrize("symbol,direction", [("BOOM600", "boom"), ("CRASH600", "crash")])
def test_prepare_clips_development_before_building_and_keeps_shared44_clock(symbol, direction, monkeypatch):
    m1 = pd.DataFrame(index=pd.date_range("2026-01-01", periods=200, freq="min", tz="UTC"))
    value = {"path": "unread-fixture.csv", "first_epoch": m1.index[0].value // 10 ** 9,
             "last_epoch": m1.index[-1].value // 10 ** 9, "sha256": "fixture", "declared_missing_minutes": 0}
    seen = []
    monkeypatch.setattr(RUNNER, "load_m1", lambda path: (m1.copy(), {"sha256": "fixture", "missing_minutes": 0}))

    def features(frame, actual_direction):
        assert actual_direction == direction
        seen.append(frame.index)
        assert len(frame) == 140
        grid = pd.date_range(frame.index[0], periods=28, freq="5min")
        rows = pd.DataFrame(1., index=grid, columns=[*FEATURE_NAMES, "atr", "close"])
        rows["feature_valid"] = True
        rows.loc[grid[11], "feature_valid"] = False
        return rows, list(FEATURE_NAMES)

    monkeypatch.setattr(RUNNER, "causal_multiframe_inputs", features)
    clipped, opportunities, names, audit, split = RUNNER.prepare(value, symbol, development=True)
    assert len(seen) == 1 and seen[0].equals(clipped.index) and len(clipped) == 140
    assert clipped.index[-1] < split["development"][1]
    assert opportunities.index.tolist() == [pd.Timestamp("2026-01-01T00:30Z"),
                                            pd.Timestamp("2026-01-01T01:30Z"),
                                            pd.Timestamp("2026-01-01T02:00Z")]
    assert names == list(FEATURE_NAMES) and audit["feature_count"] == 44
    assert audit["development_features_only"] is True
    assert audit["feature_end_exclusive"] == clipped.index[-1] + pd.Timedelta(minutes=1)
    assert audit["feature_end_exclusive"] <= split["development"][1].ceil("min")
    assert clipped.index.equals(m1.iloc[:140].index)
    assert audit["frames"] == ["H4", "H1", "M15", "M5", "M1"]


def test_prepare_refuses_undeclared_gap_before_building_features(monkeypatch):
    m1 = pd.DataFrame(index=pd.date_range("2026-01-01", periods=200, freq="min", tz="UTC"))
    value = {"path": "unread", "first_epoch": m1.index[0].value // 10 ** 9,
             "last_epoch": m1.index[-1].value // 10 ** 9, "sha256": "fixture", "declared_missing_minutes": 0}
    monkeypatch.setattr(RUNNER, "load_m1", lambda path: (m1, {"sha256": "fixture", "missing_minutes": 1}))
    monkeypatch.setattr(RUNNER, "causal_multiframe_inputs", lambda *args: pytest.fail("Unknown gap must block inputs"))
    with pytest.raises(ValueError, match="gaps"):
        RUNNER.prepare(value, "BOOM600", development=True)


def test_feature_order_and_ridge19_prediction_use_same44_eligible_clock(monkeypatch):
    names = list(FEATURE_NAMES)
    assert RUNNER.feature_order("BOOST44", names) == names
    assert RUNNER.feature_order("RIDGE44", names) == names
    assert RUNNER.feature_order("RIDGE19", names) == list(RUNNER.BASE_NAMES)
    for family, wrong in (("UNKNOWN", names), ("BOOST44", names[:-1]), ("RIDGE19", names[::-1])):
        with pytest.raises(ValueError, match="base19"):
            RUNNER.feature_order(family, wrong)
    index = pd.date_range("2026-01-01T00:30Z", periods=3, freq="30min")
    rows = pd.DataFrame(1., index=index, columns=[*names, "atr", "close"])
    model = candidate("RIDGE19", "SPIKE")["final_model"]
    frozen = deepcopy(model)
    seen = []

    def score(frame, estimator, feature_names):
        assert list(frame.columns) == list(RUNNER.BASE_NAMES)
        assert frame.index.equals(rows.index)
        assert estimator is model["estimator"] and feature_names == list(RUNNER.BASE_NAMES)
        seen.append(frame.index)
        return np.array([.2, -.1, .3])

    monkeypatch.setattr(RUNNER, "predict_ridge", score)
    signals = RUNNER.issued(rows, "BOOM600", "SPIKE", model)
    assert [item["signal_time"] for item in signals] == index[[0, 2]].tolist()
    assert seen[0].equals(index) and model == frozen


@pytest.mark.parametrize("family", list(RUNNER.FAMILIES))
def test_fit_at_uses31min_planned_purge_and_does_not_modify_nested_estimator(family, monkeypatch):
    index = pd.date_range("2026-01-01", periods=1032, freq="min", tz="UTC")
    names = list(FEATURE_NAMES)
    rows = pd.DataFrame(1., index=index, columns=names)
    labels = pd.DataFrame({"signal_time": index, "net_R": 0.5, "censored": False,
                           "planned_end": index + pd.Timedelta(minutes=31),
                           "actual_end": index + pd.Timedelta(seconds=1)})
    end = index[0] + pd.Timedelta(minutes=1032)
    estimator = {"threshold": .25, "integrity_sha256": "unchanged-nested-fixture"}
    before = deepcopy(estimator)
    captured = []

    def fit(X, y, feature_names, **parameters):
        expected_names = RUNNER.feature_order(family, names)
        assert list(X.columns) == feature_names == expected_names
        assert X.index.max() + pd.Timedelta(minutes=31) <= end
        assert len(X) == len(y) == 1002
        captured.append((X.copy(), y.copy(), parameters))
        return estimator

    monkeypatch.setattr(RUNNER, "fit_histogram_boost", fit)
    monkeypatch.setattr(RUNNER, "fit_ridge", fit)
    model = RUNNER.fit_at(rows, labels, names, family, index[0], end)
    assert estimator == before and model["estimator"] is estimator
    assert model["training_latest_issue"] == index[1001]
    assert model["training_latest_planned_end"] == end
    assert model["label_purge_minutes"] == 31 and model["training_completed_labels"] == 1002
    assert len(model["matrix_and_target_sha256"]) == 64
    if family == "BOOST44":
        assert captured[0][2] == RUNNER.BOOST_PARAMS
    else:
        assert captured[0][2] == {"penalty": .1, "quantile": .75}


@pytest.mark.parametrize("family", list(RUNNER.FAMILIES))
def test_fit_at1000_label_guard_runs_after_purge_and_unknown_filtering(family, monkeypatch):
    index = pd.date_range("2026-01-01", periods=1030, freq="min", tz="UTC")
    rows = pd.DataFrame(1., index=index, columns=FEATURE_NAMES)
    labels = pd.DataFrame({"signal_time": index, "net_R": .2, "censored": False,
                           "planned_end": index + pd.Timedelta(minutes=31)})
    labels.loc[0, "censored"] = True
    labels.loc[1, "net_R"] = np.nan
    monkeypatch.setattr(RUNNER, "fit_histogram_boost", lambda *args, **kwargs: pytest.fail("Insufficient labels"))
    monkeypatch.setattr(RUNNER, "fit_ridge", lambda *args, **kwargs: pytest.fail("Insufficient labels"))
    with pytest.raises(ValueError, match="1000 completed"):
        RUNNER.fit_at(rows, labels, list(FEATURE_NAMES), family, index[0], index[-1] + pd.Timedelta(minutes=1))


def test_synthetic_fitted_boost_checksum_stays_valid_under_runner_metadata(monkeypatch):
    index = pd.date_range("2026-01-01", periods=1000, freq="30min", tz="UTC")
    rows = pd.DataFrame(1., index=index, columns=FEATURE_NAMES)
    labels = pd.DataFrame({"signal_time": index, "net_R": .2, "censored": False,
                           "planned_end": index + pd.Timedelta(minutes=31)})
    model = RUNNER.fit_at(rows, labels, list(FEATURE_NAMES), "BOOST44", index[0],
                          index[-1] + pd.Timedelta(minutes=31))
    frozen = json.dumps(model["estimator"], sort_keys=True)
    np.testing.assert_allclose(predict_histogram_boost(rows, model["estimator"], list(FEATURE_NAMES)), .2)
    assert json.dumps(model["estimator"], sort_keys=True) == frozen
    assert "training_end" not in model["estimator"] and "training_end" in model


@pytest.mark.parametrize("existing", ["results.json", "selection.json", "selection.sha256", "declaration.json", "declaration.sha256"])
def test_guard_refuses_overwrite_before_any_new_read(existing, tmp_path, monkeypatch):
    (tmp_path / existing).write_text("preserve-fixture")
    monkeypatch.setattr(RUNNER, "source", lambda *args: pytest.fail("Guard must run first"))
    with pytest.raises(ValueError, match="overwrite"):
        RUNNER.develop(SimpleNamespace(output=tmp_path))
    assert (tmp_path / existing).read_text() == "preserve-fixture"
    if existing == "results.json":
        with pytest.raises(ValueError, match="overwrite"):
            RUNNER.evaluate(SimpleNamespace(output=tmp_path, bootstrap=9999))


@pytest.mark.parametrize("tamper", ["selection", "declaration", "code"])
def test_freeze_or_code_drift_blocks_evaluation_before_fresh_prepare(tamper, tmp_path, monkeypatch):
    output, _ = freeze_fixture(tmp_path, monkeypatch)
    if tamper == "selection":
        with (output / "selection.json").open("a") as stream:
            stream.write(" ")
    elif tamper == "declaration":
        with (output / "declaration.json").open("a") as stream:
            stream.write(" ")
    else:
        monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture_source": "changed"})
    monkeypatch.setattr(RUNNER, "prepare", lambda *args, **kwargs: pytest.fail("Integrity drift must block inputs"))
    with pytest.raises(ValueError, match="changed"):
        RUNNER.evaluate(SimpleNamespace(output=output, bootstrap=9999))


def test_primary_bootstrap_is_fixed_before_fresh_feature_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: FLAGS.copy())
    monkeypatch.setattr(RUNNER, "frozen", lambda *args: pytest.fail("Wrong inference count must block loading"))
    with pytest.raises(ValueError, match="9999"):
        RUNNER.evaluate(SimpleNamespace(output=tmp_path, bootstrap=999))


def test_development_only_prepares_older_first70_with_ordered3wf_and12_models(tmp_path, monkeypatch):
    values = sources()
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: FLAGS.copy())
    monkeypatch.setattr(RUNNER, "source", lambda folder, symbol: deepcopy(values[symbol][
        "old" if folder.endswith("_old") else "fresh"]))
    monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture_source": "unchanged"})
    monkeypatch.setattr(RUNNER, "verify_sources", lambda source_values: None)
    prepared, fit_windows, replay_windows = [], [], []

    def prepare(value, symbol, development=False):
        assert value == values[symbol]["old"] and development is True
        assert (tmp_path / "declaration.json").exists()
        assert (tmp_path / "declaration.sha256").exists()
        assert not (tmp_path / "selection.json").exists()
        prepared.append(symbol)
        start, end = split_for("old")["development"]
        m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
        return m1, pd.DataFrame(), list(FEATURE_NAMES), {}, split_for("old")

    def replay(m1, signals, config, start, end, purge_minutes):
        assert config == RUNNER.CFG and purge_minutes == 31
        assert end <= split_for("old")["development"][1]
        replay_windows.append((start, end))
        return pd.DataFrame(columns=["signal_time"]), {}

    def fit(rows, labels, names, family, start, end):
        assert end <= split_for("old")["development"][1]
        fit_windows.append((family, start, end))
        return {"family": family, "feature_names": RUNNER.feature_order(family, names),
                "estimator": {"threshold": 0.}, "training_end": end}

    monkeypatch.setattr(RUNNER, "prepare", prepare)
    monkeypatch.setattr(RUNNER, "issued", lambda *args: [])
    monkeypatch.setattr(RUNNER, "replay_timed", replay)
    monkeypatch.setattr(RUNNER, "fit_at", fit)
    monkeypatch.setattr(RUNNER, "summarize", lambda *args, **kwargs: metric())
    RUNNER.develop(SimpleNamespace(output=tmp_path))
    assert prepared == list(RUNNER.SYMBOLS)
    assert len(fit_windows) == 48 and len(replay_windows) == 40
    expected_windows = {split_for("old")[part] for part in ("train40", "train50", "train60", "development")}
    assert {(start, end) for _, start, end in fit_windows} == expected_windows
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["joint_fresh_hypotheses"] == 12
    assert selection["fresh_features_evaluated"] is selection["fresh_payoffs_evaluated"] is False
    total = 0
    for payload in selection["symbols"].values():
        assert len(payload["models"]) == 6
        for value in payload["models"].values():
            assert [(fold["training"], fold["test"]) for fold in value["walk_forward"]] == list(RUNNER.WALK_FORWARD)
            assert value["final_model"]["training_end"] == str(split_for("old")["development"][1])
            total += 1
    assert total == 12 and selection["frozen_utc"]
    assert selection["declaration_sha256"] == RUNNER.digest(tmp_path / "declaration.json")
    assert (tmp_path / "selection.sha256").read_text().strip() == RUNNER.digest(tmp_path / "selection.json")


def test_no_eligible_candidate_keeps_null_without_fallback():
    candidates = {"BOOST44_SPIKE": candidate("BOOST44", "SPIKE", False),
                  "RIDGE19_DRIFT": candidate("RIDGE19", "DRIFT", False)}
    assert RUNNER.choose(candidates) is None
    candidates["RIDGE19_DRIFT"]["development_eligible"] = True
    assert RUNNER.choose(candidates) is None
    candidates["BOOST44_SPIKE"]["development_eligible"] = True
    assert RUNNER.choose(candidates) == "BOOST44_SPIKE"


def test_profitable_eligible_ridge19_is_only_a_reference_not_combined_strategy():
    target = result_row("RIDGE19")
    RUNNER.nonlinear_gate(target)
    assert target["user_target_observed"]
    assert not target["satisfies_requested_timeframes"]
    assert not target["historical_candidate"] and not target["supports_expected_pf_1_5"]
    assert "m5_reference_not_combined_strategy" in target["rejection_reasons"]


@pytest.mark.parametrize("family", list(RUNNER.FAMILIES))
@pytest.mark.parametrize("mode,side", [("SPIKE", 1), ("DRIFT", -1)])
def test_negative_scores_never_force_signal_and_positive_cutoff_is_frozen(family, mode, side, monkeypatch):
    index = pd.date_range("2026-01-01T00:30Z", periods=5, freq="30min")
    rows = pd.DataFrame({"atr": 2., "close": 100.}, index=index)
    model = {"family": family, "estimator": {"threshold": .2}}
    monkeypatch.setattr(RUNNER, "predict", lambda frame, frozen: np.array([-.1, 0., .1, .2, .3]))
    signals = RUNNER.issued(rows, "BOOM600", mode, model)
    assert [item["signal_time"] for item in signals] == index[[3, 4]].tolist()
    assert [item["score"] for item in signals] == [.2, .3]
    assert all(item["side"] == side for item in signals)
    monkeypatch.setattr(RUNNER, "predict", lambda *args: np.full(5, -.5))
    assert RUNNER.issued(rows, "BOOM600", mode, model) == []
    assert model["estimator"]["threshold"] == .2


def test_linear_reference_conjunction_requires_all_day_and_week_p_values(monkeypatch):
    rows = {f"{family}_SPIKE": result_row(family) for family in RUNNER.FAMILIES}
    target = rows["BOOST44_SPIKE"]
    target["metrics"]["p"] = .03
    ledgers = {key: pd.DataFrame({"fixture": [key]}) for key in rows}
    day_values, week_values, seen = iter([.02, .06]), iter([.04, .05]), []

    def day(primary, baseline, *args, **kwargs):
        assert primary is ledgers["BOOST44_SPIKE"]
        seen.append(baseline.iloc[0].fixture)
        assert kwargs == {"repeats": 9999, "seed": 20261005}
        return {"p": next(day_values), "baseline_difference_ci95": [.01, .2]}

    def week(primary, baseline, *args):
        assert primary is ledgers["BOOST44_SPIKE"] and args[-1] == 9999
        return {"weekly_p": next(week_values), "weekly_difference_ci95": [.01, .2]}

    monkeypatch.setattr(RUNNER, "paired_inference", day)
    monkeypatch.setattr(RUNNER, "weekly_inference", week)
    RUNNER.compare_references(rows, ledgers, "SPIKE", pd.Timestamp("2026-01-01T00:00Z"), pd.Timestamp("2026-02-01T00:00Z"))
    assert target["metrics"]["p"] == .06
    assert seen == ["RIDGE44_SPIKE", "RIDGE19_SPIKE"]
    assert set(target["linear_references"]) == {"RIDGE44", "RIDGE19"}


def test_primary_clock_conjunction_keeps_positive_day_and_week_tests_and_fixed_cost_stresses(monkeypatch):
    start, end = pd.Timestamp("2026-01-01T00:00Z"), pd.Timestamp("2026-02-01T00:00Z")
    ledger = pd.DataFrame({"signal_time": [start + pd.Timedelta(minutes=30)],
                           "gross_R": [1.], "net_R": [.95]})
    captured = []
    monkeypatch.setattr(RUNNER, "issued", lambda *args: [])

    def replay(m1, signals, config, lower, upper, purge_minutes):
        assert purge_minutes == 31 and (lower, upper) == (start, end)
        captured.append(config)
        return ledger.copy(), {"missing_entry": 0}

    def summarize(trades, *args, **kwargs):
        result = metric()
        result["mean_net_R"] = trades.net_R.mean()
        return result

    monkeypatch.setattr(RUNNER, "replay_timed", replay)
    monkeypatch.setattr(RUNNER, "summarize", summarize)
    monkeypatch.setattr(RUNNER, "paired_inference", lambda *args, **kwargs: {
        "p": .02, "baseline_difference_ci95": [.01, .2]})
    monkeypatch.setattr(RUNNER, "weekly_inference", lambda *args: {
        "weekly_p": .05, "weekly_difference_ci95": [.02, .4], "weekly_mean_net_R_ci95": [.03, .6]})
    monkeypatch.setattr(RUNNER, "profit_factor_inference", lambda *args, **kwargs: {})
    monkeypatch.setattr(RUNNER, "tail_metrics", lambda *args: {})
    monkeypatch.setattr(RUNNER, "thirds", lambda *args: [])
    target, _, signals = RUNNER.evaluate_model(pd.DataFrame(), pd.DataFrame(), "BOOM600", "SPIKE",
                                              candidate("BOOST44", "SPIKE")["final_model"],
                                              start, end, ledger, {"missing_entry": 0})
    assert target["metrics"]["day_p"] == .02
    assert target["metrics"]["weekly_p"] == .05
    assert target["metrics"]["clock_conjunction_p"] == target["metrics"]["p"] == .05
    assert captured[0] == RUNNER.CFG and len(captured) == 4
    assert len(target["sensitivities"]) == 18
    assert {item["round_trip_cost_atr"] for item in target["sensitivities"]} == {0., .025, .05, .10, .20, .40}
    doubled = next(item for item in target["sensitivities"] if item["fill_mode"] == "adverse_extreme"
                   and item["entry_delay_minutes"] == 1 and item["round_trip_cost_atr"] == .20)
    assert doubled["mean_net_R"] == pytest.approx(.9)
    assert signals.empty


@pytest.mark.parametrize("family,day,week", [("RIDGE44", None, .1), ("RIDGE44", 0., .1),
                                           ("RIDGE19", .1, None), ("RIDGE19", .1, 0.),
                                           ("RIDGE19", -.1, .1), ("RIDGE44", np.nan, .1),
                                           ("RIDGE19", .1, np.inf)])
def test_boost_needs_positive_day_week_excess_over_both_references(family, day, week):
    target = result_row()
    target["linear_references"] = {"RIDGE44": reference(), "RIDGE19": reference()}
    target["linear_references"][family] = reference(day, week)
    RUNNER.nonlinear_gate(target)
    assert target["user_target_observed"] and not target["historical_candidate"]
    assert not target["supports_expected_pf_1_5"] and not target["nonlinear_added_value"]
    assert f"no_positive_day_week_excess_over_{family}" in target["rejection_reasons"]


@pytest.mark.parametrize("references", [{}, {"RIDGE44": reference()},
                                        {"RIDGE19": reference()}, {"OTHER": reference()}])
def test_missing_or_unknown_linear_reference_cannot_promote_boost(references):
    target = result_row()
    target["linear_references"] = deepcopy(references)
    RUNNER.nonlinear_gate(target)
    assert not target["historical_candidate"] and not target["nonlinear_added_value"]
    assert "missing_or_undeclared_linear_reference_family" in target["rejection_reasons"]


@pytest.mark.parametrize("problem", ["censored", "invalid_uncensored", "missing_entry", "development"])
def test_large_pf_does_not_rescue_unknown_references_or_failed_development(problem):
    target = result_row()
    target["linear_references"] = {"RIDGE44": reference(), "RIDGE19": reference()}
    if problem in ("censored", "invalid_uncensored"):
        target["linear_references"]["RIDGE44"]["metrics"][problem] = 1
    elif problem == "missing_entry":
        target["linear_references"]["RIDGE19"]["audit"][problem] = 1
    else:
        target["development_eligible"] = False
    RUNNER.nonlinear_gate(target)
    assert target["user_target_observed"] and not target["historical_candidate"]
    assert not target["supports_expected_pf_1_5"]


def test_combined_evaluation_freezes_all12_primary_hypotheses_and_never_refits(tmp_path, monkeypatch):
    values = sources()
    selection = {"declaration_sha256": "fixture_decl", "code_hashes": {"fixture": "unchanged"},
                 "sources": values, "symbols": {symbol: {"selected_model": None, "models": {
        f"{family}_{mode}": candidate(family, mode) for family in RUNNER.FAMILIES for mode in RUNNER.MODES
    }} for symbol in RUNNER.SYMBOLS}}
    original_selection = deepcopy(selection)
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: FLAGS.copy())
    monkeypatch.setattr(RUNNER, "frozen", lambda output: (selection, "fixture_selection"))
    prepared, model_calls, families = [], [], []

    def prepare(value, symbol):
        role = "old" if value is values[symbol]["old"] else "fresh"
        start, end = INTERVALS[role]
        first = split_for(role)["final_test"][0].ceil("30min") if role == "old" else start.ceil("30min")
        index = pd.date_range(first, periods=4, freq="30min")
        rows = pd.DataFrame(1., index=index, columns=FEATURE_NAMES)
        m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
        prepared.append((symbol, role))
        return m1, rows, list(FEATURE_NAMES), {}, split_for(role)

    def evaluate(m1, rows, symbol, mode, model, start, end, baseline, baseline_audit):
        assert len(rows) == 4 and rows.index.minute.isin([0, 30]).all()
        assert model is selection["symbols"][symbol]["models"][f"{model['family']}_{mode}"]["final_model"]
        assert model["feature_names"] == RUNNER.feature_order(model["family"], list(FEATURE_NAMES))
        model_calls.append((symbol, model["family"], mode, tuple(rows.index)))
        return result_row(model["family"], mode), pd.DataFrame(), pd.DataFrame()

    def compare(models, ledgers, mode, start, end):
        models[f"BOOST44_{mode}"]["linear_references"] = {"RIDGE44": reference(), "RIDGE19": reference()}
        models[f"BOOST44_{mode}"]["metrics"]["p"] = .02

    original_holm = RUNNER.holm

    def holm(rows):
        families.append([(row["family"], row["mode"]) for row in rows])
        original_holm(rows)

    monkeypatch.setattr(RUNNER, "prepare", prepare)
    monkeypatch.setattr(RUNNER, "issued", lambda *args: [])
    monkeypatch.setattr(RUNNER, "replay_timed", lambda *args, **kwargs: (pd.DataFrame(), {"missing_entry": 0}))
    monkeypatch.setattr(RUNNER, "evaluate_model", evaluate)
    monkeypatch.setattr(RUNNER, "compare_references", compare)
    monkeypatch.setattr(RUNNER, "holm", holm)
    monkeypatch.setattr(RUNNER, "fit_at", lambda *args, **kwargs: pytest.fail("OOS cannot refit"))
    monkeypatch.setattr(RUNNER, "choose", lambda *args: pytest.fail("OOS cannot reselect"))
    RUNNER.evaluate(SimpleNamespace(output=tmp_path, bootstrap=9999))
    assert selection == original_selection
    assert prepared == [(symbol, role) for symbol in RUNNER.SYMBOLS for role in ("old", "fresh")]
    assert len(model_calls) == 24
    assert len(families) == 1 and len(families[0]) == 12
    assert all(families[0].count((family, mode)) == 2 for family in RUNNER.FAMILIES for mode in RUNNER.MODES)
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["joint_fresh_hypotheses"] == 12 and result["selection_sha256"] == "fixture_selection"
    assert result["safety"] == FLAGS
    assert result["actual_money_profit"] == "NOT TESTED" and result["forward_test"] is False
    for payload in result["symbols"].values():
        assert payload["selected_model"] is None
        for key, target in payload["cohorts"]["old_final30"]["models"].items():
            assert target["historical_candidate"] is False and target["rejection_reasons"] == ["secondary_old_holdout"]
        primary = payload["cohorts"]["later_temporal180"]["models"]
        assert len(primary) == 6
        # Eight ridge tests p=.001 then four conjunction BOOST tests p=.02.
        assert primary["RIDGE19_SPIKE"]["metrics"]["holm_p"] == pytest.approx(.012)
        assert primary["BOOST44_SPIKE"]["metrics"]["holm_p"] == pytest.approx(.08)
        assert not primary["BOOST44_SPIKE"]["historical_candidate"]
