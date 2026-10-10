"""Native-300 chronology and immutable evaluation release, fixtures only."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native300_study_tests", ROOT / "scripts/run_spike_native300_study.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def source_value(symbol, role):
    start, end = RUNNER.EXPECTED_INTERVALS[role]
    folder = "spike_learned_transfer" if role == "old" else "spike_native300_recent"
    missing = 0 if role == "old" else 1
    return {"path": f"data/{folder}/{symbol.lower()}_m1_180d_clean.csv",
            "manifest": f"data/{folder}/{symbol}_normalization_manifest.json",
            "sha256": f"{symbol}_{role}_prices", "manifest_sha256": f"{symbol}_{role}_manifest",
            "first_epoch": start.value // 1_000_000_000,
            "last_epoch": end.value // 1_000_000_000 - 60,
            "rows": 259200 - missing, "expected_grid_rows": 259200,
            "declared_missing_minutes": missing, "declared_gaps": [], "excluded_raw_rows": [],
            "from_utc": str(start), "to_exclusive_utc": str(end)}


def sources():
    return {symbol: {role: source_value(symbol, role) for role in ("old", "fresh")}
            for symbol in RUNNER.SYMBOLS}


def split_for(role):
    start, end = RUNNER.EXPECTED_INTERVALS[role]
    span = end - start
    bounds = {q: start + span * q for q in (.4, .5, .6, .7)}
    return {"development": (start, bounds[.7]), "final_test": (bounds[.7], end),
            "train40": (start, bounds[.4]), "train50": (start, bounds[.5]),
            "train60": (start, bounds[.6]), "wf1": (bounds[.4], bounds[.5]),
            "wf2": (bounds[.5], bounds[.6]), "wf3": (bounds[.6], bounds[.7])}


def metric():
    return {"completed": 1250, "active_days": 90, "censored": 0, "invalid_uncensored": 0,
            "selection_score": .1, "mean_net_R": .2, "profit_factor": 1.8,
            "p": .001, "day_undefined_replicates": 0, "weekly_undefined_replicates": 0,
            "day_profit_factor_ci95": [1.2, 2.], "weekly_profit_factor_ci95": [1.1, 2.],
            "mean_net_R_ci95": [.1, .3], "weekly_mean_net_R_ci95": [.05, .35],
            "baseline_difference_ci95": [.02, .2], "weekly_difference_ci95": [.01, .2],
            "closed_trade_max_drawdown": .05, "equity_ruin": False}


def result_row(mode):
    return {"mode": mode, "config": asdict(RUNNER.CFG), "metrics": metric(),
            "audit": {"missing_entry": 0}, "baseline": metric(), "baseline_audit": {},
            "tail": {}, "thirds": [{"metrics": metric()} for _ in range(3)],
            "sensitivities": [{"fill_mode": "adverse_extreme", "entry_delay_minutes": 1,
                               "round_trip_cost_atr": .2, "mean_net_R": .1}]}


def candidate(mode, eligible=True, score=.1):
    return {"mode": mode, "development_eligible": eligible,
            "validation": {"selection_score": score},
            "final_model": {"feature_names": ["first"], "mode_fixture": mode}}


def freeze_fixture(tmp_path, monkeypatch, eligible=False):
    """Create tiny source-byte fixtures and a real hashed release manifest."""
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture_source": "unchanged"})
    values = sources()
    for symbol, pair in values.items():
        for role, value in pair.items():
            price = tmp_path / value["path"]
            price.parent.mkdir(parents=True, exist_ok=True)
            price.write_text(f"fixture_only,{symbol},{role}\n")
            manifest = tmp_path / value["manifest"]
            manifest.write_text(json.dumps({"fixture": True, "symbol": symbol, "role": role}))
            value["sha256"] = RUNNER.digest(price)
            value["manifest_sha256"] = RUNNER.digest(manifest)
    output = tmp_path / "study"
    output.mkdir()
    declaration = {"code_hashes": RUNNER.hashes(), "sources": values,
                   "intervals": {symbol: RUNNER.validate_intervals(pair["old"], pair["fresh"])
                                 for symbol, pair in values.items()},
                   "config": asdict(RUNNER.CFG), "feature_formulas": {"first": "fixture"},
                   "bootstrap_repeats": RUNNER.BOOTSTRAP_REPEATS}
    RUNNER.save(output / "declaration.json", declaration)
    declared_hash = RUNNER.digest(output / "declaration.json")
    (output / "declaration.sha256").write_text(declared_hash + "\n")
    selection = {**declaration, "declaration_sha256": declared_hash, "symbols": {
        symbol: {"selected_direction": "SPIKE" if eligible else None,
                 "models": {mode: candidate(mode, eligible=eligible) for mode in RUNNER.MODES}}
        for symbol in RUNNER.SYMBOLS}}
    RUNNER.save(output / "selection.json", selection)
    (output / "selection.sha256").write_text(RUNNER.digest(output / "selection.json") + "\n")
    return output, values


def test_intervals_are_exact_adjacent_old_to_new_with_known_gaps():
    pair = sources()["BOOM300N"]
    result = RUNNER.validate_intervals(pair["old"], pair["fresh"])
    assert result["old_end"] == result["fresh_start"]
    assert result["overlap_minutes"] == 0
    assert result["chronological_old_to_new"]
    assert pair["fresh"]["declared_missing_minutes"] == 1


@pytest.mark.parametrize("change, message", [
    ("overlap", "overlap"), ("gap", "adjacent"), ("wrong_span", "declared"),
])
def test_wrong_temporal_intervals_are_rejected(change, message):
    pair = sources()["BOOM300N"]
    if change == "overlap":
        pair["fresh"]["first_epoch"] -= 60
    elif change == "gap":
        pair["fresh"]["first_epoch"] += 60
    else:
        pair["old"]["first_epoch"] -= 60
    with pytest.raises(ValueError, match=message):
        RUNNER.validate_intervals(pair["old"], pair["fresh"])


def test_development_never_prepares_fresh_or_fits_old_final30(monkeypatch, tmp_path):
    values = sources()
    prepared_paths, replay_windows, fit_windows = [], [], []
    output = tmp_path / "study"
    output.mkdir()
    monkeypatch.setattr(RUNNER, "source", lambda folder, symbol: deepcopy(
        values[symbol]["old" if folder.endswith("spike_learned_transfer") else "fresh"]))
    monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture_source": "unchanged"})
    monkeypatch.setattr(RUNNER, "verify_prices", lambda value: None)

    def prepare(path, symbol):
        assert path.parent.name == "spike_learned_transfer"
        assert (output / "declaration.json").exists()
        assert (output / "declaration.sha256").exists()
        assert not (output / "selection.json").exists()
        prepared_paths.append(path)
        start, end = RUNNER.EXPECTED_INTERVALS["old"]
        m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
        return m1, pd.DataFrame(), ["first"], {
            "sha256": values[symbol]["old"]["sha256"], "missing_minutes": 0}, split_for("old")

    def replay(m1, signals, cfg, start, end, purge_minutes):
        assert cfg == RUNNER.CFG and purge_minutes == 31
        replay_windows.append((start, end))
        assert end <= split_for("old")["development"][1]
        return pd.DataFrame(columns=["signal_time"]), {}

    def fit(rows, labels, names, start, end):
        fit_windows.append((start, end))
        assert end <= split_for("old")["development"][1]
        return {"feature_names": names, "training_start": start, "training_end": end,
                "threshold": 0., "fixture": True}

    monkeypatch.setattr(RUNNER, "prepare", prepare)
    monkeypatch.setattr(RUNNER, "issued", lambda *args: [])
    monkeypatch.setattr(RUNNER, "replay_timed", replay)
    monkeypatch.setattr(RUNNER, "fit_at", fit)
    monkeypatch.setattr(RUNNER, "summarize", lambda *args, **kwargs: metric())
    RUNNER.develop(SimpleNamespace(output=output))
    assert len(prepared_paths) == 2
    assert len(replay_windows) == 4 * 4
    assert len(fit_windows) == 4 * 4
    assert set(fit_windows) == {split_for("old")[name] for name in (
        "train40", "train50", "train60", "development")}
    saved = json.loads((output / "selection.json").read_text())
    assert all(saved["symbols"][symbol]["selected_direction"] == "DRIFT" for symbol in RUNNER.SYMBOLS)
    assert saved["fresh_features_evaluated"] is saved["fresh_payoffs_evaluated"] is False
    assert saved["declaration_sha256"] == RUNNER.digest(output / "declaration.json")
    assert (output / "selection.sha256").read_text().strip() == RUNNER.digest(output / "selection.json")


def test_empty_eligibility_keeps_null_direction_without_fallback():
    candidates = {"SPIKE": candidate("SPIKE", False, 100), "DRIFT": candidate("DRIFT", False, 200)}
    assert RUNNER.choose_direction(candidates) is None
    candidates["SPIKE"] = candidate("SPIKE", True, .1)
    assert RUNNER.choose_direction(candidates) == "SPIKE"
    candidates["DRIFT"] = candidate("DRIFT", True, .1)
    assert RUNNER.choose_direction(candidates) == "DRIFT"


@pytest.mark.parametrize("tamper, message", [
    ("selection", "selection"), ("declaration", "declaration"),
    ("code", "protocol/source"), ("fresh_prices", "price source"),
    ("old_prices", "price source"), ("fresh_manifest", "manifest"),
])
def test_release_checks_every_integrity_layer_before_preparing_features(monkeypatch, tmp_path, tamper, message):
    output, values = freeze_fixture(tmp_path, monkeypatch)
    if tamper in ("selection", "declaration"):
        path = output / f"{tamper}.json"
        path.write_text(path.read_text() + " ")
    elif tamper == "code":
        monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture_source": "changed"})
    else:
        role = "old" if tamper == "old_prices" else "fresh"
        key = "manifest" if tamper == "fresh_manifest" else "path"
        (tmp_path / values["BOOM300N"][role][key]).write_text("changed fixture")
    def forbidden(*args):
        raise AssertionError("Features must not be prepared before immutable release validation")
    monkeypatch.setattr(RUNNER, "prepare", forbidden)
    with pytest.raises(ValueError, match=message):
        RUNNER.evaluate(SimpleNamespace(output=output, bootstrap=9999))


def test_evaluation_needs_a_frozen_selection_before_any_feature_preparation(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError("No frozen selection: no feature preparation")
    monkeypatch.setattr(RUNNER, "prepare", forbidden)
    with pytest.raises(FileNotFoundError):
        RUNNER.evaluate(SimpleNamespace(output=tmp_path, bootstrap=9999))


def test_develop_cannot_reselect_after_results_even_if_selection_is_removed(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text('{"fixture":true}')
    def forbidden(*args):
        raise AssertionError("Post-result selection must stop before any sources are opened")
    monkeypatch.setattr(RUNNER, "source", forbidden)
    with pytest.raises(ValueError, match="post-result"):
        RUNNER.develop(SimpleNamespace(output=tmp_path))


def test_existing_declaration_and_final_results_are_never_overwritten(monkeypatch, tmp_path):
    output, _ = freeze_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="overwrite"):
        RUNNER.develop(SimpleNamespace(output=output))
    original = (output / "selection.json").read_bytes()
    (output / "results.json").write_text('{"fixture":true}')
    with pytest.raises(ValueError, match="overwrite"):
        RUNNER.evaluate(SimpleNamespace(output=output, bootstrap=9999))
    assert (output / "selection.json").read_bytes() == original


def test_evaluation_preserves_models_and_all_four_hypotheses_without_reselection(monkeypatch, tmp_path):
    output, values = freeze_fixture(tmp_path, monkeypatch, eligible=False)
    selected_bytes = (output / "selection.json").read_bytes()
    verified, prepared, evaluations, family_sizes = [], [], [], []
    original_verify = RUNNER.verify_prices
    original_holm = RUNNER.holm

    def verify(data):
        original_verify(data)
        verified.append(True)

    def prepare(path, symbol):
        assert verified
        role = "old" if path.parent.name == "spike_learned_transfer" else "fresh"
        prepared.append((symbol, role))
        value = values[symbol][role]
        start, end = RUNNER.EXPECTED_INTERVALS[role]
        m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
        return m1, pd.DataFrame(), ["first"], {
            "sha256": value["sha256"], "missing_minutes": value["declared_missing_minutes"]}, split_for(role)

    def evaluate(m1, rows, symbol, mode, frozen, start, end, bootstrap):
        assert frozen == candidate(mode, eligible=False)["final_model"]
        assert bootstrap == 9999
        evaluations.append((symbol, mode, start, end))
        return result_row(mode), pd.DataFrame(), pd.DataFrame()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Evaluation cannot refit models or choose a replacement direction")

    def holm(family):
        family_sizes.append(len(family))
        original_holm(family)

    monkeypatch.setattr(RUNNER, "verify_prices", verify)
    monkeypatch.setattr(RUNNER, "prepare", prepare)
    monkeypatch.setattr(RUNNER, "fit_at", forbidden_fit)
    monkeypatch.setattr(RUNNER, "choose_direction", forbidden_fit)
    monkeypatch.setattr(RUNNER, "evaluate_one", evaluate)
    monkeypatch.setattr(RUNNER, "holm", holm)
    RUNNER.evaluate(SimpleNamespace(output=output, bootstrap=9999))
    assert prepared == [("BOOM300N", "old"), ("BOOM300N", "fresh"),
                        ("CRASH300N", "old"), ("CRASH300N", "fresh")]
    assert len(evaluations) == 8 and family_sizes == [4]
    assert (output / "selection.json").read_bytes() == selected_bytes
    result = json.loads((output / "results.json").read_text())
    for symbol in RUNNER.SYMBOLS:
        payload = result["symbols"][symbol]
        assert payload["selected_direction"] is None
        for mode in RUNNER.MODES:
            fresh = payload["cohorts"]["fresh_temporal180"]["models"][mode]
            assert fresh["user_target_observed"] and not fresh["historical_candidate"]
            assert fresh["selected_direction"] is False
            assert "development_rejected" in fresh["rejection_reasons"]
            assert fresh["metrics"]["holm_p"] == pytest.approx(.004)
            old = payload["cohorts"]["reused_final30"]["models"][mode]
            assert old["historical_candidate"] is False
            assert old["rejection_reasons"] == ["reused_exploratory_cohort"]


def test_primary_bootstrap_count_cannot_be_changed_for_evaluation(monkeypatch, tmp_path):
    output, _ = freeze_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="9999"):
        RUNNER.evaluate(SimpleNamespace(output=output, bootstrap=99))


def test_prepared_gap_audit_must_match_the_frozen_manifest():
    value = source_value("BOOM300N", "fresh")
    start, end = RUNNER.EXPECTED_INTERVALS["fresh"]
    m1 = pd.DataFrame(index=pd.DatetimeIndex([start, end - pd.Timedelta(minutes=1)]))
    audit = {"sha256": value["sha256"], "missing_minutes": 1}
    RUNNER.check_prepared(m1, audit, value)
    audit["missing_minutes"] = 0
    with pytest.raises(ValueError, match="gaps"):
        RUNNER.check_prepared(m1, audit, value)
