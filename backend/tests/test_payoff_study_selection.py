"""Selection and evaluation lifecycle tests use fixtures, never new payoff data."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("payoff_study_runner_tests", ROOT / "scripts/run_spike_payoff_study.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def metric(score=.25, completed=400, active_days=60, censored=0, mean=.50):
    return {
        "selection_score": score, "completed": completed, "active_days": active_days,
        "censored": censored, "mean_net_R": mean, "profit_factor": 2.0,
        "mean_net_R_ci95": [.20, .80], "baseline_difference_ci95": [.10, .40],
        "p": .01, "closed_trade_max_drawdown": .02, "win_rate": .60,
        "closed_trade_return": .40, "baseline_difference": .25,
        "break_even_cost_atr": .60, "ambiguous": 0,
    }


def candidate(name, score=.25, completed=400):
    config = RUNNER.configuration()
    return {
        "id": name, "variant": "SR_ALIGNMENT", "config": dict(config.__dict__),
        "parts": {part: metric(score=score, completed=completed) for part in RUNNER.PARTS},
    }


def test_selection_is_invariant_to_external_and_final30_results():
    rows = [candidate("winner", .30), candidate("runner_up", .20)]
    before = RUNNER.choose(rows, "development", robust=True)
    changed = deepcopy(rows)
    changed[0]["parts"].update(final_test=metric(-999, mean=-999), external_older_replication=metric(-999, mean=-999))
    changed[1]["parts"].update(final_test=metric(999, mean=999), external_older_replication=metric(999, mean=999))
    after = RUNNER.choose(changed, "development", robust=True)
    assert before["id"] == after["id"] == "winner"
    assert before["selection_score"] == after["selection_score"]
    assert before["development_eligible"] is after["development_eligible"] is True


@pytest.mark.parametrize("training", ["train40", "train50", "train60"])
def test_walk_forward_selection_ignores_future_validation_results(training):
    rows = [candidate("past_winner", .30), candidate("future_winner", .20)]
    selected = RUNNER.choose(rows, training)
    # Future fold outcomes are deliberately reversed, without touching training.
    for part in ("wf1", "wf2", "wf3"):
        rows[0]["parts"][part] = metric(-1000, completed=0, mean=-1000)
        rows[1]["parts"][part] = metric(1000, completed=10000, mean=1000)
    again = RUNNER.choose(rows, training)
    assert selected["id"] == again["id"] == "past_winner"


def test_window_enforces_common31_minute_purge_even_for_short_trades():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = start + pd.Timedelta(hours=1)
    # The last trade would finish well before partition end on its own, but
    # must remain excluded by the common 31-minute purge across bracket choices.
    trades = pd.DataFrame({
        "label": ["before_start", "boundary_allowed", "short_but_purged"],
        "signal_time": [start - pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=29),
                        start + pd.Timedelta(minutes=30)],
        "exit_time": [start, start + pd.Timedelta(minutes=35), start + pd.Timedelta(minutes=36)],
    })
    result = RUNNER.window(trades, start, end)
    assert result.label.tolist() == ["boundary_allowed"]


def test_eligible_candidate_beats_higher_scoring_ineligible_diagnostic():
    eligible = candidate("eligible", .10)
    ineligible = candidate("small_sample", 1000, completed=299)
    selected = RUNNER.choose([ineligible, eligible], "development", robust=True)
    assert selected["id"] == "eligible"
    assert selected["development_eligible"] is True


def test_fallback_is_diagnostic_and_cannot_be_promoted_by_external_success():
    rows = [candidate("fallback", .25, completed=299), candidate("other", .10, completed=299)]
    selected = RUNNER.choose(rows, "development", robust=True)
    assert selected["id"] == "fallback"
    assert selected["development_eligible"] is False
    external = {"metrics": {**metric(), "holm_p": .01}}
    passed, reasons = RUNNER.gate(external, selected)
    assert not passed
    assert "development_selection_rejected" in reasons


def test_development_never_replays_external_or_final30(monkeypatch, tmp_path):
    start = pd.Timestamp("2026-04-07T11:08:00Z")
    duration = pd.Timedelta(days=180)
    split = {
        "development": (start, start + duration * .70),
        "final_test": (start + duration * .70, start + duration),
        "train40": (start, start + duration * .40),
        "train50": (start, start + duration * .50),
        "train60": (start, start + duration * .60),
        "wf1": (start + duration * .40, start + duration * .50),
        "wf2": (start + duration * .50, start + duration * .60),
        "wf3": (start + duration * .60, start + duration * .70),
    }
    prepared_paths, replay_windows = [], []

    def fake_prepare(path, symbol):
        prepared_paths.append(path)
        assert path.parent.name == "spike_hunter"
        return pd.DataFrame(), {"sha256": "old_frozen", "from_utc": str(start)}, {name: [] for name in RUNNER.VARIANTS}, split

    def fake_replay(m1, issued, config, replay_start, replay_end, purge_minutes):
        replay_windows.append((replay_start, replay_end, purge_minutes))
        return pd.DataFrame(columns=["signal_time"]), {}

    monkeypatch.setattr(RUNNER, "prepare", fake_prepare)
    monkeypatch.setattr(RUNNER, "replay_brackets", fake_replay)
    monkeypatch.setattr(RUNNER, "summarize", lambda *a, **kw: metric())
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: {flag: False for flag in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")})
    monkeypatch.setattr(RUNNER, "hashes", lambda: {"fixture": "frozen"})
    RUNNER.develop(SimpleNamespace(output=tmp_path))
    assert len(prepared_paths) == 2
    assert len(replay_windows) == 2 * 432
    assert set(replay_windows) == {(split["development"][0], split["development"][1], 31)}
    assert (tmp_path / "selection.sha256").read_text().strip() == hashlib.sha256((tmp_path / "selection.json").read_bytes()).hexdigest()


def test_identical_primary_selected_is_one_holm_hypothesis_with_independent_gates(monkeypatch, tmp_path):
    old_start = pd.Timestamp("2026-04-07T11:08:00Z")
    old_end = old_start + pd.Timedelta(days=180)
    config = RUNNER.configuration()
    chosen = candidate(RUNNER.key("SR_ALIGNMENT", config))
    primary = deepcopy(chosen)
    selected = {**deepcopy(chosen), "development_eligible": False,
                "training_partition": "development", "selection_score": .25}
    frozen = {
        "code_hashes": {"fixture": "frozen"},
        "symbols": {"BOOM500": {"audit": {"sha256": "old_frozen", "from_utc": str(old_start)},
                                  "primary": primary, "selected": selected, "walk_forward": []}},
    }
    RUNNER.save(tmp_path / "selection.json", frozen)
    (tmp_path / "selection.sha256").write_text(hashlib.sha256((tmp_path / "selection.json").read_bytes()).hexdigest() + "\n")
    evaluate_calls, holm_counts = [], []

    def fake_prepare(path, symbol):
        if path.parent.name == "spike_hunter":
            index = pd.DatetimeIndex([old_start, old_end - pd.Timedelta(minutes=1)])
            audit = {"sha256": "old_frozen", "from_utc": str(old_start)}
        else:
            index = pd.DatetimeIndex([old_start - pd.Timedelta(days=180), old_start - pd.Timedelta(minutes=1)])
            audit = {"sha256": "external_fixture", "from_utc": str(index[0])}
        frame = pd.DataFrame(index=index)
        return frame, audit, {"SR_ALIGNMENT": [], "CLOCK_BASELINE": []}, {"final_test": (old_start + pd.Timedelta(days=126), old_end)}

    def fake_evaluate(m1, issued, variant, cfg, start, end, bootstrap):
        evaluate_calls.append((start, end, variant))
        return {"metrics": metric(), "audit": {}, "baseline": {}, "baseline_audit": {}}, pd.DataFrame()

    def fake_replay(*a, **kw):
        return pd.DataFrame({"gross_R": pd.Series([], dtype=float)}), {}

    actual_holm = RUNNER.holm

    def record_holm(rows):
        holm_counts.append(len(rows))
        actual_holm(rows)

    monkeypatch.setattr(RUNNER, "SYMBOLS", ("BOOM500",))
    monkeypatch.setattr(RUNNER, "prepare", fake_prepare)
    monkeypatch.setattr(RUNNER, "evaluate_config", fake_evaluate)
    monkeypatch.setattr(RUNNER, "replay_brackets", fake_replay)
    monkeypatch.setattr(RUNNER, "summarize", lambda *a, **kw: metric())
    monkeypatch.setattr(RUNNER, "holm", record_holm)
    monkeypatch.setattr(RUNNER, "hashes", lambda: frozen["code_hashes"])
    monkeypatch.setattr(RUNNER, "assert_offline", lambda: {flag: False for flag in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")})
    RUNNER.evaluate(SimpleNamespace(output=tmp_path, bootstrap=99))
    result = json.loads((tmp_path / "results.json").read_text())
    models = result["symbols"]["BOOM500"]["cohorts"]["external_older_replication"]["models"]
    assert len(evaluate_calls) == 2  # One unique model in each cohort.
    assert holm_counts == [1]
    assert models["primary"]["metrics"]["holm_p"] == models["selected"]["metrics"]["holm_p"] == .01
    assert models["primary"]["forward_research_candidate"] is True
    assert models["selected"]["forward_research_candidate"] is False
    assert "development_selection_rejected" in models["selected"]["gate_reasons"]
