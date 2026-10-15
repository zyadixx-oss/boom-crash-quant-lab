"""Selection, transfer-window and inference safeguards, using fixtures only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("timed_study_tests", ROOT / "scripts/run_spike_timed_study.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def candidate(name, n=400, score=.2, fold_mean=.1):
    parts = {part: {"completed": n, "active_days": 60, "censored": 0,
                    "invalid_uncensored": 0, "selection_score": score, "mean_net_R": fold_mean}
             for part in RUNNER.PARTS}
    return {"id": name, "variant": "CRT", "mode": "SPIKE", "config": {}, "parts": parts}


def ledger(times, values=None):
    n = len(times)
    values = np.ones(n) if values is None else np.asarray(values, dtype=float)
    return pd.DataFrame({"signal_time": times, "entry_time": times + pd.Timedelta(minutes=1),
                         "exit_time": times + pd.Timedelta(minutes=2), "gross_R": values + .05,
                         "net_R": values, "censored": False, "reason": "time", "ambiguous": False,
                         "holding_minutes": 1., "atr": 2.})


def test_high_score_tiny_sample_is_never_fallback():
    picked = RUNNER.choose([candidate("tiny", 1, 100), candidate("adequate", 400, -.1, -.1)],
                           "development", robust=True)
    assert picked["id"] == "adequate" and not picked["development_eligible"]
    assert RUNNER.choose([candidate("tiny", 1, 100)], "development", robust=True) is None


def test_selection_does_not_use_transfer_or_recent30_fields():
    rows = [candidate("past_winner", score=.3), candidate("other", score=.2)]
    before = RUNNER.choose(rows, "development", robust=True)
    rows[0]["transfer"] = {"mean_net_R": -999}
    rows[1]["transfer"] = {"mean_net_R": 999}
    rows[1]["parts"]["recent30"] = {"mean_net_R": 999}
    after = RUNNER.choose(rows, "development", robust=True)
    assert before == after


def test_walkforward_choice_ignores_its_future_folds():
    rows = [candidate("past_winner", score=.3), candidate("other", score=.2)]
    for name in ("wf1", "wf2", "wf3"):
        rows[0]["parts"][name]["mean_net_R"] = -100
        rows[1]["parts"][name]["mean_net_R"] = 100
    assert RUNNER.choose(rows, "train40")["id"] == "past_winner"


def test_inverse_direction_preserves_signal_time_and_atr():
    row = {"signal_time": pd.Timestamp("2026-01-01T00:00Z"), "atr": 2., "side": 1, "variant": "CRT"}
    result = RUNNER.issued_for({"CRT": [row]}, "CRT", "DRIFT")[0]
    assert result["side"] == -1 and result["signal_time"] == row["signal_time"] and result["atr"] == 2.
    assert row["side"] == 1


def test_weekly_identical_paths_never_establish_excess():
    start = pd.Timestamp("2026-01-01T00:00Z")
    trades = ledger(pd.date_range(start, periods=21, freq="D"))
    result = RUNNER.weekly_inference(trades, trades, start, start + pd.Timedelta(days=21), repeats=199)
    assert result["weekly_mean_net_R_ci95"] == [1., 1.]
    assert result["weekly_difference_ci95"] == [0., 0.]
    assert result["weekly_p"] == 1.


def test_weekly_negative_mean_cannot_pass_despite_baseline_advantage():
    start = pd.Timestamp("2026-01-01T00:00Z")
    times = pd.date_range(start, periods=21, freq="D")
    result = RUNNER.weekly_inference(ledger(times, [-1.] * 21), ledger(times, [-2.] * 21),
                                     start, start + pd.Timedelta(days=21), repeats=199)
    assert result["weekly_difference_ci95"] == [1., 1.]
    assert result["weekly_p"] == 1.


def test_tail_concentration_and_raw_quote_units_for_constant_path():
    start = pd.Timestamp("2026-01-01T00:00Z")
    trades = ledger(pd.date_range(start, periods=12, freq="D"))
    result = RUNNER.tail_metrics(trades, start, start + pd.Timedelta(days=13), 2., .10)
    assert result["winner_effective_n"] == 12
    assert result["largest_winner_positive_profit_share"] == 1 / 12
    assert result["mean_R_without_best5_trades"] == result["mean_R_without_best5_days"] == 1.
    assert result["worst_leave_one_day_out_mean_R"] == 1.
    assert np.isclose(result["raw_mean_after_modeled_cost_points"], 4.)


def test_thirds_use31minute_end_purge_at_each_boundary():
    start = pd.Timestamp("2026-01-01T00:00Z")
    times = pd.DatetimeIndex([start + pd.Timedelta(minutes=i) for i in (29, 30, 89, 90, 149, 150)])
    rows = RUNNER.thirds(ledger(times), start, start + pd.Timedelta(hours=3), 2.)
    assert [r["metrics"]["completed"] for r in rows] == [1, 1, 1]
    assert rows[0]["end"] == rows[1]["start"] == start + pd.Timedelta(hours=1)
