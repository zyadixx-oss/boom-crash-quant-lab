import numpy as np
import pandas as pd
import pytest

from app.research.payoff_metrics import paired_inference, summarize


START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-01-04", tz="UTC")


def trades(values, days=None, gross=None, censored=None):
    n = len(values)
    days = days if days is not None else [i % 3 for i in range(n)]
    index = [START + pd.Timedelta(days=day, minutes=5 * i + 1) for i, day in enumerate(days)]
    return pd.DataFrame({
        "signal_time": [t - pd.Timedelta(minutes=1) for t in index],
        "entry_time": index, "exit_time": [t + pd.Timedelta(minutes=5) for t in index],
        "entry": [100.0] * n, "exit": [100.0] * n, "atr": [1.0] * n,
        "gross_R": gross if gross is not None else values, "net_R": values,
        "reason": ["TIMEOUT"] * n, "ambiguous": [False] * n,
        "holding_minutes": [5.0] * n,
        "censored": censored if censored is not None else [False] * n,
    })


def test_net_metrics_and_break_even_cost_use_correct_atr_and_stop_units():
    frame = trades([-1.2, 1.8, 0.3], gross=[-1, 2, 0.5])
    result = summarize(frame, START, END, stop_atr=0.5)
    assert result["completed"] == 3
    assert result["wins"] == 2
    assert result["win_rate"] == 2 / 3
    assert result["mean_net_R"] == pytest.approx(0.3)
    assert result["sum_net_R"] == pytest.approx(0.9)
    assert result["mean_gross_R"] == pytest.approx(0.5)
    assert result["break_even_cost_atr"] == pytest.approx(0.25)
    assert result["profit_factor"] == pytest.approx(1.75)
    assert result["average_win_R"] == pytest.approx(1.05)
    assert result["average_loss_R"] == pytest.approx(-1.2)


def test_day_cluster_standard_error_uses_trade_weighted_mean_and_empty_days():
    frame = trades([1.0] * 10 + [-2.0], days=[0] * 10 + [1])
    result = summarize(frame, START, END, stop_atr=1)
    mean = 8 / 11
    residuals = np.array([10 - 10 * mean, -2 - mean, 0])
    se = np.sqrt(3 / 2 * np.sum(residuals ** 2)) / 11
    assert result["active_days"] == 2
    assert result["calendar_days"] == 3
    assert result["mean_net_R"] == pytest.approx(mean)
    assert result["mean_net_R"] != pytest.approx((1 - 2) / 2)
    assert result["cluster_se"] == pytest.approx(se)
    assert result["selection_score"] == pytest.approx(mean - 1.96 * se)


def test_identical_model_and_baseline_have_zero_paired_difference_in_every_draw():
    frame = trades([2.0, -1.0, 3.0, -0.5, 1.0], days=[0, 0, 1, 2, 2])
    result = paired_inference(frame, frame.copy(), START, END, repeats=9999)
    assert result["baseline_difference"] == 0
    assert result["baseline_difference_ci95"] == [0, 0]
    assert result["difference_p"] == 1
    assert result["p"] == 1
    assert result["bootstrap_difference_valid_replicates"] == 9999


def test_zero_returns_and_empty_frames_never_fabricate_pf_or_evidence():
    zero = trades([0.0, 0.0, 0.0])
    summary = summarize(zero, START, END, stop_atr=1)
    assert summary["profit_factor"] is None
    assert summary["average_win_R"] is None
    assert summary["average_loss_R"] is None
    assert summary["win_rate"] == 0
    assert summary["closed_trade_return"] == 0
    inference = paired_inference(zero, zero, START, END, repeats=99)
    assert inference["mean_net_R_ci95"] == [0, 0]
    assert inference["mean_p"] == inference["difference_p"] == inference["p"] == 1
    empty = summarize(pd.DataFrame(), START, END, stop_atr=1)
    assert empty["completed"] == 0
    assert empty["mean_net_R"] is None
    assert empty["selection_score"] is None
    unknown = paired_inference(pd.DataFrame(), zero, START, END, repeats=99)
    assert unknown["mean_net_R_ci95"] == [None, None]
    assert unknown["baseline_difference"] is None
    assert unknown["p"] == 1


def test_censored_and_nonfinite_rows_are_counted_but_excluded_from_expectancy():
    frame = trades([1.0, -1.0, 1000.0, np.nan, np.inf],
                   censored=[False, False, True, False, False])
    frame.loc[2, "ambiguous"] = True
    result = summarize(frame, START, END, stop_atr=1)
    assert result["trades"] == 5
    assert result["completed"] == 2
    assert result["censored"] == 1
    assert result["invalid_uncensored"] == 2
    assert result["ambiguous"] == 1
    assert result["mean_net_R"] == 0
    assert result["mean_gross_R"] == 0
    inference = paired_inference(frame, trades([1.0, -1.0]), START, END, repeats=99)
    assert inference["completed"] == 2
    assert inference["baseline_difference"] == 0


def test_equity_ruin_is_absorbing_and_drawdown_includes_initial_capital():
    frame = trades([-500.0, 1000.0], days=[0, 1])
    result = summarize(frame, START, END, stop_atr=1, risk_fraction=0.0025)
    assert result["equity_ruin"]
    assert result["closed_trade_return"] == -1
    assert result["closed_trade_max_drawdown"] == 1
    normal = summarize(trades([-1.0, 2.0], days=[0, 1]), START, END, stop_atr=1)
    assert normal["closed_trade_return"] == pytest.approx((1 - 0.0025) * (1 + 0.005) - 1)
    assert normal["closed_trade_max_drawdown"] == pytest.approx(0.0025)


def test_cohort_filter_removes_trades_outside_signal_interval():
    frame = trades([1.0, 100.0], days=[0, 1])
    frame.loc[1, ["signal_time", "entry_time", "exit_time"]] = START - pd.Timedelta(days=1)
    result = summarize(frame, START, END, stop_atr=1)
    assert result["trades"] == 1
    assert result["mean_net_R"] == 1


def test_no_loss_pf_remains_unknown_and_single_day_se_is_unavailable():
    frame = trades([1.0, 2.0], days=[0, 0])
    result = summarize(frame, START, START + pd.Timedelta(days=1), stop_atr=1)
    assert result["profit_factor"] is None
    assert result["cluster_se"] is None
    assert result["selection_score"] is None


def test_positive_constant_difference_uses_plus_one_bootstrap_correction():
    result = paired_inference(trades([1.0, 1.0, 1.0]), trades([0.0, 0.0, 0.0]),
                              START, END, repeats=99)
    assert result["mean_p"] == pytest.approx(0.01)
    assert result["difference_p"] == pytest.approx(0.01)
    assert result["p"] == pytest.approx(0.01)


def test_concatenated_trade_frames_need_not_have_unique_dataframe_row_labels():
    frame = pd.concat([trades([1.0], days=[0]), trades([-1.0], days=[1])])
    assert frame.index.tolist() == [0, 0]
    result = summarize(frame, START, END, stop_atr=1)
    assert result["completed"] == 2
    assert result["mean_net_R"] == 0
    assert result["active_days"] == 2
