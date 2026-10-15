import numpy as np
import pandas as pd
import pytest

from app.research.learned_metrics import profit_factor_inference


START = pd.Timestamp("2026-01-01", tz="UTC")
END = START + pd.Timedelta(days=14)


def ledger(values, days=None, censored=None):
    days = days if days is not None else [0] * len(values)
    entries = [START + pd.Timedelta(days=d, minutes=i + 1) for i, d in enumerate(days)]
    return pd.DataFrame({
        "signal_time": [t - pd.Timedelta(minutes=1) for t in entries],
        "entry_time": entries, "net_R": values,
        "censored": censored if censored is not None else [False] * len(values),
    })


def test_paired_daily_gains_and_losses_preserve_handcomputed_constant_ratio():
    values, days = [], []
    for day in range(14):
        values.extend([1.5 * (day + 1), -(day + 1)])
        days.extend([day, day])
    result = profit_factor_inference(ledger(values, days), START, END, repeats=999)
    assert result["profit_factor"] == 1.5
    assert result["gain_sum_net_R"] == 157.5
    assert result["loss_sum_net_R"] == 105
    assert result["day_profit_factor_ci95"] == pytest.approx([1.5, 1.5])
    assert result["weekly_profit_factor_ci95"] == pytest.approx([1.5, 1.5])
    assert result["day_valid_replicates"] == result["weekly_valid_replicates"] == 999
    assert result["day_undefined_replicates"] == result["weekly_undefined_replicates"] == 0
    assert result["paired_daily_gain_loss_resampling"]


def test_all_calendar_days_including_empty_days_are_resampled():
    result = profit_factor_inference(ledger([2, -1]), START,
                                     START + pd.Timedelta(days=7), repeats=199, seed=12)
    starts = np.random.default_rng(12).integers(0, 7, size=(199, 7))
    expected_valid = int((starts == 0).any(axis=1).sum())
    assert result["calendar_days"] == 7
    assert result["active_days"] == 1
    assert result["day_valid_replicates"] == expected_valid
    assert result["day_undefined_replicates"] == 199 - expected_valid
    assert result["day"]["empty_replicates"] == 199 - expected_valid
    assert result["day_profit_factor_ci95"] == [2, 2]
    # A circular seven-day block on a seven-day calendar contains each day once.
    assert result["weekly_valid_replicates"] == 199
    assert result["weekly_profit_factor_ci95"] == [2, 2]


def test_circular_weekly_blocks_wrap_and_truncate_to_exact_calendar_length():
    gains = np.array([0, 1, 4, 0, 2, 5, 0, 3, 1, 0], dtype=float)
    losses = np.array([3, 0, 1, 4, 0, 2, 1, 0, 3, 2], dtype=float)
    values = np.column_stack([gains, -losses]).ravel().tolist()
    frame = ledger(values, np.repeat(np.arange(10), 2).tolist())
    result = profit_factor_inference(frame, START, START + pd.Timedelta(days=10),
                                     repeats=73, seed=21, block_days=7)
    starts = np.random.default_rng(21).integers(0, 10, size=(73, 2))
    expected = []
    for first, last in starts:
        chosen = np.r_[(first + np.arange(7)) % 10, (last + np.arange(3)) % 10]
        assert len(chosen) == 10
        expected.append(gains[chosen].sum() / losses[chosen].sum())
    assert result["weekly_profit_factor_ci95"] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert result["profit_factor_ci95"] == result["weekly_profit_factor_ci95"]
    assert result["weekly"]["resampled_calendar_days"] == 10
    assert result["weekly"]["resampling"] == "circular_moving_utc_day_blocks"


@pytest.mark.parametrize("values", [[0, 0], [1, 2], []])
def test_zero_no_loss_and_empty_ledgers_keep_pf_and_intervals_unknown(values):
    result = profit_factor_inference(ledger(values), START, END, repeats=99)
    assert result["profit_factor"] is None
    assert result["day_profit_factor_ci95"] == [None, None]
    assert result["weekly_profit_factor_ci95"] == [None, None]
    assert result["day_valid_replicates"] == result["weekly_valid_replicates"] == 0
    assert result["day_undefined_replicates"] == result["weekly_undefined_replicates"] == 99
    assert result["day"]["no_loss_replicates"] == 99
    assert result["weekly"]["inference_status"] == "NO_FINITE_RATIOS"


def test_loss_only_ledgers_have_finite_zero_pf():
    frame = ledger([-1] * 14, list(range(14)))
    result = profit_factor_inference(frame, START, END, repeats=99)
    assert result["profit_factor"] == 0
    assert result["day_profit_factor_ci95"] == result["weekly_profit_factor_ci95"] == [0, 0]
    assert result["day_valid_replicates"] == result["weekly_valid_replicates"] == 99


def test_censored_and_nonfinite_returns_are_explicitly_counted_and_excluded():
    frame = ledger([3, -2, 1000, np.nan, np.inf], censored=[False, False, True, False, False])
    frame.loc[2, "entry_time"] = pd.NaT
    result = profit_factor_inference(frame, START, END, repeats=99)
    assert result["trades"] == 5
    assert result["completed"] == 2
    assert result["censored"] == 1
    assert result["invalid_uncensored"] == 2
    assert result["profit_factor"] == 1.5
    assert result["gain_sum_net_R"] == 3
    assert result["loss_sum_net_R"] == 2


def test_signal_interval_is_half_open_and_duplicate_dataframe_labels_are_allowed():
    frame = pd.concat([ledger([2, -1]), ledger([100])])
    frame.iloc[2, frame.columns.get_loc("signal_time")] = END
    frame.iloc[2, frame.columns.get_loc("entry_time")] = END
    result = profit_factor_inference(frame, START, END, repeats=99)
    assert frame.index.tolist() == [0, 1, 0]
    assert result["trades"] == result["completed"] == 2
    assert result["profit_factor"] == 2


def test_partial_calendar_days_are_retained():
    start = START + pd.Timedelta(hours=12)
    end = START + pd.Timedelta(days=2, hours=12)
    entries = [start + pd.Timedelta(minutes=1), START + pd.Timedelta(days=1, hours=1),
               START + pd.Timedelta(days=2, hours=1)]
    frame = pd.DataFrame({"signal_time": [t - pd.Timedelta(minutes=1) for t in entries],
                          "entry_time": entries, "net_R": [2, -1, 2], "censored": [False] * 3})
    result = profit_factor_inference(frame, start, end, repeats=99)
    assert result["calendar_days"] == result["active_days"] == 3
    assert result["profit_factor"] == 4
    assert result["weekly"]["requested_block_days"] == 7
    assert result["weekly"]["effective_block_days"] == 3
    assert result["weekly"]["inference_status"] == "INSUFFICIENT_CALENDAR_SPAN"


def test_same_seed_is_reproducible_across_multiple_batches():
    frame = ledger([2, -1, 4, -2, -3, 1], [0, 0, 3, 5, 8, 11])
    first = profit_factor_inference(frame, START, END, repeats=1501, seed=123)
    second = profit_factor_inference(frame, START, END, repeats=1501, seed=123)
    assert first == second
    assert first["day_valid_replicates"] + first["day_undefined_replicates"] == 1501
    assert first["weekly_valid_replicates"] + first["weekly_undefined_replicates"] == 1501


@pytest.mark.parametrize("argument,value", [
    ("repeats", 0), ("repeats", -1), ("repeats", True), ("repeats", 2.5),
    ("block_days", 0), ("block_days", -1), ("block_days", True), ("block_days", 1.5),
    ("seed", -1), ("seed", True), ("seed", 2.5),
])
def test_invalid_bootstrap_arguments_raise(argument, value):
    with pytest.raises(ValueError):
        profit_factor_inference(pd.DataFrame(), START, END, **{argument: value})


@pytest.mark.parametrize("end", [START, START - pd.Timedelta(minutes=1), pd.NaT])
def test_invalid_analysis_boundaries_raise(end):
    with pytest.raises(ValueError):
        profit_factor_inference(pd.DataFrame(), START, end, repeats=9)


@pytest.mark.parametrize("field,value,expected", [
    ("entry_time", START - pd.Timedelta(minutes=1), "outside"),
    ("entry_time", END, "outside"),
    ("entry_time", pd.NaT, "valid entry"),
    ("signal_time", pd.NaT, "signal timestamps"),
    ("censored", None, "censor flags"),
])
def test_invalid_completed_trade_fields_raise(field, value, expected):
    frame = ledger([2, -1])
    if field == "censored":
        frame[field] = frame[field].astype("boolean")
    frame.loc[0, field] = value
    with pytest.raises(ValueError, match=expected):
        profit_factor_inference(frame, START, END, repeats=9)


def test_entry_before_signal_and_missing_schema_raise():
    frame = ledger([2, -1])
    frame.loc[0, "entry_time"] = START
    frame.loc[0, "signal_time"] = START + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="precede"):
        profit_factor_inference(frame, START, END, repeats=9)
    with pytest.raises(ValueError, match="Missing profit-factor fields"):
        profit_factor_inference(pd.DataFrame({"net_R": [1]}), START, END, repeats=9)


def test_string_censor_flags_do_not_silently_discard_completed_trades():
    frame = ledger([2, -1])
    frame["censored"] = ["False", "False"]
    with pytest.raises(ValueError, match="boolean"):
        profit_factor_inference(frame, START, END, repeats=9)


def test_generic_selected_block_alias_leaves_day_and_weekly_outputs_available():
    frame = ledger([2, -1] * 14, np.repeat(np.arange(14), 2).tolist())
    result = profit_factor_inference(frame, START, END, repeats=99, block_days=3)
    assert result["block_days"] == result["selected"]["requested_block_days"] == 3
    assert result["selected"]["resampled_calendar_days"] == 14
    assert result["profit_factor_ci95"] == result["day_profit_factor_ci95"] == [2, 2]
    assert result["weekly_profit_factor_ci95"] == [2, 2]
