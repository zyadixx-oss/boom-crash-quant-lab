"""Synthetic-only causal600-increment feature and common44/45 join challenges."""
import math

import numpy as np
import pandas as pd
import pytest

from app.research.multiframe_signal import FEATURE_NAMES as BASE_NAMES
from app.research.tick_path_risk import (
    FEATURE_NAMES, FEATURE_FORMULAS, PATH_FEATURE_NAME, PATH_FEATURE_NAMES,
    REQUIRED_QUOTES, WINDOW_INCREMENTS, describe_tick_path_risk, join_tick_path_risk_inputs,
)


def prices(returns, side=1, start="2026-01-01T00:00:00Z"):
    quotes = np.r_[100., 100. * np.exp(np.cumsum(np.asarray(returns) * side))]
    index = pd.date_range(pd.Timestamp(start), periods=len(quotes), freq="s").as_unit("ns")
    return pd.DataFrame({"quote": quotes}, index=index)


def constant(n=601):
    return prices(np.zeros(n - 1))


def inputs(n=4):
    index = pd.date_range("2026-01-01T00:00Z", periods=n, freq="5min").as_unit("ns")
    frame = pd.DataFrame({name: np.arange(n, dtype=float) + i / 100 for i, name in enumerate(BASE_NAMES)}, index=index)
    frame["feature_valid"] = True
    frame["h4_closed_at"] = index.floor("4h")
    return frame


def descriptor(n=1501, side=1, scale=.001):
    return describe_tick_path_risk(prices(np.tile([-.002, .001], n // 2)[:n - 1]), side, scale)


def join(frame=None, described=None, names=None):
    return join_tick_path_risk_inputs(inputs() if frame is None else frame,
                                     list(BASE_NAMES) if names is None else names,
                                     descriptor() if described is None else described)


def test_exact_single_feature_formula_and_constants():
    assert WINDOW_INCREMENTS == 600 and REQUIRED_QUOTES == 601
    assert PATH_FEATURE_NAMES == ("native_adverse_semivariance600",)
    assert list(FEATURE_FORMULAS) == [PATH_FEATURE_NAME]
    assert FEATURE_NAMES == (*BASE_NAMES, PATH_FEATURE_NAME) and len(FEATURE_NAMES) == 45


@pytest.mark.parametrize("n", (1, 2, 600, 601, 602))
def test_exact601_quote_boundary_includes600_increments_only(n):
    result = describe_tick_path_risk(constant(n), 1, .001)
    valid = max(0, n - 600)
    assert len(result) == n and result.tick_path_feature_valid.sum() == valid
    assert result[PATH_FEATURE_NAME].isna().sum() == min(600, n)
    assert result.loc[result.tick_path_feature_valid, PATH_FEATURE_NAME].eq(0.).all()
    assert result.index[0] == pd.Timestamp("2026-01-01T00:00:01Z")
    if valid:
        assert result.index[600] == pd.Timestamp("2026-01-01T00:10:01Z")


@pytest.mark.parametrize("native_side", (1, -1))
@pytest.mark.parametrize("direction", ("favorable", "adverse"))
def test_constant_native_favorable_and_adverse_paths_are_hand_calculable(native_side, direction):
    increment = .002 if direction == "favorable" else -.002
    result = describe_tick_path_risk(prices(np.full(600, increment), native_side), native_side, .001)
    value = result[PATH_FEATURE_NAME].iloc[-1]
    assert result.tick_path_feature_valid.iloc[-1]
    assert value == pytest.approx(0. if direction == "favorable" else 4., abs=1e-11)


@pytest.mark.parametrize("native_side", (1, -1))
def test_mixed_returns_use_negative_native_semivariance_and_immutable_scale(native_side):
    increments = np.tile([-.001, .002, -.003, 0.], 150)
    ticks = prices(increments, native_side)
    result = describe_tick_path_risk(ticks, native_side, .002)
    expected = ((.001 / .002)**2 + (.003 / .002)**2) / 4
    assert result[PATH_FEATURE_NAME].iloc[-1] == pytest.approx(expected, abs=1e-11)
    doubled = describe_tick_path_risk(ticks, native_side, .004)
    assert doubled[PATH_FEATURE_NAME].iloc[-1] == pytest.approx(expected / 4)


def test_window_drops_an_adverse_increment_at_the_exact600_increment_boundary():
    increments = np.r_[-.01, np.zeros(601)]
    result = describe_tick_path_risk(prices(increments), 1, .001)
    assert result[PATH_FEATURE_NAME].iloc[600] == pytest.approx(100 / 600)
    assert result[PATH_FEATURE_NAME].iloc[601] == pytest.approx(0., abs=1e-15)
    assert result[PATH_FEATURE_NAME].iloc[602] == pytest.approx(0., abs=1e-15)


def test_current_quote_and_current_increment_are_excluded_from_decision_feature():
    ticks = prices(np.full(1000, -.001))
    time = pd.Timestamp("2026-01-01T00:15Z")
    a = describe_tick_path_risk(ticks, 1, .001)
    changed = ticks.copy()
    changed.loc[time, "quote"] = 1e10
    b = describe_tick_path_risk(changed, 1, .001)
    assert a.loc[time, PATH_FEATURE_NAME] == b.loc[time, PATH_FEATURE_NAME]
    pd.testing.assert_frame_equal(a.loc[:time], b.loc[:time])
    assert a.loc[time + pd.Timedelta(seconds=2), PATH_FEATURE_NAME] != b.loc[time + pd.Timedelta(seconds=2), PATH_FEATURE_NAME]


def test_future_mutation_prefix_invariance_and_missing_current_quote_do_not_rewrite_history():
    ticks = prices(np.tile([-.001, .002], 900))
    end = pd.Timestamp("2026-01-01T00:15Z")
    original = describe_tick_path_risk(ticks, 1, .001)
    changed = ticks.copy()
    changed.loc[changed.index >= end, "quote"] = np.linspace(1e3, 1e4, (changed.index >= end).sum())
    mutated = describe_tick_path_risk(changed, 1, .001)
    pd.testing.assert_frame_equal(original.loc[:end], mutated.loc[:end])
    prefix = describe_tick_path_risk(ticks.loc[ticks.index < end], 1, .001)
    pd.testing.assert_frame_equal(original.loc[:end], prefix)
    no_current = describe_tick_path_risk(ticks.drop(end), 1, .001)
    assert end in no_current.index
    assert no_current.loc[end, PATH_FEATURE_NAME] == original.loc[end, PATH_FEATURE_NAME]
    assert end + pd.Timedelta(seconds=1) not in no_current.index


@pytest.mark.parametrize("gap_seconds", (1, 2, 30, 86400))
def test_gap_requires_a_new_complete601_quote_history_and_never_carries(gap_seconds):
    ticks = prices(np.full(1400, -.001))
    last_before = ticks.index[699]
    shifted = pd.DatetimeIndex([t + pd.Timedelta(seconds=gap_seconds) if i >= 700 else t
                               for i, t in enumerate(ticks.index)]).as_unit("ns")
    ticks.index = shifted
    result = describe_tick_path_risk(ticks, 1, .001)
    first_after = ticks.index[700]
    assert result.loc[last_before + pd.Timedelta(seconds=1), "tick_path_feature_valid"]
    after = result.loc[result.index >= first_after + pd.Timedelta(seconds=1)]
    assert after.tick_path_feature_valid.iloc[:600].eq(False).all()
    assert after[PATH_FEATURE_NAME].iloc[:600].isna().all()
    assert after.tick_path_feature_valid.iloc[600]
    assert after[PATH_FEATURE_NAME].iloc[600] == pytest.approx(1., abs=1e-11)
    # Missing target rows are absent, never backfilled or encoded as zero risk.
    assert last_before + pd.Timedelta(seconds=2) not in result.index


def test_contiguous_midnight_quotes_are_valid_history_not_a_day_reset():
    ticks = prices(np.full(1000, -.001), start="2026-01-01T23:50Z")
    result = describe_tick_path_risk(ticks, 1, .001)
    assert result.loc[pd.Timestamp("2026-01-02T00:00:01Z"), "tick_path_feature_valid"]
    assert result.loc[pd.Timestamp("2026-01-02T00:00:01Z"), PATH_FEATURE_NAME] == pytest.approx(1.)


def test_native_direction_is_not_strategy_direction_and_reflected_paths_match():
    signed = np.tile([-.002, .001, -.003, .004], 200)
    boom = describe_tick_path_risk(prices(signed, 1), 1, .001)
    crash = describe_tick_path_risk(prices(signed, -1), -1, .001)
    np.testing.assert_allclose(boom[PATH_FEATURE_NAME], crash[PATH_FEATURE_NAME], atol=1e-10, equal_nan=True)
    # The same descriptor is reused for SPIKE and DRIFT; there is no mode input.
    opposite_native = describe_tick_path_risk(prices(signed, 1), -1, .001)
    assert boom[PATH_FEATURE_NAME].iloc[-1] == pytest.approx(3.25)
    assert opposite_native[PATH_FEATURE_NAME].iloc[-1] == pytest.approx(4.25)


def test_descriptor_and_join_do_not_mutate_inputs_or_modify_original44():
    ticks, frame = prices(np.tile([-.001, .002], 750)), inputs()
    before_ticks, before_frame = ticks.copy(deep=True), frame.copy(deep=True)
    described = describe_tick_path_risk(ticks, 1, .001)
    before_description = described.copy(deep=True)
    output, n44, n45 = join(frame, described)
    pd.testing.assert_frame_equal(ticks, before_ticks)
    pd.testing.assert_frame_equal(frame, before_frame)
    pd.testing.assert_frame_equal(described, before_description)
    pd.testing.assert_frame_equal(output[n44], frame[n44])
    pd.testing.assert_index_equal(output.index, frame.index)
    assert n44 == list(BASE_NAMES) and n45 == [*BASE_NAMES, PATH_FEATURE_NAME]
    assert list(output.feature_valid) == [False, False, True, True]
    assert output.feature_valid.equals(output.common_available)
    issues = frame.index + pd.Timedelta(minutes=5)
    np.testing.assert_allclose(output[PATH_FEATURE_NAME], described.reindex(issues)[PATH_FEATURE_NAME], equal_nan=True)


def test_exact_m5_join_preserves_invalid_interior_frame_and_never_falls_back():
    frame = inputs(5)
    frame.loc[frame.index[2], "feature_valid"] = False
    frame.loc[frame.index[3], BASE_NAMES[0]] = np.nan
    described = descriptor(1801)
    output, n44, n45 = join(frame, described)
    assert len(output) == len(frame)
    assert output.feature_valid.tolist() == [False, False, False, False, True]
    assert output.multiframe_feature_valid.tolist() == [True, True, False, False, True]
    assert output.tick_path_feature_valid.tolist() == [False, False, True, True, True]
    assert n45[:-1] == n44
    assert output.loc[frame.index[2], PATH_FEATURE_NAME] >= 0  # Known risk does not rescue an invalid frame.


def test_missing_exact_descriptor_second_is_unknown_even_with_neighbors_known():
    described = descriptor()
    target = pd.Timestamp("2026-01-01T00:15Z")
    assert described.loc[target - pd.Timedelta(seconds=1), "tick_path_feature_valid"]
    assert described.loc[target + pd.Timedelta(seconds=1), "tick_path_feature_valid"]
    removed = described.drop(target)
    output, _, _ = join(described=removed)
    row = output.loc[pd.Timestamp("2026-01-01T00:10Z")]
    assert not row.feature_valid and not row.tick_path_feature_valid
    assert pd.isna(row[PATH_FEATURE_NAME])
    assert output.loc[pd.Timestamp("2026-01-01T00:15Z"), "feature_valid"]


def test_current_quote_event_or_return_columns_are_ignored_by_exact_join():
    described = descriptor()
    first, _, names = join(described=described)
    described["quote"], described["tail_event"], described["current_return"] = 1e200, True, -1e200
    second, _, _ = join(described=described)
    pd.testing.assert_frame_equal(first, second)
    assert "quote" not in names and "tail_event" not in names and "current_return" not in names


@pytest.mark.parametrize("scale", (0., -1., math.nan, math.inf, -math.inf, True, np.bool_(False), "0.001", 1j, None))
def test_invalid_frozen_scales_are_refused_without_fitting_or_numerical_floor(scale):
    with pytest.raises((TypeError, ValueError)):
        describe_tick_path_risk(constant(), 1, scale)


@pytest.mark.parametrize("side", (0, 2, -2, True, False, "boom", "SPIKE", "DRIFT", 1., None))
def test_native_side_is_strict_integer_plus_or_minus_one(side):
    with pytest.raises((TypeError, ValueError)):
        describe_tick_path_risk(constant(), side, .001)


@pytest.mark.parametrize("quote", (0., -1., math.nan, math.inf, -math.inf, True, "100", 1j, None))
def test_invalid_positive_finite_quote_values_are_refused(quote):
    ticks = constant().astype(object)
    ticks.iloc[10, 0] = quote
    with pytest.raises((TypeError, ValueError)):
        describe_tick_path_risk(ticks, 1, .001)


@pytest.mark.parametrize("fault", ("unsorted", "duplicate", "naive", "nonzero_offset", "fractional_second",
                                   "nat", "empty", "missing_quote", "duplicate_column", "wrong_type"))
def test_quote_stream_schema_and_utc_grid_are_strict(fault):
    ticks = constant()
    if fault == "unsorted":
        ticks = ticks.iloc[::-1]
    elif fault == "duplicate":
        ticks = pd.concat([ticks.iloc[:10], ticks.iloc[9:]])
    elif fault == "naive":
        ticks.index = ticks.index.tz_localize(None)
    elif fault == "nonzero_offset":
        ticks.index = ticks.index.tz_convert("Asia/Riyadh")
    elif fault == "fractional_second":
        ticks.index += pd.Timedelta(milliseconds=1)
    elif fault == "nat":
        changed = ticks.index.to_list(); changed[10] = pd.NaT
        ticks.index = pd.DatetimeIndex(changed)
    elif fault == "empty":
        ticks = ticks.iloc[:0]
    elif fault == "missing_quote":
        ticks = ticks.rename(columns={"quote": "price"})
    elif fault == "duplicate_column":
        ticks = pd.concat([ticks, ticks], axis=1)
    else:
        ticks = []
    with pytest.raises((TypeError, ValueError)):
        describe_tick_path_risk(ticks, 1, .001)


def test_extreme_positive_prices_use_finite_log_return_fallback():
    values = np.tile([1e300, 1e-300], 301)[:601]
    ticks = constant(); ticks.quote = values
    result = describe_tick_path_risk(ticks, 1, 1.)
    expected = (math.log(1e-300) - math.log(1e300))**2 / 2
    assert result[PATH_FEATURE_NAME].iloc[-1] == pytest.approx(expected)


def test_nonfinite_arithmetic_is_refused_instead_of_clipped_or_zeroed():
    with pytest.raises(ValueError, match="arithmetic"):
        describe_tick_path_risk(prices(np.full(600, -.001)), 1, 1e-300)


@pytest.mark.parametrize("fault", ("swapped_names", "extra_feature", "string_names", "missing_feature", "missing_flag",
                                   "bad_bool", "infinite_base", "reserved_output", "duplicate_columns", "wrong_m5_grid",
                                   "missing_risk", "missing_risk_flag", "negative_risk", "infinite_risk", "valid_unknown_risk",
                                   "numeric_risk_bool", "bad_risk_flag", "empty_descriptor", "unsorted_descriptor", "wrong_descriptor_offset"))
def test_join_refuses_changed_schema_unknown_coercions_and_invalid_grid(fault):
    frame, described, names = inputs(), descriptor(), list(BASE_NAMES)
    if fault == "swapped_names":
        names[0], names[1] = names[1], names[0]
    elif fault == "extra_feature":
        names.append("net_R")
    elif fault == "string_names":
        names = BASE_NAMES[0]
    elif fault == "missing_feature":
        frame = frame.drop(columns=BASE_NAMES[0])
    elif fault == "missing_flag":
        frame = frame.drop(columns="feature_valid")
    elif fault == "bad_bool":
        frame["feature_valid"] = 1
    elif fault == "infinite_base":
        frame.loc[frame.index[0], BASE_NAMES[0]] = np.inf
    elif fault == "reserved_output":
        frame["signal_time"] = frame.index
    elif fault == "duplicate_columns":
        frame = pd.concat([frame, frame[[BASE_NAMES[0]]]], axis=1)
    elif fault == "wrong_m5_grid":
        frame.index += pd.Timedelta(minutes=1)
    elif fault == "missing_risk":
        described = described.drop(columns=PATH_FEATURE_NAME)
    elif fault == "missing_risk_flag":
        described = described.drop(columns="tick_path_feature_valid")
    elif fault == "negative_risk":
        described.loc[described.index[-1], PATH_FEATURE_NAME] = -.1
    elif fault == "infinite_risk":
        described.loc[described.index[-1], PATH_FEATURE_NAME] = np.inf
    elif fault == "valid_unknown_risk":
        described.loc[described.index[-1], PATH_FEATURE_NAME] = np.nan
    elif fault == "numeric_risk_bool":
        described[PATH_FEATURE_NAME] = True
    elif fault == "bad_risk_flag":
        described["tick_path_feature_valid"] = "false"
    elif fault == "empty_descriptor":
        described = described.iloc[:0]
    elif fault == "unsorted_descriptor":
        described = described.iloc[::-1]
    else:
        described.index = described.index.tz_convert("Asia/Riyadh")
    with pytest.raises((TypeError, ValueError)):
        join(frame, described, names)
