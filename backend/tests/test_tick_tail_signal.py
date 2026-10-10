"""Synthetic-only exact-clock, strict-prior-state and paired availability tests."""

import math

import numpy as np
import pandas as pd
import pytest

from app.research.multiframe_signal import FEATURE_NAMES
from app.research.tick_tail import FixedTailDetector, describe_tick_tail
from app.research.tick_tail_signal import (
    AGE_DENOMINATOR_SECONDS,
    CADENCE_MINUTES,
    TAIL_FEATURE_NAMES,
    join_tick_tail_inputs,
)


def _inputs(n=3):
    index = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC", name="m5_open")
    frame = pd.DataFrame({name: np.arange(n, dtype=float) + i / 100
                          for i, name in enumerate(FEATURE_NAMES)}, index=index)
    frame["feature_valid"] = True
    frame["h4_closed_at"] = index.floor("4h")
    return frame


def _description():
    index = pd.date_range("2026-01-01", periods=901, freq="s", tz="UTC")
    increments = np.full(len(index), -0.00001)
    increments[0] = 0
    increments[[120, 420, 700]] = 0.002
    ticks = pd.DataFrame({"quote": 100 * np.exp(np.cumsum(increments))}, index=index)
    detector = FixedTailDetector(1, 0.00001)
    return ticks, detector, describe_tick_tail(ticks, detector, 600)


def _join(frame=None, tail=None, names=None):
    return join_tick_tail_inputs(_inputs() if frame is None else frame,
                                 list(FEATURE_NAMES) if names is None else names,
                                 _description()[2] if tail is None else tail)


def test_exact_schema_transform_clock_and_inputs_remain_unmodified():
    frame = _inputs()
    _, _, tail = _description()
    original_frame, original_tail = frame.copy(deep=True), tail.copy(deep=True)
    result, names44, names46 = _join(frame, tail)
    assert CADENCE_MINUTES == 5 and AGE_DENOMINATOR_SECONDS == 600
    assert names44 == list(FEATURE_NAMES)
    assert names46 == [*FEATURE_NAMES, *TAIL_FEATURE_NAMES]
    assert len(names44) == 44 and len(names46) == 46
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_frame_equal(result[names44], frame[names44])
    issues = (frame.index + pd.Timedelta(minutes=5)).as_unit("ns")
    pd.testing.assert_index_equal(pd.DatetimeIndex(result.signal_time), issues, check_names=False)
    expected = tail.reindex(issues)
    np.testing.assert_allclose(result.tail_age_log1p,
                               np.log1p(expected.pre_event_age_seconds / 600))
    np.testing.assert_array_equal(result.prior_tail_mark, expected.prior_tail_mark)
    assert result.common_available.all()
    np.testing.assert_array_equal(result.feature_valid, result.common_available)
    assert not {"quote", "tail_event", "directed_log_return", "age_band"}.intersection(result.columns)
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_frame_equal(tail, original_tail)


def test_current_tail_quote_and_event_cannot_enter_its_own_feature_state():
    ticks, detector, original = _description()
    issue = ticks.index[600]
    changed = ticks.copy()
    changed.loc[issue:, "quote"] *= 1.03
    altered = describe_tick_tail(changed, detector, 600)
    assert not original.loc[issue, "tail_event"] and altered.loc[issue, "tail_event"]
    first, _, names = _join(tail=original)
    second, _, _ = _join(tail=altered)
    through_issue = first.signal_time <= issue
    columns = names + ["feature_valid", "common_available", "tail_feature_valid"]
    pd.testing.assert_frame_equal(first.loc[through_issue, columns],
                                 second.loc[through_issue, columns])
    following = issue + pd.Timedelta(seconds=1)
    assert altered.loc[following, "pre_event_age_seconds"] == 1
    assert original.loc[following, "pre_event_age_seconds"] == 181


def test_future_tail_perturbation_leaves_all_earlier_joined_inputs_unchanged():
    ticks, detector, tail = _description()
    cutoff = ticks.index[600]
    changed = ticks.copy()
    changed.loc[changed.index > cutoff, "quote"] *= 2
    altered = describe_tick_tail(changed, detector, 600)
    first, _, names = _join(tail=tail)
    second, _, _ = _join(tail=altered)
    columns = names + ["feature_valid", "common_available"]
    pd.testing.assert_frame_equal(first.loc[first.signal_time <= cutoff, columns],
                                 second.loc[second.signal_time <= cutoff, columns])
    prefix = tail.loc[tail.index <= cutoff]
    prefix_result, _, _ = _join(tail=prefix)
    pd.testing.assert_frame_equal(first.iloc[:2], prefix_result.iloc[:2])
    assert not prefix_result.common_available.iloc[-1]


def test_missing_exact_timestamp_never_uses_previous_or_nearest_tail_row():
    _, _, tail = _description()
    tail = tail.drop(tail.index[600])
    result, _, _ = _join(tail=tail)
    assert result.common_available.tolist() == [True, False, True]
    assert math.isnan(result.tail_age_log1p.iloc[1])
    assert math.isnan(result.prior_tail_mark.iloc[1])
    assert result.index.equals(_inputs().index)


def test_gap_and_unknown_state_have_one_shared_mask_for_both_models_and_clock():
    ticks, detector, _ = _description()
    ticks = ticks.drop(ticks.index[550])
    tail = describe_tick_tail(ticks, detector, 600)
    result, _, _ = _join(tail=tail)
    assert result.common_available.tolist() == [True, False, True]
    assert result.tail_feature_valid.tolist() == [True, False, True]
    assert result.feature_valid.equals(result.common_available)
    assert result.multiframe_feature_valid.all()


@pytest.mark.parametrize("column", ["pre_event_age_seconds", "prior_tail_mark"])
@pytest.mark.parametrize("unknown", [None, np.nan, pd.NA])
def test_unknown_state_stays_unavailable_without_row_drop(column, unknown):
    _, _, tail = _description()
    tail[column] = tail[column].astype(object)
    tail.loc[tail.index[600], column] = unknown
    result, _, _ = _join(tail=tail)
    assert len(result) == 3
    assert not result.common_available.iloc[1]
    assert result.common_available.iloc[[0, 2]].all()


@pytest.mark.parametrize("invalidate", ["flag", "unknown_feature"])
def test_invalid_interior_context_cannot_backfill_from_previous_valid_row(invalidate):
    frame = _inputs()
    if invalidate == "flag":
        frame.loc[frame.index[1], "feature_valid"] = False
    else:
        frame.loc[frame.index[1], FEATURE_NAMES[10]] = np.nan
    result, names44, _ = _join(frame=frame)
    assert result.common_available.tolist() == [True, False, True]
    assert len(result) == len(frame)
    pd.testing.assert_frame_equal(result[names44], frame[names44])
    pd.testing.assert_series_equal(result.h4_closed_at, frame.h4_closed_at)


def test_unobserved_increment_is_unavailable_even_with_supplied_prior_numbers():
    _, _, tail = _description()
    tail.loc[tail.index[600], "increment_known"] = False
    result, _, _ = _join(tail=tail)
    assert not result.common_available.iloc[1]
    assert not result.tail_feature_valid.iloc[1]


def test_noncovariate_labels_do_not_change_availability_or_ordered_model_inputs():
    frame = _inputs()
    frame["net_r_label"] = [1000, -1000, np.nan]
    first, names44, names46 = _join(frame=frame)
    frame["net_r_label"] = [-999, np.nan, 999]
    second, _, _ = _join(frame=frame)
    assert "net_r_label" not in names44 and "net_r_label" not in names46
    pd.testing.assert_frame_equal(first[names46], second[names46])
    pd.testing.assert_series_equal(first.common_available, second.common_available)


@pytest.mark.parametrize("which", ["frame", "tail"])
@pytest.mark.parametrize("fault", ["naive", "nonutc", "unsorted", "duplicate", "nat", "offgrid"])
def test_rejects_unsafe_time_axes(which, fault):
    frame, tail = _inputs(), _description()[2]
    target = frame if which == "frame" else tail
    if fault == "naive":
        target.index = target.index.tz_localize(None)
    elif fault == "nonutc":
        target.index = target.index.tz_convert("Asia/Riyadh")
    elif fault == "unsorted":
        target = target.iloc[::-1]
    elif fault == "duplicate":
        target = pd.concat([target.iloc[:1], target])
    elif fault == "nat":
        values = list(target.index)
        values[0] = pd.NaT
        target.index = pd.DatetimeIndex(values)
    else:
        target.index += pd.Timedelta(seconds=1 if which == "frame" else 0.5)
    with pytest.raises(ValueError):
        _join(frame=target if which == "frame" else frame,
              tail=target if which == "tail" else tail)


@pytest.mark.parametrize("which", ["frame", "tail"])
@pytest.mark.parametrize("value", [True, np.bool_(False), "1", 1 + 0j, np.inf, -np.inf])
def test_rejects_unsafe_numeric_scalars(which, value):
    frame, tail = _inputs(), _description()[2]
    column = FEATURE_NAMES[0] if which == "frame" else "prior_tail_mark"
    target = frame if which == "frame" else tail
    target[column] = target[column].astype(object)
    target.loc[target.index[-1], column] = value
    with pytest.raises((TypeError, ValueError)):
        _join(frame=frame, tail=tail)


@pytest.mark.parametrize("column,value", [
    ("pre_event_age_seconds", 0), ("pre_event_age_seconds", -1),
    ("pre_event_age_seconds", 1.5), ("pre_event_age_seconds", True),
    ("pre_event_age_seconds", np.inf), ("prior_tail_mark", 0),
    ("prior_tail_mark", -0.01),
])
def test_rejects_invalid_known_age_and_mark(column, value):
    _, _, tail = _description()
    tail[column] = tail[column].astype(object)
    tail.loc[tail.index[600], column] = value
    with pytest.raises((TypeError, ValueError)):
        _join(tail=tail)


@pytest.mark.parametrize("column,which", [("feature_valid", "frame"),
                                         ("increment_known", "tail")])
@pytest.mark.parametrize("value", [1, "True", pd.NA])
def test_requires_explicit_boolean_validity_flags(column, which, value):
    frame, tail = _inputs(), _description()[2]
    target = frame if which == "frame" else tail
    target[column] = target[column].astype(object)
    target.loc[target.index[-1], column] = value
    with pytest.raises(TypeError):
        _join(frame=frame, tail=tail)


@pytest.mark.parametrize("fault", ["reordered", "label_added", "missing_name", "string_names",
                                   "missing_feature", "missing_state", "duplicate_column",
                                   "reserved_output", "empty_frame", "empty_tail"])
def test_rejects_schema_changes_and_empty_inputs(fault):
    frame, tail, names = _inputs(), _description()[2], list(FEATURE_NAMES)
    if fault == "reordered":
        names = names[::-1]
    elif fault == "label_added":
        names[-1] = "net_r_label"
    elif fault == "missing_name":
        names = names[:-1]
    elif fault == "string_names":
        names = FEATURE_NAMES[0]
    elif fault == "missing_feature":
        frame = frame.drop(columns=FEATURE_NAMES[0])
    elif fault == "missing_state":
        tail = tail.drop(columns="prior_tail_mark")
    elif fault == "duplicate_column":
        frame = pd.concat([frame, frame[[FEATURE_NAMES[0]]]], axis=1)
    elif fault == "reserved_output":
        frame["common_available"] = True
    elif fault == "empty_frame":
        frame = frame.iloc[:0]
    else:
        tail = tail.iloc[:0]
    with pytest.raises(ValueError):
        _join(frame=frame, tail=tail, names=names)
