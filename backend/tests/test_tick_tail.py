"""Synthetic-only challenges for exploratory tail-tick descriptions."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pandas as pd
import pytest

from app.research.tick_tail import FixedTailDetector, describe_tick_tail, fit_fixed_tail_detector


def ticks(quotes, seconds=None, start="2026-01-01T00:00:00Z"):
    seconds = list(range(len(quotes))) if seconds is None else seconds
    index = pd.DatetimeIndex([pd.Timestamp(start)+pd.Timedelta(seconds=s) for s in seconds])
    return pd.DataFrame({"quote":quotes}, index=index)


def paths_from_returns(returns, side=1, initial=100.):
    values = [initial]
    for value in returns:
        values.append(values[-1]*math.exp(side*value))
    return ticks(values)


def test_training_median_is_only_observed_absolute_consecutive_log_returns():
    training = paths_from_returns([.001, -.002, .003, -.004])
    detector = fit_fixed_tail_detector(training, 1)
    assert detector.median_abs_log_return == pytest.approx(.0025)
    assert detector.threshold == pytest.approx(.025)


def test_detector_scale_and_direction_are_frozen():
    detector = fit_fixed_tail_detector(paths_from_returns([.001, -.001]), 1)
    with pytest.raises(FrozenInstanceError):
        detector.median_abs_log_return = .9
    with pytest.raises(FrozenInstanceError):
        detector.side = -1


def test_future_evaluation_quotes_cannot_refit_training_scale_or_rewrite_prefix():
    training = paths_from_returns([.001, -.001, .001, -.001])
    detector = fit_fixed_tail_detector(training, 1)
    original = paths_from_returns([.001, .020, -.001, .001, .030, -.001])
    perturbed = original.copy()
    perturbed.iloc[5:,0] = [1e100, 1e-100]
    a = describe_tick_tail(original, detector, 600)
    b = describe_tick_tail(perturbed, detector, 600)
    pd.testing.assert_frame_equal(a.iloc[:5], b.iloc[:5])
    assert detector.median_abs_log_return == pytest.approx(.001)


def test_current_first_event_keeps_unknown_prior_age_and_mark():
    frame = paths_from_returns([.02, -.001, .03, -.001])
    result = describe_tick_tail(frame, FixedTailDetector(1,.001), 2)
    assert bool(result.tail_event.iloc[1]) and pd.isna(result.pre_event_age_seconds.iloc[1])
    assert pd.isna(result.prior_tail_mark.iloc[1])
    assert result.pre_event_age_seconds.iloc[2] == 1
    assert result.prior_tail_mark.iloc[2] == pytest.approx(.02)
    assert result.age_band.iloc[2] == "lt_N"
    assert bool(result.tail_event.iloc[3]) and result.pre_event_age_seconds.iloc[3] == 2
    assert result.prior_tail_mark.iloc[3] == pytest.approx(.02)  # excludes current .03 mark
    assert result.age_band.iloc[3] == "ge_N"
    assert result.pre_event_age_seconds.iloc[4] == 1
    assert result.prior_tail_mark.iloc[4] == pytest.approx(.03)


def test_nominal_number_changes_only_age_band_not_detection_or_raw_features():
    frame = paths_from_returns([.02, -.001, -.001, -.001])
    detector = FixedTailDetector(1,.001)
    a = describe_tick_tail(frame,detector,2)
    b = describe_tick_tail(frame,detector,1000)
    pd.testing.assert_frame_equal(a.drop(columns="age_band"),b.drop(columns="age_band"))
    assert a.age_band.iloc[3] == "ge_N" and b.age_band.iloc[3] == "lt_N"


def test_missing_second_resets_unknown_state_and_does_not_invent_gap_event():
    frame = ticks([100.,110.,109.,1000.,1100.,1099.],[0,1,2,4,5,6])
    result = describe_tick_tail(frame,FixedTailDetector(1,.001),600)
    assert pd.isna(result.tail_event.iloc[0]) and not result.increment_known.iloc[0]
    assert pd.isna(result.tail_event.iloc[3]) and not result.increment_known.iloc[3]
    assert pd.isna(result.directed_log_return.iloc[3])
    assert pd.isna(result.pre_event_age_seconds.iloc[3])
    assert bool(result.tail_event.iloc[4]) and pd.isna(result.pre_event_age_seconds.iloc[4])
    assert pd.isna(result.prior_tail_mark.iloc[4])
    assert result.pre_event_age_seconds.iloc[5] == 1


def test_disconnected_day_resets_age_instead_of_counting_unobserved_time():
    frame = ticks([100.,110.,109.,108.,120.,119.],[0,1,2,86400,86401,86402])
    result = describe_tick_tail(frame,FixedTailDetector(1,.001),600)
    assert pd.isna(result.tail_event.iloc[3]) and pd.isna(result.pre_event_age_seconds.iloc[3])
    assert bool(result.tail_event.iloc[4]) and pd.isna(result.pre_event_age_seconds.iloc[4])
    assert result.pre_event_age_seconds.iloc[5] == 1


def test_unknown_state_survives_observed_non_events_until_first_new_tail_event():
    frame = ticks([100.,110.,109.,108.,107.,120.,119.],[0,1,2,100,101,102,103])
    result = describe_tick_tail(frame,FixedTailDetector(1,.001),600)
    assert result.increment_known.iloc[4] and not bool(result.tail_event.iloc[4])
    assert pd.isna(result.pre_event_age_seconds.iloc[4]) and pd.isna(result.prior_tail_mark.iloc[4])
    assert bool(result.tail_event.iloc[5]) and pd.isna(result.pre_event_age_seconds.iloc[5])
    assert result.pre_event_age_seconds.iloc[6] == 1


def test_actual_consecutive_midnight_quotes_keep_observable_event_state():
    frame = ticks([100.,110.,109.,108.],start="2026-01-01T23:59:58Z")
    result = describe_tick_tail(frame,FixedTailDetector(1,.001),600)
    assert result.index[2].day == 2 and result.increment_known.iloc[2]
    assert result.pre_event_age_seconds.iloc[2] == 1
    assert result.pre_event_age_seconds.iloc[3] == 2


def test_training_gap_return_is_excluded_instead_of_becoming_scale():
    frame = ticks([100.,101.,1e10,1.01e10],[0,1,100,101])
    detector = fit_fixed_tail_detector(frame,1)
    assert detector.median_abs_log_return == pytest.approx(math.log1p(.01))


@pytest.mark.parametrize("quotes,seconds",[([100.],[0]),([100.,200.],[0,2]),([100.,100.],[0,1])])
def test_training_without_positive_median_scale_is_rejected(quotes,seconds):
    with pytest.raises(ValueError):fit_fixed_tail_detector(ticks(quotes,seconds),1)


def test_even_sample_median_uses_both_middle_observed_values():
    assert fit_fixed_tail_detector(ticks([100.,100.,101.]),1).median_abs_log_return == pytest.approx(math.log1p(.01)/2)


def test_quantized_training_with_majority_flat_ticks_refuses_zero_scale_floor():
    with pytest.raises(ValueError):fit_fixed_tail_detector(ticks([100.,100.,100.,101.]),1)


def test_equal_quotes_are_observed_zero_return_not_unknown_or_event():
    result = describe_tick_tail(ticks([100.,100.]),FixedTailDetector(1,.001),600)
    assert result.increment_known.iloc[1] and result.directed_log_return.iloc[1] == 0
    assert not bool(result.tail_event.iloc[1]) and pd.isna(result.pre_event_age_seconds.iloc[1])


def test_threshold_equality_is_not_an_event():
    frame = ticks([100.,101.])
    actual = describe_tick_tail(frame,FixedTailDetector(1,.0001),600).directed_log_return.iloc[1]
    detector = FixedTailDetector(1,actual/10)
    assert detector.threshold == actual
    assert not bool(describe_tick_tail(frame,detector,600).tail_event.iloc[1])


def test_small_quantized_price_change_is_not_lost_by_subtracting_price_logs():
    previous = 1e100
    current = np.nextafter(previous,np.inf)
    assert math.log(current) == math.log(previous)
    frame = ticks([previous,current])
    detector = fit_fixed_tail_detector(frame,1)
    assert detector.median_abs_log_return > 0
    assert describe_tick_tail(frame,detector,600).directed_log_return.iloc[1] > 0


def test_extreme_positive_quotes_do_not_overflow_or_underflow_log_ratio():
    frame = ticks([1e-300,1e300,1e-300])
    result = describe_tick_tail(frame,FixedTailDetector(1,.001),600)
    assert np.isfinite(result.directed_log_return.iloc[1:]).all()
    assert result.directed_log_return.iloc[1] == pytest.approx(600*math.log(10))
    assert result.directed_log_return.iloc[2] == pytest.approx(-600*math.log(10))


def test_mirrored_directions_produce_same_directed_events_and_ages():
    returns = [.001,.02,-.001,.03,-.001]
    boom, crash = paths_from_returns(returns,1),paths_from_returns(returns,-1)
    a = describe_tick_tail(boom,FixedTailDetector(1,.001),2)
    b = describe_tick_tail(crash,FixedTailDetector(-1,.001),2)
    for name in ("directed_log_return","pre_event_age_seconds","prior_tail_mark"):
        np.testing.assert_allclose(a[name],b[name],rtol=1e-11,atol=1e-14,equal_nan=True)
    pd.testing.assert_series_equal(a.tail_event,b.tail_event)
    pd.testing.assert_series_equal(a.age_band,b.age_band)


def test_mirrored_training_paths_fit_same_absolute_scale():
    returns = [.001,-.002,.003,-.004]
    boom = fit_fixed_tail_detector(paths_from_returns(returns,1),1)
    crash = fit_fixed_tail_detector(paths_from_returns(returns,-1),-1)
    assert boom.median_abs_log_return == pytest.approx(crash.median_abs_log_return)


def test_one_observed_quote_is_describable_with_unknown_increment_and_age():
    result = describe_tick_tail(ticks([100.]),FixedTailDetector(1,.001),600)
    assert len(result) == 1 and not result.increment_known.iloc[0]
    assert pd.isna(result.tail_event.iloc[0]) and pd.isna(result.pre_event_age_seconds.iloc[0])


@pytest.mark.parametrize("frame",[None,[100.,101.],np.array([100.,101.])])
def test_non_dataframe_inputs_are_rejected(frame):
    with pytest.raises(TypeError):fit_fixed_tail_detector(frame,1)
    with pytest.raises(TypeError):describe_tick_tail(frame,FixedTailDetector(1,.001),600)


@pytest.mark.parametrize("bad",[True,np.bool_(False),1+0j,"100",None,np.nan,np.inf,-np.inf,0.,-1.])
def test_invalid_quote_types_and_values_are_rejected(bad):
    frame = ticks(pd.Series([100.,bad],dtype=object).tolist())
    with pytest.raises((TypeError,ValueError)):
        fit_fixed_tail_detector(frame,1)
    with pytest.raises((TypeError,ValueError)):
        describe_tick_tail(frame,FixedTailDetector(1,.001),600)


@pytest.mark.parametrize("side",[True,np.bool_(True),1.,"1",1+0j,0,2,-2,None])
def test_invalid_sides_are_rejected_without_coercion(side):
    with pytest.raises((TypeError,ValueError)):fit_fixed_tail_detector(ticks([100.,101.]),side)
    with pytest.raises((TypeError,ValueError)):FixedTailDetector(side,.001)


@pytest.mark.parametrize("scale",[True,".001",1+0j,None,np.nan,np.inf,0.,-1.,1e308])
def test_invalid_or_overflowing_frozen_scales_are_rejected(scale):
    with pytest.raises((TypeError,ValueError)):FixedTailDetector(1,scale)


@pytest.mark.parametrize("nominal",[True,np.bool_(False),600.,"600",0,-1,None,1+0j])
def test_nominal_band_boundary_requires_positive_integer(nominal):
    with pytest.raises((TypeError,ValueError)):
        describe_tick_tail(ticks([100.,101.]),FixedTailDetector(1,.001),nominal)


def test_numpy_real_and_integer_scalars_are_accepted():
    detector = fit_fixed_tail_detector(ticks([np.int64(100),np.float64(101)]),np.int64(1))
    assert isinstance(detector.side,int)
    assert len(describe_tick_tail(ticks([100,101]),detector,np.int64(600))) == 2


@pytest.mark.parametrize("kind",["empty","naive","nonutc","fractional","duplicate","unordered","nat","non_datetime","no_quote","duplicate_quote"])
def test_invalid_tick_schema_and_utc_grid_are_rejected(kind):
    frame = ticks([100.,101.])
    if kind == "empty":frame = frame.iloc[:0]
    elif kind == "naive":frame.index = frame.index.tz_localize(None)
    elif kind == "nonutc":frame.index = frame.index.tz_convert("Asia/Riyadh")
    elif kind == "fractional":frame.index = frame.index+pd.Timedelta(milliseconds=1)
    elif kind == "duplicate":frame.index = pd.DatetimeIndex([frame.index[0],frame.index[0]])
    elif kind == "unordered":frame = frame.iloc[::-1]
    elif kind == "nat":frame.index = pd.DatetimeIndex([frame.index[0],pd.NaT])
    elif kind == "non_datetime":frame.index = [0,1]
    elif kind == "no_quote":frame = frame.rename(columns={"quote":"close"})
    elif kind == "duplicate_quote":frame = pd.concat([frame,frame],axis=1)
    with pytest.raises((TypeError,ValueError)):fit_fixed_tail_detector(frame,1)
    with pytest.raises((TypeError,ValueError)):describe_tick_tail(frame,FixedTailDetector(1,.001),600)


def test_input_and_frozen_detector_are_not_mutated():
    frame = paths_from_returns([.001,.02,-.001])
    original = frame.copy(deep=True)
    detector = FixedTailDetector(1,.001)
    describe_tick_tail(frame,detector,600)
    pd.testing.assert_frame_equal(frame,original)
    assert detector == FixedTailDetector(1,.001)
