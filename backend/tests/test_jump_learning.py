"""Synthetic prefix calibration, shared targets and causal fixed issuance."""

from copy import deepcopy
from dataclasses import replace
import inspect
import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.research.jump_learning import (
    InsufficientTrainingEvidenceError,
    PairedRidgeFit,
    calibrate_training_detector,
    fit_paired_spike_ridge,
    issue_paired_spike,
    spike_clock_signals,
)
from app.research.jump_multiframe import PairedMultiframeInputs
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.tick_tail import fit_fixed_tail_detector


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for flag in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"):
        monkeypatch.setenv(flag, "false")


class UnreadableQuote:
    def __float__(self):
        raise AssertionError("A quote outside the training interval was examined")


def test_calibration_selects_prefix_before_reading_any_outside_quote_values():
    index = pd.date_range("2026-01-01", periods=9, freq="s", tz="UTC")
    quotes = [UnreadableQuote(), UnreadableQuote(), 100.0, 101.0, 99.0, 100.5,
              UnreadableQuote(), UnreadableQuote(), UnreadableQuote()]
    source = pd.DataFrame({"quote": quotes}, index=index)
    result = calibrate_training_detector(source, 1, index[2], index[6])
    expected = fit_fixed_tail_detector(source.iloc[2:6], 1)
    assert result.detector == expected
    assert result.metadata["training_rows"] == 4
    assert result.metadata["known_consecutive_training_increments"] == 3
    assert result.metadata["training_first_observed"] == index[2].isoformat()
    assert result.metadata["training_last_observed"] == index[5].isoformat()
    assert result.metadata["future_quote_values_examined"] is False
    assert result.metadata["cross_boundary_pairs_used"] is False


@pytest.mark.parametrize("side", (1, -1))
def test_calibration_gap_pairs_are_excluded_and_source_is_preserved(side):
    index = pd.DatetimeIndex(["2026-01-01 00:00:00Z", "2026-01-01 00:00:01Z",
                              "2026-01-01 00:00:05Z", "2026-01-01 00:00:06Z"])
    source = pd.DataFrame({"quote": [100.0, 101.0, 500.0, 490.0]}, index=index)
    before = source.copy(deep=True)
    result = calibrate_training_detector(source, side, index[0], index[-1] + pd.Timedelta(seconds=1))
    assert result.metadata["known_consecutive_training_increments"] == 2
    assert result.detector == fit_fixed_tail_detector(source, side)
    assert_frame_equal(source, before)


def test_future_valid_quote_changes_cannot_change_detector():
    index = pd.date_range("2026-01-01", periods=10, freq="s", tz="UTC")
    source = pd.DataFrame({"quote": np.arange(100.0, 110.0)}, index=index)
    first = calibrate_training_detector(source, -1, index[0], index[5])
    changed = source.copy()
    changed.loc[index[5]:, "quote"] *= 1e100
    second = calibrate_training_detector(changed, -1, index[0], index[5])
    assert first == second


@pytest.mark.parametrize("kind", ("empty_prefix", "one_quote", "zero_scale", "bad_training_quote", "bad_future_index"))
def test_invalid_calibration_has_no_alternative_window_or_scale(kind):
    index = pd.date_range("2026-01-01", periods=5, freq="s", tz="UTC")
    source = pd.DataFrame({"quote": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=index)
    start, end = index[0], index[3]
    if kind == "empty_prefix":
        start, end = index[-1] + pd.Timedelta(seconds=1), index[-1] + pd.Timedelta(seconds=2)
    elif kind == "one_quote":
        end = index[1]
    elif kind == "zero_scale":
        source["quote"] = 100.0
    elif kind == "bad_training_quote":
        source.iloc[1, 0] = math.nan
    elif kind == "bad_future_index":
        source.index = pd.DatetimeIndex([*index[:4], index[4] + pd.Timedelta(milliseconds=1)])
    with pytest.raises((TypeError, ValueError)):
        calibrate_training_detector(source, 1, start, end)


@pytest.fixture(scope="module")
def synthetic_inputs():
    index = pd.date_range("2025-12-31 23:55", periods=7500, freq="5min", tz="UTC")
    rng = np.random.default_rng(20261008)
    matrix = rng.normal(size=(len(index), len(FEATURE_NAMES)))
    raw = pd.DataFrame(matrix, index=index, columns=FEATURE_NAMES)
    transformed = pd.DataFrame(matrix * np.linspace(2.0, 4.0, len(FEATURE_NAMES)) + 3.0,
                               index=index, columns=FEATURE_NAMES)
    for frame, atr in ((raw, 2.0), (transformed, 0.02)):
        frame["feature_valid"] = True
        frame["atr"] = atr
        frame["close"] = 100.0 + np.arange(len(frame)) * 0.001
    available = pd.DataFrame({"raw_feature_valid": True, "transformed_feature_valid": True,
                              "common_feature_valid": True, "raw_execution_atr": 2.0,
                              "transformed_feature_atr": 0.02,
                              "run_id": pd.array(np.zeros(len(index)), dtype="Int64")}, index=index)
    return PairedMultiframeInputs(raw, transformed, available)


def _copy_inputs(inputs):
    return replace(inputs, raw=inputs.raw.copy(deep=True), transformed=inputs.transformed.copy(deep=True),
                   availability=inputs.availability.copy(deep=True))


def training_bounds(count=1005):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=(count - 1) * 30 + 31)
    return start, end


def labels_for(inputs, count=1005, symbol="BOOM600"):
    start, end = training_bounds(count)
    clock = spike_clock_signals(inputs, symbol, start, end)
    index = clock.signals.index
    opening = index - pd.Timedelta(minutes=5)
    y = 0.2 + 0.02 * inputs.raw.loc[opening, FEATURE_NAMES[0]].to_numpy()
    labels = pd.DataFrame({"net_R": y, "completed": True,
                           "planned_end": index + pd.Timedelta(minutes=16)}, index=index)
    return labels, start, end


@pytest.fixture(scope="module")
def fitted(synthetic_inputs):
    labels, start, end = labels_for(synthetic_inputs)
    return fit_paired_spike_ridge(synthetic_inputs, labels, "BOOM600", start, end)


def test_fixed_paired_models_share_targets_but_fit_separate_scalers(synthetic_inputs, fitted):
    labels, start, end = labels_for(synthetic_inputs)
    opening = labels.index - pd.Timedelta(minutes=5)
    assert fitted.metadata["training_completed_labels"] == 1005
    assert fitted.metadata["training_common_clock_opportunities"] == 1005
    assert fitted.metadata["training_unknown_labels"] == 0
    assert fitted.metadata["training_latest_issue"] == labels.index[-1].isoformat()
    assert labels.index[-1] + pd.Timedelta(minutes=31) == end
    assert fitted.metadata["counts_toward_profit_sample_target"] is False
    assert len(fitted.metadata["common_timestamp_target_sha256"]) == 64
    assert fitted.metadata["matrix_sha256"]["RAW44"] != fitted.metadata["matrix_sha256"]["TRANSFORMED44"]
    for family, frame in (("RAW44", synthetic_inputs.raw), ("TRANSFORMED44", synthetic_inputs.transformed)):
        model = fitted.models[family]
        assert model["penalty"] == 0.1 and model["quantile"] == 0.75
        assert model["fit_rows"] == 1005 and model["scaler_fitted_on"] == "training_rows_only"
        expected = frame.loc[opening, list(FEATURE_NAMES)].mean(axis=0)
        np.testing.assert_allclose(model["means"], expected.to_numpy(), rtol=0, atol=1e-12)
    assert fitted.models["RAW44"]["means"] != fitted.models["TRANSFORMED44"]["means"]
    assert fitted.models["RAW44"]["intercept"] == fitted.models["TRANSFORMED44"]["intercept"] == float(labels.net_R.mean())


def test_unknown_labels_are_counted_excluded_from_both_fits_and_never_zero(synthetic_inputs):
    labels, start, end = labels_for(synthetic_inputs)
    labels.loc[labels.index[:5], "completed"] = False
    labels.loc[labels.index[:5], "net_R"] = math.nan
    result = fit_paired_spike_ridge(synthetic_inputs, labels, "BOOM600", start, end)
    assert result.metadata["training_completed_labels"] == 1000
    assert result.metadata["training_unknown_labels"] == 5
    assert result.models["RAW44"]["fit_rows"] == result.models["TRANSFORMED44"]["fit_rows"] == 1000
    assert result.models["RAW44"]["intercept"] == pytest.approx(labels.net_R.mean())


def test_999_completed_labels_refuse_every_model_fallback(synthetic_inputs):
    labels, start, end = labels_for(synthetic_inputs, count=999)
    with pytest.raises(InsufficientTrainingEvidenceError, match="observed 999"):
        fit_paired_spike_ridge(synthetic_inputs, labels, "BOOM600", start, end)


@pytest.mark.parametrize("kind", ("missing", "extra_future", "off_clock", "different_planned", "planned_unknown",
                                  "complete_nan", "unknown_zero", "duplicate", "extra_target", "bool_target", "nonbool_completed"))
def test_bad_leaked_or_mismatched_training_labels_are_refused(synthetic_inputs, kind):
    labels, start, end = labels_for(synthetic_inputs)
    if kind == "missing":
        labels = labels.iloc[1:]
    elif kind == "extra_future":
        extra = labels.iloc[-1:].copy()
        extra.index += pd.Timedelta(minutes=30)
        extra["planned_end"] += pd.Timedelta(minutes=30)
        labels = pd.concat([labels, extra])
    elif kind == "off_clock":
        labels.index += pd.Timedelta(minutes=5)
    elif kind == "different_planned":
        labels.iloc[0, labels.columns.get_loc("planned_end")] += pd.Timedelta(seconds=1)
    elif kind == "planned_unknown":
        labels.iloc[0, labels.columns.get_loc("planned_end")] = pd.NaT
    elif kind == "complete_nan":
        labels.iloc[0, labels.columns.get_loc("net_R")] = math.nan
    elif kind == "unknown_zero":
        labels.iloc[0, labels.columns.get_loc("completed")] = False
        labels.iloc[0, labels.columns.get_loc("net_R")] = 0.0
    elif kind == "duplicate":
        labels = pd.concat([labels.iloc[:1], labels])
    elif kind == "extra_target":
        labels["transformed_net_R"] = labels.net_R
    elif kind == "bool_target":
        labels["net_R"] = True
    elif kind == "nonbool_completed":
        labels["completed"] = 1
    with pytest.raises((TypeError, ValueError)):
        fit_paired_spike_ridge(synthetic_inputs, labels, "BOOM600", start, end)


def test_training_future_feature_values_are_not_read(synthetic_inputs, fitted):
    inputs = _copy_inputs(synthetic_inputs)
    labels, start, end = labels_for(inputs)
    future = inputs.raw.index + pd.Timedelta(minutes=5) >= end
    for frame in (inputs.raw, inputs.transformed):
        frame[FEATURE_NAMES[0]] = frame[FEATURE_NAMES[0]].astype(object)
        frame.loc[future, FEATURE_NAMES[0]] = UnreadableQuote()
    result = fit_paired_spike_ridge(inputs, labels, "BOOM600", start, end)
    assert result == fitted


@pytest.mark.parametrize("symbol,side", (("BOOM600", 1), ("CRASH600", -1)))
def test_issuance_uses_native_side_common00_or30_and_raw_atr_only(synthetic_inputs, symbol, side):
    labels, start, training_end = labels_for(synthetic_inputs, symbol=symbol)
    fit = fit_paired_spike_ridge(synthetic_inputs, labels, symbol, start, training_end)
    end = synthetic_inputs.raw.index[-1] + pd.Timedelta(minutes=5)
    issued = issue_paired_spike(synthetic_inputs, fit, symbol, training_end, end)
    assert set(issued.signals) == {"RAW44", "TRANSFORMED44", "CLOCK"}
    assert len(issued.signals["CLOCK"]) > 0
    assert len(issued.signals["RAW44"]) > 0 and len(issued.signals["TRANSFORMED44"]) > 0
    for family, signals in issued.signals.items():
        assert signals.index.equals(pd.DatetimeIndex(signals.signal_time))
        assert signals.index.is_unique and signals.index.is_monotonic_increasing
        assert (signals.index.minute % 30 == 0).all()
        assert (signals.index.second == 0).all()
        assert (signals.side == side).all()
        assert signals.atr.eq(2.0).all()
        assert signals.index.min() >= training_end
        assert (signals.index + pd.Timedelta(minutes=31) <= end).all()
        if family != "CLOCK":
            assert signals.score.ge(fit.models[family]["threshold"]).all()
            assert signals.score.gt(0).all()
        else:
            assert signals.score.isna().all()


def test_no_labels_or_fitting_are_available_during_issuance(synthetic_inputs, fitted, monkeypatch):
    assert "labels" not in inspect.signature(issue_paired_spike).parameters
    def forbidden(*args, **kwargs):
        raise AssertionError("Issuance attempted to fit")
    monkeypatch.setattr("app.research.jump_learning.fit_ridge", forbidden)
    monkeypatch.setattr("app.research.jump_learning.fit_fixed_tail_detector", forbidden)
    _, end_training = training_bounds()
    end = synthetic_inputs.raw.index[-1] + pd.Timedelta(minutes=5)
    result = issue_paired_spike(synthetic_inputs, fitted, "BOOM600", end_training, end)
    assert result.signals["CLOCK"].shape[0] > 0


def test_eval_before_training_end_is_refused(synthetic_inputs, fitted):
    start, end = training_bounds()
    with pytest.raises(ValueError, match="before.*training end"):
        issue_paired_spike(synthetic_inputs, fitted, "BOOM600", start, end)


def test_no_common_age_availability_means_empty_clock_and_no_training_fallback(synthetic_inputs):
    inputs = _copy_inputs(synthetic_inputs)
    inputs.transformed["feature_valid"] = False
    inputs.availability["transformed_feature_valid"] = False
    inputs.availability["common_feature_valid"] = False
    start, end = training_bounds()
    clock = spike_clock_signals(inputs, "BOOM600", start, end)
    assert clock.signals.empty and clock.metadata["common_clock_opportunities"] == 0
    labels = pd.DataFrame(columns=["net_R", "completed", "planned_end"], index=pd.DatetimeIndex([], tz="UTC"))
    with pytest.raises(InsufficientTrainingEvidenceError, match="observed 0"):
        fit_paired_spike_ridge(inputs, labels, "BOOM600", start, end)


def test_mask_is_fixed_before_noncommon_numeric_values_and_future_labels(synthetic_inputs):
    inputs = _copy_inputs(synthetic_inputs)
    target_row = inputs.raw.index[0]
    inputs.transformed.loc[target_row, "feature_valid"] = False
    inputs.transformed.loc[target_row, FEATURE_NAMES[0]] = math.inf
    inputs.availability.loc[target_row, "transformed_feature_valid"] = False
    inputs.availability.loc[target_row, "common_feature_valid"] = False
    labels, start, end = labels_for(inputs)
    assert len(labels) == 1004
    result = fit_paired_spike_ridge(inputs, labels, "BOOM600", start, end)
    assert result.metadata["training_completed_labels"] == 1004


def test_one_second_past_purge_boundary_cannot_be_rescued_by_early_planned_exit(synthetic_inputs):
    start, end = training_bounds()
    full = spike_clock_signals(synthetic_inputs, "BOOM600", start, end)
    shortened = spike_clock_signals(synthetic_inputs, "BOOM600", start, end - pd.Timedelta(seconds=1))
    assert len(full.signals) == 1005 and len(shortened.signals) == 1004
    last = full.signals.index[-1]
    assert last + pd.Timedelta(minutes=16, seconds=1) < end - pd.Timedelta(seconds=1)
    assert last not in shortened.signals.index


def test_future_evaluation_feature_changes_cannot_move_earlier_frozen_issuance(synthetic_inputs, fitted):
    inputs = _copy_inputs(synthetic_inputs)
    _, start = training_bounds()
    end = inputs.raw.index[-1] + pd.Timedelta(minutes=5)
    cut = start + pd.Timedelta(days=2)
    original = issue_paired_spike(inputs, fitted, "BOOM600", start, end)
    future = inputs.raw.index + pd.Timedelta(minutes=5) >= cut
    for frame in (inputs.raw, inputs.transformed):
        frame.loc[future, list(FEATURE_NAMES)] += 100.0
    changed = issue_paired_spike(inputs, fitted, "BOOM600", start, end)
    for family in ("RAW44", "TRANSFORMED44", "CLOCK"):
        left, right = original.signals[family], changed.signals[family]
        assert_frame_equal(left.loc[left.index < cut], right.loc[right.index < cut], check_exact=True)


@pytest.mark.parametrize("kind", ("common_mismatch", "raw_flag_mismatch", "nan_valid_feature", "wrong_raw_atr",
                                  "wrong_transformed_atr", "zero_raw_atr", "negative_close", "unknown_run",
                                  "bool_run", "schema", "omitted_grid_row"))
def test_inconsistent_selected_inputs_refuse_clock_generation(synthetic_inputs, kind):
    inputs = _copy_inputs(synthetic_inputs)
    row = inputs.raw.index[0]
    if kind == "common_mismatch":
        inputs.availability.loc[row, "common_feature_valid"] = False
    elif kind == "raw_flag_mismatch":
        inputs.availability.loc[row, "raw_feature_valid"] = False
    elif kind == "nan_valid_feature":
        inputs.raw.loc[row, FEATURE_NAMES[0]] = math.nan
    elif kind == "wrong_raw_atr":
        inputs.availability.loc[row, "raw_execution_atr"] = 0.02
    elif kind == "wrong_transformed_atr":
        inputs.availability.loc[row, "transformed_feature_atr"] = 2.0
    elif kind == "zero_raw_atr":
        inputs.raw.loc[row, "atr"] = 0.0
        inputs.availability.loc[row, "raw_execution_atr"] = 0.0
    elif kind == "negative_close":
        inputs.raw.loc[row, "close"] = -1.0
    elif kind == "unknown_run":
        inputs.availability.loc[row, "run_id"] = pd.NA
    elif kind == "bool_run":
        inputs.availability["run_id"] = False
    elif kind == "schema":
        inputs = replace(inputs, feature_names=tuple(reversed(FEATURE_NAMES)))
    elif kind == "omitted_grid_row":
        inputs = replace(inputs, raw=inputs.raw.drop(inputs.raw.index[3]),
                         transformed=inputs.transformed.drop(inputs.transformed.index[3]),
                         availability=inputs.availability.drop(inputs.availability.index[3]))
    start, end = training_bounds()
    with pytest.raises((TypeError, ValueError)):
        spike_clock_signals(inputs, "BOOM600", start, end)


@pytest.mark.parametrize("kind", ("coefficient", "threshold", "training_bound", "penalty", "missing_family", "wrong_symbol"))
def test_mutated_or_mismatched_frozen_models_are_refused(synthetic_inputs, fitted, kind):
    fit = PairedRidgeFit(deepcopy(fitted.models), deepcopy(fitted.metadata))
    if kind == "coefficient":
        fit.models["RAW44"]["coefs"][0] += 1.0
    elif kind == "threshold":
        fit.models["RAW44"]["threshold"] = -1.0
    elif kind == "training_bound":
        fit.metadata["training_end"] = fit.metadata["training_start"]
    elif kind == "penalty":
        fit.models["RAW44"]["penalty"] = 0.2
    elif kind == "missing_family":
        fit.models.pop("TRANSFORMED44")
    _, start = training_bounds()
    end = synthetic_inputs.raw.index[-1] + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError):
        issue_paired_spike(synthetic_inputs, fit, "CRASH600" if kind == "wrong_symbol" else "BOOM600", start, end)


def test_sources_labels_and_fitted_models_are_not_mutated_by_fit_or_issuance(synthetic_inputs, fitted):
    labels, start, end = labels_for(synthetic_inputs)
    frames = [frame.copy(deep=True) for frame in (synthetic_inputs.raw, synthetic_inputs.transformed, synthetic_inputs.availability)]
    labels_before = labels.copy(deep=True)
    fit_paired_spike_ridge(synthetic_inputs, labels, "BOOM600", start, end)
    fit_before = deepcopy(fitted)
    issue_paired_spike(synthetic_inputs, fitted, "BOOM600", end, synthetic_inputs.raw.index[-1] + pd.Timedelta(minutes=5))
    assert_frame_equal(labels, labels_before)
    for original, before in zip((synthetic_inputs.raw, synthetic_inputs.transformed, synthetic_inputs.availability), frames, strict=True):
        assert_frame_equal(original, before)
    assert fitted == fit_before


@pytest.mark.parametrize("bad", ("2026-01-01", "2026-01-01T00:00:00+03:00", "2026-01-01T00:00:00.001Z", pd.NaT))
def test_bounds_require_explicit_whole_second_utc(synthetic_inputs, bad):
    _, end = training_bounds()
    with pytest.raises(ValueError):
        spike_clock_signals(synthetic_inputs, "BOOM600", bad, end)


@pytest.mark.parametrize("symbol", ("BOOM500", "boom600", "DRIFT", None, True))
def test_no_undeclared_symbol_or_direction(synthetic_inputs, symbol):
    start, end = training_bounds()
    with pytest.raises(ValueError):
        spike_clock_signals(synthetic_inputs, symbol, start, end)


@pytest.mark.parametrize("flag", ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"))
def test_all_entrypoints_refuse_unsafe_flags_before_data_access(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    for call in (
        lambda: calibrate_training_detector(None, 1, None, None),
        lambda: spike_clock_signals(None, "BOOM600", None, None),
        lambda: fit_paired_spike_ridge(None, None, "BOOM600", None, None),
        lambda: issue_paired_spike(None, None, "BOOM600", None, None),
    ):
        with pytest.raises(RuntimeError, match="Safety invariant"):
            call()
