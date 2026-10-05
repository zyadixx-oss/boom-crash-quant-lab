"""Closed-bar causality, missing-history handling and frozen model checks."""

import json

import numpy as np
import pandas as pd
import pytest

from app.research.learned_signal import (
    FEATURE_FORMULAS,
    FEATURE_NAMES,
    causal_inputs,
    fit_ridge,
    predict_ridge,
)


def history(n=3500):
    index = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    phase = np.arange(n)
    close = 100 + 0.5 * np.sin(phase / 15) + 0.1 * np.sin(phase / 73)
    opening = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 0.2,
        "low": np.minimum(opening, close) - 0.2, "close": close,
    }, index=index)
    # Observed large completed candles give age a known starting point.
    for i in (150, 1800):
        if i < n:
            frame.loc[index[i], "high"] += 3
    return frame


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_input_prefix_and_replaced_future_are_identical(direction):
    frame = history()
    cutoff = frame.index[2100]
    original, names = causal_inputs(frame, direction)
    assert len(names) == 19 and tuple(names) == FEATURE_NAMES
    assert set(names) == set(FEATURE_FORMULAS)
    assert original.feature_valid.any()
    modified = frame.copy()
    modified.loc[cutoff:, :] *= 10
    replaced, replaced_names = causal_inputs(modified, direction)
    prefix, prefix_names = causal_inputs(frame.loc[frame.index < cutoff], direction)
    completed = original.index + pd.Timedelta(minutes=5) <= cutoff
    assert names == replaced_names == prefix_names
    pd.testing.assert_frame_equal(original.loc[completed], replaced.loc[completed])
    pd.testing.assert_frame_equal(original.loc[completed], prefix)


@pytest.mark.parametrize("direction", ["boom", "crash"])
def test_partial_final_candle_cannot_be_eligible(direction):
    full = history()
    cutoff = full.index[2102]
    prefix, _ = causal_inputs(full.loc[full.index < cutoff], direction)
    assert prefix.index[-1] == cutoff.floor("5min")
    assert prefix.iloc[-1][["open", "high", "low", "close"]].isna().all()
    assert not prefix.feature_valid.iloc[-1]
    completed, _ = causal_inputs(full.loc[full.index < cutoff.floor("5min")], direction)
    pd.testing.assert_frame_equal(prefix.iloc[:-1], completed)


@pytest.mark.parametrize("remove_minute", [True, False])
def test_gap_is_not_compressed_and_event_age_returns_to_unknown(remove_minute):
    frame = history()
    missing = frame.index[1200]
    if remove_minute:
        frame = frame.drop(missing)
    else:
        # A partially missing OHLC record must invalidate the whole minute.
        frame.loc[missing, "high"] = np.nan
    inputs, names = causal_inputs(frame, "boom")
    gap = inputs.index.get_loc(missing.floor("5min"))
    assert inputs.feature_valid.iloc[gap - 1]
    assert inputs.iloc[gap][["open", "high", "low", "close"]].isna().all()
    assert not inputs.feature_valid.iloc[gap:gap + 120].any()
    assert inputs.log1p_large_bar_age.iloc[gap:360].isna().all()
    assert inputs.log1p_large_bar_age.iloc[360] == 0
    assert inputs.feature_valid.iloc[360]
    assert np.isfinite(inputs.loc[inputs.feature_valid, names]).all().all()
    assert (inputs.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()


def test_known_completed_event_age_caps_and_initial_history_stays_unknown():
    frame = history()
    frame.loc[frame.index[1800], "high"] -= 3
    inputs, _ = causal_inputs(frame, "boom")
    assert inputs.log1p_large_bar_age.iloc[:30].isna().all()
    assert not inputs.feature_valid.iloc[:30].any()
    assert inputs.large_completed_bar.iloc[30] == 1
    assert inputs.log1p_large_bar_age.iloc[30] == 0
    assert inputs.log1p_large_bar_age.iloc[31] == pytest.approx(np.log(2))
    assert inputs.log1p_large_bar_age.iloc[318] == pytest.approx(np.log1p(288))
    assert inputs.log1p_large_bar_age.iloc[-1] == pytest.approx(np.log1p(288))


@pytest.mark.parametrize("direction, side", [("boom", 1), ("crash", -1)])
def test_declared_inputs_match_closedbar_formulas(direction, side):
    frame = history()
    inputs, _ = causal_inputs(frame, direction)
    i = 140
    row = inputs.iloc[i]
    atr = row.atr
    for lag in (1, 3, 6, 12):
        assert row[f"aligned_return_{lag}_atr"] == pytest.approx(
            side * (row.close - inputs.close.iloc[i - lag]) / atr,
        )
    assert row.aligned_body_atr == pytest.approx(side * (row.close - row.open) / atr)
    assert row.range_atr == pytest.approx((row.high - row.low) / atr)
    assert row.body_over_range == pytest.approx(abs(row.close - row.open) / (row.high - row.low))
    assert row.log_atr_over_price == pytest.approx(np.log(atr / row.close))
    preceding = inputs.iloc[i - 12:i]
    favorable = preceding.high.max() if side == 1 else preceding.low.min()
    adverse = preceding.low.min() if side == 1 else preceding.high.max()
    assert row.aligned_distance_prior_favorable_12_atr == pytest.approx(side * (row.close - favorable) / atr)
    assert row.aligned_distance_prior_adverse_12_atr == pytest.approx(side * (row.close - adverse) / atr)
    # Neither reference includes the current candle's extreme.
    changed = frame.copy()
    current_minute = inputs.index[i]
    changed.loc[current_minute, "high"] += 10
    altered, _ = causal_inputs(changed, direction)
    if side == 1:
        assert altered.iloc[i].aligned_distance_prior_favorable_12_atr * altered.iloc[i].atr == pytest.approx(
            row.aligned_distance_prior_favorable_12_atr * row.atr,
        )
    else:
        assert altered.iloc[i].aligned_distance_prior_adverse_12_atr * altered.iloc[i].atr == pytest.approx(
            row.aligned_distance_prior_adverse_12_atr * row.atr,
        )


def training_matrix():
    rng = np.random.default_rng(34)
    X = rng.normal(size=(240, 3))
    X[:, 2] = 4  # A constant column must not cause singular scaling.
    y = 0.7 + 2 * X[:, 0] - 0.5 * X[:, 1]
    return X, y, ["first", "second", "constant"]


def test_exact_linear_fit_and_constant_column_with_tiny_penalty():
    X, y, names = training_matrix()
    model = fit_ridge(X, y, names, penalty=1e-12)
    prediction = predict_ridge(X, model, names)
    np.testing.assert_allclose(prediction, y, atol=1e-10)
    assert model["std"][2] == 1
    assert model["coefs"][2] == 0
    assert model["intercept"] == pytest.approx(y.mean())
    assert model["threshold"] == pytest.approx(max(0, np.quantile(prediction, 0.75)))
    assert model["score_kind"] == "continuous_uncalibrated_score"


def test_penalty_matches_mean_sse_objective_without_penalizing_intercept():
    X, y, names = training_matrix()
    model = fit_ridge(X, y, names, penalty=0.1)
    Z = (X - X.mean(axis=0)) / np.asarray(model["std"])
    coef = np.asarray(model["coefs"])
    residual = model["intercept"] + Z @ coef - y
    np.testing.assert_allclose((Z.T @ residual) / len(y) + 0.1 * coef, 0, atol=1e-12)
    assert residual.mean() == pytest.approx(0, abs=1e-12)
    # A large target level is not shrunk toward zero by the coefficient penalty.
    constant = fit_ridge(X, np.full(len(y), 20), names)
    np.testing.assert_allclose(predict_ridge(X, constant), 20)


def test_json_roundtrip_and_prediction_do_not_change_scaling_or_cutoff():
    X, y, names = training_matrix()
    model = fit_ridge(pd.DataFrame(X, columns=names), y, names)
    serialized = json.dumps(model, allow_nan=False, sort_keys=True)
    restored = json.loads(serialized)
    test = pd.DataFrame(X[:20] * 100 + 700, columns=names)
    expected = model["intercept"] + (
        (test.to_numpy() - np.asarray(model["means"])) / np.asarray(model["std"])
    ) @ np.asarray(model["coefs"])
    np.testing.assert_allclose(predict_ridge(test, restored), expected)
    assert json.dumps(restored, allow_nan=False, sort_keys=True) == serialized
    assert model["means"] == pytest.approx(X.mean(axis=0))
    train_scores = predict_ridge(X, restored)
    assert restored["threshold"] == pytest.approx(max(0, np.quantile(train_scores, 0.75)))
    assert not restored["features_clipped"] and not restored["target_clipped"]
    # A tail training outcome enters the fit as supplied rather than being capped.
    tail = y.copy()
    tail[-1] = 1000
    tail_model = fit_ridge(X, tail, names)
    assert tail_model["intercept"] == pytest.approx(tail.mean())


def test_negative_training_score_quantile_is_floored_at_zero():
    X, _, names = training_matrix()
    model = fit_ridge(X, np.full(len(X), -0.3), names)
    assert model["threshold"] == 0
    np.testing.assert_allclose(predict_ridge(X, model), -0.3)


def test_ordered_schema_is_enforced_for_dataframes_and_optional_array_names():
    X, y, names = training_matrix()
    model = fit_ridge(X, y, names)
    reordered = pd.DataFrame(X, columns=names)[names[::-1]]
    with pytest.raises(ValueError, match="ordered feature schema"):
        fit_ridge(reordered, y, names)
    with pytest.raises(ValueError, match="ordered feature schema"):
        predict_ridge(reordered, model)
    with pytest.raises(ValueError, match="feature order"):
        predict_ridge(X, model, names[::-1])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_and_targets_are_refused(invalid):
    X, y, names = training_matrix()
    bad = X.copy()
    bad[3, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        fit_ridge(bad, y, names)
    model = fit_ridge(X, y, names)
    with pytest.raises(ValueError, match="finite"):
        predict_ridge(bad, model)
    target = y.copy()
    target[3] = invalid
    with pytest.raises(ValueError, match="finite"):
        fit_ridge(X, target, names)


def test_insufficient_samples_and_invalid_shapes_are_refused():
    X, y, names = training_matrix()
    with pytest.raises(ValueError, match="100"):
        fit_ridge(X[:99], y[:99], names)
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_ridge(X, y.reshape(-1, 1), names)
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_ridge(X[:, 0], y, names)
    with pytest.raises(ValueError, match="unique"):
        fit_ridge(X, y, ["same"] * 3)


@pytest.mark.parametrize("change", ["bad_scale", "bad_coefficient", "bad_shape", "version"])
def test_corrupted_frozen_model_is_refused(change):
    X, y, names = training_matrix()
    model = fit_ridge(X, y, names)
    if change == "bad_scale":
        model["std"][0] = 0
    elif change == "bad_coefficient":
        model["coefs"][0] = np.nan
    elif change == "bad_shape":
        model["means"].pop()
    else:
        model["version"] = 2
    with pytest.raises(ValueError):
        predict_ridge(X, model)


@pytest.mark.parametrize("change", ["unordered", "duplicate", "naive", "seconds", "infinite", "bounds"])
def test_malformed_source_is_refused(change):
    frame = history()
    if change == "unordered":
        frame = frame.iloc[::-1]
    elif change == "duplicate":
        frame = pd.concat([frame.iloc[:10], frame.iloc[9:]])
    elif change == "naive":
        frame.index = frame.index.tz_localize(None)
    elif change == "seconds":
        frame.index += pd.Timedelta(seconds=1)
    elif change == "infinite":
        frame.iloc[10, 1] = np.inf
    else:
        frame.iloc[10, 1] = frame.iloc[10, 2] - 1
    with pytest.raises(ValueError):
        causal_inputs(frame, "boom")
