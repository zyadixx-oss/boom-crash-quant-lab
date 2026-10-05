"""Synthetic checks for frozen nonlinear scores, not historical performance."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.research.nonlinear_signal import (
    DEFAULT_PARAMETERS,
    MIN_FIT_ROWS,
    fit_histogram_boost,
    predict_histogram_boost,
)


def interaction():
    # Each feature has zero marginal signal; their quadrant agreement matters.
    X = np.repeat(np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]), 640, axis=0)
    y = np.repeat(np.array([1., -1., -1., 1.]), 640)
    return X, y, ["context", "timing"]


@pytest.fixture(scope="module")
def nonlinear_model():
    X, y, names = interaction()
    return fit_histogram_boost(X, y, names)


def small_model():
    X = np.arange(1200, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 600).astype(float) * 2
    return X, y, ["feature"], fit_histogram_boost(
        X, y, ["feature"], n_trees=2, n_bins=2,
    )


def repaired_checksum(model):
    """Exercise structural refusals even after accidental checksum replacement."""
    payload = {key: value for key, value in model.items() if key != "integrity_sha256"}
    model["integrity_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return model


def test_balanced_interaction_is_learned_without_marginal_signal(nonlinear_model):
    X, y, names = interaction()
    assert np.mean(y[X[:, 0] == 0]) == np.mean(y[X[:, 0] == 1]) == 0
    assert np.mean(y[X[:, 1] == 0]) == np.mean(y[X[:, 1] == 1]) == 0
    prediction = predict_histogram_boost(X, nonlinear_model, names)
    assert np.mean((prediction - y) ** 2) < 0.0001
    assert np.array_equal(prediction > 0, y > 0)
    root = nonlinear_model["trees"][0]
    assert root["gain"] == 0
    assert root["split_feature"] == 0
    assert root["left"]["gain"] > 0 and root["right"]["gain"] > 0
    assert len(nonlinear_model["trees"]) == 100
    assert nonlinear_model["parameters"] == DEFAULT_PARAMETERS
    assert nonlinear_model["defaults"] == DEFAULT_PARAMETERS


def test_regularized_leaf_math_and_learning_rate_are_sum_sse_units():
    X = np.repeat(np.array([[0.], [1.]]), 600, axis=0)
    y = np.repeat([0., 2.], 600)
    model = fit_histogram_boost(X, y, ["x"], n_trees=1, n_bins=2)
    root = model["trees"][0]
    assert model["initial_prediction"] == 1
    assert root["count"] == 1200 and root["residual_sum"] == 0
    assert root["value"] == 0
    for child, total in ((root["left"], -600), (root["right"], 600)):
        assert child["count"] == 600
        assert child["residual_sum"] == total
        assert child["value"] == pytest.approx(total / (600 + 20))
    assert root["gain"] == pytest.approx(2 * 600 ** 2 / 620)
    prediction = predict_histogram_boost(X, model)
    np.testing.assert_allclose(prediction[:600], 1 - .05 * 600 / 620)
    np.testing.assert_allclose(prediction[600:], 1 + .05 * 600 / 620)


def test_bins_and_threshold_remain_training_only_on_unseen_tails():
    rng = np.random.default_rng(37)
    X = rng.normal(size=(1400, 2))
    y = np.sin(X[:, 0]) + .2 * X[:, 1] ** 2
    model = fit_histogram_boost(X, y, ["a", "b"], n_trees=6)
    for feature in range(2):
        expected = np.unique(np.quantile(X[:, feature], np.arange(1, 16) / 16, method="linear"))
        np.testing.assert_array_equal(model["bin_boundaries"][feature], expected)
    serialized = json.dumps(model, sort_keys=True, allow_nan=False)
    # Only outer-bin routing can affect these predictions; neither tail refits.
    tails = np.array([[-1e9, -1e9], [1e9, 1e9]])
    scores = predict_histogram_boost(tails, model)
    bounds = np.column_stack([X.min(axis=0), X.max(axis=0)]).T
    np.testing.assert_array_equal(scores, predict_histogram_boost(bounds, model))
    assert json.dumps(model, sort_keys=True, allow_nan=False) == serialized
    training_scores = predict_histogram_boost(X, model)
    assert model["training_score_quantile"] == pytest.approx(np.quantile(training_scores, .75))
    assert model["threshold"] == max(0, model["training_score_quantile"])
    assert model["boundaries_fitted_on"] == "training_rows_only"
    assert model["cutoff_fitted_on"] == "training_predictions_only"
    assert not model["features_clipped"] and not model["target_clipped"]


def test_json_roundtrip_parity_and_no_prediction_mutation(nonlinear_model):
    X, _, names = interaction()
    original = json.dumps(nonlinear_model, sort_keys=True, allow_nan=False)
    restored = json.loads(original)
    frame = pd.DataFrame(X[::10], columns=names)
    np.testing.assert_array_equal(
        predict_histogram_boost(frame, restored),
        predict_histogram_boost(frame.to_numpy(), nonlinear_model, names),
    )
    assert json.dumps(restored, sort_keys=True, allow_nan=False) == original
    # Scoring a row alone or alongside extreme new rows cannot recalibrate it.
    solo = predict_histogram_boost(frame.iloc[:1], restored)[0]
    together = pd.concat([frame.iloc[:1], pd.DataFrame([[1e6, -1e6]], columns=names)])
    assert predict_histogram_boost(together, restored)[0] == solo


def test_fit_is_bitwise_deterministic_on_identical_order():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(1600, 4))
    y = X[:, 0] * X[:, 1] + .7 * (X[:, 2] > 0)
    first = fit_histogram_boost(X, y, ["a", "b", "c", "d"], n_trees=8)
    second = fit_histogram_boost(X, y, ["a", "b", "c", "d"], n_trees=8)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_exact_ties_use_first_feature_then_first_eligible_bin():
    column = np.arange(1600, dtype=float)
    X = np.column_stack([column, column])
    model = fit_histogram_boost(X, np.zeros(len(X)), ["a", "same"], n_trees=1, max_depth=1)
    root = model["trees"][0]
    assert root["gain"] == 0
    assert root["split_feature"] == 0
    # q1/16 has only 100 rows; q2/16 is the first with min_leaf=200.
    assert root["split_bin"] == 1
    assert root["left"]["count"] == 200


def test_zero_gain_is_allowed_only_at_root_without_epsilon():
    # A three-way parity has zero gain at its first two levels. Refuse the
    # second zero-gain split rather than claiming this learner captures it.
    corners = np.array([[a, b, c] for a in (0., 1.) for b in (0., 1.) for c in (0., 1.)])
    X = np.repeat(corners, 400, axis=0)
    target = np.prod(2 * X - 1, axis=1)
    model = fit_histogram_boost(X, target, ["a", "b", "c"], n_trees=3)
    for root in model["trees"]:
        assert root["gain"] == 0
        assert "split_feature" not in root["left"]
        assert "split_feature" not in root["right"]
    np.testing.assert_array_equal(predict_histogram_boost(X, model), 0)
    assert model["zero_gain_split_scope"] == "root_only_no_epsilon_tolerance"


def test_boundary_equality_goes_right_and_extreme_values_stay_outer_bins():
    X, _, names, model = small_model()
    boundary = model["bin_boundaries"][0][0]
    values = np.array([[-1e12], [np.nextafter(boundary, -np.inf)], [boundary], [1e12]])
    scores = predict_histogram_boost(values, model, names)
    assert scores[0] == scores[1]
    assert scores[2] == scores[3]
    assert scores[0] < scores[2]
    assert model["bin_side"] == "right"
    assert np.array_equal(X[:, 0], np.arange(1200))


def test_constant_features_produce_no_cuts_or_splits():
    X = np.full((1200, 3), [5., 7., -2.])
    y = np.linspace(-2, 4, len(X))
    model = fit_histogram_boost(X, y, ["a", "b", "c"], n_trees=4)
    assert model["bin_boundaries"] == [[], [], []]
    assert all(set(tree) == {"count", "residual_sum", "value"} for tree in model["trees"])
    np.testing.assert_allclose(predict_histogram_boost(X * 1000, model), y.mean(), atol=1e-12)


def test_discrete_columns_use_adjacent_value_midpoint():
    X = np.repeat(np.array([[0.], [1.]]), 600, axis=0)
    model = fit_histogram_boost(X, np.ones(len(X)), ["binary"], n_trees=1)
    assert model["bin_boundaries"] == [[.5]]
    assert len(model["trees"]) == 1


def test_rare_binary_feature_is_separable_even_when_all_internal_quantiles_are_zero():
    X = np.r_[np.zeros(6000), np.ones(200)].reshape(-1, 1)
    y = 2 * X[:, 0]
    assert np.unique(np.quantile(X[:, 0], np.arange(1, 16) / 16)).tolist() == [0.]
    model = fit_histogram_boost(X, y, ["rare_completed_event"])
    assert model["bin_boundaries"] == [[.5]]
    assert model["trees"][0]["right"]["count"] == 200
    prediction = predict_histogram_boost(np.array([[0.], [1.]]), model)
    assert prediction[0] < .01 and prediction[1] > 1.95


def test_low_cardinality_boundaries_are_train_only_and_unknown_categories_do_not_refit():
    X = np.repeat(np.array([[0.], [10.], [20.]]), 500, axis=0)
    y = X[:, 0] / 10
    model = fit_histogram_boost(X, y, ["category"], n_trees=4)
    assert model["bin_boundaries"] == [[5., 15.]]
    frozen = json.dumps(model, sort_keys=True)
    unknown = np.array([[-1e10], [7.], [25.]])
    known = np.array([[0.], [10.], [20.]])
    np.testing.assert_array_equal(predict_histogram_boost(unknown, model), predict_histogram_boost(known, model))
    assert json.dumps(model, sort_keys=True) == frozen


@pytest.mark.parametrize("left,right", [
    (-1e308, 1e308), (1e308, np.nextafter(1e308, np.inf)),
    (0., np.nextafter(0., np.inf)), (-np.nextafter(0., np.inf), 0.),
])
def test_low_cardinality_midpoint_arithmetic_remains_finite_and_separates_known_values(left, right):
    X = np.repeat(np.array([[left], [right]]), 600, axis=0)
    y = np.repeat([0., 1.], 600)
    model = fit_histogram_boost(X, y, ["x"], n_trees=1)
    cut = model["bin_boundaries"][0][0]
    assert np.isfinite(cut) and left < cut <= right
    prediction = predict_histogram_boost(np.array([[left], [right]]), model)
    assert prediction[0] < prediction[1]


def test_negative_score_quantile_is_floored_and_scores_are_not_probabilities():
    X = np.ones((1200, 1))
    negative = fit_histogram_boost(X, np.full(1200, -.3), ["x"], n_trees=1)
    assert negative["threshold"] == 0
    np.testing.assert_allclose(predict_histogram_boost(X, negative), -.3)
    large = fit_histogram_boost(X, np.full(1200, 20.), ["x"], n_trees=1)
    assert large["threshold"] == 20
    np.testing.assert_array_equal(predict_histogram_boost(X, large), np.full(1200, 20.))
    assert large["score_kind"] == "continuous_uncalibrated_score"


def test_large_training_outcome_is_not_capped_and_inputs_are_unchanged():
    X = np.arange(1200, dtype=float).reshape(-1, 1)
    y = np.zeros(1200)
    y[-1] = 1e7
    before_X, before_y = X.copy(), y.copy()
    model = fit_histogram_boost(X, y, ["x"], n_trees=1)
    assert model["initial_prediction"] == pytest.approx(y.mean())
    np.testing.assert_array_equal(X, before_X)
    np.testing.assert_array_equal(y, before_y)
    cuts = np.quantile(X[:, 0], np.arange(1, 16) / 16, method="linear")
    np.testing.assert_array_equal(model["bin_boundaries"][0], cuts)


def test_schema_validation_checks_dataframe_and_explicit_array_order(nonlinear_model):
    X, y, names = interaction()
    reversed_frame = pd.DataFrame(X, columns=names)[names[::-1]]
    with pytest.raises(ValueError, match="ordered feature schema"):
        fit_histogram_boost(reversed_frame, y, names)
    with pytest.raises(ValueError, match="ordered feature schema"):
        predict_histogram_boost(reversed_frame, nonlinear_model)
    with pytest.raises(ValueError, match="feature order"):
        predict_histogram_boost(X, nonlinear_model, names[::-1])
    with pytest.raises(ValueError, match="two-dimensional"):
        predict_histogram_boost(X[:, :1], nonlinear_model)


@pytest.mark.parametrize("names", [[], ["x", "x"], [""], [3], "x"])
def test_malformed_feature_names_are_refused(names):
    with pytest.raises(ValueError, match="Feature names"):
        fit_histogram_boost(np.zeros((1200, 1)), np.zeros(1200), names)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_targets_and_predictions_are_refused(value, nonlinear_model):
    X, y, names = interaction()
    X[10, 0] = value
    with pytest.raises(ValueError, match="finite"):
        fit_histogram_boost(X, y, names)
    with pytest.raises(ValueError, match="finite"):
        predict_histogram_boost(X, nonlinear_model)
    X, y, names = interaction()
    y[10] = value
    with pytest.raises(ValueError, match="finite"):
        fit_histogram_boost(X, y, names)


def test_insufficient_completed_labels_and_bad_target_shape_are_refused():
    X, y, names = interaction()
    assert MIN_FIT_ROWS == 1000
    with pytest.raises(ValueError, match="1000 completed"):
        fit_histogram_boost(X[:999], y[:999], names)
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_histogram_boost(X, y[:, None], names)
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_histogram_boost(X, y[:-1], names)
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_histogram_boost(X[:, 0], y, ["x"])


@pytest.mark.parametrize("change", [
    {"n_trees": 0}, {"n_trees": 1.0}, {"n_trees": True},
    {"max_depth": -1}, {"min_leaf": 0}, {"n_bins": 1},
    {"learning_rate": 0}, {"learning_rate": 1.1}, {"learning_rate": True},
    {"leaf_regularization": -1}, {"leaf_regularization": np.inf},
    {"quantile": -.1}, {"quantile": 1.1},
])
def test_invalid_training_parameters_are_refused(change):
    X, y, names = interaction()
    with pytest.raises(ValueError):
        fit_histogram_boost(X, y, names, **change)


def test_one_thousand_rows_are_accepted_and_empty_prediction_is_valid():
    X = np.ones((1000, 1))
    model = fit_histogram_boost(X, np.zeros(len(X)), ["x"], n_trees=1)
    assert model["fit_rows"] == 1000
    prediction = predict_histogram_boost(np.empty((0, 1)), model)
    assert prediction.shape == (0,)


def test_depth_and_child_minimum_are_respected():
    rng = np.random.default_rng(92)
    X = rng.uniform(size=(1700, 5))
    y = X[:, 0] * X[:, 1] + X[:, 2] ** 2
    model = fit_histogram_boost(X, y, ["a", "b", "c", "d", "e"], n_trees=4)

    def visit(node, depth=0):
        assert depth <= 3
        if "left" in node:
            assert node["left"]["count"] >= 200 and node["right"]["count"] >= 200
            assert node["gain"] >= 0
            visit(node["left"], depth + 1)
            visit(node["right"], depth + 1)

    for tree in model["trees"]:
        visit(tree)


@pytest.mark.parametrize("field,value", [
    ("threshold", 100), ("initial_prediction", 30),
    ("fit_rows", 1201), ("target_clipped", True),
])
def test_modified_model_fails_integrity_before_prediction(field, value):
    X, _, _, model = small_model()
    model[field] = value
    with pytest.raises(ValueError, match="integrity"):
        predict_histogram_boost(X, model)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m["bin_boundaries"].__setitem__(0, [1., 1.]), "strictly increasing"),
    (lambda m: m["bin_boundaries"].__setitem__(0, []), "outside"),
    (lambda m: m["trees"][0].__setitem__("split_feature", 7), "outside"),
    (lambda m: m["trees"][0].__setitem__("gain", -1), "nonnegative"),
    (lambda m: m["trees"][0].__setitem__("gain", 100), "sum-SSE"),
    (lambda m: m["trees"][0]["left"].__setitem__("count", 199), "node count"),
    (lambda m: m["trees"][0]["left"].__setitem__("value", 900), "regularized"),
    (lambda m: m["trees"].pop(), "tree count"),
    (lambda m: m.__setitem__("threshold", 100), "threshold"),
    (lambda m: m.__setitem__("score_kind", "probability"), "provenance"),
    (lambda m: m["parameters"].__setitem__("max_depth", 0), "max_depth"),
    (lambda m: m["trees"][0].__setitem__("count", 1201), "root count"),
    (lambda m: m["defaults"].__setitem__("n_trees", True), "n_trees"),
])
def test_structurally_invalid_model_refused_even_with_replaced_checksum(mutation, match):
    X, _, _, model = small_model()
    # Permit two boundaries to reach the strictly-increasing check for that case.
    model["parameters"]["n_bins"] = 16
    mutation(model)
    repaired_checksum(model)
    with pytest.raises(ValueError, match=match):
        predict_histogram_boost(X, model)


@pytest.mark.parametrize("model", [None, {}, {"version": 99}])
def test_missing_or_nonmapping_model_is_refused(model):
    with pytest.raises(ValueError, match="supported schema"):
        predict_histogram_boost(np.zeros((1, 1)), model)


def test_nonfinite_serialized_model_is_refused():
    X, _, _, model = small_model()
    model["bin_boundaries"][0][0] = np.nan
    with pytest.raises(ValueError, match="finite JSON"):
        predict_histogram_boost(X, model)


def test_real_multiframe_schema_can_be_fitted_using_only_synthetic_numbers():
    from app.research.multiframe_signal import FEATURE_NAMES

    rng = np.random.default_rng(104)
    frame = pd.DataFrame(rng.normal(size=(1600, 44)), columns=FEATURE_NAMES)
    target = frame.iloc[:, 0].to_numpy() * frame.iloc[:, -1].to_numpy()
    model = fit_histogram_boost(frame, target, FEATURE_NAMES, n_trees=4)
    assert model["feature_names"] == list(FEATURE_NAMES)
    assert len(model["bin_boundaries"]) == 44
    assert np.isfinite(predict_histogram_boost(frame.iloc[:30], model)).all()


def test_cyclic_model_data_is_refused_as_a_clear_validation_error():
    X, _, _, model = small_model()
    model["trees"][0]["left"] = model["trees"][0]
    with pytest.raises(ValueError, match="finite JSON"):
        predict_histogram_boost(X, model)


def test_zero_gain_descendant_is_refused_even_with_repaired_checksum(nonlinear_model):
    model = json.loads(json.dumps(nonlinear_model))
    model["trees"][0]["left"]["gain"] = 0
    repaired_checksum(model)
    with pytest.raises(ValueError, match="positive deeper"):
        predict_histogram_boost(np.array([[0., 0.]]), model)
