"""Deterministic, training-only histogram residual boosting for research.

Start with F0=mean(y). For each tree use residuals r=y-F. A leaf of n rows
and residual sum S minimizes sum((r-v)**2)+lambda*v**2 at v=S/(n+lambda).
The split gain, in sum-SSE units, is SL**2/(nL+lambda)+SR**2/(nR+lambda)
-S**2/(n+lambda). Eligible children each contain at least min_leaf rows. A
root's gain must be >= 0; deeper gains must be > 0, with no epsilon tolerance.
A zero-gain root can reveal a balanced pair interaction with useful descendants.
Balanced three-way interactions needing a second zero-gain split are therefore
not learned. Choose the largest gain, breaking exact ties by ascending feature
position, then split-bin index.
Leaf contributions enter F with the learning_rate; there is no line search,
sampling, target clipping, feature clipping, early stopping or outcome filtering.

For a feature with <=n_bins distinct TRAINING values, use adjacent-value
midpoints (constant=>no cuts); compute a/2+b/2 to avoid subtraction overflow.
If floating precision yields midpoint<=a, use b, so the two known values remain
separable. Otherwise use unique internal training quantiles at k/n_bins,
k=1..n_bins-1, with numpy's linear method; minimum/maximum cuts are discarded.
All boundaries are fitted once. Binning uses searchsorted(side="right").
Thus a split at bin k sends x < boundary[k] left and x >= boundary[k] right.
Unseen tails route to the outer bins without replacing their numeric inputs or
refitting boundaries. The frozen issuance cutoff is max(0, q75(training F)).
Scores are continuous regression outputs, never calibrated probabilities.

The caller must supply completed, partition-purged training labels. This module
has no price collection, chronology selection, trading or historical-file API.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Real

import numpy as np
import pandas as pd


MIN_FIT_ROWS = 1000
DEFAULT_PARAMETERS = {
    "n_trees": 100,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_leaf": 200,
    "n_bins": 16,
    "leaf_regularization": 20.0,
    "quantile": 0.75,
}
_PROVENANCE = {
    "score_kind": "continuous_uncalibrated_score",
    "objective": "sum_squared_residual_error_plus_lambda_times_squared_leaf_values",
    "boundaries_fitted_on": "training_rows_only",
    "cutoff_fitted_on": "training_predictions_only",
    "quantile_method": "linear",
    "boundary_method": "training_low_cardinality_adjacent_midpoints_else_unique_internal_linear_quantiles",
    "bin_side": "right",
    "split_tie_break": "ascending_feature_position_then_split_bin",
    "allow_zero_gain_split": True,
    "zero_gain_split_scope": "root_only_no_epsilon_tolerance",
    "target_clipped": False,
    "features_clipped": False,
    "completed_labels_required": True,
}
_MODEL_KEYS = {
    "version", "feature_names", "parameters", "defaults", "fit_rows",
    "initial_prediction", "bin_boundaries", "trees", "training_score_quantile",
    "threshold", "integrity_sha256", *_PROVENANCE,
}


def _names(feature_names: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(feature_names, (list, tuple)):
        raise ValueError("Feature names must be an ordered list or tuple")
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Feature names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique and ordered")
    return names


def _matrix(X: np.ndarray | pd.DataFrame, names: list[str]) -> np.ndarray:
    if isinstance(X, pd.DataFrame) and list(X.columns) != names:
        raise ValueError("DataFrame columns must exactly match the ordered feature schema")
    try:
        values = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("X must contain finite numeric inputs") from error
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("X must be a two-dimensional matrix matching the feature schema")
    if not np.isfinite(values).all():
        raise ValueError("Every input feature must be finite")
    return values


def _real(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _integer(value: object, field: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _parameters(parameters: dict) -> dict:
    if not isinstance(parameters, dict) or set(parameters) != set(DEFAULT_PARAMETERS):
        raise ValueError("Model parameters must match the declared parameter schema")
    result = {name: _integer(parameters[name], name, minimum) for name, minimum in (
        ("n_trees", 1), ("max_depth", 0), ("min_leaf", 1), ("n_bins", 2),
    )}
    for name in ("learning_rate", "leaf_regularization", "quantile"):
        result[name] = _real(parameters[name], name)
    if not 0 < result["learning_rate"] <= 1:
        raise ValueError("learning_rate must be > 0 and <= 1")
    if result["leaf_regularization"] < 0:
        raise ValueError("leaf_regularization must be nonnegative")
    if not 0 <= result["quantile"] <= 1:
        raise ValueError("quantile must lie between zero and one")
    return result


def _digest(model: dict) -> str:
    try:
        payload = json.dumps(
            {key: value for key, value in model.items() if key != "integrity_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Frozen model must be finite JSON-serializable data") from error
    return hashlib.sha256(payload).hexdigest()


def _bin_values(values: np.ndarray, boundaries: list[list[float]]) -> np.ndarray:
    return np.column_stack([
        np.searchsorted(cuts, values[:, feature], side="right")
        for feature, cuts in enumerate(boundaries)
    ])


def _score(total: float, count: int, regularization: float) -> float:
    result = total * (total / (count + regularization))
    if not np.isfinite(result):
        raise ValueError("Training arithmetic produced non-finite split scores")
    return result


def _tree(bins: np.ndarray, residual: np.ndarray, indices: np.ndarray,
          boundaries: list[list[float]], parameters: dict, depth: int = 0) -> dict:
    count = len(indices)
    total = float(residual[indices].sum())
    if not np.isfinite(total):
        raise ValueError("Training arithmetic produced non-finite residual sums")
    regularization = parameters["leaf_regularization"]
    node = {"count": count, "residual_sum": total,
            "value": total / (count + regularization)}
    if depth >= parameters["max_depth"] or count < 2 * parameters["min_leaf"]:
        return node
    parent_score = _score(total, count, regularization)
    best = None
    best_gain = -np.inf
    for feature, cuts in enumerate(boundaries):
        if not cuts:
            continue
        codes = bins[indices, feature]
        counts = np.bincount(codes, minlength=len(cuts) + 1)
        sums = np.bincount(codes, weights=residual[indices], minlength=len(cuts) + 1)
        left_counts = counts.cumsum()[:-1]
        left_sums = sums.cumsum()[:-1]
        for split_bin, (left_count, left_sum) in enumerate(zip(left_counts, left_sums)):
            left_count = int(left_count)
            right_count = count - left_count
            if min(left_count, right_count) < parameters["min_leaf"]:
                continue
            gain = (_score(float(left_sum), left_count, regularization)
                    + _score(total - float(left_sum), right_count, regularization)
                    - parent_score)
            # Traversal order is the tie break; equal gains never replace best.
            eligible_gain = gain >= 0 if depth == 0 else gain > 0
            if eligible_gain and gain > best_gain:
                best = (feature, split_bin)
                best_gain = gain
    if best is None:
        return node
    feature, split_bin = best
    left_indices = indices[bins[indices, feature] <= split_bin]
    right_indices = indices[bins[indices, feature] > split_bin]
    left = _tree(bins, residual, left_indices, boundaries, parameters, depth + 1)
    right = _tree(bins, residual, right_indices, boundaries, parameters, depth + 1)
    # Store gain from the actual child sums, avoiding histogram-roundoff drift.
    gain = (_score(left["residual_sum"], left["count"], regularization)
            + _score(right["residual_sum"], right["count"], regularization)
            - parent_score)
    if gain < 0 or (depth > 0 and gain == 0):
        return node
    node.update({"split_feature": feature, "split_bin": split_bin,
                 "gain": gain, "left": left, "right": right})
    return node


def _tree_prediction(bins: np.ndarray, tree: dict) -> np.ndarray:
    prediction = np.empty(len(bins), dtype=float)

    def visit(node: dict, indices: np.ndarray) -> None:
        if "split_feature" not in node:
            prediction[indices] = node["value"]
            return
        mask = bins[indices, node["split_feature"]] <= node["split_bin"]
        visit(node["left"], indices[mask])
        visit(node["right"], indices[~mask])

    visit(tree, np.arange(len(bins)))
    return prediction


def fit_histogram_boost(
    X: np.ndarray | pd.DataFrame, y: np.ndarray,
    feature_names: list[str] | tuple[str, ...], *,
    n_trees: int = 100, learning_rate: float = 0.05, max_depth: int = 3,
    min_leaf: int = 200, n_bins: int = 16, leaf_regularization: float = 20.0,
    quantile: float = 0.75,
) -> dict:
    """Fit complete training labels only; retain the entire serializable model.

    Default parameters define the fixed research learner. Explicit overrides
    support synthetic verification or a separately declared study; all actual
    values and defaults are stored in the model. No validation inputs or labels
    enter binning, fitting or the issuance cutoff.
    """
    names = _names(feature_names)
    values = _matrix(X, names)
    if len(values) < MIN_FIT_ROWS:
        raise ValueError(f"At least {MIN_FIT_ROWS} completed training rows are required")
    try:
        target = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("y must contain finite numeric targets") from error
    if target.ndim != 1 or len(target) != len(values) or not np.isfinite(target).all():
        raise ValueError("y must be a finite one-dimensional target matching X")
    parameters = _parameters({
        "n_trees": n_trees, "learning_rate": learning_rate, "max_depth": max_depth,
        "min_leaf": min_leaf, "n_bins": n_bins,
        "leaf_regularization": leaf_regularization, "quantile": quantile,
    })
    boundaries = []
    fractions = np.arange(1, parameters["n_bins"]) / parameters["n_bins"]
    for feature in range(len(names)):
        column = values[:, feature]
        distinct = np.unique(column)
        if len(distinct) <= parameters["n_bins"]:
            left, right = distinct[:-1], distinct[1:]
            cuts = left / 2 + right / 2
            cuts = np.where(cuts <= left, right, cuts)
        else:
            cuts = np.unique(np.quantile(column, fractions, method="linear"))
            cuts = cuts[(cuts > column.min()) & (cuts < column.max())]
        if not np.isfinite(cuts).all():
            raise ValueError("Training arithmetic produced non-finite bin boundaries")
        boundaries.append(cuts.tolist())
    bins = _bin_values(values, boundaries)
    initial = float(target.mean())
    if not np.isfinite(initial):
        raise ValueError("Training arithmetic produced a non-finite initial prediction")
    prediction = np.full(len(values), initial)
    trees = []
    indices = np.arange(len(values))
    for _ in range(parameters["n_trees"]):
        residual = target - prediction
        if not np.isfinite(residual).all():
            raise ValueError("Training arithmetic produced non-finite residuals")
        tree = _tree(bins, residual, indices, boundaries, parameters)
        prediction += parameters["learning_rate"] * _tree_prediction(bins, tree)
        if not np.isfinite(prediction).all():
            raise ValueError("Training arithmetic produced non-finite predictions")
        trees.append(tree)
    score_quantile = float(np.quantile(prediction, parameters["quantile"], method="linear"))
    model = {
        "version": 1, "feature_names": names, "parameters": parameters,
        "defaults": DEFAULT_PARAMETERS.copy(), "fit_rows": len(values),
        "initial_prediction": initial, "bin_boundaries": boundaries,
        "trees": trees, "training_score_quantile": score_quantile,
        "threshold": max(0.0, score_quantile), **_PROVENANCE,
    }
    model["integrity_sha256"] = _digest(model)
    _validated_model(model)
    return model


def _validated_model(model: dict) -> tuple[list[str], dict, list[list[float]]]:
    if not isinstance(model, dict) or set(model) != _MODEL_KEYS:
        raise ValueError("Frozen model fields must exactly match the supported schema")
    if type(model["version"]) is not int or model["version"] != 1:
        raise ValueError("Unsupported histogram model version")
    if model["integrity_sha256"] != _digest(model):
        raise ValueError("Frozen model integrity check failed: model was modified")
    names = _names(model["feature_names"])
    parameters = _parameters(model["parameters"])
    defaults = _parameters(model["defaults"])
    if defaults != DEFAULT_PARAMETERS or any(
        type(model[key]) is not type(value) or model[key] != value
        for key, value in _PROVENANCE.items()
    ):
        raise ValueError("Frozen model provenance differs from the supported learner")
    fit_rows = _integer(model["fit_rows"], "fit_rows", MIN_FIT_ROWS)
    _real(model["initial_prediction"], "initial_prediction")
    score_quantile = _real(model["training_score_quantile"], "training_score_quantile")
    threshold = _real(model["threshold"], "threshold")
    if threshold != max(0.0, score_quantile):
        raise ValueError("Frozen threshold must equal max(0, training score quantile)")
    boundaries = model["bin_boundaries"]
    if not isinstance(boundaries, list) or len(boundaries) != len(names):
        raise ValueError("Frozen bin boundary dimensions must match the feature schema")
    for cuts in boundaries:
        if not isinstance(cuts, list) or len(cuts) >= parameters["n_bins"]:
            raise ValueError("Frozen bin boundaries must be lists with fewer than n_bins entries")
        numbers = [_real(cut, "bin boundary") for cut in cuts]
        if any(right <= left for left, right in zip(numbers, numbers[1:])):
            raise ValueError("Frozen bin boundaries must be strictly increasing")
    trees = model["trees"]
    if not isinstance(trees, list) or len(trees) != parameters["n_trees"]:
        raise ValueError("Frozen tree count differs from n_trees")
    regularization = parameters["leaf_regularization"]

    def validate_node(node: dict, depth: int, root: bool = False) -> tuple[int, float]:
        leaf_keys = {"count", "residual_sum", "value"}
        split_keys = {*leaf_keys, "split_feature", "split_bin", "gain", "left", "right"}
        if not isinstance(node, dict) or set(node) not in (leaf_keys, split_keys):
            raise ValueError("Frozen tree node has malformed fields")
        count = _integer(node["count"], "node count", 1 if root else parameters["min_leaf"])
        if root and count != fit_rows:
            raise ValueError("Frozen root count differs from fit_rows")
        total = _real(node["residual_sum"], "node residual_sum")
        value = _real(node["value"], "node value")
        if not np.isclose(value, total / (count + regularization), rtol=1e-12, atol=1e-12):
            raise ValueError("Frozen leaf value differs from the regularized residual mean")
        if set(node) == leaf_keys:
            return count, total
        if depth >= parameters["max_depth"]:
            raise ValueError("Frozen tree exceeds max_depth")
        feature = _integer(node["split_feature"], "split_feature", 0)
        split_bin = _integer(node["split_bin"], "split_bin", 0)
        if feature >= len(names) or split_bin >= len(boundaries[feature]):
            raise ValueError("Frozen tree split falls outside the feature bins")
        gain = _real(node["gain"], "split gain")
        if gain < 0 or (depth > 0 and gain == 0):
            raise ValueError("Frozen split gain must be nonnegative at the root and positive deeper")
        left_count, left_sum = validate_node(node["left"], depth + 1)
        right_count, right_sum = validate_node(node["right"], depth + 1)
        if left_count + right_count != count:
            raise ValueError("Frozen child counts do not sum to their parent")
        tolerance = 1e-10 * max(1.0, abs(total), abs(left_sum), abs(right_sum))
        if abs(left_sum + right_sum - total) > tolerance:
            raise ValueError("Frozen child residual sums do not sum to their parent")
        expected_gain = (_score(left_sum, left_count, regularization)
                         + _score(right_sum, right_count, regularization)
                         - _score(total, count, regularization))
        if not np.isclose(gain, expected_gain, rtol=1e-10, atol=1e-10):
            raise ValueError("Frozen split gain differs from regularized sum-SSE improvement")
        return count, total

    try:
        for tree in trees:
            validate_node(tree, 0, root=True)
    except RecursionError as error:
        raise ValueError("Frozen tree contains excessive or cyclic nesting") from error
    return names, parameters, boundaries


def predict_histogram_boost(
    X: np.ndarray | pd.DataFrame, model: dict,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Apply frozen bins and trees without refitting or mutating the model.

    DataFrame columns must match exactly. Array columns follow the frozen model
    order; supplying feature_names additionally checks the caller's array schema.
    Empty, correctly shaped prediction matrices produce an empty score vector.
    An integrity checksum and structural/math checks refuse modified model data;
    the checksum is corruption detection, not a cryptographic trust signature.
    """
    names, parameters, boundaries = _validated_model(model)
    if feature_names is not None and _names(feature_names) != names:
        raise ValueError("Prediction feature order differs from the frozen model")
    values = _matrix(X, names)
    bins = _bin_values(values, boundaries)
    prediction = np.full(len(values), float(model["initial_prediction"]))
    for tree in model["trees"]:
        prediction += parameters["learning_rate"] * _tree_prediction(bins, tree)
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction arithmetic produced non-finite scores")
    return prediction
