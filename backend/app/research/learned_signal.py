"""Causal continuous inputs and a frozen ridge score for offline research.

Every feature refers to a completed M5 candle, indexed by its opening time.
Its earliest issue time is therefore ``index + 5 minutes``. In the formulas
below s = +1 for Boom and -1 for Crash; A is the current mean TR(14), R is
high-low, B is abs(close-open), and every rolling window requires all rows.
Missing M5 candles remain on the grid and invalidate affected windows. Scores
are continuous regression outputs, never calibrated probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.research.spike_hunter import features


FEATURE_FORMULAS = {
    "aligned_return_1_atr": "s * (close - close.shift(1)) / A",
    "aligned_return_3_atr": "s * (close - close.shift(3)) / A",
    "aligned_return_6_atr": "s * (close - close.shift(6)) / A",
    "aligned_return_12_atr": "s * (close - close.shift(12)) / A",
    "aligned_body_atr": "s * (close - open) / A",
    "range_atr": "R / A",
    "favorable_wick_atr": "(high-max(open,close) for Boom; min(open,close)-low for Crash) / A",
    "adverse_wick_atr": "(min(open,close)-low for Boom; high-max(open,close) for Crash) / A",
    "favorable_close_position": "(close-low for Boom; high-close for Crash) / R",
    "atr_5_over_30": "TR.rolling(5).mean() / TR.rolling(30).mean()",
    "bb_width_over_prior_median_100": "W / W.shift(1).rolling(100).median(); W=4*std(close,20,ddof=0)/mean(close,20)",
    "range_over_prior_median_20": "R / R.shift(1).rolling(20).median()",
    "body_over_range": "B / R",
    "mean_range_3_over_prior_median_20": "R.rolling(3).mean() / R.shift(3).rolling(20).median()",
    "large_completed_bar": "float(R > 3 * A.shift(1)); unknown if candle or lagged A is missing",
    "log1p_large_bar_age": "log1p(min(M5 bars since last large_completed_bar,288)); unknown before first or after a gap",
    "log_atr_over_price": "log(A / close)",
    "aligned_distance_prior_favorable_12_atr": "s * (close - favorable preceding12bar extreme) / A; high for Boom, low for Crash",
    "aligned_distance_prior_adverse_12_atr": "s * (close - adverse preceding12bar extreme) / A; low for Boom, high for Crash",
}
FEATURE_NAMES = tuple(FEATURE_FORMULAS)
MIN_FIT_ROWS = 100


def _validated_m1(m1: pd.DataFrame) -> pd.DataFrame:
    columns = ["open", "high", "low", "close"]
    if not isinstance(m1, pd.DataFrame) or not set(columns).issubset(m1.columns):
        raise ValueError("Expected M1 OHLC DataFrame")
    if not isinstance(m1.index, pd.DatetimeIndex) or m1.index.tz is None:
        raise ValueError("M1 index must contain timezone-aware UTC minute openings")
    if m1.empty or not m1.index.is_unique or not m1.index.is_monotonic_increasing:
        raise ValueError("M1 index must be nonempty, sorted and unique")
    frame = m1.loc[:, columns].astype(float).copy()
    frame.index = frame.index.tz_convert("UTC").as_unit("ns")
    if not frame.index.floor("min").equals(frame.index):
        raise ValueError("M1 timestamps must be minute openings")
    values = frame.to_numpy()
    if np.isinf(values).any():
        raise ValueError("M1 prices cannot be infinite")
    known = np.isfinite(values).all(axis=1)
    if (values[known] <= 0).any():
        raise ValueError("Known M1 prices must be positive")
    complete = frame.loc[known]
    if ((complete.high < complete[["open", "close", "low"]].max(axis=1)) |
            (complete.low > complete[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid M1 OHLC bounds")
    # A minute missing any OHLC component contributes no usable candle.
    frame.loc[~known, :] = np.nan
    return frame


def causal_inputs(m1: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, list[str]]:
    """Return the original M5 grid and its nineteen ordered continuous inputs.

    Apply ``feature_valid`` before fitting, predicting or issuing signals. An
    incomplete final candle and candles containing missing minutes are retained
    as invalid rows; removing them first would compress rolling time windows.
    """
    m5 = features(_validated_m1(m1), direction)
    side = 1 if direction == "boom" else -1
    atr = m5.atr.where(m5.atr > 0)
    ranges = m5.high - m5.low
    denominator = ranges.where(ranges > 0)
    body = (m5.close - m5.open).abs()
    previous = m5.close.shift(1)
    tr = pd.concat([ranges, (m5.high - previous).abs(),
                    (m5.low - previous).abs()], axis=1).max(axis=1)
    complete = m5[["open", "high", "low", "close"]].notna().all(axis=1)
    tr = tr.where(complete & previous.notna())

    for lag in (1, 3, 6, 12):
        m5[f"aligned_return_{lag}_atr"] = side * (m5.close - m5.close.shift(lag)) / atr
    m5["aligned_body_atr"] = side * (m5.close - m5.open) / atr
    m5["range_atr"] = ranges / atr
    upper = m5.high - m5[["open", "close"]].max(axis=1)
    lower = m5[["open", "close"]].min(axis=1) - m5.low
    favorable, adverse = (upper, lower) if side == 1 else (lower, upper)
    m5["favorable_wick_atr"] = favorable / atr
    m5["adverse_wick_atr"] = adverse / atr
    m5["favorable_close_position"] = (
        m5.close - m5.low if side == 1 else m5.high - m5.close
    ) / denominator
    m5["atr_5_over_30"] = tr.rolling(5).mean() / tr.rolling(30).mean()
    width = 4 * m5.close.rolling(20).std(ddof=0) / m5.close.rolling(20).mean()
    m5["bb_width_over_prior_median_100"] = width / width.shift(1).rolling(100).median()
    m5["range_over_prior_median_20"] = ranges / ranges.shift(1).rolling(20).median()
    m5["body_over_range"] = body / denominator
    m5["mean_range_3_over_prior_median_20"] = (
        ranges.rolling(3).mean() / ranges.shift(3).rolling(20).median()
    )

    lag_atr = atr.shift(1)
    event_known = complete & lag_atr.notna()
    large = (ranges > 3 * lag_atr).where(event_known)
    m5["large_completed_bar"] = large.astype(float)
    age = np.full(len(m5), np.nan)
    last_age = None
    # Missing candles may conceal an event; preserve unknown instead of zero.
    for i, known in enumerate(event_known.to_numpy(bool)):
        if not known:
            last_age = None
        elif bool(large.iloc[i]):
            last_age = 0
        elif last_age is not None:
            last_age = min(last_age + 1, 288)
        if last_age is not None:
            age[i] = last_age
    m5["log1p_large_bar_age"] = np.log1p(age)
    m5["log_atr_over_price"] = np.log(atr / m5.close)
    preceding_high = m5.high.shift(1).rolling(12).max()
    preceding_low = m5.low.shift(1).rolling(12).min()
    favorable_extreme, adverse_extreme = (
        (preceding_high, preceding_low) if side == 1 else (preceding_low, preceding_high)
    )
    m5["aligned_distance_prior_favorable_12_atr"] = side * (m5.close - favorable_extreme) / atr
    m5["aligned_distance_prior_adverse_12_atr"] = side * (m5.close - adverse_extreme) / atr
    finite = np.isfinite(m5.loc[:, FEATURE_NAMES].to_numpy()).all(axis=1)
    m5["feature_valid"] = m5.feature_valid & complete & finite
    return m5, list(FEATURE_NAMES)


def _names(feature_names: list[str] | tuple[str, ...]) -> list[str]:
    result = list(feature_names)
    if not result or not all(isinstance(name, str) and name for name in result):
        raise ValueError("Feature names must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("Feature names must be unique and ordered")
    return result


def _matrix(X: np.ndarray | pd.DataFrame, names: list[str]) -> np.ndarray:
    if isinstance(X, pd.DataFrame) and list(X.columns) != names:
        raise ValueError("DataFrame columns must exactly match the ordered feature schema")
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("X must be a two-dimensional matrix matching the feature schema")
    if not np.isfinite(values).all():
        raise ValueError("Every input feature must be finite")
    return values


def fit_ridge(X: np.ndarray | pd.DataFrame, y: np.ndarray,
              feature_names: list[str] | tuple[str, ...], penalty: float = 0.1,
              quantile: float = 0.75) -> dict:
    """Fit train-only standardization and mean-SSE ridge with free intercept.

    Minimize ``mean((y-intercept-Z@coef)**2) + penalty*sum(coef**2)``.
    X and y are never clipped. Constant columns receive scale one. The issuance
    cutoff is ``max(0, quantile(training_predictions, quantile))``. Callers must
    supply only completed training labels, with their partition purge applied.
    """
    names = _names(feature_names)
    values = _matrix(X, names)
    target = np.asarray(y, dtype=float)
    if len(values) < MIN_FIT_ROWS:
        raise ValueError(f"At least {MIN_FIT_ROWS} completed training rows are required")
    if target.ndim != 1 or len(target) != len(values) or not np.isfinite(target).all():
        raise ValueError("y must be a finite one-dimensional target matching X")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Ridge penalty must be finite and nonnegative")
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("Training quantile must lie between zero and one")
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    Z = (values - means) / scales
    intercept = float(target.mean())
    centered = target - intercept
    if penalty == 0:
        coefficients = np.linalg.lstsq(Z, centered, rcond=None)[0]
    else:
        coefficients = np.linalg.solve(
            Z.T @ Z + len(values) * penalty * np.eye(len(names)), Z.T @ centered,
        )
    scores = intercept + Z @ coefficients
    threshold = max(0.0, float(np.quantile(scores, quantile)))
    if not np.isfinite(np.r_[means, scales, coefficients, intercept, threshold]).all():
        raise ValueError("Training arithmetic produced a non-finite model")
    return {
        "version": 1, "feature_names": names,
        "means": means.tolist(), "std": scales.tolist(),
        "coefs": coefficients.tolist(), "intercept": intercept,
        "threshold": threshold, "penalty": float(penalty),
        "quantile": float(quantile), "fit_rows": int(len(values)),
        "score_kind": "continuous_uncalibrated_score",
        "objective": "mean_squared_error_plus_penalty_times_squared_coefficients",
        "scaler_fitted_on": "training_rows_only", "target_clipped": False,
        "features_clipped": False,
    }


def predict_ridge(X: np.ndarray | pd.DataFrame, model: dict,
                  feature_names: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
    """Apply frozen scaling and coefficients; ndarray order follows the model.

    DataFrame columns are checked automatically. For ndarray inputs the caller
    may additionally supply ``feature_names`` to verify their ordered schema.
    This function never refits the scaler or the issuance threshold.
    """
    if model.get("version") != 1:
        raise ValueError("Unsupported ridge model version")
    names = _names(model["feature_names"])
    if feature_names is not None and _names(feature_names) != names:
        raise ValueError("Prediction feature order differs from the frozen model")
    values = _matrix(X, names)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["std"], dtype=float)
    coefficients = np.asarray(model["coefs"], dtype=float)
    intercept = float(model["intercept"])
    if any(vector.shape != (len(names),) for vector in (means, scales, coefficients)):
        raise ValueError("Frozen scaler and coefficient dimensions must match the feature schema")
    if not np.isfinite(np.r_[means, scales, coefficients, intercept]).all() or (scales <= 0).any():
        raise ValueError("Frozen model contains invalid scaling or coefficients")
    scores = intercept + ((values - means) / scales) @ coefficients
    if not np.isfinite(scores).all():
        raise ValueError("Prediction arithmetic produced non-finite scores")
    return scores
