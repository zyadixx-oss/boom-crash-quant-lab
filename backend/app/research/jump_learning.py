"""Fixed SPIKE-only paired ridge learning and issuance for offline research.

Detector calibration sees only the explicitly declared training quote prefix.
Paired models share the same common-available closed-M5 UTC00/30 opportunities,
completed original-quote labels and a 31-minute planned partition purge. Each model
has its own training-only scaler, with unchanged ridge penalty0.1 and positive
floor q75 threshold. Unknown labels remain unknown and are counted, never zero.

Issuance has no label argument, detector fitting, outcome calculation, execution
or fallback model. Execution ATR always belongs to the raw observed series.
The caller remains responsible for source lineage, numerically valid feature
preparation, raw-tick label reproduction and all economic evidence gates.
Synthetic training rows or predictions do not establish independent trades,
profit, calibrated probabilities or permission to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd

from app.research.jump_multiframe import PairedMultiframeInputs
from app.research.learned_signal import fit_ridge, predict_ridge
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.spike_hunter import assert_offline
from app.research.tick_tail import FixedTailDetector, fit_fixed_tail_detector, _side
from app.research.tick_tail_signal import _boolean, _index, _values


MIN_TRAINING_LABELS = 1000
RIDGE_PENALTY = 0.1
TRAINING_QUANTILE = 0.75
PURGE_MINUTES = 31
PLANNED_HOLD_AND_DELAY_MINUTES = 16
SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
SIDES = {"BOOM600": 1, "CRASH600": -1}
FAMILIES = ("RAW44", "TRANSFORMED44")
LABEL_COLUMNS = ("net_R", "completed", "planned_end")
SIGNAL_COLUMNS = ("signal_time", "atr", "side", "variant", "signal_close", "score")


class InsufficientTrainingEvidenceError(ValueError):
    """No model is fitted when fewer than 1000 completed paired labels exist."""


@dataclass(frozen=True, slots=True)
class TrainingDetectorCalibration:
    detector: FixedTailDetector
    metadata: dict


@dataclass(frozen=True, slots=True)
class PairedRidgeFit:
    models: dict[str, dict]
    metadata: dict


@dataclass(frozen=True, slots=True)
class ClockIssuance:
    signals: pd.DataFrame
    metadata: dict


@dataclass(frozen=True, slots=True)
class PairedSpikeIssuance:
    signals: dict[str, pd.DataFrame]
    metadata: dict


def _boundary(value: object, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a whole-second UTC timestamp") from error
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    stamp = stamp.tz_convert("UTC").as_unit("ns")
    if stamp.value % SECOND_NS:
        raise ValueError(f"{name} must be a whole-second UTC timestamp")
    return stamp


def _bounds(start: object, end: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _boundary(start, "start"), _boundary(end, "end")
    if start >= end:
        raise ValueError("start must be strictly before end")
    return start, end


def _native_side(symbol: object) -> int:
    if not isinstance(symbol, str) or symbol not in SIDES:
        raise ValueError("Only the declared BOOM600/CRASH600 SPIKE sides are permitted")
    return SIDES[symbol]


def _json_hash(value: dict) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Model metadata must be finite JSON values") from error
    return hashlib.sha256(encoded).hexdigest()


def _target_hash(index: pd.DatetimeIndex, target: np.ndarray) -> str:
    digest = hashlib.sha256(b"jump-paired-target-v1\0")
    digest.update(np.asarray(index.asi8, dtype=">i8").tobytes())
    digest.update(np.asarray(target, dtype=">f8").tobytes())
    return digest.hexdigest()


def calibrate_training_detector(ticks: pd.DataFrame, native_side: int,
                               train_start: object, train_end: object) -> TrainingDetectorCalibration:
    """Fit the unchanged detector on quote values wholly inside [start,end).

    Full-source index/schema validation examines timestamps, not quote values.
    Malformed future quote values therefore cannot enter calibration or its
    validity checks. A pair straddling either boundary is excluded by selecting
    the prefix before calling the immutable detector fitter.
    """
    safety = assert_offline()
    side = _side(native_side)
    start, end = _bounds(train_start, train_end)
    index = _index(ticks, "ticks", SECOND_NS)
    if "quote" not in ticks:
        raise ValueError("A quote column is required")
    chosen = (index >= start) & (index < end)
    prefix = ticks.loc[chosen, ["quote"]].copy(deep=True)
    if prefix.empty:
        raise ValueError("Training detector prefix is empty")
    # Never validate or convert quote values outside this selected prefix.
    detector = fit_fixed_tail_detector(prefix, side)
    observed = prefix.index.tz_convert("UTC").as_unit("ns")
    pairs = int(np.count_nonzero(np.diff(observed.asi8) == SECOND_NS))
    metadata = {
        "training_start": start.isoformat(), "training_end": end.isoformat(),
        "training_rows": len(prefix), "known_consecutive_training_increments": pairs,
        "training_first_observed": observed[0].isoformat(),
        "training_last_observed": observed[-1].isoformat(), "native_side": side,
        "median_abs_log_return": detector.median_abs_log_return,
        "threshold": detector.threshold, "threshold_multiplier": 10,
        "fit_scope": "exact_declared_training_quote_values_only",
        "future_quote_values_examined": False, "cross_boundary_pairs_used": False,
        "safety": safety,
    }
    return TrainingDetectorCalibration(detector, metadata)


def _paired_index(inputs: PairedMultiframeInputs) -> pd.DatetimeIndex:
    if not isinstance(inputs, PairedMultiframeInputs):
        raise TypeError("inputs must be PairedMultiframeInputs")
    if tuple(inputs.feature_names) != FEATURE_NAMES:
        raise ValueError("Exactly the original ordered44 features are required")
    opening = _index(inputs.raw, "raw_inputs", 5 * MINUTE_NS)
    if not opening.equals(_index(inputs.transformed, "transformed_inputs", 5 * MINUTE_NS)):
        raise ValueError("Raw and transformed feature populations differ")
    if not opening.equals(_index(inputs.availability, "availability", 5 * MINUTE_NS)):
        raise ValueError("Availability and feature populations differ")
    if not opening.equals(pd.date_range(opening[0], opening[-1], freq="5min").as_unit("ns")):
        raise ValueError("The full UTC M5 grid must remain present")
    for frame in (inputs.raw, inputs.transformed):
        if any(name not in frame for name in (*FEATURE_NAMES, "feature_valid", "atr", "close")):
            raise ValueError("Both frames require unchanged44 features, validity, atr and close")
    required = ("raw_feature_valid", "transformed_feature_valid", "common_feature_valid",
                "raw_execution_atr", "transformed_feature_atr", "run_id")
    if any(name not in inputs.availability for name in required):
        raise ValueError("Exact paired availability and separate ATR columns are required")
    return opening


def _clock_rows(inputs: PairedMultiframeInputs, start: pd.Timestamp, end: pd.Timestamp):
    opening = _paired_index(inputs)
    issues = opening + pd.Timedelta(minutes=5)
    inside = (issues >= start) & (issues < end)
    # Read feature/availability values only in the requested temporal interval.
    raw = inputs.raw.loc[inside].copy(deep=True)
    transformed = inputs.transformed.loc[inside].copy(deep=True)
    available = inputs.availability.loc[inside].copy(deep=True)
    issues = issues[inside]
    raw_flag = _boolean(raw.feature_valid, "raw_feature_valid")
    transformed_flag = _boolean(transformed.feature_valid, "transformed_feature_valid")
    declared_raw = _boolean(available.raw_feature_valid, "declared_raw_feature_valid")
    declared_transformed = _boolean(available.transformed_feature_valid, "declared_transformed_feature_valid")
    common = _boolean(available.common_feature_valid, "common_feature_valid")
    if (not np.array_equal(raw_flag, declared_raw)
            or not np.array_equal(transformed_flag, declared_transformed)
            or not np.array_equal(common, raw_flag & transformed_flag)):
        raise ValueError("Common availability must equal both intrinsic44 validity flags")
    clock = issues.asi8 % (30 * MINUTE_NS) == 0
    planned_end = issues + pd.Timedelta(minutes=PLANNED_HOLD_AND_DELAY_MINUTES)
    purged = ((issues + pd.Timedelta(minutes=PURGE_MINUTES) <= end)
              & (planned_end + pd.Timedelta(seconds=1) <= end))
    selected = clock & purged & common
    # Invalid/nonclock/purged rows remain in availability counts; their numeric
    # indicators are not fit or issued. Warmup can legitimately contain inf/NaN
    # ratios, so validate values only after the past-only shared mask is fixed.
    chosen_raw, chosen_transformed, chosen_available = raw.loc[selected], transformed.loc[selected], available.loc[selected]
    for frame in (chosen_raw, chosen_transformed):
        values = np.column_stack([_values(frame[name], name) for name in FEATURE_NAMES])
        if not np.isfinite(values).all():
            raise ValueError("Every selected common44 feature row must be finite")
    raw_atr = _values(chosen_available.raw_execution_atr, "raw_execution_atr")
    feature_atr = _values(chosen_available.transformed_feature_atr, "transformed_feature_atr")
    if (not np.array_equal(raw_atr, _values(chosen_raw.atr, "raw.atr"), equal_nan=True)
            or not np.array_equal(feature_atr, _values(chosen_transformed.atr, "transformed.atr"), equal_nan=True)):
        raise ValueError("Execution and transformed ATR lineage must match their separate frames")
    if not np.isfinite(raw_atr).all() or np.any(raw_atr <= 0):
        raise ValueError("Common opportunities require finite positive raw execution ATR")
    if not np.isfinite(feature_atr).all() or np.any(feature_atr <= 0):
        raise ValueError("Common opportunities require finite positive transformed feature ATR")
    close = _values(chosen_raw.close, "raw.close")
    if not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("Common opportunities require finite positive original close")
    run = chosen_available.run_id.array
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0 for value in run):
        raise ValueError("Common opportunities require a known nonnegative integer representation run")
    metadata = {
        "start": start.isoformat(), "end": end.isoformat(), "m5_rows_in_bounds": len(raw),
        "clock_rows_in_bounds": int(clock.sum()), "clock_planned_purged": int((clock & ~purged).sum()),
        "raw_valid_clock_rows": int((clock & purged & raw_flag).sum()),
        "transformed_valid_clock_rows": int((clock & purged & transformed_flag).sum()),
        "common_clock_opportunities": int(selected.sum()), "purge_minutes": PURGE_MINUTES,
        "closed_m5_issue_offset_minutes": 5, "clock": "UTC00_or30_only",
        "past_only_common_availability": True, "execution_atr_source": "original_raw_quote_prices",
    }
    return chosen_raw, chosen_transformed, chosen_available, issues[selected], metadata


def _signals(raw: pd.DataFrame, availability: pd.DataFrame, issues: pd.DatetimeIndex,
             symbol: str, family: str, scores: np.ndarray | None = None) -> pd.DataFrame:
    side = _native_side(symbol)
    index = issues.copy().rename("signal_time")
    frame = pd.DataFrame({
        "signal_time": index, "atr": availability.raw_execution_atr.to_numpy(dtype=float),
        "side": np.full(len(raw), side, dtype=np.int64), "variant": family + "_SPIKE",
        "signal_close": raw.close.to_numpy(dtype=float),
        "score": np.full(len(raw), np.nan) if scores is None else scores,
    }, index=index)
    return frame.loc[:, SIGNAL_COLUMNS]


def spike_clock_signals(inputs: PairedMultiframeInputs, symbol: str,
                        start: object, end: object) -> ClockIssuance:
    """Generate the fixed common raw clock before raw-tick labels are available."""
    safety = assert_offline()
    side = _native_side(symbol)
    start, end = _bounds(start, end)
    raw, _, available, issues, metadata = _clock_rows(inputs, start, end)
    return ClockIssuance(_signals(raw, available, issues, symbol, "CLOCK"),
                         {**metadata, "symbol": symbol, "native_side": side, "safety": safety})


def fit_paired_spike_ridge(inputs: PairedMultiframeInputs, labels: pd.DataFrame,
                           symbol: str, train_start: object, train_end: object) -> PairedRidgeFit:
    """Fit two fixed estimators on one exact paired original-quote target vector.

    Labels have a UTC issue-time index and exactly net_R/completed/planned_end.
    Their index must equal the complete eligible purged training clock,
    including explicit unknown rows. Missing/extra/noncommon/future labels are
    refused. Only completed finite labels enter either fit; their common count
    must reach 1000. This function does not calculate or certify those labels.
    """
    safety = assert_offline()
    side = _native_side(symbol)
    start, end = _bounds(train_start, train_end)
    raw, transformed, available, issues, clock_metadata = _clock_rows(inputs, start, end)
    if isinstance(labels, pd.DataFrame) and labels.empty and len(issues) == 0:
        if not labels.columns.is_unique or set(labels.columns) != set(LABEL_COLUMNS):
            raise ValueError("Labels require exactly net_R, completed and planned_end")
        index = labels.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError("Empty labels require an explicit UTC issue-time index")
        if str(index.tz) not in ("UTC", "Etc/UTC"):
            raise ValueError("Empty labels require an explicit UTC issue-time index")
        raise InsufficientTrainingEvidenceError("At least 1000 completed paired training labels are required; observed 0")
    label_index = _index(labels, "labels", MINUTE_NS)
    if set(labels.columns) != set(LABEL_COLUMNS):
        raise ValueError("Labels require exactly net_R, completed and planned_end")
    if not label_index.equals(issues):
        raise ValueError("Label timestamps must equal the exact common purged training clock")
    completed = _boolean(labels.completed, "completed")
    target = _values(labels.net_R, "net_R")
    if np.any(completed & ~np.isfinite(target)) or np.any(~completed & np.isfinite(target)):
        raise ValueError("Completed labels must be finite; unknown labels must stay unknown")
    planned = pd.DatetimeIndex([_boundary(value, "planned_end") for value in labels.planned_end])
    expected_end = issues + pd.Timedelta(minutes=PLANNED_HOLD_AND_DELAY_MINUTES)
    if not planned.equals(expected_end):
        raise ValueError("Every planned_end must equal issue + 16 minutes")
    if (np.any(issues + pd.Timedelta(minutes=PURGE_MINUTES) > end)
            or np.any(planned + pd.Timedelta(seconds=1) > end)):
        raise ValueError("Training labels breach the declared planned partition purge")
    count = int(completed.sum())
    if count < MIN_TRAINING_LABELS:
        raise InsufficientTrainingEvidenceError(f"At least 1000 completed paired training labels are required; observed {count}")
    y = target[completed].copy()
    chosen_issues = issues[completed]
    target_hash = _target_hash(chosen_issues, y)
    models, matrix_hashes = {}, {}
    for family, frame in (("RAW44", raw), ("TRANSFORMED44", transformed)):
        matrix = frame.loc[completed, list(FEATURE_NAMES)].copy(deep=True)
        estimator = fit_ridge(matrix, y.copy(), list(FEATURE_NAMES),
                              penalty=RIDGE_PENALTY, quantile=TRAINING_QUANTILE)
        models[family] = estimator
        digest = hashlib.sha256(target_hash.encode())
        digest.update(np.asarray(matrix.to_numpy(dtype=float), dtype=">f8").tobytes())
        matrix_hashes[family] = digest.hexdigest()
    metadata = {
        "version": 1, "symbol": symbol, "native_side": side, "mode": "SPIKE",
        "training_start": start.isoformat(), "training_end": end.isoformat(),
        "training_completed_labels": count, "training_unknown_labels": int((~completed).sum()),
        "training_common_clock_opportunities": len(labels),
        "training_first_issue": chosen_issues[0].isoformat(),
        "training_latest_issue": chosen_issues[-1].isoformat(),
        "common_timestamp_target_sha256": target_hash, "matrix_sha256": matrix_hashes,
        "model_sha256": {family: _json_hash(model) for family, model in models.items()},
        "feature_names": list(FEATURE_NAMES), "penalty": RIDGE_PENALTY,
        "quantile": TRAINING_QUANTILE, "minimum_training_labels": MIN_TRAINING_LABELS,
        "training_rows_are_independent_trades": False, "counts_toward_profit_sample_target": False,
        "label_source_contract": "completed_original_raw_tick_net_R",
        "clock": clock_metadata, "safety": safety,
    }
    metadata["metadata_sha256"] = _json_hash(metadata)
    return PairedRidgeFit(models, metadata)


def _validate_fit(fit: PairedRidgeFit, symbol: str) -> pd.Timestamp:
    if not isinstance(fit, PairedRidgeFit) or set(fit.models) != set(FAMILIES):
        raise ValueError("A complete fixed paired ridge fit is required; no fallback")
    metadata = fit.metadata
    if (not isinstance(metadata, dict) or metadata.get("metadata_sha256")
            != _json_hash({key: value for key, value in metadata.items() if key != "metadata_sha256"})):
        raise ValueError("Frozen paired fit metadata hash changed")
    if (metadata.get("version") != 1 or metadata.get("symbol") != symbol
            or metadata.get("native_side") != _native_side(symbol) or metadata.get("mode") != "SPIKE"
            or metadata.get("feature_names") != list(FEATURE_NAMES)
            or metadata.get("penalty") != RIDGE_PENALTY or metadata.get("quantile") != TRAINING_QUANTILE
            or metadata.get("minimum_training_labels") != MIN_TRAINING_LABELS):
        raise ValueError("Paired fit identity or fixed learner settings differ")
    start, end = _bounds(metadata.get("training_start"), metadata.get("training_end"))
    count = metadata.get("training_completed_labels")
    if isinstance(count, bool) or not isinstance(count, int) or count < MIN_TRAINING_LABELS:
        raise ValueError("Paired fit must retain at least 1000 completed training labels")
    first = _boundary(metadata.get("training_first_issue"), "training_first_issue")
    latest = _boundary(metadata.get("training_latest_issue"), "training_latest_issue")
    if not start <= first <= latest or latest + pd.Timedelta(minutes=PURGE_MINUTES) > end:
        raise ValueError("Frozen training issue bounds breach the declared partition purge")
    if metadata.get("safety") != dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False):
        raise ValueError("Frozen fit must retain all false safety flags")
    for family, model in fit.models.items():
        if (not isinstance(model, dict) or model.get("version") != 1
                or model.get("feature_names") != list(FEATURE_NAMES)
                or model.get("penalty") != RIDGE_PENALTY or model.get("quantile") != TRAINING_QUANTILE
                or model.get("fit_rows") != count or model.get("scaler_fitted_on") != "training_rows_only"
                or model.get("objective") != "mean_squared_error_plus_penalty_times_squared_coefficients"
                or model.get("score_kind") != "continuous_uncalibrated_score"
                or model.get("features_clipped") is not False or model.get("target_clipped") is not False
                or not isinstance(model.get("threshold"), (float, int))
                or isinstance(model.get("threshold"), bool) or not np.isfinite(model["threshold"])
                or model["threshold"] < 0):
            raise ValueError("Frozen estimators must retain the fixed training-only ridge semantics")
        if metadata.get("model_sha256", {}).get(family) != _json_hash(model):
            raise ValueError("Frozen paired estimator hash changed")
        # Validate dimensional/scaler arithmetic even if evaluation is empty.
        predict_ridge(np.empty((0, len(FEATURE_NAMES))), model, FEATURE_NAMES)
    return end


def issue_paired_spike(inputs: PairedMultiframeInputs, fit: PairedRidgeFit, symbol: str,
                       eval_start: object, eval_end: object) -> PairedSpikeIssuance:
    """Issue fixed SPIKE signals from past-only features, with no label argument."""
    safety = assert_offline()
    side = _native_side(symbol)
    training_end = _validate_fit(fit, symbol)
    start, end = _bounds(eval_start, eval_end)
    if start < training_end:
        raise ValueError("Evaluation issuance cannot begin before the frozen training end")
    raw, transformed, available, issues, metadata = _clock_rows(inputs, start, end)
    result = {"CLOCK": _signals(raw, available, issues, symbol, "CLOCK")}
    for family, frame in (("RAW44", raw), ("TRANSFORMED44", transformed)):
        scores = predict_ridge(frame.loc[:, list(FEATURE_NAMES)], fit.models[family], FEATURE_NAMES)
        chosen = (scores >= fit.models[family]["threshold"]) & (scores > 0.0)
        result[family] = _signals(raw.loc[chosen], available.loc[chosen], issues[chosen],
                                  symbol, family, scores[chosen])
    return PairedSpikeIssuance(result, {**metadata, "symbol": symbol, "native_side": side,
                                       "model_sha256": dict(fit.metadata["model_sha256"]),
                                       "score_kind": "continuous_uncalibrated_score", "safety": safety})
