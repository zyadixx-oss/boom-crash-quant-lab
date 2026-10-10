"""Fixed paired nonlinear conditional-region learner; no historical I/O.

This adapter changes only the estimator used on the already defined completed
CLOCK-region label population. No new features, parameters or fill selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from app.research.jump_learning import _bounds, _clock_rows, _native_side, _signals
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.nonlinear_signal import (DEFAULT_PARAMETERS, fit_histogram_boost,
                                           predict_histogram_boost)
from app.research.spike_hunter import assert_offline
from scripts.region_reward_learning import (InsufficientRegionTraining, LABELS, NONFILL,
                                           UNKNOWN, fingerprint, region_training_labels)

ARMS = ("RAW_BOOST_REGION", "HYBRID_BOOST_REGION")
ALGORITHM = "app.research.nonlinear_signal.fit_histogram_boost"
FIXED_PARAMETERS = {"n_trees": 100, "learning_rate": .05, "max_depth": 3,
                    "min_leaf": 200, "n_bins": 16, "leaf_regularization": 20.,
                    "quantile": .75}


@dataclass(frozen=True, slots=True)
class RegionBoostFit:
    models: dict
    metadata: dict


def _parameters():
    if DEFAULT_PARAMETERS != FIXED_PARAMETERS:
        raise ValueError("The inherited fixed histogram learner defaults changed")
    return FIXED_PARAMETERS.copy()


def fit_region_boost(inputs, labels, symbol, start, end):
    """Fit both fixed learners on identical purged completed CLOCK-region rows.

    Validation intentionally follows the immutable region-reward population:
    no-fill, unknown and occupied rows retain NaN reward and distinct status.
    Trees consume the original cached features without a new scaler.
    """
    safety = assert_offline()
    side = _native_side(symbol)
    start, end = _bounds(start, end)
    parameters = _parameters()
    raw, hybrid, _, issues, clock = _clock_rows(inputs, start, end)
    if set(labels.columns) != set(LABELS) or not labels.columns.is_unique:
        raise ValueError("Exact filled-region label schema required")
    if not isinstance(labels.index, pd.DatetimeIndex) or labels.index.tz is None:
        raise ValueError("Explicit UTC issue-time labels required")
    if not labels.index.as_unit("ns").equals(issues):
        raise ValueError("Labels must equal the full common training clock")
    flags = []
    for name in ("completed", "known_nonfill", "unknown_path", "exposure_skipped"):
        if not labels[name].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("Known boolean training disposition flags required")
        flags.append(labels[name].to_numpy(bool))
    if not np.all(np.column_stack(flags).sum(axis=1) == 1):
        raise ValueError("Every training opportunity has exactly one disposition category")
    for values, expected in zip(flags, (("completed",), NONFILL, UNKNOWN, ("overlap_skipped",)), strict=True):
        if not np.array_equal(values, labels.status.isin(expected)):
            raise ValueError("Training status and category differ")
    if pd.api.types.is_bool_dtype(labels.net_R.dtype):
        raise ValueError("Numeric reward cannot be boolean")
    target = labels.net_R.to_numpy(float)
    completed = flags[0]
    if not np.isfinite(target[completed]).all() or not np.isnan(target[~completed]).all():
        raise ValueError("Only completed filled paths may carry finite reward labels")
    planned = pd.DatetimeIndex(pd.to_datetime(labels.planned_end, utc=True)).as_unit("ns")
    if not planned.equals(issues + pd.Timedelta(minutes=30, seconds=1)) or np.any(planned >= end):
        raise ValueError("Exact planned region horizon and training purge required")
    count = int(completed.sum())
    if count < 1000:
        raise InsufficientRegionTraining(f"1000 completed shared region training paths required; observed {count}")
    y = target[completed].copy()
    models, matrices = {}, {}
    h = hashlib.sha256(b"paired-region-filled-target-v1\0")
    h.update(np.asarray(issues[completed].asi8, dtype=">i8").tobytes())
    h.update(np.asarray(y, dtype=">f8").tobytes())
    target_hash = h.hexdigest()
    for arm, frame in zip(ARMS, (raw, hybrid), strict=True):
        x = frame.loc[completed, list(FEATURE_NAMES)].copy(deep=True)
        models[arm] = fit_histogram_boost(x, y, FEATURE_NAMES, **parameters)
        matrix = hashlib.sha256(target_hash.encode())
        matrix.update(np.asarray(x.to_numpy(float), dtype=">f8").tobytes())
        matrices[arm] = matrix.hexdigest()
    metadata = {"version": 1, "symbol": symbol, "native_side": side,
        "algorithm": ALGORITHM, "parameters": parameters, "standardization": False,
        "target": "conditional_completed_original_quote_CLOCK_region_net_R",
        "training_start": start.isoformat(), "training_end": end.isoformat(),
        "clock": clock, "completed": count, "known_nonfill": int(flags[1].sum()),
        "unknown_path": int(flags[2].sum()), "exposure_skipped": int(flags[3].sum()),
        "opportunities": len(labels), "target_sha256": target_hash, "matrix_sha256": matrices,
        "feature_names": list(FEATURE_NAMES), "quantile": .75,
        "model_sha256": {key: fingerprint(value) for key, value in models.items()},
        "training_is_independent_trade_sample": False,
        "label_selection_is_conditional_on_CLOCK_fill": True, "safety": safety}
    metadata["metadata_sha256"] = fingerprint(metadata)
    return RegionBoostFit(models, metadata)


def issue_region_boost(inputs, fit, symbol, start, end):
    """Apply already frozen bins/trees on past-only common rows; accepts no labels."""
    safety = assert_offline()
    side = _native_side(symbol)
    start, end = _bounds(start, end)
    parameters = _parameters()
    if not isinstance(fit, RegionBoostFit) or set(fit.models) != set(ARMS):
        raise ValueError("Complete paired filled-region boost fit required; no fallback")
    metadata = fit.metadata
    if (metadata.get("metadata_sha256") != fingerprint({k: v for k, v in metadata.items() if k != "metadata_sha256"})
            or metadata.get("version") != 1 or metadata.get("symbol") != symbol
            or metadata.get("native_side") != side or metadata.get("algorithm") != ALGORITHM
            or metadata.get("parameters") != parameters or metadata.get("standardization") is not False
            or metadata.get("feature_names") != list(FEATURE_NAMES) or metadata.get("completed", 0) < 1000
            or metadata.get("quantile") != .75 or metadata.get("safety") != safety):
        raise ValueError("Frozen region boost identity/metadata changed")
    if start < pd.Timestamp(metadata["training_end"]):
        raise ValueError("Issuance before frozen training end")
    raw, hybrid, available, issues, _ = _clock_rows(inputs, start, end)
    output = {}
    for arm, frame in zip(ARMS, (raw, hybrid), strict=True):
        model = fit.models[arm]
        if (fingerprint(model) != metadata["model_sha256"][arm]
                or model.get("parameters") != parameters or model.get("defaults") != parameters
                or model.get("fit_rows") != metadata["completed"]):
            raise ValueError("Frozen region boost estimator changed")
        scores = predict_histogram_boost(frame.loc[:, list(FEATURE_NAMES)], model, FEATURE_NAMES)
        chosen = (scores >= model["threshold"]) & (scores > 0)
        output[arm] = _signals(raw.loc[chosen], available.loc[chosen], issues[chosen], symbol, arm, scores[chosen])
    return output
