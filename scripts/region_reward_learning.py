"""Paired conditional filled-region reward learning; research only.

Every training opportunity retains its real region disposition. Only completed
filled paths are regression labels. Known no-fills, occupancy skips and unknown
paths remain distinct and never become fabricated zero-return trades. Both arms
share the same completed training target and exact past-only feature clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from app.research.jump_learning import _bounds, _clock_rows, _native_side, _signals
from app.research.learned_signal import fit_ridge, predict_ridge
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.spike_hunter import assert_offline

ARMS = ("RAW_REGION", "HYBRID_REGION")
UNKNOWN = ("waiting_gap", "entry_gap", "path_gap")
NONFILL = ("invalidated", "expired")
STATUSES = (*UNKNOWN, *NONFILL, "completed", "overlap_skipped")
LABELS = ("net_R", "completed", "known_nonfill", "unknown_path", "exposure_skipped", "status", "planned_end")


class InsufficientRegionTraining(ValueError): pass


@dataclass(frozen=True, slots=True)
class RegionRewardFit:
    models: dict
    metadata: dict


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def region_training_labels(clock, events):
    """Exact shared clock, preserving actual one-pending/open CLOCK behavior."""
    assert_offline()
    issues = pd.DatetimeIndex(pd.to_datetime(clock.signal_time, utc=True)).as_unit("ns")
    event_issues = pd.DatetimeIndex(pd.to_datetime(events.signal_time, utc=True)).as_unit("ns")
    if issues.has_duplicates or not issues.equals(event_issues):
        raise ValueError("Region events must equal the exact ordered shared clock")
    required = {"status", "censored", "entry_time", "net_R"}
    if not required.issubset(events): raise ValueError("Complete saved dispositions required")
    if not events.status.isin(STATUSES).all(): raise ValueError("Unexpected unpurged training disposition")
    if not events.censored.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Actual boolean censor flags required")
    status = events.status.to_numpy(str)
    complete = np.isin(status, ["completed"])
    unknown = np.isin(status, UNKNOWN)
    nofill = np.isin(status, NONFILL)
    skipped = np.isin(status, ["overlap_skipped"])
    if pd.api.types.is_bool_dtype(events.net_R.dtype): raise ValueError("Numeric reward cannot be boolean")
    value = pd.to_numeric(events.net_R, errors="raise").to_numpy(float)
    if (not np.array_equal(events.censored.to_numpy(bool), unknown)
            or np.any(complete & ~np.isfinite(value)) or np.any(~complete & np.isfinite(value))
            or np.any(complete & pd.isna(events.entry_time).to_numpy())):
        raise ValueError("Completed/unknown/nonfilled region label semantics differ")
    return pd.DataFrame({"net_R": value, "completed": complete, "known_nonfill": nofill,
        "unknown_path": unknown, "exposure_skipped": skipped, "status": status,
        "planned_end": issues + pd.Timedelta(minutes=30, seconds=1)}, index=issues).loc[:, LABELS]


def fit_region_reward(inputs, labels, symbol, train_start, train_end):
    safety = assert_offline(); side = _native_side(symbol)
    start, end = _bounds(train_start, train_end)
    raw, hybrid, available, issues, clock = _clock_rows(inputs, start, end)
    if set(labels.columns) != set(LABELS) or not labels.columns.is_unique:
        raise ValueError("Exact filled-region label schema required")
    if not isinstance(labels.index, pd.DatetimeIndex) or labels.index.tz is None:
        raise ValueError("Explicit UTC issue-time labels required")
    if not labels.index.as_unit("ns").equals(issues): raise ValueError("Labels must equal the full common training clock")
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
    if pd.api.types.is_bool_dtype(labels.net_R.dtype): raise ValueError("Numeric reward cannot be boolean")
    target = labels.net_R.to_numpy(float); completed = flags[0]
    if not np.isfinite(target[completed]).all() or not np.isnan(target[~completed]).all():
        raise ValueError("Only completed filled paths may carry finite reward labels")
    planned = pd.DatetimeIndex(pd.to_datetime(labels.planned_end, utc=True)).as_unit("ns")
    if not planned.equals(issues + pd.Timedelta(minutes=30, seconds=1)) or np.any(planned >= end):
        raise ValueError("Exact planned region horizon and training purge required")
    count = int(completed.sum())
    if count < 1000: raise InsufficientRegionTraining(f"1000 completed shared region training paths required; observed {count}")
    y = target[completed].copy(); models, matrices = {}, {}
    h = hashlib.sha256(b"paired-region-filled-target-v1\0")
    h.update(np.asarray(issues[completed].asi8, dtype=">i8").tobytes())
    h.update(np.asarray(y, dtype=">f8").tobytes()); target_hash = h.hexdigest()
    for arm, frame in (("RAW_REGION", raw), ("HYBRID_REGION", hybrid)):
        x = frame.loc[completed, list(FEATURE_NAMES)].copy(deep=True)
        models[arm] = fit_ridge(x, y, FEATURE_NAMES, penalty=.1, quantile=.75)
        m = hashlib.sha256(target_hash.encode()); m.update(np.asarray(x.to_numpy(float), dtype=">f8").tobytes())
        matrices[arm] = m.hexdigest()
    metadata = {"version": 1, "symbol": symbol, "native_side": side,
        "target": "conditional_completed_original_quote_CLOCK_region_net_R",
        "training_start": start.isoformat(), "training_end": end.isoformat(),
        "clock": clock, "completed": count, "known_nonfill": int(flags[1].sum()),
        "unknown_path": int(flags[2].sum()), "exposure_skipped": int(flags[3].sum()),
        "opportunities": len(labels), "target_sha256": target_hash, "matrix_sha256": matrices,
        "feature_names": list(FEATURE_NAMES), "penalty": .1, "quantile": .75,
        "model_sha256": {k: fingerprint(v) for k, v in models.items()},
        "training_is_independent_trade_sample": False,
        "label_selection_is_conditional_on_CLOCK_fill": True, "safety": safety}
    metadata["metadata_sha256"] = fingerprint(metadata)
    return RegionRewardFit(models, metadata)


def issue_region_reward(inputs, fit, symbol, eval_start, eval_end):
    assert_offline(); side = _native_side(symbol)
    start, end = _bounds(eval_start, eval_end)
    if not isinstance(fit, RegionRewardFit) or set(fit.models) != set(ARMS):
        raise ValueError("Complete paired filled-region fit required; no fallback")
    m = fit.metadata
    if (m.get("metadata_sha256") != fingerprint({k: v for k, v in m.items() if k != "metadata_sha256"})
            or m.get("symbol") != symbol or m.get("native_side") != side
            or m.get("feature_names") != list(FEATURE_NAMES) or m.get("completed", 0) < 1000
            or m.get("penalty") != .1 or m.get("quantile") != .75
            or m.get("safety") != assert_offline()):
        raise ValueError("Frozen region learner identity/metadata changed")
    if start < pd.Timestamp(m["training_end"]): raise ValueError("Issuance before frozen training end")
    raw, hybrid, available, issues, _ = _clock_rows(inputs, start, end)
    result = {}
    for arm, frame in (("RAW_REGION", raw), ("HYBRID_REGION", hybrid)):
        model = fit.models[arm]
        if (fingerprint(model) != m["model_sha256"][arm] or model["penalty"] != .1
                or model["quantile"] != .75 or model["fit_rows"] != m["completed"]):
            raise ValueError("Frozen region estimator changed")
        scores = predict_ridge(frame.loc[:, list(FEATURE_NAMES)], model, FEATURE_NAMES)
        chosen = (scores >= model["threshold"]) & (scores > 0)
        result[arm] = _signals(raw.loc[chosen], available.loc[chosen], issues[chosen], symbol, arm, scores[chosen])
    return result
