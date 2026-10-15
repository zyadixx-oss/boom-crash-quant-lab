"""A distinct representation: transformed continuous context, raw observed age.

The native event clock is deliberately measured on original closed M5 bars.
Unknown raw ages remain unknown. This does not repair or overwrite the rejected
all-transformed44 study, and has no labels, learning, execution or order API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research.jump_multiframe import PairedMultiframeInputs, _stable_bb_inputs
from app.research.multiframe_signal import FEATURE_NAMES
from app.research.spike_hunter import assert_offline

AGE = "log1p_large_bar_age"


def raw_event_clock_inputs(inputs: PairedMultiframeInputs, direction: str) -> PairedMultiframeInputs:
    """Keep43 transformed values and substitute the same-row observed raw age.

    Re-evaluate intrinsic validity within each original contiguous representation
    run. Recomputing validity on a concatenated history would import pre-gap
    rolling windows. Both model populations must share the resulting mask.
    """
    assert_offline()
    if direction not in ("boom", "crash"):
        raise ValueError("direction must be boom or crash")
    if not isinstance(inputs, PairedMultiframeInputs) or tuple(inputs.feature_names) != FEATURE_NAMES:
        raise ValueError("Original paired44 input schema required")
    raw, hybrid = inputs.raw.copy(deep=True), inputs.transformed.copy(deep=True)
    availability = inputs.availability.copy(deep=True)
    if not raw.index.equals(hybrid.index) or not raw.index.equals(availability.index):
        raise ValueError("Paired feature populations must match")
    for frame in (raw, hybrid):
        if not frame.representation_run_id.equals(availability.run_id.rename("representation_run_id")):
            raise ValueError("Raw event clock and transformed context must share original runs")
    hybrid[AGE] = raw[AGE].copy()
    hybrid["feature_valid"] = False
    for run in availability.run_id.dropna().unique():
        mask = availability.run_id.eq(run).fillna(False)
        group = hybrid.loc[mask].copy(deep=True)
        _stable_bb_inputs(group, direction)
        # Only the age and validity semantics change; all43 other values stay
        # byte-for-byte those of the original transformed feature computation.
        hybrid.loc[mask, "feature_valid"] = group.feature_valid.to_numpy(bool)
    finite = np.isfinite(hybrid.loc[:, FEATURE_NAMES].to_numpy()).all(axis=1)
    if np.any(hybrid.feature_valid & ~finite):
        raise ValueError("Hybrid validity cannot enable unknown inputs")
    availability["transformed_feature_valid"] = hybrid.feature_valid.copy()
    availability["common_feature_valid"] = raw.feature_valid & hybrid.feature_valid
    return PairedMultiframeInputs(raw, hybrid, availability)


def named_signals(issued):
    """Keep the frozen paired learner internally unchanged; label the new arm."""
    if set(issued) != {"CLOCK", "RAW44", "TRANSFORMED44"}:
        raise ValueError("Complete paired issuance required")
    result = {"CLOCK": issued["CLOCK"].copy(), "RAW44": issued["RAW44"].copy(),
              "HYBRID44": issued["TRANSFORMED44"].copy()}
    result["HYBRID44"]["variant"] = "HYBRID44_SPIKE"
    return result
