"""Retain the existing evidence gates at the user's revised positive-profit target."""
from __future__ import annotations

from copy import deepcopy
import math

from app.research.zone_qualification import historical_zone_gate


def above(value, limit):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > limit


def stable_positive_gate(**arguments):
    # The immutable parent includes a PF1.5 point floor. Replace only that
    # criterion; leave development, uncertainty, stability and risk unchanged.
    output = deepcopy(historical_zone_gate(**arguments))
    reasons = [r for r in output["historical_rejection_reasons"] if r != "PF_below_1.5_or_unknown"]
    metrics = arguments["metrics"]
    if not (above(metrics.get("profit_factor"), 1) and above(metrics.get("mean_net_R"), 0)):
        reasons.append("PF_not_above_1_or_mean_not_positive_or_unknown")
    output.pop("PF_1_5_lower_CI_supported", None)
    output["historical_rejection_reasons"] = reasons
    output["historical_criteria_passed"] = not reasons
    output["point_criterion"] = "finite PF > 1 AND finite mean_net_R > 0"
    return output
