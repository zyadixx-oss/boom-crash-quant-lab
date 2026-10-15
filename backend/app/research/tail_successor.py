"""One fixed post-latency public-feed response, with day-cluster uncertainty."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from numbers import Integral

import numpy as np

from app.research.tick_tail import FixedTailDetector, _log_return, _validated_ticks

COUNT_FIELDS = ("anchor_rows", "detector_pair_exclusions", "detector_eligible_anchors",
    "detected_event_anchors", "boundary_excluded_anchors", "boundary_excluded_events",
    "reference_eligible", "reference_known", "reference_unknown", "event_eligible",
    "event_known", "event_unknown", "overlapping_consecutive_event_windows")
SUM_FIELDS = ("reference_response_sum", "event_response_sum")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(value)


def collect_successor_days(ticks, detector, start_epoch, end_exclusive_epoch):
    """Classify anchors from current/past quotes, then preserve unknown outcomes."""
    if not isinstance(detector, FixedTailDetector):
        raise TypeError("Exact frozen FixedTailDetector required")
    start = _integer(start_epoch, "start_epoch")
    end = _integer(end_exclusive_epoch, "end_exclusive_epoch")
    if end <= start:
        raise ValueError("Nonempty chronological partition required")
    index, quotes = _validated_ticks(ticks)
    epochs = index.asi8 // 1_000_000_000
    first, last = np.searchsorted(epochs, [start, end])
    days = {}
    for i in range(int(first), int(last)):
        t = int(epochs[i]); day_start = t // 86400 * 86400
        date = datetime.fromtimestamp(day_start, timezone.utc).strftime("%Y-%m-%d")
        if date not in days:
            days[date] = {"date": date, **dict.fromkeys(COUNT_FIELDS, 0),
                          "_reference": [], "_event": [], "_previous_event": None}
        day = days[date]; day["anchor_rows"] += 1
        if i == 0 or int(epochs[i - 1]) != t - 1 or int(epochs[i - 1]) < day_start:
            day["detector_pair_exclusions"] += 1
            continue
        event = detector.side * _log_return(float(quotes[i - 1]), float(quotes[i])) > detector.threshold
        day["detector_eligible_anchors"] += 1
        day["detected_event_anchors"] += int(event)
        if t + 2 >= end or t + 2 >= day_start + 86400:
            day["boundary_excluded_anchors"] += 1
            day["boundary_excluded_events"] += int(event)
            continue
        day["reference_eligible"] += 1
        if event:
            day["event_eligible"] += 1
            previous_event = day["_previous_event"]
            if previous_event is not None and t - previous_event <= 2:
                day["overlapping_consecutive_event_windows"] += 1
            day["_previous_event"] = t
        known = i + 2 < len(epochs) and int(epochs[i + 1]) == t + 1 and int(epochs[i + 2]) == t + 2
        day["reference_known" if known else "reference_unknown"] += 1
        if event:
            day["event_known" if known else "event_unknown"] += 1
        if known:
            response = detector.side * _log_return(float(quotes[i + 1]), float(quotes[i + 2]))
            day["_reference"].append(response)
            if event:
                day["_event"].append(response)
    result = []
    for day in days.values():
        day["reference_response_sum"] = math.fsum(day.pop("_reference"))
        day["event_response_sum"] = math.fsum(day.pop("_event"))
        day.pop("_previous_event")
        result.append(day)
    return result


def _validated_days(days):
    if not isinstance(days, list) or not days:
        raise ValueError("At least one observed UTC day required")
    labels = []
    for day in days:
        if set(day) != {"date", *COUNT_FIELDS, *SUM_FIELDS}:
            raise ValueError("Exact daily sufficient-statistic schema required")
        label = day["date"]
        if not isinstance(label, str) or datetime.strptime(label, "%Y-%m-%d").strftime("%Y-%m-%d") != label:
            raise ValueError("Canonical UTC day required")
        labels.append(label)
        for key in COUNT_FIELDS:
            _integer(day[key], key)
        if day["anchor_rows"] == 0:
            raise ValueError("Resampling units must contain observed partition rows")
        for key in SUM_FIELDS:
            if isinstance(day[key], bool) or not isinstance(day[key], (float, int)) or not math.isfinite(day[key]):
                raise ValueError("Finite response sum required")
        if (day["anchor_rows"] != day["detector_pair_exclusions"] + day["detector_eligible_anchors"]
                or day["detector_eligible_anchors"] != day["boundary_excluded_anchors"] + day["reference_eligible"]
                or day["detected_event_anchors"] != day["boundary_excluded_events"] + day["event_eligible"]
                or day["reference_eligible"] != day["reference_known"] + day["reference_unknown"]
                or day["event_eligible"] != day["event_known"] + day["event_unknown"]
                or day["event_known"] > day["reference_known"]
                or day["event_unknown"] > day["reference_unknown"]
                or day["boundary_excluded_events"] > day["boundary_excluded_anchors"]
                or day["overlapping_consecutive_event_windows"] > max(0, day["event_eligible"] - 1)
                or (day["event_known"] == 0 and day["event_response_sum"] != 0)
                or (day["reference_known"] == 0 and day["reference_response_sum"] != 0)):
            raise ValueError("Daily cohort identities failed")
    if labels != sorted(set(labels)):
        raise ValueError("Observed days must be unique and chronological")
    return days


def summarize_successor(days, repeats=9999, seed=20261007):
    """Shared resamples of observed days; ticks are never independent resamples."""
    days = _validated_days(days)
    repeats = _integer(repeats, "repeats", 1); seed = _integer(seed, "seed")
    totals = {key: sum(day[key] for day in days) for key in COUNT_FIELDS}
    totals.update({key: math.fsum(day[key] for day in days) for key in SUM_FIELDS})
    ecount = np.array([day["event_known"] for day in days], dtype=float)
    esum = np.array([day["event_response_sum"] for day in days], dtype=float)
    weights = np.array([day["event_eligible"] for day in days], dtype=float)
    rcount = np.array([day["reference_known"] for day in days], dtype=float)
    rsum = np.array([day["reference_response_sum"] for day in days], dtype=float)
    rmean = np.divide(rsum, rcount, out=np.zeros(len(days)), where=rcount > 0)
    reference_missing = (weights > 0) & (rcount == 0)
    event_mean = totals["event_response_sum"] / totals["event_known"] if totals["event_known"] else None
    reference_mean = (math.fsum(float(w * r) for w, r in zip(weights, rmean, strict=True)) / totals["event_eligible"]
                      if totals["event_eligible"] and not reference_missing.any() else None)
    excess = event_mean - reference_mean if event_mean is not None and reference_mean is not None else None
    draws = np.random.default_rng(seed).multinomial(len(days), [1 / len(days)] * len(days), size=repeats)
    en = draws @ ecount; ew = draws @ weights
    event_values = np.divide(draws @ esum, en, out=np.full(repeats, np.nan), where=en > 0)
    valid_reference = (ew > 0) & ((draws @ reference_missing.astype(int)) == 0)
    reference_values = np.divide(draws @ (weights * rmean), ew, out=np.full(repeats, np.nan), where=valid_reference)
    difference_values = event_values - reference_values
    uncertainty = {}
    for name, values in (("event_mean", event_values), ("day_matched_reference_mean", reference_values),
                         ("excess_mean", difference_values)):
        finite = values[np.isfinite(values)]
        uncertainty[name] = {"finite_draws": int(len(finite)), "undefined_draws": repeats - int(len(finite)),
                             "conditional_ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None}
    complete = totals["event_unknown"] == totals["reference_unknown"] == 0
    interval = uncertainty["excess_mean"]["conditional_ci95"]
    rejected = (repeats == 9999 and seed == 20261007 and totals["event_eligible"] >= 100 and complete
                and uncertainty["excess_mean"]["undefined_draws"] == 0
                and interval is not None and interval[1] <= 0)
    return {"observed_days": [day["date"] for day in days], "observed_day_clusters": len(days),
            "totals": totals, "event_mean_known_subset": event_mean,
            "day_matched_reference_mean_known_subset": reference_mean, "excess_mean_known_subset": excess,
            "complete_responses": complete, "bootstrap_repeats": repeats, "bootstrap_seed": seed,
            "bootstrap": uncertainty, "conditional_positive_excess_rejected": bool(rejected),
            "interpretation": ("specified_positive_excess_conditionally_rejected" if rejected
                               else "insufficient_for_bounded_rejection"),
            "historical_strategy_candidate": False, "strategy_eligible": False, "profit_factor": None,
            "actual_money_profit": "NOT TESTED", "prospective_paper": "NOT TESTED"}
