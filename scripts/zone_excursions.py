"""Fixed directional-excursion diagnostics for unchanged saved region policies.

This is a label/measurement module, not a strategy or profit calculation.
Targets after an economic stop/expiry remain forecast outcomes, never wins.
All required seconds through the horizon must exist, even after an early hit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


MULTIPLIERS = (1.5, 2.0, 3.0)
HORIZONS = (5, 10, 15, 30)
ENDPOINTS = ("ISSUE_DELAYED", "ENTRY_CONDITIONAL")
DEFINITIONS = tuple((a, h) for a in MULTIPLIERS for h in HORIZONS)


def definition_key(multiplier, horizon):
    return f"a{multiplier:g}_h{horizon:02d}"


def utc_seconds(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tz is None or stamp.value % 1_000_000_000:
        raise ValueError("A whole-second UTC-aware timestamp is required")
    return stamp.tz_convert("UTC").value // 1_000_000_000


def label_anchors(times, prices, anchors, *, start, end, endpoint):
    """Label fixed anchors with all twelve definitions, preserving all unknowns.

    Input is strictly sorted original quote arrays and a DataFrame with
    issue_time, anchor_time, atr, side. ISSUE_DELAYED must anchor at issue+61s.
    ENTRY_CONDITIONAL anchors must lie within [issue+62s,issue+901s]. Purges
    are32/46minutes from issue for ALL definitions, before seeing paths.
    Output y_key is−1unknown,0nohit,1hit; tts_key is seconds from the anchor.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError("Unknown endpoint")
    start, end = utc_seconds(start), utc_seconds(end)
    if start >= end:
        raise ValueError("start must precede end")
    if not isinstance(times, np.ndarray) or times.dtype.kind not in "iu" or times.ndim != 1:
        raise ValueError("Original quote seconds must be a one-dimensional integer array")
    if not isinstance(prices, np.ndarray) or prices.ndim != 1 or len(prices) != len(times):
        raise ValueError("One quote per original timestamp is required")
    if len(times) and np.any(np.diff(times) <= 0):
        raise ValueError("Original quote timestamps must be sorted and unique")
    selected = (times >= start) & (times < end)
    times = times[selected]
    raw = prices[selected]
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw) & (raw > 0)):
        raise ValueError("In-partition quotes must be positive finite real numbers")
    prices = raw.astype(float, copy=False)
    required = {"issue_time", "anchor_time", "atr", "side"}
    if not isinstance(anchors, pd.DataFrame) or not required.issubset(anchors):
        raise ValueError("Declared anchor metadata is missing")
    result = anchors.copy(deep=True).reset_index(drop=True)
    if anchors.duplicated(["issue_time", "anchor_time"]).any():
        raise ValueError("Duplicate opportunities in one endpoint/policy are refused")
    n = len(result)
    labels = np.full((n, len(DEFINITIONS)), -1, dtype=np.int8)
    firsts = np.full((n, len(DEFINITIONS)), np.nan)
    marks = np.full(n, np.nan)
    reasons = {h: np.full(n, "unprocessed", dtype=object) for h in HORIZONS}
    purge_seconds = (32 if endpoint == "ISSUE_DELAYED" else 46) * 60
    for ordinal, row in enumerate(result.itertuples(index=False)):
        issue, anchor = utc_seconds(row.issue_time), utc_seconds(row.anchor_time)
        if isinstance(row.side, (bool, np.bool_)) or not isinstance(row.side, (int, np.integer)) or row.side not in (-1, 1):
            raise ValueError("Native side must be integer +1/-1")
        if isinstance(row.atr, (bool, np.bool_)) or not isinstance(row.atr, (int, float, np.number)) or not math.isfinite(row.atr) or row.atr <= 0:
            raise ValueError("Raw issue ATR must be positive finite")
        if issue % 60 or (endpoint == "ISSUE_DELAYED" and anchor != issue + 61) or (
                endpoint == "ENTRY_CONDITIONAL" and not issue + 62 <= anchor <= issue + 901):
            raise ValueError("Anchor chronology differs from its frozen endpoint")
        if not start <= issue < end:
            reason = "outside_partition"
        elif issue + purge_seconds > end:
            reason = "planned_purge"
        else:
            reason = ""
        if reason:
            for h in HORIZONS: reasons[h][ordinal] = reason
            continue
        position = int(np.searchsorted(times, anchor))
        if position >= len(times) or int(times[position]) != anchor:
            for h in HORIZONS: reasons[h][ordinal] = "missing_anchor"
            continue
        price = float(prices[position])
        marks[ordinal] = price
        for h in HORIZONS:
            end_time = anchor + 60 * h
            right = int(np.searchsorted(times, end_time, side="right"))
            window_times, window_prices = times[position:right], prices[position:right]
            if len(window_times) != 60 * h + 1 or int(window_times[-1]) != end_time or np.any(np.diff(window_times) != 1):
                reasons[h][ordinal] = "incomplete_full_horizon"
                continue
            reasons[h][ordinal] = "known_full_horizon"
            excursions = int(row.side) * (window_prices[1:] - price)
            for multiplier in MULTIPLIERS:
                column = DEFINITIONS.index((multiplier, h))
                crossed = np.flatnonzero(excursions >= multiplier * float(row.atr))
                labels[ordinal, column] = int(len(crossed) > 0)
                if len(crossed):
                    firsts[ordinal, column] = int(window_times[int(crossed[0]) + 1]) - anchor
    result["anchor_price"] = marks
    for h in HORIZONS: result[f"reason_h{h:02d}"] = reasons[h]
    for column, (a, h) in enumerate(DEFINITIONS):
        key = definition_key(a, h)
        result[f"y_{key}"] = labels[:, column]
        result[f"tts_{key}"] = firsts[:, column]
    return result


def bootstrap_weights(start, end, *, repeats=9999, seed=20261008):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D")
    n = len(days)
    if n == 0 or repeats < 1:
        raise ValueError("A nonempty frozen calendar/repeat count is required")
    day = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=repeats).astype(float)
    starts = np.random.default_rng(seed).integers(0, n, size=(repeats, math.ceil(n / min(7, n))))
    sample_days = ((starts[:, :, None] + np.arange(min(7, n))) % n).reshape(repeats, -1)[:, :n]
    week = np.zeros((repeats, n), dtype=float)
    for i in range(repeats): week[i] = np.bincount(sample_days[i], minlength=n)
    return days, {"day": day, "week": week}


def _ratio(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b > 0)


def _ci(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]


def _p(samples, estimate):
    if estimate is None or estimate <= 0:
        return 1.0
    return (int((~np.isfinite(samples) | (samples - estimate >= estimate)).sum()) + 1) / (len(samples) + 1)


def compare_labels(model, baseline, multiplier, horizon, *, days, weights, opportunity_recall):
    """Paired calendar inference; opportunity recall is only a subset-clock rate.

    Counts include unknown outcomes explicitly. Complete-case precision/lift
    have separate partial-identification bounds; missing outcomes are not zero.
    No quantity returned here is a profit factor or monetary return.
    """
    key = definition_key(multiplier, horizon)
    def population(frame):
        y = frame[f"y_{key}"].to_numpy(np.int8)
        reasons = frame[f"reason_h{horizon:02d}"]
        planned = ~reasons.isin(("planned_purge", "outside_partition")).to_numpy()
        known = (y >= 0) & planned
        hits = y == 1
        stamps = pd.to_datetime(frame.issue_time, utc=True)
        ids = ((stamps.dt.floor("D") - days[0]) // pd.Timedelta(days=1)).to_numpy(int)
        if np.any((ids < 0) | (ids >= len(days))):
            raise ValueError("Opportunity leaves the frozen calendar")
        counts = np.bincount(ids[known], minlength=len(days)).astype(float)
        sums = np.bincount(ids[hits], minlength=len(days)).astype(float)
        n, k, issued = int(known.sum()), int(hits.sum()), int(planned.sum())
        unknown = issued - n
        return y, known, counts, sums, {"offered": len(frame), "planned": issued, "purged": len(frame) - issued,
                "known": n, "unknown": unknown, "hits": k, "active_known_days": int((counts > 0).sum()),
                "rate": k / n if n else None,
                "rate_bounds_with_unknown": [k / issued, (k + unknown) / issued] if issued else [None, None]}
    my, mk, mn, mh, m = population(model)
    by, bk, bn, bh, b = population(baseline)
    precision, base = m["rate"], b["rate"]
    lift = precision / base if precision is not None and base else None
    delta = precision - base if precision is not None and base is not None else None
    recall = m["hits"] / b["hits"] if opportunity_recall and b["hits"] else None
    if opportunity_recall:
        # A subset comparison must share the exact target at each opportunity.
        base_map = dict(zip(pd.to_datetime(baseline.issue_time, utc=True), by, strict=True))
        if len(base_map) != len(baseline) or any(base_map.get(t) != int(y) for t, y in
                zip(pd.to_datetime(model.issue_time, utc=True), my, strict=True)):
            raise ValueError("Opportunity recall requires exact subset-clock labels")
    known_hits = model.loc[model[f"y_{key}"].eq(1)]
    times = known_hits[f"tts_{key}"].to_numpy(float)
    delays = (pd.to_datetime(known_hits.anchor_time, utc=True) - pd.to_datetime(known_hits.issue_time, utc=True)).dt.total_seconds().to_numpy(float)
    inference = {}
    for kind, w in weights.items():
        mc, bc = np.einsum("ij,j->i", w, mn), np.einsum("ij,j->i", w, bn)
        ms, bs = np.einsum("ij,j->i", w, mh), np.einsum("ij,j->i", w, bh)
        mp, bp = _ratio(ms, mc), _ratio(bs, bc)
        differences, lifts = mp - bp, _ratio(mp, bp)
        r = _ratio(ms, bs) if opportunity_recall else np.full(len(w), np.nan)
        inference[kind] = {"precision_ci95": _ci(mp), "base_rate_ci95": _ci(bp),
                           "lift_ci95": _ci(lifts), "difference_ci95": _ci(differences),
                           "opportunity_recall_ci95": _ci(r), "p": _p(differences, delta),
                           "valid_lift_replicates": int(np.isfinite(lifts).sum()),
                           "undefined_lift_replicates": int((~np.isfinite(lifts)).sum())}
    lower, upper = m["rate_bounds_with_unknown"]
    bl, bu = b["rate_bounds_with_unknown"]
    lift_bounds = [lower / bu if lower is not None and bu else None,
                   upper / bl if upper is not None and bl else None]
    return {"atr_multiplier": multiplier, "horizon_minutes": horizon, "model": m, "baseline": b,
            "precision": precision, "base_rate": base, "lift": lift, "difference": delta,
            "opportunity_recall": recall, "unique_event_recall": None,
            "recall_interpretation": "positive_subset_clock_opportunities" if opportunity_recall else "not_identified_for_different_entry_anchors",
            "lift_bounds_with_unknown": lift_bounds,
            "median_time_to_excursion_minutes_from_anchor": float(np.median(times) / 60) if len(times) else None,
            "median_time_to_excursion_minutes_from_issue": float(np.median(times + delays) / 60) if len(times) else None,
            "inference": inference, "p": max(inference["day"]["p"], inference["week"]["p"]),
            "QUALIFIED": False, "profit_factor": "UNCHANGED_SEPARATE_ECONOMIC_RESULT"}
