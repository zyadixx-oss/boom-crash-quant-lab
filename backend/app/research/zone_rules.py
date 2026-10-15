"""Three fixed causal price-region hypotheses and a geometric context control.

Only original complete M1 prices enter these rules. Every00/30 issuance keeps
four audit rows, including rejected/unknown regions. H4/H1 context and M5 trigger
are closed native-direction bodies. M1 supplies the exact latest complete issue
price, without a directional-body filter. M15 setup differs by fixed family.
There are no future-confirmed pivots, fitted thresholds, outcomes or execution.
All rolling histories and context restart at an actual missing native second.
Conceptual CRT/Fibonacci/trend geometry is not evidence of predictive profit.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd

from app.research.learned_signal import _validated_m1
from app.research.spike_hunter import aggregate_complete, assert_offline
from app.research.tick_tail_signal import _index, _boolean
from app.research.zone_dataset import ZoneDataset


VARIANTS = ("CRT_RETEST", "FIB_RETRACE", "TREND_RETEST", "CONTEXT_GEOMETRIC")
OHLC = ("open", "high", "low", "close")
OUTPUT_COLUMNS = ("signal_time", "issue_time", "variant", "zone_id", "side", "zone_low", "zone_high",
                  "invalidation", "atr", "issue_price", "source_close", "eligible", "reason")
FRAME_MINUTES = (1, 5, 15, 60, 240)
MINUTE_NS = 60_000_000_000


def _validate(dataset: ZoneDataset):
    if not isinstance(dataset, ZoneDataset):
        raise TypeError("dataset must be ZoneDataset")
    index = _index(dataset.m1, "m1", MINUTE_NS)
    minute = _validated_m1(dataset.m1)
    coverage_index = _index(dataset.coverage, "coverage", MINUTE_NS)
    expected = pd.date_range(index[0], index[-1], freq="min").as_unit("ns")
    if not index.equals(expected) or not index.equals(coverage_index):
        raise ValueError("Dataset requires one shared full UTC minute grid")
    coverage = dataset.coverage.copy(deep=True)
    if any(name not in coverage for name in ("minute_valid", "observed_seconds", "run_id")):
        raise ValueError("Exact second-population coverage is required")
    valid = _boolean(coverage.minute_valid, "minute_valid")
    counts = coverage.observed_seconds
    if (not pd.api.types.is_integer_dtype(counts.dtype) or pd.api.types.is_bool_dtype(counts.dtype)
            or counts.isna().any() or not counts.between(0, 60).all()):
        raise ValueError("Observed second counts must be integers from 0 through 60")
    run = coverage.run_id
    known = run.dropna()
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0 for value in known):
        raise ValueError("Run ids must be nonnegative integers or unknown")
    if not known.is_monotonic_increasing:
        raise ValueError("Known run identities must be chronological")
    if not np.array_equal(valid, counts.eq(60).to_numpy() & run.notna().to_numpy()):
        raise ValueError("Minute validity must equal the complete60-second known-run population")
    if (counts.eq(0) & run.notna()).any():
        raise ValueError("Empty minutes cannot have a known run")
    original = dataset.m1.loc[:, OHLC].to_numpy(dtype=float)
    complete = np.isfinite(original).all(axis=1)
    if not np.array_equal(valid, complete) or not (complete | np.isnan(original).all(axis=1)).all():
        raise ValueError("Original OHLC must be complete exactly on valid minutes")
    return minute, coverage


def _frames(minute: pd.DataFrame, coverage: pd.DataFrame):
    result = {}
    for size in FRAME_MINUTES:
        pieces = []
        for run in coverage.run_id.dropna().unique():
            mask = coverage.run_id.eq(run).fillna(False).to_numpy(bool)
            source = minute.loc[mask]
            source = source.reindex(pd.date_range(source.index[0], source.index[-1], freq="min").as_unit("ns"))
            frame = source.copy() if size == 1 else aggregate_complete(source, size, size)
            frame["run_id"] = int(run)
            if size == 5:
                previous = frame.close.shift(1)
                tr = pd.concat([frame.high - frame.low, (frame.high - previous).abs(),
                                (frame.low - previous).abs()], axis=1).max(axis=1)
                tr = tr.where(frame.loc[:, OHLC].notna().all(axis=1) & previous.notna())
                frame["atr"] = tr.rolling(14, min_periods=14).mean()
            if size == 15:
                frame["ema20"] = frame.close.ewm(span=20, adjust=False, min_periods=20).mean()
                frame["ema20"] = frame.ema20.where(frame.close.rolling(20, min_periods=20).count().eq(20))
                frame["previous_ema20"] = frame.ema20.shift(1)
            pieces.append(frame)
        freq = f"{size}min"
        grid = pd.date_range(minute.index[0].floor(freq), minute.index[-1].floor(freq), freq=freq).as_unit("ns")
        if pieces:
            combined = pd.concat(pieces)
            combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index().reindex(grid)
        else:
            combined = pd.DataFrame(index=grid, columns=[*OHLC, "run_id"], dtype=float)
            if size == 5:
                combined["atr"] = np.nan
            if size == 15:
                combined["ema20"] = combined["previous_ema20"] = np.nan
        result[size] = combined
    return result


def _at(frame: pd.DataFrame, opening: pd.Timestamp):
    if opening not in frame.index:
        return None
    row = frame.loc[opening]
    values = row.loc[list(OHLC)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or pd.isna(row.run_id):
        return None
    return row


def _same_run(row, run):
    return row is not None and int(row.run_id) == run


def zone_candidates(dataset: ZoneDataset, direction: str) -> pd.DataFrame:
    """Return all fixed00/30 candidate dispositions, with immutable issue bands.

    Eligible row dictionaries use the exact region-replay input schema.
    ``eligible=False`` means no region was issued; its boundaries remain NaN.
    Waiting, activation, touch, invalidation, fill and exit belong to the replay
    module, not to these source-only geometric rules.
    """
    assert_offline()
    if direction not in ("boom", "crash"):
        raise ValueError("direction must be boom or crash")
    side = 1 if direction == "boom" else -1
    minute, coverage = _validate(dataset)
    frames = _frames(minute, coverage)
    end = minute.index[-1] + pd.Timedelta(minutes=1)
    issues = pd.date_range(minute.index[0].ceil("30min"), end, freq="30min", inclusive="left").as_unit("ns")
    output = []
    for issue in issues:
        rows = {size: _at(frames[size], issue.floor(f"{size}min") - pd.Timedelta(minutes=size))
                for size in FRAME_MINUTES}
        common_reason = "context_unavailable"
        atr, price, run = math.nan, math.nan, None
        if all(row is not None for row in rows.values()):
            run = int(rows[1].run_id)
            atr, price = float(rows[5].atr), float(rows[1].close)
            if not all(_same_run(row, run) for row in rows.values()):
                common_reason = "context_crosses_native_gap"
            elif not math.isfinite(atr) or atr <= 0:
                common_reason = "raw_m5_atr_unavailable"
            elif side * (rows[240].close - rows[240].open) < 0 or side * (rows[60].close - rows[60].open) < 0:
                common_reason = "h4_h1_context_disagrees"
            elif side * (rows[5].close - rows[5].open) <= 0:
                common_reason = "m5_trigger_absent"
            else:
                common_reason = ""
        for variant in VARIANTS:
            record = {"signal_time": issue, "issue_time": issue, "variant": variant,
                      "zone_id": f"{direction}:{variant}:{issue.isoformat()}", "side": side,
                      "zone_low": math.nan, "zone_high": math.nan, "invalidation": math.nan,
                      "atr": atr, "issue_price": price, "source_close": price,
                      "eligible": False, "reason": common_reason}
            if common_reason:
                output.append(record)
                continue
            m15 = rows[15]
            opening15 = issue - pd.Timedelta(minutes=15)
            lo, hi = math.nan, math.nan
            if variant == "CRT_RETEST":
                # Reference closed BEFORE the M15 setup began, even at :00.
                reference = _at(frames[60], opening15.floor("h") - pd.Timedelta(hours=1))
                if not _same_run(reference, run):
                    record["reason"] = "crt_reference_unavailable"
                else:
                    crl, crh = float(reference.low), float(reference.high)
                    depth = crl - m15.low if side == 1 else m15.high - crh
                    if not (0.10 * atr <= depth <= 0.65 * atr and crl < m15.close < crh):
                        record["reason"] = "crt_sweep_reclaim_absent"
                    else:
                        lo, hi = ((crl, crl + 0.10 * atr) if side == 1 else
                                  (crh - 0.10 * atr, crh))
            elif variant == "FIB_RETRACE":
                impulse = [_at(frames[15], opening15 - pd.Timedelta(minutes=15 * back)) for back in (3, 2, 1, 0)]
                if not all(_same_run(row, run) for row in impulse):
                    record["reason"] = "fib_impulse_unavailable"
                else:
                    lows, highs = [float(row.low) for row in impulse], [float(row.high) for row in impulse]
                    low_at, high_at = int(np.argmin(lows)), int(np.argmax(highs))
                    low, high = lows[low_at], highs[high_at]
                    ordered = low_at < high_at if side == 1 else high_at < low_at
                    if not ordered or not high > low or side * (m15.close - m15.open) <= 0:
                        record["reason"] = "fib_ordered_native_impulse_absent"
                    else:
                        span = high - low
                        lo, hi = ((high - 0.786 * span, high - 0.618 * span) if side == 1 else
                                  (low + 0.618 * span, low + 0.786 * span))
            elif variant == "TREND_RETEST":
                ema, previous = float(m15.ema20), float(m15.previous_ema20)
                if not math.isfinite(ema) or not math.isfinite(previous):
                    record["reason"] = "trend_ema_unavailable"
                elif side * (ema - previous) <= 0 or side * (m15.close - ema) <= 0:
                    record["reason"] = "trend_native_retest_absent"
                else:
                    lo, hi = ema - 0.05 * atr, ema + 0.05 * atr
            else:
                # Context-conditioned geometry, not unconditional market entry.
                endpoints = (price - side * 0.55 * atr, price - side * 0.45 * atr)
                lo, hi = min(endpoints), max(endpoints)
            if math.isfinite(lo) and math.isfinite(hi):
                invalidation = lo - 0.25 * atr if side == 1 else hi + 0.25 * atr
                if not all(math.isfinite(value) and value > 0 for value in (lo, hi, invalidation)) or not lo < hi:
                    record["reason"] = "region_not_positive_finite"
                elif not (hi < price if side == 1 else price < lo):
                    record["reason"] = "region_not_ahead_of_future_retracement"
                else:
                    record.update(zone_low=lo, zone_high=hi, invalidation=invalidation,
                                  eligible=True, reason="issued_fixed_region")
            output.append(record)
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)
