"""Causal CRT/alternative signals and paired inference for offline M1 research."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

SAFETY_FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
VARIANTS = (
    "CRT", "CRT_MSS", "CRT_DISPLACEMENT", "CRT_MSS_DISPLACEMENT",
    "CRT_MSS_DISPLACEMENT_FVG", "ATR_COMPRESSION", "BB_SQUEEZE",
    "CANDLE_COMPRESSION", "CANDLE_STRUCTURE", "SR_ALIGNMENT",
    "ATR_BB_COMPRESSION", "CRT_ATR", "CRT_BB", "CRT_CANDLE_COMPRESSION",
    "CRT_CANDLE_STRUCTURE", "CRT_SR",
)


def assert_offline() -> dict:
    # Settings also checks the repository .env; do not silently override unsafe input.
    from app.config import Settings

    settings = Settings()
    settings.assert_safe()
    return {name: False for name in SAFETY_FLAGS}


def load_m1(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    cols = ["epoch", "open", "high", "low", "close"]
    if not set(cols).issubset(raw):
        raise ValueError("Expected real M1 CSV with epoch/open/high/low/close")
    raw = raw[cols].apply(pd.to_numeric, errors="raise")
    if raw.empty or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Empty or non-finite source data")
    if (raw.epoch % 60 != 0).any():
        raise ValueError("Epochs must be UTC minute openings")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if ((raw.high < raw[["open", "close", "low"]].max(axis=1)) |
            (raw.low > raw[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bounds")
    duplicates = int(raw.duplicated("epoch").sum())
    if raw.groupby("epoch")[cols[1:]].nunique().gt(1).any().any():
        raise ValueError("Conflicting prices at duplicate epochs")
    raw = raw.drop_duplicates("epoch").sort_values("epoch")
    if len(raw) < 1000 or (raw.epoch.diff().dropna().mode().iloc[0] != 60):
        raise ValueError("Source is not a sufficiently long M1 grid")
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.epoch, unit="s", utc=True)).as_unit("ns")
    frame = raw[cols[1:]].reindex(pd.date_range(raw.index[0], raw.index[-1], freq="min").as_unit("ns"))
    missing = int(frame.close.isna().sum())
    audit = {
        "source_file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_rows": len(raw), "grid_minutes": len(frame),
        "duplicate_equal_rows_removed": duplicates, "missing_minutes": missing,
        "from_utc": str(frame.index[0]),
        "to_exclusive_utc": str(frame.index[-1] + pd.Timedelta(minutes=1)),
        "gap_intervals": int((raw.epoch.diff() > 60).sum()),
    }
    return frame, audit


def aggregate_complete(frame: pd.DataFrame, minutes: int, expected: int) -> pd.DataFrame:
    group = frame.resample(f"{minutes}min", closed="left", label="left")
    out = group.agg(open=("open", "first"), high=("high", "max"),
                    low=("low", "min"), close=("close", "last"))
    counts = group.close.count()
    out.loc[counts != expected, :] = np.nan
    return out


def features(m1: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction not in ("boom", "crash"):
        raise ValueError("direction must be boom or crash")
    m5 = aggregate_complete(m1, 5, 5)
    prev = m5.close.shift(1)
    tr = pd.concat([m5.high - m5.low, (m5.high - prev).abs(),
                    (m5.low - prev).abs()], axis=1).max(axis=1)
    tr = tr.where(m5.close.notna() & prev.notna())
    m5["atr"] = tr.rolling(14).mean()
    m5["sweep_atr"] = m5.atr.shift(1)
    body = (m5.close - m5.open).abs()
    ranges = m5.high - m5.low
    med_body = body.shift(1).rolling(20).median()
    directional = m5.close > m5.open if direction == "boom" else m5.close < m5.open
    m5["disp"] = directional & (med_body > 0) & (body >= 1.30 * med_body)
    if direction == "boom":
        m5["mss"] = m5.close > m5.high.shift(1).rolling(5).max()
        gap = m5.low - m5.high.shift(2)
        wick = m5[["open", "close"]].min(axis=1) - m5.low
        position = (m5.close - m5.low) / ranges.replace(0, np.nan)
        sr_level = m5.low.shift(1).rolling(12).min()
    else:
        m5["mss"] = m5.close < m5.low.shift(1).rolling(5).min()
        gap = m5.low.shift(2) - m5.high
        wick = m5.high - m5[["open", "close"]].max(axis=1)
        position = (m5.high - m5.close) / ranges.replace(0, np.nan)
        sr_level = m5.high.shift(1).rolling(12).max()
    m5["fvg"] = (gap >= 0.08 * m5.atr) & directional
    m5["atr_compression"] = tr.rolling(5).mean() / tr.rolling(30).mean() <= 0.80
    width = 4 * m5.close.rolling(20).std(ddof=0) / m5.close.rolling(20).mean()
    q35 = width.shift(1).rolling(100).quantile(0.35)
    m5["bb_squeeze"] = width <= q35
    comp_ratio = ranges.rolling(3).mean() / ranges.shift(3).rolling(20).median()
    body_ratio = (body / ranges.replace(0, np.nan)).rolling(3).mean()
    m5["candle_compression"] = (comp_ratio <= 0.65) & (body_ratio <= 0.35)
    m5["candle_structure"] = ((wick >= 0.5 * ranges) & (position >= 0.65) &
                               (ranges > 0) & (ranges <= 1.20 * m5.atr))
    m5["sr_alignment"] = (m5.close - sr_level).abs() <= 0.25 * m5.atr
    m5["feature_valid"] = (q35.notna() & med_body.notna() & sr_level.notna() &
                             tr.rolling(30).count().eq(30) & m5.atr.gt(0))
    return m5


def signals(m5: pd.DataFrame, direction: str) -> dict[str, np.ndarray]:
    """Offline replay of closed-bar state transitions; outputs are prefix invariant."""
    out = {name: [] for name in VARIANTS}
    closes = m5.index + pd.Timedelta(minutes=5)

    def add(name: str, i: int) -> None:
        if not bool(m5.feature_valid.iloc[i]):
            return
        out[name].append(i)

    standalone = {"ATR_COMPRESSION": "atr_compression", "BB_SQUEEZE": "bb_squeeze",
                  "CANDLE_COMPRESSION": "candle_compression",
                  "CANDLE_STRUCTURE": "candle_structure", "SR_ALIGNMENT": "sr_alignment"}
    # Do not suppress any signal because its later outcome window is incomplete.
    for i in range(len(m5)):
        for name, key in standalone.items():
            if m5[key].iloc[i]:
                add(name, i)
        if m5.atr_compression.iloc[i] and m5.bb_squeeze.iloc[i]:
            add("ATR_BB_COMPRESSION", i)

    h1 = aggregate_complete(m5[["open", "high", "low", "close"]], 60, 12)
    hmap = h1.to_dict("index")
    hour_groups = m5.groupby(m5.index.floor("h")).indices
    filters = {"CRT_ATR": "atr_compression", "CRT_BB": "bb_squeeze",
               "CRT_CANDLE_COMPRESSION": "candle_compression",
               "CRT_CANDLE_STRUCTURE": "candle_structure", "CRT_SR": "sr_alignment"}
    for hour, idxs in hour_groups.items():
        ref = hmap.get(hour - pd.Timedelta(hours=1))
        if ref is None or not np.isfinite([ref["high"], ref["low"]]).all():
            continue
        crh, crl = ref["high"], ref["low"]
        sweep = None
        for idx in idxs:
            a = m5.sweep_atr.iloc[idx]
            depth = crl - m5.low.iloc[idx] if direction == "boom" else m5.high.iloc[idx] - crh
            if m5.feature_valid.iloc[idx] and a > 0 and 0.10 * a <= depth <= 0.65 * a:
                sweep = int(idx)
                break
        if sweep is None:
            continue
        a = float(m5.sweep_atr.iloc[sweep])
        reclaim = None
        extreme = m5.low.iloc[sweep] if direction == "boom" else m5.high.iloc[sweep]
        for j in range(sweep, min(sweep + 3, len(m5))):
            row = m5.iloc[j]
            depth = crl - row.low if direction == "boom" else row.high - crh
            if not np.isfinite(row.close) or depth > 0.65 * a:
                break
            extreme = min(extreme, row.low) if direction == "boom" else max(extreme, row.high)
            if crl < row.close < crh:
                reclaim = j
                break
        if reclaim is None:
            continue
        add("CRT", reclaim)
        for name, key in filters.items():
            if m5[key].iloc[reclaim]:
                add(name, reclaim)
        done = set()
        for j in range(reclaim, min(reclaim + 4, len(m5))):
            row = m5.iloc[j]
            invalid = not np.isfinite(row.close) or (row.close < extreme if direction == "boom" else row.close > extreme)
            if invalid:
                break
            tests = {"CRT_MSS": row.mss, "CRT_DISPLACEMENT": row.disp,
                     "CRT_MSS_DISPLACEMENT": row.mss and row.disp,
                     "CRT_MSS_DISPLACEMENT_FVG": row.mss and row.disp and row.fvg}
            for name, qualifies in tests.items():
                if qualifies and name not in done:
                    add(name, j)
                    done.add(name)
    cooled = {}
    # Overlapping references can confirm out of order; cooldown must run by issue time.
    for name, indices in out.items():
        kept, last = [], None
        for i in sorted(set(indices)):
            if last is None or (closes[i] - last).total_seconds() >= 1800:
                kept.append(i)
                last = closes[i]
        cooled[name] = np.asarray(kept, dtype=int)
    return cooled


def outcomes(m1: pd.DataFrame, m5: pd.DataFrame, direction: str,
             multiplier: float, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcome uses only M1 candles starting at/after the M5 signal close."""
    times = (m5.index + pd.Timedelta(minutes=5)).as_unit("ns").asi8
    minute_times = m1.index.as_unit("ns").asi8
    pos = np.searchsorted(minute_times, times)
    in_bounds = pos + horizon <= len(m1)
    positions = np.minimum(pos[:, None] + np.arange(horizon), len(m1) - 1)
    future = m1.high.to_numpy()[positions] if direction == "boom" else m1.low.to_numpy()[positions]
    expected = times[:, None] + np.arange(horizon) * 60_000_000_000
    exact = (minute_times[positions] == expected).all(axis=1)
    valid = in_bounds & exact & np.isfinite(future).all(axis=1) & m5.feature_valid.to_numpy(bool)
    entry = m5.close.to_numpy()[:, None]
    move = future - entry if direction == "boom" else entry - future
    hit_mask = move >= multiplier * m5.atr.to_numpy()[:, None]
    hit = hit_mask.any(axis=1) & valid
    tts = np.where(hit, hit_mask.argmax(axis=1) + 1, np.nan).astype(float)
    return hit, tts, valid


def partitions(m1: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    start = m1.index[0]
    end = m1.index[-1] + pd.Timedelta(minutes=1)
    duration = end - start
    t40, t50, t60, t70 = [start + duration * f for f in (0.4, 0.5, 0.6, 0.7)]
    return {"development": (start, t70), "final_test": (t70, end),
            "wf1": (t40, t50), "wf2": (t50, t60), "wf3": (t60, t70)}


def eligible(m5: pd.DataFrame, valid30: np.ndarray,
             start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    close_times = m5.index + pd.Timedelta(minutes=5)
    return valid30 & (close_times >= start) & (close_times + pd.Timedelta(minutes=30) <= end)


def wilson(k: int, n: int) -> list[float | None]:
    if not n:
        return [None, None]
    z = 1.959963984540054
    p, den = k / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def bootstrap_context(m5: pd.DataFrame, mask: np.ndarray, start: pd.Timestamp,
                      end: pd.Timestamp, repeats: int, block_hours: int = 24) -> dict:
    step = pd.Timedelta(hours=block_hours)
    origin = start.floor(f"{block_hours}h")
    nblocks = int(math.ceil((end - origin) / step))
    ids = ((m5.index + pd.Timedelta(minutes=5) - origin) // step).to_numpy(int)
    ids = np.clip(ids, 0, nblocks - 1)
    rng = np.random.default_rng(20261004 + block_hours)
    weights = rng.multinomial(nblocks, np.ones(nblocks) / nblocks, repeats).astype(float)
    baseline_n = np.bincount(ids[mask], minlength=nblocks)
    return {"ids": ids, "weights": weights, "blocks": nblocks,
            "baseline_n": baseline_n, "evaluated_blocks": int((baseline_n > 0).sum()),
            "block_hours": block_hours}


def interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]


def summarize(hit: np.ndarray, tts: np.ndarray, mask: np.ndarray, signal_indices: np.ndarray,
              ctx: dict, m5: pd.DataFrame) -> dict:
    selected = signal_indices[mask[signal_indices]]
    n, k = len(selected), int(hit[selected].sum())
    nb, kb = int(mask.sum()), int(hit[mask].sum())
    base = kb / nb if nb else None
    precision = k / n if n else None
    lift = precision / base if n and base else None
    ids, w, blocks = ctx["ids"], ctx["weights"], ctx["blocks"]
    signal_n = np.bincount(ids[selected], minlength=blocks)
    signal_k = np.bincount(ids[selected[hit[selected]]], minlength=blocks)
    baseline_k = np.bincount(ids[mask & hit], minlength=blocks)
    bs_n, bs_k = w @ signal_n, w @ signal_k
    bb_n, bb_k = w @ ctx["baseline_n"], w @ baseline_k
    with np.errstate(divide="ignore", invalid="ignore"):
        bp, bb = bs_k / bs_n, bb_k / bb_n
        bl, delta = bp / bb, bp - bb
    bp[bs_n == 0] = np.nan
    bl[(bs_n == 0) | (bb_k == 0)] = np.nan
    delta[(bs_n == 0) | (bb_n == 0)] = np.nan
    actual_delta = precision - base if n and base is not None else None
    finite_delta = delta[np.isfinite(delta)]
    if actual_delta is None or actual_delta <= 0 or not len(finite_delta):
        pvalue = 1.0
    else:
        pvalue = (1 + int((finite_delta - actual_delta >= actual_delta).sum())) / (len(finite_delta) + 1)
    observed_tts = tts[selected]
    observed_tts = observed_tts[np.isfinite(observed_tts)]
    days = (m5.index[selected] + pd.Timedelta(minutes=5)).floor("D")
    return {
        "signals": n, "hits": k, "opportunities": nb, "positive_opportunities": kb,
        "precision": precision, "base_rate": base, "lift": lift,
        "opportunity_recall": k / kb if kb else None,
        "wilson_precision_ci95": wilson(k, n),
        "block_precision_ci95": interval(bp), "block_base_rate_ci95": interval(bb),
        "block_lift_ci95": interval(bl), "block_difference_ci95": interval(delta),
        "bootstrap_one_sided_p": pvalue,
        "bootstrap_valid_replicates": len(finite_delta),
        "block_hours": ctx["block_hours"], "evaluated_blocks": ctx["evaluated_blocks"],
        "days_with_signals": int(days.nunique()),
        "median_time_to_spike_min_upper_bound": float(np.median(observed_tts)) if len(observed_tts) else None,
        "sample_status": "ADEQUATE_FOR_RESEARCH_GATE" if n >= 200 and k >= 30 and days.nunique() >= 20 else "INSUFFICIENT_FOR_GATE",
    }


def holm_adjust(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda x: x["bootstrap_one_sided_p"])
    previous = 0.0
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, max(previous, (len(rows) - rank) * row["bootstrap_one_sided_p"]))
        row["holm_adjusted_p"] = adjusted
        previous = adjusted


def research_gate(test: dict, folds: list[dict]) -> tuple[bool, list[str]]:
    reasons = []
    if test["signals"] < 200:
        reasons.append("fewer_than_200_final_signals")
    if test["hits"] < 30:
        reasons.append("fewer_than_30_final_hits")
    if test["days_with_signals"] < 20 or test["evaluated_blocks"] < 20:
        reasons.append("fewer_than_20_signal_or_evaluation_days")
    lower = test["block_lift_ci95"][0]
    if lower is None or lower <= 1:
        reasons.append("day_block_lift_CI_does_not_exclude_one")
    if test.get("holm_adjusted_p", 1) >= 0.05:
        reasons.append("family_adjusted_p_not_below_0.05")
    if any(f["signals"] < 30 or f["lift"] is None or f["lift"] <= 1 for f in folds):
        reasons.append("not_positive_with_30_signals_in_each_development_fold")
    return not reasons, reasons
