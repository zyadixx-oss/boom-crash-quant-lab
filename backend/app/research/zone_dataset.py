"""Stream original native quotes into complete minute bars for zone research.

The caller declares the entire end-exclusive interval. Source timestamps must
remain chronological even outside it; only quote values within it are read and
validated. Missing observations stay unknown. An actual missing second starts
a new run, while a chunk boundary or midnight alone never resets a run.

There is no source I/O, transformation, detector, indicator, label or execution
in this module. Outputs necessarily scale with the planned minute grid, but
native tick memory is bounded by the largest supplied chunk plus59 pending rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from app.research.tick_tail import SECOND_NS, _validated_ticks


MINUTE_NS = 60 * SECOND_NS
OHLC_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class ZoneDataset:
    """Full original-price minute grid and its observed-second coverage.

    A known ``coverage.run_id`` means all observed seconds in that minute
    belong to one run. OHLC is usable only when ``minute_valid`` is true,
    requiring every second00..59. Empty or mixed-run minutes have unknown IDs.
    """

    m1: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict[str, int | str | None]


def _bound(value: pd.Timestamp, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value) or value.tz is None:
        raise ValueError(f"{name} must be a timezone-aware UTC Timestamp")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    value = value.tz_convert("UTC").as_unit("ns")
    if value.value % MINUTE_NS:
        raise ValueError(f"{name} must be minute aligned")
    return value


def _source_index(chunk: pd.DataFrame, last_source_ns: int | None) -> pd.DatetimeIndex:
    """Validate the entire source chronology without inspecting quote values."""
    if not isinstance(chunk, pd.DataFrame):
        raise TypeError("Each tick chunk must be a DataFrame")
    if not chunk.columns.is_unique or "quote" not in chunk:
        raise ValueError("Each tick chunk requires a unique quote column")
    index = chunk.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Source timestamps must be timezone-aware UTC seconds")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Every source index must be strictly sorted and unique")
    if str(index.tz) != "UTC" and any(stamp.utcoffset() != timedelta(0) for stamp in index):
        raise ValueError("Source timestamps must be UTC")
    index = index.tz_convert("UTC").as_unit("ns").rename(None)
    if np.any(index.asi8 % SECOND_NS):
        raise ValueError("Source timestamps must be whole UTC seconds")
    if len(index) and last_source_ns is not None and int(index.asi8[0]) <= last_source_ns:
        raise ValueError("Source chunks must be strictly later without overlap")
    return index


def build_zone_dataset(chunks: Iterable[pd.DataFrame], *, start: pd.Timestamp,
                       end: pd.Timestamp) -> ZoneDataset:
    """Build exact complete raw M1 candles over an explicit planned interval.

    Bounds are minute-aligned UTC ``Timestamp`` objects, with start < end.
    Empty chunks and an empty iterable are supported. Every source index is
    validated before the interval filter, including timestamps after ``end``;
    quote values are validated only after filtering. No earlier or future quote
    can initialize a run or affect an output candle. Indices are unnamed UTC.
    """
    start, end = _bound(start, "start"), _bound(end, "end")
    if start >= end:
        raise ValueError("The planned interval must have start < end")
    grid = pd.date_range(start, end, freq="min", inclusive="left").as_unit("ns")
    n = len(grid)
    ohlc = np.full((n, 4), np.nan)
    counts = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    run_values = np.zeros(n, dtype=np.int64)
    run_known = np.zeros(n, dtype=bool)
    pending: pd.DataFrame | None = None
    last_source_ns: int | None = None
    last_observed_ns: int | None = None
    current_run = -1
    observed = 0
    first_observed: str | None = None

    def store_minutes(description: pd.DataFrame) -> None:
        if description.empty:
            return
        groups = description.groupby(description.index.floor("min"), sort=True)
        aggregate = groups.agg(
            open=("quote", "first"), high=("quote", "max"),
            low=("quote", "min"), close=("quote", "last"),
            observed_seconds=("quote", "size"), run_count=("run_id", "nunique"),
            run_id=("run_id", "first"),
        )
        positions = (aggregate.index.asi8 - start.value) // MINUTE_NS
        one_run = aggregate.run_count.eq(1).to_numpy()
        complete = aggregate.observed_seconds.eq(60).to_numpy() & one_run
        values = aggregate.loc[:, OHLC_COLUMNS].to_numpy(copy=True)
        values[~complete] = np.nan
        ohlc[positions] = values
        counts[positions] = aggregate.observed_seconds.to_numpy(dtype=np.int64)
        valid[positions] = complete
        run_known[positions] = one_run
        run_values[positions] = aggregate.run_id.to_numpy(dtype=np.int64)

    for chunk in chunks:
        index = _source_index(chunk, last_source_ns)
        if len(index):
            last_source_ns = int(index.asi8[-1])
        selected = (index >= start) & (index < end)
        if not selected.any():
            continue
        bounded = chunk.loc[selected, ["quote"]].copy()
        bounded.index = index[selected]
        bounded_index, quotes = _validated_ticks(bounded)
        times = bounded_index.asi8
        changes = np.empty(len(times), dtype=np.int64)
        changes[0] = int(last_observed_ns is None or int(times[0]) - last_observed_ns != SECOND_NS)
        changes[1:] = np.diff(times) != SECOND_NS
        runs = current_run + np.cumsum(changes, dtype=np.int64)
        current_run = int(runs[-1])
        last_observed_ns = int(times[-1])
        observed += len(times)
        first_observed = first_observed or bounded_index[0].isoformat()
        description = pd.DataFrame({"quote": quotes, "run_id": runs}, index=bounded_index)
        if pending is not None:
            description = pd.concat([pending, description])
            pending = None
        # An observed second59 closes the minute's population. Otherwise keep
        # its observed part, so splitting a source minute never loses coverage.
        last_minute = description.index[-1].floor("min")
        if int(description.index.asi8[-1]) % MINUTE_NS == 59 * SECOND_NS:
            store_minutes(description)
        else:
            before = description.index < last_minute
            store_minutes(description.loc[before])
            pending = description.loc[~before].copy()
    if pending is not None:
        store_minutes(pending)
    identities = pd.array(run_values, dtype="Int64")
    identities[~run_known] = pd.NA
    m1 = pd.DataFrame(ohlc, index=grid, columns=OHLC_COLUMNS)
    coverage = pd.DataFrame({"observed_seconds": counts, "minute_valid": valid,
                             "run_id": identities}, index=grid)
    summary: dict[str, int | str | None] = {
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "planned_minutes": n, "planned_seconds": n * 60,
        "observed_seconds": observed, "missing_seconds": n * 60 - observed,
        "first_observed": first_observed,
        "last_observed": (pd.Timestamp(last_observed_ns, tz="UTC").isoformat()
                          if last_observed_ns is not None else None),
        "runs": current_run + 1,
        "known_consecutive_increments": observed - (current_run + 1),
        "complete_minutes": int(valid.sum()),
        "incomplete_minutes": int((~valid).sum()),
        "empty_minutes": int((counts == 0).sum()),
    }
    return ZoneDataset(m1, coverage, summary)
