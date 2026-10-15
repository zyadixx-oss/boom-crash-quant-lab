"""Offline stop-or-time exits with uncapped upside; never a broker fill claim."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from app.research.payoff import TRADE_COLUMNS
from app.research.spike_hunter import assert_offline

MINUTE_NS = 60_000_000_000


@dataclass(frozen=True)
class TimedExitConfig:
    stop_atr: float
    max_hold_minutes: int
    entry_delay_minutes: int = 1
    round_trip_cost_atr: float = 0.10
    fill_mode: str = "adverse_extreme"

    def __post_init__(self) -> None:
        if isinstance(self.stop_atr, bool) or not math.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        for name, minimum in (("max_hold_minutes", 1), ("entry_delay_minutes", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.round_trip_cost_atr, bool)
                or not math.isfinite(self.round_trip_cost_atr) or self.round_trip_cost_atr < 0):
            raise ValueError("round_trip_cost_atr must be finite and nonnegative")
        if self.fill_mode not in ("barrier_proxy", "adverse_extreme"):
            raise ValueError("Unknown fill_mode")


def _utc_ns(value: object, name: str, require_minute: bool = False) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp")
    if require_minute and stamp.value % MINUTE_NS:
        raise ValueError(f"{name} must be on a minute boundary")
    return stamp.tz_convert("UTC").value


def replay_timed(
    m1: pd.DataFrame,
    issued: list[dict],
    config: TimedExitConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    purge_minutes: int = 31,
) -> tuple[pd.DataFrame, dict]:
    """Replay exact delayed opens, with one position and no take-profit barrier.

    A known opening SL gap can fill at that open under barrier_proxy. The stress
    model consumes the complete minute and uses its worst quote. Intrabar stop
    exits are timed at minute close; timeout consumes exactly H minutes. Missing
    paths reserve occupancy through planned_end and never receive a measured R.
    Signal cooldown belongs to the signal generator, not this execution layer.
    """
    safety = assert_offline()
    if not isinstance(config, TimedExitConfig):
        raise TypeError("config must be TimedExitConfig")
    if isinstance(purge_minutes, bool) or not isinstance(purge_minutes, (int, np.integer)) or purge_minutes < 0:
        raise ValueError("purge_minutes must be a nonnegative integer")
    start_ns, end_ns = _utc_ns(start, "start"), _utc_ns(end, "end")
    if start_ns >= end_ns:
        raise ValueError("start must be before end")
    if not isinstance(m1.index, pd.DatetimeIndex) or m1.index.tz is None:
        raise ValueError("M1 index must be timezone-aware")
    if not m1.index.is_monotonic_increasing or not m1.index.is_unique:
        raise ValueError("M1 index must be sorted and unique")
    if not {"open", "high", "low", "close"}.issubset(m1.columns):
        raise ValueError("M1 requires open/high/low/close")
    minute_times = m1.index.as_unit("ns").asi8
    if np.any(minute_times % MINUTE_NS):
        raise ValueError("M1 timestamps must be UTC minute openings")
    op, hi, lo, cl = (m1[key].to_numpy(dtype=float, copy=False) for key in ("open", "high", "low", "close"))
    finite = np.isfinite(op) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl)
    valid_open = np.isfinite(op) & (op > 0)
    bounds = (lo > 0) & (hi >= op) & (hi >= cl) & (hi >= lo) & (lo <= op) & (lo <= cl)
    prepared = []
    for ordinal, signal in enumerate(issued):
        signal_ns = _utc_ns(signal["signal_time"], "signal_time", require_minute=True)
        atr = float(signal["atr"])
        side = signal["side"]
        if not math.isfinite(atr) or atr <= 0:
            raise ValueError("Signal ATR must be finite and positive")
        if isinstance(side, bool) or side not in (1, -1):
            raise ValueError("Signal side must be 1 or -1")
        prepared.append((signal_ns, ordinal, atr, int(side), signal.get("variant", "")))
    prepared.sort(key=lambda row: (row[0], row[1]))
    delay_ns = int(config.entry_delay_minutes) * MINUTE_NS
    hold_ns = int(config.max_hold_minutes) * MINUTE_NS
    purge_ns = int(purge_minutes) * MINUTE_NS
    audit = {
        "issued": len(prepared), "filled": 0, "completed": 0, "censored": 0,
        "missing_entry": 0, "outside_partition": 0, "purged": 0,
        "overlap_skipped": 0, "ambiguous": 0, "safety": safety,
        "config": {**asdict(config), "max_hold_minutes": int(config.max_hold_minutes),
                   "entry_delay_minutes": int(config.entry_delay_minutes)},
        "purge_minutes": int(purge_minutes), "take_profit": None,
        "fill_interpretation": "hypothetical_ohlc_proxy_or_extreme_stress_not_measured_execution",
    }
    records = []
    busy_until = start_ns
    for signal_ns, _, atr, side, variant in prepared:
        entry_ns = signal_ns + delay_ns
        planned_end = entry_ns + hold_ns
        if not start_ns <= signal_ns < end_ns:
            audit["outside_partition"] += 1
            continue
        if signal_ns + purge_ns > end_ns or planned_end > end_ns:
            audit["purged"] += 1
            continue
        if entry_ns < busy_until:
            audit["overlap_skipped"] += 1
            continue
        position = int(np.searchsorted(minute_times, entry_ns))
        if position >= len(minute_times) or minute_times[position] != entry_ns or not valid_open[position]:
            audit["missing_entry"] += 1
            continue
        entry = float(op[position])
        risk = config.stop_atr * atr
        stop = entry - side * risk
        if not np.isfinite([risk, stop]).all() or stop <= 0:
            raise ValueError("Nonpositive or non-finite stop price")
        record = dict(signal_time=signal_ns, entry_time=entry_ns, exit_time=planned_end,
                      entry=entry, exit=np.nan, atr=atr, gross_R=np.nan, net_R=np.nan,
                      reason="censored_missing_path", ambiguous=False,
                      holding_minutes=float(config.max_hold_minutes), censored=False,
                      variant=variant, planned_end=planned_end, missing_time=None)
        audit["filled"] += 1
        for minute in range(config.max_hold_minutes):
            stamp_ns = entry_ns + minute * MINUTE_NS
            row = position + minute
            if row >= len(minute_times) or minute_times[row] != stamp_ns or not valid_open[row]:
                record.update(censored=True, missing_time=stamp_ns)
                break
            opening = float(op[row])
            opening_stop = opening <= stop if side == 1 else opening >= stop
            if opening_stop and config.fill_mode == "barrier_proxy":
                record.update(exit=opening, exit_time=stamp_ns, reason="sl")
                break
            if not finite[row]:
                record.update(censored=True, missing_time=stamp_ns)
                break
            if not bounds[row]:
                raise ValueError(f"Invalid OHLC bounds in execution path at {pd.Timestamp(stamp_ns, tz='UTC')}")
            stop_hit = opening_stop or (lo[row] <= stop if side == 1 else hi[row] >= stop)
            if stop_hit:
                price = float(lo[row]) if side == 1 else float(hi[row])
                record.update(exit=price if config.fill_mode == "adverse_extreme" else stop,
                              exit_time=stamp_ns + MINUTE_NS, reason="sl")
                break
            if minute == config.max_hold_minutes - 1:
                record.update(exit=float(cl[row]), exit_time=planned_end, reason="time")
        if record["censored"]:
            audit["censored"] += 1
            busy_until = planned_end
        else:
            record["gross_R"] = side * (record["exit"] - entry) / risk
            record["net_R"] = record["gross_R"] - config.round_trip_cost_atr / config.stop_atr
            audit["completed"] += 1
            busy_until = int(record["exit_time"])
        record["holding_minutes"] = (int(record["exit_time"]) - entry_ns) / MINUTE_NS
        records.append(record)
    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)
    for name in ("signal_time", "entry_time", "exit_time", "planned_end", "missing_time"):
        trades[name] = pd.to_datetime(trades[name], unit="ns", utc=True).astype("datetime64[ns, UTC]")
    for name in ("entry", "exit", "atr", "gross_R", "net_R", "holding_minutes"):
        trades[name] = trades[name].astype(float)
    for name in ("ambiguous", "censored"):
        trades[name] = trades[name].astype(bool)
    assert audit["filled"] == audit["completed"] + audit["censored"] == len(trades)
    assert audit["issued"] == (audit["filled"] + audit["outside_partition"] + audit["purged"]
                               + audit["overlap_skipped"] + audit["missing_entry"])
    return trades, audit
