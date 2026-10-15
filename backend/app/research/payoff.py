"""Offline M1 bracket research; prices and costs are hypotheses, never live fills."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from app.research.spike_hunter import assert_offline

MINUTE_NS = 60_000_000_000
TRADE_COLUMNS = [
    "signal_time", "entry_time", "exit_time", "entry", "exit", "atr",
    "gross_R", "net_R", "reason", "ambiguous", "holding_minutes", "censored",
    "variant", "planned_end", "missing_time",
]


@dataclass(frozen=True)
class BracketConfig:
    stop_atr: float
    target_atr: float
    max_hold_minutes: int
    entry_delay_minutes: int = 1
    round_trip_cost_atr: float = 0.10
    fill_mode: str = "adverse_extreme"
    same_bar_policy: str = "stop_first"

    def __post_init__(self) -> None:
        for name in ("stop_atr", "target_atr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("max_hold_minutes", 1), ("entry_delay_minutes", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.round_trip_cost_atr) or self.round_trip_cost_atr < 0:
            raise ValueError("round_trip_cost_atr must be finite and nonnegative")
        if self.fill_mode not in ("barrier_proxy", "adverse_extreme"):
            raise ValueError("Unknown fill_mode")
        if self.same_bar_policy not in ("stop_first", "target_first"):
            raise ValueError("Unknown same_bar_policy")


def _utc_ns(value: object, name: str, require_minute: bool = False) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp")
    if require_minute and stamp.value % MINUTE_NS:
        raise ValueError(f"{name} must be on a minute boundary")
    return stamp.tz_convert("UTC").value


def replay_brackets(
    m1: pd.DataFrame,
    issued: list[dict],
    config: BracketConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    purge_minutes: int = 31,
) -> tuple[pd.DataFrame, dict]:
    """Replay each exact delayed opening, with one concurrent hypothetical position.

    Intrabar exit times are upper bounds at the minute close. Barrier-proxy opening
    gaps exit at the opening; adverse-extreme stop fills consume the complete minute
    as a deliberately pessimistic stress, so their exit time is its close. Censored
    paths reserve the position until planned_end and have no measured PnL.
    """
    safety = assert_offline()
    if not isinstance(config, BracketConfig):
        raise TypeError("config must be BracketConfig")
    if isinstance(purge_minutes, bool) or not isinstance(purge_minutes, (int, np.integer)) or purge_minutes < 0:
        raise ValueError("purge_minutes must be a nonnegative integer")
    start_ns, end_ns = _utc_ns(start, "start"), _utc_ns(end, "end")
    if start_ns >= end_ns:
        raise ValueError("start must be before end")
    if not isinstance(m1.index, pd.DatetimeIndex) or m1.index.tz is None:
        raise ValueError("M1 index must be timezone-aware")
    if not m1.index.is_monotonic_increasing or not m1.index.is_unique:
        raise ValueError("M1 index must be sorted and unique")
    if not set(("open", "high", "low", "close")).issubset(m1.columns):
        raise ValueError("M1 requires open/high/low/close")
    minute_times = m1.index.as_unit("ns").asi8
    if np.any(minute_times % MINUTE_NS):
        raise ValueError("M1 timestamps must be UTC minute openings")
    # Array positions are searched once per entry; exact timestamps are checked
    # during replay, including sparse gaps. No per-minute pandas lookups occur.
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
        "purge_minutes": int(purge_minutes),
        "fill_interpretation": "hypothetical_ohlc_proxy_or_extreme_stress_not_measured_execution",
    }
    records = []
    busy_until = start_ns
    for signal_ns, _, atr, side, variant in prepared:
        entry_ns = signal_ns + delay_ns
        planned_end = entry_ns + hold_ns
        if signal_ns < start_ns or signal_ns >= end_ns:
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
        target = entry + side * config.target_atr * atr
        if not np.isfinite([risk, stop, target]).all() or min(stop, target) <= 0:
            raise ValueError("Nonpositive or non-finite bracket prices")
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
            opening_stop = side * (opening - stop) <= 0
            opening_target = side * (opening - target) >= 0
            if opening_target:
                record.update(exit=target, exit_time=stamp_ns, reason="tp")
                break
            if opening_stop and config.fill_mode == "barrier_proxy":
                record.update(exit=opening, exit_time=stamp_ns, reason="sl")
                break
            if not finite[row]:
                record.update(censored=True, missing_time=stamp_ns)
                break
            if not bounds[row]:
                raise ValueError(f"Invalid OHLC bounds in execution path at {pd.Timestamp(stamp_ns, tz='UTC')}")
            stop_hit = opening_stop or (lo[row] <= stop if side == 1 else hi[row] >= stop)
            target_hit = hi[row] >= target if side == 1 else lo[row] <= target
            ambiguous = stop_hit and target_hit and not opening_stop
            stop_selected = opening_stop or (stop_hit and (not target_hit or config.same_bar_policy == "stop_first"))
            if stop_selected:
                fill = min(opening, float(lo[row])) if side == 1 else max(opening, float(hi[row]))
                record.update(exit=fill if config.fill_mode == "adverse_extreme" else stop,
                              exit_time=stamp_ns + MINUTE_NS, reason="sl", ambiguous=bool(ambiguous))
                break
            if target_hit:
                record.update(exit=target, exit_time=stamp_ns + MINUTE_NS,
                              reason="tp", ambiguous=bool(ambiguous))
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
            audit["ambiguous"] += int(record["ambiguous"])
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
