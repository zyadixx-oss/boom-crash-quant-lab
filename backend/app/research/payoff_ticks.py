"""Causal public-tick quote proxies; no broker fills, orders or live promotion.

The nominal order time is issue+delay. Entry uses the first observed quote
STRICTLY AFTER it, within the declared maximum gap. Nominal expiry remains
nominal order time+hold, independent of the entry's one-tick delay. A stop is
triggered by the first subsequent adverse crossing at or before that expiry.
Primary stop exit uses the next observed quote, even after a jump or recovery;
latency=0 is a separately named optimistic trigger-quote sensitivity. A trigger
at expiry takes priority over timeout. Timeout uses the first quote strictly
after expiry. Quotes at a barrier are never fabricated and losses are not capped.

Required quote gaps before trigger/exit/timeout censor the path; they are never
interpolated. Data gaps after a completed exit are irrelevant. Planned partition
purges are evaluated before outcomes, including a maximum-gap exit allowance.
Missing entries and censored paths reserve occupancy to nominal expiry. These
rules specify historical quote-path arithmetic, not account execution, latency,
bid/ask prices, measured costs or monetary returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd

from app.research.payoff import TRADE_COLUMNS
from app.research.spike_hunter import assert_offline


SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
TICK_TRADE_COLUMNS = [*TRADE_COLUMNS, "nominal_entry_time", "trigger_time", "trigger_quote", "side"]


def _number(value: object, name: str, minimum: float, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (strict and result == minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return result


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class TickExitConfig:
    stop_atr: float = 2.0
    max_hold_minutes: int = 15
    entry_delay_minutes: int = 1
    round_trip_cost_atr: float = 0.10
    max_gap_seconds: int = 1
    stop_latency_ticks: int = 1

    def __post_init__(self) -> None:
        _number(self.stop_atr, "stop_atr", 0., strict=True)
        _number(self.round_trip_cost_atr, "round_trip_cost_atr", 0.)
        _integer(self.max_hold_minutes, "max_hold_minutes", 1)
        _integer(self.entry_delay_minutes, "entry_delay_minutes", 0)
        _integer(self.max_gap_seconds, "max_gap_seconds", 1)
        _integer(self.stop_latency_ticks, "stop_latency_ticks", 0)
        if self.stop_latency_ticks not in (0, 1):
            raise ValueError("stop_latency_ticks must be 0 or 1")


def _utc_ns(value: object, name: str, require_minute: bool = False) -> int:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError(f"{name} must be a valid timezone-aware timestamp")
        result = stamp.tz_convert("UTC").value
    except (TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp") from error
    if require_minute and result % MINUTE_NS:
        raise ValueError(f"{name} must be on a UTC minute boundary")
    return result


def _tick_arrays(ticks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(ticks, pd.DataFrame) or "quote" not in ticks.columns:
        raise ValueError("ticks must be a DataFrame with a quote column")
    if not isinstance(ticks.index, pd.DatetimeIndex) or ticks.index.tz is None or ticks.index.hasnans:
        raise ValueError("Tick index must be a valid timezone-aware DatetimeIndex")
    if not ticks.index.is_monotonic_increasing or not ticks.index.is_unique:
        raise ValueError("Tick index must be strictly increasing and unique")
    times = ticks.index.tz_convert("UTC").as_unit("ns").asi8
    if np.any(times % SECOND_NS):
        raise ValueError("Tick timestamps must use whole UTC seconds")
    quote = ticks["quote"]
    if (not isinstance(quote, pd.Series) or pd.api.types.is_bool_dtype(quote.dtype)
            or pd.api.types.is_complex_dtype(quote.dtype) or not pd.api.types.is_numeric_dtype(quote.dtype)):
        raise ValueError("Tick quotes must be real numeric values, not boolean, complex or textual")
    prices = quote.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Tick quotes must be finite and positive")
    return times, prices


def replay_ticks(
    ticks: pd.DataFrame, issued: list[dict], config: TickExitConfig,
    start: pd.Timestamp, end: pd.Timestamp, purge_minutes: int = 31,
) -> tuple[pd.DataFrame, dict]:
    """Replay frozen signals with exact chronological quotes and one position.

    Config's one-second primary tolerance requires an unbroken observed second
    grid over every needed path segment. Generic larger tolerances are explicit
    assumptions and never inferred from a path's outcome. No fitting, signal
    generation, collection or order API is available in this module.
    """
    safety = assert_offline()
    if not isinstance(config, TickExitConfig):
        raise TypeError("config must be TickExitConfig")
    purge_minutes = _integer(purge_minutes, "purge_minutes", 0)
    start_ns, end_ns = _utc_ns(start, "start"), _utc_ns(end, "end")
    if start_ns >= end_ns:
        raise ValueError("start must be before end")
    times, prices = _tick_arrays(ticks)
    if not isinstance(issued, list):
        raise TypeError("issued must be a list of signal dictionaries")
    prepared = []
    for ordinal, signal in enumerate(issued):
        if not isinstance(signal, dict) or not {"signal_time", "atr", "side"}.issubset(signal):
            raise ValueError("Every signal requires signal_time, atr and side")
        stamp = _utc_ns(signal["signal_time"], "signal_time", require_minute=True)
        atr = _number(signal["atr"], "signal atr", 0., strict=True)
        side = signal["side"]
        if isinstance(side, (bool, np.bool_)) or not isinstance(side, (int, np.integer)) or side not in (-1, 1):
            raise ValueError("Signal side must be integer 1 or -1")
        variant = signal.get("variant", "")
        if not isinstance(variant, str):
            raise ValueError("Signal variant must be a string")
        prepared.append((stamp, ordinal, atr, int(side), variant))
    prepared.sort(key=lambda row: (row[0], row[1]))
    delay_ns = int(config.entry_delay_minutes) * MINUTE_NS
    hold_ns = int(config.max_hold_minutes) * MINUTE_NS
    gap_ns = int(config.max_gap_seconds) * SECOND_NS
    purge_ns = purge_minutes * MINUTE_NS
    configuration = asdict(config)
    for name in ("max_hold_minutes", "entry_delay_minutes", "max_gap_seconds", "stop_latency_ticks"):
        configuration[name] = int(configuration[name])
    for name in ("stop_atr", "round_trip_cost_atr"):
        configuration[name] = float(configuration[name])
    audit = {
        "issued": len(prepared), "filled": 0, "completed": 0, "censored": 0,
        "missing_entry": 0, "missing_path": 0, "outside_partition": 0,
        "purged": 0, "overlap_skipped": 0, "ambiguous": 0,
        "safety": safety, "config": configuration, "purge_minutes": purge_minutes,
        "take_profit": None, "fill_interpretation": "quoteproxy_notbrokerfills",
        "entry_rule": "first_observed_quote_strictly_after_nominal_order_time",
        "stop_rule": ("next_observed_quote_strictly_after_trigger" if config.stop_latency_ticks == 1
                      else "optimistic_trigger_quote_sensitivity"),
        "expiry_rule": "first_observed_quote_strictly_after_nominal_expiry",
        "expiry_anchor": "nominal_entry_time_not_actual_entry_time",
        "deadline_priority": "stop_trigger_at_expiry_precedes_timeout",
        "partition_exit_allowance_seconds": int(config.max_gap_seconds),
    }
    records = []
    busy_until = start_ns
    for signal_ns, _, atr, side, variant in prepared:
        nominal_entry = signal_ns + delay_ns
        planned_end = nominal_entry + hold_ns
        if not start_ns <= signal_ns < end_ns:
            audit["outside_partition"] += 1
            continue
        # Both guards are planned; an early realized exit never rescues a purge.
        if signal_ns + purge_ns > end_ns or planned_end + gap_ns > end_ns:
            audit["purged"] += 1
            continue
        if nominal_entry < busy_until:
            audit["overlap_skipped"] += 1
            continue
        position = int(np.searchsorted(times, nominal_entry, side="right"))
        if position >= len(times) or times[position] - nominal_entry > gap_ns or times[position] >= planned_end:
            audit["missing_entry"] += 1
            busy_until = planned_end
            continue
        entry_time, entry = int(times[position]), float(prices[position])
        risk = float(config.stop_atr) * atr
        stop = entry - side * risk
        if not np.isfinite([risk, stop]).all() or risk <= 0 or stop <= 0:
            raise ValueError("Stop risk and price must be finite and positive")
        record = dict(
            signal_time=signal_ns, nominal_entry_time=nominal_entry, entry_time=entry_time,
            exit_time=planned_end, entry=entry, exit=np.nan, atr=atr,
            gross_R=np.nan, net_R=np.nan, reason="censored_missing_path", ambiguous=False,
            holding_minutes=(planned_end - entry_time) / MINUTE_NS, censored=False,
            variant=variant, planned_end=planned_end, missing_time=None,
            trigger_time=None, trigger_quote=np.nan, side=side,
        )
        audit["filled"] += 1
        previous = entry_time
        row = position + 1
        while True:
            if row >= len(times):
                record.update(censored=True, missing_time=previous + gap_ns)
                break
            stamp = int(times[row])
            if stamp - previous > gap_ns:
                record.update(censored=True, missing_time=previous + gap_ns)
                break
            price = float(prices[row])
            if stamp > planned_end:
                if stamp - planned_end > gap_ns:
                    record.update(censored=True, missing_time=planned_end + gap_ns)
                else:
                    record.update(exit=price, exit_time=stamp, reason="time")
                break
            crossed = price <= stop if side == 1 else price >= stop
            if crossed:
                record.update(trigger_time=stamp, trigger_quote=price)
                fill_row = row + int(config.stop_latency_ticks)
                if fill_row >= len(times):
                    record.update(censored=True, missing_time=stamp + gap_ns)
                    break
                fill_time = int(times[fill_row])
                if fill_time - stamp > gap_ns:
                    record.update(censored=True, missing_time=stamp + gap_ns)
                    break
                record.update(exit=float(prices[fill_row]), exit_time=fill_time, reason="sl")
                break
            previous = stamp
            row += 1
        if record["censored"]:
            audit["censored"] += 1
            audit["missing_path"] += 1
            busy_until = planned_end
        else:
            gross_R = side * (record["exit"] - entry) / risk
            net_R = gross_R - float(config.round_trip_cost_atr) / float(config.stop_atr)
            if not math.isfinite(gross_R) or not math.isfinite(net_R):
                raise ValueError("Quote-path return arithmetic produced a non-finite value")
            record.update(gross_R=gross_R, net_R=net_R)
            audit["completed"] += 1
            busy_until = int(record["exit_time"])
        record["holding_minutes"] = (int(record["exit_time"]) - entry_time) / MINUTE_NS
        records.append(record)
    trades = pd.DataFrame(records, columns=TICK_TRADE_COLUMNS)
    for name in ("signal_time", "nominal_entry_time", "entry_time", "exit_time", "planned_end", "missing_time", "trigger_time"):
        trades[name] = pd.to_datetime(trades[name], unit="ns", utc=True).astype("datetime64[ns, UTC]")
    for name in ("entry", "exit", "atr", "gross_R", "net_R", "holding_minutes", "trigger_quote"):
        trades[name] = trades[name].astype(float)
    for name in ("ambiguous", "censored"):
        trades[name] = trades[name].astype(bool)
    trades["side"] = trades.side.astype(int)
    assert audit["filled"] == audit["completed"] + audit["censored"] == len(trades)
    assert audit["censored"] == audit["missing_path"]
    assert audit["issued"] == (audit["filled"] + audit["missing_entry"] + audit["outside_partition"]
                               + audit["purged"] + audit["overlap_skipped"])
    return trades, audit
