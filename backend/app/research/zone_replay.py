"""Replay pre-issued price regions against original public quotes, offline only.

A region is immutable at issue. Observe invalidation immediately, allow a touch
after the declared activation delay, and fill at the next actual second. An
untouched region expires; jumps over the region are not fabricated touches.
One pending region OR position occupies each supplied single-model stream.
Unknown seconds censor the decision/path and reserve its maximum planned time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from app.research.payoff_ticks import (
    MINUTE_NS, SECOND_NS, _integer, _number, _tick_arrays, _utc_ns,
)
from app.research.spike_hunter import assert_offline


@dataclass(frozen=True)
class ZoneReplayConfig:
    activation_delay_minutes: int = 1
    wait_minutes: int = 15
    hold_minutes: int = 15
    stop_atr: float = 2.0
    cost_atr: float = 0.10
    purge_minutes: int = 31

    def __post_init__(self):
        _integer(self.activation_delay_minutes, "activation_delay_minutes", 0)
        _integer(self.wait_minutes, "wait_minutes", 1)
        _integer(self.hold_minutes, "hold_minutes", 1)
        _integer(self.purge_minutes, "purge_minutes", 0)
        _number(self.stop_atr, "stop_atr", 0., strict=True)
        _number(self.cost_atr, "cost_atr", 0.)
        if self.activation_delay_minutes >= self.wait_minutes:
            raise ValueError("Activation must precede the wait deadline")


TIME_FIELDS = ("signal_time", "activation_time", "wait_end", "maximum_end",
               "touch_time", "entry_time", "exit_time", "trigger_time", "missing_time")
FIELDS = (*TIME_FIELDS, "variant", "zone_id", "side", "zone_low", "zone_high",
          "invalidation", "issue_price", "atr", "entry", "exit", "status", "reason",
          "censored", "ambiguous", "gross_R", "net_R", "holding_minutes")


def replay_zones(ticks: pd.DataFrame, zones: list[dict], config: ZoneReplayConfig,
                 start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Return every issued region's disposition, including failures to enter.

    Input regions require a unique zone_id, variant, signal_time, side (+1/-1),
    positive zone_low/high, invalidation, issue_price and raw M5 atr. Boom regions
    lie strictly below issue_price and Crash regions strictly above it. A region
    cannot be invented after its touch. The caller supplies one symbol/variant.
    Costs are assumed quote-proxy costs, not measured broker costs.
    """
    safety = assert_offline()
    if not isinstance(config, ZoneReplayConfig):
        raise TypeError("config must be ZoneReplayConfig")
    start_ns, end_ns = _utc_ns(start, "start"), _utc_ns(end, "end")
    if start_ns >= end_ns:
        raise ValueError("start must precede end")
    if not isinstance(ticks, pd.DataFrame) or not isinstance(ticks.index, pd.DatetimeIndex):
        raise ValueError("ticks require a DatetimeIndex")
    # Out-of-partition values cannot alter arithmetic or availability.
    bounded = ticks.loc[(ticks.index >= start) & (ticks.index < end)].infer_objects()
    times, prices = _tick_arrays(bounded)
    if not isinstance(zones, list):
        raise TypeError("zones must be a list")
    prepared, identities, variants, sides = [], set(), set(), set()
    required = {"signal_time", "variant", "zone_id", "side", "zone_low", "zone_high",
                "invalidation", "issue_price", "atr"}
    for z in zones:
        if not isinstance(z, dict) or not required.issubset(z):
            raise ValueError("Region is missing required fields")
        stamp = _utc_ns(z["signal_time"], "signal_time", require_minute=True)
        side = z["side"]
        if isinstance(side, (bool, np.bool_)) or side not in (-1, 1) or not isinstance(side, (int, np.integer)):
            raise ValueError("side must be integer +1/-1")
        if not isinstance(z["variant"], str) or not z["variant"]:
            raise ValueError("variant must be a nonempty string")
        if not isinstance(z["zone_id"], str) or not z["zone_id"] or z["zone_id"] in identities:
            raise ValueError("zone_id must be unique and nonempty")
        values = {k: _number(z[k], k, 0., strict=True) for k in
                  ("zone_low", "zone_high", "invalidation", "issue_price", "atr")}
        lo, hi, inv, p = (values[k] for k in ("zone_low", "zone_high", "invalidation", "issue_price"))
        if not lo < hi or not (inv < lo < hi < p if side == 1 else p < lo < hi < inv):
            raise ValueError("Region must be ahead of a future adverse retracement with exterior invalidation")
        identities.add(z["zone_id"])
        variants.add(z["variant"])
        sides.add(side)
        prepared.append({**values, "signal_time": stamp, "side": side,
                         "variant": z["variant"], "zone_id": z["zone_id"]})
    if len(variants) > 1 or len(sides) > 1:
        raise ValueError("Replay accepts one direction and variant at a time")
    prepared.sort(key=lambda z: (z["signal_time"], z["zone_id"]))
    records, busy_until = [], start_ns
    for z in prepared:
        issue, side, atr = z["signal_time"], z["side"], z["atr"]
        activation = issue + config.activation_delay_minutes * MINUTE_NS
        wait_end = issue + config.wait_minutes * MINUTE_NS
        maximum_end = wait_end + config.hold_minutes * MINUTE_NS + SECOND_NS
        r = dict.fromkeys(FIELDS)
        r.update(z, activation_time=activation, wait_end=wait_end, maximum_end=maximum_end,
                 status="pending", censored=False, ambiguous=False,
                 entry=np.nan, exit=np.nan, gross_R=np.nan, net_R=np.nan, holding_minutes=np.nan)
        records.append(r)
        if not start_ns <= issue < end_ns:
            r.update(status="outside_partition", reason="outside_partition")
            continue
        if maximum_end >= end_ns or issue + config.purge_minutes * MINUTE_NS > end_ns:
            r.update(status="purged", reason="planned_partition_purge")
            continue
        if issue < busy_until:
            r.update(status="overlap_skipped", reason="pending_region_or_position")
            continue
        row = int(np.searchsorted(times, issue, side="right"))
        previous, touch = issue, None
        while previous < wait_end:
            if row >= len(times) or int(times[row]) != previous + SECOND_NS:
                r.update(status="waiting_gap", reason="unknown_before_touch", censored=True,
                         missing_time=previous + SECOND_NS)
                busy_until = maximum_end
                break
            stamp, price = int(times[row]), float(prices[row])
            invalid = side * (price - z["invalidation"]) <= 0
            if invalid:
                r.update(status="invalidated", reason="invalidation_before_touch", exit_time=stamp)
                busy_until = stamp
                break
            if stamp > activation and z["zone_low"] <= price <= z["zone_high"]:
                touch = stamp
                r["touch_time"] = touch
                break
            previous, row = stamp, row + 1
        if touch is None:
            if r["status"] == "pending":
                r.update(status="expired", reason="no_observed_touch", exit_time=wait_end)
                busy_until = wait_end
            continue
        row += 1
        if row >= len(times) or int(times[row]) != touch + SECOND_NS:
            r.update(status="entry_gap", reason="unknown_next_quote_entry", censored=True,
                     missing_time=touch + SECOND_NS)
            busy_until = maximum_end
            continue
        entry_time, entry = int(times[row]), float(prices[row])
        risk = config.stop_atr * atr
        stop = entry - side * risk
        if not math.isfinite(risk) or risk <= 0 or not math.isfinite(stop) or stop <= 0:
            raise ValueError("Nonpositive/nonfinite stop price or risk")
        expiry = touch + config.hold_minutes * MINUTE_NS
        r.update(entry_time=entry_time, entry=entry, status="open")
        row += 1
        previous = entry_time
        while True:
            if row >= len(times) or int(times[row]) != previous + SECOND_NS:
                r.update(status="path_gap", reason="unknown_after_entry", censored=True,
                         missing_time=previous + SECOND_NS, exit_time=expiry)
                busy_until = maximum_end
                break
            stamp, price = int(times[row]), float(prices[row])
            if stamp > expiry:
                r.update(status="completed", reason="time", exit_time=stamp, exit=price)
                break
            if side * (price - stop) <= 0:
                r["trigger_time"] = stamp
                row += 1
                if row >= len(times) or int(times[row]) != stamp + SECOND_NS:
                    r.update(status="path_gap", reason="unknown_stop_fill", censored=True,
                             missing_time=stamp + SECOND_NS, exit_time=expiry)
                    busy_until = maximum_end
                else:
                    r.update(status="completed", reason="sl", exit_time=int(times[row]),
                             exit=float(prices[row]))
                break
            previous, row = stamp, row + 1
        r["holding_minutes"] = (r["exit_time"] - entry_time) / MINUTE_NS
        if r["status"] == "completed":
            gross = side * (r["exit"] - entry) / risk
            net = gross - config.cost_atr / config.stop_atr
            if not math.isfinite(gross) or not math.isfinite(net):
                raise ValueError("Nonfinite payoff")
            r.update(gross_R=gross, net_R=net)
            busy_until = r["exit_time"]
    table = pd.DataFrame(records, columns=FIELDS)
    for field in TIME_FIELDS:
        # Int64 avoids loss of nanosecond precision when missing values exist.
        table[field] = pd.to_datetime(pd.array(table[field], dtype="Int64"), unit="ns", utc=True)
    for field in ("censored", "ambiguous"):
        table[field] = table[field].astype(bool)
    for field in ("zone_low", "zone_high", "invalidation", "issue_price", "atr", "entry",
                  "exit", "gross_R", "net_R", "holding_minutes"):
        table[field] = pd.to_numeric(table[field], errors="raise").astype(float)
    table["side"] = table.side.astype(np.int64)
    counts = {str(k): int(v) for k, v in table.status.value_counts().items()}
    return table, {"issued": len(table), "statuses": counts, "config": asdict(config),
                   "safety": safety, "quote_proxy_only": True, "broker_execution": "NOT TESTED",
                   "unknown": int(table.censored.sum()),
                   "filled": int(table.entry_time.notna().sum())}


def zone_trades(events: pd.DataFrame) -> pd.DataFrame:
    """Metrics-compatible ledger; all unfilled outcomes remain in events."""
    return events.loc[events.entry_time.notna()].copy()
