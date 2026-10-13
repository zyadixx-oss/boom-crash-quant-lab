"""Deterministic conditional event-time quote model, never actual arrival/fill evidence.

The caller supplies an already issued decision and explicit horizon. Equal-ms
quotes form an atomic bundle: preserve every row, use the adverse execution side,
and resolve simultaneous barrier evidence stop first. A bundle at T is closed
only when the next distinct timestamp arrives; barrier decisions use that later
timestamp. No ordering inside a bundle is claimed. The entry bundle is also
monitored for spread stops, subject to the same confirmation delay.
Barriers use effective entry (including assumed entry slippage); exit prices
always use observed opposite-side quotes, never clipped barrier levels.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Iterable, Literal


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(name + " must be an integer in the permitted range")
    return value


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be finite and numeric")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(name + " must be finite and numeric") from None
    if not math.isfinite(value):
        raise ValueError(name + " must be finite and numeric")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + " must be in the permitted range")
    return value


@dataclass(frozen=True)
class Quote:
    time_msc: int
    bid: float
    ask: float

    def __post_init__(self):
        _integer(self.time_msc, "time_msc")
        object.__setattr__(self, "bid", _real(self.bid, "bid", positive=True))
        object.__setattr__(self, "ask", _real(self.ask, "ask", positive=True))
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")


@dataclass(frozen=True)
class ExecutionConfig:
    entry_latency_ms: int = 250
    exit_latency_ms: int = 250
    max_gap_ms: int = 5000
    adverse_slippage_per_leg: float = 0.0

    def __post_init__(self):
        _integer(self.entry_latency_ms, "entry_latency_ms")
        _integer(self.exit_latency_ms, "exit_latency_ms")
        _integer(self.max_gap_ms, "max_gap_ms", positive=True)
        object.__setattr__(self, "adverse_slippage_per_leg",
                           _real(self.adverse_slippage_per_leg, "adverse_slippage_per_leg"))


@dataclass(frozen=True)
class TradeOutcome:
    status: Literal["COMPLETE", "UNKNOWN", "SKIP"]
    reason: str
    decision_ms: int
    direction: int
    horizon_end_ms: int
    config: ExecutionConfig
    stop_distance: float
    target_distance: float | None
    entry_time_msc: int | None = None
    entry_quote_price: float | None = None
    entry_price: float | None = None
    trigger_source_time_msc: int | None = None
    trigger_time_msc: int | None = None
    trigger_reason: str | None = None
    exit_time_msc: int | None = None
    exit_quote_price: float | None = None
    exit_price: float | None = None
    signed_move_after_spread_and_slippage: float | None = None
    both_barriers_in_bundle: bool = False
    entry_bundle: tuple[Quote, ...] = ()
    exit_bundle: tuple[Quote, ...] = ()
    model_label: str = field(init=False, default="CONDITIONAL_EVENT_TIME_NOT_REAL_ARRIVAL_OR_FILL")
    move_units: str = field(init=False, default="QUOTE_PRICE_UNITS_ONLY")
    fees_model: str = field(init=False, default="NOT_INCLUDED_NOT_VERIFIED")
    costs_verified: bool = field(init=False, default=False)
    execution_verified: bool = field(init=False, default=False)
    cash_pnl: None = field(init=False, default=None)


def _bundles(quotes):
    rows = tuple(quotes)
    bundles = []
    previous = None
    for quote in rows:
        if not isinstance(quote, Quote):
            raise ValueError("quotes must contain Quote instances")
        _integer(quote.time_msc, "time_msc")
        bid = _real(quote.bid, "bid", positive=True)
        ask = _real(quote.ask, "ask", positive=True)
        if bid > ask:
            raise ValueError("bid must not exceed ask")
        if previous is not None and quote.time_msc < previous:
            raise ValueError("quote timestamps must be nondecreasing; no sorting is performed")
        if not bundles or quote.time_msc != previous:
            bundles.append([])
        bundles[-1].append(quote)
        previous = quote.time_msc
    return tuple(tuple(bundle) for bundle in bundles)


def simulate_trade(
    quotes: Iterable[Quote],
    decision_ms: int,
    direction: int,
    horizon_end_ms: int,
    stop_distance: float,
    target_distance: float | None = None,
    config: ExecutionConfig | None = None,
) -> TradeOutcome:
    """Simulate one conditional exposure; overlap and strategy rules belong to callers.

    A barrier in a bundle at T is confirmed at the next distinct timestamp
    Tnext; exit latency begins at Tnext. If Tnext >= horizon_end_ms, the scheduled
    horizon wins. A timeout issues its exit at horizon_end_ms and needs a quote at
    or after horizon+exit_latency. No last-available quote is substituted.
    Missing entry/path/exit evidence produces UNKNOWN with no signed move.
    An entry deadline at/after the horizon, or its first eligible bundle after
    the horizon, produces SKIP without exposure. Max-gap checks include the
    decision-to-entry and trigger-to-exit latency intervals.
    """
    decision_ms = _integer(decision_ms, "decision_ms")
    horizon_end_ms = _integer(horizon_end_ms, "horizon_end_ms")
    if horizon_end_ms <= decision_ms:
        raise ValueError("horizon must follow the issued decision")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    stop_distance = _real(stop_distance, "stop_distance", positive=True)
    if target_distance is not None:
        target_distance = _real(target_distance, "target_distance", positive=True)
    config = ExecutionConfig() if config is None else config
    if not isinstance(config, ExecutionConfig):
        raise ValueError("config must be ExecutionConfig")
    _integer(config.entry_latency_ms, "entry_latency_ms")
    _integer(config.exit_latency_ms, "exit_latency_ms")
    _integer(config.max_gap_ms, "max_gap_ms", positive=True)
    _real(config.adverse_slippage_per_leg, "adverse_slippage_per_leg")
    groups = _bundles(quotes)
    times = tuple(bundle[0].time_msc for bundle in groups)
    base = dict(decision_ms=decision_ms, direction=direction, horizon_end_ms=horizon_end_ms,
                config=config, stop_distance=stop_distance, target_distance=target_distance)

    def outcome(status, reason, **values):
        return TradeOutcome(status=status, reason=reason, **base, **values)

    entry_due = decision_ms + config.entry_latency_ms
    if entry_due >= horizon_end_ms:
        return outcome("SKIP", "ENTRY_LATENCY_REACHES_HORIZON")
    entry_index = bisect_left(times, entry_due)
    if entry_index == len(groups):
        return outcome("UNKNOWN", "MISSING_ENTRY_QUOTE")
    if times[entry_index] >= horizon_end_ms:
        return outcome("SKIP", "NO_ENTRY_BEFORE_HORIZON")

    last_observed = decision_ms
    for index in range(bisect_left(times, decision_ms), entry_index + 1):
        if times[index] - last_observed > config.max_gap_ms:
            return outcome("UNKNOWN", "GAP_BEFORE_ENTRY")
        last_observed = times[index]
    entry_group = groups[entry_index]
    raw_entry = max(q.ask for q in entry_group) if direction == 1 else min(q.bid for q in entry_group)
    entry_price = raw_entry + direction * config.adverse_slippage_per_leg
    stop_price = entry_price - direction * stop_distance
    target_price = None if target_distance is None else entry_price + direction * target_distance
    if entry_price <= 0 or not math.isfinite(entry_price) or not math.isfinite(stop_price) or (
        target_price is not None and not math.isfinite(target_price)
    ):
        raise ValueError("derived entry/barrier prices must be finite")
    observed = dict(entry_time_msc=times[entry_index], entry_quote_price=raw_entry,
                    entry_price=entry_price, entry_bundle=entry_group)

    trigger_time = horizon_end_ms
    trigger_source_time = None
    trigger_reason = "HORIZON"
    both = False
    checked_index = entry_index
    for index in range(entry_index, len(groups)):
        time_msc = times[index]
        if time_msc > horizon_end_ms:
            break
        if time_msc - last_observed > config.max_gap_ms:
            return outcome("UNKNOWN", "GAP_DURING_HOLD", **observed)
        last_observed = time_msc
        checked_index = index
        # Seeing the first quote of the next timestamp closes this bundle.
        # Scheduled horizon issuance does not wait for a later bundle closure.
        if index + 1 == len(groups) or times[index + 1] >= horizon_end_ms:
            break
        side = [q.bid if direction == 1 else q.ask for q in groups[index]]
        stop = min(side) <= stop_price if direction == 1 else max(side) >= stop_price
        target = target_price is not None and (
            max(side) >= target_price if direction == 1 else min(side) <= target_price
        )
        if stop or target:
            confirmation_index = index + 1
            confirmation_time = times[confirmation_index]
            if confirmation_time - last_observed > config.max_gap_ms:
                return outcome("UNKNOWN", "GAP_DURING_HOLD", **observed)
            trigger_source_time = time_msc
            trigger_time = confirmation_time
            trigger_reason = "STOP" if stop else "TARGET"
            both = bool(stop and target)
            checked_index = confirmation_index
            last_observed = confirmation_time
            break

    triggered = dict(**observed, trigger_source_time_msc=trigger_source_time,
                     trigger_time_msc=trigger_time, trigger_reason=trigger_reason,
                     both_barriers_in_bundle=both)
    exit_due = trigger_time + config.exit_latency_ms
    exit_index = bisect_left(times, exit_due, lo=checked_index)
    if exit_index == len(groups):
        reason = "MISSING_HORIZON_EXIT_QUOTE" if trigger_reason == "HORIZON" else "MISSING_EXIT_QUOTE"
        return outcome("UNKNOWN", reason, **triggered)
    for index in range(checked_index + 1, exit_index + 1):
        if times[index] - last_observed > config.max_gap_ms:
            return outcome("UNKNOWN", "GAP_BEFORE_EXIT", **triggered)
        last_observed = times[index]
    exit_group = groups[exit_index]
    raw_exit = min(q.bid for q in exit_group) if direction == 1 else max(q.ask for q in exit_group)
    exit_price = raw_exit - direction * config.adverse_slippage_per_leg
    move = direction * (exit_price - entry_price)
    if exit_price <= 0 or not math.isfinite(exit_price) or not math.isfinite(move):
        raise ValueError("derived exit price and signed move must be finite")
    reason = "STOP_PRIORITY_SAME_MILLISECOND" if both else trigger_reason
    return outcome("COMPLETE", reason, **triggered, exit_time_msc=times[exit_index],
                   exit_quote_price=raw_exit, exit_price=exit_price, exit_bundle=exit_group,
                   signed_move_after_spread_and_slippage=move)
