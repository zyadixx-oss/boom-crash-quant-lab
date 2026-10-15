"""Individually replayed overlapping opportunity targets from frozen tick replay.

These labels are training targets, not a one-position portfolio, profit-factor
evidence or completed simulations toward the strategy sample target. No I/O,
signal discovery, fitting, acquisition or execution is available here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research.payoff_ticks import (
    MINUTE_NS, SECOND_NS, TICK_TRADE_COLUMNS, TickExitConfig,
    _integer, _number, _tick_arrays, _utc_ns, replay_ticks,
)
from app.research.spike_hunter import assert_offline

COUNTERS = ("issued", "filled", "completed", "censored", "missing_entry", "missing_path",
            "outside_partition", "purged", "overlap_skipped", "ambiguous")


def independent_tick_labels(
    ticks: pd.DataFrame, issued: list[dict], config: TickExitConfig,
    start: pd.Timestamp, end: pd.Timestamp, purge_minutes: int = 31,
) -> tuple[pd.DataFrame, dict]:
    """Replay each supplied opportunity separately using a bounded quote view.

    "Independent" in this API name means singleton execution only. Overlapping
    targets share quote paths; no statistical independence claim is made.
    Validate the full observed source once. Preserve missing seconds and use
    the original caller partition in every replay: never replace its end with
    a candidate's deadline. Callers must supply the effective training/fold end
    and separately enforce training size and strategy qualification policies.
    Only the fixed one-second/one-successor convention is supported.
    """
    assert_offline()
    if not isinstance(config, TickExitConfig):
        raise TypeError("config must be TickExitConfig")
    if config.max_gap_seconds != 1 or config.stop_latency_ticks != 1:
        raise ValueError("Independent labels require max_gap_seconds=1 and stop_latency_ticks=1")
    purge_minutes = _integer(purge_minutes, "purge_minutes", 0)
    start_ns, end_ns = _utc_ns(start, "start"), _utc_ns(end, "end")
    if start_ns >= end_ns:
        raise ValueError("start must be before end")
    times, _ = _tick_arrays(ticks)
    if not isinstance(issued, list):
        raise TypeError("issued must be a list of signal dictionaries")
    prepared, seen_times = [], set()
    for ordinal, signal in enumerate(issued):
        if not isinstance(signal, dict) or not {"signal_time", "atr", "side"}.issubset(signal):
            raise ValueError("Every signal requires signal_time, atr and side")
        stamp = _utc_ns(signal["signal_time"], "signal_time", require_minute=True)
        if stamp in seen_times:
            raise ValueError("Duplicate signal_time timestamps cannot inflate training opportunities")
        seen_times.add(stamp)
        _number(signal["atr"], "signal atr", 0., strict=True)
        side = signal["side"]
        if isinstance(side, (bool, np.bool_)) or not isinstance(side, (int, np.integer)) or side not in (-1, 1):
            raise ValueError("Signal side must be integer 1 or -1")
        if not isinstance(signal.get("variant", ""), str):
            raise ValueError("Signal variant must be a string")
        prepared.append((stamp, ordinal, dict(signal)))
    prepared.sort(key=lambda row: (row[0], row[1]))
    # The frozen engine supplies its exact typed empty schema and policy audit.
    empty, aggregate = replay_ticks(ticks.iloc[:0], [], config, start, end, purge_minutes)
    frames = []
    delay = int(config.entry_delay_minutes) * MINUTE_NS
    hold = int(config.max_hold_minutes) * MINUTE_NS
    for stamp, _, signal in prepared:
        nominal = stamp + delay
        deadline = nominal + hold
        left = int(np.searchsorted(times, nominal, side="left"))
        right = int(np.searchsorted(times, deadline + 2 * SECOND_NS, side="right"))
        # Positional slices are immutable views of the original gap-preserving
        # source. Including +2s is harmless and reveals a missing +1s successor.
        window = ticks.iloc[left:right]
        ledger, audit = replay_ticks(window, [signal], config, start, end, purge_minutes)
        for field in COUNTERS:
            aggregate[field] += audit[field]
        if not ledger.empty:
            frames.append(ledger)
    ledger = pd.concat(frames, ignore_index=True) if frames else empty
    assert list(ledger.columns) == TICK_TRADE_COLUMNS
    assert aggregate["issued"] == len(prepared) == sum(aggregate[k] for k in
                                                     ("filled", "missing_entry", "outside_partition", "purged", "overlap_skipped"))
    assert aggregate["filled"] == aggregate["completed"] + aggregate["censored"] == len(ledger)
    assert aggregate["censored"] == aggregate["missing_path"]
    assert aggregate["overlap_skipped"] == 0
    aggregate.update(
        label_kind="individually_replayed_overlapping_opportunity_training_targets",
        independent_signal_replays=len(prepared), source_rows_validated=len(times),
        labels_may_overlap=True, one_open_portfolio_constraint=False,
        strategy_returns=False, profit_factor_evidence=False,
        counts_toward_profit_sample_target=False, training_size_floor_enforced=False,
        no_statistical_independence_claim=True,
        partition_bounds_preserved=True,
        quote_window_rule="nominal_entry_through_nominal_deadline_plus2seconds_inclusive",
    )
    return ledger, aggregate
