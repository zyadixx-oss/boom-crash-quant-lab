#!/usr/bin/env python3
"""Independent saved-region/path/metric audit; never imports research engines.

This verifies immutable saved candidates, all pending-region dispositions,
actual original-quote entries/exits, ledger arithmetic and day/week PF intervals.
It does not independently derive the CRT/Fibonacci/EMA feature rules, and cannot
establish fresh OOS, broker fills or profitability. Missing rows remain unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
VARIANTS = ("CRT_RETEST", "FIB_RETRACE", "TREND_RETEST", "CONTEXT_GEOMETRIC")
SYMBOLS = ("BOOM600", "CRASH600")
TIME_FIELDS = ("signal_time", "activation_time", "wait_end", "maximum_end", "touch_time",
               "entry_time", "exit_time", "trigger_time", "missing_time")


def sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def epoch(value):
    if value is None or value == "" or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tz is None or stamp.value % 1_000_000_000:
        raise ValueError("Audit timestamps require complete UTC seconds")
    return stamp.value // 1_000_000_000


class Audit:
    def __init__(self):
        self.checks, self.errors, self.pins = 0, [], {}

    def check(self, condition, message):
        self.checks += 1
        if not condition and len(self.errors) < 100:
            self.errors.append(message)

    def equal(self, actual, expected, message):
        if expected is None:
            valid = actual is None or pd.isna(actual)
        elif isinstance(expected, float):
            valid = isinstance(actual, (int, float, np.number)) and math.isfinite(actual) and math.isclose(
                float(actual), expected, rel_tol=2e-12, abs_tol=2e-12)
        else:
            valid = actual == expected
        self.check(bool(valid), f"{message}: {actual!r} != {expected!r}")

    def pin(self, name, expected=None):
        path = (ROOT / name).resolve()
        path.relative_to(ROOT.resolve())
        actual = sha(path)
        self.check(expected is None or actual == expected, f"Pin changed: {name}")
        self.pins[str(path.relative_to(ROOT))] = actual
        if expected is not None and actual != expected:
            raise ValueError(f"Required pin changed before decoding: {name}")
        return path


def _first(mask, times):
    hits = np.flatnonzero(mask)
    return int(times[hits[0]]) if len(hits) else None


def independent_path(times, quotes, z, cfg, start, end, busy):
    """Vector first-event oracle, separately written from production state loop."""
    issue, side, atr = epoch(z["signal_time"]), int(z["side"]), float(z["atr"])
    activation = issue + 60 * cfg["activation_delay_minutes"]
    deadline = issue + 60 * cfg["wait_minutes"]
    maximum = deadline + 60 * cfg["hold_minutes"] + 1
    result = {k: None for k in TIME_FIELDS}
    result.update(signal_time=issue, activation_time=activation, wait_end=deadline, maximum_end=maximum,
                  status=None, reason=None, censored=False, ambiguous=False, entry=None, exit=None,
                  gross_R=None, net_R=None, holding_minutes=None)
    if not start <= issue < end:
        result.update(status="outside_partition", reason="outside_partition")
        return result, busy
    if maximum >= end or issue + 60 * cfg["purge_minutes"] > end:
        result.update(status="purged", reason="planned_partition_purge")
        return result, busy
    if issue < busy:
        result.update(status="overlap_skipped", reason="pending_region_or_position")
        return result, busy
    left, right = np.searchsorted(times, [issue + 1, deadline + 1])
    t, p = times[left:right], quotes[left:right]
    # First absent second is earlier than any subsequent observed event.
    expected = np.arange(issue + 1, deadline + 1, dtype=np.int64)
    missing = expected[~np.isin(expected, t, assume_unique=True)]
    gap = int(missing[0]) if len(missing) else None
    invalid = _first(side * (p - float(z["invalidation"])) <= 0, t)
    touch = _first((t > activation) & (p >= float(z["zone_low"])) & (p <= float(z["zone_high"])), t)
    events = [(stamp, priority, kind) for stamp, priority, kind in
              ((gap, 0, "gap"), (invalid, 1, "invalid"), (touch, 2, "touch")) if stamp is not None]
    chosen = min(events) if events else (deadline, 3, "expired")
    at, _, kind = chosen
    if kind == "gap":
        result.update(status="waiting_gap", reason="unknown_before_touch", censored=True, missing_time=at)
        return result, maximum
    if kind == "invalid":
        result.update(status="invalidated", reason="invalidation_before_touch", exit_time=at)
        return result, at
    if kind == "expired":
        result.update(status="expired", reason="no_observed_touch", exit_time=deadline)
        return result, deadline
    result["touch_time"] = touch
    entry_at = touch + 1
    entry_pos = int(np.searchsorted(times, entry_at))
    if entry_pos >= len(times) or int(times[entry_pos]) != entry_at:
        result.update(status="entry_gap", reason="unknown_next_quote_entry", censored=True,
                      missing_time=entry_at)
        return result, maximum
    entry = float(quotes[entry_pos])
    result.update(entry_time=entry_at, entry=entry)
    expiry = touch + 60 * cfg["hold_minutes"]
    risk = cfg["stop_atr"] * atr
    stop = entry - side * risk
    left, right = np.searchsorted(times, [entry_at + 1, expiry + 1])
    t, p = times[left:right], quotes[left:right]
    expected = np.arange(entry_at + 1, expiry + 1, dtype=np.int64)
    missing = expected[~np.isin(expected, t, assume_unique=True)]
    gap = int(missing[0]) if len(missing) else None
    trigger = _first(side * (p - stop) <= 0, t)
    if gap is not None and (trigger is None or gap < trigger):
        result.update(status="path_gap", reason="unknown_after_entry", censored=True,
                      missing_time=gap, exit_time=expiry)
    else:
        exit_at = (trigger if trigger is not None else expiry) + 1
        pos = int(np.searchsorted(times, exit_at))
        if trigger is not None:
            result["trigger_time"] = trigger
        if pos >= len(times) or int(times[pos]) != exit_at:
            result.update(status="path_gap", reason="unknown_stop_fill" if trigger is not None else "unknown_after_entry",
                          censored=True, missing_time=exit_at, exit_time=expiry)
        else:
            exit_price = float(quotes[pos])
            gross = side * (exit_price - entry) / risk
            result.update(status="completed", reason="sl" if trigger is not None else "time",
                          exit_time=exit_at, exit=exit_price, gross_R=gross,
                          net_R=gross - cfg["cost_atr"] / cfg["stop_atr"])
    result["holding_minutes"] = (result["exit_time"] - entry_at) / 60
    return result, maximum if result["censored"] else result["exit_time"]


def decode_symbol(audit, sources):
    total = sum(row["rows"] for row in sources)
    times, prices = np.empty(total, dtype=np.int64), np.empty(total, dtype=float)
    cursor = 0
    for source in sources:
        path = audit.pin(source["path"], source["sha256"])
        # NumPy text parsing is independent from the production pandas reader.
        data = np.loadtxt(path, delimiter=",", skiprows=1,
                          dtype=[("epoch", np.int64), ("quote", np.float64)], ndmin=1)
        audit.equal(len(data), source["rows"], f"daily rows {source['path']}")
        n = len(data)
        times[cursor:cursor + n], prices[cursor:cursor + n] = data["epoch"], data["quote"]
        cursor += n
    audit.check(np.all(np.diff(times) > 0), "Native quote chronology")
    audit.check(np.all(np.isfinite(prices) & (prices > 0)), "Native positive finite quotes")
    return times, prices


def interval(samples):
    finite = samples[np.isfinite(samples)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]


def read_events(path):
    """Normalize CSV numeric representations without hiding missing outcomes."""
    frame = pd.read_csv(path, keep_default_na=False)
    for field in ("zone_low", "zone_high", "invalidation", "issue_price", "atr", "entry",
                  "exit", "gross_R", "net_R", "holding_minutes"):
        values = frame[field].map(lambda value: np.nan if value == "" else value)
        frame[field] = pd.to_numeric(values, errors="raise").astype(float)
    frame["side"] = pd.to_numeric(frame.side, errors="raise").astype(np.int64)
    for field in ("censored", "ambiguous"):
        if len(frame) and not frame[field].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("Unknown/nonboolean saved path flag")
        frame[field] = frame[field].astype(bool)
    for field in TIME_FIELDS:
        frame[field] = frame[field].astype(str)
    for field in ("variant", "zone_id", "status", "reason"):
        frame[field] = frame[field].astype(str)
    return frame


def independent_pf_intervals(values, entry_days, start, end, repeats, seed):
    first = pd.Timestamp(start, unit="s", tz="UTC").floor("D")
    last = pd.Timestamp(end - 1, unit="s", tz="UTC").floor("D")
    n = (last - first).days + 1
    ids = (entry_days - first.value // 10**9) // 86400
    gains = np.bincount(ids, weights=np.maximum(values, 0), minlength=n)
    losses = np.bincount(ids, weights=np.maximum(-values, 0), minlength=n)
    output = {}
    for block, label in ((1, "day_profit_factor_ci95"), (7, "weekly_profit_factor_ci95")):
        block = min(block, n)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n, size=(repeats, math.ceil(n / block)))
        selected = ((starts[:, :, None] + np.arange(block)) % n).reshape(repeats, -1)[:, :n]
        gs, ls = gains[selected].sum(axis=1), losses[selected].sum(axis=1)
        samples = np.divide(gs, ls, out=np.full(repeats, np.nan), where=ls > 0)
        output[label] = interval(samples)
    return output


def metric_checks(audit, row, ledger, cfg, repeats, seed):
    values = pd.to_numeric(ledger.net_R, errors="raise").to_numpy(float)
    done = ledger.loc[~ledger.censored & np.isfinite(values)].copy()
    net = pd.to_numeric(done.net_R).to_numpy(float)
    gross = pd.to_numeric(done.gross_R).to_numpy(float)
    m = row["metrics"]
    prefix = f"{row['symbol']}/{row['variant']}/{row['partition']}"
    audit.equal(m["trades"], len(ledger), prefix + " trades")
    audit.equal(m["completed"], len(done), prefix + " completed")
    audit.equal(m["censored"], int(ledger.censored.sum()), prefix + " censored")
    audit.equal(m["sum_net_R"], float(math.fsum(net)), prefix + " sum")
    audit.equal(m["mean_net_R"], float(net.mean()) if len(net) else None, prefix + " mean")
    audit.equal(m["mean_gross_R"], float(gross.mean()) if len(gross) else None, prefix + " gross mean")
    gain, loss = math.fsum(net[net > 0]), -math.fsum(net[net < 0])
    audit.equal(m["profit_factor"], gain / loss if loss > 0 else None, prefix + " PF")
    audit.equal(m["active_days"], done.entry_time.str[:10].nunique(), prefix + " active days")
    audit.equal(m["wins"], int((net > 0).sum()), prefix + " wins")
    audit.equal(m["losses"], int((net < 0).sum()), prefix + " losses")
    start, end = epoch(row["start"]), epoch(row["end_exclusive"])
    days = np.array([epoch(t) // 86400 * 86400 for t in done.entry_time], dtype=np.int64)
    calculated = independent_pf_intervals(net, days, start, end, repeats, seed)
    for field, expected in calculated.items():
        for actual, target in zip(row["profit_factor_inference"][field], expected, strict=True):
            audit.equal(actual, target, prefix + " " + field)
    doubled = net - cfg["cost_atr"] / cfg["stop_atr"]
    audit.equal(row["double_cost_metrics"]["mean_net_R"], float(doubled.mean()) if len(net) else None,
                prefix + " doubled cost mean")
    audit.check(row["QUALIFIED"] is False, prefix + " unqualified")


def verify(folder, audit):
    declaration_path = audit.pin(folder / "declaration.json")
    declaration_sha = sha(declaration_path)
    audit.check((declaration_path.parent / "declaration.sha256").read_text().strip() == declaration_sha, "Declaration SHA sidecar")
    declaration = json.loads(declaration_path.read_text())
    results_path = audit.pin(folder / "results.json")
    results = json.loads(results_path.read_text())
    audit.equal(results["declaration_sha256"], declaration_sha, "Result declaration")
    audit.check((results_path.parent / "results.sha256").read_text().strip() == sha(results_path), "Result SHA sidecar")
    for name, fingerprint in declaration["code_sha256"].items():
        audit.pin(name, fingerprint)
    audit.pin(declaration["sources"]["audit_path"], declaration["sources"]["audit_sha256"])
    for obj in (declaration, results):
        for flag in FLAGS:
            audit.check(obj["safety"].get(flag) is False, f"Disabled {flag}")
        audit.check(obj["QUALIFIED"] is False, "No historical promotion")
    protocol = declaration["protocol"]
    cfg = protocol["config"]
    expected = {(s, v, p) for s in SYMBOLS for v in VARIANTS for p in
                (*protocol["partitions"], "walk_forward_combined")}
    rows = results["rows"]
    audit.check(len(rows) == len(expected) and {(r["symbol"], r["variant"], r["partition"]) for r in rows} == expected,
                "Exact56 unique result cells")
    event_count = filled_count = 0
    for symbol in SYMBOLS:
        times, quotes = decode_symbol(audit, declaration["sources"]["symbols"][symbol])
        candidate_artifact = results["symbols"][symbol]["candidate_artifact"]
        candidates = pd.read_csv(audit.pin(candidate_artifact["path"], candidate_artifact["sha256"]),
                                 keep_default_na=False, low_memory=False)
        audit.check(not candidates.duplicated(["signal_time", "variant"]).any(), "Unique scheduled candidate rows")
        ledgers = {}
        for row in (r for r in rows if r["symbol"] == symbol and r["partition"] != "walk_forward_combined"):
            prefix = f"{symbol}/{row['variant']}/{row['partition']}"
            start, end = epoch(row["start"]), epoch(row["end_exclusive"])
            audit.equal([row["start"], row["end_exclusive"]], protocol["partitions"][row["partition"]], prefix + " bounds")
            artifacts = row["artifacts"]
            regions = json.loads(audit.pin(artifacts["zones"]["path"], artifacts["zones"]["sha256"]).read_text())
            events = read_events(audit.pin(artifacts["events"]["path"], artifacts["events"]["sha256"]))
            saved_ledger = read_events(audit.pin(artifacts["ledger"]["path"], artifacts["ledger"]["sha256"]))
            selected = candidates.loc[candidates.eligible & candidates.variant.eq(row["variant"])]
            selected = selected.loc[selected.signal_time.map(epoch).ge(start) & selected.signal_time.map(epoch).lt(end)]
            audit.equal([z["zone_id"] for z in regions], selected.zone_id.tolist(), prefix + " issued population")
            selected = selected.set_index("zone_id")
            audit.equal(len(events), len(regions), prefix + " event population")
            busy, filled_ids = start, []
            bounded = (times >= start) & (times < end)
            t, p = times[bounded], quotes[bounded]
            for z, event in zip(regions, events.to_dict("records"), strict=True):
                cid = z["zone_id"]
                candidate = selected.loc[cid]
                for field in ("zone_low", "zone_high", "invalidation", "atr", "issue_price", "side"):
                    audit.equal(z[field], float(candidate[field]), prefix + " fixed source " + field)
                    audit.equal(float(event[field]), float(z[field]), prefix + " event source " + field)
                audit.equal(event["zone_id"], cid, prefix + " identity")
                actual, busy = independent_path(t, p, z, cfg, start, end, busy)
                for field, target in actual.items():
                    value = event[field]
                    if field in TIME_FIELDS:
                        value = epoch(value)
                    elif field in ("entry", "exit", "gross_R", "net_R", "holding_minutes"):
                        value = float(value) if not pd.isna(value) else None
                    audit.equal(value, target, prefix + ":" + cid + ":" + field)
                if actual["entry_time"] is not None:
                    filled_ids.append(cid)
                event_count += 1
            audit.equal(saved_ledger.zone_id.tolist(), filled_ids, prefix + " filled ledger identity")
            expected_ledger = events.loc[events.entry_time.ne("")].reset_index(drop=True)
            audit.check(saved_ledger.equals(expected_ledger), prefix + " filled ledger equals events")
            audit.equal(row["replay_audit"]["statuses"], dict(Counter(events.status)), prefix + " dispositions")
            audit.equal(row["replay_audit"]["unknown"], int(events.censored.sum()), prefix + " all unknown regions")
            audit.equal(row["replay_audit"]["filled"], len(filled_ids), prefix + " fills")
            filled_count += len(filled_ids)
            metric_checks(audit, row, saved_ledger, cfg, protocol["bootstrap_repeats"], protocol["bootstrap_seed"])
            ledgers[(row["variant"], row["partition"])] = saved_ledger
        for row in (r for r in rows if r["symbol"] == symbol and r["partition"] == "walk_forward_combined"):
            artifact = row["artifacts"]["ledger"]
            ledger = read_events(audit.pin(artifact["path"], artifact["sha256"]))
            combined = pd.concat([ledgers[(row["variant"], f)] for f in ("wf1", "wf2", "wf3")], ignore_index=True)
            audit.check(ledger.equals(combined), "Nonoverlapping walk-forward ledger union")
            metric_checks(audit, row, ledger, cfg, protocol["bootstrap_repeats"], protocol["bootstrap_seed"])
        del times, quotes
    # Pin consistency is checked again after use, without reinterpreting outcomes.
    for name, fingerprint in tuple(audit.pins.items()):
        audit.equal(sha(ROOT / name), fingerprint, "Final stable input " + name)
    return {"events_checked": event_count, "filled_paths_checked": filled_count,
            "cells_checked": len(rows), "scope": "saved_regions_actual_quotes_dispositions_R_PF_day_week_PF_CI",
            "not_independently_recomputed": ["zone_feature_generation", "paired_mean_inference", "qualification_logic"],
            "fresh_out_of_sample": False, "broker_execution": "NOT TESTED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="docs/zone_study_20261008")
    parser.add_argument("--output", default="docs/zone_study_20261008/independent_economic_audit.json")
    args = parser.parse_args()
    folder = Path(args.folder)
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT.resolve())
    if output.exists() or output.with_suffix(".sha256").exists():
        raise ValueError("Audit output is exclusive")
    audit, detail = Audit(), {}
    try:
        detail = verify(folder, audit)
    except Exception as error:
        audit.errors.append(f"{type(error).__name__}: {error}")
    report = {"stage": "independent_zone_quote_path_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
              "passed": not audit.errors, "checks": audit.checks, "errors": audit.errors,
              "input_sha256": audit.pins, "verifier_sha256": sha(Path(__file__)),
              "safety": {flag: False for flag in FLAGS}, "QUALIFIED": False, **detail}
    with output.open("x") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with output.with_suffix(".sha256").open("x") as target:
        target.write(sha(output) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
