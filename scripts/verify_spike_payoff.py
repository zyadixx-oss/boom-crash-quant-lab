#!/usr/bin/env python3
"""Independent, standard-library-only audit of frozen external payoff ledgers.

This reads saved public M1 data and research outputs. It neither imports the
study engine nor connects to any API. Signal eligibility/completeness and
bootstrap intervals are outside this ledger audit's scope.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
COHORT = "external_older_replication"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def epoch(text: str) -> int:
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None or stamp.timestamp() % 60:
        raise ValueError(f"Expected a timezone-aware minute: {text}")
    return int(stamp.timestamp())


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.safety_values_checked = 0
        self.max_numeric_error = {}

    def equal(self, label, actual, expected):
        self.checks += 1
        if isinstance(actual, float) and isinstance(expected, (float, int)):
            passed = math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9)
            field = label.rsplit("/", 1)[-1]
            self.max_numeric_error[field] = max(
                self.max_numeric_error.get(field, 0), abs(actual - expected))
        else:
            passed = actual == expected
        if not passed:
            self.failures.append({"check": label, "actual": actual, "expected": expected})

    def safety(self, value, label):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FLAGS:
                    self.safety_values_checked += 1
                    self.equal(f"{label}/{key}", item is False, True)
                self.safety(item, f"{label}/{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                self.safety(item, f"{label}/{i}")


def read_minutes(path: Path):
    minutes = {}
    previous = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            t = int(row["epoch"])
            values = tuple(float(row[key]) for key in ("open", "high", "low", "close"))
            o, h, low, c = values
            if t % 60 or (previous is not None and t != previous + 60):
                raise ValueError(f"Noncontiguous/duplicate/off-grid external M1 at {t}")
            if not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError(f"Invalid M1 price at {t}")
            if not low <= min(o, c) <= max(o, c) <= h:
                raise ValueError(f"Invalid M1 bounds at {t}")
            minutes[t] = values
            previous = t
    return minutes


def causal_atr(minutes, signal):
    """Direct 14-bar arithmetic mean of M5 true ranges before signal close."""
    ranges = []
    for end in range(signal - 13 * 300, signal + 1, 300):
        start = end - 300
        bars = [minutes[t] for t in range(start, end, 60)]
        previous_close = minutes[start - 60][3]
        high, low = max(b[1] for b in bars), min(b[2] for b in bars)
        ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return math.fsum(ranges) / 14


def scalar_trade(minutes, signal, atr, side, config):
    """Chronological scalar walk using the primary adverse-extreme convention."""
    if config["fill_mode"] != "adverse_extreme" or config["same_bar_policy"] != "stop_first":
        raise ValueError("This verifier audits the primary fill convention only")
    if config["entry_delay_minutes"] != 1:
        raise ValueError("Primary entry delay must remain one minute")
    entry_time = signal + 60
    opening = minutes[entry_time][0]
    risk = atr * config["stop_atr"]
    stop = opening - side * risk
    target = opening + side * atr * config["target_atr"]
    planned_end = entry_time + 60 * config["max_hold_minutes"]
    ambiguous = False
    for t in range(entry_time, planned_end, 60):
        o, h, low, c = minutes[t]
        at_target = o >= target if side == 1 else o <= target
        at_stop = o <= stop if side == 1 else o >= stop
        if at_target:
            price, exit_time, reason = target, t, "tp"
            break
        reaches_stop = low <= stop if side == 1 else h >= stop
        reaches_target = h >= target if side == 1 else low <= target
        if at_stop or reaches_stop:
            ambiguous = reaches_target and not at_stop
            price, exit_time, reason = (low if side == 1 else h), t + 60, "sl"
            break
        if reaches_target:
            price, exit_time, reason = target, t + 60, "tp"
            break
    else:
        price, exit_time, reason = c, planned_end, "time"
    gross = side * (price - opening) / risk
    return {"signal_time": signal, "entry_time": entry_time, "exit_time": exit_time,
            "entry": opening, "exit": price, "atr": atr, "gross_R": gross,
            "net_R": gross - config["round_trip_cost_atr"] / config["stop_atr"],
            "reason": reason, "ambiguous": ambiguous,
            "holding_minutes": (exit_time - entry_time) / 60,
            "censored": False, "planned_end": planned_end, "missing_time": ""}


def summary(rows, start, end, stop_atr, risk_fraction):
    values = [r["net_R"] for r in rows]
    gross = [r["gross_R"] for r in rows]
    winners = [r for r in values if r > 0]
    losers = [r for r in values if r < 0]
    days = list(range(start // 86400, (end - 1) // 86400 + 1))
    by_day = {day: [] for day in days}
    for row in rows:
        by_day[row["entry_time"] // 86400].append(row["net_R"])
    mean = statistics.fmean(values) if values else None
    se = (math.sqrt(len(days) / (len(days) - 1) * math.fsum(
        (math.fsum(v) - mean * len(v)) ** 2 for v in by_day.values())) / len(values)
        if values and len(days) > 1 else None)
    equity = peak = 1.0
    dd = 0.0
    ruined = False
    for row in sorted(rows, key=lambda r: (r["exit_time"], r["entry_time"])):
        factor = 1 + risk_fraction * row["net_R"]
        ruined = ruined or factor <= 0
        equity = 0.0 if ruined else equity * factor
        peak = max(peak, equity)
        dd = max(dd, 1 - equity / peak)
    mean_gross = statistics.fmean(gross) if gross else None
    return {"trades": len(rows), "completed": len(rows), "censored": 0,
            "invalid_uncensored": 0, "wins": len(winners), "losses": len(losers),
            "flat_trades": len(values) - len(winners) - len(losers),
            "win_rate": len(winners) / len(rows) if rows else None,
            "mean_net_R": mean, "median_net_R": statistics.median(values) if rows else None,
            "sum_net_R": math.fsum(values), "mean_gross_R": mean_gross,
            "profit_factor": math.fsum(winners) / -math.fsum(losers) if losers else None,
            "average_win_R": statistics.fmean(winners) if winners else None,
            "average_loss_R": statistics.fmean(losers) if losers else None,
            "active_days": sum(bool(v) for v in by_day.values()), "calendar_days": len(days),
            "median_holding_minutes": statistics.median(r["holding_minutes"] for r in rows) if rows else None,
            "ambiguous": sum(r["ambiguous"] for r in rows),
            "break_even_cost_atr": mean_gross * stop_atr if mean_gross is not None else None,
            "cluster_se": se, "selection_score": mean - 1.96 * se if se is not None else None,
            "closed_trade_return": equity - 1, "closed_trade_max_drawdown": dd,
            "equity_ruin": ruined, "illustrative_risk_fraction": risk_fraction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_payoff_20261004")
    args = parser.parse_args()
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() != "false":
            raise SystemExit(f"Offline audit refuses {flag} other than false")
    audit = Audit()
    results = json.loads((args.study / "results.json").read_text())
    selection = json.loads((args.study / "selection.json").read_text())
    audit.safety(results, "results")
    audit.safety(selection, "selection")
    audit.equal("safety/top_level", results["safety"], dict.fromkeys(FLAGS, False))
    selection_hash = sha256(args.study / "selection.json")
    audit.equal("hash/selection", selection_hash, results["selection_sha256"])
    audit.equal("hash/selection_file", selection_hash, (args.study / "selection.sha256").read_text().strip())
    for path, expected in results["code_hashes"].items():
        audit.equal(f"hash/{path}", sha256(ROOT / path), expected)
    ledgers = []
    for symbol, payload in results["symbols"].items():
        cohort = payload["cohorts"][COHORT]
        start, end = epoch(cohort["start"]), epoch(cohort["end"])
        data = ROOT / "data/spike_payoff_external" / cohort["audit"]["source_file"]
        raw = data.with_name(data.name.replace("_clean", ""))
        audit.equal(f"{symbol}/input_sha256", sha256(data), cohort["audit"]["sha256"])
        audit.equal(f"{symbol}/raw_equals_clean_sha256", sha256(raw), sha256(data))
        minutes = read_minutes(raw)
        audit.equal(f"{symbol}/M1_count", len(minutes), cohort["audit"]["source_rows"])
        audit.equal(f"{symbol}/M1_start", min(minutes), start)
        audit.equal(f"{symbol}/M1_end", max(minutes) + 60, end)
        for role, model in cohort["models"].items():
            path = args.study / f"{symbol}_{COHORT}_{role}_trades.csv"
            config = model["config"]
            label = f"{symbol}/{role}"
            reconstructed = []
            previous_signal = previous_exit = None
            failure_count = len(audit.failures)
            with path.open(newline="") as stream:
                for i, saved in enumerate(csv.DictReader(stream)):
                    signal = epoch(saved["signal_time"])
                    atr = causal_atr(minutes, signal)
                    calculated = scalar_trade(minutes, signal, atr, 1 if symbol.startswith("BOOM") else -1, config)
                    audit.equal(f"{label}/{i}/signal_on_M5_close", signal % 300, 0)
                    audit.equal(f"{label}/{i}/inside_partition", start <= signal < end and signal + 31 * 60 <= end, True)
                    audit.equal(f"{label}/{i}/variant", saved["variant"], model["variant"])
                    if previous_signal is not None:
                        audit.equal(f"{label}/{i}/cooldown_at_least_30m", signal - previous_signal >= 1800, True)
                        audit.equal(f"{label}/{i}/no_overlap", calculated["entry_time"] >= previous_exit, True)
                    for key, value in calculated.items():
                        if key.endswith("time") and key != "missing_time" or key == "planned_end":
                            observed = epoch(saved[key])
                        elif isinstance(value, bool):
                            if saved[key] not in ("True", "False"):
                                raise ValueError(f"Invalid boolean at {label}/{i}/{key}")
                            observed = saved[key] == "True"
                        elif isinstance(value, (int, float)):
                            observed = float(saved[key])
                        else:
                            observed = saved[key]
                        audit.equal(f"{label}/{i}/{key}", observed, value)
                    reconstructed.append(calculated)
                    previous_signal, previous_exit = signal, calculated["exit_time"]
            independent = summary(reconstructed, start, end, config["stop_atr"], model["metrics"]["illustrative_risk_fraction"])
            for key, value in independent.items():
                audit.equal(f"{label}/metrics/{key}", model["metrics"][key], value)
            for key in ("filled", "completed"):
                audit.equal(f"{label}/audit/{key}", model["audit"][key], len(reconstructed))
            audit.equal(f"{label}/audit/ambiguous", model["audit"]["ambiguous"], independent["ambiguous"])
            audit.equal(f"{label}/audit/censored", model["audit"]["censored"], 0)
            ledgers.append({"symbol": symbol, "role": role, "id": model["id"],
                            "trades_verified": len(reconstructed), "ledger_sha256": sha256(path),
                            "raw_data_sha256": sha256(raw), "mean_net_R": independent["mean_net_R"],
                            "profit_factor": independent["profit_factor"],
                            "mismatches": len(audit.failures) - failure_count})
    report = {"status": "PASS" if not audit.failures else "FAIL",
              "run_utc": datetime.now(timezone.utc).isoformat(), "safety": dict.fromkeys(FLAGS, False),
              "implementation": "independent scalar Python standard library; no study engine or metrics imports",
              "script_sha256": sha256(Path(__file__)), "results_sha256": sha256(args.study / "results.json"),
              "selection_sha256": selection_hash,
              "scope": ["all four external primary and selected ledgers", "raw and clean M1 hash equivalence",
                        "causal ATR14 independently aggregated from closed M5 bars", "exact signal+1-minute entry",
                        "stop/target/timeout chronology with adverse-extreme stop-first fills",
                        "entry/exit prices and times, gross/net R, ambiguity and holding time",
                        "30-minute signal cooldown, no overlapping positions, 31-minute end purge",
                        "summary counts, means, PF, UTC-day cluster SE and illustrative closed-trade equity",
                        "frozen source/selection hashes and all saved safety flags"],
              "not_verified": ["signal eligibility and completeness independently regenerated",
                               "clock baseline replay", "bootstrap confidence intervals and p-values",
                               "sensitivity ledgers", "actual broker execution or monetary profits"],
              "checks": audit.checks, "saved_safety_values_checked": audit.safety_values_checked,
              "total_trades_verified": sum(row["trades_verified"] for row in ledgers),
              "ledgers": ledgers, "max_absolute_numeric_differences": audit.max_numeric_error,
              "mismatch_count": len(audit.failures), "mismatches": audit.failures}
    output = args.study / "independent_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "trades": report["total_trades_verified"],
                      "checks": audit.checks, "mismatches": len(audit.failures), "audit": str(output)}))
    raise SystemExit(bool(audit.failures))


if __name__ == "__main__":
    main()
