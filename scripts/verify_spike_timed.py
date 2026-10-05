#!/usr/bin/env python3
"""Independent scalar audit of transferred uncapped trades; public saved data only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_spike_payoff import Audit, FLAGS, causal_atr, epoch, read_minutes, sha256, summary


def scalar(minutes, signal, atr, side, config):
    if config["fill_mode"] != "adverse_extreme" or config["entry_delay_minutes"] != 1:
        raise ValueError("Verifier covers primary one-minute/extreme execution only")
    entry_time = signal + 60
    entry = minutes[entry_time][0]
    risk = atr * config["stop_atr"]
    stop = entry - side * risk
    planned_end = entry_time + config["max_hold_minutes"] * 60
    for t in range(entry_time, planned_end, 60):
        opening, high, low, close = minutes[t]
        at_stop = opening <= stop if side == 1 else opening >= stop
        reaches_stop = low <= stop if side == 1 else high >= stop
        if at_stop or reaches_stop:
            price, exit_time, reason = (low if side == 1 else high), t + 60, "sl"
            break
    else:
        price, exit_time, reason = close, planned_end, "time"
    gross = side * (price - entry) / risk
    return {"signal_time": signal, "entry_time": entry_time, "exit_time": exit_time,
            "entry": entry, "exit": price, "atr": atr, "gross_R": gross,
            "net_R": gross - config["round_trip_cost_atr"] / config["stop_atr"],
            "reason": reason, "ambiguous": False, "censored": False,
            "holding_minutes": (exit_time - entry_time) / 60,
            "planned_end": planned_end, "missing_time": ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "docs/spike_timed_20261005")
    args = parser.parse_args()
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() != "false":
            raise SystemExit(f"Requires {flag}=false")
    audit = Audit()
    result = json.loads((args.study / "results.json").read_text())
    audit.safety(result, "results")
    audit.equal("selection_hash", sha256(args.study / "selection.json"), result["selection_sha256"])
    audit.equal("saved_selection_hash", (args.study / "selection.sha256").read_text().strip(), result["selection_sha256"])
    for path, expected in result["code_hashes"].items():
        audit.equal(f"source/{path}", sha256(ROOT / path), expected)
    ledgers = []
    for training_symbol, payload in result["symbols"].items():
        cohort = payload["cohorts"]["cross_symbol_replication"]
        symbol = cohort["symbol"]
        clean = ROOT / "data/spike_timed_transfer" / cohort["audit"]["source_file"]
        raw = clean.with_name(clean.name.replace("_clean", ""))
        audit.equal(f"{symbol}/clean_hash", sha256(clean), cohort["audit"]["sha256"])
        audit.equal(f"{symbol}/raw_clean_hash", sha256(raw), sha256(clean))
        minutes = read_minutes(raw)
        start, end = epoch(cohort["start"]), epoch(cohort["end"])
        audit.equal(f"{symbol}/rows", len(minutes), cohort["audit"]["source_rows"])
        audit.equal(f"{symbol}/start", min(minutes), start)
        audit.equal(f"{symbol}/end", max(minutes) + 60, end)
        seen = {}
        for role, model in cohort["models"].items():
            if model is None:
                continue
            actual_role = seen.setdefault(model["id"], role)
            path = args.study / f"{symbol}_cross_symbol_replication_{actual_role}_trades.csv"
            side = (1 if symbol.startswith("BOOM") else -1) * (1 if model["mode"] == "SPIKE" else -1)
            reconstructed = []
            previous_signal = previous_exit = None
            failures_before = len(audit.failures)
            label = f"{symbol}/{role}"
            with path.open(newline="") as stream:
                for ordinal, saved in enumerate(csv.DictReader(stream)):
                    signal = epoch(saved["signal_time"])
                    calculated = scalar(minutes, signal, causal_atr(minutes, signal), side, model["config"])
                    audit.equal(f"{label}/{ordinal}/M5_close", signal % 300, 0)
                    audit.equal(f"{label}/{ordinal}/partition", start <= signal and signal + 31 * 60 <= end, True)
                    audit.equal(f"{label}/{ordinal}/variant", saved["variant"], f"{model['mode']}::{model['variant']}")
                    if previous_signal is not None:
                        audit.equal(f"{label}/{ordinal}/cooldown", signal - previous_signal >= 1800, True)
                        audit.equal(f"{label}/{ordinal}/no_overlap", calculated["entry_time"] >= previous_exit, True)
                    for key, value in calculated.items():
                        if key in ("signal_time", "entry_time", "exit_time", "planned_end"):
                            observed = epoch(saved[key])
                        elif isinstance(value, bool):
                            if saved[key] not in ("True", "False"):
                                raise ValueError("Invalid saved boolean")
                            observed = saved[key] == "True"
                        elif isinstance(value, (int, float)):
                            observed = float(saved[key])
                        else:
                            observed = saved[key]
                        audit.equal(f"{label}/{ordinal}/{key}", observed, value)
                    reconstructed.append(calculated)
                    previous_signal, previous_exit = signal, calculated["exit_time"]
            independent = summary(reconstructed, start, end, model["config"]["stop_atr"],
                                  model["metrics"]["illustrative_risk_fraction"])
            for field, value in independent.items():
                audit.equal(f"{label}/metrics/{field}", model["metrics"][field], value)
            ledgers.append({"training_symbol": training_symbol, "evaluation_symbol": symbol, "role": role,
                            "id": model["id"], "trades": len(reconstructed), "ledger_sha256": sha256(path),
                            "mean_net_R": independent["mean_net_R"], "profit_factor": independent["profit_factor"],
                            "mismatches": len(audit.failures) - failures_before})
    report = {"status": "PASS" if not audit.failures else "FAIL", "run_utc": datetime.now(timezone.utc).isoformat(),
              "safety": dict.fromkeys(FLAGS, False), "script_sha256": sha256(Path(__file__)),
              "helper_sha256": sha256(ROOT / "scripts/verify_spike_payoff.py"),
              "selection_sha256": result["selection_sha256"], "results_sha256": sha256(args.study / "results.json"),
              "scope": "Independent scalar standard-library replay of all transfer primary/selected ledgers, causal ATR, paths and summary arithmetic; no timed engine or metrics imports",
              "not_verified": ["signal eligibility/completeness", "baseline replay", "bootstrap", "sensitivities", "actual fills"],
              "checks": audit.checks, "total_trades": sum(x["trades"] for x in ledgers), "ledgers": ledgers,
              "saved_false_safety_values": audit.safety_values_checked,
              "mismatch_count": len(audit.failures), "mismatches": audit.failures,
              "max_numeric_differences": audit.max_numeric_error}
    (args.study / "independent_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "checks", "total_trades", "mismatch_count")}))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
