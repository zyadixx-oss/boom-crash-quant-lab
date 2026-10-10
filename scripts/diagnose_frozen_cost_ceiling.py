#!/usr/bin/env python3
"""Verify the nonnegative-cost PF ceiling of twelve already-observed ledgers.

No price sources, research engine, fitting, alternate fills or fresh predictions
are used. The bound concerns fixed paths and weights, not every possible policy.
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
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = "docs/spike_nonlinear_20261005"
COHORT = "later_temporal180"
SYMBOLS = ("BOOM600", "CRASH600")
FAMILIES = ("BOOST44", "RIDGE44", "RIDGE19")
MODES = ("SPIKE", "DRIFT")
SAFETY = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)
RESULT_SHA = "4d00095faaa6b839452bcf61ad3cdd984a9bf1f632c7ece768f83ae52355e180"
AUDIT_SHA = "9da900082a7393c486b0e2e30190e836781d7859f04e9104e3e0d6e74adda19a"
PRIMARY = {"stop_atr": 2.0, "max_hold_minutes": 15, "entry_delay_minutes": 1,
           "round_trip_cost_atr": 0.1, "fill_mode": "adverse_extreme"}
FIELDS = ["signal_time", "entry_time", "exit_time", "entry", "exit", "atr", "gross_R",
          "net_R", "reason", "ambiguous", "holding_minutes", "censored", "variant",
          "planned_end", "missing_time"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected repository-relative input")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Input symlinks are forbidden")
    if not current.is_file():
        raise ValueError("Missing input: " + name)
    return current


def pin(root, name, expected, pins):
    path = safe_file(root, name)
    actual = digest(path)
    if actual != expected:
        raise ValueError("Changed frozen input: " + name)
    pins[name] = actual
    return path


def safety(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SAFETY and item is not False:
                raise ValueError("Saved execution flag is not false: " + key)
            safety(item)
    elif isinstance(value, list):
        for item in value:
            safety(item)


def guard():
    for key in SAFETY:
        if os.environ.get(key, "false").strip().lower() != "false":
            raise ValueError(key + " must remain false")


def number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a return")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Unknown or nonfinite return")
    return result


def summary(values):
    numbers = [number(x) for x in values]
    gains = math.fsum(max(x, 0) for x in numbers)
    losses = math.fsum(max(-x, 0) for x in numbers)
    return {"completed": len(numbers), "gains_R": gains, "losses_R": losses,
            "sum_R": math.fsum(numbers), "mean_R": math.fsum(numbers) / len(numbers) if numbers else None,
            "profit_factor": gains / losses if losses > 0 else None}


def cost_ceiling(gross, costs):
    """For fixed g_i and c_i>=0, gains(g-c)<=gains(g), losses(g-c)>=losses(g)."""
    gross, costs = [number(x) for x in gross], [number(x) for x in costs]
    if len(gross) != len(costs) or any(c < 0 for c in costs):
        raise ValueError("One nonnegative finite additive cost per fixed path is required")
    before, after = summary(gross), summary([g - c for g, c in zip(gross, costs)])
    return {"gross": before, "net": after,
            "finite_zero_cost_ceiling": before["profit_factor"],
            "target_1_5_impossible_by_cost_reduction_alone": (
                before["profit_factor"] is not None and before["profit_factor"] < 1.5)}


def close(actual, expected, name):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError("Null mismatch: " + name)
    elif not math.isclose(number(actual), number(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Numeric mismatch: " + name)


def stamp(value):
    date = datetime.fromisoformat(value)
    if date.utcoffset() is None or date.utcoffset().total_seconds() != 0 or date.microsecond:
        raise ValueError("Whole-second UTC timestamp required")
    return int(date.timestamp())


def read_ledger(path, symbol, family, mode, start, end):
    side = (1 if symbol == "BOOM600" else -1) * (1 if mode == "SPIKE" else -1)
    gross, net, days, identities = [], [], set(), set()
    previous_signal = previous_exit = None
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Wrong or duplicate CSV schema")
        for row in reader:
            if set(row) != set(FIELDS) or any(value is None for value in row.values()):
                raise ValueError("Ragged ledger row")
            signal, entry, exit_, planned = [stamp(row[k]) for k in (
                "signal_time", "entry_time", "exit_time", "planned_end")]
            if (signal in identities or signal % 1800 or not start <= signal < end
                    or signal + 31 * 60 > end or entry != signal + 60
                    or planned != entry + 15 * 60 or not entry <= exit_ <= planned
                    or (previous_signal is not None and signal <= previous_signal)
                    or (previous_exit is not None and entry < previous_exit)):
                raise ValueError("Duplicate, overlap or invalid chronology")
            if (row["censored"] != "False" or row["ambiguous"] != "False" or row["missing_time"]
                    or row["reason"] not in {"sl", "time"}
                    or row["variant"] != f"NONLINEAR_{family}_{mode}"):
                raise ValueError("Unknown, censored or mismatched saved path")
            first, last, atr = [number(row[k]) for k in ("entry", "exit", "atr")]
            if min(first, last, atr) <= 0:
                raise ValueError("Nonpositive price or ATR")
            g, n = number(row["gross_R"]), number(row["net_R"])
            close(g, side * (last - first) / (2 * atr), "saved gross quote arithmetic")
            close(n, g - 0.05, "assumed .10ATR/2ATR cost")
            close(number(row["holding_minutes"]), (exit_ - entry) / 60, "holding duration")
            gross.append(g); net.append(n)
            days.add(datetime.fromtimestamp(entry, timezone.utc).date().isoformat())
            identities.add(signal); previous_signal, previous_exit = signal, exit_
    return gross, net, sorted(days)


def json_document(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate metadata key")
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite metadata")))


def verify(root):
    guard(); pins = {}
    result = json_document(pin(root, STUDY + "/results.json", RESULT_SHA, pins))
    audit = json_document(pin(root, STUDY + "/independent_audit.json", AUDIT_SHA, pins))
    if not (audit["status"] == "PASS" and audit["pass"] is True and audit["error_count"] == 0
            and audit["errors"] == [] and audit["results_sha256"] == RESULT_SHA):
        raise ValueError("Prior primary-path audit must have passed")
    for name in ("declaration", "selection"):
        document = json_document(pin(root, f"{STUDY}/{name}.json", result[name + "_sha256"], pins))
        safety(document)
        if document["code_hashes"] != result["code_hashes"]:
            raise ValueError("Science lineage mismatch")
    if audit["selection_sha256"] != result["selection_sha256"]:
        raise ValueError("Selection audit mismatch")
    for name, expected in {**result["code_hashes"], **audit["helper_sha256"],
                           "scripts/verify_spike_nonlinear.py": audit["script_sha256"]}.items():
        pin(root, name, expected, pins)
    safety(result); safety(audit)
    expected_keys = {(s, f, m) for s in SYMBOLS for f in FAMILIES for m in MODES}
    audited = {}
    for record in audit["ledgers"]:
        key = (record["symbol"], record["family"], record["mode"])
        if key in audited or record["cohort"] != COHORT:
            raise ValueError("Duplicated or wrong audited cohort")
        audited[key] = record
    if set(audited) != expected_keys or set(result["symbols"]) != set(SYMBOLS):
        raise ValueError("The exact twelve fixed models are required")
    rows = []
    for symbol in SYMBOLS:
        cohort = result["symbols"][symbol]["cohorts"][COHORT]
        if set(cohort["models"]) != {f"{f}_{m}" for f in FAMILIES for m in MODES}:
            raise ValueError("Saved model family changed")
        for family in FAMILIES:
            for mode in MODES:
                model = cohort["models"][f"{family}_{mode}"]
                prior = audited[symbol, family, mode]
                if model["config"] != PRIMARY or model["audit"]["config"] != PRIMARY:
                    raise ValueError("Primary execution changed")
                for key in ("censored", "missing_entry", "outside_partition", "purged", "overlap_skipped", "ambiguous"):
                    if type(model["audit"][key]) is not int or model["audit"][key] != 0:
                        raise ValueError("Incomplete or selected primary cohort")
                for key in ("censored", "purged", "missing_entries", "mismatches"):
                    if prior[key] != 0:
                        raise ValueError("Prior path audit contains unknowns")
                prefix = f"{STUDY}/{symbol}_{COHORT}_{family}_{mode}"
                path = pin(root, prefix + "_trades.csv", prior["ledger_sha256"], pins)
                pin(root, prefix + "_signals.csv", prior["signals_sha256"], pins)
                gross, net, days = read_ledger(path, symbol, family, mode, stamp(cohort["start"]), stamp(cohort["end"]))
                bound = cost_ceiling(gross, [0.05] * len(gross))
                stored = model["metrics"]
                zeros = [x for x in model["sensitivities"] if x["fill_mode"] == PRIMARY["fill_mode"]
                         and x["entry_delay_minutes"] == 1 and x["round_trip_cost_atr"] == 0]
                if len(zeros) != 1:
                    raise ValueError("Exactly one matching zero-cost sensitivity required")
                for metric, saved in ((bound["gross"], zeros[0]), (summary(net), stored)):
                    for a, b in (("completed", "completed"), ("mean_R", "mean_net_R"),
                                 ("sum_R", "sum_net_R"), ("profit_factor", "profit_factor")):
                        close(metric[a], saved[b], a)
                    close(len(days), saved["active_days"], "active days")
                for n in (prior["trades"], prior["completed"], prior["issued_signals"],
                          model["audit"]["issued"], model["audit"]["filled"], model["audit"]["completed"]):
                    if type(n) is not int or n != len(gross):
                        raise ValueError("Missing or omitted primary paths")
                close(summary(net)["profit_factor"], prior["profit_factor"], "independent audit PF")
                close(summary(net)["mean_R"], prior["mean_net_R"], "independent audit mean")
                rows.append({"symbol": symbol, "family": family, "mode": mode, "active_days": len(days),
                             "is_combined44": family != "RIDGE19", "bound": bound,
                             "completed_sample_gate": len(gross) >= 1000,
                             "active_day_gate": len(days) >= 60,
                             "development_eligible": model["development_eligible"],
                             "historical_candidate": False, "live_candidate": False})
    for name, expected in pins.items():
        pin(root, name, expected, {})
    return {"status": "PASS", "rows": rows, "input_sha256": pins,
            "total_primary_records": sum(r["bound"]["gross"]["completed"] for r in rows),
            "all_fixed_models_below_target_even_without_cost": all(
                r["bound"]["target_1_5_impossible_by_cost_reduction_alone"] for r in rows),
            "pooled_strategy_sample_claim": False}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); guard()
    args.output.mkdir(parents=True, exist_ok=False)
    files = ["scripts/diagnose_frozen_cost_ceiling.py", "backend/tests/test_frozen_cost_ceiling.py",
             "docs/FROZEN_COST_CEILING_DIAGNOSTIC.md"]
    declaration = {"started_utc": datetime.now(timezone.utc).isoformat(), "safety": SAFETY,
                   "source_sha256": {n: digest(safe_file(ROOT, n)) for n in files},
                   "python_version": sys.version, "prior_cost_sensitivities_already_seen": True,
                   "ledger_schema_and_example_rows_already_seen": True,
                   "fresh_price_or_label_holdout": False, "new_paths_or_models": False,
                   "scope": "fixed audited ledgers and additive nonnegative costs only"}
    save(args.output / "declaration.json", declaration)
    try:
        result = verify(ROOT)
        for name, expected in declaration["source_sha256"].items():
            pin(ROOT, name, expected, {})
    except Exception as exc:
        result = {"status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)}}
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(), safety=SAFETY,
                  declaration_sha256=digest(args.output / "declaration.json"), goal_achieved=False,
                  historical_candidate=False, actual_money_profit="NOT TESTED",
                  prospective_paper="NOT TESTED", executable_CFD_costs="NOT TESTED",
                  limitation="No inference about altered fills/signals/weights, rebates, positive carry or other policies")
    save(args.output / "result.json", result)
    (args.output / "result.sha256").write_text(digest(args.output / "result.json") + "  result.json\n")
    print(json.dumps({k: result[k] for k in ("status", "goal_achieved", "safety")}, sort_keys=True))
    if result["status"] == "FAIL":
        print(json.dumps(result["error"]))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
