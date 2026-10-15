"""Independent stdlib audit of the sixteen saved generation-2 WF1 cases.

This does not import a study/execution module, load raw price tapes, inspect a
holdout, or access MT5. Decimal checks public price arithmetic; exact math.fsum
checks saved ledger-to-day/stage aggregation. It is not a strategy replay.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DECL_SHA = "a3f9b73838b63abfcb0e85cc0fce076c5e5ae7c31fd8545ad05038f208dbe0e6"
DAY = 86_400_000
EMBARGO = 1_800_000
SCALE = 100_000_000
START = int(datetime(2025, 2, 1, tzinfo=timezone.utc).timestamp() * 1000)
END = START + 20 * DAY
SAFETY = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES",
          "live_trading", "ready_for_live", "live_allowed", "opened_trades")
SYMBOLS = {"BOOM500": 1, "CRASH500": -1}
POLICIES = ("CLOCK", "CRT_SWEEP_RECLAIM", "PULLBACK_CONTINUATION", "SQUEEZE_BREAKOUT")
CONFIGS = {"BASE": (250, Decimal("0.25")), "STRESS": (2000, Decimal("0.5"))}
FIELDS = ("observed_sum_R", "observed_sum_quote", "observed_gain_R", "observed_loss_R")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def decimal(value):
    require(finite(value), "Expected finite non-boolean numeric value")
    return Decimal(str(value))


def exact(actual, expected, label):
    require(type(actual) is type(expected) or (finite(actual) and finite(expected)),
            label + ": incompatible value types")
    require(actual == expected, label + ": exact comparison failed")


def arithmetic(actual, expected, label):
    require(finite(actual), label + ": nonfinite or missing value")
    error = abs(actual - float(expected))
    require(math.isclose(actual, float(expected), rel_tol=8e-15, abs_tol=8e-15),
            label + ": arithmetic mismatch")
    return error


def safety(obj, label):
    require(isinstance(obj, dict) and all(obj.get(key) is False for key in SAFETY),
            label + ": all eight safety flags must be literal false")


def safety_recursive(value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SAFETY:
                require(item is False, label + ": unsafe flag " + key)
            safety_recursive(item, label)
    elif isinstance(value, list):
        for item in value:
            safety_recursive(item, label)


def components(rows):
    return {
        "observed_sum_R": math.fsum(row["R"] for row in rows),
        "observed_sum_quote": math.fsum(row["signed_move_after_spread_and_slippage"] for row in rows),
        "observed_gain_R": math.fsum(max(0.0, row["R"]) for row in rows),
        "observed_loss_R": math.fsum(max(0.0, -row["R"]) for row in rows),
    }


def profit_factor(totals):
    gain, loss = totals["observed_gain_R"], totals["observed_loss_R"]
    return gain / loss if loss > 0 else None


def check_bundle(bundle, stamp, label):
    require(isinstance(bundle, list) and bundle, label + ": missing saved bundle")
    for quote in bundle:
        require(type(quote.get("time_msc")) is int and quote["time_msc"] == stamp,
                label + ": inconsistent timestamp")
        require(finite(quote.get("bid")) and finite(quote.get("ask")) and
                0 < quote["bid"] <= quote["ask"], label + ": invalid saved quote")
        for field in ("bid", "ask"):
            scaled = decimal(quote[field]) * SCALE
            require(scaled == scaled.to_integral_value(), label + ": quote is not 1e-8 quantized")


def audit_case(case, rows, name, unit):
    symbol, _, tail = name.partition("_WF1_")
    policy, _, scenario = tail.rpartition("_")
    direction = SYMBOLS[symbol]
    baseline = policy == "CLOCK"
    identity = "CLOCK_00_30" if baseline else policy
    require(case["symbol"] == symbol and case["stage"] == "WF1" and
            case["scenario"] == scenario and case["policy"] == policy,
            name + ": filename identity differs")
    exact(case["policy_id"], identity, name + "/policy_id")
    exact(case["baseline"], baseline, name + "/baseline")
    exact(case["declaration_sha256"], DECL_SHA, name + "/declaration")
    for field, value in (("stage_start_ms", START), ("stage_end_exclusive_ms", END),
                         ("eligible_decision_start_ms", START + EMBARGO),
                         ("eligible_decision_end_exclusive_ms", END - EMBARGO),
                         ("price_scale_divisor", SCALE)):
        exact(case[field], value, name + "/" + field)
    exact(case["U_scaled"], float(unit * SCALE), name + "/training_unit")
    safety(case["safety"], name)
    safety_recursive(case, name)
    for key in ("costs_verified", "cost_verified", "execution_verified", "profit_qualified",
                "cash_profit_tested", "full_tick_coverage_verified"):
        require(case[key] is False, name + "/" + key)
    require(case["cash_net"] is None, name + ": cash must stay null")
    require(isinstance(rows, list), name + ": ledger must be a list")
    completed, unknown, skipped = [], [], []
    reserved = previous = None
    max_r_error = 0.0
    latency, slip_multiplier = CONFIGS[scenario]
    slip = unit * slip_multiplier
    for number, row in enumerate(rows):
        label = name + "/row/" + str(number)
        safety_recursive(row, label)
        require(not unknown, label + ": row after terminal UNKNOWN")
        decision, horizon = row["decision_ms"], row["horizon_end_ms"]
        require(type(decision) is int and START + EMBARGO <= decision < END - EMBARGO,
                label + ": decision outside embargoed stage")
        require(previous is None or decision >= previous, label + ": unordered decisions")
        require(reserved is None or decision > reserved, label + ": inclusive busy exclusion failed")
        require(type(horizon) is int and horizon == decision + 900_000 and
                horizon + latency < END, label + ": wrong frozen horizon or stage exit budget")
        previous = decision
        exact(row["direction"], direction, label + "/direction")
        exact(row["policy_id"], identity, label + "/identity")
        exact(row["baseline"], baseline, label + "/baseline")
        exact(row["public_price_scale_divisor"], SCALE, label + "/scale")
        require(row["costs_verified"] is False and row["execution_verified"] is False and
                row["cash_net"] is None and row["cash_pnl"] is None, label + ": financial claim")
        require(row["fees_model"] == "NOT_INCLUDED_NOT_VERIFIED", label + ": fee model changed")
        cfg = row["config"]
        exact(cfg["entry_latency_ms"], latency, label + "/entry_latency")
        exact(cfg["exit_latency_ms"], latency, label + "/exit_latency")
        exact(cfg["max_gap_ms"], 5000, label + "/max_gap")
        require(decimal(cfg["adverse_slippage_per_leg"]) == slip, label + ": slippage changed")
        require(decimal(row["training_unit_quote"]) == unit, label + ": training unit changed")
        stop, target = decimal(row["stop_distance"]), decimal(row["target_distance"])
        require(stop > 0, label + ": nonpositive stop")
        arithmetic(float(target), stop * 2, label + "/target_twice_stop")
        status = row["status"]
        require(status in ("COMPLETE", "UNKNOWN", "SKIP"), label + ": invalid status")
        if row["entry_time_msc"] is not None:
            entry_time = row["entry_time_msc"]
            require(type(entry_time) is int and decision + latency <= entry_time < horizon,
                    label + ": invalid saved entry timing")
            check_bundle(row["entry_bundle"], entry_time, label + "/entry_bundle")
            entry_quote = (max(decimal(q["ask"]) for q in row["entry_bundle"]) if direction == 1
                           else min(decimal(q["bid"]) for q in row["entry_bundle"]))
            require(decimal(row["entry_quote_price"]) == entry_quote,
                    label + ": saved entry is not adverse bundle side")
            require(decimal(row["entry_price"]) == entry_quote + direction * slip,
                    label + ": adverse entry cost mismatch")
        if status != "COMPLETE":
            for key in ("R", "signed_move_after_spread_and_slippage", "exit_time_msc",
                        "exit_quote_price", "exit_price"):
                require(row[key] is None, label + ": noncomplete result must be null/" + key)
            require(row["exit_bundle"] == [], label + ": noncomplete exit bundle")
            (unknown if status == "UNKNOWN" else skipped).append(row)
            continue
        require(row["entry_time_msc"] is not None, label + ": complete without entry")
        exit_time, trigger_time = row["exit_time_msc"], row["trigger_time_msc"]
        require(type(exit_time) is int and type(trigger_time) is int and
                START <= row["entry_time_msc"] <= trigger_time <= exit_time < END and
                exit_time >= trigger_time + latency, label + ": exit/trigger chronology")
        check_bundle(row["exit_bundle"], exit_time, label + "/exit_bundle")
        exit_quote = (min(decimal(q["bid"]) for q in row["exit_bundle"]) if direction == 1
                      else max(decimal(q["ask"]) for q in row["exit_bundle"]))
        require(decimal(row["exit_quote_price"]) == exit_quote,
                label + ": saved exit is not adverse bundle side")
        require(decimal(row["exit_price"]) == exit_quote - direction * slip,
                label + ": adverse exit cost mismatch")
        move = direction * (decimal(row["exit_price"]) - decimal(row["entry_price"]))
        require(decimal(row["signed_move_after_spread_and_slippage"]) == move,
                label + ": signed decimal movement mismatch")
        max_r_error = max(max_r_error, arithmetic(row["R"], move / stop, label + "/R"))
        source_time = row["trigger_source_time_msc"]
        if row["trigger_reason"] == "HORIZON":
            require(source_time is None and trigger_time == horizon and row["reason"] == "HORIZON",
                    label + ": horizon trigger mismatch")
        else:
            require(row["trigger_reason"] in ("STOP", "TARGET") and type(source_time) is int and
                    row["entry_time_msc"] <= source_time < trigger_time < horizon and
                    trigger_time - source_time <= 5000, label + ": barrier closure timing")
            require(row["reason"] == ("STOP_PRIORITY_SAME_MILLISECOND" if row["both_barriers_in_bundle"]
                                      else row["trigger_reason"]), label + ": reason mismatch")
            if row["both_barriers_in_bundle"]:
                require(row["trigger_reason"] == "STOP", label + ": same-bundle stop priority")
        completed.append(row)
        reserved = exit_time
    totals = components(completed)
    summary = case["summary"]
    for key, value in totals.items():
        exact(summary[key], value, name + "/summary/" + key)
    exact(summary["completed"], len(completed), name + "/completed")
    exact(summary["unknown"], len(unknown), name + "/unknown")
    exact(summary["skipped"], len(skipped), name + "/skipped")
    expected_consumed = (len(rows) + summary["blocked_overlap"] +
                         summary["excluded_before_embargo"] + summary["excluded_after_embargo"])
    exact(summary["input_signals_consumed"], expected_consumed, name + "/consumed_identity")
    for key in ("blocked_overlap", "excluded_before_embargo", "excluded_after_embargo"):
        require(type(summary[key]) is int and summary[key] >= 0, name + ": invalid signal count")
    censored = bool(unknown)
    require(len(unknown) <= 1, name + ": multiple UNKNOWN despite terminal censor rule")
    for key in ("remaining_censored", "unknown_and_remaining_censored"):
        exact(summary[key], censored, name + "/" + key)
    exact(summary["primary_measurement_usable"], not censored, name + "/usable")
    exact(summary["censored_from_ms"], unknown[0]["decision_ms"] if censored else None, name + "/censored_from")
    exact(summary["censored_until_exclusive_ms"], END - EMBARGO if censored else None, name + "/censored_until")
    exact(summary["remaining_opportunities_count"], None if censored else 0, name + "/remaining")
    pf = profit_factor(totals)
    exact(summary["observed_pf"], pf, name + "/observed_pf")
    for key, value in (("sum_R", totals["observed_sum_R"]),
                       ("sum_quote", totals["observed_sum_quote"]), ("pf", pf)):
        exact(summary[key], None if censored else value, name + "/primary/" + key)
    ending = 10000 + 25 * totals["observed_sum_R"]
    exact(summary["observed_completed_prefix_ending_equity_index"], ending, name + "/observed_ending")
    exact(summary["ending_equity_index"], None if censored else ending, name + "/ending")
    exact(summary["median_trade_R_observed"], statistics.median([r["R"] for r in completed])
          if completed else None, name + "/median_R")
    exact(summary["median_holding_ms_observed"], statistics.median([
        r["exit_time_msc"] - r["entry_time_msc"] for r in completed]) if completed else None,
        name + "/median_holding")
    require(summary["cash_net"] is None and summary["qualified_for_profit_claim"] is False,
            name + ": summary financial claim")
    if censored:
        for key in ("marked_drawdown_pct", "drawdown_peak_time_ms", "drawdown_trough_time_ms"):
            require(summary[key] is None, name + ": censored primary drawdown must be null")
    daily = case["daily"]
    require(len(daily) == 20, name + ": wrong daily calendar")
    positive_days = negative_days = flat_days = 0
    for number, day in enumerate(daily):
        stamp = START + number * DAY
        exact(day["day_start_ms"], stamp, name + "/daily/stamp")
        exact(day["date_utc"], datetime.fromtimestamp(stamp / 1000, timezone.utc).date().isoformat(),
              name + "/daily/date")
        day_rows = [r for r in completed if stamp <= r["exit_time_msc"] < stamp + DAY]
        day_totals = components(day_rows)
        for key, value in day_totals.items():
            exact(day[key], value, name + "/day/" + str(number) + "/" + key)
        exact(day["completed"], len(day_rows), name + "/daily/completed")
        exact(day["unknown"], sum(stamp <= r["decision_ms"] < stamp + DAY for r in unknown),
              name + "/daily/unknown")
        exact(day["primary_measurement_censored"], censored, name + "/daily/censored")
        exact(day["sum_R"], None if censored else day_totals["observed_sum_R"], name + "/daily/primary_R")
        exact(day["sum_quote"], None if censored else day_totals["observed_sum_quote"], name + "/daily/primary_quote")
        require(day["cash_net"] is None, name + ": daily cash must remain null")
        safety(day["safety"], name + "/daily")
        positive_days += day_totals["observed_sum_R"] > 0
        negative_days += day_totals["observed_sum_R"] < 0
        flat_days += day_totals["observed_sum_R"] == 0
    residuals = {key: math.fsum(day[key] for day in daily) - totals[key] for key in FIELDS}
    for key, value in residuals.items():
        exact(case["accounting_reconciliation"]["daily_to_stage_fsum_residual"][key], value,
              name + "/daily_fsum_rounding_residual/" + key)
    return {"case": name, "status": "PASS", "complete_rows": len(completed), "unknown_rows": len(unknown),
            "skip_rows": len(skipped), "row_R_max_absolute_error_decimal_normalized_prices": max_r_error,
            "stage_totals_exact_from_ledger_fsum": totals, "observed_pf_exact": pf,
            "daily_totals_exact_from_exit_UTC_day": True,
            "positive_observed_days": positive_days, "negative_observed_days": negative_days,
            "flat_observed_days": flat_days, "daily_to_stage_fsum_residual": residuals,
            "saved_exposure_reservations_nonoverlapping_inclusively": True,
            "blocked_signals_not_saved_in_ledger_not_reproduced": summary["blocked_overlap"],
            "unknown_censors_whole_stage_and_all_primary_days": censored}


def audit():
    decl_path = OUT / "declaration.json"
    require(digest(decl_path) == DECL_SHA, "Generation-2 declaration pin differs")
    decl = read(decl_path)
    safety(decl, "declaration")
    old_path = ROOT / "work/independent_profit_study_20261013/declaration.json"
    require(digest(old_path) == decl["original_declaration_sha256"], "Original declaration differs")
    old = read(old_path)
    require(decl["stages"]["WF1"] == [30, 50, 30], "Frozen WF1 calendar differs")
    index_path = OUT / "artifact_index.json"
    index_snapshot = read(index_path)
    inputs, pin_checks = {}, []
    for group, pins in (("generation2_code", decl["source_pins"]),
                        ("generation1_code", old["source_pins"]),
                        ("reused_training_fits", decl["reused_training_fit_pins"])):
        for name, expected in pins.items():
            path = ROOT / name
            require(path.is_file() and digest(path) == expected, "Changed pinned dependency: " + name)
            pin_checks.append({"group": group, "path": name, "sha256": expected})
            inputs[name] = expected
    for name_key, sha_key in (("protocol_path", "protocol_sha256"),
                              ("premeasurement_validation_path", "premeasurement_validation_sha256")):
        name, expected = decl[name_key], decl[sha_key]
        require(digest(ROOT / name) == expected, "Changed declaration document: " + name)
        inputs[name] = expected
    inputs[str(old_path.relative_to(ROOT))] = digest(old_path)
    inputs[str(decl_path.relative_to(ROOT))] = DECL_SHA
    reports, metadata, tapes_skipped, fixtures = [], {}, set(), []
    source_by_symbol = {}
    with localcontext() as ctx:
        ctx.prec = 50
        for symbol in SYMBOLS:
            fit_path = ROOT / "work/independent_profit_study_20261013/fits" / (symbol + "_TRAIN0_30.json")
            unit = decimal(read(fit_path)["U_scaled"]) / SCALE
            for policy in POLICIES:
                for scenario in CONFIGS:
                    name = f"{symbol}_WF1_{policy}_{scenario}"
                    case_path = OUT / "cases" / (name + ".json")
                    require(index_snapshot.get(str(case_path.relative_to(OUT))) == digest(case_path),
                            "WF1 case is changed/unregistered: " + name)
                    case = read(case_path)
                    ledger_path = OUT / "ledgers" / (name + ".json.gz")
                    require(case["ledger_path"] == str(ledger_path.relative_to(ROOT)) and
                            digest(ledger_path) == case["ledger_sha256"], "Ledger hash differs: " + name)
                    with gzip.open(ledger_path, "rt") as stream:
                        rows = json.load(stream)
                    inputs[str(case_path.relative_to(ROOT))] = digest(case_path)
                    inputs[str(ledger_path.relative_to(ROOT))] = digest(ledger_path)
                    source_pins = case["source_files"]
                    require(len(source_pins) == 60, "Wrong number of declared WF1 source files")
                    if symbol in source_by_symbol:
                        require(source_by_symbol[symbol] == source_pins, "Case source pins differ within symbol")
                    source_by_symbol[symbol] = source_pins
                    for source, expected in source_pins.items():
                        relative = Path(source)
                        require(relative.parts[:3] == ("data", "independent_profit_ticks_20261013", "development"),
                                "Forbidden source path in WF1 case")
                        require(symbol in str(relative), "Wrong source symbol")
                        if relative.name.endswith("_ticks.tsv"):
                            tapes_skipped.add(source)
                            continue
                        if source not in metadata:
                            require(digest(ROOT / source) == expected, "WF1 metadata source pin differs: " + source)
                            metadata[source] = expected
                    reports.append(audit_case(case, rows, name, unit))
                    fixtures.append((case, rows, name, unit))
    # The worker may add later-stage receipts, but cannot replace any audited input.
    for source, expected in inputs.items():
        require(digest(ROOT / source) == expected, "Audited input changed during audit: " + source)
    final_index = read(index_path)
    for entry, expected in index_snapshot.items():
        if "_WF1_" in entry or entry == "declaration.json":
            require(final_index.get(entry) == expected, "Audited index pin changed")
    report = {
        "status": "PASS", "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "INDEPENDENT_SAVED_WF1_LEDGER_ARITHMETIC_AND_ACCOUNTING_ONLY",
        "declaration_sha256": DECL_SHA, "auditor_sha256": digest(Path(__file__)),
        "audited_case_count": len(reports), "complete_rows_checked": sum(r["complete_rows"] for r in reports),
        "unknown_rows_checked": sum(r["unknown_rows"] for r in reports),
        "skip_rows_checked": sum(r["skip_rows"] for r in reports), "mismatches": 0,
        "daily_records_checked": 20 * len(reports), "cases": reports,
        "dependency_pin_checks": pin_checks, "small_WF1_metadata_sha256_checked": metadata,
        "raw_WF1_tick_files_declared_but_not_read_count": len(tapes_skipped),
        "input_sha256": inputs, "arithmetic_tolerances": {
            "entry_exit_prices_and_signed_quote_movement": "EXACT_DECIMAL_FROM_SAVED_JSON_PRICE_STRINGS",
            "R_and_twice_stop_target": {"relative": 8e-15, "absolute": 8e-15},
            "summary_daily_fsum_counts_PF": "EXACT_FLOAT64_VALUES_RECOMPUTED_WITH_STDLIB_MATH_FSUM",
            "R_tolerance_reason": "Public stop distance normalized by 1e8 can roundtrip by a float ULP",
        },
        "coverage": [
            "All sixteen fixed WF1 case and gzip-ledger SHA256 pins",
            "Every COMPLETE saved entry/exit bundle adverse side and configured per-leg slippage",
            "Every COMPLETE signed quote movement and R; exact ledger fsum, gain, loss, PF and medians",
            "All 320 daily rows independently grouped by actual exit UTC day",
            "Embargoed decision bounds, stage chronology and saved-trade inclusive busy exclusion",
            "UNKNOWN null results, terminal position in ledger, primary whole-stage/day censorship",
            "Both generations' declared code hashes, reused fit hashes and small WF1 metadata hashes",
        ],
        "not_verified": [
            "Raw-tape saved quote authentication or full provider tick completeness",
            "First eligible quote, within-exposure gaps, trigger barrier path or stop/target reproduction",
            "Causal bar construction, setup/issuance features or skipped/blocked signal eligibility",
            "TRAIN unit refitting, marked-to-market drawdown path or marked-quote row counts",
            "Other walk-forward folds, validation, sealed FINAL or any private financial data",
            "Historical broker fees, actual fills, cash profitability or tradeable strategy qualification",
        ], "cash_net": None, "costs_verified": False, "execution_verified": False,
        "profit_qualified": False, "final_prices_read": False, "market_or_account_API_calls": 0,
        **{key: False for key in SAFETY},
    }
    return report, fixtures


def mutation_checks(fixtures):
    base = next(f for f in fixtures if f[2] == "BOOM500_WF1_CRT_SWEEP_RECLAIM_BASE")
    unknown = next(f for f in fixtures if f[2] == "BOOM500_WF1_CLOCK_BASE")
    checks = []
    mutations = [
        (base, "R arithmetic", lambda c, r: r[0].update(R=r[0]["R"] + 0.01)),
        (base, "adverse entry cost", lambda c, r: r[0].update(entry_price=r[0]["entry_price"] - 0.1)),
        (base, "daily accounting", lambda c, r: c["daily"][2].update(observed_sum_R=99.0)),
        (base, "summary PF", lambda c, r: c["summary"].update(pf=99.0)),
        (base, "embargo", lambda c, r: r[0].update(decision_ms=START)),
        (base, "inclusive busy", lambda c, r: r[1].update(decision_ms=r[0]["exit_time_msc"])),
        (unknown, "UNKNOWN cannot become zero", lambda c, r: r[-1].update(R=0.0)),
        (unknown, "censored daily cannot become zero", lambda c, r: c["daily"][-1].update(sum_R=0.0)),
        (base, "safety flag", lambda c, r: c["safety"].update(LIVE_ALLOWED=True)),
    ]
    for fixture, label, mutate in mutations:
        case, rows, name, unit = copy.deepcopy(fixture)
        mutate(case, rows)
        try:
            audit_case(case, rows, name, unit)
        except (ValueError, KeyError, TypeError):
            checks.append({"mutation": label, "detected": True})
        else:
            raise ValueError("Auditor failed to detect mutation: " + label)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Also reject nine in-memory evidence mutations")
    args = parser.parse_args()
    report, fixtures = audit()
    report["in_memory_mutation_checks"] = mutation_checks(fixtures) if args.self_test else []
    output = OUT / "independent_WF1_accounting_audit.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "cases": report["audited_case_count"],
                      "complete_rows": report["complete_rows_checked"],
                      "unknown_rows": report["unknown_rows_checked"],
                      "daily_rows": report["daily_records_checked"],
                      "mismatches": report["mismatches"],
                      "mutation_checks": len(report["in_memory_mutation_checks"]),
                      "report_sha256": digest(output), "report": str(output.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
