"""Audit saved final ledgers; never import the engine or run a market simulation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import fcntl
import json
import math
from pathlib import Path
import statistics
import time

FLAGS = {"LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES",
         "ready_for_live", "live_allowed", "opened_trades"}
STEPS = {"M1": 60, "M2": 120, "M3": 180, "M5": 300, "M10": 600,
         "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
INITIAL = 10000.0
RISK = 0.005
EXPECTED_LATEST = "c7ac7cbc3d6009131edd516c512f388fa9b34c8f30b65f9b09ae563d9b0fb0d0"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    stream = gzip.open(path, "rt") if str(path).endswith(".gz") else Path(path).open()
    with stream:
        return json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.errors = []
        self.error_count = 0
        self.max_absolute_difference = defaultdict(float)

    def check(self, condition, label, detail=None):
        self.counts[label] += 1
        if not condition:
            self.error_count += 1
            if len(self.errors) < 100:
                self.errors.append({"check": label, "detail": detail})

    def close(self, actual, expected, label, detail=None, *, cash=False):
        valid = (type(actual) in (float, int) and math.isfinite(actual)
                 and type(expected) in (float, int) and math.isfinite(expected))
        if valid:
            self.max_absolute_difference[label] = max(self.max_absolute_difference[label], abs(actual - expected))
        # Independent summation and price-to-cash arithmetic can differ by
        # floating-point roundoff. These tolerances are fixed before the audit.
        self.check(valid and math.isclose(actual, expected, rel_tol=5e-10,
                                         abs_tol=1e-8 if cash else 1e-11), label, detail)


def audit_result(artifact, checks=None, measurement=None, declared_costs=None):
    c = checks or Checks()
    r = artifact["final"]
    spec, costs, metrics = r["strategy"], r["costs"], r["metrics"]
    trades, equity = r["trades"], r["equity"]
    group = f"{r['symbol']}/{r['timeframe']}/{spec['family']}"
    before_errors = c.error_count
    identity = artifact["identity"]
    c.check(r["strategy"] == artifact["spec"] and r["strategy_id"] == artifact["spec"]["id"],
            "nested_strategy_matches_original_wrapper", group)
    c.check(r["symbol"] == artifact["symbol"] == identity["symbol"]
            and r["timeframe"] == artifact["timeframe"] == identity["timeframe"]
            and r["partition"] == identity["phase"] == "final"
            and r["start"] == identity["split"]["validation_end"]
            and r["end"] == identity["split"]["end"], "nested_final_identity", group)
    result_identity = {k: r[k] for k in ("strategy", "symbol", "timeframe", "start", "end", "partition", "costs")}
    expected_id = hashlib.sha256(json.dumps(result_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    c.check(r["id"] == expected_id, "result_identity_hash", group)
    if measurement is not None:
        c.check(group == measurement["selection_group"] and r["strategy_id"] == measurement["strategy_id"]
                and r["symbol"] == measurement["symbol"] and r["timeframe"] == measurement["timeframe"]
                and identity["input_sha256"] == measurement["source_sha256"],
                "nested_result_matches_selected_measurement", group)
    if declared_costs is not None:
        c.check(costs == declared_costs, "cost_assumptions_match_frozen_declaration", group)
    direction = 1 if r["symbol"].startswith("BOOM") else -1
    step = STEPS[r["timeframe"]]
    start, end = r["start"], r["end"]
    c.check(r["partition"] == "final", "final_partition", group)
    c.check(set(r["safety"]) == FLAGS and all(v is False for v in r["safety"].values()),
            "strict_false_safety", group)
    c.check(spec["trailing_stop"] is False, "no_trailing_stop", group)
    c.close(metrics["initial_equity"], INITIAL, "fixed_initial_capital", group)
    c.check(bool(equity) and equity[0] == {"epoch": start, "equity": INITIAL, "kind": "initial"},
            "initial_equity_point", group)
    cash = INITIAL
    previous_exit = start
    reconstructed_pnl, reconstructed_r = [], []
    months = defaultdict(list)
    for index, trade in enumerate(trades):
        item = f"{group}:{index}"
        atr, entry, exit_price = trade["entry_atr"], trade["entry_price"], trade["exit_price"]
        unit_risk = spec["stop_atr"] * atr
        c.check(atr > 0 and unit_risk > 0 and cash > 0, "positive_risk_and_pretrade_cash", item)
        c.check(trade["direction"] == direction, "native_direction", item)
        c.close(trade["risk_fraction"], RISK, "fixed_risk_fraction", item)
        c.close(trade["equity_before"], cash, "cash_chain_before", item, cash=True)
        risk_cash = cash * RISK
        quantity = risk_cash / unit_risk
        fee_price = costs["round_trip_atr"] * atr + costs["round_trip_price"]
        c.close(trade["risk_cash"], risk_cash, "risk_cash", item, cash=True)
        c.close(trade["quantity"], quantity, "position_quantity", item)
        c.close(trade["round_trip_cost_price"], fee_price, "round_trip_cost_once", item)
        c.close(trade["stop_price"], entry - direction * unit_risk, "fixed_stop", item)
        c.close(trade["target_price"], entry + direction * spec["target_atr"] * atr,
                "fixed_target", item)
        # Use price change times quantity, less one full-cycle fee. This is
        # separate from the production engine's risk_cash * net_R computation.
        gross_cash = direction * (exit_price - entry) * quantity
        fee_cash = quantity * fee_price
        pnl = gross_cash - fee_cash
        gross_r = gross_cash / risk_cash
        net_r = pnl / risk_cash
        c.close(trade["gross_R"], gross_r, "gross_R_from_cash", item)
        c.close(trade["net_R"], net_r, "net_R_from_cash", item)
        c.close(trade["net_pnl"], pnl, "net_cash_from_prices_quantity_fee", item, cash=True)
        c.close(trade["equity_after"], cash + pnl, "cash_chain_after", item, cash=True)
        c.check(trade["signal_epoch"] == trade["entry_epoch"], "signal_close_next_open", item)
        c.check(trade["entry_bar_index"] == trade["signal_bar_index"] + 1,
                "next_bar_index", item)
        c.check(start <= trade["entry_epoch"] <= trade["exit_epoch"] <= end,
                "trade_partition_bounds", item)
        c.check(trade["entry_epoch"] >= previous_exit, "one_position_chronology", item)
        c.check(trade["planned_end_epoch"] == trade["entry_epoch"] + spec["max_hold_bars"] * step
                and trade["exit_epoch"] <= trade["planned_end_epoch"] <= end,
                "planned_horizon", item)
        reason = trade["exit_reason"]
        if reason in {"STOP", "AMBIGUOUS_STOP_FIRST"}:
            c.close(exit_price, trade["stop_price"], "stop_exit_price", item)
        elif reason == "TARGET":
            c.close(exit_price, trade["target_price"], "target_exit_price", item)
        elif reason == "GAP_STOP":
            c.check(direction * (exit_price - trade["stop_price"]) <= 1e-9,
                    "uncapped_adverse_stop_gap", item)
        elif reason == "GAP_TARGET":
            c.check(direction * (exit_price - trade["target_price"]) >= -1e-9,
                    "favorable_target_gap", item)
        elif reason == "TIMEOUT":
            c.check(trade["exit_epoch"] == trade["planned_end_epoch"], "timeout_at_horizon", item)
        else:
            c.check(False, "recognized_candle_exit_reason", item)
        c.check(trade["execution_mode"] == "assumed_candle_stop_first", "disclosed_candle_mode", item)
        c.check(trade["ambiguous_candle"] is (reason == "AMBIGUOUS_STOP_FIRST"),
                "ambiguity_flag", item)
        reconstructed_pnl.append(pnl)
        reconstructed_r.append(net_r)
        month = datetime.fromtimestamp(trade["exit_epoch"], timezone.utc).strftime("%Y-%m")
        months[month].append((trade, pnl, net_r))
        # Continue with independently computed cash; never floor a gap loss.
        cash += pnl
        previous_exit = trade["exit_epoch"]
    c.close(metrics["final_equity"], cash, "metric_final_cash", group, cash=True)
    c.close(metrics["net_profit"], math.fsum(reconstructed_pnl), "metric_net_profit", group, cash=True)
    c.close(metrics["net_return"], cash / INITIAL - 1, "metric_net_return", group)
    c.check(type(metrics["completed_trades"]) is int and type(metrics["sample_size"]) is int
            and metrics["completed_trades"] == len(trades) == metrics["sample_size"], "trade_counts", group)
    # Classify exact stored cash signs after independently validating every
    # amount. Reordering floating operations near zero is not a new win/loss.
    saved_cash = [t["net_pnl"] for t in trades]
    positive = math.fsum(x for x in saved_cash if x > 0)
    negative = -math.fsum(x for x in saved_cash if x < 0)
    if negative:
        c.close(metrics["profit_factor"], positive / negative, "cash_profit_factor", group)
    else:
        c.check(metrics["profit_factor"] is None, "undefined_PF_without_losses", group)
    if trades:
        c.close(metrics["win_rate"], sum(x > 0 for x in saved_cash) / len(trades),
                "strict_positive_cash_win_rate", group)
        c.close(metrics["mean_net_R"], math.fsum(reconstructed_r) / len(trades), "mean_net_R", group)
        c.close(metrics["median_holding_seconds"], statistics.median(
            x["exit_epoch"] - x["entry_epoch"] for x in trades), "median_holding_not_spike_time", group)
    else:
        c.check(all(metrics[k] is None for k in ("win_rate", "mean_net_R", "median_holding_seconds")),
                "empty_ledger_null_metrics", group)
    c.check(type(metrics["active_days"]) is int
            and metrics["active_days"] == len({t["entry_epoch"] // 86400 for t in trades}),
            "UTC_active_days", group)
    c.close(metrics["period_days"], (end - start) / 86400, "calendar_period_days", group)
    # Retain duplicate-epoch extrema in their saved order. This checks DD
    # arithmetic on that path, not whether those marks match actual ticks.
    peak = -math.inf
    drawdown = 0.0
    previous_epoch = start
    realized = []
    for point in equity:
        value = point["equity"]
        c.check(type(value) in (int, float) and math.isfinite(value), "finite_equity", group)
        c.check(previous_epoch <= point["epoch"] <= end, "equity_chronology", group)
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / max(peak, 1e-12))
        if point["kind"] == "realized":
            realized.append(point)
        previous_epoch = point["epoch"]
    c.close(metrics["max_drawdown"], drawdown, "full_saved_path_drawdown", group)
    c.close(equity[-1]["equity"], cash, "last_saved_mark_equals_final_cash", group, cash=True)
    c.check(len(realized) == len(trades), "one_realized_point_per_trade", group)
    for index, (point, trade) in enumerate(zip(realized, trades)):
        c.check(point["epoch"] == trade["exit_epoch"], "realized_epoch", f"{group}:{index}")
        c.close(point["equity"], trade["equity_after"], "realized_equity", f"{group}:{index}", cash=True)
    completed = [x for x in r["dispositions"] if x["status"] == "COMPLETED"]
    c.check([x["trade_index"] for x in completed] == list(range(len(trades))),
            "completed_disposition_bijection", group)
    for index, (disposition, trade) in enumerate(zip(completed, trades)):
        c.check(all(disposition[k] == trade[k] for k in
                    ("signal_epoch", "entry_epoch", "planned_end_epoch")),
                "completed_disposition_identity", f"{group}:{index}")
    c.check(r["unknown_paths"] == sum(x["status"].startswith("UNKNOWN_") for x in r["dispositions"]),
            "unknown_paths_not_zero_return_trades", group)
    c.check(r["purged_signals"] == sum(x["status"] == "PURGED_FULL_PLANNED_HORIZON"
                                     for x in r["dispositions"]), "purged_signals", group)
    c.check(r["ambiguous_candles"] == sum(t["ambiguous_candle"] for t in trades),
            "ambiguous_candle_count", group)
    saved_months = {x["month"]: x for x in r["monthly"]}
    c.check(len(saved_months) == len(r["monthly"]) and set(saved_months) == set(months),
            "complete_unique_UTC_months", group)
    for month, rows in months.items():
        if month not in saved_months:
            continue
        saved = saved_months[month]
        c.check(type(saved["completed_trades"]) is int and saved["completed_trades"] == len(rows),
                "monthly_trade_count", group)
        for field, expected in {
            "net_profit": math.fsum(x[1] for x in rows),
            "sum_net_R": math.fsum(x[2] for x in rows),
            "equity_start": rows[0][0]["equity_before"],
            "equity_end": rows[-1][0]["equity_after"],
            "net_return": rows[-1][0]["equity_after"] / rows[0][0]["equity_before"] - 1,
        }.items():
            c.close(saved[field], expected, "monthly_" + field, f"{group}:{month}", cash=True)
    for key, expected_direction in (("buy_and_hold", 1), ("native_direction_hold", direction)):
        ref = r["reference"].get(key)
        if ref is None:
            continue
        c.check(ref["direction"] == expected_direction, "reference_direction", group)
        net = (expected_direction * (ref["exit_price"] - ref["entry_price"])
               - ref["round_trip_cost_price"]) / ref["entry_price"]
        c.close(ref["net_return"], net, "reference_return_from_saved_quotes", group)
        c.close(ref["net_profit"], INITIAL * net, "reference_assumed_cash", group, cash=True)
    return {"group": group, "strategy_id": r["strategy_id"], "trades": len(trades),
            "equity_points": len(equity), "months": len(months),
            "errors": c.error_count - before_errors, "assumed_final_cash": cash,
            "saved_path_drawdown": drawdown}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    campaign = root / "data/single_frame_campaign_20261011"
    if output.exists() or output.is_relative_to(campaign) or not output.is_relative_to(root / "work"):
        raise ValueError("Use a new output file only under work, outside the campaign")
    lock = output.with_suffix(".lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    replay = root / "work/final_replay_20261011"
    latest_path = replay / "latest.json"
    index_path = replay / "receipts_index.json"
    latest, index = read_json(latest_path), read_json(index_path)
    if digest(latest_path) != EXPECTED_LATEST or index["campaign"] != latest["campaign"]:
        raise ValueError("Completed authoritative replay or index identity changed")
    if latest["status"] != "PASS" or latest["selected_final_and_WF_reproduction_complete"] is not True:
        raise ValueError("Requires the completed unchanged replay")
    pins = {str(latest_path): digest(latest_path), str(index_path): digest(index_path)}
    script_path = Path(__file__).resolve()
    pins[str(script_path)] = digest(script_path)
    declaration_path = campaign / "declaration.json"
    pins[str(declaration_path)] = latest["campaign"]["declaration_sha256"]
    if digest(declaration_path) != pins[str(declaration_path)]:
        raise ValueError("Declaration seal differs")
    declaration = read_json(declaration_path)
    for name, expected in latest["campaign"]["source_pins"].items():
        path = root / "research_campaign" / name
        if digest(path) != expected:
            raise ValueError("Frozen scientific source differs")
        pins[str(path)] = expected
    plan = []
    seen_receipts = set()
    for entry in latest["receipts"]:
        ref = entry["receipt"]
        path = (replay / ref["path"]).resolve()
        if not path.is_relative_to(replay) or digest(path) != ref["sha256"]:
            raise ValueError("Invalid receipt path/hash")
        rec = read_json(path)
        key = path.stem
        report = rec["report"]
        measurement = rec["identity"]["measurement"]
        if (key in seen_receipts or index["receipts"].get(key) != ref
                or rec["identity"]["campaign"] != latest["campaign"] or rec["status"] != "PASS"
                or entry["selection_group"] != measurement["selection_group"]
                or rec["report_sha256"] != hashlib.sha256(encoded(report)).hexdigest()
                or report["status"] != "PASS" or report["cache_used"] is not False
                or report["phase"] != measurement["phase"] or report["artifact"] != measurement["artifact"]):
            raise ValueError("Receipt identity/status mismatch")
        seen_receipts.add(key)
        pins[str(path)] = ref["sha256"]
        if rec["identity"]["measurement"]["phase"] == "final":
            pins[str(path)] = ref["sha256"]
            plan.append(rec["identity"]["measurement"])
    if seen_receipts != set(index["receipts"]):
        raise ValueError("Latest and index receipt populations differ")
    if len(plan) != 1300 or len({p["selection_group"] for p in plan}) != 1300:
        raise ValueError("Expected exactly 1300 selected final policies")
    c = Checks()
    policies = []
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    for measurement in plan:
        path = (campaign / measurement["artifact"]["path"]).resolve()
        expected = measurement["artifact"]["sha256"]
        if not path.is_relative_to(campaign / "artifacts") or digest(path) != expected:
            raise ValueError("Invalid original artifact path/hash")
        pins[str(path)] = expected
        artifact = read_json(path)
        c.check(artifact["identity"]["phase"] == "final"
                and artifact["symbol"] == measurement["symbol"]
                and artifact["timeframe"] == measurement["timeframe"]
                and artifact["spec"]["id"] == measurement["strategy_id"],
                "artifact_matches_selected_identity", measurement["selection_group"])
        policies.append(audit_result(artifact, c, measurement, declaration["costs"]))
        if len(policies) % 100 == 0:
            print(json.dumps({"policies": len(policies), "errors": c.error_count}), flush=True)
    for name, expected in pins.items():
        if digest(name) != expected:
            raise ValueError("Input changed during the read-only audit: " + name)
    report = {
        "schema_version": 1, "status": "PASS" if not c.error_count else "FAIL",
        "scope": "Independent arithmetic on all selected final saved ledgers and equity marks; no strategy simulation, signal generation, bootstrap or engine imports",
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - timer, "policies": policies,
        "totals": {"policies": len(policies), "trades": sum(x["trades"] for x in policies),
                   "equity_points": sum(x["equity_points"] for x in policies)},
        "checks": dict(c.counts), "check_count": sum(c.counts.values()),
        "max_absolute_difference_by_check": dict(c.max_absolute_difference),
        "numeric_tolerance": {"relative": 5e-10, "absolute_cash": 1e-8, "absolute_other": 1e-11},
        "error_count": c.error_count, "first_errors": c.errors,
        "input_sha256_before_and_after": pins, "inputs_unchanged": True,
        "new_backtests": 0, "production_engine_imported": False,
        "historical_CFD_costs_verified": False, "cash_contract_mapping_verified": False,
        "original_tick_execution_verified": False, "qualification_changed": False,
        "safety": {key: False for key in sorted(FLAGS)},
        "limitations": [
            "Saved fill prices and ATR values are inputs, not independently proven executable observations.",
            "Full saved-path drawdown arithmetic is checked; underlying OHLC envelope chronology and tick marks are not independently reconstructed.",
            "Reference arithmetic uses saved quotes/costs, without verifying CFD financing or cost reality.",
            "Weekly uncertainty, Holm, strategy signals and the development search are outside this independent arithmetic audit.",
            "Trade and equity-point totals overlap policies; they are not independent market-event sample sizes.",
        ],
    }
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "totals", "check_count", "error_count", "runtime_seconds")}), flush=True)
    raise SystemExit(1 if c.error_count else 0)


if __name__ == "__main__":
    main()
