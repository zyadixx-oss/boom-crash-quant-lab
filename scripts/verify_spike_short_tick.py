#!/usr/bin/env python3
"""Independent stdlib audit of frozen SHORT1 tick/M1 quote arithmetic.

Only scalar paths and saved issuance are reconstructed. Feature generation,
fitting, bootstrap draws, inference and financial execution are excluded.
Historical execution requires a completed frozen study and authorization.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_spike_short_target as old
from scripts import verify_spike_tick_execution as tick7

FLAGS = old.FLAGS
SYMBOLS, MODES = old.SYMBOLS, old.MODES
FAMILIES = ("SHORT44", "LONG44", "CLOCK")
DATES = tick7.DATES
BOOTSTRAP, SEED, PURGE = 9999, 20261005, 31
TICK_CONFIG = {"stop_atr": 2., "max_hold_minutes": 1, "entry_delay_minutes": 1,
               "round_trip_cost_atr": .10, "max_gap_seconds": 1, "stop_latency_ticks": 1}
COARSE_CONFIG = old.CONFIG
SIGNAL_COLUMNS = ["signal_time", "atr", "side", "signal_close", "variant", "score"]
COARSE_COLUMNS = ["signal_time", "entry_time", "exit_time", "entry", "exit", "atr", "gross_R", "net_R",
                  "reason", "ambiguous", "holding_minutes", "censored", "variant", "planned_end", "missing_time"]
TICK_COLUMNS = [*COARSE_COLUMNS, "nominal_entry_time", "trigger_time", "trigger_quote", "side"]
MATCHED_COLUMNS = ["signal_time", "net_R_tick", "net_R_coarse", "tick_minus_coarse_R"]
CONFIG = {"tick": TICK_CONFIG, "coarse": COARSE_CONFIG, "purge_minutes": PURGE,
          "bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED,
          "scheduled_observed_days": 12, "nominal_clock_opportunities": 576,
          "maximum_common_purged_opportunities_per_symbol_model": 564,
          "bootstrap_basis": "twelve_scheduled_observed_UTC_days_including_observed_empty_days",
          "unsampled_calendar_days_are_zero": False, "weekly_resampling": False,
          "new_scores_features_fits_or_thresholds": False, "unknown_outcomes_are_zero": False}
HISTORY = {"known_prices": True, "known_prior_payoffs": True, "price_oos": False,
           "new_label_holdout": False, "prospective": False,
           "all_twelve_tick_days_postdate_frozen_older70_fit": True,
           "study_kind": "fixed_known_history_short_execution_fidelity_not_profit_validation"}
TOTAL_COUNTERS = (
    "issued", "filled", "completed", "censored", "missing_entry", "outside_partition",
    "purged", "overlap_skipped", "ambiguous")
COUNTERS = ("issued", "filled", "completed", "censored", "missing_entry", "missing_path",
            "outside_partition", "purged", "overlap_skipped", "ambiguous")
DEPENDENCIES = {**old.DEPENDENCIES,
               "scripts/verify_spike_short_target.py": "8c2ddcc5c6015d2e799baa0908530d9431711f49a6f78f7ab0843e0947ff0fb4"}
Audit, compare, finite = old.Audit, old.compare, old.finite
digest = old.digest


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_float=finite, parse_constant=invalid, object_pairs_hook=pairs)


def stamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.0+)?(?:Z|\+00:00)", value):
        raise ValueError("Whole aware UTC timestamps required")
    return old.stamp(value)


def run_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Aware UTC run chronology required")
    return parsed


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected canonical repository-relative POSIX path")
    lexical = PurePosixPath(value)
    if lexical.is_absolute() or ".." in lexical.parts or str(lexical) != value or value == ".":
        raise ValueError("Path aliases, traversal and absolute paths are refused")
    path = ROOT
    for part in lexical.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Artifact symlink aliases are refused")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Artifact symlink escapes the repository")
    return path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def exact(audit, label, actual, expected):
    """Canonical JSON distinguishes bool/int and int/float aliases."""
    audit.require(label, canonical_hash(actual) == canonical_hash(expected))


def artifact(value, audit):
    compare(audit, "artifact_descriptor", value, {"local_ignored_artifact": True})
    audit.require("artifact_rows_integer", type(value["rows"]) is int and value["rows"] >= 0)
    audit.require("artifact_columns_strings_unique", isinstance(value["columns"], list) and
                  all(isinstance(x, str) and x for x in value["columns"]) and
                  len(set(value["columns"])) == len(value["columns"]))
    path = relative_path(value["path"])
    audit.pin(path, value["sha256"])
    return path


def csv_rows(value, audit):
    with artifact(value, audit).open(newline="") as stream:
        reader = csv.DictReader(stream)
        audit.require("artifact_exact_csv_schema", reader.fieldnames == value["columns"])
        rows = list(reader)
    audit.require("artifact_rectangular_csv_rows", all(set(r) == set(value["columns"]) and
                  all(cell is not None for cell in r.values()) for r in rows))
    audit.equal("artifact_exact_row_count", len(rows), value["rows"])
    return rows


def side_for(symbol, mode):
    if symbol not in SYMBOLS or mode not in MODES:
        raise ValueError("Unknown native symbol/direction")
    return (1 if symbol.startswith("BOOM") else -1) * (1 if mode == "SPIKE" else -1)


def signals_from_rows(rows, symbol, key, identity=None):
    family, mode = key.rsplit("_", 1)
    if family not in FAMILIES or mode not in MODES:
        raise ValueError("Unknown saved family/mode")
    columns = {"signal_time", "atr", "side", "signal_close", "variant", "score"}
    result, previous = [], None
    for raw in rows:
        if set(raw) != columns:
            raise ValueError("Exact saved issuance schema required")
        t = stamp(raw["signal_time"])
        if t % 1800 or previous is not None and t <= previous:
            raise ValueError("Strict sorted unique UTC00/30 issuance required")
        side = raw["side"]
        if isinstance(side, str) and side in ("-1", "1"):
            side = int(side)
        if type(side) is not int or side != side_for(symbol, mode) or raw["variant"] != key:
            raise ValueError("Frozen issuance family/direction identity mismatch")
        row = {"signal_time": t, "atr": finite(raw["atr"]), "side": side,
               "signal_close": finite(raw["signal_close"]), "variant": key}
        if row["atr"] <= 0 or row["signal_close"] <= 0:
            raise ValueError("Positive frozen ATR/closed quote required")
        if family == "CLOCK":
            if raw["score"] not in (None, ""):
                raise ValueError("Saved CLOCK score must remain blank")
        else:
            row["score"] = finite(raw["score"])
            if row["score"] <= 0:
                raise ValueError("Saved learned issuance must retain positive score")
            if identity is not None and (row["score"] < finite(identity["frozen_score_threshold"]) or
                                         t < stamp(identity["training_end"])):
                raise ValueError("Saved frozen score threshold/training chronology changed")
        result.append(row); previous = t
    return result


def scheduled(signals, date):
    start, end = tick7.date_bounds(date)
    return [row for row in signals if start <= row["signal_time"] < end]


def tick_replay(ticks, signals, start, end):
    """Second-grid scalar SHORT1 walk independent of inherited hold15 code."""
    if type(start) is not int or type(end) is not int or start >= end:
        raise ValueError("Ordered integer UTC day bounds required")
    rows, busy, previous = [], start, None
    counts = dict.fromkeys(COUNTERS, 0); counts["issued"] = len(signals)
    for signal in signals:
        issue, atr, side = signal["signal_time"], finite(signal["atr"]), signal["side"]
        if type(issue) is not int or type(side) is not int or side not in (-1, 1) or atr <= 0:
            raise ValueError("Valid scalar signal required")
        if previous is not None and issue <= previous:
            raise ValueError("Sorted unique scalar signals required")
        previous = issue
        nominal, planned = issue + 60, issue + 120
        if not start <= issue < end:
            counts["outside_partition"] += 1; continue
        if issue + PURGE * 60 > end or planned + 1 > end:
            counts["purged"] += 1; continue
        if nominal < busy:
            counts["overlap_skipped"] += 1; continue
        entry_time = nominal + 1
        if entry_time not in ticks:
            counts["missing_entry"] += 1; busy = planned; continue
        entry, risk = finite(ticks[entry_time]), 2 * atr
        if entry <= 0 or not math.isfinite(risk) or risk <= 0:
            raise ValueError("Finite positive entry/risk required")
        stop = entry - side * risk
        if not math.isfinite(stop) or stop <= 0:
            raise ValueError("Finite positive stop required")
        row = {"signal_time": issue, "nominal_entry_time": nominal, "entry_time": entry_time,
               "exit_time": planned, "entry": entry, "exit": None, "atr": atr,
               "gross_R": None, "net_R": None, "reason": "censored_missing_path",
               "ambiguous": False, "holding_minutes": (planned - entry_time) / 60,
               "censored": False, "variant": signal.get("variant", ""), "planned_end": planned,
               "missing_time": None, "trigger_time": None, "trigger_quote": None, "side": side}
        counts["filled"] += 1
        for t in range(entry_time + 1, planned + 2):
            if t not in ticks:
                row.update(censored=True, missing_time=t); break
            quote = finite(ticks[t])
            if quote <= 0:
                raise ValueError("Finite positive path quote required")
            if t == planned + 1:
                row.update(exit=quote, exit_time=t, reason="time"); break
            if side * (quote - stop) <= 0:
                row.update(trigger_time=t, trigger_quote=quote)
                if t + 1 not in ticks:
                    row.update(censored=True, missing_time=t + 1)
                else:
                    exit_quote = finite(ticks[t + 1])
                    if exit_quote <= 0:
                        raise ValueError("Finite positive successor quote required")
                    row.update(exit=exit_quote, exit_time=t + 1, reason="sl")
                break
        if row["censored"]:
            counts["censored"] += 1; counts["missing_path"] += 1; busy = planned
        else:
            row["gross_R"] = side * (row["exit"] - entry) / risk
            row["net_R"] = row["gross_R"] - .05
            if not math.isfinite(row["gross_R"]) or not math.isfinite(row["net_R"]):
                raise ValueError("Finite completed return required")
            counts["completed"] += 1; busy = row["exit_time"]
        row["holding_minutes"] = (row["exit_time"] - entry_time) / 60
        rows.append(row)
    return rows, counts


def known(rows):
    result, times = [], set()
    for row in rows:
        if type(row["censored"]) is not bool:
            raise ValueError("Strict censored boolean required")
        if not row["censored"] and row["net_R"] is not None:
            finite(row["net_R"]); finite(row["gross_R"])
            if row["signal_time"] in times:
                raise ValueError("Completed issuance must be unique")
            times.add(row["signal_time"]); result.append(row)
    return result


def point_metrics(rows, dates=DATES):
    if not dates or len(set(dates)) != len(dates) or any(d not in DATES for d in dates):
        raise ValueError("Nonempty unique scheduled day units required")
    completed = known(rows)
    if any(tick7.date_of(r["signal_time"]) not in dates for r in rows):
        raise ValueError("Unscheduled path in metrics")
    values = [r["net_R"] for r in completed]; gross = [r["gross_R"] for r in completed]
    n = len(values); mean = statistics.fmean(values) if n else None
    gains, losses = [v for v in values if v > 0], [v for v in values if v < 0]
    equity = peak = 1.; drawdown, ruined = 0., False
    for row in sorted(completed, key=lambda r: (r["exit_time"], r["entry_time"])):
        factor = 1 + .0025 * row["net_R"]
        ruined = ruined or factor <= 0
        equity = 0. if ruined else equity * factor
        peak = max(peak, equity); drawdown = max(drawdown, 1 - equity / peak)
    day_values = {d: [r["net_R"] for r in completed if tick7.date_of(r["signal_time"]) == d] for d in dates}
    residual = math.fsum((math.fsum(v) - mean * len(v)) ** 2 for v in day_values.values()) if n else None
    mean_gross = statistics.fmean(gross) if n else None
    return {"trades": len(rows), "completed": n, "censored": sum(r["censored"] for r in rows),
            "invalid_uncensored": len(rows) - sum(r["censored"] for r in rows) - n,
            "wins": len(gains), "losses": len(losses), "flat_trades": sum(v == 0 for v in values),
            "win_rate": len(gains) / n if n else None, "mean_net_R": mean,
            "median_net_R": statistics.median(values) if n else None, "sum_net_R": math.fsum(values),
            "mean_gross_R": mean_gross,
            "profit_factor": math.fsum(gains) / -math.fsum(losses) if losses else None,
            "average_win_R": statistics.fmean(gains) if gains else None,
            "average_loss_R": statistics.fmean(losses) if losses else None,
            "active_days": sum(bool(v) for v in day_values.values()), "calendar_days": len(dates),
            "span_calendar_days": (tick7.date_bounds(dates[-1])[1] - tick7.date_bounds(dates[0])[0]) // 86400,
            "median_holding_minutes": statistics.median(r["holding_minutes"] for r in completed) if n else None,
            "ambiguous": sum(r["ambiguous"] for r in rows),
            "break_even_cost_atr": mean_gross * 2 if mean_gross is not None else None,
            "cluster_se": math.sqrt(len(dates) / (len(dates) - 1) * residual) / n if n and len(dates) > 1 else None,
            "selection_score": None, "closed_trade_return": equity - 1,
            "closed_trade_max_drawdown": drawdown, "equity_ruin": ruined,
            "illustrative_risk_fraction": .0025,
            "drawdown_basis": "closed trades only; fixed fractional risk illustration"}


def matched_known(tick, coarse):
    a = {r["signal_time"]: r for r in known(tick)}
    b = {r["signal_time"]: r for r in known(coarse)}
    rows = [{"signal_time": t, "net_R_tick": a[t]["net_R"], "net_R_coarse": b[t]["net_R"],
             "delta": a[t]["net_R"] - b[t]["net_R"]} for t in sorted(a.keys() & b.keys())]
    return rows, {"matched_completed": len(rows),
                  "tick_minus_coarse_mean_R": statistics.fmean(r["delta"] for r in rows) if rows else None,
                  "unmatched_tick_completed": len(a) - len(rows),
                  "unmatched_coarse_completed": len(b) - len(rows)}


def reference_difference(left, right):
    a, b = known(left), known(right)
    return statistics.fmean(r["net_R"] for r in a) - statistics.fmean(r["net_R"] for r in b) if a and b else None


def ci_contract(saved, audit, label, interval_key, valid_key, undefined_key, *, empty=False, defined=True):
    """Saved CI count/schema checks only; no bootstrap draw regeneration."""
    compare(audit, label + "/bootstrap", saved, {"bootstrap_repeats": BOOTSTRAP, "bootstrap_seed": SEED})
    valid, undefined = saved[valid_key], saved[undefined_key]
    audit.require(label + "/integer_draw_counts", type(valid) is int and type(undefined) is int)
    audit.require(label + "/bounded_draw_counts", 0 <= valid <= BOOTSTRAP and 0 <= undefined <= BOOTSTRAP)
    audit.require(label + "/complete_draw_accounting", valid + undefined == BOOTSTRAP)
    interval = saved[interval_key]
    audit.require(label + "/pair_interval", isinstance(interval, list) and len(interval) == 2)
    if valid == 0:
        exact(audit, label + "/undefined_interval", interval, [None, None])
    else:
        audit.require(label + "/numeric_interval", all(type(v) in (int, float) for v in interval))
        low, high = (finite(v) for v in interval)
        audit.require(label + "/ordered_finite_interval", low <= high)
    if empty or not defined:
        audit.require(label + "/fully_undefined_case", valid == 0 and undefined == BOOTSTRAP)


def audit_ledger(value, expected, audit, label, columns=None):
    if columns is not None:
        exact(audit, label + "/exact_ledger_columns", value["columns"], columns)
    saved = csv_rows(value, audit)
    audit.equal(label + "/ledger_records", len(saved), len(expected))
    for i, (raw, row) in enumerate(zip(saved, expected, strict=True)):
        for key, wanted in row.items():
            if key in tick7.TIMES:
                got = stamp(raw[key]) if raw[key] not in ("", "NaT") else None
            elif key in tick7.NUMBERS or key in ("net_R_tick", "net_R_coarse", "tick_minus_coarse_R"):
                got = finite(raw[key], optional=True)
            elif key in tick7.BOOLS:
                got = tick7.boolean(raw[key])
            elif key == "side":
                if raw[key] not in ("-1", "1"):
                    raise ValueError("Canonical saved ledger side required")
                got = int(raw[key])
            else:
                got = raw[key]
            compare(audit, f"{label}/{i}/{key}", got, wanted)
    return expected


def prepared_audit(source, minutes):
    times = list(minutes); first, last = times[0], times[-1]
    return {"source_file": Path(source["path"]).name, "sha256": source["sha256"],
        "source_rows": len(minutes), "grid_minutes": (last-first)//60+1,
        "duplicate_equal_rows_removed": 0, "missing_minutes": (last-first)//60+1-len(minutes),
        "from_utc": old.iso(first), "to_exclusive_utc": old.iso(last+60),
        "gap_intervals": sum(b-a > 60 for a, b in zip(times, times[1:]))}


def dependency_pins(audit):
    for path, expected in DEPENDENCIES.items():
        audit.pin(relative_path(path), expected)


def frozen_document(study, name, audit):
    path = relative_path(str((study / (name + ".json")).relative_to(ROOT)))
    sidecar = relative_path(str((study / (name + ".sha256")).relative_to(ROOT)))
    audit.pin(sidecar)
    audit.pin(path, sidecar.read_text().strip())
    document = read_json(path)
    audit.safety(document, name)
    exact(audit, name + "/four_false_safety", document["safety"], dict.fromkeys(FLAGS, False))
    return document


def metadata(study, audit):
    declaration = frozen_document(study, "declaration", audit)
    result = frozen_document(study, "results", audit)
    dependency_pins(audit)
    for name, value in (("declaration", declaration), ("results", result)):
        for key, wanted in {"adaptive_round": 12, "config": CONFIG, "history": HISTORY,
                            "dates": list(DATES), "goal_achieved": False,
                            "historical_strategy_candidate": False, "live_candidate": False}.items():
            exact(audit, name + "/" + key, value[key], wanted)
    exact(audit, "declaration/stage", declaration["stage"], "short_tick_premeasurement_declaration")
    exact(audit, "result/stage", result["stage"], "short_tick_known_history_fidelity")
    for key, wanted in {"symbols": list(SYMBOLS), "families": list(FAMILIES), "modes": list(MODES),
                        "prices_or_signal_csv_decoded": False, "new_execution_ledgers_computed": False,
                        "new_model_or_scores_computed": False, "study_can_meet_sample_gate": False,
                        "bootstrap_is_discovery_test": False,
                        "user_target": {"profit_factor": 1.5, "completed_per_symbol_model": 1000,
                                        "active_heldout_days": 60}}.items():
        exact(audit, "declaration/" + key, declaration[key], wanted)
    audit.require("result/exact_symbol_set", set(result["symbols"]) == set(SYMBOLS))
    exact(audit, "result/declaration_hash", result["declaration_sha256"], digest(study / "declaration.json"))
    audit.require("declaration_before_result", run_time(declaration["run_utc"]) <= run_time(result["run_utc"]))
    for key in ("science_code_hashes", "lineage"):
        exact(audit, "result/copied_" + key, result[key], declaration[key])
    for key in ("actual_money_profit", "prospective_paper"):
        exact(audit, "result/" + key, result[key], "NOT TESTED")
    lineage = declaration["lineage"]
    for group in (declaration["science_code_hashes"], lineage["files"]):
        for name, sha in group.items():
            audit.pin(relative_path(name), sha)
    required = {"docs/SPIKE_SHORT_TICK_PROTOCOL.md", "scripts/run_spike_short_tick.py",
                "backend/tests/test_spike_short_tick.py", "scripts/run_spike_short_target.py"}
    audit.require("science/required_frozen_sources", required <= set(declaration["science_code_hashes"]))
    old_study = ROOT / "docs/spike_short_target_20261006"
    prior = {name: frozen_document(old_study, name, audit)
             for name in ("declaration", "selection", "results", "independent_audit")}
    for name in ("declaration", "selection", "results", "independent_audit"):
        field = "round11_" + ("audit" if name == "independent_audit" else name) + "_sha256"
        exact(audit, "lineage/" + field, lineage[field], digest(old_study / (name + ".json")))
    old_audit = prior["independent_audit"]
    compare(audit, "passing_round11_audit", old_audit, {"status": "PASS", "pass": True,
        "error_count": 0, "errors": [], "script_sha256": digest(relative_path("scripts/verify_spike_short_target.py")),
        "declaration_sha256": lineage["round11_declaration_sha256"],
        "selection_sha256": lineage["round11_selection_sha256"], "results_sha256": lineage["round11_results_sha256"]})
    selected, old_result = prior["selection"], prior["results"]
    for key in ("science_code_hashes", "lineage", "config"):
        exact(audit, "round11/result_snapshot_" + key, old_result[key], selected[key])
    exact(audit, "round11/history", old_result["history"], old.HISTORY)
    compare(audit, "round11/result_contract", old_result, {"stage": "short_target_known_history_evaluation",
        "adaptive_round": 11, "goal_achieved": False, "historical_strategy_candidate": False,
        "live_candidate": False, "declaration_sha256": lineage["round11_declaration_sha256"],
        "selection_sha256": lineage["round11_selection_sha256"]})
    exact(audit, "lineage/round11_science", lineage["round11_science_code_hashes"], selected["science_code_hashes"])
    audit.require("science/all_parent_sources_retained", set(selected["science_code_hashes"]) <= set(declaration["science_code_hashes"]))
    for key, sha in selected["science_code_hashes"].items():
        exact(audit, "science/parent_identity/" + key, declaration["science_code_hashes"][key], sha)
    tick = lineage["tick_lineage"]
    tick_study = ROOT / "docs/spike_tick_execution_20261005"
    tick_documents = {}
    for name in ("declaration", "tick_sources", "results", "independent_audit"):
        path = tick_study / (name + ".json")
        field = "round7_" + {"independent_audit": "audit", "tick_sources": "tick_sources"}.get(name, name) + "_sha256"
        audit.pin(path, tick[field]); tick_documents[name] = read_json(path)
        audit.safety(tick_documents[name], "round7/" + name)
    tick_audit = tick_documents["independent_audit"]
    compare(audit, "round7/audit_pass", tick_audit, {"passed": True, "errors": [],
        "stage": "independent_primary_tick_execution_audit"})
    audit.require("round7/source_count", len(tick_audit["sources"]) == 24)
    audit.pin(relative_path(tick_audit["verifier_file"]), tick_audit["verifier_sha256"])
    exact(audit, "round7/source_snapshot", tick["sources"], tick_documents["tick_sources"]["sources"])
    exact(audit, "round7/shared_ancestor", tick["round6_selection_sha256"], selected["lineage"]["round6_selection_sha256"])
    for name, sha in tick["inherited_files"].items():
        audit.pin(relative_path(name), sha)
    exact(audit, "lineage/m1_symbol_set", sorted(lineage["m1_sources"]), sorted(SYMBOLS))
    exact(audit, "lineage/signals_symbol_set", sorted(lineage["sampled_signals"]), sorted(SYMBOLS))
    identities = set()
    for symbol in SYMBOLS:
        source = selected["lineage"]["sources"][symbol]["fresh"]
        exact(audit, symbol + "/m1_source_identity", lineage["m1_sources"][symbol], source)
        cohort = old_result["symbols"][symbol]["cohorts"]["later180_known_history_primary"]
        exact(audit, symbol + "/issuance_source_identity", cohort["source"], source)
        expected_keys = {f"{family}_{mode}" for family in FAMILIES for mode in MODES}
        exact(audit, symbol + "/six_stream_keys", sorted(lineage["sampled_signals"][symbol]), sorted(expected_keys))
        exact(audit, symbol + "/tick_dates", sorted(tick["sources"][symbol]), sorted(DATES))
        for key, identity in lineage["sampled_signals"][symbol].items():
            family, mode = key.rsplit("_", 1); saved = cohort["models"][key]
            candidate = selected["symbols"][symbol]["models"].get(key)
            model = candidate["final_model"] if candidate else None
            dev = candidate["development_eligible"] if candidate else False
            wanted = {"signals": saved["artifacts"]["signals"], "family": family, "mode": mode,
                "model_sha256": canonical_hash(model) if model else None, "development_eligible": dev,
                "reference_only": family != "SHORT44", "round11_selected_model": saved["selected_model"],
                "training_end": model["training_end"] if model else None,
                "frozen_score_threshold": model["estimator"]["threshold"] if model and model["estimator"] else None}
            exact(audit, symbol + "/" + key + "/identity", identity, wanted)
            compare(audit, symbol + "/" + key + "/old_report", saved, {"family": family, "mode": mode,
                "model_sha256": wanted["model_sha256"], "development_eligible": dev,
                "reference_only": family != "SHORT44", "historical_candidate": False, "config": COARSE_CONFIG})
            descriptor = identity["signals"]
            exact(audit, "six_column_saved_signal_descriptor", descriptor["columns"], SIGNAL_COLUMNS)
            artifact(descriptor, audit)
            audit.require("unique_original_issuance_path", descriptor["path"] not in identities)
            identities.add(descriptor["path"])
            if model:
                audit.require("all_twelve_days_after_frozen_fit", stamp(model["training_end"]) < tick7.date_bounds(DATES[0])[0])
    audit.equal("twelve_distinct_saved_streams", len(identities), 12)
    return declaration, result


def copied_inputs(descriptor, expected, audit, label, *, tick=False, bounds=None):
    path = artifact(descriptor, audit)
    columns = ["quote_time", "quote"] if tick else ["minute_open", "open", "high", "low", "close"]
    exact(audit, label + "/descriptor_columns", descriptor["columns"], columns[1:])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        exact(audit, label + "/copied_csv_schema", reader.fieldnames, columns)
        count, previous = 0, None
        for raw in reader:
            audit.require(label + "/rectangular_copy_rows", set(raw) == set(columns) and all(v is not None for v in raw.values()))
            t = stamp(raw[columns[0]])
            audit.require(label + "/sorted_unique", previous is None or previous < t)
            if tick or bounds is None:
                audit.require(label + "/original_quote_membership", t in expected)
            else:
                audit.require(label + "/dense_minute_grid_membership", bounds[0] <= t < bounds[1] and t % 60 == 0)
            if not tick and t not in expected:
                audit.require(label + "/missing_M1_remains_unknown", all(raw[k] == "" for k in columns[1:]))
            else:
                got = finite(raw["quote"]) if tick else tuple(finite(raw[k]) for k in columns[1:])
                compare(audit, label + "/" + str(t), got, expected[t])
            previous, count = t, count + 1
    wanted_rows = len(expected) if tick or bounds is None else (bounds[1]-bounds[0])//60
    audit.equal(label + "/all_original_rows_copied", count, wanted_rows)
    audit.equal(label + "/descriptor_rows", descriptor["rows"], count)


def audit_metrics(saved, rows, audit, label, date=None):
    computed = point_metrics(rows, [date] if date else DATES)
    if date:
        computed.pop("span_calendar_days")
    compare(audit, label + "/point_metrics", saved, computed)
    if date:
        audit.require(label + "/per_day_no_bootstrap", not any(k in saved for k in
            ("mean_net_R_ci95", "profit_factor_ci95", "bootstrap_repeats", "bootstrap_seed")))
    else:
        compare(audit, label + "/inference_contract", saved, {
            "calendar_basis": "twelve scheduled observed days; no unsampled calendar padding",
            "no_contiguous_weekly_claim": True, "scheduled_observed_day_clusters": 12,
            "inference_kind": "descriptive_scheduled_day_bootstrap_not_discovery", "unknowns_are_not_zero": True})
        ci_contract(saved, audit, label + "/mean_CI", "mean_net_R_ci95", "mean_valid_replicates",
                    "mean_undefined_replicates", empty=not computed["completed"])
        ci_contract(saved, audit, label + "/PF_CI", "profit_factor_ci95", "pf_valid_replicates",
                    "pf_undefined_replicates", defined=bool(computed["losses"]))
    return computed


def unknown(rows, counts):
    censored = sum(r["censored"] for r in rows)
    return {"censored": censored, "invalid_uncensored": len(rows)-censored-len(known(rows)),
            "missing_entry": counts["missing_entry"]}


def audit_matched(saved, tick, coarse, ta, ca, audit, label, *, aggregate=False):
    expected, point = matched_known(tick, coarse)
    for row in expected:
        row["tick_minus_coarse_R"] = row.pop("delta")
    audit_ledger(saved["ledger"], expected, audit, label, MATCHED_COLUMNS)
    a, b = unknown(tick, ta), unknown(coarse, ca)
    point.update(tick_unknown_counts=a, coarse_unknown_counts=b,
        policy_unknowns_present=any(v for record in (a, b) for v in record.values()),
        descriptive_only=True,
        interpretation="matched known outcomes; entry stop and expiry quotes all change, not stop-only causality",
        inference_status="FINITE_DRAW_DESCRIPTIVE_ONLY" if aggregate else "NOT_REQUESTED_PER_DAY_POINT_SUMMARY")
    compare(audit, label + "/point", saved["metrics"], point)
    if aggregate:
        ci_contract(saved["metrics"], audit, label + "/CI", "difference_ci95", "valid_replicates",
                    "undefined_replicates", empty=not expected)
    else:
        compare(audit, label + "/no_per_day_bootstrap", saved["metrics"], {"difference_ci95": [None, None],
            "valid_replicates": 0, "undefined_replicates": 0, "bootstrap_repeats": 0, "bootstrap_seed": None})


def audit_counts(saved, expected, audit, label, *, tick=False, aggregate=False):
    compare(audit, label + "/counters", saved, {k: expected[k] for k in TOTAL_COUNTERS})
    if aggregate:
        exact(audit, label + "/exact_total_fields", sorted(saved), sorted(TOTAL_COUNTERS))
    else:
        if tick:
            compare(audit, label + "/tick_semantics", saved, {"missing_path": expected["missing_path"],
                "config": TICK_CONFIG, "purge_minutes": PURGE, "take_profit": None,
                "safety": dict.fromkeys(FLAGS, False), "fill_interpretation": "quoteproxy_notbrokerfills",
                "entry_rule": "first_observed_quote_strictly_after_nominal_order_time",
                "stop_rule": "next_observed_quote_strictly_after_trigger",
                "expiry_rule": "first_observed_quote_strictly_after_nominal_expiry",
                "expiry_anchor": "nominal_entry_time_not_actual_entry_time",
                "deadline_priority": "stop_trigger_at_expiry_precedes_timeout", "partition_exit_allowance_seconds": 1})
        else:
            old.audit_counts(saved, expected, audit, label, COARSE_CONFIG)
    audit.equal(label + "/filled_identity", saved["filled"], saved["completed"]+saved["censored"])
    audit.equal(label + "/issued_identity", saved["issued"], sum(saved[k] for k in
        ("filled", "missing_entry", "outside_partition", "purged", "overlap_skipped")))


def audit_reference(saved, left, right, lc, rc, audit, label):
    a, b = unknown(left, lc), unknown(right, rc)
    compare(audit, label + "/point", saved, {"mean_R_difference": reference_difference(left, right),
        "model_unknown_counts": a, "reference_unknown_counts": b,
        "policy_unknowns_present": any(v for record in (a, b) for v in record.values()),
        "descriptive_only": True,
        "comparison": "same scheduled dates, different saved issuance exposure; per-completed-path means"})
    ci_contract(saved, audit, label + "/CI", "difference_ci95", "valid_replicates", "undefined_replicates",
                empty=not known(left) or not known(right))


def minute_reconciliation(ticks, minutes, date, audit):
    expected = tick7.reconcile(ticks, minutes, date, audit)
    # Quote distances have float units even when max() selects its integer-zero seed.
    distance = expected["max_abs_difference"]
    expected["max_abs_difference"] = None if distance is None else float(distance)
    return expected


def verify(study, audit):
    """Historical entry point; caller authorizes this only after frozen evaluation."""
    for flag in FLAGS:
        audit.require("offline_environment/" + flag, os.getenv(flag, "false").lower() in ("false", "0", "no", "off"))
    study = relative_path(str(Path(study).absolute().relative_to(ROOT)))
    declaration, result = metadata(study, audit)
    lineage = declaration["lineage"]
    record_counts = {"tick_policy_records": 0, "coarse_policy_records": 0, "matched_known_records": 0}
    for symbol in SYMBOLS:
        report = result["symbols"][symbol]
        source = lineage["m1_sources"][symbol]
        minutes = tick7.read_m1(relative_path(source["path"]))
        audit.equal(symbol + "/m1_source_rows", len(minutes), source["rows"])
        compare(audit, symbol + "/m1_prepared", report["m1_audit"], prepared_audit(source, minutes))
        audit.equal(symbol + "/m1_first", min(minutes), source["first_epoch"])
        audit.equal(symbol + "/m1_last", max(minutes), source["last_epoch"])
        audit.equal(symbol + "/m1_gap_count", prepared_audit(source, minutes)["missing_minutes"], source["declared_missing_minutes"])
        exact(audit, symbol + "/twelve_input_dates", sorted(report["inputs"]), sorted(DATES))
        contexts = {}
        for date in DATES:
            start, end = tick7.date_bounds(date); values = report["inputs"][date]
            tick_source = lineage["tick_lineage"]["sources"][symbol][date]
            exact(audit, symbol + "/" + date + "/tick_source_identity", values["tick_source"], tick_source)
            ticks = tick7.float_quotes(tick7.read_ticks(relative_path(tick_source["clean_file"])))
            audit.equal("normalized_tick_rows", len(ticks), tick_source["rows"])
            audit.require("normalized_tick_day_bounds", bool(ticks) and min(ticks) >= start and max(ticks) < end)
            audit.equal("normalized_missing_seconds", 86400-len(ticks), tick_source["missing_seconds"])
            exact(audit, "normalized_gap_flag", tick_source["gap_free"], len(ticks) == 86400)
            day_minutes = {t: q for t, q in minutes.items() if start <= t < end}
            compare(audit, "reconciliation/" + symbol + "/" + date, values["reconciliation"],
                    minute_reconciliation(ticks, day_minutes, date, audit))
            copied_inputs(values["ticks"], ticks, audit, symbol + "/" + date + "/ticks", tick=True)
            copied_inputs(values["m1"], day_minutes, audit, symbol + "/" + date + "/m1", bounds=(start, end))
            contexts[date] = ticks, day_minutes
        identities = lineage["sampled_signals"][symbol]
        exact(audit, symbol + "/six_result_streams", sorted(report["models"]), sorted(identities))
        streams = {}
        for key, identity in identities.items():
            row = report["models"][key]; label = symbol + "/" + key
            signals = signals_from_rows(csv_rows(identity["signals"], audit), symbol, key, identity)
            compare(audit, label + "/identity_and_refusal", row, {k: identity[k] for k in
                ("family", "mode", "model_sha256", "development_eligible", "reference_only", "round11_selected_model")})
            compare(audit, label + "/refusal_gates", row, {"selected_model": False, "historical_candidate": False,
                "supports_expected_pf_1_5": False, "user_target_observed": False,
                "conditional_economic_gates_pass": False, "actual_money_profit": "NOT TESTED",
                "prospective_validation": "NOT TESTED"})
            reasons = ["fixed_execution_diagnostic_not_candidate", "max564_paths_and12_days_below1000_and60",
                       "reused_adaptive_known_history", "measured_costs_fills_not_tested"]
            if not identity["development_eligible"] and identity["family"] == "SHORT44":
                reasons.append("round11_development_rejected")
            exact(audit, label + "/rejection_reasons", row["rejection_reasons"], reasons)
            exact(audit, label + "/ordered_twelve_day_reports", [d["date"] for d in row["days"]], list(DATES))
            tick_all, coarse_all, tas, cas = [], [], [], []
            for day in row["days"]:
                date = day["date"]; start, end = tick7.date_bounds(date); sublabel = label + "/" + date
                day_signals = scheduled(signals, date)
                copied = signals_from_rows(csv_rows(day["signals"], audit), symbol, key, identity)
                exact(audit, sublabel + "/saved_issuance_exact_membership_identity", copied, day_signals)
                audit.require(sublabel + "/half_hour_opportunity_bound", len(day_signals) <= 48)
                ticks, day_minutes = contexts[date]
                tick, ta = tick_replay(ticks, day_signals, start, end)
                coarse, ca = old.replay(day_minutes, day_signals, start, end, COARSE_CONFIG)
                for proxy, paths, counts in (("tick", tick, ta), ("coarse", coarse, ca)):
                    audit_counts(day[proxy]["audit"], counts, audit, sublabel + "/" + proxy, tick=proxy == "tick")
                    audit_ledger(day[proxy]["ledger"], paths, audit, sublabel + "/" + proxy,
                                 TICK_COLUMNS if proxy == "tick" else COARSE_COLUMNS)
                    audit_metrics(day[proxy]["metrics"], paths, audit, sublabel + "/" + proxy, date)
                audit_matched(day["matched_known"], tick, coarse, ta, ca, audit, sublabel + "/matched")
                audit.equal(sublabel + "/common_purge", ta["purged"], ca["purged"])
                tick_all.extend(tick); coarse_all.extend(coarse); tas.append(ta); cas.append(ca)
            totals = [{k: sum(c[k] for c in collection) for k in TOTAL_COUNTERS} for collection in (tas, cas)]
            for i, (proxy, paths) in enumerate((("tick", tick_all), ("coarse", coarse_all))):
                audit_counts(row[proxy]["audit_totals"], totals[i], audit, label + "/" + proxy, aggregate=True)
                audit_ledger(row[proxy]["ledger"], paths, audit, label + "/" + proxy + "/all",
                             TICK_COLUMNS if proxy == "tick" else COARSE_COLUMNS)
                metrics = audit_metrics(row[proxy]["metrics"], paths, audit, label + "/" + proxy)
                audit.require(label + "/" + proxy + "/cannot_pass_sample_gate", metrics["completed"] <= 564 and metrics["active_days"] <= 12)
            audit_matched(row["matched_known"], tick_all, coarse_all, *totals, audit, label + "/matched_all", aggregate=True)
            record_counts["tick_policy_records"] += len(tick_all)
            record_counts["coarse_policy_records"] += len(coarse_all)
            record_counts["matched_known_records"] += len(matched_known(tick_all, coarse_all)[0])
            streams[key] = (tick_all, coarse_all, totals)
        for key, (tick, coarse, counts) in streams.items():
            row = report["models"][key]; family, mode = key.rsplit("_", 1)
            for field, reference in (("clock_comparisons", "CLOCK_" + mode), ("long_comparisons", "LONG44_" + mode)):
                required = family != "CLOCK" if field == "clock_comparisons" else family == "SHORT44"
                audit.require(symbol + "/" + key + "/comparison_presence/" + field, (field in row) is required)
                if required:
                    exact(audit, "reference_proxy_keys", sorted(row[field]), ["coarse", "tick"])
                    ref = streams[reference]
                    for i, paths in enumerate((tick, coarse)):
                        proxy = ("tick", "coarse")[i]
                        audit_reference(row[field][proxy], paths, ref[i], counts[i], ref[2][i], audit,
                                        symbol + "/" + key + "/" + field + "/" + proxy)
    audit.pin(study / "metrics.csv")
    audit_metrics_csv(study / "metrics.csv", result, audit)
    for path, sha in list(audit.inputs.items()):
        audit.equal("unchanged_after_verification/" + path, digest(Path(path)), sha)
    return {**record_counts, "tick_sources": 24, "m1_sources": 2, "scheduled_observed_days": 12,
            "saved_issuance_streams": 12, "declaration_sha256": digest(study / "declaration.json"),
            "results_sha256": digest(study / "results.json")}


def audit_metrics_csv(path, result, audit):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(s, key, proxy): row[proxy]["metrics"] for s, value in result["symbols"].items()
                for key, row in value["models"].items() for proxy in ("tick", "coarse")}
    audit.equal("metrics_csv/twenty_four_rows", len(rows), 24)
    seen = set()
    for raw in rows:
        audit.require("metrics_csv/rectangular_rows", None not in raw and all(v is not None for v in raw.values()))
        key = (raw["symbol"], raw["model"], raw["proxy"])
        audit.require("metrics_csv/unique_identity", key in expected and key not in seen); seen.add(key)
        for field, wanted in expected[key].items():
            value = raw[field]
            if wanted is None:
                got = None if value == "" else value
            elif isinstance(wanted, bool):
                got = tick7.boolean(value)
            elif isinstance(wanted, int):
                got = int(value)
            elif isinstance(wanted, float):
                got = finite(value)
            elif isinstance(wanted, list):
                # pandas preserves Python list repr, including None in undefined intervals.
                import ast
                got = ast.literal_eval(value)
            else:
                got = value
            compare(audit, "metrics_csv/" + "/".join(key) + "/" + field, got, wanted)
    exact(audit, "metrics_csv/complete_identities", sorted(seen), sorted(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refuse audit output overwrite")
    audit = Audit()
    evidence = {}
    try:
        evidence = verify(Path(args.study), audit)
    except Exception as exc:
        audit.failures.append({"check": "exception", "type": type(exc).__name__, "message": str(exc)})
    helper_hashes = {}
    for name in DEPENDENCIES:
        try:
            helper_hashes[name] = digest(relative_path(name))
        except Exception as exc:
            helper_hashes[name] = None
            audit.failures.append({"check": "report_helper_identity/" + name, "type": type(exc).__name__, "message": str(exc)})
    report = {"stage": "independent_fixed_short_tick_fidelity_audit", "run_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if not audit.failures else "FAIL", "pass": not audit.failures,
              "checks": audit.checks, "script_sha256": digest(Path(__file__)),
              "helper_sha256": helper_hashes,
              **evidence,
              "errors": len(audit.failures), "failures": audit.failures,
              "safety_values_checked": audit.safety_values_checked,
              "max_numeric_error": audit.max_numeric_error, "input_hashes": audit.inputs,
              "safety": dict.fromkeys(FLAGS, False),
              "scope": "frozen hashes and saved issuance identity; scalar SHORT1 tick/M1 paths; point metrics; matched and control means; CI schema/counts only; refusal gates",
              "exclusions": ["raw_wire_renormalization", "full44_feature_regeneration", "complete_eligible_universe_regeneration",
                  "model_refitting_or_predictions", "bootstrap_draw_or_CI_regeneration", "inferential_validity",
                  "new_price_or_label_holdout", "independence", "broker_fills_measured_costs_or_money_profit"]}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "checks", "errors", "safety_values_checked")}, sort_keys=True))
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
