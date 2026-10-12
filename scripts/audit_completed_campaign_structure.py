"""Read-only structural closeout audit, separate from strategy numerical replay.

Outputs stay in work/. Earlier development bytes are anchored to the original
19-cycle receipt and the separately executed delta receipt. No result is tuned.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import csv
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_campaign.state import SAFETY, digest, encoded, guard, read_json, validate_state

DIRECTORY = ROOT / "data/single_frame_campaign_20261011"
OUTPUT = ROOT / "work/completed_campaign_audit_20261012"
BASELINE_SHA = "23c0a4e2a8ac5713a2f30c7ae9479b76817269283c2e6403c0a450eef6f1ee61"
RESULT_SAFETY = {**SAFETY, "ready_for_live":False, "live_allowed":False, "opened_trades":False}


def exact_false_schema(value, expected):
    return isinstance(value, dict) and set(value) == set(expected) and all(v is False for v in value.values())


def utc():
    return datetime.now(timezone.utc).isoformat()


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def reduce_family(path):
    """Stream canonical pretty JSON and omit only three bulky ledger arrays.

    Their entry counts are preserved; the whole compressed source is separately
    hashed. This neither reconstructs numerical backtests nor recomputes fills.
    """
    output, ledgers = [], []
    skipping, counts, depth = None, {}, 0
    current_trade = None
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if skipping:
                if line.startswith("      ]"):
                    output[-1] = '      "' + skipping + '": []' + ("," if line.rstrip().endswith(",") else "") + "\n"
                    skipping = None
                    continue
                if line == "        {\n":
                    counts["entries"] += 1
                    if counts["field"] == "trades":
                        current_trade = {}
                elif current_trade is not None:
                    if line.startswith("        }"):
                        counts["first_signal_epoch"] = min(counts.get("first_signal_epoch", current_trade["signal_epoch"]), current_trade["signal_epoch"])
                        counts["last_exit_epoch"] = max(counts.get("last_exit_epoch", current_trade["exit_epoch"]), current_trade["exit_epoch"])
                        counts["chronology_violations"] += int(not (current_trade["signal_epoch"] == current_trade["entry_epoch"] <= current_trade["exit_epoch"] <= current_trade["planned_end_epoch"]))
                        counts["adjacent_bar_index_violations"] += int(current_trade["entry_bar_index"] != current_trade["signal_bar_index"] + 1)
                        current_trade = None
                    elif line.startswith("          ") and not line.startswith("           "):
                        key, value = line.strip().rstrip(",").split(":", 1)
                        key = json.loads(key)
                        if key in {"signal_epoch", "entry_epoch", "exit_epoch", "planned_end_epoch", "signal_bar_index", "entry_bar_index"}:
                            current_trade[key] = json.loads(value)
                if counts["field"] == "dispositions" and line.startswith('          "status": '):
                    counts["statuses"][json.loads(line.strip().split(":", 1)[1].rstrip(","))] += 1
                continue
            field = None
            if line.startswith('      "'):
                field = line.split('"', 2)[1]
            if field in {"dispositions", "equity", "trades"}:
                counts = {"field": field, "entries": 0, "statuses": Counter(),
                          "chronology_violations": 0, "adjacent_bar_index_violations": 0}
                ledgers.append(counts)
                if line.rstrip().endswith("[],") or line.rstrip().endswith("[]"):
                    output.append(line)
                elif line.rstrip().endswith("["):
                    output.append('      "' + field + '": [],\n')
                    skipping = field
                else:
                    raise ValueError("Unexpected canonical ledger encoding")
            else:
                output.append(line)
    if skipping:
        raise ValueError("Unclosed final ledger array")
    family = json.loads("".join(output))
    if len(ledgers) != 3 * len(family["results"]):
        raise ValueError("Ledger scope does not match final results")
    grouped = [dict((r["field"], r) for r in ledgers[i:i+3]) for i in range(0, len(ledgers), 3)]
    return family, grouped


def expected_gates(row):
    m, u, refs, repro, costs = (row[k] for k in ("metrics", "uncertainty", "reference", "reproducibility", "costs"))
    pf, mean_ci, pf_ci = m["profit_factor"], u.get("mean_net_R_ci95"), u.get("profit_factor_ci95")
    bh, native = refs.get("buy_and_hold"), refs.get("native_direction_hold")
    verified = all(costs[k] for k in ("mapping_verified", "historical_costs_verified", "execution_verified")) and bool(costs["evidence"])
    folds = row["development_walk_forward"]
    return {
        "final_partition": row["partition"] == "final",
        "minimum_100_completed": m["completed_trades"] >= 100,
        "maximum_drawdown_20pct": m["max_drawdown"] <= .20,
        "profit_factor_1_10": pf is not None and math.isfinite(pf) and pf >= 1.10,
        "positive_net_profit": m["net_profit"] > 0,
        "outperforms_buy_and_hold": bool(bh and m["net_return"] > bh["net_return"]),
        "outperforms_native_reference": bool(native and m["net_return"] > native["net_return"]),
        "data_quality": bool(row["data_quality"]["ok"] and row["requested_data_quality_ok"] and not row["unknown_paths"]),
        "future_leak_checked": bool(repro["signal_causality_verified"]),
        "reproducible_real_source": bool(repro["deterministic"] and repro["raw_source_hash_verified"] and row["provenance"].get("real_deriv_data", False)),
        "actual_contract_costs_verified": bool(verified),
        "original_tick_execution_verified": bool(row["execution_tick_replayed"] and costs["execution_verified"] and not row["unknown_paths"]),
        "statistical_uncertainty": bool(mean_ci and pf_ci and mean_ci[0] > 0 and pf_ci[0] > 1),
        "multiplicity_adjusted": bool(u.get("multiplicity_adjusted") and u.get("holm_p_value", 1) <= .05),
        "all_live_gates_false": exact_false_schema(row["safety"], RESULT_SAFETY),
        "development_walk_forward_stability": len(folds) == 3 and all(f.get("status") == "TESTED" and f["metrics"]["net_profit"] > 0 and f["metrics"]["profit_factor"] is not None and f["metrics"]["profit_factor"] >= 1.10 for f in folds),
    }


def main():
    started, clock = utc(), time.monotonic()
    guard()
    errors, checks = [], Counter()

    def check(condition, label, context=None):
        checks[label] += 1
        if not condition:
            errors.append({"check": label, "context": context})
            print(json.dumps({"discrepancy": errors[-1]}, ensure_ascii=False), flush=True)

    def hashed(path, pin, label):
        actual = digest(path)
        check(actual == pin, label, str(path))
        return actual

    state = read_json(DIRECTORY / "state.json")
    validate_state(state)
    state_pin = digest(DIRECTORY / "state.json")
    declaration = read_json(DIRECTORY / "declaration.json")
    hashed(DIRECTORY / "declaration.json", state["declaration_sha256"], "frozen_declaration_hash")
    check(state["id"] == "single-frame-8af790ef59a5" and state["status"] == "COMPLETED" and state["completed_cycles"] == 32 and state["final_accessed"] is True, "actual_completed_campaign_identity")
    request = ROOT / "docs/single_frame_campaign_20261011/REQUEST.ar.md"
    prep = read_json(ROOT / "work/final_requirements_checklist_20261011.json")
    hashed(request, prep["request_sha256"], "authoritative_request_unchanged")
    baseline_path = ROOT / "docs/single_frame_campaign_20261011/progress_audit_20261011.json"
    hashed(baseline_path, BASELINE_SHA, "trusted_nineteen_cycle_audit")
    baseline = read_json(baseline_path)
    delta_path = ROOT / "work/progress_delta_completed_20261012.json"
    delta = read_json(delta_path)
    check(delta["audit_status"] == "PASS" and not delta["errors"] and delta["counts"]["through_completed_cycle"] == 32 and delta["trusted_baseline"]["sha256"] == BASELINE_SHA, "executed_delta_audit_pass_through_32")
    initial_failed_path = ROOT/"work/progress_delta_completed_20261012_initial_counter_mismatch.json"
    initial_failed = read_json(initial_failed_path)
    check(initial_failed["audit_status"] == "FAIL" and len(initial_failed["errors"]) == 13 and all(e["check"] == "update_run_source_collector_safety" for e in initial_failed["errors"]) and {e["context"] for e in initial_failed["errors"]} == set(range(20,33)), "first_counter_contract_mismatch_evidence_preserved")
    counter_test_path = ROOT/"work/progress_delta_counter_contract_20261012.json"
    counter_test = read_json(counter_test_path)
    check(counter_test["status"] == "PASS" and len(counter_test["cases"]) == 6 and [r["accepted"] for r in counter_test["cases"]] == [True,False,False,False,False,False] and counter_test["original_failed_receipt_sha256"] == digest(initial_failed_path), "focused_integer_zero_counter_contract_test")
    compatibility = delta["summary_receipts"]["compatible_receipt_reuse_evidence"]
    check(compatibility["summary_function_ast_identical"] is True and compatibility["all_artifact_bytes_rehashed_again"] is True and compatibility["legacy_cache_misses_allowed"] is False and delta["summary_receipts"]["diagnostics"].get("summary_receipts_reused") == 15500 and not delta["summary_receipts"]["diagnostics"].get("new_artifact_summaries_streamed",0), "honest_delta_reused_summaries_not_new_computations")
    targets, frames = set(declaration["symbol_catalogue"]), set(declaration["timeframes"])
    cells = {f"{s}/{f}" for s in targets for f in frames}
    family_names = {s["family"] for s in declaration["candidate_rounds"][0]}
    groups = {f"{cell}/{family}" for cell in cells for family in family_names}
    winners, seen, cycles = {}, set(), []
    attempts = fresh = reused = aliases = 0
    last_start = last_end = None
    for number in range(1, 33):
        path = DIRECTORY / "cycles" / f"{number:02d}.json"
        cycle = read_json(path)
        check(cycle["number"] == number and cycle["status"] == "COMPLETED", "actual_journal_completed", number)
        check(all(cycle[k] == v for k, v in state["cycles"][number-1].items()), "journal_matches_completed_state_record", number)
        check(set(cycle["coverage"]) == cells and all(r["status"] == "TESTED" for r in cycle["coverage"].values()), "all_hundred_cells_tested", number)
        row_groups = [f'{r["symbol"]}/{r["timeframe"]}/{r["family"]}' for r in cycle["results"]]
        check(len(row_groups) == 1300 and set(row_groups) == groups, "exact_thirteen_families_per_cell", number)
        start, end, due = (epoch(cycle[k]) for k in ("started_at", "completed_at", "due_at"))
        expected_due = max(state["started_epoch"] + (number - 1) * 900, (last_start + 900) if last_start is not None else state["started_epoch"])
        check(abs(due - expected_due) < 1e-5 and start >= due and end >= start and (last_end is None or start >= last_end), "actual_noncompressed_cycle_chronology", number)
        if last_start is not None:
            check(start - last_start >= 900, "actual_minimum_900_second_start_gap", number)
        nf = nh = na = 0
        for row, group in zip(cycle["results"], row_groups):
            hit = row["artifact"]["path"] in seen
            check(type(row["cache_hit"]) is bool and row["cache_hit"] == hit, "actual_cache_hit_not_new_measurement", (number, group))
            seen.add(row["artifact"]["path"])
            nf += int(not hit)
            nh += int(hit)
            na += int(row["strategy_id"] != row["attempted_strategy_id"])
            old = winners.get(group)
            if old is None or (row["validation_score"] is not None and (old.get("validation_score") is None or row["validation_score"] > old["validation_score"])):
                winners[group] = row
        check(nf == cycle["tested_count"] and nh == cycle["cache_hits"], "exact_new_reused_cycle_counters", number)
        attempts += len(cycle["results"]); fresh += nf; reused += nh; aliases += na
        cycles.append({"number": number, "started_at": cycle["started_at"], "completed_at": cycle["completed_at"], "start_gap_seconds": None if last_start is None else start-last_start, "duration_seconds": end-start, "new_measurements": nf, "cache_hits": nh, "journal_sha256": digest(path)})
        last_start, last_end = start, end
    check((attempts, fresh, reused, aliases, len(seen)) == (41600, 38400, 3200, 3200, 38400), "actual_full_campaign_counts")
    print(json.dumps({"stage":"all_32_journals_and_selection_reconstruction","errors":len(errors)}),flush=True)
    check(winners == state["winners"], "winners_reconstructed_from_development_only")
    selection_path = DIRECTORY / "selection_frozen.json"
    selected = read_json(selection_path)
    hashed(selection_path, state["selection_sha256"], "frozen_selection_hash")
    check(selected["source"] == state["declaration_sha256"] and selected["selection"] == winners and selected["final_data_accessed"] is False, "selection_sealed_from_validation_before_final")
    check(set(selected["selection"]) == groups and set(selected["walk_forward"]) == groups, "selection_and_wf_cover_all_1300_groups")
    guard_epoch = state["started_epoch"] + 32*900
    check(epoch(state["deadline_at"]) == guard_epoch and epoch(selected["frozen_at"]) >= guard_epoch and epoch(selected["frozen_at"]) >= last_end, "full_eight_hour_guard_and_all_cycles_before_seal")
    code = {}
    for name, pin in declaration["source_pins"].items():
        code[name] = {"declared": pin, "current": hashed(ROOT/"research_campaign"/name, pin, "current_frozen_code_hash"), "archived": hashed(DIRECTORY/"frozen_source"/name, pin, "archived_frozen_code_hash")}
    runtime = {"python":platform.python_version(), "numpy":importlib.metadata.version("numpy"), "pandas":importlib.metadata.version("pandas")}
    check(runtime == declaration["runtime"], "frozen_numerical_runtime_unchanged")
    hashed(declaration["source_manifest"], declaration["source_manifest_sha256"], "frozen_initial_manifest_hash")
    sources = []
    for symbol, row in declaration["datasets"].items():
        for file_key, hash_key in (("file", "sha256"), ("requested_file", "requested_sha256"), ("manifest_file", "manifest_sha256")):
            hashed(row[file_key], row[hash_key], "frozen_original_data_and_manifest_hash")
        split = declaration["splits"][symbol]
        span = split["end"] - split["start"]
        check(split["train_end"] == split["start"] + span*3//5 and split["validation_end"] == split["start"] + span*4//5 and split["fractions"] == [.6,.2,.2], "exact_per_symbol_60_20_20_utc_split", symbol)
        sources.append({"symbol": symbol, "api_symbol": row["api_symbol"], "requested_file": row["requested_file"], "requested_sha256": row["requested_sha256"], "original_source_audit": baseline["source_reaudits"][symbol], "split": split})
    fold_artifacts = {}
    for group, folds in selected["walk_forward"].items():
        check(len(folds) == 3 and [f["label"] for f in folds] == ["wf1", "wf2", "wf3"] and all(f["status"] == "TESTED" for f in folds), "three_tested_development_folds_per_selected_group", group)
        for fold in folds:
            pointer = fold["artifact"]
            if pointer["path"] not in fold_artifacts:
                hashed(DIRECTORY/pointer["path"], pointer["sha256"], "saved_walk_forward_artifact_byte_hash")
                fold_artifacts[pointer["path"]] = pointer["sha256"]
    print(json.dumps({"stage":"all_frozen_sources_and_3900_fold_byte_hashes","errors":len(errors)}),flush=True)
    family_path = DIRECTORY/state["final_artifact"]["path"]
    hashed(family_path, state["final_artifact"]["sha256"], "whole_final_family_compressed_hash")
    family, ledgers = reduce_family(family_path)
    print(json.dumps({"stage":"streamed_final_family_and_ledger_counts","results":len(family["results"]),"errors":len(errors)}),flush=True)
    check(family["family_size"] == len(family["results"]) == len(family["original_measurements"]) == 1300 and family["selected_before_access"] is True and family["selection_sha256"] == state["selection_sha256"], "exact_sealed_final_family_scope")
    check(set(family["original_measurements"]) == groups and {r["selection_group"] for r in family["results"]} == groups, "final_all_selected_groups_once")
    gate_failures, outcomes, original_pins = Counter(), [], {}
    for row, ledger in zip(family["results"], ledgers):
        group = row["selection_group"]; winner = selected["selection"][group]
        check(row["strategy"] == winner["spec"] and row["symbol"] == winner["symbol"] and row["timeframe"] == winner["timeframe"] and row["selection_sha256"] == state["selection_sha256"], "final_policy_exactly_selected_spec", group)
        check(epoch(row["measured_at"]) >= epoch(selected["frozen_at"]) >= guard_epoch, "all_final_measurement_timestamps_after_seal_and_guard", group)
        split = declaration["splits"][row["symbol"]]
        check(row["start"] == split["validation_end"] and row["end"] == split["end"] and row["partition"] == "final", "same_untuned_final_calendar", group)
        check(row["costs"] == declaration["costs"] and row["development_walk_forward"] == selected["walk_forward"][group], "frozen_costs_and_fold_evidence_retained", group)
        expected = expected_gates(row)
        check(row["gates"] == expected and row["qualified"] == all(expected.values()) and row["rejection_reasons"] == [k for k,v in expected.items() if not v], "independently_applied_final_gate_logic", group)
        source = declaration["datasets"][row["symbol"]]
        provenance = row["provenance"]
        check(provenance["source_file"] == source["requested_file"] and provenance["source_sha256"] == source["requested_sha256"] and provenance["source_manifest"] == source["manifest_file"] and provenance["api_symbol"] == source["api_symbol"] and provenance["fresh_oos"] is False and provenance["actual_prefix_invariance_checked"] is True, "final_original_source_binding_and_exposure_disclosed", group)
        check(exact_false_schema(row["safety"], RESULT_SAFETY), "each_final_result_exact_seven_live_flags_false", group)
        check(ledger["trades"]["entries"] == row["metrics"]["completed_trades"] == ledger["dispositions"]["statuses"].get("COMPLETED", 0), "saved_trade_ledger_and_disposition_counts_match_metrics", group)
        check(ledger["trades"]["chronology_violations"] == ledger["trades"]["adjacent_bar_index_violations"] == 0, "saved_trade_close_to_next_open_chronology", group)
        if ledger["trades"]["entries"]:
            check(ledger["trades"]["first_signal_epoch"] >= row["start"] and ledger["trades"]["last_exit_epoch"] <= row["end"], "all_saved_trades_within_final_partition", group)
        pointer = family["original_measurements"][group]
        original_pins[group] = hashed(DIRECTORY/pointer["path"], pointer["sha256"], "original_final_measurement_byte_hash")
        gate_failures.update(k for k,v in row["gates"].items() if not v)
        outcomes.append({"group": group, "symbol": row["symbol"], "timeframe": row["timeframe"], "strategy_id": row["strategy_id"], "metrics": row["metrics"], "gates": row["gates"], "rejection_reasons": row["rejection_reasons"], "qualified": row["qualified"], "measured_at": row["measured_at"], "ledger_counts": {k:dict(v) for k,v in ledger.items()}})
    # Fresh Holm arithmetic over the recorded p-values; no bootstrap is rerun.
    last = 0.0
    for rank, row in enumerate(sorted(family["results"], key=lambda r:(float(r["uncertainty"].get("p_value",1)),r["id"]))):
        p = float(row["uncertainty"].get("p_value",1))
        check(math.isfinite(p) and 0 <= p <= 1, "final_p_value_valid", row["selection_group"])
        last = max(last,min(1.0,(1300-rank)*p))
        u = row["uncertainty"]
        check(u["holm_p_value"] == last and u["family_size"] == 1300 and u["multiplicity_adjusted"] is True, "joint_1300_holm_from_saved_p_values", row["selection_group"])
    qualified = [r for r in outcomes if r["qualified"]]
    check(len(qualified) == state["qualified_policies"] == 0, "no_qualified_policy_and_no_forced_winner")
    check(all(exact_false_schema(value, SAFETY) for value in (family["safety"], declaration["safety"], state["safety"])) and declaration["trailing_stop"] is False and declaration["live_execution"] is False, "global_exact_four_safety_and_no_trailing_stop")
    public = read_json(ROOT/"frontend/public/campaign/latest.json")
    check(public["campaign"]["status"] == "COMPLETED" and public["campaign"]["completed_cycles"] == 32 and public["campaign"]["final_accessed"] is True and exact_false_schema(public["safety"], SAFETY), "public_snapshot_actual_completion_and_safety")
    check(len(public["strategies"]) == 1300 and len(public["coverage"]) == 100 and len(public["symbols"]) == 10 and all(r["best_strategy_id"] is None and r["status"] == "NO_QUALIFIED_STRATEGY" and r["reason"] == "لا توجد استراتيجية مؤهلة لهذا المؤشر." for r in public["symbols"]), "public_all_ten_symbols_honest_no_qualified_strategy")
    csv_path = ROOT/"frontend/public/campaign/search.csv"
    csv_rows = 0
    csv_dispositions = Counter()
    csv_current_cycle, csv_journal_rows = None, {}
    final_by_group = {r["selection_group"]:r for r in family["results"]}
    with csv_path.open(newline="") as stream:
        for exported in csv.DictReader(stream):
            csv_rows += 1
            number = int(exported["cycle"])
            if number != csv_current_cycle:
                journal = read_json(DIRECTORY/"cycles"/f"{number:02d}.json")
                csv_journal_rows = {(r["symbol"],r["timeframe"],r["family"],r["attempted_strategy_id"]):r for r in journal["results"]}
                csv_current_cycle = number
            row = csv_journal_rows[(exported["symbol"],exported["timeframe"],exported["family"],exported["attempted_strategy_id"])]
            group = f'{row["symbol"]}/{row["timeframe"]}/{row["family"]}'
            for name in ("completed_trades", "profit_factor", "net_profit", "max_drawdown"):
                expected = row["validation_metrics"][name]
                check((exported[name] == "" if expected is None else float(exported[name]) == expected), "every_csv_numeric_metric_is_validation_not_final", (number,group,name))
            selected_row = row["strategy_id"] == winners[group]["strategy_id"]
            expected_disposition = "REJECTED_FINAL" if selected_row else "NOT_SELECTED"
            expected_reasons = final_by_group[group]["rejection_reasons"] if selected_row else ["NOT_BEST_VALIDATION_SCORE_WITHIN_FAMILY"]
            check(exported["selection_disposition"] == expected_disposition and exported["reasons"] == ";".join(expected_reasons), "every_csv_disposition_reason_partition_matched", (number,group))
            csv_dispositions[exported["selection_disposition"]] += 1
    check(csv_rows == attempts == 41600, "published_search_csv_all_actual_attempt_rows")
    export_qa_path = ROOT/"work/final_export_browser_qa_20261012.json"
    export_qa = read_json(export_qa_path)
    final_export = export_qa["selected_final_export"]
    hashed(final_export["path"], final_export["sha256"], "actual_browser_downloaded_selected_final_csv_hash")
    hashed(ROOT/"frontend/public/campaign/latest.json", final_export["snapshot_sha256"], "downloaded_final_csv_snapshot_reference_hash")
    exported_final_ids = set()
    snapshot_by_key = {(r["symbol"],r["timeframe"],r["settings"]["id"]):r for r in public["strategies"]}
    with Path(final_export["path"]).open(newline="",encoding="utf-8-sig") as stream:
        for downloaded in csv.DictReader(stream):
            settings = json.loads(downloaded["settings"])
            identifier = settings["id"]
            # Strategy IDs are shared across symbols/frames, so the coverage
            # identity is the full symbol/frame/family key.
            key = (downloaded["symbol"],downloaded["timeframe"],identifier)
            check(key in snapshot_by_key, "final_csv_unique_selected_symbol_frame_policy", key)
            strategy = snapshot_by_key[key]
            export_key = (strategy["symbol"],strategy["timeframe"],strategy["family"])
            check(export_key not in exported_final_ids and downloaded["scope"] == "oos" and settings == strategy["settings"] and downloaded["status"] == strategy["status"] and downloaded["reasons"] == " | ".join(strategy["reasons"]), "final_csv_explicit_scope_settings_status_reasons", export_key)
            exported_final_ids.add(export_key)
            for name in ("completed_trades","profit_factor","win_rate","net_profit","max_drawdown"):
                value = strategy["oos"][name]
                check((downloaded[name] == "" if value is None else float(downloaded[name]) == value), "final_csv_all_selected_oos_metrics_exact", (export_key,name))
    check(len(exported_final_ids) == final_export["rows"] == 1300, "downloaded_final_csv_all_1300_selected_policies")
    clarification = export_qa["ui_clarification"]
    hashed(ROOT/clarification["source_file"], clarification["source_sha256"], "ui_partition_clarification_current_source_hash")
    hashed(clarification["screenshot"], clarification["screenshot_sha256"], "ui_partition_clarification_actual_capture_hash")
    check(clarification["actual_dom_and_visual_verified"] is True and clarification["checks"]["frontend_tests_passed"] == 50 and clarification["checks"]["typecheck"] == clarification["checks"]["build"] == "PASS", "saved_actual_ui_partition_clarification_and_checks")
    browser_path = ROOT/"work/dashboard_final_browser_qa_20261012.json"
    browser = read_json(browser_path)
    check(browser["completed_cycles_displayed"] == 32 and browser["covered_cells"] == 100 and browser["displayed_selected_policies"] == 1300 and browser["displayed_unique_development_measurements"] == 38400 and browser["qualified_displayed"] == 0, "saved_actual_final_browser_scope")
    check(browser["lang"] == "ar" and browser["dir"] == "rtl" and browser["desktop"]["viewport_width"] == browser["desktop"]["document_scroll_width"] == 1280 and browser["mobile"]["viewport_width"] == browser["mobile"]["document_scroll_width"] == 390 and browser["viewport_restored"] is True, "saved_actual_desktop_and_mobile_rtl_evidence")
    for name, receipt in browser["files"].items():
        hashed(ROOT/"work"/name, receipt["sha256"], "saved_browser_capture_byte_hash")
    ci_path = ROOT/"work/campaign_publication/helper_ci_run.json"
    ci = read_json(ci_path)
    jobs = read_json(ROOT/"work/campaign_publication/helper_ci_jobs.json")
    actual_head = subprocess.run(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True,check=True,capture_output=True).stdout.strip()
    git_status = subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,check=True,capture_output=True).stdout.splitlines()
    check(ci["head_sha"] == actual_head == "1ec28d0e5c5e41c4cda3a052e3b7aa8a14bc822a" and ci["status"] == "completed" and ci["conclusion"] == "success" and ci["id"] == 38183355758, "saved_ci_success_exact_current_head")
    job_statuses = {j["name"]:j["conclusion"] for j in jobs["jobs"]}
    check(job_statuses.get("verify") == job_statuses.get("frontend") == "success" and job_statuses.get("historical-study") == "skipped", "saved_ci_job_scope_and_skipped_history_disclosed")
    safety_diagnosis_path = ROOT/"work/completed_campaign_safety_schema_diagnosis_20261012.json"
    safety_diagnosis = read_json(safety_diagnosis_path)
    safety_test_path = ROOT/"work/completed_campaign_safety_schema_tests_20261012.json"
    safety_test = read_json(safety_test_path)
    safety_initial_failed_path = ROOT/"work/completed_campaign_audit_20261012_initial_safety_schema_mismatch.json"
    hashed(safety_initial_failed_path, safety_diagnosis["initial_failed_audit_sha256"], "first_safety_schema_checker_failure_preserved")
    check(safety_diagnosis["all_other_15_gates_match"] is True and list(safety_diagnosis["gate_differences"]) == ["all_live_gates_false"] and safety_test["status"] == "PASS" and all(c["accepted"] is c["expected"] for c in safety_test["cases"]), "state_four_vs_engine_seven_exact_schema_contract_tests")
    hashed(DIRECTORY/"state.json", state_pin, "completed_state_unchanged_during_audit")
    hashed(selection_path, state["selection_sha256"], "selection_unchanged_during_audit")
    hashed(family_path, state["final_artifact"]["sha256"], "family_unchanged_during_audit")
    limitations = [
        "The original 19-cycle receipt anchors old development artifact hashes; this run does not reread those 20.4 GB. The separately executed delta receipt verifies the new development bytes and summaries through cycle 32.",
        "Saved final ledger counts, timing, gates and Holm arithmetic were checked. No signal generation, backtest path, bootstrap, cash-PnL formula or walk-forward numerical result was independently rerun by this audit. Root numerical replay has its own receipt and scope.",
        "All 13 strategy families were fixed before round 1; 32 parameter rounds do not prove a new strategy idea was invented each cycle.",
        "Historical CFD bid/ask, account mapping, financing, fills and full original-tick strategy execution remain unverified. Round-trip 0.10 ATR is an assumed scenario, not measured transaction costs.",
        "All ten frozen histories retain an internal missing minute and internal invalid/off-grid source row. Byte/source integrity preserves defects and does not make data quality pass.",
        "A requested year is not proof of the universal maximum history available. Adaptive history is not fresh prospective OOS; development walk-forward folds are selection-dependent.",
        "This report audits the 11 request sections structurally; existing browser/testing evidence is inherited with its stated scope. It is not a full user-request compliance certificate or an executable profitability claim.",
        "search.csv numeric completed_trades/profit_factor/net_profit/max_drawdown columns are validation metrics for all 41,600 attempt rows. For a selected policy, selection_disposition and reasons describe final gates. The CSV lacks an explicit metrics_partition column; its mixed semantics must not be interpreted as final performance. Selected aliases can repeat the same final disposition in several development-attempt rows.",
        "The CUA-downloaded selected-results CSV explicitly labels scope=oos and all 1,300 rows were independently matched here to the saved snapshot. The UI now explains raw search-CSV semantics; its changes were tested locally and are outside the frozen scientific core. CI is for the cited published HEAD, not any uncommitted UI edits.",
        "The summed 1,292,148 saved completed final trades overlap across policies and hypotheses. This is a ledger coverage count, not that many independent market events or an aggregate strategy sample.",
    ]
    section_evidence = []
    # Preserve every preparatory requirement and append current evidence; do not
    # silently upgrade requirements this structural audit cannot establish.
    overrides = {
        "2.1":("VERIFIED","32 سجلًا فعليًا؛ 100 خلية و 13 عائلة في كل دورة، وضبط مستقل لكل رمز وفريم."),
        "2.2":("VERIFIED","طُبقت البوابات النهائية على 1,300 سياسة، ولم يتأهل فريم؛ لا يُفرض فائز."),
        "3.1":("VERIFIED","13 عائلة معلنة اختبرت فعليًا بجميع الخلايا والدورات؛ لا يثبت ذلك صلاحية تداول أو ابتكار عائلة كل دورة."),
        "4.10":("PARTIAL","سجلات التطوير مرتبطة ببصمات ملفاتها؛ عُدّت جميع صفقات الأسرة النهائية وطابقت المقاييس والتصرفات. إعادة حساب البحث التطويري كله غير منفذة."),
        "4.12":("VERIFIED","3,900 طية تطوير محفوظة: ثلاث لكل سياسة من 1,300؛ طابقت جميع بصمات ملفاتها. هذه طيات تابعة لاختيار التطوير وليست OOS مستقلاً."),
        "4.13":("VERIFIED","أعيد بناء الاختيارات من validation فقط، وطابق selection_frozen؛ التثبيت بعد آخر دورة وحد الثماني ساعات، وجميع measured_at النهائية بعده."),
        "5.1":("VERIFIED","أعيد تطبيق بوابة 100 صفقة من المقاييس النهائية على جميع السياسات، وطابق عدد الصفقات سجلاتها الأصلية."),
        "5.2":("VERIFIED","أعيد تطبيق حد DD 20% على المقاييس النهائية؛ إعادة مسار رأس المال عددياً خارج هذا التدقيق."),
        "5.3":("VERIFIED","أعيد تطبيق PF≥1.10 على جميع المقاييس النهائية دون تخفيف الحد؛ إعادة الحساب العددي مستقلة عن هذه المراجعة."),
        "5.4":("PARTIAL","طابقت بوابة التفوق على الشراء والاحتفاظ والمرجع الأصلي؛ كلا المرجعين تحت فرضية تكاليف غير موثقة."),
        "5.7":("PARTIAL","تطابق المصدر والملفات محفوظ، وبوابة جودة البيانات تفشل لكل السياسات. إعادة حساب كامل بحث التطوير غير منفذة."),
        "5.8":("VERIFIED","مقاييس القبول النهائية من partition=final وحدوده المجمدة؛ لم تستخدم مقاييس التدريب بديلًا."),
        "6.1":("VERIFIED","32 دورة فعلية متصلة، 31 فاصل بدء ≥ 900 ثانية، وآخر إكمال واختيار نهائي بعد حد ثماني ساعات."),
        "6.2":("VERIFIED","كل دورة تحمل إيصال تحديث السوق؛ التقرير الأصلي وتدقيق delta يثبتان بيانات التحديث وسلسلة المصدر للرموز العشرة."),
        "6.4":("PARTIAL","41,600 محاولة تعادل 38,400 قياسًا مختلفًا و 3,200 استخدامًا سابقًا؛ 13 عائلة ثابتة و32 توليفة معلنة، لا ابتكار عائلة جديدة في كل دورة."),
        "6.5":("VERIFIED","تثبت جميع السجلات الـ32 100 خلية TESTED × 13 عائلة؛ لا خلايا مفقودة أو نتائج مصطنعة."),
        "6.7":("VERIFIED","نتائج نهائية لـ 1,300 سياسة مع أسباب رفض صريحة، و 41,600 محاولة تطوير منشورة في CSV؛ لا استراتيجية مؤهلة."),
        "7.1":("VERIFIED","المؤشرات العشرة جميعها NO_QUALIFIED_STRATEGY؛ طابق العرض العام غياب أفضل سياسة بدل فرض اختيار."),
        "7.2":("VERIFIED","حُفظت مقاييس وإعدادات ونتائج جميع السياسات النهائية؛ لا إعدادات فائز مؤهل لعدم وجوده."),
        "7.3":("PARTIAL","أعيد تطبيق بوابة ثبات الطيات ومضاعفة Holm من الأرقام المسجلة؛ تاريخ مكشوف وطيات اختيار تابعة وتكاليف غير موثقة تمنع إثبات ثبات قابل للتنفيذ."),
        "8.2":("VERIFIED","اللقطة المنشورة بعد الإكمال ترفض تعيين فريم أفضل لكل المؤشرات العشرة لعدم اجتياز الشروط."),
        "8.8":("PARTIAL","CSV العام يحتوي 41,600 محاولة؛ طابقت جميع الأرقام ببيانات validation وجميع حالات/أسباب السياسات المختارة ببوابات final. الأعمدة الرقمية غير مسماة بالجزء رغم مزجها بالحالة النهائية؛ لا يصح قراءتها كمقاييس final."),
        "8.10":("PARTIAL","تحققت من CSV نهائي محمّل فعليًا: 1,300 صف scope=oos طابقت مقاييسه وإعداداته وحالته وأسبابه بالكامل. الواجهة تشرح مزج search.csv المحفوظ، ويبقى التصدير الأصلي لكل المحاولات بلا metrics_partition؛ تصدير جميع المحاولات المعلّم المستقل لم يدقق هنا بعد."),
        "8.11":("VERIFIED","إيصال CUA فعلي نهائي يثبت 32/32 وRTL؛ سطح المكتب 1280/scroll 1280 والهاتف 390/scroll 390 بلا تجاوز. البصرية اعتمدها المنسق بعد إعادة تحميل الهاتف؛ دققت بصمات الصور ولم أكرر الفحص البصري. تغطي اللقطة viewport وليست كل الأقسام."),
        "11.1":("VERIFIED","لا توجد استراتيجية مؤهلة لهذا المؤشر لكل واحد من الرموز العشرة؛ لا تُختلق إعدادات أو فريم مؤهل."),
        "11.2":("PARTIAL","السجلات التاريخية والتطويرية والنهائية موجودة ومثبتة البصمات ضمن النطاق؛ إعادة الحساب النهائية الفعلية منفصلة، وإعادة كل بحث التطوير غير مثبتة."),
        "11.3":("VERIFIED","عدد السياسات المؤهلة صفر؛ لا ترتيب نجاحات أو إعلان ربحية."),
        "11.4":("PARTIAL","لكل سياسة نهائية أسباب تطابق تمامًا البوابات الفاشلة، وكامل محاولات البحث 41,600 في CSV. حالات NOT_SELECTED تخص اختيار التطوير، وREJECTED_FINAL تخص السياسة المختارة؛ الأرقام تبقى validation ويجب بيان هذا المزج أو توفير تصدير معلّم مستقل."),
        "11.5":("VERIFIED","32 سجل دورة مكتمل طابق الحالة والفواصل الفعلية؛ ليس حسابًا نظريًا من مرور الوقت."),
    }
    for section in prep["sections"]:
        items = []
        for source in section["requirements"]:
            item = dict(source)
            item["preparatory_status"] = item["status"]
            if item["id"] in overrides:
                item["status"], item["completion_audit_finding_ar"] = overrides[item["id"]]
                item["completion_audit_evidence"] = ["completed_campaign_audit_20261012.json", "progress_delta_completed_20261012.json", "selection_frozen.json", "final_family_sha256", "frontend/public/campaign/latest.json"]
            else:
                item["completion_audit_finding_ar"] = "لم يثبت هذا التدقيق البنيوي دليلاً إضافياً يغلق القيد؛ يحتفظ البند بنطاق الدليل السابق وحدوده."
            items.append(item)
        section_evidence.append({"section": section["section"], "title_ar":section["title_ar"], "requirements":items})
    status_counts = Counter(r["status"] for s in section_evidence for r in s["requirements"])
    gaps = [c["start_gap_seconds"] for c in cycles if c["start_gap_seconds"] is not None]
    report = {
        "schema_version":1, "document_type":"COMPLETED_CAMPAIGN_STRUCTURAL_AUDIT", "audit_status":"PASS" if not errors else "FAIL",
        "completion_claim":"The 32-cycle controller run completed; the full user request is not certified complete by this structural audit.",
        "started_at_utc":started, "completed_at_utc":utc(), "runtime_seconds":time.monotonic()-clock,
        "script_sha256":digest(Path(__file__)), "campaign":{k:v for k,v in state.items() if k not in {"cycles","coverage","winners"}},
        "request_sha256":digest(request), "check_count":sum(checks.values()), "checks":dict(checks), "errors":errors,
        "counts":{"actual_cycles":32,"cells_per_cycle":100,"families_per_cell":13,"candidate_attempts":attempts,"unique_new_development_measurements":fresh,"cache_reuse_rows":reused,"different_declared_id_alias_rows":aliases,"selected_final_policies":1300,"walk_forward_saved_artifacts":len(fold_artifacts),"saved_final_completed_trades":sum(r["metrics"]["completed_trades"] for r in outcomes),"qualified_policies":len(qualified),"published_search_csv_rows":csv_rows},
        "cadence":{"start_gap_min_seconds":min(gaps),"start_gap_max_seconds":max(gaps),"deadline_at":state["deadline_at"],"last_cycle_completed_at":cycles[-1]["completed_at"],"last_cycle_completed_elapsed_seconds":last_end-state["started_epoch"],"selection_frozen_at":selected["frozen_at"],"first_final_measured_at":min(r["measured_at"] for r in outcomes),"last_final_measured_at":max(r["measured_at"] for r in outcomes),"family_finalized_at":family["finalized_at"],"state_completed_at":state["completed_at"]},
        "cycles":cycles,"code_pins":code,"runtime":runtime,"frozen_source_datasets":sources,
        "development_audit_chain":{"baseline":{"path":str(baseline_path),"sha256":BASELINE_SHA,"scope_cycles":19},"delta":{"path":str(delta_path),"sha256":digest(delta_path),"scope_cycles":[20,32],"counts":delta["counts"],"check_count":delta["check_count"]}},
        "operational_checker_counter_correction":{"initial_failed_receipt":{"path":str(initial_failed_path),"sha256":digest(initial_failed_path),"errors":initial_failed["errors"],"checker_sha256":initial_failed["script_sha256"]},"reason":"Frozen data.py emits integer real_orders_sent=0. The original operational checker incorrectly required identity with boolean False. Only the operational predicate was corrected, with strict int-zero acceptance; the producer and data were not changed.","synthetic_contract_test":{"path":str(counter_test_path),"sha256":digest(counter_test_path),"receipt":counter_test},"rerun_checker_sha256":delta["script_sha256"],"summary_receipts":delta["summary_receipts"]},
        "operational_checker_safety_schema_correction":{"initial_failed_audit_path":str(safety_initial_failed_path),"initial_failed_audit_sha256":digest(safety_initial_failed_path),"diagnosis":{"path":str(safety_diagnosis_path),"sha256":digest(safety_diagnosis_path),"receipt":safety_diagnosis},"focused_synthetic_test":{"path":str(safety_test_path),"sha256":digest(safety_test_path),"receipt":safety_test},"state_family_declaration_public_exact_schema":dict(SAFETY),"engine_result_exact_schema":dict(RESULT_SAFETY),"every_value_must_be_strict_false":True,"missing_extra_true_integer_zero_rejected":True,"scientific_source_or_gate_modified":False},
        "selection":{"path":str(selection_path),"sha256":state["selection_sha256"],"reconstructed_from_development_only":True,"final_data_accessed_in_seal":selected["final_data_accessed"]},
        "final_family":{"path":str(family_path),"sha256":state["final_artifact"]["sha256"],"compressed_bytes":family_path.stat().st_size,"original_measurements_rehashed":len(original_pins),"saved_gate_failure_counts":dict(gate_failures),"results":outcomes},
        "public_snapshot":{"path":"frontend/public/campaign/latest.json","sha256":digest(ROOT/"frontend/public/campaign/latest.json"),"generated_at":public["generated_at"],"symbols":public["symbols"],"search_csv_sha256":digest(csv_path)},
        "search_csv_partition_contract":{"rows":csv_rows,"numeric_columns_partition":"validation","selected_disposition_and_reasons_partition":"final","not_selected_disposition_partition":"development_selection","explicit_metrics_partition_column":False,"disposition_row_counts":dict(csv_dispositions),"every_numeric_column_and_disposition_matched_to_original_journals":True,"limitation":"Do not interpret validation metric columns beside REJECTED_FINAL as final metrics; aliases may repeat one selected policy's final disposition."},
        "executed_browser_evidence":{"path":str(browser_path),"sha256":digest(browser_path),"receipt":browser,"visual_review_by_this_auditor":False},
        "final_export_and_ui_clarification_evidence":{"path":str(export_qa_path),"sha256":digest(export_qa_path),"receipt":export_qa,"selected_oos_rows_independently_matched":len(exported_final_ids),"visual_review_by_this_auditor":False},
        "ci_evidence":{"saved_run_path":str(ci_path),"saved_run_sha256":digest(ci_path),"head_sha":actual_head,"run_id":ci["id"],"conclusion":ci["conclusion"],"jobs":job_statuses,"html_url":ci["html_url"],"fresh_network_check_by_this_auditor":False,"working_tree_status_at_audit":git_status,"limitation":"CI success is for the exact published HEAD; local UI clarification is separately tested and has not been covered by this saved CI run."},
        "requirements_sections":section_evidence,"requirements_status_counts":dict(status_counts),
        "numerical_replay_performed_by_this_audit":False,"all_development_search_measurements_replayed":False,"historical_costs_verified":False,"execution_realism_verified":False,"safety":dict(SAFETY),"limitations":limitations,
        "actions":{"scientific_source_modified":False,"campaign_state_modified":False,"research_or_order_process_started":False,"outputs_only_under_work":True},
    }
    OUTPUT.with_suffix(".json").write_bytes(encoded(report))
    note = (f"نتيجة التدقيق البنيوي الختامي للحملة: **{report['audit_status']}**، {report['check_count']:,} فحصًا و{len(errors)} اختلافات. هذا يثبت اكتمال تشغيل الحملة؛ لا يشهد باستيفاء الطلب كله أو ربحية قابلة للتنفيذ.\n\n"
        f"دققت السجلات الـ32: في كل دورة 100 خلية × 13 عائلة، بإجمالي **{attempts:,} محاولة**، منها **{fresh:,} قياس تطويري مختلف** و**{reused:,} إعادة استخدام**. فواصل البدء الفعلية {min(gaps):.6f}–{max(gaps):.6f} ثانية. انتهت الدورة 32 في {cycles[-1]['completed_at']} بعد {last_end-state['started_epoch']:.6f} ثانية من البداية.\n\n"
        f"أعدت بناء الاختيار من validation فقط؛ طابق سجل الحالة وملف الاختيار المثبت `{state['selection_sha256']}`. ثبت الاختيار في {selected['frozen_at']}، بعد جميع الدورات وبعد حد ثماني ساعات. جميع القياسات النهائية الـ 1,300 مؤرخة بعد هذا التثبيت. طابقت بصمات ملفات الكود الثمانية ونسخها المؤرشفة، وملفات البيانات المجمدة العشرة ومصادرها.\n\n"
        "تدقيق التطوير مرتبط بإيصال 19 السابق وبفحص delta فعلي للدورات 20–32؛ ليس إعادة حساب لكل البحث. فُحصت بصمات 3,900 ملف لطيات التطوير و 1,300 قياس نهائي أصلي والأسرة النهائية كاملة. عُدّت صفقات الأسرة النهائية والتصرفات وتحققت مطابقة العدد والحدود الزمنية، وأعيد تطبيق بوابات القبول وHolm على الأرقام المسجلة. لم يُعد هذا التدقيق حساب الإشارات أو محاكاة مسارات الصفقات أو bootstrap؛ إعادة الحساب الفعلية التي ينفذها المنسق لها إيصال منفصل.\n\n"
        "حُفظ فشل التدقيق التشغيلي الأول كما هو: كان يتوقع real_orders_sent is False بينما المنتج المجمد يسجل عدادًا صحيحًا 0. ثبت عقد الأنواع بست حالات ثم صحح checker فقط لقبول int=0 ورفض bool وغير الصفر. في الإعادة أعيدت بصمات جميع الملفات واستُخدم 15,500 ملخص سابق بعد تطابق AST لدالة الاستخراج؛ عدد الملخصات المحسوبة حديثًا صفر، وليس ذلك بحثًا جديدًا.\n\n"
        "وحُفظ فشل checker البنيوي الأول أيضًا: خلط مخطط الحالة ذي4 مفاتيح بمخطط نتيجة المحرك ذي7 مفاتيح الذي يضيف نظائر العرض الصغيرة. تبين على الأصل أن القيم السبع كلها false، وأن الفرق الوحيد في البوابات16 هو توقع checker الخاطئ لمخطط الأمان. اختُبر المخططان مستقلين مع رفض المفاتيح الناقصة/الزائدة وtrue و0؛ لم تخفف بوابة المنتج أو تُعدل البيانات.\n\n"
        f"عدد السياسات المؤهلة: **{len(qualified)}**. عرض جميع المؤشرات العشرة: **لا توجد استراتيجية مؤهلة لهذا المؤشر.** اللقطة العامة COMPLETED 32، وCSV يحتوي {csv_rows:,} محاولة فعلية.\n\n"
        f"قيد تصدير مهم: طابقت الأرقام الأربعة لكل صف CSV بالتطوير؛ completed_trades وPF وصافي الربح وDD هي validation دائمًا، بينما {csv_dispositions.get('REJECTED_FINAL',0):,} صفًا يحمل حالة/أسباب final للسياسة المختارة، والبقية NOT_SELECTED في اختيار التطوير. لا يوجد metrics_partition؛ لا تنسب الأرقام بجانب REJECTED_FINAL إلى الاختبار النهائي. قد تتكرر الحالة لسياسة واحدة عبر aliases. التصدير النهائي المحمّل فعليًا بديل مستقل واضح scope=oos؛ طابقت جميع صفوفه الـ1,300 مع اللقطة، والواجهة تشرح نطاق الملف الخام. لم أدقق بعد تصدير جميع المحاولات المعلّم المستقل، ويبقى المصدر المجمد محفوظًا.\n\n"
        "قيود صريحة باقية: 13 عائلة ثابتة لا تثبت ابتكار فكرة في كل دورة؛ تكاليف وتعبئات CFD التاريخية غير موثقة و 0.10 ATR مفترض؛ لكل مصدر دقيقة داخلية مفقودة وصف غير صالح؛ سنة البيانات ليست إثبات أقصى تاريخ متاح؛ التاريخ مكشوف وطيات التطوير تابعة للاختيار. لا يتحول نجاح السلامة أو الاختبارات البرمجية إلى ربحية تداول.\n\n"
        f"أرفقت تحديث أدلة البنود الـ79 في الأقسام 11؛ الحالات {dict(status_counts)}، وهي ليست نسبة اكتمال. لم أغلق بندًا يتطلب إثباتًا عدديًا أو بصريًا أو تكاليف تنفيذ غير متوفرة.\n\n"
        "LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false، ولم يُعدّل الكود العلمي أو الحالة أو البيانات أو الإعدادات ولم تُرسل أوامر تداول.\n\n"
        f"التفاصيل: [JSON]({OUTPUT.name}.json)، SHA256 `{digest(OUTPUT.with_suffix('.json'))}`.\n")
    OUTPUT.with_suffix(".ar.md").write_text(note)
    print(json.dumps({"audit_status":report["audit_status"],"check_count":report["check_count"],"errors":errors,"counts":report["counts"],"cadence":report["cadence"],"requirement_status_counts":dict(status_counts),"json_sha256":digest(OUTPUT.with_suffix('.json'))},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
