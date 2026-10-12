"""Read-only receipt, identity, coverage and export audit; never run a backtest."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import csv
import gzip
import json
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "data/single_frame_campaign_20261011"
REPLAY = ROOT / "work/final_replay_20261011"
OUT = ROOT / "work/final_replay_closeout_independent_20261012.json"
FLAGS = {"LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"}


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    start = utc()
    timer = time.monotonic()
    checks = Counter()
    errors = []
    hashes = {}
    hashed_bytes = 0

    def check(ok, label, item=None):
        checks[label] += 1
        if not ok:
            errors.append({"check": label, "item": item})

    def digest(path, expected=None):
        nonlocal hashed_bytes
        path = Path(path).resolve()
        if str(path) not in hashes:
            with path.open("rb") as stream:
                actual = __import__("hashlib").file_digest(stream, "sha256").hexdigest()
            hashes[str(path)] = actual
            hashed_bytes += path.stat().st_size
        actual = hashes[str(path)]
        if expected is not None:
            check(actual == expected, "pinned_file_hash", str(path))
        return actual

    def safe(value):
        return isinstance(value, dict) and set(value) == FLAGS and all(v is False for v in value.values())

    latest_path = REPLAY / "latest.json"
    latest_sha = digest(latest_path)
    latest = read(latest_path)
    index_path = REPLAY / "receipts_index.json"
    index_sha = digest(index_path)
    index = read(index_path)
    state = read(CAMPAIGN / "state.json")
    declaration = read(CAMPAIGN / "declaration.json")
    selection = read(CAMPAIGN / "selection_frozen.json")
    for path in [CAMPAIGN / "state.json", CAMPAIGN / "declaration.json", CAMPAIGN / "selection_frozen.json"]:
        digest(path)
    check(state["status"] == "COMPLETED" and state["completed_cycles"] == 32 and state["final_accessed"] is True, "completed_campaign")
    check(safe(state["safety"]) and safe(declaration["safety"]), "state_declaration_safety")
    check(selection["final_data_accessed"] is False and selection["source"] == state["declaration_sha256"], "pre_final_seal")
    check(digest(CAMPAIGN / "declaration.json") == state["declaration_sha256"], "declaration_seal")
    check(digest(CAMPAIGN / "selection_frozen.json") == state["selection_sha256"], "selection_seal")
    campaign_identity = {
        "campaign_directory": str(CAMPAIGN), "declaration_sha256": state["declaration_sha256"],
        "selection_sha256": state["selection_sha256"], "final_artifact": state["final_artifact"],
        "source_manifest_sha256": declaration["source_manifest_sha256"], "source_pins": declaration["source_pins"],
        "runtime": declaration["runtime"], "helper_sha256": digest(ROOT / "scripts/replay_campaign_final.py"),
        "helper_schema_version": 1,
    }
    check(latest["campaign"] == campaign_identity and index["campaign"] == campaign_identity, "exact_replay_campaign_identity")
    for name, pin in declaration["source_pins"].items():
        digest(ROOT / "research_campaign" / name, pin)
        digest(CAMPAIGN / "frozen_source" / name, pin)
    digest(Path(declaration["source_manifest"]), declaration["source_manifest_sha256"])
    for row in declaration["datasets"].values():
        digest(Path(row["requested_file"]), row["requested_sha256"])
    family_path = CAMPAIGN / state["final_artifact"]["path"]
    digest(family_path, state["final_artifact"]["sha256"])
    # The canonical family begins with this small index, before the large ledgers.
    buffer = []
    active = False
    with gzip.open(family_path, "rt") as stream:
        for line in stream:
            if line.startswith('  "original_measurements": {'):
                active = True
                buffer.append("{\n")
            elif active and line.startswith("  },"):
                buffer.append("}\n")
                break
            elif active:
                buffer.append(line)
    original_measurements = json.loads("".join(buffer))
    winners = {g: value for g, value in selection["selection"].items() if value["status"] == "TESTED"}
    check(set(winners) == set(original_measurements) and len(winners) == 1300, "exact_selected_original_family")
    expected_plan = {}
    for group in sorted(winners):
        winner = winners[group]
        row = declaration["datasets"][winner["symbol"]]
        common = {"selection_group": group, "symbol": winner["symbol"], "timeframe": winner["timeframe"],
                  "strategy_id": winner["spec"]["id"], "source_file": row["requested_file"], "source_sha256": row["requested_sha256"]}
        check(group == f'{winner["symbol"]}/{winner["timeframe"]}/{winner["spec"]["family"]}' and winner["strategy_id"] == winner["spec"]["id"], "winner_identity", group)
        entries = [{**common, "phase": "final", "artifact": original_measurements[group]}]
        folds = selection["walk_forward"][group]
        check(len(folds) == 3, "three_frozen_folds", group)
        for number, fold in enumerate(folds, 1):
            check(fold["label"] == f"wf{number}" and fold["status"] == "TESTED", "actual_tested_fold", f"{group}/{number}")
            entries.append({**common, "phase": "walk_forward", "fold_number": number, "artifact": fold.get("artifact"), "fold_status": fold["status"]})
        for item in entries:
            identity = {"campaign": campaign_identity, "measurement": item}
            name = sha256(encoded(identity)).hexdigest()
            check(name not in expected_plan, "unique_expected_receipt", name)
            expected_plan[name] = identity

    check(len(expected_plan) == 5200 and set(index["receipts"]) == set(expected_plan), "exact_5200_index_coverage")
    pointers = {p["receipt"]["path"]: p for p in latest["receipts"]}
    check(len(pointers) == len(latest["receipts"]) == 5200, "unique_latest_coverage")
    check(set(pointers) == {f"receipts/{key}.json" for key in expected_plan}, "latest_index_same_coverage")
    check({p.name for p in (REPLAY / "receipts").glob("*.json")} == {f"{key}.json" for key in expected_plan}, "no_missing_or_undeclared_receipt_files")
    phase_counts = Counter()
    ledger_matches = {phase: Counter() for phase in ["final", "walk_forward"]}
    timestamps = []
    receipt_bytes = 0
    for name, identity in expected_plan.items():
        pointer = index["receipts"][name]
        check(pointer["path"] == f"receipts/{name}.json" and pointers[pointer["path"]]["receipt"] == pointer, "pointer_identity_and_equality", name)
        path = REPLAY / pointer["path"]
        digest(path, pointer["sha256"])
        receipt_bytes += path.stat().st_size
        receipt = read(path)
        item = identity["measurement"]
        report = receipt["report"]
        phase = item["phase"]
        check(receipt["identity"] == identity and pointers[pointer["path"]]["selection_group"] == item["selection_group"], "exact_receipt_identity", name)
        check(receipt["report_sha256"] == sha256(encoded(report)).hexdigest(), "inner_report_hash", name)
        check(receipt["schema_version"] == 1 and receipt["status"] == report["status"] == "PASS", "receipt_and_report_pass", name)
        check(safe(receipt["safety"]) and safe(report["safety"]), "receipt_strict_false_safety", name)
        check(report["phase"] == phase and report["cache_used"] is False and type(report["matched_measurements"]) is int and report["matched_measurements"] == 1, "fresh_phase_and_measurement", name)
        check(report["artifact"] == item["artifact"] and report["selection_sha256"] == state["selection_sha256"] and all(report[k] == item[k] for k in ["source_sha256", "symbol", "timeframe", "strategy_id"]), "report_measurement_identity", name)
        verification = report["verification"]
        check(verification["declaration_sha256"] == campaign_identity["declaration_sha256"] and verification["source_manifest_sha256"] == campaign_identity["source_manifest_sha256"] and verification["runtime"] == campaign_identity["runtime"] and verification["replay_code_sha256"] == declaration["source_pins"]["replay.py"] and verification["replay_code_declared"] is True and verification["code_files_rehashed"] == 8, "verified_frozen_runtime_and_code", name)
        check(report["real_execution_costs"] == "NOT_VERIFIED_BY_REPLAY", "honest_cost_scope", name)
        if phase == "final":
            check(report["full_signal_fingerprint_matched"] is True and report["matched_partitions"] == 1, "full_final_signal_match", name)
        else:
            check(type(report["backtests_run"]) is int and report["backtests_run"] == 1 and report["matched_partitions"] == 1, "fresh_walk_forward_backtest", name)
        for key in ["matched_trades", "matched_equity_points", "matched_dispositions"]:
            check(type(report[key]) is int and report[key] >= 0, "ledger_match_counter", f"{name}/{key}")
            ledger_matches[phase][key] += report[key]
        t = datetime.fromisoformat(report["replayed_at"])
        end = datetime.fromisoformat(receipt["completed_at"])
        check(datetime.fromisoformat(latest["started_at"]) <= t <= end <= datetime.fromisoformat(latest["completed_at"]), "receipt_within_fresh_invocation", name)
        timestamps.append(end)
        phase_counts[phase] += 1
        digest(CAMPAIGN / item["artifact"]["path"], item["artifact"]["sha256"])
    print(json.dumps({"stage": "5200_receipts_and_original_artifact_hashes", "errors": len(errors), "counts": dict(phase_counts)}), flush=True)
    check(latest["status"] == "PASS" and latest["errors"] == [] and safe(latest["safety"]), "overall_replay_pass_and_safety")
    check(latest["selected_final_and_WF_reproduction_complete"] is True and latest["all_final_measurements_replayed"] is True and latest["all_development_search_measurements_replayed"] is False, "exact_complete_scope")
    for key in ["attempted_now", "backtests_now", "measured_now", "completed_total_verified_this_invocation", "requested_measurements_and_folds"]:
        check(type(latest[key]) is int and latest[key] == 5200, "fresh_summary_count", key)
    for key in ["reused_verified_receipts", "recovered_orphan_receipts", "remaining_requested_measurements_and_folds", "remaining_final_measurements", "remaining_walk_forward_folds"]:
        check(type(latest[key]) is int and latest[key] == 0, "zero_reused_remaining_or_recovered", key)
    check(latest["expected_final_measurements"] == 1300 and latest["expected_walk_forward_folds"] == 3900 and latest["include_walk_forward"] is True and latest["max_new"] is None, "requested_plan_scope")
    for phase, count in [("final", 1300), ("walk_forward", 3900)]:
        check(phase_counts[phase] == count and latest["phase_counts"][phase] == {"backtests_now": count, "measured_now": count, "reused": 0, "verified_total": count}, "phase_count_matches_actual_receipts", phase)
    check(latest["holm"]["status"] == "PASS" and latest["holm"]["calculated_this_invocation"] is True and latest["holm"]["policies_compared"] == 1300, "full_joint_holm_and_family_gates")
    run_files = list(REPLAY.glob("run-*.json"))
    check(len(run_files) == 1 and run_files[0].read_bytes() == latest_path.read_bytes(), "single_completed_run_and_latest_bytes")

    export_path = ROOT / "work/search_phase_labelled_20261012.receipt.json"
    export_sha = digest(export_path)
    export = read(export_path)
    byte_path = ROOT / "work/search_phase_labelled_bytecheck_20261012.json"
    byte_sha = digest(byte_path)
    byte = read(byte_path)
    check(export["status"] == "VERIFIED_JOURNAL_EXPORT_WITH_EXPLICIT_PHASE_LABELS" and safe(export["safety"]), "annotated_export_pass_safety")
    check(export["completed_cycles"] == 32 and export["campaign_id"] == state["id"] and export["metrics_partition"] == "validation", "annotated_export_campaign_and_scope")
    check(export["appended_columns"] == ["metrics_partition", "disposition_partition"] and export["original_source_bytes_unchanged"] is True and export["original_record_bytes_retained_before_added_labels"] is True, "annotated_export_preservation_contract")
    for label in ["source", "output"]:
        p = Path(export[label]["path"])
        digest(p, export[label]["sha256"])
        check(p.stat().st_size == export[label]["bytes"], "export_byte_count", label)
    digest(ROOT / "scripts/annotate_campaign_search_export.py", export["script_sha256"])
    for path, pin in export["source_and_evidence_pins_before_after"].items():
        digest(Path(path), pin)
    check(len(export["source_and_evidence_pins_before_after"]) == 1365, "annotated_export_all_1365_pins")
    export_counts = Counter()
    with Path(export["source"]["path"]).open(newline="", encoding="utf-8-sig") as source, Path(export["output"]["path"]).open(newline="", encoding="utf-8-sig") as output:
        source_reader, output_reader = csv.DictReader(source), csv.DictReader(output)
        check(source_reader.fieldnames == export["original_columns"] and output_reader.fieldnames == source_reader.fieldnames + export["appended_columns"], "exact_appended_export_header")
        from itertools import zip_longest
        for number, (original, annotated) in enumerate(zip_longest(source_reader, output_reader), 1):
            check(original is not None and annotated is not None, "no_missing_export_rows", number)
            if original is None or annotated is None:
                continue
            check(all(annotated[k] == original[k] for k in original), "every_original_export_cell_unchanged", number)
            phase = "final_selection" if original["selection_disposition"] == "REJECTED_FINAL" else "development_selection"
            check(annotated["metrics_partition"] == "validation" and annotated["disposition_partition"] == phase, "explicit_partition_label_each_row", number)
            export_counts[phase] += 1
            export_counts["rows"] += 1
            export_counts["cache_hits" if original["cache_hit"] == "True" else "new_measurements"] += 1
    # Each producer record is one physical line; assert that before comparing
    # exact original quoting and line endings, excluding only the appended suffix.
    with Path(export["source"]["path"]).open("rb") as source, Path(export["output"]["path"]).open("rb") as output:
        for number, (original, annotated) in enumerate(zip_longest(source, output), 0):
            if original is None or annotated is None:
                check(False, "raw_record_byte_preservation", number)
                continue
            end = b"\r\n" if original.endswith(b"\r\n") else b"\n" if original.endswith(b"\n") else b""
            core = original[:-len(end)] if end else original
            check(annotated.startswith(core + b",") and (not end or annotated.endswith(end)), "raw_record_byte_preservation", number)
            suffix = annotated[len(core):]
            check(suffix in [b",metrics_partition,disposition_partition" + end, b",validation,final_selection" + end, b",validation,development_selection" + end], "only_two_appended_fields", number)
    check(export_counts == Counter({"rows": 41600, "final_selection": 1500, "development_selection": 40100, "cache_hits": 3200, "new_measurements": 38400}), "all_export_population_counts")
    check(byte["status"] == "ACTUAL_ALL_RECORD_BYTES_AND_COLUMNS_PRESERVED" and byte["zero_mismatches"] is True and byte["records_including_header"] == 41601 and byte["original_data_rows"] == 41600, "previous_independent_bytecheck_consistent")
    for path, pin in byte["before_after_file_hashes"].items():
        digest(Path(path), pin)
    for path, expected in [(latest_path, latest_sha), (index_path, index_sha)]:
        with path.open("rb") as stream:
            check(__import__("hashlib").file_digest(stream, "sha256").hexdigest() == expected, "replay_receipt_package_unchanged_after_independent_check", str(path))
    report = {
        "schema_version": 1, "status": "PASS" if not errors else "FAIL", "started_at": start, "completed_at": utc(),
        "runtime_seconds": time.monotonic() - timer, "scope": "Independent byte/hash/identity/coverage/safety verification of the already completed numerical replay and labelled export. No backtests or formula reruns by this audit.",
        "script_sha256": digest(Path(__file__)), "campaign_id": state["id"], "campaign_identity": campaign_identity,
        "latest": {"path": str(latest_path), "sha256": latest_sha, "completed_at": latest["completed_at"]},
        "receipts_index": {"path": str(index_path), "sha256": index_sha}, "receipt_files_verified": len(expected_plan), "receipt_bytes": receipt_bytes,
        "phase_counts_independently_verified": dict(phase_counts), "fresh_backtests_recorded": 5200,
        "root_reused_verified_receipts": latest["reused_verified_receipts"], "root_recovered_orphans": latest["recovered_orphan_receipts"],
        "recorded_ledger_matches": {k: dict(v) for k, v in ledger_matches.items()},
        "first_receipt_completion": min(timestamps).isoformat(), "last_receipt_completion": max(timestamps).isoformat(),
        "selected_final_and_WF_reproduction_complete": latest["selected_final_and_WF_reproduction_complete"],
        "all_development_search_measurements_replayed": False, "holm_scope_receipt_verified": latest["holm"],
        "original_final_and_WF_artifacts_rehashed": 5200, "unique_pinned_files_hashed": len(hashes), "total_unique_file_bytes_hashed": hashed_bytes,
        "phase_labelled_export": {"receipt_path": str(export_path), "receipt_sha256": export_sha, "bytecheck_path": str(byte_path), "bytecheck_sha256": byte_sha,
                                   "source": export["source"], "output": export["output"], "counts_independently_verified": dict(export_counts), "metrics_partition": "validation", "all_original_columns_and_record_bytes_retained": True},
        "check_count": sum(checks.values()), "checks": dict(checks), "errors": errors, "file_hashes": hashes,
        "new_backtests_performed_by_this_audit": 0, "historical_costs_verified": False, "execution_realism_verified": False,
        "safety": {key: False for key in sorted(FLAGS)},
        "limitations": ["Root replay uses original frozen production formulas, not an independent mathematical implementation.",
                        "Only selected 1300 final originals and 3900 selected development folds were numerically replayed; 38400 development-search measurements were not all numerically rerun.",
                        "Receipt and ledger match totals overlap policies and are not independent market-event samples.",
                        "Historical CFD costs, cash mapping, financing, fills and full original-tick strategy execution remain unverified.",
                        "Previously exposed history is not fresh prospective OOS; development folds depend on policy selection.",
                        "All 13 families were predeclared; 32 parameter rounds do not prove one newly invented family per cycle.",
                        "The original raw search CSV is unchanged; metrics are validation and selected dispositions are final. The added partition labels clarify this mix."]
    }
    OUT.write_bytes(encoded(report))
    print(json.dumps({"status": report["status"], "checks": report["check_count"], "errors": errors[:10], "receipt_files": len(expected_plan), "phase_counts": dict(phase_counts), "unique_hashed_files": len(hashes), "output": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
