"""Incremental journal/source integrity audit; never evaluate a holdout result.

The trusted earlier receipt anchors old artifact hashes. Only artifacts first
referenced after that receipt are rehashed. Resumption rehashes their bytes but
reuses a pinned summary receipt to avoid repeatedly decompressing large ledgers.
This is an integrity audit, not numerical replay or a profitability test.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_campaign.source_audit import audit_catalogue
from research_campaign.state import SAFETY, atomic_json, digest, encoded, guard, read_json, utc, validate_state

BASELINE_SHA = "23c0a4e2a8ac5713a2f30c7ae9479b76817269283c2e6403c0a450eef6f1ee61"
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"


def no_real_order_count(value):
    """The frozen collector emits an integer counter, never a boolean flag."""
    return type(value) is int and value == 0


def _safe(value):
    return isinstance(value, dict) and set(value) == set(SAFETY) and all(v is False for v in value.values())


def _epoch(value):
    return datetime.fromisoformat(value).timestamp()


def _same(a, b):
    return encoded(a) == encoded(b)


def _artifact_summary(path):
    """Stream canonical pretty JSON; discard trade/equity/disposition arrays.

    Fixed indentation distinguishes the outer headers and train/validation
    metrics. JSON strings cannot contain literal unescaped newlines. Metrics
    are decoded exactly, and the immutable whole gzip byte hash is checked by
    the caller; this does not numerically reconstruct any ledger.
    """
    header = ""
    rolling = ""
    metrics = []
    score = None
    score_found = False
    marker = '\n    "metrics": '
    score_marker = '\n  "validation_score": '
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt") as stream:
        collecting = False
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            if '\n  "train":' not in header and len(header) < 65536:
                header += chunk
            rolling += chunk
            while True:
                if collecting:
                    try:
                        value, end = decoder.raw_decode(rolling.lstrip())
                    except json.JSONDecodeError:
                        if len(rolling) > 65536:
                            raise ValueError("Unexpectedly large or malformed metric summary")
                        break
                    metrics.append(value)
                    rolling = rolling.lstrip()[end:]
                    collecting = False
                location = rolling.find(marker)
                if location >= 0:
                    rolling = rolling[location + len(marker):]
                    collecting = True
                    continue
                location = rolling.find(score_marker)
                if location >= 0:
                    tail = rolling[location + len(score_marker):]
                    if "\n" in tail:
                        score = json.loads(tail.split("\n", 1)[0].rstrip(","))
                        score_found = True
                rolling = rolling[-max(len(marker), len(score_marker), 80):]
                break
    if '\n  "train":' not in header or len(metrics) != 2 or not score_found:
        raise ValueError("Expected one closed development train/validation summary and score")
    metadata = json.loads(header.split('\n  "train":', 1)[0].rstrip().removesuffix(",") + "\n}")
    return {"metadata": metadata, "training_metrics": metrics[0],
            "validation_metrics": metrics[1], "validation_score": score}


def audit(directory, baseline_file, baseline_sha, output_prefix, receipt_dir,
          through_cycle=None, requested_snapshot=None, reuse_summary_script=None,
          reuse_summary_script_sha256=None):
    guard()
    directory, baseline_file = Path(directory).resolve(), Path(baseline_file).resolve()
    output_prefix, receipt_dir = Path(output_prefix).resolve(), Path(receipt_dir).resolve()
    if any(p == directory or p.is_relative_to(directory) for p in (output_prefix, receipt_dir)):
        raise ValueError("Audit outputs must stay outside frozen campaign data")
    if digest(baseline_file) != baseline_sha:
        raise ValueError("Earlier audit receipt hash differs from its trusted pin")
    baseline = read_json(baseline_file)
    if baseline.get("audit_status") != "PASS" or baseline.get("errors"):
        raise ValueError("Earlier audit did not pass")
    start = utc()
    clock = time.monotonic()
    state_bytes = (directory / "state.json").read_bytes()
    state = json.loads(state_bytes)
    validate_state(state)
    old_count = baseline["counts"]["completed_cycles_audited"]
    through = state["completed_cycles"] if through_cycle is None else through_cycle
    if type(through) is not int or not old_count <= through <= state["completed_cycles"]:
        raise ValueError("Explicit scope exceeds actual completed cycles or precedes baseline")
    declaration = read_json(directory / "declaration.json")
    errors, checks, diagnostics = [], Counter(), Counter()

    def check(value, label, context=None):
        checks[label] += 1
        if not value:
            item = {"check": label, "context": context}
            errors.append(item)
            print(json.dumps({"discrepancy": item}), flush=True)

    def hashed(path, expected, label):
        path = Path(path)
        check(path.exists() and digest(path) == expected, label, str(path))

    check(_safe(state["safety"]) and _safe(declaration["safety"]), "state_and_declaration_safety_false")
    hashed(directory / "declaration.json", state["declaration_sha256"], "current_declaration_pin")
    check(state["declaration_sha256"] == baseline["declaration_sha256"], "same_declaration_as_baseline")
    check(declaration["source_pins"] == baseline["code_pins"], "same_frozen_code_pins_as_baseline")
    for name, pin in declaration["source_pins"].items():
        hashed(ROOT / "research_campaign" / name, pin, "current_frozen_code_pin")
    hashed(declaration["source_manifest"], declaration["source_manifest_sha256"], "original_manifest_pin")
    check(declaration["source_manifest_sha256"] == baseline["initial_source_manifest"]["sha256"], "same_initial_source_manifest")
    targets, frames = set(declaration["symbol_catalogue"]), set(declaration["timeframes"])
    cells = {f"{s}/{f}" for s in targets for f in frames}
    check(len(cells) == 100, "ten_symbols_ten_frames")
    family_sets = [{s["family"] for s in specs} for specs in declaration["candidate_rounds"]]
    check(len(family_sets) == 32 and len(family_sets[0]) == 13 and all(v == family_sets[0] for v in family_sets),
          "same_thirteen_families_predeclared_in_thirty_two_rounds")
    by_id = {s["id"]: s for row in declaration["candidate_rounds"] for s in row}
    old_journals = {r["number"]: r for r in baseline["cycles"]}
    seen, new_artifacts, delta_cycles, all_cycle_pins = {}, {}, [], {}
    previous_start = previous_end = None
    for number in range(1, through + 1):
        path = directory / "cycles" / f"{number:02d}.json"
        cycle = read_json(path)
        cycle_sha = digest(path)
        all_cycle_pins[number] = cycle_sha
        if number <= old_count:
            check(cycle_sha == old_journals[number]["journal_sha256"], "baseline_completed_journal_unchanged", number)
            for row in cycle["results"]:
                seen.setdefault(row["artifact"]["path"], row)
            previous_start, previous_end = _epoch(cycle["started_at"]), _epoch(cycle["completed_at"])
            continue
        check(cycle["number"] == number and cycle["status"] == "COMPLETED", "delta_completed_journal_identity", number)
        summary = state["cycles"][number - 1]
        check(all(cycle[k] == summary[k] for k in summary), "delta_journal_matches_state_counter_record", number)
        check(set(cycle["coverage"]) == cells and all(v["status"] == "TESTED" for v in cycle["coverage"].values()),
              "delta_all_hundred_cells_tested", number)
        actual_start, actual_end, due = (_epoch(cycle[k]) for k in ("started_at", "completed_at", "due_at"))
        expected_due = max(state["started_epoch"] + (number - 1) * 900, previous_start + 900)
        check(abs(due - expected_due) < 1e-5 and actual_start >= due, "delta_not_before_actual_due", number)
        check(actual_end >= actual_start >= previous_end, "delta_start_and_completion_chronology", number)
        check(actual_start - previous_start >= 900, "delta_no_compressed_cycle", number)
        planned = {s["family"]: s for s in declaration["candidate_rounds"][number - 1]}
        rowkeys = set()
        fresh = hits = aliases = 0
        for row in cycle["results"]:
            symbol, frame, family = row["symbol"], row["timeframe"], row["family"]
            key = (symbol, frame, family)
            attempted = planned[family]
            check(key not in rowkeys, "delta_no_duplicate_cell_family", str((number, key)))
            rowkeys.add(key)
            check(symbol in targets and frame in frames and row["attempted_strategy_id"] == attempted["id"],
                  "delta_exact_declared_round_attempt", str((number, key)))
            check(row["spec"] == by_id.get(row["strategy_id"]) and row["spec"]["scientific_id"] == attempted["scientific_id"],
                  "delta_actual_spec_and_scientific_alias", str((number, key)))
            split = declaration.get("splits", {}).get(symbol, declaration["split"])
            identity = {"symbol": symbol, "timeframe": frame, "scientific_spec": attempted["scientific_id"],
                        "input_sha256": declaration["datasets"][symbol]["requested_sha256"],
                        "split": split, "phase": "development", "source_pins": declaration["source_pins"]}
            cache_key = sha256(encoded(identity)).hexdigest()
            artifact = row["artifact"]
            check(artifact["path"] == f"artifacts/{cache_key}.json.gz", "delta_artifact_scientific_identity", str((number, key)))
            check(read_json(directory / "cache" / f"{cache_key}.json") == artifact, "delta_cache_pointer_pin", str((number, key)))
            prior = seen.get(artifact["path"])
            hit = row["cache_hit"]
            check(type(hit) is bool and hit == (prior is not None), "delta_cache_hit_matches_original_attempt", str((number, key)))
            if prior is not None:
                check(all(_same(row[k], prior[k]) for k in ("artifact", "spec", "tested_at", "strategy_id",
                    "training_metrics", "validation_metrics", "validation_score")), "delta_alias_retains_original_measurement_and_summary", str((number, key)))
            else:
                seen[artifact["path"]] = row
                new_artifacts[artifact["path"]] = {"row": row, "identity": identity}
            check(row["status"] == "TESTED", "delta_measurement_tested", str((number, key)))
            fresh += int(row["status"] == "TESTED" and not hit)
            hits += int(hit)
            aliases += int(row["strategy_id"] != row["attempted_strategy_id"])
        check(len(rowkeys) == len(cycle["results"]) == 1300, "delta_thirteen_attempts_per_hundred_cells", number)
        check(fresh == cycle["tested_count"] and hits == cycle["cache_hits"], "delta_exact_new_and_reused_counters", number)
        delta_cycles.append({"number": number, "started_at": cycle["started_at"], "due_at": cycle["due_at"],
            "completed_at": cycle["completed_at"], "duration_seconds": actual_end - actual_start,
            "start_gap_seconds": actual_start - previous_start, "candidate_rows": len(cycle["results"]),
            "new_measurements": fresh, "cache_hits": hits, "different_id_aliases": aliases,
            "coverage_cells": len(cycle["coverage"]), "journal_sha256": cycle_sha, "market_update": cycle.get("market_update")})
        previous_start, previous_end = actual_start, actual_end
    check(len(seen) == baseline["counts"]["unique_development_measurement_artifacts"] + len(new_artifacts), "cumulative_unique_artifact_count")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    script_sha = digest(Path(__file__))
    summary_script_sha = script_sha
    reuse_evidence = None
    if reuse_summary_script is not None:
        old_path = Path(reuse_summary_script).resolve()
        if not reuse_summary_script_sha256 or digest(old_path) != reuse_summary_script_sha256:
            raise ValueError("Prior summary checker does not match its explicit byte pin")
        def summary_ast(path):
            tree = ast.parse(Path(path).read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_artifact_summary"), include_attributes=False)
        if summary_ast(old_path) != summary_ast(Path(__file__)):
            raise ValueError("Cannot reuse receipts from another summary extraction algorithm")
        summary_script_sha = reuse_summary_script_sha256
        reuse_evidence = {"previous_script_path": str(old_path), "previous_script_sha256": summary_script_sha,
                          "summary_function_ast_identical": True, "all_artifact_bytes_rehashed_again": True,
                          "legacy_cache_misses_allowed": False}
    index_path = receipt_dir / "index.json"
    index = read_json(index_path) if index_path.exists() else {"script_sha256": summary_script_sha, "baseline_sha256": baseline_sha, "artifacts": {}}
    if index["script_sha256"] != summary_script_sha or index["baseline_sha256"] != baseline_sha:
        raise ValueError("Summary receipt cache belongs to another script or trusted baseline")
    byte_count = 0
    for i, (relative, item) in enumerate(new_artifacts.items(), 1):
        row, identity = item["row"], item["identity"]
        path = directory / relative
        byte_count += path.stat().st_size
        hashed(path, row["artifact"]["sha256"], "delta_full_new_artifact_byte_hash")
        receipt_key = sha256(encoded({"artifact": row["artifact"], "identity": identity, "script_sha256": summary_script_sha})).hexdigest()
        receipt_path = receipt_dir / f"{receipt_key}.json"
        pointer = index["artifacts"].get(receipt_key)
        if pointer:
            hashed(receipt_path, pointer["sha256"], "delta_resumed_summary_receipt_pin")
            receipt = read_json(receipt_path)
            diagnostics["summary_receipts_reused"] += 1
        elif receipt_path.exists():
            receipt = read_json(receipt_path)
            diagnostics["orphan_summary_receipts_recovered"] += 1
        else:
            if reuse_evidence is not None:
                raise ValueError("A requested legacy summary receipt is missing; use a fresh receipt directory to compute it honestly")
            receipt = {"artifact": row["artifact"], "identity": identity, "script_sha256": script_sha,
                       "summary": _artifact_summary(path), "verified_at_utc": utc()}
            receipt["summary_sha256"] = sha256(encoded(receipt["summary"])).hexdigest()
            atomic_json(receipt_path, receipt, exclusive=True)
            diagnostics["new_artifact_summaries_streamed"] += 1
        check(receipt["artifact"] == row["artifact"] and receipt["identity"] == identity
              and receipt["script_sha256"] == summary_script_sha and receipt["summary_sha256"] == sha256(encoded(receipt["summary"])).hexdigest(),
              "delta_saved_summary_identity", relative)
        saved = receipt["summary"]
        metadata = saved["metadata"]
        check(metadata["identity"] == identity and all(_same(metadata[k], row[k]) for k in
            ("spec", "symbol", "timeframe", "tested_at", "status")), "delta_new_artifact_header_matches_journal", relative)
        check(all(_same(saved[k], row[k]) for k in ("training_metrics", "validation_metrics", "validation_score")),
              "delta_full_saved_metric_summaries_match_journal", relative)
        verified_pointer = {"sha256": digest(receipt_path)}
        if index["artifacts"].get(receipt_key) != verified_pointer:
            index["artifacts"][receipt_key] = verified_pointer
            atomic_json(index_path, index)
        if i % 500 == 0:
            print(json.dumps({"new_artifacts_checked": i, "total_new": len(new_artifacts), "discrepancies": len(errors)}), flush=True)
    checked_manifests, checked_pages, checked_csvs = {}, set(), set()

    def page_receipt(page, symbol, api_symbol):
        if page["wire_file"] in checked_pages:
            return
        checked_pages.add(page["wire_file"])
        check(_safe(page["safety"]) and page.get("authentication_used") is False and page["endpoint"] == ENDPOINT,
              "update_original_page_public_only", symbol)
        check(page.get("collector_sha256") == declaration["source_pins"]["data.py"], "update_page_collector_pin", symbol)
        hashed(page["wire_file"], page["file_sha256"], "update_original_wire_file_pin")
        raw = gzip.decompress(Path(page["wire_file"]).read_bytes())
        check(sha256(raw).hexdigest() == page["wire_sha256"], "update_decoded_wire_pin", symbol)
        response = json.loads(raw)
        request = page["request"]
        check(set(request) == {"ticks_history", "style", "granularity", "count", "start", "end", "req_id"}
              and request["ticks_history"] == api_symbol and request["style"] == "candles" and request["granularity"] == 60,
              "update_allowlisted_actual_symbol_request", symbol)
        check(response.get("req_id") == request["req_id"] and response.get("echo_req") == request,
              "update_original_request_response_identity", symbol)
        candles = response.get("candles", [])
        check(not response.get("error") and len(candles) == page["returned_rows"], "update_original_received_row_count", symbol)
        check(all(type(c["epoch"]) is int and all(math.isfinite(float(c[k])) and float(c[k]) > 0
              for k in ("open", "high", "low", "close")) and float(c["low"]) <= min(float(c["open"]), float(c["close"]))
              and float(c["high"]) >= max(float(c["open"]), float(c["close"])) for c in candles), "update_wire_finite_valid_ohlc", symbol)

    def chain(path, pin, symbol, api_symbol):
        anchor = declaration["datasets"][symbol]
        links = 0
        visited = set()
        while (str(path), pin) != (anchor["manifest_file"], anchor["manifest_sha256"]):
            key = (str(path), pin)
            if key in visited:
                raise ValueError("Cyclic update source lineage")
            visited.add(key)
            links += 1
            if key not in checked_manifests:
                hashed(path, pin, "update_ancestor_manifest_pin")
                row = read_json(Path(path))
                check(row["symbol"] == symbol and row["api_symbol"] == api_symbol and _safe(row["safety"])
                      and row.get("authentication_used") is False and row.get("collector_sha256") == declaration["source_pins"]["data.py"],
                      "update_ancestor_symbol_safety_collector", symbol)
                for field in ("file", "requested_file"):
                    check(Path(row[field]).exists(), "update_ancestor_source_snapshot_exists", row[field])
                for page in row.get("pages", []):
                    page_receipt(page, symbol, api_symbol)
                checked_manifests[key] = row
            row = checked_manifests[key]
            parent = row.get("parent_manifest")
            if not parent or not row.get("parent_manifest_sha256"):
                raise ValueError("Update source lineage does not reach trusted initial source")
            parent_row = read_json(Path(parent))
            check(parent_row["cutoff_exclusive_epoch"] <= row["cutoff_exclusive_epoch"], "update_source_cutoffs_monotone", symbol)
            path, pin = parent, row["parent_manifest_sha256"]
        check(Path(path).exists(), "update_lineage_reaches_existing_initial_anchor", symbol)
        return links

    updates = []
    for cycle in delta_cycles:
        update = cycle["market_update"]
        if not update or "manifest_file" not in update:
            updates.append({"cycle": cycle["number"], "status": "RECORDED_UPDATE_ERROR", "receipt": update})
            diagnostics["recorded_market_update_errors"] += 1
            continue
        hashed(update["manifest_file"], update["manifest_sha256"], "update_run_manifest_pin")
        manifest = read_json(Path(update["manifest_file"]))
        check(manifest.get("source") == "Official Deriv public M1 candles" and manifest.get("endpoint") == ENDPOINT
              and _safe(manifest["safety"]) and no_real_order_count(manifest.get("real_orders_sent"))
              and manifest.get("cost_model_verified") is False and manifest.get("collector_sha256") == declaration["source_pins"]["data.py"],
              "update_run_source_collector_safety", cycle["number"])
        check(set(manifest["symbols"]) == targets and update["statuses"] == {s: r["status"] for s, r in manifest["symbols"].items()},
              "update_ten_symbol_status_inventory", cycle["number"])
        symbols = audit_catalogue(manifest["catalogue"])
        rows = []
        for symbol, row in manifest["symbols"].items():
            check(row["api_symbol"] == symbols[symbol] == declaration["datasets"][symbol]["api_symbol"],
                  "update_independent_catalogue_symbol_binding", symbol)
            if row["status"] != "READY":
                diagnostics["recorded_update_symbol_errors"] += 1
                rows.append({"symbol": symbol, "status": row["status"]})
                continue
            for field, hash_field in (("file", "sha256"), ("requested_file", "requested_sha256")):
                pair = (row[field], row[hash_field])
                if pair not in checked_csvs:
                    hashed(*pair, "update_scoped_source_snapshot_byte_pin")
                    checked_csvs.add(pair)
            own = read_json(Path(row["manifest_file"]))
            check(all(_same(own[k], row[k]) for k in ("symbol", "api_symbol", "status", "file", "sha256",
                "requested_file", "requested_sha256", "parent_manifest", "parent_manifest_sha256", "pages", "safety")),
                "update_run_row_matches_own_manifest", symbol)
            links = chain(row["manifest_file"], row["manifest_sha256"], symbol, row["api_symbol"])
            rows.append({"symbol": symbol, "status": "READY", "requested_rows": row["requested_rows"],
                "source_file": row["requested_file"], "source_sha256": row["requested_sha256"],
                "lineage_links_to_initial_anchor": links, "new_network_pages": row["new_network_pages"]})
        updates.append({"cycle": cycle["number"], "manifest_file": update["manifest_file"],
                        "manifest_sha256": update["manifest_sha256"], "symbols": rows})
    for number, pin in all_cycle_pins.items():
        hashed(directory / "cycles" / f"{number:02d}.json", pin, "completed_journal_stable_during_delta_audit")
    for name, pin in declaration["source_pins"].items():
        hashed(ROOT / "research_campaign" / name, pin, "frozen_code_stable_during_delta_audit")
    end_state = read_json(directory / "state.json")
    validate_state(end_state)
    check(_safe(end_state["safety"]), "ending_safety_false")
    check(end_state["completed_cycles"] >= through, "ending_actual_count_not_reversed")
    count = {"baseline_completed_cycles": old_count, "through_completed_cycle": through,
             "delta_completed_cycles": through - old_count, "delta_candidate_attempts": sum(c["candidate_rows"] for c in delta_cycles),
             "delta_unique_new_artifacts": len(new_artifacts), "delta_new_measurement_rows": sum(c["new_measurements"] for c in delta_cycles),
             "delta_cache_hits": sum(c["cache_hits"] for c in delta_cycles), "new_compressed_artifact_bytes_rehashed": byte_count,
             "previous_artifact_bytes_rehashed": 0, "cumulative_unique_artifacts": len(seen),
             "cumulative_candidate_attempts": baseline["counts"]["candidate_attempt_rows"] + sum(c["candidate_rows"] for c in delta_cycles),
             "update_ancestor_manifests_verified": len(checked_manifests), "update_original_pages_verified": len(checked_pages),
             "scoped_update_csv_snapshots_rehashed": len(checked_csvs)}
    report = {"schema_version": 1, "audit_status": "PASS" if not errors else "FAIL", "started_at_utc": start,
        "completed_at_utc": utc(), "runtime_seconds": time.monotonic() - clock, "script_sha256": script_sha,
        "trusted_baseline": {"path": str(baseline_file), "sha256": baseline_sha}, "requested_snapshot": requested_snapshot,
        "state_at_audit_start": {k: state.get(k) for k in ("id", "status", "completed_cycles", "current_cycle", "final_accessed", "updated_at")},
        "state_at_audit_start_sha256": sha256(state_bytes).hexdigest(),
        "state_at_audit_end": {k: end_state.get(k) for k in ("status", "completed_cycles", "current_cycle", "final_accessed", "updated_at")},
        "declaration_sha256": state["declaration_sha256"], "counts": count, "cycles": delta_cycles,
        "market_update_source_audits": updates, "checks": dict(checks), "check_count": sum(checks.values()),
        "summary_receipts": {"directory": str(receipt_dir), "index_sha256": digest(index_path) if index_path.exists() else None,
                             "diagnostics": dict(diagnostics), "summary_algorithm_script_sha256": summary_script_sha,
                             "compatible_receipt_reuse_evidence": reuse_evidence}, "errors": errors, "safety": dict(SAFETY),
        "scope": "incremental completed-development journal, artifact byte/summary and update source-lineage integrity; no numerical strategy replay",
        "final_outcomes_opened": False, "profitability_tested_by_this_audit": False,
        "limitations": ["Earlier artifact bytes rely on the pinned baseline receipt; they were not reread.",
            "New saved metrics were matched to journals; no trade/equity result was numerically reconstructed.",
            "Scoped update CSV bytes and all intervening manifest/page pins were checked; ancestor CSV existence was checked without rehashing every older snapshot or reconstructing CSV rows from wire.",
            "Main search uses 13 fixed predeclared families and 32 parameter rounds, not autonomous invention of new families each cycle.",
            "Historical CFD costs/fills remain unverified and frozen original-source defects remain disqualifying."]}
    atomic_json(output_prefix.with_suffix(".json"), report)
    report_sha = digest(output_prefix.with_suffix(".json"))
    gaps = [c["start_gap_seconds"] for c in delta_cycles]
    note = (f"نتيجة التدقيق التراكمي: **{report['audit_status']}**؛ {len(errors)} اختلافات، و{report['check_count']:,} فحصًا. "
            f"النطاق المثبت الدورات **{old_count + 1}–{through}**، مرتبط بتقرير {old_count} السابق وبصمته `{baseline_sha}`. "
            f"لم تُعد قراءة ملفات القياس السابقة؛ عدد بايتاتها المقروءة مجددًا صفر.\n\n"
            f"راجعت {count['delta_candidate_attempts']:,} محاولة، و{count['delta_unique_new_artifacts']:,} ملف قياس جديد، و{count['delta_cache_hits']:,} إعادة استخدام. "
            f"لكل دورة 100 خلية × 13 محاولة؛ طابقت المواصفات والتقسيم وبصمات المصدر والكود وهويات aliases والتوقيت والعدّادات. "
            f"أعيد حساب بصمات {byte_count:,} بايت مضغوط جديد، وطابقت ملخصات التدريب والتحقق الأصلية بالسجلات دون إعادة حساب الاستراتيجية.\n\n"
            f"فاصل بدايات الدورات ضمن النطاق: {min(gaps) if gaps else None}–{max(gaps) if gaps else None} ثانية، مع تحقق موعد البدء والانتهاء وحد 900 ثانية. "
            f"الإجمالي المربوط حتى الدورة {through}: {count['cumulative_candidate_attempts']:,} محاولة و{count['cumulative_unique_artifacts']:,} قياس علمي مختلف؛ ليس إعلانًا باكتمال 32 دورة قبل إثباتها.\n\n"
            f"راجعت تحديثات السوق للدورات المحددة: {len(updates)} manifests، و{len(checked_manifests)} روابط مصدر، و{len(checked_pages)} صفحة أصلية، "
            f"و{len(checked_csvs)} بصمة CSV حديثة. تصل سلسلة المصدر إلى manifest البداية المجمد؛ الردود عامة دون توثيق حساب أو أوامر. "
            "الأسعار المحدثة منفصلة عن مدخلات الدراسة المجمدة. لم أعد بناء CSV لكل تحديث من الردود الأصلية، ولم أعد تجزئة محتوى كل snapshot أقدم.\n\n"
            "هذا تدقيق سلامة وربط، وليس numerical replay أو اختبار ربح. لم أفتح نتائج holdout أو أغير العامل أو الحالة أو الكود العلمي. "
            "العائلات الـ13 ثابتة قبل القياس؛ الدورات تغير إعدادات معلنة، ولا تثبت ابتكار عائلة جديدة كل 15 دقيقة. تكاليف CFD غير موثقة وعيوب المصدر المجمد محفوظة. "
            "LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false.\n\n"
            f"اكتمل الفحص في {report['completed_at_utc']}؛ ملف التفاصيل [JSON]({output_prefix.name}.json)، SHA256 `{report_sha}`.\n")
    output_prefix.with_suffix(".ar.md").write_text(note)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "data/single_frame_campaign_20261011")
    parser.add_argument("--baseline", type=Path, default=ROOT / "docs/single_frame_campaign_20261011/progress_audit_20261011.json")
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "work/progress_delta_audit_20261011")
    parser.add_argument("--receipt-dir", type=Path, default=ROOT / "work/progress_delta_artifact_receipts")
    parser.add_argument("--through-cycle", type=int)
    parser.add_argument("--requested-snapshot-json", help="Optional earlier scope-lock metadata, not an authority for completion")
    parser.add_argument("--reuse-summary-script", type=Path, help="Explicit pinned prior checker; permits reuse only when summary-extractor AST is identical")
    parser.add_argument("--reuse-summary-script-sha256")
    args = parser.parse_args()
    report = audit(args.state_dir, args.baseline, args.baseline_sha256, args.output_prefix, args.receipt_dir,
                   args.through_cycle, json.loads(args.requested_snapshot_json) if args.requested_snapshot_json else None,
                   args.reuse_summary_script, args.reuse_summary_script_sha256)
    print(json.dumps({k: report[k] for k in ("audit_status", "counts", "runtime_seconds", "check_count", "errors")}, indent=2))
    if report["audit_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
