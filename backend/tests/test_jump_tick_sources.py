"""Fabricated public-source files only; no historical quote reads or network."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts import verify_jump_tick_sources as verifier


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def key(request):
    return hashlib.sha256(json.dumps({"endpoint": verifier.ENDPOINT, "request": request},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture_day(root, pages=None, retry=None, empty=False):
    """One whole declared day with a tiny explicit observed population."""
    symbol, day = "BOOM600", "2025-10-09"
    start = int(datetime(2025, 10, 9, tzinfo=timezone.utc).timestamp()); end = start + 86400
    if pages is None:
        pages = [[]] if empty else [[(0, "100.0"), (1, "99.00")]]
    names = verifier.source_names(symbol, day)
    files = {name: root / label for name, label in names.items()}
    records, audits, errors, lineage, prices = [], [], [], [], {}
    cursor, duplicate_total = end - 1, 0
    for number, offsets in enumerate(pages):
        request = {"ticks_history": symbol, "count": 1000, "start": start,
                   "end": cursor, "style": "ticks", "req_id": 1000 + number}
        attempt = 1
        if retry and number == 0:
            error_wire = None
            error_hash = None
            error_type = "TimeoutError"
            if retry == "rate":
                error_wire = json.dumps({"req_id": request["req_id"], "error": {"code": "RateLimit", "message": "Synthetic"}})
                error_hash = hashlib.sha256(error_wire.encode()).hexdigest(); error_type = "RateLimitError"
            error = {"endpoint": verifier.ENDPOINT, "request": request, "request_key": key(request),
                     "attempt": 1, "error_type": error_type, "error": "Synthetic retry", "wire_sha256": error_hash}
            records.append({"endpoint": verifier.ENDPOINT, "request": request, "request_key": key(request),
                "attempt": 1, "received_at_utc": "2026-10-08T00:00:00+00:00", "response_wire": error_wire,
                "wire_sha256": error_hash, "error": error})
            audits.append({"endpoint": verifier.ENDPOINT, "request": request, "request_key": key(request),
                "attempt": 1, "accepted": False, "wire_sha256": error_hash, "recoverable": True, "error": error})
            errors.append(error); attempt = 2
        epochs = [start + offset for offset, _ in offsets]
        wire = json.dumps({"echo_req": request, "req_id": request["req_id"], "msg_type": "history",
                           "history": {"times": epochs, "prices": [quote for _, quote in offsets]}})
        wire_hash = hashlib.sha256(wire.encode()).hexdigest()
        stamp = datetime(2026, 10, 8, tzinfo=timezone.utc) + timedelta(seconds=len(records))
        records.append({"endpoint": verifier.ENDPOINT, "request": request, "request_key": key(request),
            "attempt": attempt, "received_at_utc": stamp.isoformat(), "response_wire": wire, "wire_sha256": wire_hash})
        duplicates = 0
        for offset, quote in offsets:
            if start + offset in prices:
                duplicates += 1
            else:
                prices[start + offset] = quote
        duplicate_total += duplicates
        audits.append({"endpoint": verifier.ENDPOINT, "request": request, "request_key": key(request),
            "attempt": attempt, "accepted": True, "wire_sha256": wire_hash, "rows": len(offsets),
            "equal_duplicates_removed": duplicates, "oldest_epoch": epochs[0] if epochs else None,
            "newest_epoch": epochs[-1] if epochs else None})
        if retry and number == 0:
            lineage.append({"error_index": 0, "successful_page_audit_index": 1, "request_key": key(request),
                            "successful_wire_sha256": wire_hash, "endpoint": verifier.ENDPOINT,
                            "start_epoch": start, "end_epoch": cursor})
        cursor = epochs[0] - 1 if epochs else start - 1
    write_lines(files["raw"], records); write_lines(files["audit"], audits)
    epochs = sorted(prices)
    files["clean"].write_text("epoch,quote\n" + "".join(f"{epoch},{prices[epoch]}\n" for epoch in epochs))
    missing = []; before = start - 1
    for after in [*epochs, end]:
        if after > before + 1:
            missing.append({"start_epoch": before + 1, "end_exclusive_epoch": after, "missing_seconds": after - before - 1})
        before = after
    manifest = {"symbol": symbol, "date": day, "start_epoch": start, "end_exclusive_epoch": end,
        "start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "end_exclusive_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "cadence_seconds": 1, "endpoint": verifier.ENDPOINT, "config_sha256": verifier.CONFIG_SHA256,
        "collector_file": "scripts/collect_spike_ticks.py", "collector_sha256": verifier.COLLECTOR_SHA256,
        "clean_file": names["clean"], "clean_sha256": verifier.digest(files["clean"]),
        "raw_pages_file": names["raw"], "raw_pages_sha256": verifier.digest(files["raw"]),
        "page_audit_file": names["audit"], "page_audit_sha256": verifier.digest(files["audit"]),
        "rows": len(prices), "expected_grid_rows": 86400, "missing_seconds": 86400 - len(prices), "gaps": missing,
        "first_epoch": epochs[0] if epochs else None, "last_epoch": epochs[-1] if epochs else None,
        "gap_free": len(prices) == 86400, "equal_duplicates_removed": duplicate_total,
        "fills_or_interpolations": False, "authentication_used": False, "safety": dict(verifier.SAFETY),
        "features_labels_or_tick_outcomes_computed": False, "request_errors": errors,
        "recovered_retry_lineage": lineage, "successful_page_count": len(pages), "normalization_valid": True,
        "completed_at_utc": "2026-10-08T01:00:00+00:00"}
    dump(files["manifest"], manifest)
    return {"files": files, "manifest": manifest, "raw": records, "audit": audits, "symbol": symbol, "day": day,
            "start": start, "end": end}


def refresh(fixture):
    """Allow adversarial content with internally refreshed hashes to test logic."""
    files, manifest = fixture["files"], fixture["manifest"]
    write_lines(files["raw"], fixture["raw"]); write_lines(files["audit"], fixture["audit"])
    for name, field in (("raw", "raw_pages_sha256"), ("audit", "page_audit_sha256"), ("clean", "clean_sha256")):
        manifest[field] = verifier.digest(files[name])
    dump(files["manifest"], manifest)


def verify(root, fixture):
    return verifier.verify_day(verifier.Audit(root), fixture["symbol"], fixture["day"],
                               datetime(2026, 10, 7, tzinfo=timezone.utc))


def test_no_production_modules_or_nonstdlib_computation_imported():
    tree = ast.parse(Path(verifier.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("app", "scripts", "numpy", "pandas"))
        if isinstance(node, ast.Import):
            assert all(not value.name.startswith(("app", "scripts", "numpy", "pandas")) for value in node.names)


def test_tiny_population_preserves_entire_missing_day_grid(tmp_path):
    fixture = fixture_day(tmp_path)
    result = verify(tmp_path, fixture)
    assert result["rows"] == 2 and result["missing_seconds"] == 86398
    assert result["gap_free"] is False
    assert result["successful_pages"] == result["attempts"] == 1
    assert result["equal_duplicates_removed"] == result["recovered_errors"] == 0


def test_empty_source_is_integrity_valid_but_not_gap_free(tmp_path):
    result = verify(tmp_path, fixture_day(tmp_path, empty=True))
    assert result["rows"] == 0 and result["missing_seconds"] == 86400 and result["gap_free"] is False


def test_multiple_pages_reconstruct_backward_cursor_and_sorted_csv(tmp_path):
    result = verify(tmp_path, fixture_day(tmp_path, pages=[[(2, "101")], [(0, "100")]]))
    assert result["successful_pages"] == 2 and result["rows"] == 2 and result["missing_seconds"] == 86398


@pytest.mark.parametrize("retry", ["transport", "rate"])
def test_exact_same_request_recovery_is_retained(tmp_path, retry):
    result = verify(tmp_path, fixture_day(tmp_path, retry=retry))
    assert result["recovered_errors"] == 1 and result["attempts"] == 2


def test_equal_duplicate_keeps_first_canonical_decimal_text(tmp_path):
    fixture = fixture_day(tmp_path, pages=[[(0, "100.0"), (0, "100.00"), (1, "99")]])
    result = verify(tmp_path, fixture)
    assert result["rows"] == 2 and result["equal_duplicates_removed"] == 1
    assert "100.0\n" in fixture["files"]["clean"].read_text()


def test_unequal_duplicate_is_rejected(tmp_path):
    fixture = fixture_day(tmp_path, pages=[[(0, "100"), (0, "101")]])
    with pytest.raises(ValueError, match="Unequal duplicate"):
        verify(tmp_path, fixture)


@pytest.mark.parametrize("which", ["raw", "audit", "clean"])
def test_changed_original_bytes_fail_before_normalization(tmp_path, which):
    fixture = fixture_day(tmp_path)
    with fixture["files"][which].open("a") as stream: stream.write(" ")
    with pytest.raises(ValueError, match="Changed input bytes"):
        verify(tmp_path, fixture)


@pytest.mark.parametrize("change", ["cursor", "reqid", "start", "count", "request_key", "wire_hash", "bounds",
    "page_count", "duplicate_count", "missing_range", "lineage", "unresolved", "semantic", "timestamp", "clean", "extra_request"])
def test_internally_rehashed_tampering_is_rejected(tmp_path, change):
    fixture = fixture_day(tmp_path, retry="transport")
    raw, pages, manifest = fixture["raw"], fixture["audit"], fixture["manifest"]
    if change in ("cursor", "reqid", "start", "count"):
        field = {"cursor": "end", "reqid": "req_id", "start": "start", "count": "count"}[change]
        raw[-1]["request"] = dict(raw[-1]["request"]); raw[-1]["request"][field] += 1
    elif change == "request_key": raw[-1]["request_key"] = "0" * 64
    elif change == "wire_hash": raw[-1]["wire_sha256"] = "0" * 64
    elif change == "bounds": pages[-1]["oldest_epoch"] += 1
    elif change == "page_count": manifest["successful_page_count"] += 1
    elif change == "duplicate_count": manifest["equal_duplicates_removed"] = 1
    elif change == "missing_range": manifest["gaps"][0]["start_epoch"] += 1
    elif change == "lineage": manifest["recovered_retry_lineage"] = []
    elif change == "unresolved": raw.pop(); pages.pop()
    elif change == "semantic": pages[0]["recoverable"] = False
    elif change == "timestamp": raw[-1]["received_at_utc"] = "2025-01-01T00:00:00+00:00"
    elif change == "clean": fixture["files"]["clean"].write_text("epoch,quote\n" + str(fixture["start"]) + ",100.00\n")
    elif change == "extra_request": raw.append(deepcopy(raw[-1])); pages.append(deepcopy(pages[-1]))
    refresh(fixture)
    with pytest.raises(ValueError): verify(tmp_path, fixture)


@pytest.mark.parametrize("key", verifier.FLAGS)
def test_saved_safety_gate_cannot_be_promoted(tmp_path, key):
    fixture = fixture_day(tmp_path); fixture["manifest"]["safety"][key] = True; refresh(fixture)
    with pytest.raises(ValueError, match=key): verify(tmp_path, fixture)


@pytest.mark.parametrize("key", ["authentication_used", "features_labels_or_tick_outcomes_computed", "fills_or_interpolations"])
def test_source_scope_must_remain_false(tmp_path, key):
    fixture = fixture_day(tmp_path); fixture["manifest"][key] = True; refresh(fixture)
    with pytest.raises(ValueError, match="False source claim"): verify(tmp_path, fixture)


@pytest.mark.parametrize("kind", ["duplicate_key", "nan", "bool_epoch", "fraction_epoch", "out_of_window", "unordered", "zero_quote", "nan_quote", "boolean_quote", "wrong_echo", "wrong_reqid", "ragged_arrays"])
def test_bad_wire_is_rejected_without_return_computation(tmp_path, kind):
    fixture = fixture_day(tmp_path); request = fixture["raw"][0]["request"]
    value = json.loads(fixture["raw"][0]["response_wire"])
    if kind == "duplicate_key": text = '{"req_id":1000,"req_id":1000}'
    elif kind == "nan": text = '{"history":NaN}'
    else:
        if kind == "bool_epoch": value["history"]["times"][0] = True
        elif kind == "fraction_epoch": value["history"]["times"][0] += .5
        elif kind == "out_of_window": value["history"]["times"][0] -= 1
        elif kind == "unordered": value["history"]["times"].reverse()
        elif kind == "zero_quote": value["history"]["prices"][0] = 0
        elif kind == "nan_quote": value["history"]["prices"][0] = "NaN"
        elif kind == "boolean_quote": value["history"]["prices"][0] = True
        elif kind == "wrong_echo": value["echo_req"]["style"] = "candles"
        elif kind == "wrong_reqid": value["req_id"] += 1
        elif kind == "ragged_arrays": value["history"]["prices"].pop()
        text = json.dumps(value)
    with pytest.raises(ValueError): verifier.wire_rows(text, request, verifier.Audit(tmp_path))


@pytest.mark.parametrize("label", ["../x", "/tmp/x", "./x", "a//b", "", 1])
def test_source_path_cannot_escape_or_alias(tmp_path, label):
    with pytest.raises(ValueError): verifier.Audit(tmp_path).path(label)


def test_source_symlink_refused(tmp_path):
    (tmp_path / "real").write_text("data"); (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValueError, match="symlink"): verifier.Audit(tmp_path).path("link")


@pytest.mark.parametrize("key", verifier.FLAGS)
def test_enabled_environment_blocks_before_metadata_access(monkeypatch, tmp_path, key):
    for name in verifier.FLAGS: monkeypatch.setenv(name, "false")
    monkeypatch.setenv(key, "true")
    monkeypatch.setattr(verifier, "frozen_contract", lambda *_: pytest.fail("metadata opened"))
    with pytest.raises(ValueError, match=key): verifier.inspect(verifier.Audit(tmp_path), "snapshot")


def test_complete_requires_explicit_authorization_before_input_access(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "frozen_contract", lambda *_: pytest.fail("metadata opened"))
    with pytest.raises(ValueError, match="sources-ready"):
        verifier.inspect(verifier.Audit(tmp_path), "complete")


def test_complete_requires_all722_manifests_before_quote_decoding(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "frozen_contract", lambda *_: {})
    monkeypatch.setattr(verifier, "verify_day", lambda *_: pytest.fail("quotes decoded"))
    with pytest.raises(ValueError, match="All722"):
        verifier.inspect(verifier.Audit(tmp_path), "complete", sources_ready=True)


def test_partial_snapshot_never_reports_completeness_pass_or_decodes_quotes(monkeypatch, tmp_path):
    fixture_day(tmp_path)
    monkeypatch.setattr(verifier, "frozen_contract", lambda *_: {"declared_at_utc": "2026-10-07T00:00:00+00:00"})
    monkeypatch.setattr(verifier, "verify_day", lambda *_: pytest.fail("quotes decoded"))
    monkeypatch.setattr(verifier, "wire_rows", lambda *_: pytest.fail("wire decoded"))
    result = verifier.inspect(verifier.Audit(tmp_path), "snapshot")
    assert result["passed"] is None and result["completeness_passed"] is False
    assert result["integrity_of_available_sources"] == "NOT TESTED"
    assert result["available_manifests"] == 1 and result["missing_manifests"] == 721
    assert result["sources"][0]["reported_rows_unverified"] == 2
    assert result["returns_features_labels_or_payoffs_computed"] is False


def test_recognized_transport_error_cannot_be_replaced_by_semantic_failure(tmp_path):
    fixture = fixture_day(tmp_path, retry="transport")
    fixture["raw"][0]["error"]["error_type"] = "ValueError"
    refresh(fixture)
    with pytest.raises(ValueError, match="transport exception"): verify(tmp_path, fixture)


def test_raw_audit_length_mismatch_is_rejected(tmp_path):
    fixture = fixture_day(tmp_path); fixture["audit"].append(deepcopy(fixture["audit"][0])); refresh(fixture)
    with pytest.raises(ValueError, match="attempt counts"): verify(tmp_path, fixture)


def test_interruption_resume_can_restart_attempt_numbering(tmp_path):
    fixture = fixture_day(tmp_path, retry="transport")
    fixture["raw"][-1]["attempt"] = 1; fixture["audit"][-1]["attempt"] = 1; refresh(fixture)
    result = verify(tmp_path, fixture)
    assert result["recovered_errors"] == 1


@pytest.mark.parametrize("actual,expected", [(True, 1), (False, 0), ({"index": False}, {"index": 0}),
    ([True], [1]), ({"a": 1, "extra": 2}, {"a": 1})])
def test_manifest_metadata_cannot_hide_type_or_schema_tampering(actual, expected):
    assert verifier.identical(actual, expected) is False


def test_retry_lineage_boolean_index_is_not_an_integer(tmp_path):
    fixture = fixture_day(tmp_path, retry="transport")
    fixture["manifest"]["recovered_retry_lineage"][0]["error_index"] = False; refresh(fixture)
    with pytest.raises(ValueError, match="retry lineage"): verify(tmp_path, fixture)


def contract_fixture(root, monkeypatch):
    collector = root / "scripts/collect_spike_ticks.py"; collector.parent.mkdir(parents=True)
    collector.write_text("# Immutable fabricated collector bytes; never executed.\n")
    collector_hash = verifier.digest(collector); monkeypatch.setattr(verifier, "COLLECTOR_SHA256", collector_hash)
    declaration = {"symbols": list(verifier.SYMBOLS), "dates": list(verifier.DATES),
        "expected_days_per_symbol": 361, "expected_sources": 722, "expected_1s_grid_rows": 62380800,
        "source_file": "scripts/collect_spike_ticks.py", "source_sha256": collector_hash,
        "features_labels_or_tick_outcomes_computed": False, "authentication_used": False,
        "account_or_order_requests": False, "full_continuous_tick_coverage": "NOT TESTED",
        "safety": dict(verifier.SAFETY), "declared_at_utc": "2026-10-07T00:00:00+00:00"}
    config = {"version": 1, "endpoint": verifier.ENDPOINT, "symbols": list(verifier.SYMBOLS),
        "dates": list(verifier.DATES), "page_size": 1000, "requested_delay_seconds": .5,
        "effective_min_request_spacing_seconds": .5, "cadence_seconds": 1,
        "collector_sha256": collector_hash, "safety": dict(verifier.SAFETY)}
    dp = root / verifier.DECLARATION; cp = root / verifier.DATA / "acquisition_config.json"
    dump(dp, declaration); dump(cp, config)
    monkeypatch.setattr(verifier, "DECLARATION_SHA256", verifier.digest(dp))
    monkeypatch.setattr(verifier, "CONFIG_SHA256", verifier.digest(cp))
    return declaration, {"declaration": dp, "config": cp, "collector": collector}


def test_exact_frozen_contract_reads_metadata_only(tmp_path, monkeypatch):
    expected, _ = contract_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(verifier, "wire_rows", lambda *_: pytest.fail("quotes decoded"))
    audit = verifier.Audit(tmp_path)
    assert verifier.frozen_contract(audit) == expected
    assert len(audit.inputs) == 3


@pytest.mark.parametrize("file", ["declaration", "config", "collector"])
def test_frozen_contract_detects_changed_input_bytes(tmp_path, monkeypatch, file):
    _, files = contract_fixture(tmp_path, monkeypatch)
    with files[file].open("a") as stream: stream.write(" ")
    with pytest.raises(ValueError, match="Changed input bytes"):
        verifier.frozen_contract(verifier.Audit(tmp_path))


def test_metadata_snapshot_lists_undeclared_manifest_without_claiming_pass(tmp_path, monkeypatch):
    _, _ = contract_fixture(tmp_path, monkeypatch)
    extra = tmp_path / verifier.DATA / "BOOM600_1900-01-01_manifest.json"
    extra.write_text("{}\n")
    result = verifier.inspect(verifier.Audit(tmp_path), "snapshot")
    assert result["unexpected_manifest_files"] == [str(extra.relative_to(tmp_path))]
    assert result["completeness_passed"] is False and result["passed"] is None


def test_cli_preserves_failure_report_and_hash_without_quote_access(tmp_path, monkeypatch):
    destination = tmp_path / "failed.json"
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "frozen_contract", lambda *_: pytest.fail("input opened before ready gate"))
    monkeypatch.setattr(sys, "argv", ["verify", "--mode", "complete", "--output", str(destination)])
    with pytest.raises(SystemExit) as stopped: verifier.main()
    assert stopped.value.code == 1
    report = json.loads(destination.read_text())
    assert report["passed"] is False and report["completeness_passed"] is False
    assert report["checks"] == 0 and "sources-ready" in report["errors"][0]["message"]
    assert destination.with_suffix(".sha256").read_text().strip() == verifier.digest(destination)


@pytest.mark.parametrize("sidecar", [False, True])
def test_cli_refuses_output_overwrite_before_audit(tmp_path, monkeypatch, sidecar):
    destination = tmp_path / "kept.json"
    existing = destination.with_suffix(".sha256") if sidecar else destination
    existing.write_text("preserved\n")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "inspect", lambda *_: pytest.fail("audit started before overwrite guard"))
    monkeypatch.setattr(sys, "argv", ["verify", "--mode", "snapshot", "--output", str(destination)])
    with pytest.raises(ValueError, match="overwrite"): verifier.main()
    assert existing.read_text() == "preserved\n"
