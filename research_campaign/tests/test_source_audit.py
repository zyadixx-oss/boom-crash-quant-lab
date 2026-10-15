"""Synthetic receipt fixtures only; no fixture establishes market performance."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research_campaign import source_audit as a


def candle(epoch, price=100.12345678901234):
    return {"epoch": epoch, "open": price, "high": price + .2,
            "low": price - .2, "close": price + .01}


def receipt(tmp_path, rows, *, end=599, request=None):
    request = request or {"ticks_history": "B500", "style": "candles", "granularity": 60,
                          "count": 5000, "start": 0, "end": end, "req_id": 1}
    response = {"req_id": request["req_id"], "echo_req": request,
                "candles": rows, "fixture_only": True}
    raw = json.dumps(response).encode()
    path = tmp_path / (hashlib.sha256(raw).hexdigest() + ".json.gz")
    path.write_bytes(gzip.compress(raw, mtime=0))
    return {"wire_file": str(path), "file_sha256": a._hash(path),
            "wire_sha256": hashlib.sha256(raw).hexdigest(), "request": request,
            "endpoint": a.ENDPOINT, "authentication_used": False, "safety": dict(a.SAFETY)}


def source(tmp_path, rows=None, *, start=0, end=600, raw_rows=None):
    rows = rows if rows is not None else [candle(i * 60) for i in range(10)]
    path = tmp_path / "canonical.csv"
    pd.DataFrame(rows, columns=a.FIELDS).to_csv(path, index=False)
    return {"symbol": "BOOM500", "api_symbol": "B500", "requested_file": str(path),
            "requested_sha256": a._hash(path), "requested_start_epoch": start,
            "cutoff_exclusive_epoch": end,
            "pages": [receipt(tmp_path, raw_rows if raw_rows is not None else rows, end=end - 1)]}


def catalogue(tmp_path, api_symbol="B500"):
    request = {"active_symbols": "brief", "req_id": 1}
    raw = json.dumps({"req_id": 1, "echo_req": request,
                      "active_symbols": [{"underlying_symbol_name": "Boom 500 Index",
                                          "underlying_symbol": api_symbol}]}).encode()
    path = tmp_path / "catalogue.json.gz"
    path.write_bytes(gzip.compress(raw, mtime=0))
    return {"wire_file": str(path), "file_sha256": a._hash(path),
            "wire_sha256": hashlib.sha256(raw).hexdigest(), "request": request,
            "endpoint": a.ENDPOINT, "authentication_used": False, "safety": dict(a.SAFETY)}


def test_exact_original_floats_and_effective_interval(tmp_path):
    row = source(tmp_path)
    result = a.audit_source_row(row, catalogue_provenance=catalogue(tmp_path))
    assert result["status"] == "PASS"
    assert result["catalogue_binding_checked"] is True
    assert result["canonical_rows"] == 10
    assert result["effective_start_epoch"] == 0
    assert result["effective_end_exclusive_epoch"] == 600
    assert result["internal_missing_minutes"] == result["internal_invalid_rows"] == 0
    assert result["performance_accessed"] is False
    assert result["gap_free_segment_selected"] is False


def test_boundary_loss_does_not_remove_internal_gap(tmp_path):
    row = source(tmp_path, [candle(i) for i in (120, 180, 300, 360, 420)])
    result = a.audit_source_row(row)
    assert (result["effective_start_epoch"], result["effective_end_exclusive_epoch"]) == (120, 480)
    assert result["leading_boundary_loss_minutes"] == 2
    assert result["trailing_boundary_loss_minutes"] == 2
    assert result["internal_missing_minutes"] == 1
    assert result["requested_missing_minutes"] == 5
    assert result["internal_missing_ranges"] == [{"start_epoch": 240, "end_exclusive_epoch": 300,
                                                   "missing_minutes": 1}]


@pytest.mark.parametrize("offgrid,expected_invalid", [(61, 0), (181, 1)])
def test_offgrid_rows_recorded_without_fabricated_candles(tmp_path, offgrid, expected_invalid):
    rows = [candle(i) for i in (120, 180, 240, 300)]
    row = source(tmp_path, rows, raw_rows=[candle(offgrid), *rows])
    result = a.audit_source_row(row)
    assert result["canonical_rows"] == 4
    assert result["excluded_counts"]["off_grid"] == 1
    assert result["internal_invalid_rows"] == expected_invalid


def test_equal_original_duplicates_deduplicated(tmp_path):
    rows = [candle(i * 60) for i in range(10)]
    result = a.audit_source_row(source(tmp_path, rows, raw_rows=[*rows, rows[0]]))
    assert result["equal_original_duplicates_removed"] == 1


def test_conflicting_original_duplicates_rejected(tmp_path):
    rows = [candle(i * 60) for i in range(10)]
    row = source(tmp_path, rows, raw_rows=[*rows, candle(0, 101)])
    with pytest.raises(ValueError, match="Conflicting original"):
        a.audit_source_row(row)


def test_modified_csv_with_new_hash_still_rejected_against_wire(tmp_path):
    row = source(tmp_path)
    frame = pd.read_csv(row["requested_file"], float_precision="round_trip")
    frame.loc[1, "close"] += .001
    frame.to_csv(row["requested_file"], index=False)
    row["requested_sha256"] = a._hash(row["requested_file"])
    with pytest.raises(ValueError, match="differs from canonical"):
        a.audit_source_row(row)


@pytest.mark.parametrize("mutation,pattern", [
    (lambda r: r.update(requested_sha256="wrong"), "normalized source hash"),
    (lambda r: r["pages"][0].update(file_sha256="wrong"), "compressed wire hash"),
    (lambda r: r["pages"][0].update(wire_sha256="wrong"), "decoded wire hash"),
    (lambda r: r["pages"][0].update(endpoint="wss://untrusted.invalid"), "public-only receipt"),
    (lambda r: r["pages"][0].update(authentication_used=True), "public-only receipt"),
    (lambda r: r["pages"][0]["safety"].update(LIVE_TRADING=True), "public-only receipt"),
    (lambda r: r["pages"][0]["safety"].update(LIVE_TRADING=0), "public-only receipt"),
    (lambda r: r["pages"][0]["request"].update(req_id=2), "request identity"),
    (lambda r: r.update(symbol="BTC"), "ten named"),
])
def test_mutated_identity_or_hash_fail_closed(tmp_path, mutation, pattern):
    row = source(tmp_path)
    mutation(row)
    with pytest.raises(ValueError, match=pattern):
        a.audit_source_row(row)


@pytest.mark.parametrize("extra", [{"authorize": "synthetic_not_a_token"}, {"buy": "synthetic"}])
def test_receipt_request_allowlist_blocks_account_fields(tmp_path, extra):
    row = source(tmp_path)
    request = {**row["pages"][0]["request"], **extra}
    row["pages"] = [receipt(tmp_path, [candle(i * 60) for i in range(10)], request=request)]
    with pytest.raises(ValueError, match="nonallowlisted"):
        a.audit_source_row(row)


def test_wrong_api_symbol_page_rejected(tmp_path):
    row = source(tmp_path)
    request = {**row["pages"][0]["request"], "ticks_history": "R100"}
    row["pages"] = [receipt(tmp_path, [candle(i * 60) for i in range(10)], request=request)]
    with pytest.raises(ValueError, match="wrong symbol"):
        a.audit_source_row(row)


@pytest.mark.parametrize("bad", [
    {"epoch": 1.5}, {"epoch": True}, {"low": 0}, {"close": float("nan")},
    {"high": 90}, {"low": 110}, {"open": "invalid"}, {"open": True},
])
def test_malformed_received_candles_fail_closed(tmp_path, bad):
    row = source(tmp_path)
    original = [candle(i * 60) for i in range(10)]
    original[0].update(bad)
    row["pages"] = [receipt(tmp_path, original)]
    with pytest.raises(ValueError):
        a.audit_source_row(row)


def test_provider_boundary_exclusion_is_verified(tmp_path):
    row = source(tmp_path)
    boundary = receipt(tmp_path, [candle(1200)], end=599)
    boundary["excluded_provider_boundary_response"] = True
    row["pages"].append(boundary)
    result = a.audit_source_row(row)
    assert result["excluded_provider_boundary_rows"] == 1
    assert result["canonical_rows"] == 10


def test_falsely_declared_boundary_cannot_hide_actual_rows(tmp_path):
    row = source(tmp_path)
    row["pages"][0]["excluded_provider_boundary_response"] = True
    with pytest.raises(ValueError, match="inside its requested window"):
        a.audit_source_row(row)


def test_unmarked_offwindow_provider_page_rejected(tmp_path):
    row = source(tmp_path)
    row["pages"].append(receipt(tmp_path, [candle(1200)], end=599))
    with pytest.raises(ValueError, match="exceeds its requested end"):
        a.audit_source_row(row)


def test_pinned_parent_lineage_reconstructs_resumed_cached_history(tmp_path):
    old = source(tmp_path, [candle(i * 60) for i in range(5)], end=300)
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(old))
    row = source(tmp_path)
    row["pages"] = [receipt(tmp_path, [candle(i * 60) for i in range(5, 10)])]
    row.update(parent_manifest=str(parent), parent_manifest_sha256=a._hash(parent))
    result = a.audit_source_row(row)
    assert result["manifest_chain_length"] == 2
    assert result["canonical_rows"] == result["original_received_rows"] == 10
    row["parent_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="parent manifest hash"):
        a.audit_source_row(row)


def test_missing_ancestor_originals_cannot_claim_reconstructed_history(tmp_path):
    row = source(tmp_path)
    row["pages"] = [receipt(tmp_path, [candle(i * 60) for i in range(5, 10)])]
    with pytest.raises(ValueError, match="differs from canonical"):
        a.audit_source_row(row)


def test_catalogue_prevents_other_instrument_substitution(tmp_path):
    row = source(tmp_path)
    with pytest.raises(ValueError, match="independently resolved public catalogue"):
        a.audit_source_row(row, catalogue_provenance=catalogue(tmp_path, "R100"))


def test_empty_source_still_requires_original_wire_and_stays_empty(tmp_path):
    row = source(tmp_path, [])
    result = a.audit_source_row(row)
    assert result["canonical_rows"] == 0
    assert result["effective_start_epoch"] is result["effective_end_exclusive_epoch"] is None
    assert result["requested_missing_minutes"] == 10


def test_audit_is_deterministic(tmp_path):
    row = source(tmp_path)
    assert a.audit_source_row(row) == a.audit_source_row(deepcopy(row))
