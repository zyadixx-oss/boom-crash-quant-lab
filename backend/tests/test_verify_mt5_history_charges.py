"""Artificial charge components only; never read actual private account receipts."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import stat

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/verify_mt5_history_charges.py"
spec = importlib.util.spec_from_file_location("private_charge_verifier", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
T = 1700000001000


def rows():
    return [
        dict(zip(mod.RAW_FIELDS, map(str, (
            T, "Boom 500 Index", 0, 0, "0.5", -2, 0, "-0.1",
            0, 0, 0, 0, 0, 0, "true", "true", "true")))),
        dict(zip(mod.RAW_FIELDS, map(str, (
            T+1, "Boom 500 Index", 1, 1, 2, 0, -1, 0,
            0, 0, 0, 0, 0, 0, "true", "true", "true")))),
    ]


def metadata():
    return {
        **mod.FIXED, "status": "CHARGE_COMPONENTS_OBSERVED_NOT_PROVEN", "failure_reason": "",
        "server": "Deriv-Demo", "company": "Deriv (SVG) LLC", "account_mode": "DEMO",
        "capture_time_gmt": "2026.10.13 01:02:03 GMT", "account_currency_current": "USD",
        "account_currency_capture_time_gmt": "2026.10.13 01:02:00 GMT", "account_currency_error": "0",
        "input_start_server_epoch": "1700000000", "input_end_server_epoch": "0",
        "requested_start_server_epoch": "1700000000", "requested_end_server_epoch": "1710000000",
        "server_clock_at_query_epoch": "1710000000", "server_clock_error": "0",
        "end_resolved_from_timecurrent": "true", "history_select_called": "true",
        "history_select_call_count": "1", "history_select_returned": "true", "history_select_error": "0",
        "history_deals_total_called": "true", "history_deals_total_returned": "2",
        "history_deals_total_error": "0", "processed_records": "2", "matched_records": "2",
        "raw_rows_written": "2", "unrelated_trade_records": "0", "nontrade_records": "0",
        "unclassified_records": "0", "property_error_rows": "0", "numeric_invalid_rows": "0",
        "out_of_requested_bounds_rows": "0", "backward_matched_time_rows": "0",
        "first_property_error": "0", "last_property_error": "0",
        "first_matched_time_msc": str(T), "last_matched_time_msc": str(T+1),
        "minimum_matched_time_msc": str(T), "maximum_matched_time_msc": str(T+1),
        "context_stable_before_query": "true", "context_stable_after_query": "true",
        "context_stable_before_loop": "true", "context_stable_after_loop": "true",
        "selected_list_loop_complete": "true", "query_elapsed_ms": "3", "loop_elapsed_ms": "4",
        "connection_wait_limit_seconds": "30", "connection_wait_elapsed_ms": "0",
        "terminal_trade_allowed_observed": "false", "mql_trade_allowed_observed": "false",
        "guard_error": "0", "write_error": "0", "raw_file": "synthetic_charges.tsv",
        "raw_partial_file": "", "raw_attempts": "1", "matching_rows_with_errors_retained": "true",
        "private_financial_records_exported": "true",
    }


def tsv(header, items):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(items)
    return buffer.getvalue().encode()


def save(tmp_path, meta=None, raw=None):
    folder = tmp_path.resolve()
    meta = metadata() if meta is None else meta
    raw = rows() if raw is None else raw
    receipt = folder / "synthetic_receipt.tsv"
    receipt.write_bytes(tsv(("field", "value"), meta.items()))
    (folder / "synthetic_charges.tsv").write_bytes(tsv(mod.RAW_FIELDS, ([row[key] for key in mod.RAW_FIELDS] for row in raw)))
    return receipt


def diagnostic(meta):
    meta.update(status="FAILED_PROPERTY_OR_NUMERIC",
                failure_reason="retained_diagnostics_include_unavailable_invalid_or_outside_rows")


def test_hand_computed_signed_components_and_per_volume_ranges(tmp_path):
    report = mod.verify_receipt(save(tmp_path))
    assert report["status"] == "COMPONENTS_VALIDATED_NOT_COST_PROOF"
    item = report["observed_components_by_symbol"]["Boom 500 Index"]
    assert item["observed_rows"] == 2 and item["observed_volume_sum"] == "2.5"
    assert item["components"]["commission"]["observed_native_sum"] == "-2"
    assert item["components"]["commission"]["per_volume_min"] == "-4"
    assert item["components"]["commission"]["per_volume_max"] == "0"
    assert item["components"]["swap"]["per_volume_min"] == "-0.5"
    assert item["components"]["fee"]["per_volume_min"] == "-0.2"
    assert item["components"]["fee"]["zero_count"] == 1
    assert item["components"]["fee"]["nonzero_count"] == 1
    assert report["cost_verified"] is report["execution_verified"] is False


def test_all_zero_components_are_not_a_verified_free_cost_model(tmp_path):
    raw = rows()
    for row in raw:
        row.update(commission="0", swap="-0", fee="0")
    report = mod.verify_receipt(save(tmp_path, raw=raw))
    assert report["cost_verified"] is False
    assert report["zero_components_prove_all_dates_or_brokers_free"] is False
    assert report["observed_components_by_symbol"]["Boom 500 Index"]["components"]["swap"]["zero_count"] == 2


def test_empty_matching_raw_is_valid_selected_diagnostic_not_global_absence(tmp_path):
    meta = metadata()
    meta.update(status="NO_MATCHING_CHARGES_OBSERVED_NOT_PROVEN",
                history_deals_total_returned="3", processed_records="3", matched_records="0",
                raw_rows_written="0", unrelated_trade_records="2", nontrade_records="1",
                first_matched_time_msc="NONE", last_matched_time_msc="NONE",
                minimum_matched_time_msc="NONE", maximum_matched_time_msc="NONE",
                private_financial_records_exported="false")
    report = mod.verify_receipt(save(tmp_path, meta, []))
    assert report["status"] == "EMPTY_MATCHING_SELECTION_NOT_GLOBAL_ABSENCE"
    assert report["observed_matching_components_available"] is False
    assert report["observed_components_by_symbol"] == {}


def test_positive_credit_and_multiple_symbols_preserve_native_signs(tmp_path):
    raw = rows()
    raw[1].update(symbol="Crash 1000 Index", commission="1")
    report = mod.verify_receipt(save(tmp_path, raw=raw))
    assert report["observed_components_by_symbol"]["Crash 1000 Index"]["components"]["commission"]["per_volume_min"] == "0.5"


def test_error_row_is_retained_but_excluded_from_usable_component_stats(tmp_path):
    meta, raw = metadata(), rows()
    diagnostic(meta)
    meta.update(property_error_rows="1", numeric_invalid_rows="1",
                first_property_error="4303", last_property_error="4303")
    raw[0].update(commission="NOT_AVAILABLE", commission_error_code="4303", numeric_valid="false")
    report = mod.verify_receipt(save(tmp_path, meta, raw))
    assert report["status"] == "DIAGNOSTIC_VALIDATED_NOT_USABLE"
    assert report["counts"]["usable_retained_rows"] == 1
    assert report["observed_matching_components_available"] is None


def test_nonfinite_diagnostic_is_never_used_as_a_component(tmp_path):
    meta, raw = metadata(), rows()
    diagnostic(meta)
    meta["numeric_invalid_rows"] = "1"
    raw[0].update(commission="NaN", numeric_valid="false")
    report = mod.verify_receipt(save(tmp_path, meta, raw))
    assert report["counts"]["usable_retained_rows"] == 1


@pytest.mark.parametrize("field,bad", [("volume", "0"), ("commission", "Infinity"), ("deal_entry", "4")])
def test_claimed_numeric_valid_row_rejects_invalid_values(tmp_path, field, bad):
    raw = rows()
    raw[0][field] = bad
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, raw=raw))


def test_unknown_symbol_and_non_buy_sell_type_are_rejected(tmp_path):
    for changes in ({"symbol": "Other Index"}, {"deal_type": "2"}):
        raw = rows()
        raw[0].update(changes)
        with pytest.raises(mod.VerificationError):
            mod.verify_receipt(save(tmp_path, raw=raw))


def test_decimal_exponent_must_be_representable_by_source_double(tmp_path):
    raw = rows()
    raw[0]["commission"] = "1e999999"
    with pytest.raises(mod.VerificationError, match="BINARY64_DOMAIN_VIOLATION"):
        mod.verify_receipt(save(tmp_path, raw=raw))


def test_reserved_prefix_is_not_accepted_as_a_source_output(tmp_path):
    receipt = save(tmp_path)
    reserved = receipt.with_name("CON_receipt.tsv")
    reserved.write_bytes(receipt.read_bytes())
    with pytest.raises(mod.VerificationError, match="UNSAFE_PREFIX"):
        mod.verify_receipt(reserved)


def test_currency_capture_must_remain_nonidentity_current_metadata(tmp_path):
    meta = metadata()
    meta["account_currency_current"] = "SYNTHETIC_SECRET_PERSON"
    with pytest.raises(mod.VerificationError, match="UNSAFE_CURRENCY_TEXT"):
        mod.verify_receipt(save(tmp_path, meta))


def test_error_code_cannot_coexist_with_successful_numeric_value(tmp_path):
    raw = rows()
    raw[0]["fee_error_code"] = "4303"
    with pytest.raises(mod.VerificationError, match="ERROR_VALUE_MISMATCH"):
        mod.verify_receipt(save(tmp_path, raw=raw))


def test_row_flags_and_metadata_diagnostic_totals_must_agree(tmp_path):
    meta, raw = metadata(), rows()
    raw[0]["numeric_valid"] = "false"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, meta, raw))
    meta = metadata()
    meta["numeric_invalid_rows"] = "1"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, meta))


def test_partition_counts_and_retention_truth_are_checked(tmp_path):
    for changes in ({"processed_records": "3"}, {"matched_records": "3"},
                    {"matching_rows_with_errors_retained": "false"}):
        meta = metadata()
        meta.update(changes)
        with pytest.raises(mod.VerificationError):
            mod.verify_receipt(save(tmp_path, meta))


def test_private_financial_truth_flag_cannot_hide_exported_rows(tmp_path):
    meta = metadata()
    meta["private_financial_records_exported"] = "false"
    with pytest.raises(mod.VerificationError, match="PRIVATE_FINANCIAL_FLAG_MISMATCH"):
        mod.verify_receipt(save(tmp_path, meta))


def test_post_context_currency_and_eight_safety_gates_are_required(tmp_path):
    for changes in ({"context_stable_after_loop": "false"}, {"account_currency_error": "4303"},
                    {"account_currency_scope": "HISTORICAL"}, {"safety_upper_live_allowed": "true"},
                    {"orders_sent": "1"}):
        meta = metadata()
        meta.update(changes)
        with pytest.raises(mod.VerificationError):
            mod.verify_receipt(save(tmp_path, meta))


def test_latched_row_context_failure_cannot_claim_stable_post_loop(tmp_path):
    raw = rows()
    raw[0]["row_context_stable"] = "false"
    with pytest.raises(mod.VerificationError, match="LATCHED_CONTEXT_MISMATCH"):
        mod.verify_receipt(save(tmp_path, raw=raw))


def test_server_time_semantics_max_span_and_end_milliseconds(tmp_path):
    meta, raw = metadata(), rows()
    meta["requested_time_semantics"] = "UTC"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, meta))
    meta = metadata()
    meta.update(input_start_server_epoch="1", requested_start_server_epoch="1")
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, meta))
    meta = metadata()
    raw[1]["time_msc"] = str(1710000000*1000+999)
    meta.update(last_matched_time_msc=raw[1]["time_msc"], maximum_matched_time_msc=raw[1]["time_msc"])
    assert mod.verify_receipt(save(tmp_path, meta, raw))["status"] == "COMPONENTS_VALIDATED_NOT_COST_PROOF"
    raw[1]["time_msc"] = str(1710000000*1000+1000)
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, meta, raw))


def test_raw_hash_pin_and_source_hash_are_checked(tmp_path):
    receipt = save(tmp_path)
    raw_hash = mod.sha((tmp_path/"synthetic_charges.tsv").read_bytes())
    assert mod.verify_receipt(receipt, expected_raw_sha256=raw_hash)["expected_raw_sha256_checked"] is True
    with pytest.raises(mod.VerificationError, match="EXPECTED_RAW_HASH_MISMATCH"):
        mod.verify_receipt(receipt, expected_raw_sha256="0"*64)
    changed = tmp_path/"changed.mq5"
    changed.write_bytes(mod.SOURCE.read_bytes()+b"\n")
    with pytest.raises(mod.VerificationError, match="SOURCE_PIN_MISMATCH"):
        mod.verify_receipt(receipt, source_path=changed)


def test_duplicate_metadata_and_unexpected_raw_columns_rejected(tmp_path):
    receipt = save(tmp_path)
    meta = list(metadata().items())
    meta[-1] = meta[0]
    receipt.write_bytes(tsv(("field", "value"), meta))
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(receipt)
    receipt = save(tmp_path)
    (tmp_path/"synthetic_charges.tsv").write_bytes(tsv((*mod.RAW_FIELDS, "ticket"), []))
    with pytest.raises(mod.VerificationError, match="INVALID_HEADER"):
        mod.verify_receipt(receipt)


def test_raw_traversal_symlink_and_truncation_are_rejected(tmp_path):
    meta = metadata()
    meta["raw_file"] = "../other.tsv"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, meta))
    receipt = save(tmp_path)
    raw = tmp_path/"synthetic_charges.tsv"
    original = raw.read_bytes()
    raw.unlink()
    alias_target = tmp_path/"other.tsv"
    alias_target.write_bytes(original)
    raw.symlink_to(alias_target)
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(receipt)
    raw.unlink()
    raw.write_bytes(original.rstrip(b"\n"))
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(receipt)


def test_changed_raw_during_verification_is_detected(tmp_path, monkeypatch):
    receipt = save(tmp_path)
    original_reader = mod.read_regular
    seen = 0

    def changed(target, limit):
        nonlocal seen
        data = original_reader(target, limit)
        if Path(target).name == "synthetic_charges.tsv":
            seen += 1
            if seen == 2:
                return data+b"\n"
        return data

    monkeypatch.setattr(mod, "read_regular", changed)
    with pytest.raises(mod.VerificationError, match="RAW_CHANGED"):
        mod.verify_receipt(receipt)


def test_cli_stdout_is_status_only_private_json_is_exclusive_0600(tmp_path, monkeypatch, capsys):
    private = tmp_path.resolve()/"private"
    private.mkdir()
    monkeypatch.setattr(mod, "PRIVATE_ROOT", private)
    receipt = save(private)
    output = private/"report.json"
    args = ["--receipt", str(receipt), "--output", str(output)]
    assert mod.main(args) == 0
    captured = capsys.readouterr()
    assert captured.out == "COMPONENTS_VALIDATED_NOT_COST_PROOF\n" and captured.err == ""
    original = output.read_bytes()
    assert json.loads(original)["cost_verified"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert mod.main(args) == 2 and capsys.readouterr().out == "VERIFY_FAILED\n"
    assert output.read_bytes() == original


def test_cli_rejects_public_output_and_does_not_echo_private_payload(tmp_path, monkeypatch, capsys):
    private = tmp_path.resolve()/"private"
    private.mkdir()
    monkeypatch.setattr(mod, "PRIVATE_ROOT", private)
    meta = metadata()
    meta["account_login"] = "SYNTHETIC_SECRET"
    receipt = save(private, meta)
    output = private/"failure.json"
    assert mod.main(["--receipt", str(receipt), "--output", str(output)]) == 2
    assert capsys.readouterr().out == "VERIFY_FAILED\n"
    assert "SYNTHETIC_SECRET" not in output.read_text()
    public = tmp_path.resolve()/"public.json"
    assert mod.main(["--receipt", str(receipt), "--output", str(public)]) == 2
    assert capsys.readouterr().out == "VERIFY_FAILED\n" and not public.exists()
