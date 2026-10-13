"""Synthetic protocol fixtures only; no actual account metadata or platform calls."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import stat

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/verify_mt5_history_availability.py"
spec = importlib.util.spec_from_file_location("private_history_verifier", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    return {
        "schema_version": "1", "scope": "ACCOUNT_HISTORY_AVAILABILITY_COUNTS_ONLY",
        "status": "COUNTS_OBSERVED_NOT_PROVEN", "failure_reason": "",
        "server": "Deriv-Demo", "company": "Deriv (SVG) LLC", "account_mode": "DEMO",
        "capture_time_gmt": "2026.10.13 01:02:03 GMT",
        "requested_time_semantics": "server_time", "requested_bounds_are_utc": "false",
        "server_timezone_offset_verified": "false",
        "input_start_server_epoch": "1700000000", "input_end_server_epoch": "0",
        "requested_start_server_epoch": "1700000000",
        "requested_end_server_epoch": "1710000000", "server_clock_at_query_epoch": "1710000000",
        "server_clock_error": "0",
        "server_clock_semantics": "TimeCurrent_last_known_existing_server_time",
        "end_resolved_from_timecurrent": "true", "max_span_days": "400",
        "selection_bounds_semantics": "HistorySelect_server_time_bounds_no_UTC_conversion",
        "history_select_called": "true", "history_select_call_count": "1",
        "history_select_returned": "true", "history_select_error": "0",
        "history_deals_total_called": "true", "history_deals_total_returned": "7",
        "history_deals_total_error": "0", "history_orders_total_called": "true",
        "history_orders_total_returned": "3", "history_orders_total_error": "0",
        "counts_accepted": "true", "context_stable_before_query": "true",
        "context_stable_after_query": "true", "query_elapsed_ms": "3",
        "connection_wait_limit_seconds": "30", "connection_wait_elapsed_ms": "0",
        "terminal_trade_allowed_observed": "false", "mql_trade_allowed_observed": "false",
        "guard_error": "0", "selection_scope": "ALL_ACCOUNT_HISTORY_IN_REQUESTED_RANGE_NOT_SYMBOL_FILTERED",
        "selection_completeness": "NOT_PROVEN", "whole_history_completeness": "NOT_PROVEN",
        "same_server_same_mode_account_switch_detection": "false",
        "context_comparison_scope": "server_company_account_mode_only",
        "boom_crash_relation_verified": "false", "deal_count_is_trade_fill_count": "false",
        "record_type_classification_performed": "false",
        "live_trading": "false", "ready_for_live": "false", "live_allowed": "false",
        "opened_trades": "false", "safety_upper_live_trading": "false",
        "safety_upper_ready_for_live": "false", "safety_upper_live_allowed": "false",
        "safety_upper_opened_trades": "false", "costs_verified": "false",
        "historical_costs_verified": "false", "execution_verified": "false",
        "cfd_fills_verified": "false", "prices_exported": "false", "pnl_exported": "false",
        "tickets_exported": "false", "individual_records_exported": "false",
        "personal_data_exported": "false", "account_login_name_balance_recorded": "false",
        "credentials_recorded": "false", "login_or_account_creation_attempted": "false",
        "orders_sent": "0", "output_file": "synthetic_availability.tsv",
        "limitation": "Counts describe selected account-history lists only; no symbol relation, "
        "trade-fill classification, costs, prices, profit, or history completeness is established.",
    }


def encoded(values, *, header=("field", "value"), rows=None):
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(values.items() if rows is None else rows)
    return text.getvalue().encode()


def save(tmp_path, values=None):
    target = tmp_path.resolve() / "synthetic_availability.tsv"
    target.write_bytes(encoded(fixture() if values is None else values))
    return target


def cli_paths(tmp_path, monkeypatch, values=None):
    private = tmp_path.resolve() / "private"
    private.mkdir()
    monkeypatch.setattr(mod, "PRIVATE_ROOT", private)
    return save(private, values), private / "report.json"


def test_positive_counts_only_establish_selected_list_availability(tmp_path):
    report = mod.verify_receipt(save(tmp_path))
    assert report["status"] == "COUNTS_VALIDATED"
    assert report["availability"] is True
    assert report["counts"] == {"deals": 7, "orders": 3}
    assert report["trade_fill_count_verified"] is False
    assert report["historical_costs_verified"] is False
    assert report["execution_verified"] is False
    assert report["receipt_producer_identity_verified"] is False


def test_zero_is_not_evidence_of_absence_at_all_brokers(tmp_path):
    values = fixture()
    values.update(history_deals_total_returned="0", history_orders_total_returned="0")
    report = mod.verify_receipt(save(tmp_path, values))
    assert report["availability"] is False
    assert report["availability_determined"] is True
    assert report["zero_count_proves_all_brokers_absence"] is False
    assert report["whole_history_completeness"] == "NOT_PROVEN"


def test_failed_selection_never_reads_previous_selected_counts(tmp_path):
    values = fixture()
    values.update(status="FAILED_SELECTION_OR_COUNTS",
                  failure_reason="selection_return_error_or_nonnegative_counts_check_failed",
                  history_select_returned="false", history_select_error="4401", counts_accepted="false")
    for label in ("deals", "orders"):
        values["history_" + label + "_total_called"] = "false"
        values["history_" + label + "_total_returned"] = "NOT_QUERIED"
    report = mod.verify_receipt(save(tmp_path, values))
    assert report["availability"] is None
    assert report["counts"] == {"deals": None, "orders": None}


def test_count_error_keeps_availability_unknown(tmp_path):
    values = fixture()
    values.update(status="FAILED_SELECTION_OR_COUNTS",
                  failure_reason="selection_return_error_or_nonnegative_counts_check_failed",
                  history_deals_total_error="4401", counts_accepted="false")
    assert mod.verify_receipt(save(tmp_path, values))["availability"] is None


def test_disconnected_receipt_has_no_resolved_window_or_count(tmp_path):
    values = fixture()
    values.update(status="FAILED_DISCONNECTED",
                  failure_reason="existing_server_connection_unavailable_after_bounded_wait",
                  server="", company="", account_mode="UNKNOWN",
                  requested_end_server_epoch="NOT_RESOLVED", server_clock_at_query_epoch="NOT_READ",
                  end_resolved_from_timecurrent="false", history_select_called="false",
                  history_select_call_count="0", history_select_returned="false",
                  history_deals_total_called="false", history_deals_total_returned="NOT_QUERIED",
                  history_orders_total_called="false", history_orders_total_returned="NOT_QUERIED",
                  counts_accepted="false", context_stable_before_query="false",
                  context_stable_after_query="false", query_elapsed_ms="0",
                  connection_wait_elapsed_ms="30000")
    report = mod.verify_receipt(save(tmp_path, values))
    assert report["status"] == "NO_USABLE_COUNTS"
    assert report["availability"] is None
    assert report["counts"] == {"deals": None, "orders": None}


def test_claimed_success_requires_disabled_trading_after_query(tmp_path):
    values = fixture()
    values["terminal_trade_allowed_observed"] = "true"
    with pytest.raises(mod.VerificationError, match="POST_QUERY_PROTECTION_FAILED"):
        mod.verify_receipt(save(tmp_path, values))


def test_counts_cannot_be_claimed_without_history_selection(tmp_path):
    values = fixture()
    values.update(history_select_called="false", history_select_call_count="0",
                  history_select_returned="false", context_stable_before_query="false",
                  context_stable_after_query="false", query_elapsed_ms="0")
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))


def test_context_change_discards_acceptance_of_observed_counts(tmp_path):
    values = fixture()
    values.update(status="FAILED_ACCOUNT_CONTEXT",
                  failure_reason="server_company_or_account_mode_changed_during_query",
                  context_stable_after_query="false", counts_accepted="false")
    report = mod.verify_receipt(save(tmp_path, values))
    assert report["availability"] is None
    values.update(status="COUNTS_OBSERVED_NOT_PROVEN", failure_reason="", counts_accepted="true")
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))


@pytest.mark.parametrize("bad", ["-1", "true", "1.5"])
def test_counts_require_nonnegative_integers(tmp_path, bad):
    values = fixture()
    values["history_deals_total_returned"] = bad
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))


@pytest.mark.parametrize("field", ["safety_upper_live_allowed", "personal_data_exported", "orders_sent"])
def test_safety_and_no_personal_or_trading_claims_are_mandatory(tmp_path, field):
    values = fixture()
    values[field] = "1" if field == "orders_sent" else "true"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))


def test_exact_header_and_unique_field_set(tmp_path):
    target = save(tmp_path)
    target.write_bytes(encoded(fixture(), header=("value", "field")))
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(target)
    rows = list(fixture().items())
    rows[5] = ("server", "Deriv-Demo")
    target.write_bytes(encoded({}, rows=rows))
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(target)


def test_extra_private_field_and_truncated_file_rejected(tmp_path):
    values = fixture()
    values["account_login"] = "SYNTHETIC_SECRET"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))
    target = save(tmp_path)
    target.write_bytes(target.read_bytes().rstrip(b"\n"))
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(target)


def test_server_time_cannot_be_relabelled_utc(tmp_path):
    values = fixture()
    values["requested_time_semantics"] = "UTC"
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))


def test_span_over_400_days_and_future_end_rejected(tmp_path):
    values = fixture()
    values.update(input_start_server_epoch="1", requested_start_server_epoch="1")
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))
    values = fixture()
    values.update(input_end_server_epoch="1710000001", requested_end_server_epoch="1710000001",
                  end_resolved_from_timecurrent="false")
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(save(tmp_path, values))


def test_explicit_past_end_is_valid_without_utc_conversion(tmp_path):
    values = fixture()
    values.update(input_end_server_epoch="1709999999", requested_end_server_epoch="1709999999",
                  end_resolved_from_timecurrent="false")
    assert mod.verify_receipt(save(tmp_path, values))["requested_end_server_epoch"] == 1709999999


def test_no_overwrite_and_stdout_contains_only_status(tmp_path, monkeypatch, capsys):
    receipt, output = cli_paths(tmp_path, monkeypatch)
    assert mod.main(["--receipt", str(receipt), "--output", str(output)]) == 0
    assert capsys.readouterr().out == "COUNTS_VALIDATED\n"
    original = output.read_bytes()
    assert json.loads(original)["counts"] == {"deals": 7, "orders": 3}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert mod.main(["--receipt", str(receipt), "--output", str(output)]) == 2
    assert capsys.readouterr().out == "VERIFY_FAILED\n"
    assert output.read_bytes() == original


def test_invalid_secret_content_never_reaches_stdout_or_stderr(tmp_path, monkeypatch, capsys):
    values = fixture()
    values["server"] = "SYNTHETIC_SECRET_ACCOUNT_IDENTIFIER"
    receipt, output = cli_paths(tmp_path, monkeypatch, values)
    assert mod.main(["--receipt", str(receipt), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "VERIFY_FAILED\n" and captured.err == ""
    assert "SYNTHETIC_SECRET" not in output.read_text()


def test_output_outside_private_root_is_rejected(tmp_path, monkeypatch, capsys):
    receipt, _ = cli_paths(tmp_path, monkeypatch)
    outside = tmp_path.resolve() / "public-report.json"
    assert mod.main(["--receipt", str(receipt), "--output", str(outside)]) == 2
    assert not outside.exists()
    assert capsys.readouterr().out == "VERIFY_FAILED\n"


def test_symlink_receipt_and_output_ancestor_rejected(tmp_path, monkeypatch):
    receipt, _ = cli_paths(tmp_path, monkeypatch)
    alias = receipt.with_name("alias_availability.tsv")
    alias.symlink_to(receipt)
    with pytest.raises(mod.VerificationError):
        mod.verify_receipt(alias)
    linked = receipt.parent / "linked"
    linked.symlink_to(tmp_path.resolve(), target_is_directory=True)
    with pytest.raises(mod.VerificationError):
        mod.private_path(linked / "report.json", output=True)


def test_source_pin_is_required(tmp_path):
    receipt = save(tmp_path)
    altered = tmp_path / "altered.mq5"
    altered.write_bytes(mod.SOURCE.read_bytes() + b"\n")
    with pytest.raises(mod.VerificationError, match="SOURCE_PIN_MISMATCH"):
        mod.verify_receipt(receipt, source_path=altered)


def test_input_change_during_verification_is_detected(tmp_path, monkeypatch):
    receipt = save(tmp_path)
    original_reader = mod.read_regular
    reads = 0

    def changing_reader(target, limit):
        nonlocal reads
        data = original_reader(target, limit)
        if Path(target) == receipt:
            reads += 1
            if reads == 2:
                return data + b"\n"
        return data

    monkeypatch.setattr(mod, "read_regular", changing_reader)
    with pytest.raises(mod.VerificationError, match="RECEIPT_CHANGED"):
        mod.verify_receipt(receipt)
