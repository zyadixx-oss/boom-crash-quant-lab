"""Offline private charge-component verification; stdout contains a status token only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "work/private_history_availability_20261013"
SOURCE = ROOT / "scripts/mt5/InspectResearchHistoryCharges.mq5"
SOURCE_SHA256 = "20b1163673b98fd04e4e6da4d66853cfa48ee43007f22f6000a3e29fb675592c"
FIELDS = frozenset("""
schema_version scope status failure_reason server company account_mode capture_time_gmt
account_currency_current account_currency_scope account_currency_capture_time_gmt account_currency_error
historical_currency_applicability_verified requested_time_semantics requested_bounds_are_utc
server_timezone_offset_verified input_start_server_epoch input_end_server_epoch requested_start_server_epoch
requested_end_server_epoch server_clock_at_query_epoch server_clock_error server_clock_semantics
end_resolved_from_timecurrent max_span_days selection_bounds_semantics matched_time_bounds_semantics
history_select_called history_select_call_count history_select_returned history_select_error
history_deals_total_called history_deals_total_returned history_deals_total_error maximum_selected_records
processed_records matched_records raw_rows_written unrelated_trade_records nontrade_records unclassified_records
property_error_rows numeric_invalid_rows out_of_requested_bounds_rows backward_matched_time_rows
first_property_error last_property_error first_matched_time_msc last_matched_time_msc minimum_matched_time_msc
maximum_matched_time_msc context_stable_before_query context_stable_after_query context_stable_before_loop
context_stable_after_loop selected_list_loop_complete query_elapsed_ms loop_elapsed_ms
connection_wait_limit_seconds connection_wait_elapsed_ms terminal_trade_allowed_observed
mql_trade_allowed_observed guard_error write_error raw_file raw_partial_file private_output allowed_symbols
allowed_deal_types record_order_preserved deduplication_applied raw_retention_scope raw_attempts
unclassified_record_details_retained matching_rows_with_errors_retained current_cost_completeness
historical_cost_completeness history_selection_completeness same_server_same_mode_account_switch_detection
context_comparison_scope all_period_fees_established actual_slippage_evidence historical_contract_mapping_verified
live_trading ready_for_live live_allowed opened_trades safety_upper_live_trading safety_upper_ready_for_live
safety_upper_live_allowed safety_upper_opened_trades costs_verified historical_costs_verified execution_verified
cfd_fills_verified prices_exported pnl_exported tickets_exported position_order_ids_exported
direct_account_identity_fields_exported private_financial_records_exported account_login_name_balance_recorded
credentials_recorded login_or_account_creation_attempted orders_sent limitation
""".split())
RAW_FIELDS = (
    "time_msc", "symbol", "deal_type", "deal_entry", "volume", "commission", "swap", "fee",
    "time_error_code", "entry_error_code", "volume_error_code", "commission_error_code",
    "swap_error_code", "fee_error_code", "numeric_valid", "time_in_requested_server_bounds",
    "row_context_stable",
)
SYMBOLS = frozenset(f"{side} {index} Index" for side in ("Boom", "Crash")
                    for index in (300, 500, 600, 900, 1000))
FIXED = {
    "schema_version": "1", "scope": "PRIVATE_ALLOWED_SYMBOL_HISTORY_CHARGE_COMPONENTS_ONLY",
    "account_currency_scope": "CURRENT_NOT_HISTORICAL", "requested_time_semantics": "server_time",
    "server_clock_semantics": "TimeCurrent_last_known_existing_server_time", "max_span_days": "400",
    "selection_bounds_semantics": "HistorySelect_server_time_bounds_no_UTC_conversion",
    "matched_time_bounds_semantics": "inclusive_server_seconds_milliseconds_end_plus999",
    "maximum_selected_records": "50000", "private_output": "true",
    "allowed_symbols": "Boom/Crash exact Index names for 300,500,600,900,1000",
    "allowed_deal_types": "BUY_SELL_ONLY", "record_order_preserved": "true",
    "raw_retention_scope": "SINGLE_SELECTION_MATCHING_ROWS_ONLY",
    "current_cost_completeness": "NOT_PROVEN", "historical_cost_completeness": "NOT_PROVEN",
    "history_selection_completeness": "NOT_PROVEN",
    "context_comparison_scope": "server_company_account_mode_current_currency_only",
    "orders_sent": "0",
    "limitation": "Private matching BUY/SELL deal charge components only; no prices, PnL, identity, "
                 "pairing, full-period costs, historical currency applicability, completeness, "
                 "broker-fill or slippage validation.",
}
FALSE_FIELDS = """
historical_currency_applicability_verified requested_bounds_are_utc server_timezone_offset_verified
deduplication_applied unclassified_record_details_retained same_server_same_mode_account_switch_detection
all_period_fees_established actual_slippage_evidence historical_contract_mapping_verified
live_trading ready_for_live live_allowed opened_trades safety_upper_live_trading safety_upper_ready_for_live
safety_upper_live_allowed safety_upper_opened_trades costs_verified historical_costs_verified
execution_verified cfd_fills_verified prices_exported pnl_exported tickets_exported position_order_ids_exported
direct_account_identity_fields_exported account_login_name_balance_recorded credentials_recorded
login_or_account_creation_attempted
""".split()
FIXED.update(dict.fromkeys(FALSE_FIELDS, "false"))
SUCCESS = {"CHARGE_COMPONENTS_OBSERVED_NOT_PROVEN", "NO_MATCHING_CHARGES_OBSERVED_NOT_PROVEN"}
REASONS = {
    "CHARGE_COMPONENTS_OBSERVED_NOT_PROVEN": {""}, "NO_MATCHING_CHARGES_OBSERVED_NOT_PROVEN": {""},
    "FAILED_INPUT": {"invalid_start_end_or_connection_wait", "invalid_or_future_server_window_or_span_over_400_days"},
    "FAILED_UNSAFE": {"terminal_or_script_trading_not_disabled_or_guard_unavailable"},
    "FAILED_DISCONNECTED": {"existing_server_connection_unavailable_after_bounded_wait", "terminal_disconnected"},
    "FAILED_ACCOUNT_CONTEXT": {"unavailable_or_non_deriv_server_context", "unsupported_account_mode",
                               "server_company_or_account_mode_changed_during_query",
                               "currency_unavailable_unsafe_or_changed"},
    "FAILED_STOPPED": {"script_stopped_before_query", "script_stopped_during_capture", "script_stopped_at_capture_completion"},
    "FAILED_SELECTION_OR_COUNT": {"history_selection_or_count_unavailable"},
    "FAILED_RECORD_LIMIT": {"selected_list_exceeds_50000_no_records_read"},
    "FAILED_FILE_WRITE": {"cannot_create_private_raw_staging", "raw_staging_incomplete_retained",
                          "raw_publication_failed_partial_retained"},
    "FAILED_PROPERTY_OR_NUMERIC": {"retained_diagnostics_include_unavailable_invalid_or_outside_rows"},
}


class VerificationError(ValueError):
    """Fixed codes only; never interpolate private metadata or numeric values."""


def require(condition, code):
    if not condition:
        raise VerificationError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_symlinks(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK_PATH")
    return path


def read_regular(path, limit):
    path = no_symlinks(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "NOT_REGULAR_FILE")
            data = stream.read(limit + 1)
        require(len(data) <= limit, "INPUT_TOO_LARGE")
        return data
    except OSError:
        raise VerificationError("INPUT_UNAVAILABLE") from None


def integer(value, code="INVALID_INTEGER", *, signed=False, maximum=2**63-1):
    pattern = r"0|-?[1-9][0-9]{0,18}" if signed else r"0|[1-9][0-9]{0,18}"
    require(isinstance(value, str) and re.fullmatch(pattern, value) is not None, code)
    result = int(value)
    require(abs(result) <= maximum, code)
    return result


def boolean(value):
    require(value in ("true", "false"), "INVALID_BOOLEAN")
    return value == "true"


def table(data, header, max_rows):
    try:
        text = data.decode("utf-8-sig")
        require(text.endswith("\n") and "\x00" not in text, "TRUNCATED_TSV")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
        require(next(reader, None) == list(header), "INVALID_HEADER")
        rows = list(reader)
        require(len(rows) <= max_rows and all(len(row) == len(header) for row in rows), "INVALID_ROWS")
        require(all(len(value) <= 2048 and not any(c in value for c in "\t\r\n")
                    for row in rows for value in row), "INVALID_CELL")
        return rows
    except (UnicodeError, csv.Error):
        raise VerificationError("INVALID_TSV") from None


def timestamp(value):
    if value == "NOT_AVAILABLE":
        return None
    return integer(value, signed=True)


def decimal_value(value):
    if value == "NOT_AVAILABLE":
        return None
    require(re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value)
            or value.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"), "INVALID_DECIMAL_TEXT")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise VerificationError("INVALID_DECIMAL_TEXT") from None
    if result.is_finite() and result:
        require(-324 <= result.adjusted() <= 308 and
                abs(result) <= Decimal("1.7976931348623157e308"), "BINARY64_DOMAIN_VIOLATION")
    return result


def verify_receipt(receipt_path, *, source_path=None, expected_raw_sha256=None):
    receipt_path = Path(receipt_path)
    name_match = re.fullmatch(r"([A-Za-z0-9_-]{1,64})_receipt\.tsv", receipt_path.name)
    require(name_match is not None, "INVALID_RECEIPT_NAME")
    prefix = name_match.group(1)
    require(prefix.lower() not in {"con", "prn", "aux", "nul"} and
            re.fullmatch(r"(?:com|lpt)[1-9]", prefix, re.IGNORECASE) is None, "UNSAFE_PREFIX")
    verifier_source = read_regular(Path(__file__), 100_000)
    source_path = SOURCE if source_path is None else Path(source_path)
    source = read_regular(source_path, 100_000)
    require(sha(source) == SOURCE_SHA256, "SOURCE_PIN_MISMATCH")
    receipt_bytes = read_regular(receipt_path, 100_000)
    pairs = table(receipt_bytes, ("field", "value"), 106)
    meta = dict(pairs)
    require(len(pairs) == len(meta) == 106 and set(meta) == FIELDS, "EXACT_METADATA_REQUIRED")
    require(all(meta[key] == value for key, value in FIXED.items()), "FIXED_PROTOCOL_MISMATCH")
    require(meta["status"] in REASONS and meta["failure_reason"] in REASONS[meta["status"]],
            "STATUS_REASON_MISMATCH")
    require(not meta["account_currency_current"] or
            re.fullmatch(r"[A-Za-z0-9]{3,12}", meta["account_currency_current"]) is not None,
            "UNSAFE_CURRENCY_TEXT")
    require(bool(meta["account_currency_current"]) ==
            (meta["account_currency_capture_time_gmt"] != "NOT_CAPTURED"), "CURRENCY_CAPTURE_MISMATCH")
    try:
        datetime.strptime(meta["capture_time_gmt"], "%Y.%m.%d %H:%M:%S GMT")
        if meta["account_currency_capture_time_gmt"] != "NOT_CAPTURED":
            datetime.strptime(meta["account_currency_capture_time_gmt"], "%Y.%m.%d %H:%M:%S GMT")
    except ValueError:
        raise VerificationError("INVALID_CAPTURE_TIME") from None

    raw_name, part_name = meta["raw_file"], meta["raw_partial_file"]
    require(not raw_name or raw_name == prefix + "_charges.tsv", "RAW_BASENAME_MISMATCH")
    require(not part_name or part_name == prefix + "_charges.part.tsv", "PARTIAL_BASENAME_MISMATCH")
    require(not (raw_name and part_name), "CONFLICTING_RAW_FILES")
    retained_name = raw_name or part_name
    raw_path = receipt_path.parent / retained_name if retained_name else None
    raw_bytes = read_regular(raw_path, 32_000_000) if raw_path is not None else None
    rows = [dict(zip(RAW_FIELDS, row)) for row in table(raw_bytes, RAW_FIELDS, 50000)] if raw_bytes is not None else []
    if expected_raw_sha256 is not None:
        require(isinstance(expected_raw_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_raw_sha256) is not None
                and raw_bytes is not None and sha(raw_bytes) == expected_raw_sha256, "EXPECTED_RAW_HASH_MISMATCH")

    n = {key: integer(meta[key]) for key in (
        "processed_records", "matched_records", "raw_rows_written", "unrelated_trade_records",
        "nontrade_records", "unclassified_records", "property_error_rows", "numeric_invalid_rows",
        "out_of_requested_bounds_rows", "backward_matched_time_rows", "first_property_error",
        "last_property_error", "account_currency_error", "server_clock_error", "history_select_error",
        "history_deals_total_error", "guard_error", "write_error", "query_elapsed_ms", "loop_elapsed_ms",
        "connection_wait_limit_seconds", "connection_wait_elapsed_ms")}
    b = {key: boolean(meta[key]) for key in (
        "history_select_called", "history_select_returned", "history_deals_total_called",
        "end_resolved_from_timecurrent", "context_stable_before_query", "context_stable_after_query",
        "context_stable_before_loop", "context_stable_after_loop", "selected_list_loop_complete",
        "terminal_trade_allowed_observed", "mql_trade_allowed_observed",
        "matching_rows_with_errors_retained", "private_financial_records_exported")}
    require(n["connection_wait_limit_seconds"] <= 30, "INVALID_WAIT_LIMIT")
    require(integer(meta["history_select_call_count"]) == integer(meta["raw_attempts"])
            == int(b["history_select_called"]), "SELECT_CALL_COUNT")
    require(b["context_stable_before_query"] == b["history_select_called"], "QUERY_GUARD_SEQUENCE")
    require(b["history_select_called"] or (not b["history_select_returned"]
            and n["history_select_error"] == 0 and not b["context_stable_after_query"]), "UNREQUESTED_SELECTION")
    expected_count_call = b["history_select_returned"] and n["history_select_error"] == 0 and b["history_select_called"]
    require(b["history_deals_total_called"] == expected_count_call, "COUNT_CALL_SEQUENCE")
    total = integer(meta["history_deals_total_returned"], maximum=2**31-1) if b["history_deals_total_called"] else None
    require(total is not None or (meta["history_deals_total_returned"] == "NOT_QUERIED"
            and n["history_deals_total_error"] == 0), "UNQUERIED_COUNT")
    require(n["raw_rows_written"] == len(rows) <= n["matched_records"] <= n["processed_records"], "ROW_COUNT_MISMATCH")
    require(n["processed_records"] == sum(n[key] for key in (
        "matched_records", "unrelated_trade_records", "nontrade_records", "unclassified_records")), "PARTITION_COUNT_MISMATCH")
    require((total is not None and n["processed_records"] <= total) or
            (total is None and n["processed_records"] == 0), "PROCESSED_TOTAL_MISMATCH")
    require(total is None or total <= 50000 or n["processed_records"] == 0, "RECORD_LIMIT_VIOLATION")
    require(b["private_financial_records_exported"] == (len(rows) > 0), "PRIVATE_FINANCIAL_FLAG_MISMATCH")
    require(b["matching_rows_with_errors_retained"] == (raw_path is not None and len(rows) == n["matched_records"]),
            "RETENTION_FLAG_MISMATCH")
    require(not b["context_stable_before_loop"] or (expected_count_call and total is not None
            and total <= 50000 and n["history_deals_total_error"] == 0 and b["context_stable_after_query"]),
            "LOOP_GUARD_SEQUENCE")
    require(raw_path is None or b["context_stable_before_loop"], "RAW_WITHOUT_LOOP_GUARD")
    require(not b["selected_list_loop_complete"] or (total is not None and n["processed_records"] == total
            and b["context_stable_before_loop"] and b["context_stable_after_loop"]), "INCOMPLETE_LOOP_CLAIM")

    start = integer(meta["input_start_server_epoch"])
    input_end = integer(meta["input_end_server_epoch"])
    require(integer(meta["requested_start_server_epoch"]) == start, "START_MISMATCH")
    clock = None if meta["server_clock_at_query_epoch"] == "NOT_READ" else integer(meta["server_clock_at_query_epoch"])
    end = None if meta["requested_end_server_epoch"] == "NOT_RESOLVED" else integer(meta["requested_end_server_epoch"])
    require(b["end_resolved_from_timecurrent"] == (input_end == 0 and clock is not None and clock > 0), "END_RESOLUTION_FLAG")
    if clock is not None:
        require(end == (clock if input_end == 0 else input_end), "END_RESOLUTION_MISMATCH")
    else:
        require(end is None and not b["history_select_called"], "UNREAD_CLOCK")
    if b["history_select_called"]:
        require(n["server_clock_error"] == 0 and clock is not None and end is not None
                and 0 < start < end <= clock and end-start <= 400*86400, "INVALID_SERVER_WINDOW")
        require(re.fullmatch(r"Deriv(?:SVG|BVI|FX|V|Mauritius|Cayman)?(?:[- ._][A-Za-z0-9 ._()-]+)?",
                             meta["server"], re.IGNORECASE) is not None, "UNIDENTIFIED_SERVER")
        require(re.fullmatch(r"Deriv(?:[ .(][A-Za-z0-9 .(),-]*)?", meta["company"], re.IGNORECASE)
                is not None and meta["account_mode"] in ("DEMO", "LIVE"), "UNIDENTIFIED_CONTEXT")
    require(not raw_path or (re.fullmatch(r"[A-Za-z0-9]{3,12}", meta["account_currency_current"]) is not None
            and meta["account_currency_capture_time_gmt"] != "NOT_CAPTURED"), "CURRENCY_UNAVAILABLE")

    usable, valid_times, property_rows, invalid_rows, outside_rows, row_context_failures = [], [], 0, 0, 0, 0
    raw_error_codes = []
    for row in rows:
        require(row["symbol"] in SYMBOLS, "UNDECLARED_SYMBOL")
        require(integer(row["deal_type"]) in (0, 1), "NON_BUY_SELL_ROW")
        errors = {key: integer(row[key], maximum=2**31-1) for key in RAW_FIELDS[8:14]}
        raw_error_codes.extend(value for value in errors.values() if value)
        time_value = timestamp(row["time_msc"])
        entry = timestamp(row["deal_entry"])
        values = {key: decimal_value(row[key]) for key in ("volume", "commission", "swap", "fee")}
        for field, error_key in (("time_msc", "time_error_code"), ("deal_entry", "entry_error_code"),
                                 ("volume", "volume_error_code"), ("commission", "commission_error_code"),
                                 ("swap", "swap_error_code"), ("fee", "fee_error_code")):
            require(errors[error_key] == 0 or row[field] == "NOT_AVAILABLE", "ERROR_VALUE_MISMATCH")
        properties_ok = (not any(errors.values()) and time_value is not None and entry is not None
                         and all(value is not None for value in values.values()))
        numeric = properties_ok and all(value.is_finite() for value in values.values()) and values["volume"] > 0 and 0 <= entry <= 3
        in_bounds = (time_value is not None and errors["time_error_code"] == 0
                     and end is not None and start*1000 <= time_value <= end*1000+999)
        row_context = boolean(row["row_context_stable"])
        require(boolean(row["numeric_valid"]) == numeric, "NUMERIC_FLAG_MISMATCH")
        require(boolean(row["time_in_requested_server_bounds"]) == in_bounds, "BOUNDS_FLAG_MISMATCH")
        property_rows += not properties_ok
        invalid_rows += not numeric
        outside_rows += not in_bounds
        row_context_failures += not row_context
        if time_value is not None and errors["time_error_code"] == 0:
            valid_times.append(time_value)
        if numeric and in_bounds and row_context:
            usable.append((row["symbol"], values))
    retained_all = len(rows) == n["matched_records"]
    for key, observed in (("property_error_rows", property_rows+n["unclassified_records"]),
                          ("numeric_invalid_rows", invalid_rows), ("out_of_requested_bounds_rows", outside_rows)):
        require(n[key] == observed if retained_all else n[key] >= observed, "DIAGNOSTIC_COUNT_MISMATCH")
    require(n["property_error_rows"] <= n["unclassified_records"]+n["matched_records"]
            and n["numeric_invalid_rows"] <= n["matched_records"]
            and n["out_of_requested_bounds_rows"] <= n["matched_records"], "UNBOUNDED_DIAGNOSTIC_COUNTS")
    if retained_all:
        require(n["backward_matched_time_rows"] == sum(b < a for a, b in zip(valid_times, valid_times[1:])),
                "BACKWARD_COUNT_MISMATCH")
        expected_times = (valid_times[0], valid_times[-1], min(valid_times), max(valid_times)) if valid_times else ("NONE",)*4
        for key, value in zip(("first_matched_time_msc", "last_matched_time_msc",
                               "minimum_matched_time_msc", "maximum_matched_time_msc"), expected_times):
            require(meta[key] == str(value), "MATCHED_TIME_MISMATCH")
    if n["unclassified_records"] == 0 and retained_all:
        require(n["first_property_error"] == (raw_error_codes[0] if raw_error_codes else 0)
                and n["last_property_error"] == (raw_error_codes[-1] if raw_error_codes else 0), "PROPERTY_ERROR_BOUND_MISMATCH")
    require(not row_context_failures or not b["context_stable_after_loop"], "LATCHED_CONTEXT_MISMATCH")

    clean = meta["status"] in SUCCESS
    if clean:
        require(b["selected_list_loop_complete"] and raw_name and retained_all and
                all(b[key] for key in ("context_stable_before_query", "context_stable_after_query",
                                      "context_stable_before_loop", "context_stable_after_loop"))
                and not b["terminal_trade_allowed_observed"] and not b["mql_trade_allowed_observed"]
                and all(n[key] == 0 for key in ("guard_error", "write_error", "account_currency_error",
                                              "property_error_rows", "numeric_invalid_rows", "out_of_requested_bounds_rows"))
                and len(usable) == len(rows), "SUCCESS_GUARD_MISMATCH")
        require((meta["status"] == "CHARGE_COMPONENTS_OBSERVED_NOT_PROVEN") == (len(rows) > 0),
                "SUCCESS_AVAILABILITY_MISMATCH")
    stats = {}
    with localcontext() as context:
        context.prec = 50
        for symbol in sorted({symbol for symbol, values in usable}):
            values = [item for row_symbol, item in usable if row_symbol == symbol]
            components = {}
            for key in ("commission", "swap", "fee"):
                native = [item[key] for item in values]
                normalized = [item[key]/item["volume"] for item in values]
                components[key] = {
                    "observed_native_sum": str(sum(native, Decimal(0))),
                    "native_min": str(min(native)), "native_max": str(max(native)),
                    "per_volume_min": str(min(normalized)), "per_volume_max": str(max(normalized)),
                    "zero_count": sum(value == 0 for value in native),
                    "nonzero_count": sum(value != 0 for value in native),
                }
            stats[symbol] = {"observed_rows": len(values), "observed_volume_sum": str(sum(
                (item["volume"] for item in values), Decimal(0))), "components": components}
    require(sha(read_regular(receipt_path, 100_000)) == sha(receipt_bytes), "RECEIPT_CHANGED")
    if raw_path is not None:
        require(sha(read_regular(raw_path, 32_000_000)) == sha(raw_bytes), "RAW_CHANGED")
    require(sha(read_regular(source_path, 100_000)) == SOURCE_SHA256, "SOURCE_CHANGED")
    require(sha(read_regular(Path(__file__), 100_000)) == sha(verifier_source), "VERIFIER_CHANGED")
    status = ("COMPONENTS_VALIDATED_NOT_COST_PROOF" if rows else "EMPTY_MATCHING_SELECTION_NOT_GLOBAL_ABSENCE") if clean else "DIAGNOSTIC_VALIDATED_NOT_USABLE"
    return {
        "schema_version": 1, "status": status, "scope": "PRIVATE_OBSERVED_COMPONENT_ARITHMETIC_ONLY",
        "observed_matching_components_available": bool(rows) if clean else None,
        "counts": {**{key: n[key] for key in ("processed_records", "matched_records", "raw_rows_written",
                                            "unrelated_trade_records", "nontrade_records", "unclassified_records")},
                   "selected_deals": total, "usable_retained_rows": len(usable)},
        "observed_components_by_symbol": stats, "current_currency": meta["account_currency_current"],
        "currency_scope": "CURRENT_NOT_HISTORICAL", "receipt_status": meta["status"],
        "receipt_sha256": sha(receipt_bytes), "raw_sha256": sha(raw_bytes) if raw_bytes is not None else None,
        "expected_raw_sha256_checked": expected_raw_sha256 is not None, "source_sha256": SOURCE_SHA256,
        "verifier_sha256": sha(verifier_source),
        "requested_time_semantics": "server_time", "requested_start_server_epoch": start,
        "requested_end_server_epoch": end, "raw_order_preserved": True,
        "backward_matched_time_rows": n["backward_matched_time_rows"],
        "private_financial_records_exported": bool(rows), "direct_account_identity_fields_exported": False,
        "cost_verified": False, "historical_costs_verified": False, "execution_verified": False,
        "historical_contract_mapping_verified": False, "historical_currency_applicability_verified": False,
        "all_period_fees_established": False, "actual_slippage_evidence": False,
        "zero_components_prove_all_dates_or_brokers_free": False, "paired_round_trip_cost_verified": False,
        "receipt_producer_identity_verified": False, "history_completeness": "NOT_PROVEN",
        "limitations": ["Observed signed components only; excluded/unclassified/nontrade charges can exist.",
                        "Other dates, historical contract/currency applicability, pairing, fills and slippage remain unknown."],
    }


def private_path(path, *, output=False):
    path = no_symlinks(path)
    allowed = no_symlinks(PRIVATE_ROOT).resolve()
    resolved = path.resolve()
    require(resolved != allowed and resolved.is_relative_to(allowed), "PRIVATE_PATH_REQUIRED")
    require(not any(resolved.is_relative_to((ROOT/name).resolve()) for name in
                    ("docs", "data", "research", "research_campaign", "frontend")), "PROTECTED_PATH")
    if output:
        require(path.suffix == ".json" and not path.exists() and path.parent.is_dir(), "NEW_PRIVATE_JSON_REQUIRED")
    return path


def save_private(path, report):
    path = private_path(path, output=True)
    descriptor = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise VerificationError("CLI_ARGUMENTS")


def main(argv=None):
    try:
        parser = StatusParser(description=__doc__)
        parser.add_argument("--receipt", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--expected-raw-sha256")
        args = parser.parse_args(argv)
        output, receipt = private_path(args.output, output=True), private_path(args.receipt)
        try:
            report = verify_receipt(receipt, expected_raw_sha256=args.expected_raw_sha256)
        except VerificationError as error:
            report = {"schema_version": 1, "status": "VERIFY_FAILED", "error_code": str(error),
                      "cost_verified": False, "execution_verified": False}
        save_private(output, report)
        print(report["status"])
        return 0 if report["status"] in ("COMPONENTS_VALIDATED_NOT_COST_PROOF",
                                         "EMPTY_MATCHING_SELECTION_NOT_GLOBAL_ABSENCE") else 2
    except (Exception, KeyboardInterrupt):
        print("VERIFY_FAILED")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
