"""Verify private count-only MQL5 receipts offline; CLI prints a status token only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "work/private_history_availability_20261013"
SOURCE = ROOT / "scripts/mt5/InspectResearchHistoryAvailability.mq5"
SOURCE_SHA256 = "9e641987455c5349a03d2675d526bfc4f26993749b4c0797ad22bb16115c7e82"
FIELDS = frozenset("""
schema_version scope status failure_reason server company account_mode capture_time_gmt
requested_time_semantics requested_bounds_are_utc server_timezone_offset_verified
input_start_server_epoch input_end_server_epoch requested_start_server_epoch
requested_end_server_epoch server_clock_at_query_epoch server_clock_error
server_clock_semantics end_resolved_from_timecurrent max_span_days selection_bounds_semantics
history_select_called history_select_call_count history_select_returned history_select_error
history_deals_total_called history_deals_total_returned history_deals_total_error
history_orders_total_called history_orders_total_returned history_orders_total_error
counts_accepted context_stable_before_query context_stable_after_query query_elapsed_ms
connection_wait_limit_seconds connection_wait_elapsed_ms terminal_trade_allowed_observed
mql_trade_allowed_observed guard_error selection_scope selection_completeness
whole_history_completeness same_server_same_mode_account_switch_detection
context_comparison_scope boom_crash_relation_verified deal_count_is_trade_fill_count
record_type_classification_performed live_trading ready_for_live live_allowed opened_trades
safety_upper_live_trading safety_upper_ready_for_live safety_upper_live_allowed
safety_upper_opened_trades costs_verified historical_costs_verified execution_verified
cfd_fills_verified prices_exported pnl_exported tickets_exported individual_records_exported
personal_data_exported account_login_name_balance_recorded credentials_recorded
login_or_account_creation_attempted orders_sent output_file limitation
""".split())
FALSE_FIELDS = frozenset("""
requested_bounds_are_utc server_timezone_offset_verified
same_server_same_mode_account_switch_detection boom_crash_relation_verified
deal_count_is_trade_fill_count record_type_classification_performed
live_trading ready_for_live live_allowed opened_trades safety_upper_live_trading
safety_upper_ready_for_live safety_upper_live_allowed safety_upper_opened_trades
costs_verified historical_costs_verified execution_verified cfd_fills_verified
prices_exported pnl_exported tickets_exported individual_records_exported
personal_data_exported account_login_name_balance_recorded credentials_recorded
login_or_account_creation_attempted
""".split())
FIXED = {
    "schema_version": "1",
    "scope": "ACCOUNT_HISTORY_AVAILABILITY_COUNTS_ONLY",
    "requested_time_semantics": "server_time",
    "server_clock_semantics": "TimeCurrent_last_known_existing_server_time",
    "max_span_days": "400",
    "selection_bounds_semantics": "HistorySelect_server_time_bounds_no_UTC_conversion",
    "selection_scope": "ALL_ACCOUNT_HISTORY_IN_REQUESTED_RANGE_NOT_SYMBOL_FILTERED",
    "selection_completeness": "NOT_PROVEN",
    "whole_history_completeness": "NOT_PROVEN",
    "context_comparison_scope": "server_company_account_mode_only",
    "orders_sent": "0",
    "limitation": (
        "Counts describe selected account-history lists only; no symbol relation, "
        "trade-fill classification, costs, prices, profit, or history completeness is established."
    ),
}
REASONS = {
    "COUNTS_OBSERVED_NOT_PROVEN": {""},
    "FAILED_INPUT": {"invalid_start_end_or_connection_wait",
                     "invalid_or_future_server_time_window_or_span_over_400_days"},
    "FAILED_UNSAFE": {"terminal_or_script_trading_not_disabled_or_guard_unavailable"},
    "FAILED_DISCONNECTED": {"existing_server_connection_unavailable_after_bounded_wait",
                            "terminal_disconnected"},
    "FAILED_ACCOUNT_CONTEXT": {"unavailable_or_non_deriv_server_context",
                               "unsupported_account_mode",
                               "server_company_or_account_mode_changed_during_query"},
    "FAILED_STOPPED": {"script_stopped_before_history_query"},
    "FAILED_SELECTION_OR_COUNTS": {"selection_return_error_or_nonnegative_counts_check_failed"},
}


class VerificationError(ValueError):
    """Fixed error codes only; never incorporate metadata, counts or paths."""


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
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "NOT_REGULAR_FILE")
            data = stream.read(limit + 1)
        require(len(data) <= limit, "INPUT_TOO_LARGE")
        return data
    except VerificationError:
        raise
    except OSError:
        raise VerificationError("INPUT_UNAVAILABLE") from None


def number(value, code, maximum=2**63 - 1):
    require(isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]{0,18}", value) is not None, code)
    result = int(value)
    require(result <= maximum, code)
    return result


def boolean(value):
    require(value in ("true", "false"), "INVALID_BOOLEAN")
    return value == "true"


def parse_receipt(data):
    try:
        text = data.decode("utf-8-sig")
        require(text.endswith("\n") and "\x00" not in text, "TRUNCATED_OR_BINARY_TSV")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
        require(next(reader, None) == ["field", "value"], "INVALID_HEADER")
        rows = list(reader)
        require(len(rows) == len(FIELDS), "FIELD_COUNT")
        require(all(len(row) == 2 for row in rows), "MALFORMED_ROW")
        require(all("\n" not in v and "\r" not in v and "\t" not in v and len(v) <= 2048
                    for row in rows for v in row), "MULTILINE_OR_OVERSIZED_VALUE")
        metadata = dict(rows)
        require(len(metadata) == len(rows) and set(metadata) == FIELDS, "DUPLICATE_OR_UNKNOWN_FIELD")
        return metadata
    except VerificationError:
        raise
    except (UnicodeError, csv.Error):
        raise VerificationError("INVALID_TSV") from None


def verify_receipt(receipt_path, *, source_path=None):
    receipt_path = Path(receipt_path)
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,64}_availability\.tsv", receipt_path.name) is not None,
            "INVALID_RECEIPT_NAME")
    source_path = SOURCE if source_path is None else Path(source_path)
    source = read_regular(source_path, 100_000)
    require(sha(source) == SOURCE_SHA256, "SOURCE_PIN_MISMATCH")
    raw = read_regular(receipt_path, 65_536)
    meta = parse_receipt(raw)
    require(meta["output_file"] == receipt_path.name, "OUTPUT_BASENAME_MISMATCH")
    require(all(meta[key] == value for key, value in FIXED.items()), "FIXED_PROTOCOL_MISMATCH")
    require(all(meta[key] == "false" for key in FALSE_FIELDS), "UNSAFE_OR_UNSUPPORTED_CLAIM")
    status = meta["status"]
    require(status in REASONS and meta["failure_reason"] in REASONS[status], "STATUS_REASON_MISMATCH")
    try:
        datetime.strptime(meta["capture_time_gmt"], "%Y.%m.%d %H:%M:%S GMT")
    except ValueError:
        raise VerificationError("INVALID_CAPTURE_TIME") from None
    selected = boolean(meta["history_select_called"])
    returned = boolean(meta["history_select_returned"])
    before = boolean(meta["context_stable_before_query"])
    after = boolean(meta["context_stable_after_query"])
    accepted = boolean(meta["counts_accepted"])
    resolved = boolean(meta["end_resolved_from_timecurrent"])
    terminal_allowed = boolean(meta["terminal_trade_allowed_observed"])
    program_allowed = boolean(meta["mql_trade_allowed_observed"])
    require(number(meta["history_select_call_count"], "SELECT_CALL_COUNT", 1) == int(selected),
            "SELECT_CALL_COUNT")
    require(before == selected and (selected or not after), "CONTEXT_CALL_SEQUENCE")
    select_error = number(meta["history_select_error"], "INVALID_SELECT_ERROR", 2**31 - 1)
    guard_error = number(meta["guard_error"], "INVALID_GUARD_ERROR", 2**31 - 1)
    clock_error = number(meta["server_clock_error"], "INVALID_CLOCK_ERROR", 2**31 - 1)
    require(selected or (not returned and select_error == 0), "UNREQUESTED_SELECTION_RETURN")
    wait_limit = number(meta["connection_wait_limit_seconds"], "INVALID_WAIT_LIMIT", 30)
    waited = number(meta["connection_wait_elapsed_ms"], "INVALID_WAIT_DURATION")
    elapsed = number(meta["query_elapsed_ms"], "INVALID_QUERY_DURATION")
    require(selected or elapsed == 0, "UNREQUESTED_QUERY_DURATION")

    counts, count_errors, called = {}, {}, {}
    for label in ("deals", "orders"):
        stem = "history_" + label + "_total"
        called[label] = boolean(meta[stem + "_called"])
        count_errors[label] = number(meta[stem + "_error"], "INVALID_COUNT_ERROR", 2**31 - 1)
        if called[label]:
            counts[label] = number(meta[stem + "_returned"], "INVALID_NONNEGATIVE_COUNT", 2**31 - 1)
        else:
            require(meta[stem + "_returned"] == "NOT_QUERIED" and count_errors[label] == 0,
                    "UNQUERIED_COUNT_VALUE")
            counts[label] = None
    should_count = selected and returned and select_error == 0
    require(called["deals"] == called["orders"] == should_count, "COUNT_CALL_SEQUENCE")

    start = number(meta["input_start_server_epoch"], "INVALID_START")
    input_end = number(meta["input_end_server_epoch"], "INVALID_END_INPUT")
    require(number(meta["requested_start_server_epoch"], "INVALID_REQUEST_START") == start,
            "REQUEST_START_MISMATCH")
    clock = (None if meta["server_clock_at_query_epoch"] == "NOT_READ"
             else number(meta["server_clock_at_query_epoch"], "INVALID_SERVER_CLOCK"))
    end = (None if meta["requested_end_server_epoch"] == "NOT_RESOLVED"
           else number(meta["requested_end_server_epoch"], "INVALID_REQUEST_END"))
    require(resolved == (input_end == 0 and clock is not None and clock > 0), "END_RESOLUTION_FLAG")
    if clock is not None:
        require(end == (clock if input_end == 0 else input_end), "RESOLVED_END_MISMATCH")
    else:
        require(end is None and not selected, "UNREAD_CLOCK_WITH_SELECTION")
    if selected:
        require(clock_error == 0 and clock is not None and clock > 0 and end is not None
                and 0 < start < end <= clock and end - start <= 400 * 86400,
                "INVALID_SERVER_TIME_WINDOW")
        require(re.fullmatch(r"Deriv(?:SVG|BVI|FX|V|Mauritius|Cayman)?(?:[- ._][A-Za-z0-9 ._()-]+)?",
                             meta["server"], re.IGNORECASE) is not None, "UNIDENTIFIED_SERVER")
        require(re.fullmatch(r"Deriv(?:[ .(][A-Za-z0-9 .(),-]*)?", meta["company"], re.IGNORECASE)
                is not None, "UNIDENTIFIED_COMPANY")
        require(meta["account_mode"] in ("DEMO", "LIVE"), "INVALID_ACCOUNT_MODE")
    else:
        require(meta["account_mode"] in ("DEMO", "LIVE", "UNKNOWN"), "INVALID_ACCOUNT_MODE")
        # No arbitrary text is accepted as nonpersonal broker provenance.
        require(not meta["server"] or re.fullmatch(r"[A-Za-z0-9 ._()-]{1,128}", meta["server"]),
                "INVALID_SERVER_TEXT")
        require(not meta["company"] or re.fullmatch(r"[A-Za-z0-9 .(),-]{1,128}", meta["company"]),
                "INVALID_COMPANY_TEXT")

    expected_acceptance = (should_count and after and count_errors["deals"] == 0
                           and count_errors["orders"] == 0)
    require(accepted == expected_acceptance, "COUNT_ACCEPTANCE_MISMATCH")
    require((status == "COUNTS_OBSERVED_NOT_PROVEN") == accepted, "SUCCESS_STATUS_MISMATCH")
    if accepted:
        require(not terminal_allowed and not program_allowed and guard_error == 0
                and before and after, "POST_QUERY_PROTECTION_FAILED")
    require(sha(read_regular(receipt_path, 65_536)) == sha(raw), "RECEIPT_CHANGED")
    require(sha(read_regular(source_path, 100_000)) == SOURCE_SHA256, "SOURCE_CHANGED")
    return {
        "schema_version": 1,
        "status": "COUNTS_VALIDATED" if accepted else "NO_USABLE_COUNTS",
        "scope": "PRIVATE_SELECTED_ACCOUNT_HISTORY_AVAILABILITY_ONLY",
        "availability": bool(counts["deals"] or counts["orders"]) if accepted else None,
        "availability_determined": accepted,
        "availability_scope": "SELECTED_ACCOUNT_LISTS_IN_REQUESTED_SERVER_TIME_RANGE_ONLY",
        "counts": counts,
        "receipt_status": status,
        "receipt_sha256": sha(raw),
        "source_sha256": SOURCE_SHA256,
        "verifier_sha256": sha(read_regular(Path(__file__), 100_000)),
        "requested_time_semantics": "server_time",
        "requested_start_server_epoch": start,
        "requested_end_server_epoch": end,
        "source_return_errors": {"selection": select_error, **count_errors},
        "context_stable_after_query": after,
        "connection_wait_limit_seconds": wait_limit,
        "connection_wait_elapsed_ms": waited,
        "whole_history_completeness": "NOT_PROVEN",
        "zero_count_proves_all_brokers_absence": False,
        "trade_fill_count_verified": False,
        "boom_crash_relation_verified": False,
        "historical_costs_verified": False,
        "execution_verified": False,
        "personal_data_exported": False,
        "receipt_producer_identity_verified": False,
        "compiled_binary_identity_verified": False,
    }


def private_path(path, *, output=False):
    path = no_symlinks(path)
    allowed = no_symlinks(PRIVATE_ROOT).resolve()
    resolved = path.resolve()
    require(resolved.is_relative_to(allowed) and resolved != allowed, "PRIVATE_PATH_REQUIRED")
    for name in ("docs", "data", "research", "research_campaign", "frontend"):
        require(not resolved.is_relative_to((ROOT / name).resolve()), "PROTECTED_OUTPUT_PATH")
    if output:
        require(not path.exists() and path.parent.is_dir(), "NEW_OUTPUT_REQUIRED")
        require(path.suffix == ".json", "JSON_OUTPUT_REQUIRED")
    return path


def save_private(path, report):
    path = private_path(path, output=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
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
        args = parser.parse_args(argv)
        output = private_path(args.output, output=True)
        receipt = private_path(args.receipt)
        try:
            report = verify_receipt(receipt)
        except VerificationError as error:
            report = {"schema_version": 1, "status": "VERIFY_FAILED", "error_code": str(error),
                      "availability": None, "availability_determined": False}
        save_private(output, report)
        print(report["status"])
        return 0 if report["status"] == "COUNTS_VALIDATED" else 2
    except (VerificationError, OSError, ValueError, KeyboardInterrupt):
        print("VERIFY_FAILED")
        return 2
    except Exception:
        print("VERIFY_FAILED")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
