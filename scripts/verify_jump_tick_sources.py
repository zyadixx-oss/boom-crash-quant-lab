#!/usr/bin/env python3
"""Independent public-source integrity audit; never computes returns or features.

The complete mode reconstructs one declared UTC day at a time from preserved
wire messages. Snapshot mode reads metadata only and cannot report completeness
PASS. No production acquisition/normalization module is imported.
"""
from __future__ import annotations

import argparse
import builtins
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import socket
import ssl

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = "docs/jump_representation_20261007/acquisition_declaration.json"
DECLARATION_SHA256 = "05b9e682e112d187d61a6a9321cc106fcf78b55eb0fb6710eac6ccc3b25b5db9"
CONFIG_SHA256 = "112f03efd3e247bb280fa16d229ba64abb47d72ff6800476a98cf31c59f8a7e6"
COLLECTOR_SHA256 = "087b071e28b948cd45e8d58bf03b370c42ac70827fc6010ace544c60ec9204a7"
DATA = "data/jump_representation_ticks_20261007"
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOLS = ("BOOM600", "CRASH600")
DATES = tuple((date(2025, 10, 9) + timedelta(days=i)).isoformat() for i in range(361))
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SAFETY = dict.fromkeys(FLAGS, False)
RAW_KEYS = {"endpoint", "request", "request_key", "attempt", "received_at_utc", "response_wire", "wire_sha256"}
PAGE_KEYS = {"endpoint", "request", "request_key", "attempt", "accepted", "wire_sha256"}
ERROR_KEYS = {"endpoint", "request", "request_key", "attempt", "error_type", "error", "wire_sha256"}
TRANSPORT_ERRORS = {name for module in (builtins, socket, ssl) for name, value in vars(module).items()
                    if isinstance(value, type) and issubclass(value, OSError)} | {
    "ConnectionClosed", "ConnectionClosedError", "ConnectionClosedOK", "InvalidStatus"}


def offline():
    for name in FLAGS:
        if os.environ.get(name, "false").strip().lower() != "false":
            raise ValueError(name + " must remain false")
    return dict(SAFETY)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(text, object_pairs_hook=unique, parse_float=Decimal, parse_constant=invalid)


def json_file(path):
    return parse_json(Path(path).read_text())


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def utc_time(value):
    if not isinstance(value, str):
        raise ValueError("UTC timestamp must be text")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("UTC timestamp required")
    return parsed


def date_bounds(day):
    parsed = date.fromisoformat(day)
    if parsed.isoformat() != day:
        raise ValueError("Canonical UTC date required")
    start = int(datetime.combine(parsed, datetime.min.time(), timezone.utc).timestamp())
    return start, start + 86400


def source_names(symbol, day):
    prefix = DATA + "/" + symbol.lower() + "_" + day
    return {"manifest": DATA + "/" + symbol + "_" + day + "_manifest.json",
            "raw": prefix + "_raw_pages.jsonl", "audit": prefix + "_page_audit.jsonl",
            "clean": prefix + "_ticks_clean.csv"}


def request_key(request):
    content = json.dumps({"endpoint": ENDPOINT, "request": request}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def identical(actual, expected):
    """JSON metadata equality without bool-as-int or extra-key equivalence."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(identical(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(identical(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


class Audit:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.checks = 0
        self.inputs = {}
        self.false_values = 0

    def require(self, condition, label):
        self.checks += 1
        if condition is not True:
            raise ValueError(label)

    def path(self, label):
        if (not isinstance(label, str) or not label or Path(label).is_absolute()
                or str(Path(label)) != label or any(part in (".", "..") for part in Path(label).parts)):
            raise ValueError("Canonical repository-relative input required")
        path = self.root
        for part in Path(label).parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Input symlinks forbidden")
        if not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Regular repository input required: " + label)
        return path

    def pin(self, label, expected=None):
        actual = digest(self.path(label))
        if expected is not None:
            self.require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected) is not None,
                         "Canonical SHA256 required")
            self.require(actual == expected, "Changed input bytes: " + label)
        self.inputs[label] = actual
        return actual

    def safety(self, value):
        self.require(isinstance(value, dict) and set(value) == set(FLAGS), "All four safety gates required")
        for name in FLAGS:
            self.require(value[name] is False, name + " must remain false")
            self.false_values += 1


def frozen_contract(audit):
    """Metadata/hash checks only; no wire message or quote CSV is decoded."""
    offline()
    audit.pin(DECLARATION, DECLARATION_SHA256)
    audit.pin("scripts/collect_spike_ticks.py", COLLECTOR_SHA256)
    audit.pin(DATA + "/acquisition_config.json", CONFIG_SHA256)
    declaration = json_file(audit.path(DECLARATION))
    config = json_file(audit.path(DATA + "/acquisition_config.json"))
    audit.safety(declaration["safety"])
    audit.safety(config["safety"])
    audit.require(declaration["symbols"] == list(SYMBOLS) and declaration["dates"] == list(DATES), "Fixed acquisition calendar")
    audit.require(declaration["expected_days_per_symbol"] == 361 and declaration["expected_sources"] == 722
                  and declaration["expected_1s_grid_rows"] == 62380800, "Fixed expected coverage arithmetic")
    audit.require(declaration["source_file"] == "scripts/collect_spike_ticks.py"
                  and declaration["source_sha256"] == COLLECTOR_SHA256, "Fixed collector identity")
    audit.require(config == {"version": 1, "endpoint": ENDPOINT, "symbols": list(SYMBOLS), "dates": list(DATES),
                  "page_size": 1000, "requested_delay_seconds": Decimal("0.5"),
                  "effective_min_request_spacing_seconds": Decimal("0.5"), "cadence_seconds": 1,
                  "collector_sha256": COLLECTOR_SHA256, "safety": SAFETY}, "Exact frozen collector config")
    for name in ("features_labels_or_tick_outcomes_computed", "authentication_used", "account_or_order_requests"):
        audit.require(declaration[name] is False, "Acquisition scope: " + name)
    audit.require(declaration["full_continuous_tick_coverage"] == "NOT TESTED", "Declaration must not preclaim completeness")
    return declaration


def manifest_metadata(audit, symbol, day):
    names = source_names(symbol, day)
    fingerprint = audit.pin(names["manifest"])
    manifest = json_file(audit.path(names["manifest"]))
    start, end = date_bounds(day)
    audit.safety(manifest["safety"])
    expected = {"symbol": symbol, "date": day, "start_epoch": start, "end_exclusive_epoch": end,
                "start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                "end_exclusive_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                "cadence_seconds": 1, "endpoint": ENDPOINT, "config_sha256": CONFIG_SHA256,
                "collector_file": "scripts/collect_spike_ticks.py", "collector_sha256": COLLECTOR_SHA256,
                "clean_file": names["clean"], "raw_pages_file": names["raw"], "page_audit_file": names["audit"],
                "expected_grid_rows": 86400, "normalization_valid": True}
    for name, value in expected.items():
        audit.require(type(manifest[name]) is type(value) and manifest[name] == value, "Manifest identity: " + name)
    for name in ("fills_or_interpolations", "authentication_used", "features_labels_or_tick_outcomes_computed"):
        audit.require(manifest[name] is False, "False source claim: " + name)
    rows = integer(manifest["rows"], "manifest rows")
    missing = integer(manifest["missing_seconds"], "missing seconds")
    audit.require(rows + missing == 86400, "Observed plus missing must equal UTC-day grid")
    audit.require(manifest["gap_free"] is (missing == 0), "Manifest gap-free identity")
    utc_time(manifest["completed_at_utc"])
    return manifest, names, fingerprint


def wire_rows(wire, request, audit):
    """Validate positive prices; no differences, returns, signals or targets."""
    value = parse_json(wire)
    audit.require(isinstance(value, dict), "History response object required")
    audit.require(type(value.get("req_id")) is int and value["req_id"] == request["req_id"], "Response req_id")
    if "echo_req" in value:
        audit.require(value["echo_req"] == request, "Response echo request")
    audit.require(not value.get("error") and value.get("msg_type") == "history", "Accepted page must be history without API error")
    history = value.get("history")
    audit.require(isinstance(history, dict) and {"times", "prices"} <= set(history), "History arrays required")
    times, prices = history["times"], history["prices"]
    audit.require(isinstance(times, list) and isinstance(prices, list)
                  and len(times) == len(prices) and len(times) <= request["count"], "History array lengths")
    rows = []
    previous = None
    for epoch, price in zip(times, prices, strict=True):
        audit.require(type(epoch) is int and request["start"] <= epoch <= request["end"], "Exact requested integer epoch")
        audit.require(previous is None or epoch >= previous, "History epochs ordered")
        audit.require(not isinstance(price, bool) and isinstance(price, (str, int, Decimal)), "Numeric public quote")
        try:
            number = Decimal(str(price))
        except InvalidOperation as error:
            raise ValueError("Invalid decimal quote") from error
        audit.require(number.is_finite() and number > 0, "Finite positive quote")
        rows.append((epoch, format(number, "f")))
        previous = epoch
    return rows


def gap_ranges(epochs, start, end):
    result = []
    previous = start - 1
    for epoch in (*epochs, end):
        if epoch > previous + 1:
            result.append({"start_epoch": previous + 1, "end_exclusive_epoch": epoch,
                           "missing_seconds": epoch - previous - 1})
        previous = epoch
    return result


def verify_day(audit, symbol, day, declared_at=None):
    manifest, names, manifest_sha = manifest_metadata(audit, symbol, day)
    start, end = date_bounds(day)
    for name, field in (("raw", "raw_pages_sha256"), ("audit", "page_audit_sha256"), ("clean", "clean_sha256")):
        audit.pin(names[name], manifest[field])
    prices = {}
    cursor, successes, attempts, duplicates = end - 1, 0, 0, 0
    errors, recovered = [], []
    previous_attempt, previous_time = None, None
    pending_errors = []
    with audit.path(names["raw"]).open() as raw_stream, audit.path(names["audit"]).open() as audit_stream:
        for ordinal, pair in enumerate(zip_longest(raw_stream, audit_stream)):
            raw_line, page_line = pair
            audit.require(raw_line is not None and page_line is not None, "Raw/audit attempt counts differ")
            raw, page = parse_json(raw_line), parse_json(page_line)
            audit.require(isinstance(raw, dict) and isinstance(page, dict), "Attempt records must be objects")
            audit.require(cursor >= start, "No requests permitted after day cursor exhaustion")
            request = {"ticks_history": symbol, "count": 1000, "start": start,
                       "end": cursor, "style": "ticks", "req_id": 1000 + successes}
            key = request_key(request)
            for record in (raw, page):
                audit.require(record["endpoint"] == ENDPOINT and record["request"] == request
                              and record["request_key"] == key, "Exact public request/page chain")
                for name in ("count", "start", "end", "req_id"):
                    integer(record["request"][name], name)
            attempt = integer(raw["attempt"], "attempt", 1)
            audit.require(attempt <= 4 and type(page["attempt"]) is int and page["attempt"] == attempt, "Attempt ordinal1..4")
            # A verified resume can restart at1 after any interrupted attempt.
            audit.require(attempt == 1 or previous_attempt is not None and attempt == previous_attempt + 1,
                          "Attempt order (or explicitly permitted resume restart at1)")
            stamp = utc_time(raw["received_at_utc"])
            audit.require(previous_time is None or stamp >= previous_time, "Attempt timestamp order")
            audit.require(declared_at is None or stamp >= declared_at, "Declaration precedes source requests")
            previous_time, previous_attempt = stamp, attempt
            wire = raw["response_wire"]
            if wire is None:
                audit.require(raw["wire_sha256"] is None and page["wire_sha256"] is None, "No-wire hash must be null")
            else:
                audit.require(isinstance(wire, str), "Wire message must remain text")
                fingerprint = hashlib.sha256(wire.encode()).hexdigest()
                audit.require(raw["wire_sha256"] == page["wire_sha256"] == fingerprint, "Preserved wire SHA256")
            audit.require(type(page["accepted"]) is bool, "Accepted must be boolean")
            if page["accepted"]:
                audit.require(set(raw) == RAW_KEYS and set(page) == PAGE_KEYS | {"rows", "equal_duplicates_removed", "oldest_epoch", "newest_epoch"}, "Successful attempt schema")
                audit.require(wire is not None, "Accepted history requires preserved wire")
                rows = wire_rows(wire, request, audit)
                removed = 0
                for epoch, quote in rows:
                    if epoch in prices:
                        audit.require(Decimal(prices[epoch]) == Decimal(quote), "Unequal duplicate quote")
                        removed += 1
                    else:
                        prices[epoch] = quote
                audit.require(type(page["rows"]) is int and page["rows"] == len(rows), "Accepted page row count")
                audit.require(type(page["equal_duplicates_removed"]) is int and page["equal_duplicates_removed"] == removed, "Equal duplicate count")
                audit.require(page["oldest_epoch"] == (rows[0][0] if rows else None)
                              and page["newest_epoch"] == (rows[-1][0] if rows else None), "Accepted page bounds")
                for error_index in pending_errors:
                    recovered.append({"error_index": error_index, "successful_page_audit_index": ordinal,
                        "request_key": key, "successful_wire_sha256": fingerprint, "endpoint": ENDPOINT,
                        "start_epoch": start, "end_epoch": cursor})
                pending_errors.clear()
                cursor = rows[0][0] - 1 if rows else start - 1
                duplicates += removed
                successes += 1
                previous_attempt = None
            else:
                audit.require(set(raw) == RAW_KEYS | {"error"}
                              and set(page) == PAGE_KEYS | {"recoverable", "error"}, "Failed attempt schema")
                audit.require(page["recoverable"] is True, "Semantic/unrecoverable source failure cannot pass")
                error = raw["error"]
                audit.require(isinstance(error, dict) and set(error) == ERROR_KEYS and error == page["error"], "Exact preserved error record")
                audit.require(error["endpoint"] == ENDPOINT and error["request"] == request
                              and error["request_key"] == key and error["attempt"] == attempt
                              and error["wire_sha256"] == raw["wire_sha256"], "Error request identity")
                integer(error["attempt"], "error attempt", 1)
                audit.require(isinstance(error["error_type"], str) and bool(error["error_type"])
                              and isinstance(error["error"], str), "Preserved error description")
                if wire is not None:
                    value = parse_json(wire)
                    audit.require(isinstance(value, dict) and value.get("req_id") == request["req_id"]
                                  and isinstance(value.get("error"), dict) and value["error"].get("code") == "RateLimit"
                                  and error["error_type"] == "RateLimitError", "Only rate-limit wire failures are recoverable")
                else:
                    audit.require(error["error_type"] in TRANSPORT_ERRORS, "No-wire failure must be a recognized transport exception")
                pending_errors.append(len(errors)); errors.append(error)
            attempts += 1
    audit.require(attempts > 0 and successes > 0 and cursor < start, "Source must finish cursor traversal")
    audit.require(not pending_errors, "Unresolved retries cannot pass")
    audit.require(identical(manifest["request_errors"], errors), "Manifest retained error sequence")
    audit.require(identical(manifest["recovered_retry_lineage"], recovered), "Exact successful same-request retry lineage")
    audit.require(manifest["successful_page_count"] == successes and type(manifest["successful_page_count"]) is int, "Successful page count")
    audit.require(manifest["equal_duplicates_removed"] == duplicates and type(manifest["equal_duplicates_removed"]) is int, "Manifest duplicate total")
    epochs = sorted(prices)
    gaps = gap_ranges(epochs, start, end)
    audit.require(identical(manifest["gaps"], gaps), "Exact unknown-second ranges")
    audit.require(manifest["missing_seconds"] == sum(x["missing_seconds"] for x in gaps)
                  and manifest["rows"] == len(epochs), "Reconstructed coverage counts")
    audit.require(manifest["first_epoch"] == (epochs[0] if epochs else None)
                  and manifest["last_epoch"] == (epochs[-1] if epochs else None), "Reconstructed first/last quote")
    audit.require(utc_time(manifest["completed_at_utc"]) >= previous_time, "Manifest completion follows attempts")
    with audit.path(names["clean"]).open(newline="") as stream:
        reader = csv.reader(stream)
        audit.require(next(reader, None) == ["epoch", "quote"], "Exact clean CSV schema")
        for epoch, row in zip_longest(epochs, reader):
            audit.require(epoch is not None and row is not None, "Clean CSV row count")
            audit.require(row == [str(epoch), prices[epoch]], "Clean CSV canonical epoch/decimal quote equals original wire")
    for label in names.values():
        audit.require(digest(audit.path(label)) == audit.inputs[label], "Source changed during independent reconstruction")
    return {"symbol": symbol, "date": day, "rows": len(epochs), "missing_seconds": manifest["missing_seconds"],
            "gap_free": not gaps, "attempts": attempts, "successful_pages": successes,
            "recovered_errors": len(errors), "equal_duplicates_removed": duplicates,
            "manifest_sha256": manifest_sha}


def inspect(audit, mode, sources_ready=False):
    offline()
    if mode not in ("snapshot", "complete"):
        raise ValueError("Unknown audit mode")
    if mode == "complete" and sources_ready is not True:
        raise ValueError("Explicit --sources-ready required before decoding source quotes")
    declaration = frozen_contract(audit)
    available, missing = [], []
    for symbol in SYMBOLS:
        for day in DATES:
            label = source_names(symbol, day)["manifest"]
            (available if (audit.root / label).exists() else missing).append((symbol, day))
    declared_names = {source_names(symbol, day)["manifest"] for symbol in SYMBOLS for day in DATES}
    discovered_names = {str(path.relative_to(audit.root)) for path in (audit.root / DATA).glob("*_manifest.json")}
    unexpected = sorted(discovered_names - declared_names)
    if mode == "complete":
        audit.require(not missing and len(available) == 722, "All722 declared sources are required before a complete audit")
        audit.require(not unexpected, "Undeclared source manifests cannot enter a complete acquisition audit")
    summaries = []
    declared_at = utc_time(declaration["declared_at_utc"])
    for symbol, day in available:
        if mode == "complete":
            summaries.append(verify_day(audit, symbol, day, declared_at))
        else:
            manifest, _, fingerprint = manifest_metadata(audit, symbol, day)
            summaries.append({"symbol": symbol, "date": day, "manifest_sha256": fingerprint,
                              "reported_rows_unverified": manifest["rows"],
                              "reported_missing_seconds_unverified": manifest["missing_seconds"]})
    if mode == "complete":
        for label, fingerprint in audit.inputs.items():
            audit.require(digest(audit.path(label)) == fingerprint, "Pinned input changed before audit completion")
    return {"stage": "independent_complete_jump_tick_source_audit" if mode == "complete" else "jump_tick_source_metadata_snapshot",
            "run_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "passed": True if mode == "complete" else None, "completeness_passed": mode == "complete",
            "integrity_of_available_sources": "PASS" if mode == "complete" else "NOT TESTED",
            "gap_free_full_calendar": all(row["gap_free"] for row in summaries) if mode == "complete" else None,
            "expected_sources": 722, "available_manifests": len(available), "missing_manifests": len(missing),
            "unexpected_manifest_files": unexpected,
            "expected_grid_seconds": 62380800, "sources": summaries,
            "missing_source_identities": [{"symbol": symbol, "date": day} for symbol, day in missing],
            "checks": audit.checks, "errors": [], "input_sha256": audit.inputs,
            "saved_false_values_checked": audit.false_values, "safety": offline(),
            "returns_features_labels_or_payoffs_computed": False,
            "strategy_payoff": "NOT TESTED", "cash_profit": "NOT TESTED",
            "scope": "Source integrity and preserved missing seconds only. Every completed source may contain explicitly audited gaps. No returns, targets, physical-spike census, model, economic or execution inference. Attempt ordinals may restart at1 on verified resume; total retry attempts across sessions are not bounded at4."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("snapshot", "complete"), required=True)
    parser.add_argument("--sources-ready", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); offline()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT) or destination.exists() or destination.with_suffix(".sha256").exists():
        raise ValueError("New repository-local output required; refusing overwrite")
    audit = Audit()
    try:
        report = inspect(audit, args.mode, args.sources_ready)
    except Exception as error:
        report = {"stage": "independent_jump_tick_source_audit_failed", "run_utc": datetime.now(timezone.utc).isoformat(),
                  "mode": args.mode, "passed": False, "completeness_passed": False,
                  "checks": audit.checks, "errors": [{"type": type(error).__name__, "message": str(error)}],
                  "input_sha256": audit.inputs, "safety": offline(),
                  "returns_features_labels_or_payoffs_computed": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        report["verifier_file"] = "scripts/verify_jump_tick_sources.py"
        report["verifier_sha256"] = digest(Path(__file__))
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    fingerprint = digest(destination)
    with destination.with_suffix(".sha256").open("x") as stream:
        stream.write(fingerprint + "\n")
    print(json.dumps({"mode": args.mode, "passed": report["passed"], "checks": report["checks"],
                      "output_sha256": fingerprint, "errors": report["errors"]}), flush=True)
    if report["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
