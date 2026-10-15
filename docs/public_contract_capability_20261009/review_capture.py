#!/usr/bin/env python3
"""Independent, offline stdlib review of the frozen public capability capture.

No probe imports, sockets, account calls, price modelling or profitability tests.
An internally consistent INCOMPLETE capture never becomes a completed PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOLS = ("BOOM500", "CRASH500", "BOOM600", "CRASH600", "R_100")
TYPES = ("CALL", "PUT", "MULTUP", "MULTDOWN")
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
FROZEN_V1 = {
    "probe.py": "e4f23509b1cc55540a4d9bfd9e5031743f1c137f8dfcda625d2418056becf608",
    "test_probe.py": "64b28627066db6a17476ff89465ad67f36709215798466492a37a4bebbe169cc",
    "PROTOCOL.md": "8ef504678c16b673c4060d448d53b739bd7724e58606dd03f71e735ed2bb23cd",
}
FROZEN_V2 = {
    "probe_v2.py": "7bb49f36833822374675658de36c0e04e0988e4c2d76ccc35c5739f294abbe14",
    "test_probe_v2.py": "0e1ae028ba1779db87a019edaebe5e026cdea7dda3816641dc3ba4dbdd527bc0",
    "PROTOCOL_v2.md": "0a6d255c680fdeb2f51f76c2112f75c180caa998ab185e4ad68bedb29d90ae50",
}
DATA_FILES = ("declaration.json", "requests.jsonl", "received_text_messages.jsonl",
              "observations.jsonl", "result.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def decode(text):
    if not isinstance(text, str):
        raise ValueError("Expected text response")
    def bad(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    obj = json.loads(text, object_pairs_hook=unique, parse_constant=bad)
    if not isinstance(obj, dict):
        raise ValueError("Expected response object")
    return obj


def exact(a, b):
    """JSON value equality with booleans distinct from numeric one/zero."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def positive_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def catalogue_request(symbol):
    return {"contracts_for": symbol, "req_id": SYMBOLS.index(symbol) + 1}


def derive_quote(symbol, kind, available):
    """Independently reconstruct only the predeclared metadata rule."""
    records = [(i, row) for i, row in enumerate(available)
               if isinstance(row, dict) and row.get("underlying_symbol") == symbol
               and row.get("contract_type") == kind]
    reason = "not_advertised_or_no_matching_symbol"
    selected = None
    lists = []
    for _, row in records:
        if kind in ("CALL", "PUT"):
            def ticks(value):
                match = re.fullmatch(r"([0-9]+)t", value) if isinstance(value, str) else None
                return int(match[1]) if match else None
            low, high = ticks(row.get("min_contract_duration")), ticks(row.get("max_contract_duration"))
            if (type(row.get("barriers")) in (int, float) and row["barriers"] == 0
                    and row.get("expiry_type") == "tick" and low is not None
                    and high is not None and low <= 5 <= high):
                selected = {"duration": 5, "duration_unit": "t"}
                break
            reason = "no_explicit_zero_barrier_tick_interval_containing_5t"
        else:
            values = row.get("multiplier_range")
            if not isinstance(values, list) or not values or not all(positive_number(x) for x in values):
                return None, "no_valid_explicit_numeric_multiplier_list", [i for i, _ in records]
            lists.append(sorted(set(values)))
    if lists:
        if any(values != lists[0] for values in lists[1:]):
            return None, "conflicting_advertised_multiplier_lists", [i for i, _ in records]
        selected = {"multiplier": min(lists[0])}
    if selected is None:
        return None, reason, [i for i, _ in records]
    request = {"proposal": 1, "amount": 10, "basis": "stake", "currency": "USD",
               "underlying_symbol": symbol, "contract_type": kind,
               "req_id": 100 + 4 * SYMBOLS.index(symbol) + TYPES.index(kind), **selected}
    return request, "eligible_under_frozen_catalogue_rule", [i for i, _ in records]


def observation(request, response, version=1):
    operation = "contracts_for" if "contracts_for" in request else "proposal"
    if type(response.get("req_id")) is not int or response["req_id"] != request["req_id"]:
        raise ValueError("Response request-ID mismatch")
    allowed_echoes = [request]
    if version == 2 and operation == "proposal" and request.get("contract_type") in ("MULTUP", "MULTDOWN"):
        allowed_echoes.append({**request, "duration_unit": "s"})
    # V1's terminal failure stays fatal; V2 permits only its separately declared addition.
    if not any(exact(response.get("echo_req"), expected) for expected in allowed_echoes):
        raise ValueError("Response echoed request mismatch")
    if response.get("msg_type") != operation:
        raise ValueError("Response message-type mismatch")
    row = {"req_id": request["req_id"], "operation": operation,
           "symbol": request.get("contracts_for", request.get("underlying_symbol")),
           "top_level_keys": sorted(response)}
    if "error" in response:
        return {**row, "status": "API_ERROR", "error": response["error"]}
    obj = response.get(operation)
    if not isinstance(obj, dict):
        raise ValueError("Missing response object")
    row.update(status="SUCCESS", object_keys=sorted(obj))
    if operation == "contracts_for":
        available = obj.get("available")
        if not isinstance(available, list) or not all(isinstance(x, dict) for x in available):
            raise ValueError("Invalid catalogue records")
        row.update(available_records=len(available),
                   advertised_types=sorted({str(x.get("contract_type")) for x in available}),
                   contract_record_keys=sorted({key for x in available for key in x}))
    else:
        row.update(contract_type=request["contract_type"],
                   field_types={key: type(value).__name__ for key, value in obj.items()})
        if not isinstance(obj.get("id"), str) or not obj["id"]:
            row["status"] = "UNUSABLE_PROPOSAL"
    return row


class Audit:
    def __init__(self):
        self.checks = 0
        self.errors = []
    def check(self, condition, pointer, detail):
        self.checks += 1
        if not condition:
            self.errors.append({"pointer": pointer, "detail": detail})
        return bool(condition)
    def equal(self, actual, expected, pointer):
        return self.check(exact(actual, expected), pointer,
                          {"expected": expected, "actual": actual})
    def time(self, value, pointer):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                raise ValueError("timestamp must explicitly be UTC")
            self.check(True, pointer, "UTC timestamp")
            return parsed
        except (AttributeError, TypeError, ValueError) as exc:
            self.check(False, pointer, str(exc))
            return None


def field_inventory(value, pointer=""):
    """Preserve values, types and exact pointers; field names imply no hidden state."""
    rows = [{"pointer": pointer or "/", "type": type(value).__name__}]
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            rows.extend(field_inventory(child, pointer + "/" + escaped))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            rows.extend(field_inventory(child, pointer + "/" + str(i)))
    else:
        rows[0]["value"] = value
    return rows


def review_capture(capture, frozen_directory=None):
    capture = Path(capture)
    frozen_directory = Path(frozen_directory) if frozen_directory else HERE
    audit = Audit()
    inputs = {}
    documents = {}
    try:
        predeclaration = decode((capture / "declaration.json").read_text())
        version = predeclaration.get("audit_version", 1)
    except (OSError, UnicodeError, ValueError):
        version = 1
    audit.check(type(version) is int and version in (1, 2), "declaration.json/audit_version", "Only frozen versions 1/2")
    frozen_sources = FROZEN_V2 if version == 2 else FROZEN_V1
    capture_files = (*frozen_sources, *DATA_FILES)
    for name in capture_files:
        path = capture / name
        try:
            data = path.read_bytes()
            inputs[name] = sha(data)
            if name.endswith(".json"):
                documents[name] = decode(data.decode("utf-8"))
            elif name.endswith(".jsonl"):
                documents[name] = [decode(line) for line in data.decode("utf-8").splitlines()]
        except (OSError, UnicodeError, ValueError) as exc:
            audit.check(False, name, f"Cannot read exact artifact: {type(exc).__name__}: {exc}")
    report = {"reviewed_utc": datetime.now(timezone.utc).isoformat(),
              "status": "FAIL", "pass": False, "evidence_consistent": False,
              "capture_complete": False, "input_sha256": inputs,
              "reviewer_sha256": sha(Path(__file__).read_bytes()),
              "audit_version": version, "frozen_source_sha256": frozen_sources,
              "scope": "offline byte, request-matrix, timestamp and saved-summary consistency only",
              "limitations": [
                  "No independent TLS/network packet, actual endpoint identity or connection-count audit.",
                  "Wall-clock ordering is checked; monotonic timeout enforcement is not independently measured.",
                  "Saved declaration timestamps are not external attestations of premeasurement chronology.",
                  "Frozen test bytes are verified; this reviewer does not establish their execution result.",
                  "Response fields are descriptive; no hidden generator state or predictive meaning is inferred.",
                  "No CFD mapping, account fills, broker costs, probability, strategy or profit is verified."],
              "prediction_tested": False, "profit_tested": False, "QUALIFIED": False}
    required = ("declaration.json", "result.json", "requests.jsonl",
                "received_text_messages.jsonl", "observations.jsonl")
    if not all(name in documents for name in required):
        return {**report, "checks": audit.checks, "errors": audit.errors}
    declaration, result = documents["declaration.json"], documents["result.json"]
    requests, packets, saved_obs = (documents[name] for name in required[2:])
    for filename, expected in frozen_sources.items():
        audit.equal(inputs.get(filename), expected, filename + "/sha256")
        audit.equal(declaration.get("source_sha256", {}).get(filename), expected,
                    "declaration.json/source_sha256/" + filename)
        try:
            audit.equal(sha((frozen_directory / filename).read_bytes()), expected,
                        "frozen_directory/" + filename)
        except OSError as exc:
            audit.check(False, "frozen_directory/" + filename, str(exc))
    audit.equal(declaration.get("source_sha256"), frozen_sources,
                "declaration.json/source_sha256")
    expected_pins = {name: inputs.get(name) for name in capture_files if name != "result.json"}
    audit.equal(result.get("artifact_sha256"), expected_pins, "result.json/artifact_sha256")
    constants = {"endpoint": ENDPOINT, "symbols": list(SYMBOLS), "types": list(TYPES),
                 "max_requests": 25, "max_request_work_seconds": 300,
                 "receive_send_timeout": 15, "connection_timeout": 20,
                 "close_timeout": 5, "http_redirects_allowed": False}
    for key, expected in constants.items():
        audit.equal(declaration.get(key), expected, "declaration.json/" + key)
    for name, doc in (("declaration.json", declaration), ("result.json", result)):
        audit.equal(doc.get("safety"), dict.fromkeys(FLAGS, False), name + "/safety")
        for key in ("authentication_used", "profit_tested", "prediction_tested", "cfd_mapping_verified",
                    "actual_fills_verified", "measured_broker_costs_verified"):
            audit.equal(doc.get(key), False, name + "/" + key)
        audit.equal(doc.get("orders_sent"), 0, name + "/orders_sent")
    audit.equal(result.get("scope"), "public capability only", "result.json/scope")
    declared = audit.time(declaration.get("declared_utc"), "declaration.json/declared_utc")
    finished = audit.time(result.get("finished_utc"), "result.json/finished_utc")
    error, close_error = result.get("error"), result.get("close_error")
    for name, value in (("error", error), ("close_error", close_error)):
        audit.check(value is None or (isinstance(value, dict) and set(value) == {"type", "message"}
                    and isinstance(value["type"], str) and isinstance(value["message"], str)),
                    "result.json/" + name, "Expected null or preserved type/message failure context")
    claimed_complete = error is None and close_error is None
    audit.equal(result.get("transport_status"), "COMPLETED" if claimed_complete else "INCOMPLETE",
                "result.json/transport_status")
    audit.check(type(result.get("connection_closed")) is bool, "result.json/connection_closed", "Expected bool")
    if claimed_complete:
        audit.equal(result.get("connection_closed"), True, "result.json/connection_closed")
    if declared and finished:
        audit.check(declared <= finished, "timestamps", "Declaration must precede finish")
        audit.check((finished - declared).total_seconds() <= 305, "timestamps/recorded_total_cap",
                    "Recorded declaration-to-finish span fits 300s work plus 5s cleanup")

    attempts, completed, request_times = [], {}, []
    awaiting = None
    previous = declared
    for i, event in enumerate(requests):
        pointer = f"requests.jsonl/{i}"
        timestamp = audit.time(event.get("utc"), pointer + "/utc")
        if timestamp and previous:
            audit.check(timestamp >= previous, pointer + "/utc", "Request log is chronological")
        if timestamp and finished:
            audit.check(timestamp <= finished, pointer + "/utc", "No request after finish")
        previous = timestamp or previous
        if event.get("event") == "send_attempt":
            audit.check(awaiting is None, pointer, "Previous send must complete before another attempt")
            awaiting = len(attempts)
            attempts.append(event)
            request_times.append(timestamp)
            text = event.get("text")
            try:
                parsed = decode(text)
                audit.equal(parsed, event.get("request"), pointer + "/text_decoded")
                audit.equal(event.get("text_sha256"), sha(text.encode()), pointer + "/text_sha256")
                audit.equal(text, json.dumps(event.get("request"), separators=(",", ":"), allow_nan=False),
                            pointer + "/text_exact_encoding")
            except (ValueError, TypeError) as exc:
                audit.check(False, pointer + "/text", str(exc))
        elif event.get("event") == "send_completed":
            if audit.check(awaiting is not None, pointer, "Completion needs preceding send attempt"):
                audit.equal(event.get("req_id"), attempts[awaiting].get("request", {}).get("req_id"), pointer + "/req_id")
                completed[awaiting] = timestamp
                if timestamp and request_times[awaiting]:
                    audit.check((timestamp - request_times[awaiting]).total_seconds() <= 15,
                                pointer + "/utc", "Recorded send duration must fit declared 15s cap")
                awaiting = None
        else:
            audit.check(False, pointer + "/event", "Unrecognized request event")
    audit.check(len(attempts) <= 25 and len(packets) <= 25, "request_response_caps", "At most 25 attempts/responses")
    audit.equal(result.get("request_attempts"), len(attempts), "result.json/request_attempts")
    audit.equal(result.get("response_observations"), len(saved_obs), "result.json/response_observations")
    audit.equal(result.get("observations"), saved_obs, "result.json/observations")
    audit.check(len(packets) <= len(completed) <= len(attempts), "counts", "Responses need completed sends")
    audit.check(awaiting is None or (not claimed_complete and error is not None and awaiting == len(attempts) - 1),
                "requests.jsonl/terminal_send", "Only a failed terminal send may lack completion")

    rebuilt_obs, responses, failures, inventories = [], [], [], []
    previous_receive = declared
    for i, packet in enumerate(packets):
        pointer = f"received_text_messages.jsonl/{i}"
        audit.equal(packet.get("frame_number"), i + 1, pointer + "/frame_number")
        received = audit.time(packet.get("received_utc"), pointer + "/received_utc")
        if received and previous_receive:
            audit.check(received >= previous_receive, pointer, "Received messages are chronological")
        if received and i in completed and completed[i]:
            audit.check(received >= completed[i], pointer, "Receive follows completed send")
            audit.check((received - completed[i]).total_seconds() <= 15, pointer, "Recorded receive fits 15s cap")
        if received and i + 1 < len(request_times) and request_times[i + 1]:
            audit.check(received <= request_times[i + 1], pointer, "Prior response precedes next attempt")
        if received and finished:
            audit.check(received <= finished, pointer, "Receive precedes finish")
        if received and request_times and request_times[0]:
            audit.check((received - request_times[0]).total_seconds() <= 300,
                        pointer, "Recorded request-work span fits 300s cap")
        previous_receive = received or previous_receive
        response = None
        try:
            if packet.get("payload_type") == "text":
                raw = packet.get("text")
                if not isinstance(raw, str):
                    raise ValueError("Expected text response")
                audit.equal(packet.get("binary_hex"), None, pointer + "/binary_hex")
                audit.equal(packet.get("payload_sha256"), sha(raw.encode()), pointer + "/payload_sha256")
                response = decode(raw)
            elif packet.get("payload_type") == "binary":
                audit.equal(packet.get("text"), None, pointer + "/text")
                audit.equal(packet.get("payload_sha256"), sha(bytes.fromhex(packet["binary_hex"])), pointer + "/payload_sha256")
                raise ValueError("Expected text response")
            else:
                raise ValueError("Unrecognized payload type")
            inventories.append({"frame_number": i + 1, "raw_text_pointer": pointer + "/text",
                                "fields": field_inventory(response)})
            if i >= len(attempts):
                raise ValueError("Response has no attempted request")
            rebuilt_obs.append(observation(attempts[i].get("request", {}), response, version))
        except (ValueError, TypeError, KeyError) as exc:
            failure = {"frame_number": i + 1, "raw_text_pointer": pointer + "/text",
                       "type": type(exc).__name__, "message": str(exc)}
            if response is not None:
                request = attempts[i].get("request", {}) if i < len(attempts) else {}
                echo = response.get("echo_req")
                if isinstance(echo, dict):
                    failure["echo_difference"] = {
                        "added": {k: v for k, v in echo.items() if k not in request},
                        "missing": {k: v for k, v in request.items() if k not in echo},
                        "changed": {k: {"sent": request[k], "echo": echo[k]}
                                    for k in request.keys() & echo.keys() if not exact(request[k], echo[k])}}
            failures.append(failure)
        responses.append(response)
    audit.equal(rebuilt_obs, saved_obs, "observations.jsonl/reconstructed")
    if failures:
        audit.check(len(failures) == 1 and failures[0]["frame_number"] == len(packets) == len(attempts),
                    "terminal_failure", "Only final response may stop sequential matrix")
        audit.equal(error, {k: failures[0][k] for k in ("type", "message")}, "result.json/error/reconstructed")
        audit.equal(result.get("transport_status"), "INCOMPLETE", "terminal_failure/transport_status")
    elif claimed_complete:
        audit.equal(len(packets), len(attempts), "complete_response_count")
    elif error is not None:
        audit.check(len(packets) in (len(attempts), len(attempts) - 1) or not attempts,
                    "incomplete_prefix", "Failure must preserve a sequential observed prefix")

    # The first five calls must be the catalogues, even if the last fails.
    for i, attempt in enumerate(attempts[:5]):
        audit.equal(attempt.get("request"), catalogue_request(SYMBOLS[i]), f"requests/catalogue/{i}")
    accepted_catalogues = {}
    for i, row in enumerate(rebuilt_obs[:5]):
        if row.get("operation") == "contracts_for" and i < len(responses):
            accepted_catalogues[SYMBOLS[i]] = responses[i]
    selections, matrix, quotes = [], [], []
    for symbol in SYMBOLS:
        response = accepted_catalogues.get(symbol)
        catalogue_index = SYMBOLS.index(symbol)
        matrix.append({"symbol": symbol, "operation": "contracts_for",
                       "request": catalogue_request(symbol), "observed": response is not None,
                       "status": ("API_ERROR" if response is not None and "error" in response else
                                  "SUCCESS" if response is not None else "NOT_ACCEPTED_OR_NOT_REACHED"),
                       "observation_pointer": f"observations.jsonl/{catalogue_index}" if response is not None else None})
        if response is not None and "error" in response:
            selections.append({"symbol": symbol, "status": "catalogue_error_no_proposals"})
        for kind in TYPES:
            row = {"symbol": symbol, "operation": "proposal", "contract_type": kind,
                   "catalogue_raw_text_pointer": f"received_text_messages.jsonl/{catalogue_index}/text"}
            if response is None or "error" in response:
                row.update(request=None, reason="catalogue_error" if response is not None else "catalogue_not_accepted",
                           matching_record_pointers=[])
            else:
                request, reason, record_indices = derive_quote(symbol, kind, response["contracts_for"]["available"])
                selection = {"symbol": symbol, "contract_type": kind, "request": request, "reason": reason}
                selections.append(selection)
                row.update(selection, matching_record_pointers=[f"/contracts_for/available/{j}" for j in record_indices])
                if request is not None:
                    quotes.append(request)
            matrix.append(row)
    if len(attempts) > 5:
        audit.equal(len(accepted_catalogues), 5, "proposal_prerequisite",)
    expected_requests = [catalogue_request(symbol) for symbol in SYMBOLS] + quotes
    audit.check(len(attempts) <= len(expected_requests), "requests/matrix", "No request outside reconstructed matrix")
    for i, attempt in enumerate(attempts):
        if i < len(expected_requests):
            audit.equal(attempt.get("request"), expected_requests[i], f"requests/matrix/{i}")
    saved_selections = result.get("selections")
    audit.check(isinstance(saved_selections, list), "result.json/selections", "Expected list")
    saved_selections = saved_selections if isinstance(saved_selections, list) else []
    audit.equal(saved_selections, selections[:len(saved_selections)], "result.json/selections/reconstructed_prefix")
    attempted_quote_ids = [x.get("request", {}).get("req_id") for x in attempts[5:]]
    selected_quote_ids = [x["request"]["req_id"] for x in saved_selections if x.get("request") is not None]
    audit.check(selected_quote_ids == attempted_quote_ids or
                (error is not None and selected_quote_ids[:-1] == attempted_quote_ids),
                "selections/attempted_quotes", "Selections include attempted quotes and at most terminal pre-send failure")
    if failures and attempted_quote_ids:
        audit.check(bool(saved_selections) and saved_selections[-1].get("request", {}).get("req_id") == attempted_quote_ids[-1],
                    "selections/terminal_failure", "Matrix stops at the failed quote")
    if claimed_complete or (error is None and close_error is not None):
        audit.equal(saved_selections, selections, "result.json/selections/complete")
        audit.equal(len(attempts), len(expected_requests), "requests/complete_matrix")
        audit.equal(len(accepted_catalogues), 5, "catalogues/complete")
    selected_keys = {(x.get("symbol"), x.get("contract_type")) for x in saved_selections}
    catalogue_error_symbols = {x["symbol"] for x in saved_selections if x.get("status") == "catalogue_error_no_proposals"}
    observation_by_id = {row["req_id"]: row for row in rebuilt_obs}
    failure_ids = {attempts[x["frame_number"] - 1]["request"]["req_id"] for x in failures
                   if x["frame_number"] <= len(attempts)}
    for row in matrix:
        if row["operation"] != "proposal":
            continue
        selected = (row["symbol"], row["contract_type"]) in selected_keys or row["symbol"] in catalogue_error_symbols
        row["selection_evaluated_before_stop"] = selected
        request = row.get("request")
        if request is not None and request["req_id"] in failure_ids:
            row["status"] = "RECEIVED_NOT_ACCEPTED_FATAL_VALIDATION"
        elif request is not None and request["req_id"] in observation_by_id:
            row["status"] = observation_by_id[request["req_id"]]["status"]
        elif request is not None and request["req_id"] in attempted_quote_ids:
            row["status"] = "ATTEMPTED_NO_ACCEPTED_RESPONSE"
        elif selected and request is None:
            row["status"] = "SKIPPED_UNDER_FROZEN_RULE"
        else:
            row["status"] = "NOT_REACHED"
    proposal_rows = [row for row in matrix if row["operation"] == "proposal"]
    elapsed = (finished - declared).total_seconds() if declared and finished else None
    report["pass"] = claimed_complete and not audit.errors
    report.update(checks=audit.checks, errors=audit.errors,
                  evidence_consistent=not audit.errors, capture_complete=claimed_complete and not audit.errors,
                  status="FAIL" if audit.errors else "PASS" if claimed_complete else "INCOMPLETE",
                  asserted_transport_status=result.get("transport_status"),
                  asserted_error=error, asserted_close_error=close_error,
                  terminal_response_failures=failures, matrix=matrix,
                  full_metadata_derived_selections=selections,
                  observed_field_inventory=inventories,
                  summary={"catalogue_requests_planned": 5, "quote_type_slots": 20,
                           "metadata_eligible_quotes": len(quotes),
                           "metadata_ineligible_quotes": sum(x.get("request") is None and
                               x.get("reason") not in ("catalogue_error", "catalogue_not_accepted") for x in proposal_rows),
                           "metadata_unresolved_quote_slots": sum(x.get("reason") in
                               ("catalogue_error", "catalogue_not_accepted") for x in proposal_rows),
                           "requests_attempted": len(attempts), "sends_completed": len(completed),
                           "responses_preserved": len(packets), "accepted_observations": len(rebuilt_obs),
                           "accepted_catalogues": len(accepted_catalogues),
                           "accepted_proposal_observations": sum(x["operation"] == "proposal" for x in rebuilt_obs),
                           "selections_evaluated": len(saved_selections),
                           "quote_slot_status_counts": dict(Counter(x["status"] for x in proposal_rows)),
                           "recorded_declaration_to_finish_seconds": elapsed,
                           "all_four_saved_flags_false": all(doc.get("safety") == dict.fromkeys(FLAGS, False)
                                                              for doc in (declaration, result)),
                           "orders_in_expected_request_matrix": 0,
                           "observed_order_request_count": sum(any(key in x.get("request", {}) for key in
                               ("buy", "sell", "cancel", "sell_expired")) for x in attempts),
                           "asserted_orders_sent": result.get("orders_sent")})
    return report


def write_exclusive(path, report):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def self_test():
    def save(path, value):
        path.write_text(json.dumps(value, allow_nan=False) + "\n")
    def lines(path, values):
        path.write_text("".join(json.dumps(x, allow_nan=False) + "\n" for x in values))
    def read_lines(path):
        return [decode(x) for x in path.read_text().splitlines()]
    def repin(root):
        result = decode((root / "result.json").read_text())
        result["artifact_sha256"] = {p.name: sha(p.read_bytes()) for p in root.iterdir()
                                     if p.is_file() and p.name != "result.json"}
        save(root / "result.json", result)
    def fixture(root, version=1, bad_echo=False, catalogue_errors=False, unusable=False):
        frozen = FROZEN_V2 if version == 2 else FROZEN_V1
        root.mkdir()
        for name in frozen:
            (root / name).write_bytes((HERE / name).read_bytes())
        declaration = {"declared_utc": "2026-01-01T00:00:00+00:00", "endpoint": ENDPOINT,
                       "symbols": list(SYMBOLS), "types": list(TYPES), "safety": dict.fromkeys(FLAGS, False),
                       "authentication_used": False, "max_requests": 25, "max_request_work_seconds": 300,
                       "receive_send_timeout": 15, "connection_timeout": 20, "close_timeout": 5,
                       "source_sha256": frozen, "http_redirects_allowed": False, "orders_sent": 0,
                       **dict.fromkeys(("profit_tested", "prediction_tested", "cfd_mapping_verified",
                                        "actual_fills_verified", "measured_broker_costs_verified"), False)}
        if version == 2:
            declaration["audit_version"] = 2
        save(root / "declaration.json", declaration)
        request_log, packet_log, obs, selections, calls = [], [], [], [], []
        catalogues = {}
        for symbol in SYMBOLS:
            available = []
            for kind in TYPES:
                row = {"underlying_symbol": symbol, "contract_type": kind}
                if kind in ("CALL", "PUT"):
                    row.update(barriers=0, expiry_type="tick", min_contract_duration="1t", max_contract_duration="10t")
                else:
                    row["multiplier_range"] = [50, 10, 20]
                available.append(row)
            request = catalogue_request(symbol)
            response = {"req_id": request["req_id"], "echo_req": request, "msg_type": "contracts_for"}
            response.update({"error": {"code": "Fixture", "message": "catalogue unavailable"}} if catalogue_errors
                            else {"contracts_for": {"available": available}})
            catalogues[symbol] = response
            calls.append((request, response))
        for symbol in SYMBOLS:
            if catalogue_errors:
                selections.append({"symbol": symbol, "status": "catalogue_error_no_proposals"})
                continue
            for kind in TYPES:
                request, reason, _ = derive_quote(symbol, kind, catalogues[symbol]["contracts_for"]["available"])
                selections.append({"symbol": symbol, "contract_type": kind, "request": request, "reason": reason})
                response = {"req_id": request["req_id"], "echo_req": request.copy(), "msg_type": "proposal",
                            "proposal": {"id": None if unusable else "fixture-id", "ask_price": "10", "payout": None}}
                if version == 2 and kind in ("MULTUP", "MULTDOWN"):
                    response["echo_req"]["duration_unit"] = "s"
                if bad_echo:
                    response["echo_req"]["added_fixture_field"] = True
                calls.append((request, response))
                if bad_echo:
                    break
            if bad_echo:
                break
        for i, (request, response) in enumerate(calls, 1):
            text = json.dumps(request, separators=(",", ":"))
            request_log.extend([
                {"utc": f"2026-01-01T00:00:{i:02d}.000000+00:00", "event": "send_attempt",
                 "request": request, "text": text, "text_sha256": sha(text.encode())},
                {"utc": f"2026-01-01T00:00:{i:02d}.010000+00:00", "event": "send_completed", "req_id": request["req_id"]}])
            raw = json.dumps(response)
            packet_log.append({"received_utc": f"2026-01-01T00:00:{i:02d}.020000+00:00", "frame_number": i,
                               "payload_type": "text", "text": raw, "binary_hex": None, "payload_sha256": sha(raw.encode())})
            if not (bad_echo and i == len(calls)):
                obs.append(observation(request, response, version))
        lines(root / "requests.jsonl", request_log)
        lines(root / "received_text_messages.jsonl", packet_log)
        lines(root / "observations.jsonl", obs)
        result = {"finished_utc": "2026-01-01T00:00:30+00:00", "scope": "public capability only",
                  "transport_status": "INCOMPLETE" if bad_echo else "COMPLETED", "request_attempts": len(calls),
                  "response_observations": len(obs), "observations": obs, "selections": selections,
                  "error": {"type": "ValueError", "message": "Response echoed request mismatch"} if bad_echo else None,
                  "close_error": None, "connection_closed": True, "safety": dict.fromkeys(FLAGS, False),
                  "authentication_used": False, "orders_sent": 0,
                  **dict.fromkeys(("profit_tested", "prediction_tested", "cfd_mapping_verified",
                                   "actual_fills_verified", "measured_broker_costs_verified"), False)}
        save(root / "result.json", result)
        repin(root)
    class Boundaries(unittest.TestCase):
        def test_tick_units_and_barriers(self):
            row = {"underlying_symbol": "BOOM500", "contract_type": "CALL", "barriers": 0,
                   "expiry_type": "tick", "min_contract_duration": "1t", "max_contract_duration": "10t"}
            self.assertEqual(derive_quote("BOOM500", "CALL", [row])[0]["duration"], 5)
            for update in ({"barriers": False}, {"min_contract_duration": "1"},
                           {"max_contract_duration": "4t"}, {"expiry_type": "intraday"}):
                self.assertIsNone(derive_quote("BOOM500", "CALL", [{**row, **update}])[0])
        def test_multiplier_all_records(self):
            def row(values):
                return {"underlying_symbol": "BOOM500", "contract_type": "MULTUP", "multiplier_range": values}
            self.assertEqual(derive_quote("BOOM500", "MULTUP", [row([50, 10, 10]), row([10, 50])])[0]["multiplier"], 10)
            for values in ([20, 50], [], [True], ["10"], [float("inf")], [10 ** 10000]):
                self.assertIsNone(derive_quote("BOOM500", "MULTUP", [row([10, 50]), row(values)])[0])
        def test_echo_addition_is_fatal(self):
            request, _, _ = derive_quote("BOOM500", "MULTUP", [{"underlying_symbol": "BOOM500",
                                         "contract_type": "MULTUP", "multiplier_range": [100]}])
            response = {"req_id": request["req_id"], "msg_type": "proposal", "proposal": {"id": "x"},
                        "echo_req": {**request, "duration_unit": "s"}}
            with self.assertRaisesRegex(ValueError, "echoed request mismatch"):
                observation(request, response)
            self.assertEqual(observation(request, response, version=2)["status"], "SUCCESS")
            for update in ({"duration_unit": "t"}, {"buy": 1}, {"amount": True}):
                with self.assertRaisesRegex(ValueError, "echoed request mismatch"):
                    observation(request, {**response, "echo_req": {**request, **update}}, version=2)
        def test_minimal_and_unusable_proposals(self):
            request = {"req_id": 100, "underlying_symbol": "BOOM500", "contract_type": "CALL"}
            for obj in ({}, {"id": None}, {"id": ""}, {"id": 1}, {"id": False}):
                response = {"req_id": 100, "msg_type": "proposal", "echo_req": request, "proposal": obj}
                self.assertEqual(observation(request, response)["status"], "UNUSABLE_PROPOSAL")
            obj = {"id": "x", "ask_price": "10", "payout": None}
            response = {"req_id": 100, "msg_type": "proposal", "echo_req": request, "proposal": obj}
            self.assertEqual(observation(request, response)["status"], "SUCCESS")
        def test_strict_json_and_boolean_id(self):
            for raw in ('{"a":1,"a":2}', '{"a":NaN}', '[]', b'{}'):
                with self.assertRaises(ValueError):
                    decode(raw)
            request = catalogue_request("BOOM500")
            with self.assertRaisesRegex(ValueError, "request-ID"):
                observation(request, {"req_id": True, "echo_req": request, "msg_type": "contracts_for"})
            self.assertFalse(exact({"n": True}, {"n": 1}))
        def test_exclusive_output(self):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "review.json"
                write_exclusive(path, {"pass": False})
                with self.assertRaises(FileExistsError):
                    write_exclusive(path, {})
        def test_missing_capture_never_passes(self):
            with tempfile.TemporaryDirectory() as temp:
                report = review_capture(Path(temp) / "missing")
                self.assertEqual(report["status"], "FAIL")
                self.assertFalse(report["pass"])
                self.assertFalse(report["capture_complete"])
        def test_complete_full_matrix_both_versions(self):
            for version in (1, 2):
                with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp) / "capture"
                    fixture(root, version=version)
                    report = review_capture(root)
                    self.assertEqual(report["errors"], [])
                    self.assertEqual(report["status"], "PASS")
                    self.assertEqual(report["summary"]["requests_attempted"], 25)
                    self.assertEqual(report["summary"]["accepted_proposal_observations"], 20)
        def test_consistent_incomplete_not_pass(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root, bad_echo=True)
                report = review_capture(root)
                self.assertEqual(report["errors"], [])
                self.assertTrue(report["evidence_consistent"])
                self.assertEqual(report["status"], "INCOMPLETE")
                self.assertFalse(report["pass"])
                result = decode((root / "result.json").read_text())
                result.update(transport_status="COMPLETED", error=None)
                save(root / "result.json", result)
                self.assertFalse(review_capture(root)["evidence_consistent"])
        def test_catalogue_errors_complete_without_proposals(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root, catalogue_errors=True)
                report = review_capture(root)
                self.assertEqual(report["errors"], [])
                self.assertTrue(report["pass"])
                self.assertEqual(report["summary"]["requests_attempted"], 5)
                self.assertEqual(report["summary"]["accepted_proposal_observations"], 0)
                self.assertEqual(report["summary"]["metadata_ineligible_quotes"], 0)
                self.assertEqual(report["summary"]["metadata_unresolved_quote_slots"], 20)
        def test_unusable_proposals_not_erased(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root, unusable=True)
                report = review_capture(root)
                self.assertEqual(report["errors"], [])
                self.assertTrue(report["pass"])
                self.assertEqual(report["summary"]["quote_slot_status_counts"], {"UNUSABLE_PROPOSAL": 20})
        def test_rehashed_order_request_not_accepted(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root)
                events = read_lines(root / "requests.jsonl")
                request = {**events[10]["request"], "buy": "fixture-order"}
                text = json.dumps(request, separators=(",", ":"))
                events[10].update(request=request, text=text, text_sha256=sha(text.encode()))
                lines(root / "requests.jsonl", events)
                packets = read_lines(root / "received_text_messages.jsonl")
                response = decode(packets[5]["text"])
                response["echo_req"] = request
                raw = json.dumps(response)
                packets[5].update(text=raw, payload_sha256=sha(raw.encode()))
                lines(root / "received_text_messages.jsonl", packets)
                repin(root)
                report = review_capture(root)
                self.assertFalse(report["evidence_consistent"])
                self.assertEqual(report["summary"]["observed_order_request_count"], 1)
                self.assertTrue(any(x["pointer"] == "requests/matrix/5" for x in report["errors"]))
        def test_rehashed_chronology_violation(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root)
                packets = read_lines(root / "received_text_messages.jsonl")
                packets[0]["received_utc"] = "2025-12-31T23:59:59+00:00"
                lines(root / "received_text_messages.jsonl", packets)
                repin(root)
                report = review_capture(root)
                self.assertFalse(report["evidence_consistent"])
                self.assertTrue(any(x["pointer"] == "received_text_messages.jsonl/0" for x in report["errors"]))
        def test_rehashed_source_and_safety_violation(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root)
                (root / "PROTOCOL.md").write_text("mutated after declaration\n")
                result = decode((root / "result.json").read_text())
                result["safety"]["LIVE_ALLOWED"] = True
                result["orders_sent"] = 1
                save(root / "result.json", result)
                repin(root)
                report = review_capture(root)
                self.assertFalse(report["evidence_consistent"])
                pointers = {x["pointer"] for x in report["errors"]}
                self.assertIn("PROTOCOL.md/sha256", pointers)
                self.assertIn("result.json/safety", pointers)
                self.assertIn("result.json/orders_sent", pointers)
        def test_preserved_binary_terminal_failure(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root, bad_echo=True)
                packets = read_lines(root / "received_text_messages.jsonl")
                payload = b"binary-fixture"
                packets[-1].update(payload_type="binary", text=None, binary_hex=payload.hex(), payload_sha256=sha(payload))
                lines(root / "received_text_messages.jsonl", packets)
                result = decode((root / "result.json").read_text())
                result["error"] = {"type": "ValueError", "message": "Expected text response"}
                save(root / "result.json", result)
                repin(root)
                report = review_capture(root)
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["status"], "INCOMPLETE")
                self.assertFalse(report["pass"])
        def test_missing_terminal_response_preserves_unknown(self):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "capture"
                fixture(root)
                packets = read_lines(root / "received_text_messages.jsonl")[:-1]
                obs = read_lines(root / "observations.jsonl")[:-1]
                lines(root / "received_text_messages.jsonl", packets)
                lines(root / "observations.jsonl", obs)
                result = decode((root / "result.json").read_text())
                result.update(transport_status="INCOMPLETE", observations=obs, response_observations=len(obs),
                              error={"type": "TimeoutError", "message": "fixture timeout"})
                save(root / "result.json", result)
                repin(root)
                report = review_capture(root)
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["status"], "INCOMPLETE")
                self.assertEqual(report["summary"]["quote_slot_status_counts"]["ATTEMPTED_NO_ACCEPTED_RESPONSE"], 1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=HERE / "capture01")
    parser.add_argument("--frozen-directory", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-status", choices=("PASS", "INCOMPLETE"),
                        help="Allow CI to verify expected incomplete evidence without claiming completion")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    report = review_capture(args.capture, args.frozen_directory)
    if args.output:
        write_exclusive(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    if args.expect_status:
        return 0 if report["evidence_consistent"] and report["status"] == args.expect_status else 1
    # 2 explicitly distinguishes consistent incomplete evidence from completed PASS.
    return 0 if report["pass"] else 2 if report["evidence_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
