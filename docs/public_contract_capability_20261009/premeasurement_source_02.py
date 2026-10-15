#!/usr/bin/env python3
"""Frozen, bounded, unauthenticated public contract metadata/quote observation."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import ssl
import sys
import time

import certifi
import websockets

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOLS = ("BOOM500", "CRASH500", "BOOM600", "CRASH600", "R_100")
TYPES = ("CALL", "PUT", "MULTUP", "MULTDOWN")
SAFETY = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)
HERE = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def safety_guard():
    for name in SAFETY:
        if os.environ.get(name, "false").strip().lower() != "false":
            raise ValueError(f"Public audit requires {name}=false")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(raw):
    if not isinstance(raw, str):
        raise ValueError("Expected text response")

    def bad_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=bad_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected response object")
    return value


def number(value):
    try:
        return (type(value) in (int, float) and math.isfinite(value) and value > 0)
    except (OverflowError, TypeError):
        return False


def tick_duration(value):
    match = re.fullmatch(r"([0-9]+)t", value) if isinstance(value, str) else None
    return int(match[1]) if match else None


def catalog_request(symbol):
    return {"contracts_for": symbol, "req_id": SYMBOLS.index(symbol) + 1}


def eligible_quote(symbol, contract_type, available):
    records = [r for r in available if isinstance(r, dict)
               and r.get("underlying_symbol") == symbol
               and r.get("contract_type") == contract_type]
    selected = None
    reason = "not_advertised_or_no_matching_symbol"
    multiplier_lists = []
    for record in records:
        if contract_type in ("CALL", "PUT"):
            minimum = tick_duration(record.get("min_contract_duration"))
            maximum = tick_duration(record.get("max_contract_duration"))
            if (type(record.get("barriers")) in (int, float) and record["barriers"] == 0
                    and record.get("expiry_type") == "tick" and minimum is not None
                    and maximum is not None and minimum <= 5 <= maximum):
                selected = {"duration": 5, "duration_unit": "t"}
                break
            reason = "no_explicit_zero_barrier_tick_interval_containing_5t"
        else:
            choices = record.get("multiplier_range")
            if isinstance(choices, list) and choices and all(number(v) for v in choices):
                multiplier_lists.append(sorted(set(choices)))
                candidate = min(choices)
                if selected is None or candidate < selected["multiplier"]:
                    selected = {"multiplier": candidate}
            else:
                return None, "no_valid_explicit_numeric_multiplier_list"
    if multiplier_lists and any(choices != multiplier_lists[0] for choices in multiplier_lists):
        return None, "conflicting_advertised_multiplier_lists"
    if selected is None:
        return None, reason
    request = {"proposal": 1, "amount": 10, "basis": "stake", "currency": "USD",
               "underlying_symbol": symbol, "contract_type": contract_type,
               "req_id": 100 + 4 * SYMBOLS.index(symbol) + TYPES.index(contract_type)}
    request.update(selected)
    return request, "eligible_under_frozen_catalogue_rule"


def validate_request(request):
    if "contracts_for" in request:
        symbol = request["contracts_for"]
        if (symbol not in SYMBOLS or type(request.get("req_id")) is not int
                or request != catalog_request(symbol)):
            raise ValueError("Request outside frozen catalogue matrix")
        return
    symbol = request.get("underlying_symbol")
    kind = request.get("contract_type")
    if symbol not in SYMBOLS or kind not in TYPES:
        raise ValueError("Request outside frozen quote matrix")
    fixed = {"proposal": 1, "amount": 10, "basis": "stake", "currency": "USD",
             "underlying_symbol": symbol, "contract_type": kind,
             "req_id": 100 + 4 * SYMBOLS.index(symbol) + TYPES.index(kind)}
    if kind in ("CALL", "PUT"):
        fixed.update(duration=5, duration_unit="t")
    else:
        if not number(request.get("multiplier")):
            raise ValueError("Invalid advertised multiplier")
        fixed["multiplier"] = request["multiplier"]
    if request != fixed or any(type(request[k]) is bool for k in ("proposal", "amount", "req_id")):
        raise ValueError("Request differs from frozen informational quote")


def summarize(request, response):
    expected_type = "contracts_for" if "contracts_for" in request else "proposal"
    if type(response.get("req_id")) is not int or response["req_id"] != request["req_id"]:
        raise ValueError("Response request-ID mismatch")
    if response.get("echo_req") != request:
        raise ValueError("Response echoed request mismatch")
    if response.get("msg_type") != expected_type:
        raise ValueError("Response message-type mismatch")
    row = {"req_id": request["req_id"], "operation": expected_type,
           "symbol": request.get("contracts_for", request.get("underlying_symbol")),
           "top_level_keys": sorted(response)}
    if "error" in response:
        row.update(status="API_ERROR", error=response["error"])
        return row
    obj = response.get(expected_type)
    if not isinstance(obj, dict):
        raise ValueError("Missing response object")
    row.update(status="SUCCESS", object_keys=sorted(obj))
    if expected_type == "contracts_for":
        available = obj.get("available")
        if not isinstance(available, list) or not all(isinstance(r, dict) for r in available):
            raise ValueError("Invalid catalogue records")
        row.update(available_records=len(available),
                   advertised_types=sorted({str(r.get("contract_type")) for r in available}),
                   contract_record_keys=sorted({k for r in available for k in r}))
    else:
        row.update(contract_type=request["contract_type"],
                   field_types={k: type(v).__name__ for k, v in obj.items()})
        if not isinstance(obj.get("id"), str) or not obj["id"]:
            row["status"] = "UNUSABLE_PROPOSAL"
    return row


async def probe(output, connector=None):
    safety_guard()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for filename in ("probe.py", "test_probe.py", "PROTOCOL.md"):
        data = (HERE / filename).read_bytes()
        (output / filename).write_bytes(data)
        sources[filename] = sha(data)
    declaration = {"declared_utc": utc(), "endpoint": ENDPOINT, "symbols": SYMBOLS,
                   "types": TYPES, "safety": SAFETY, "authentication_used": False,
                   "max_requests": 25, "max_request_work_seconds": 300, "receive_send_timeout": 15,
                   "connection_timeout": 20, "close_timeout": 5, "source_sha256": sources,
                   "python": sys.version, "platform": platform.platform(),
                   "websockets_version": importlib.metadata.version("websockets"),
                   "certifi_version": importlib.metadata.version("certifi"),
                   "profit_tested": False, "prediction_tested": False,
                   "cfd_mapping_verified": False, "actual_fills_verified": False,
                   "measured_broker_costs_verified": False, "orders_sent": 0}
    save(output / "declaration.json", declaration)
    for filename in ("requests.jsonl", "received_text_messages.jsonl", "observations.jsonl"):
        (output / filename).touch(exist_ok=False)
    started = time.monotonic()
    ws = None
    count = 0
    observations = []
    selections = []
    error = None
    close_error = None
    connection_closed = False

    async def call(request):
        nonlocal count
        safety_guard()
        validate_request(request)
        if count >= 25:
            raise ValueError("Request cap reached")
        remaining = 300 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Overall audit deadline")
        text = json.dumps(request, separators=(",", ":"), allow_nan=False)
        append(output / "requests.jsonl", {"utc": utc(), "event": "send_attempt",
               "request": request, "text": text, "text_sha256": sha(text.encode())})
        count += 1
        await asyncio.wait_for(ws.send(text), timeout=min(15, remaining))
        append(output / "requests.jsonl", {"utc": utc(), "event": "send_completed",
                                           "req_id": request["req_id"]})
        remaining = 300 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Overall audit deadline before response")
        raw = await asyncio.wait_for(ws.recv(), timeout=min(15, remaining))
        payload = raw.encode() if isinstance(raw, str) else raw
        append(output / "received_text_messages.jsonl", {"received_utc": utc(),
               "frame_number": count, "payload_type": "text" if isinstance(raw, str) else "binary",
               "text": raw if isinstance(raw, str) else None,
               "binary_hex": raw.hex() if isinstance(raw, bytes) else None,
               "payload_sha256": sha(payload)})
        response = decode(raw)
        observation = summarize(request, response)
        observations.append(observation)
        append(output / "observations.jsonl", observation)
        return response

    try:
        connect = connector or websockets.connect
        ws = await asyncio.wait_for(connect(ENDPOINT,
                ssl=ssl.create_default_context(cafile=certifi.where()), open_timeout=20,
                close_timeout=5, ping_interval=20, ping_timeout=20, max_size=1048576,
                max_queue=16), timeout=20)
        catalogues = {}
        for symbol in SYMBOLS:
            catalogues[symbol] = await call(catalog_request(symbol))
        for symbol in SYMBOLS:
            response = catalogues[symbol]
            if "error" in response:
                selections.append({"symbol": symbol, "status": "catalogue_error_no_proposals"})
                continue
            available = response["contracts_for"]["available"]
            for kind in TYPES:
                request, reason = eligible_quote(symbol, kind, available)
                selections.append({"symbol": symbol, "contract_type": kind,
                                   "request": request, "reason": reason})
                if request is not None:
                    await call(request)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=5)
                connection_closed = True
            except Exception as exc:
                close_error = {"type": type(exc).__name__, "message": str(exc)}
    result = {"finished_utc": utc(), "scope": "public capability only",
              "transport_status": "COMPLETED" if error is None and close_error is None else "INCOMPLETE",
              "request_attempts": count, "response_observations": len(observations),
              "observations": observations, "selections": selections,
              "error": error, "close_error": close_error, "connection_closed": connection_closed,
              "safety": SAFETY, "authentication_used": False, "orders_sent": 0,
              "profit_tested": False, "prediction_tested": False,
              "cfd_mapping_verified": False, "actual_fills_verified": False,
              "measured_broker_costs_verified": False,
              "artifact_sha256": {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(probe(args.output))
    print(json.dumps({k: result[k] for k in ("transport_status", "request_attempts",
                     "response_observations", "error", "close_error", "safety")}, indent=2))
    return 0 if result["transport_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
