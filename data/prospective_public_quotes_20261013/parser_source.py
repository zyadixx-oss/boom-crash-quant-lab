#!/usr/bin/env python3
"""Bounded public quote schema observation; no returns, signals, accounts or orders.

The observed bid/ask are public-feed fields. They have not been established as
executable CFD prices, historical spreads, actual fills or broker costs.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import ssl
import statistics
import sys
import time

import certifi
import websockets

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOLS = ("BOOM500", "CRASH500", "BOOM600", "CRASH600")
SAFETY = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)
DOCS = ["https://developers.deriv.com/docs/options/ws-public/",
        "https://developers.deriv.com/comparison/ticks/",
        "https://github.com/deriv-com/deriv-api-schemas/blob/master/schemas/ticks_response.schema.json"]


class ContractError(ValueError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safety_guard():
    for name in SAFETY:
        if os.environ.get(name, "false").strip().lower() != "false":
            raise ContractError(f"Public observation requires {name}=false")


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ContractError(f"{name} must be an integer in [{low}, {high}]")


def validate_request(request):
    if set(request) != {"ticks", "subscribe", "req_id"}:
        raise ContractError("Only the exact public ticks request is allowed")
    if request["ticks"] not in SYMBOLS or type(request["subscribe"]) is not int or request["subscribe"] != 1:
        raise ContractError("Unknown symbol or subscription mode")
    integer(request["req_id"], "req_id", 1, len(SYMBOLS))
    if SYMBOLS[request["req_id"] - 1] != request["ticks"]:
        raise ContractError("Request symbol/ID mismatch")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def positive_decimal(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ContractError(f"{name} is not a JSON number")
    number = Decimal(value)
    if not number.is_finite() or number <= 0:
        raise ContractError(f"{name} must be finite and positive")
    return number


def validate_response(raw):
    def invalid_constant(value):
        raise ContractError(f"Invalid JSON number: {value}")
    if not isinstance(raw, str):
        raise ContractError("Expected a text WebSocket message")
    value = json.loads(raw, parse_float=Decimal, object_pairs_hook=unique_object,
                       parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ContractError("Response must be an object")
    if "error" in value:
        raise ContractError(f"Public API error: {value['error']!r}")
    if value.get("msg_type") != "tick" or not isinstance(value.get("tick"), dict):
        raise ContractError("Expected tick response")
    req_id = value.get("req_id")
    integer(req_id, "response req_id", 1, len(SYMBOLS))
    tick = value["tick"]
    if tick.get("symbol") != SYMBOLS[req_id - 1]:
        raise ContractError("Response symbol/ID mismatch")
    integer(tick.get("epoch"), "epoch", 1, 2**53 - 1)
    quote = positive_decimal(tick.get("quote"), "quote")
    # Missing or null optional sides remain unknown, never substituted by quote.
    bid = None if tick.get("bid") is None else positive_decimal(tick["bid"], "bid")
    ask = None if tick.get("ask") is None else positive_decimal(tick["ask"], "ask")
    complete = bid is not None and ask is not None
    if (bid is not None and bid > quote) or (ask is not None and ask < quote):
        raise ContractError("Public quote violates observed bid/quote/ask ordering")
    return {"symbol": tick["symbol"], "req_id": req_id, "epoch": tick["epoch"],
            "quote": str(quote), "bid": None if bid is None else str(bid),
            "ask": None if ask is None else str(ask), "tick_keys": sorted(tick),
            "bid_ask_status": "PRESENT" if complete else "UNKNOWN",
            "quote_within_bid_ask": True if complete else None,
            "public_feed_spread_price_units": str(ask - bid) if complete else None}


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def artifact_hashes(output):
    return {path.name: sha(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}


async def probe(output, *, samples=60, duration=180, receive_timeout=20, connector=None):
    safety_guard()
    integer(samples, "samples", 1, 120)
    integer(duration, "duration", 1, 300)
    integer(receive_timeout, "receive_timeout", 1, 30)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (output / "probe_source.py").write_bytes(script)
    test_path = ROOT / "backend/tests/test_public_quote_contract.py"
    test_sha = sha(test_path.read_bytes()) if test_path.is_file() else None
    if test_path.is_file():
        (output / "probe_tests.py").write_bytes(test_path.read_bytes())
    started = time.monotonic()
    declaration = {"started_utc": utc(), "endpoint": ENDPOINT, "symbols": list(SYMBOLS),
                   "samples_per_symbol": samples, "max_seconds": duration,
                   "receive_timeout_seconds": receive_timeout, "max_frames": 8 * samples + 32,
                   "safety": SAFETY, "authentication_used": False, "docs": DOCS,
                   "source_sha256": sha(script), "tests_sha256": test_sha,
                   "python_version": sys.version, "platform": platform.platform(),
                   "websockets_version": importlib.metadata.version("websockets"),
                   "certifi_version": importlib.metadata.version("certifi"),
                   "scope": "public feed schema and quote ordering only",
                   "economic_hypotheses_evaluated": False, "price_csvs_read": False,
                   "executable_cfd_verified": False, "actual_fills_verified": False,
                   "broker_costs_verified": False, "profit_tested": False}
    save_json(output / "declaration.json", declaration)
    for name in ("requests.jsonl", "received_text_messages.jsonl", "observations.jsonl"):
        (output / name).touch(exist_ok=False)
    rows = {symbol: {} for symbol in SYMBOLS}
    counts = {symbol: {"distinct_epochs": 0, "duplicate_epochs": 0,
                       "post_cap_messages": 0, "with_bid_ask": 0, "unknown_bid_ask": 0,
                       "bid_quote_ask_checks_passed": 0} for symbol in SYMBOLS}
    ws = None
    error = None
    close_error = None
    frame_count = 0
    connection_closed = False
    try:
        connect = connector or websockets.connect
        ws = await asyncio.wait_for(connect(ENDPOINT,
                    ssl=ssl.create_default_context(cafile=certifi.where()), open_timeout=20,
                    close_timeout=5, ping_interval=20, ping_timeout=20, max_size=65536, max_queue=16),
                    timeout=min(20, duration))
        for req_id, symbol in enumerate(SYMBOLS, 1):
            request = {"ticks": symbol, "subscribe": 1, "req_id": req_id}
            validate_request(request)
            text = json.dumps(request, separators=(",", ":"))
            append_json(output / "requests.jsonl", {"utc": utc(), "event": "send_attempt",
                        "request": request, "text": text, "text_sha256": sha(text.encode())})
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Overall deadline before all subscriptions sent")
            await asyncio.wait_for(ws.send(text), timeout=min(receive_timeout, remaining))
            append_json(output / "requests.jsonl", {"utc": utc(), "event": "send_completed", "req_id": req_id})
        while any(len(items) < samples for items in rows.values()):
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Overall observation deadline")
            if frame_count >= declaration["max_frames"]:
                raise ContractError("Bounded message cap reached before sample target")
            raw = await asyncio.wait_for(ws.recv(), timeout=min(receive_timeout, remaining))
            frame_count += 1
            received = utc()
            # Exact decoded text is preserved; WebSocket framing/TLS bytes are not captured.
            raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
            append_json(output / "received_text_messages.jsonl", {
                "received_utc": received, "frame_number": frame_count,
                "payload_type": "text" if isinstance(raw, str) else "binary",
                "text": raw if isinstance(raw, str) else None,
                "binary_hex": raw.hex() if isinstance(raw, bytes) else None,
                "payload_sha256": sha(raw_bytes)})
            record = validate_response(raw)
            symbol = record["symbol"]
            epoch = record["epoch"]
            previous = rows[symbol]
            if epoch in previous:
                if any(record[key] != previous[epoch][key] for key in ("quote", "bid", "ask")):
                    raise ContractError("Conflicting quotes at an already observed epoch")
                counts[symbol]["duplicate_epochs"] += 1
                action = "duplicate_epoch"
            elif previous and epoch < max(previous):
                raise ContractError("Out-of-order public epoch")
            elif len(previous) == samples:
                counts[symbol]["post_cap_messages"] += 1
                action = "outside_sample_cap"
            else:
                previous[epoch] = record
                counts[symbol]["distinct_epochs"] += 1
                complete = record["bid_ask_status"] == "PRESENT"
                counts[symbol]["with_bid_ask"] += int(complete)
                counts[symbol]["unknown_bid_ask"] += int(not complete)
                counts[symbol]["bid_quote_ask_checks_passed"] += int(complete)
                action = "sampled"
            append_json(output / "observations.jsonl", {"received_utc": received,
                        "frame_number": frame_count, "action": action, **record})
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=6)
                connection_closed = True
            except Exception as exc:
                close_error = {"type": type(exc).__name__, "message": str(exc)}
    for symbol, items in rows.items():
        epochs = sorted(items)
        counts[symbol]["first_epoch"] = epochs[0] if epochs else None
        counts[symbol]["last_epoch"] = epochs[-1] if epochs else None
        counts[symbol]["missing_seconds_within_observed_span"] = (
            epochs[-1] - epochs[0] + 1 - len(epochs) if epochs else None)
        spreads = [Decimal(row["public_feed_spread_price_units"])
                   for row in items.values() if row["bid_ask_status"] == "PRESENT"]
        counts[symbol]["public_feed_spread_summary_price_units"] = {
            "observations": len(spreads),
            "minimum": str(min(spreads)) if spreads else None,
            "median": str(statistics.median(spreads)) if spreads else None,
            "maximum": str(max(spreads)) if spreads else None,
            "interpretation": "observed_public_feed_only_not_executable_CFD_cost",
        }
    result = {"status": "PASS" if error is None and close_error is None else "FAIL",
              "finished_utc": utc(), "elapsed_seconds": time.monotonic() - started,
              "counts": counts, "received_messages": frame_count,
              "connection_closed": connection_closed, "error": error, "close_error": close_error,
              "safety": SAFETY, "authentication_used": False,
              "executable_cfd_verified": False, "actual_fills_verified": False,
              "broker_costs_verified": False, "profit_tested": False,
              "transport_timing_is_order_execution_latency": False,
              "artifact_sha256": artifact_hashes(output)}
    save_json(output / "result.json", result)
    (output / "result.sha256").write_text(sha((output / "result.json").read_bytes()) + "  result.json\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--receive-timeout", type=int, default=20)
    args = parser.parse_args()
    result = asyncio.run(probe(args.output, samples=args.samples, duration=args.duration,
                               receive_timeout=args.receive_timeout))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
