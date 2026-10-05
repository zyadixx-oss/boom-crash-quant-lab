#!/usr/bin/env python3
"""Public-only UTC-day tick acquisition with preserved wire/retry provenance.

Python3.12+. No indicators, targets, quote returns, interpolation or orders.
The current public endpoint empirically caps tick pages at1000. Explicit start
is mandatory: older end-only tick requests can silently return current data.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import time

import certifi
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SAFETY = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)
SYMBOLS = ("BOOM600", "CRASH600")
CADENCE = 1
MAX_PAGE = 1000
MAX_ATTEMPTS = 4
PUBLIC_KEYS = {"ticks_history", "count", "start", "end", "style", "req_id"}


class SemanticError(ValueError):
    """Unsafe or inconsistent public response; never automatically retried."""


class RateLimitError(Exception):
    pass


def safety_guard():
    for flag in SAFETY:
        if os.environ.get(flag, "false").strip().lower() != "false":
            raise ValueError(f"Public research collector requires {flag}=false")
    return dict(SAFETY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def key_for(request):
    return hashlib.sha256(encoded({"endpoint": ENDPOINT, "request": request})).hexdigest()


def portable(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Tick output files must remain inside the repository")
    return str(resolved.relative_to(ROOT))


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def read_json(path):
    def invalid(value):
        raise SemanticError(f"Invalid JSON constant {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value, immutable=False):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if immutable:
        with path.open("x") as file:
            file.write(data)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(data)
        os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def lines(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def date_bounds(value):
    stamp = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if stamp.strftime("%Y-%m-%d") != value:
        raise ValueError("Dates must use canonical YYYY-MM-DD")
    start = int(stamp.timestamp())
    return start, start + 86400


def validate_request(request):
    if set(request) == {"active_symbols", "req_id"} and request["active_symbols"] == "brief":
        return
    if set(request) != PUBLIC_KEYS or request.get("ticks_history") not in SYMBOLS or request.get("style") != "ticks":
        raise SemanticError("Only declared public tick history and active_symbols are permitted")
    for field in ("start", "end", "count", "req_id"):
        if isinstance(request[field], bool) or not isinstance(request[field], int):
            raise SemanticError(f"Public request {field} must be an integer")
    if not 0 <= request["start"] <= request["end"] or not 1 <= request["count"] <= MAX_PAGE:
        raise SemanticError("Invalid explicit window or unverified tick page size")


def tick_rows(response, request):
    if response.get("req_id") != request["req_id"]:
        raise SemanticError("Unmatched public history response")
    if response.get("error"):
        error = response["error"]
        if error.get("code") == "RateLimit":
            raise RateLimitError(json.dumps(error))
        raise SemanticError(f"Nonrecoverable public API error: {json.dumps(error)}")
    if response.get("msg_type") != "history":
        raise SemanticError("Unexpected public history message type")
    history = response.get("history")
    if not isinstance(history, dict) or set(history) & {"times", "prices"} != {"times", "prices"}:
        raise SemanticError("Missing tick history arrays")
    times, prices = history["times"], history["prices"]
    if not isinstance(times, list) or not isinstance(prices, list) or len(times) != len(prices) or len(times) > request["count"]:
        raise SemanticError("Invalid tick history array sizes")
    rows, previous = [], None
    for stamp, price in zip(times, prices, strict=True):
        if isinstance(stamp, bool) or not isinstance(stamp, int) or not request["start"] <= stamp <= request["end"]:
            raise SemanticError("Tick response lies outside the explicit UTC-second window")
        if isinstance(price, bool):
            raise SemanticError("Invalid boolean quote")
        try:
            quote = Decimal(str(price))
        except Exception as exc:
            raise SemanticError("Invalid quote") from exc
        if not quote.is_finite() or quote <= 0 or previous is not None and stamp < previous:
            raise SemanticError("Invalid quote or unordered ticks")
        rows.append((stamp, format(quote, "f")))
        previous = stamp
    return rows


def insert_rows(prices, rows):
    duplicates = 0
    for stamp, quote in rows:
        if stamp in prices:
            if Decimal(prices[stamp]) != Decimal(quote):
                raise SemanticError("Unequal duplicate quotes cannot be sequenced safely")
            duplicates += 1
        else:
            prices[stamp] = quote
    return duplicates


def missing_ranges(epochs, start, end):
    ranges, before = [], start - 1
    for after in (*epochs, end):
        if after > before + 1:
            ranges.append({"start_epoch": before + 1, "end_exclusive_epoch": after,
                           "missing_seconds": after - before - 1})
        before = after
    return ranges


def files_for(output, symbol, date):
    prefix = f"{symbol.lower()}_{date}"
    return {"raw": output / f"{prefix}_raw_pages.jsonl", "audit": output / f"{prefix}_page_audit.jsonl",
            "clean": output / f"{prefix}_ticks_clean.csv", "checkpoint": output / f"{prefix}_checkpoint.json",
            "manifest": output / f"{symbol}_{date}_manifest.json"}


class PublicClient:
    def __init__(self, delay=.35):
        self.ws = None
        self.delay = max(float(delay), .5)
        self.last_sent = None

    async def reset(self):
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def request(self, request):
        safety_guard()
        validate_request(request)
        if self.ws is None:
            self.ws = await websockets.connect(ENDPOINT, ssl=ssl.create_default_context(cafile=certifi.where()),
                                               open_timeout=20, ping_interval=20, ping_timeout=20, max_size=8_000_000)
        if self.last_sent is not None:
            await asyncio.sleep(max(0., self.delay - (time.monotonic() - self.last_sent)))
        self.last_sent = time.monotonic()
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(self.ws.recv(), timeout=35)


def checkpoint(files, state):
    for field in ("raw", "audit"):
        state[f"{field}_sha256"] = digest(files[field])
    write_json(files["checkpoint"], state)


def verified_state(files, config_hash, symbol, date, start, end, page_size):
    state = read_json(files["checkpoint"])
    expected = {"config_sha256": config_hash, "symbol": symbol, "date": date,
                "start_epoch": start, "end_exclusive_epoch": end, "endpoint": ENDPOINT, "page_size": page_size}
    for name, value in expected.items():
        if state.get(name) != value:
            raise SemanticError(f"Resume checkpoint changed: {name}")
    if state.get("semantic_failure"):
        raise SemanticError("A semantic failure requires separate reviewed acquisition, not resume")
    for name in ("raw", "audit"):
        if digest(files[name]) != state[f"{name}_sha256"]:
            raise SemanticError("Immutable raw/audit bytes differ from checkpoint")
    raw, audit = lines(files["raw"]), lines(files["audit"])
    if len(raw) != len(audit):
        raise SemanticError("Raw/audit attempt counts differ")
    prices, cursor, success_count, errors = {}, end - 1, 0, []
    for record, page in zip(raw, audit, strict=True):
        request = record["request"]
        validate_request(request)
        if request["ticks_history"] != symbol or request["start"] != start or request["end"] != cursor:
            raise SemanticError("Checkpoint public page chain changed")
        if request["count"] != page_size or request["req_id"] != 1000 + success_count:
            raise SemanticError("Checkpoint page-size or request-ID sequence changed")
        if (record["endpoint"] != ENDPOINT or page["endpoint"] != ENDPOINT or page["request"] != request or
                page["request_key"] != record["request_key"] or page["request_key"] != key_for(request)):
            raise SemanticError("Checkpoint endpoint/request changed")
        if record.get("response_wire") is not None:
            wire = record["response_wire"].encode()
            if hashlib.sha256(wire).hexdigest() != record["wire_sha256"] or page["wire_sha256"] != record["wire_sha256"]:
                raise SemanticError("Checkpoint wire hash changed")
        if page["accepted"]:
            response = json.loads(record["response_wire"], parse_float=Decimal)
            rows = tick_rows(response, request)
            if (page["rows"] != len(rows) or page["oldest_epoch"] != (rows[0][0] if rows else None) or
                    page["newest_epoch"] != (rows[-1][0] if rows else None)):
                raise SemanticError("Checkpoint successful-page bounds/count changed")
            insert_rows(prices, rows)
            cursor = rows[0][0] - 1 if rows else start - 1
            success_count += 1
        elif page.get("recoverable") is not True:
            raise SemanticError("Checkpoint contains an unmatched semantic failure")
        else:
            if record["error"] != page["error"]:
                raise SemanticError("Checkpoint error audit changed")
            errors.append(record["error"])
    if cursor != state["next_end"] or success_count != state["success_pages"]:
        raise SemanticError("Checkpoint cursor/count differs from preserved wire pages")
    if errors != state["request_errors"]:
        raise SemanticError("Checkpoint retained request errors changed")
    return state, prices


def recovery_lineage(errors, pages):
    successes = {}
    for i, page in enumerate(pages):
        if page["accepted"]:
            if page["request_key"] in successes:
                raise SemanticError("Duplicate successful request key")
            successes[page["request_key"]] = (i, page)
    result = []
    for i, error in enumerate(errors):
        if error["request_key"] not in successes:
            raise SemanticError("Request error has no matching successful retry")
        ordinal, page = successes[error["request_key"]]
        if error["request"] != page["request"] or error["endpoint"] != page["endpoint"]:
            raise SemanticError("Recovered retry changed endpoint/start/end/request")
        result.append({"error_index": i, "successful_page_audit_index": ordinal,
                       "request_key": error["request_key"], "successful_wire_sha256": page["wire_sha256"],
                       "endpoint": ENDPOINT, "start_epoch": page["request"]["start"], "end_epoch": page["request"]["end"]})
    return result


async def collect_day(client, output, symbol, date, config_hash, page_size=MAX_PAGE, resume=False):
    safety_guard()
    start, end = date_bounds(date)
    files = files_for(output, symbol, date)
    if files["manifest"].exists():
        if not resume:
            raise FileExistsError("Completed source is immutable; use --resume to verify and skip")
        manifest = read_json(files["manifest"])
        if manifest["config_sha256"] != config_hash or manifest["safety"] != SAFETY:
            raise SemanticError("Completed source config/safety changed")
        identity = {"symbol": symbol, "date": date, "start_epoch": start, "end_exclusive_epoch": end,
                    "endpoint": ENDPOINT, "cadence_seconds": CADENCE, "normalization_valid": True,
                    "fills_or_interpolations": False, "authentication_used": False}
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise SemanticError("Completed source identity or public-data contract changed")
        for field, filename in (("clean_sha256", "clean"), ("raw_pages_sha256", "raw"), ("page_audit_sha256", "audit")):
            if digest(files[filename]) != manifest[field]:
                raise SemanticError("Completed source bytes changed")
        return manifest
    if files["checkpoint"].exists():
        if not resume:
            raise FileExistsError("Incomplete source exists; exact verified --resume required")
        state, prices = verified_state(files, config_hash, symbol, date, start, end, page_size)
    else:
        if any(path.exists() for path in files.values()):
            raise FileExistsError("Refusing to overwrite partial tick artifacts")
        files["raw"].touch(); files["audit"].touch()
        state = {"config_sha256": config_hash, "symbol": symbol, "date": date, "start_epoch": start,
                 "end_exclusive_epoch": end, "endpoint": ENDPOINT, "next_end": end-1,
                 "page_size": page_size, "success_pages": 0, "request_errors": [], "semantic_failure": False}
        prices = {}
        checkpoint(files, state)
    while state["next_end"] >= start:
        request = {"ticks_history": symbol, "count": page_size, "start": start,
                   "end": state["next_end"], "style": "ticks", "req_id": 1000 + state["success_pages"]}
        validate_request(request)
        request_key = key_for(request)
        recovered = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record = {"endpoint": ENDPOINT, "request": request, "request_key": request_key,
                      "attempt": attempt, "received_at_utc": datetime.now(timezone.utc).isoformat(),
                      "response_wire": None, "wire_sha256": None}
            page = {"endpoint": ENDPOINT, "request": request, "request_key": request_key,
                    "attempt": attempt, "accepted": False, "wire_sha256": None}
            retry = False
            try:
                raw = await client.request(request)
                wire = raw.decode() if isinstance(raw, bytes) else raw
                record.update(response_wire=wire, wire_sha256=hashlib.sha256(wire.encode()).hexdigest())
                page["wire_sha256"] = record["wire_sha256"]
                response = json.loads(wire, parse_float=Decimal)
                rows = tick_rows(response, request)
                duplicate_count = insert_rows(prices, rows)
                page.update(accepted=True, rows=len(rows), equal_duplicates_removed=duplicate_count,
                            oldest_epoch=rows[0][0] if rows else None, newest_epoch=rows[-1][0] if rows else None)
                state["next_end"] = rows[0][0] - 1 if rows else start - 1
                state["success_pages"] += 1
                recovered = True
            except (RateLimitError, OSError, asyncio.TimeoutError, ConnectionClosed, InvalidStatus) as exc:
                retry = True
                error = {"endpoint": ENDPOINT, "request": request, "request_key": request_key,
                         "attempt": attempt, "error_type": type(exc).__name__, "error": str(exc),
                         "wire_sha256": record["wire_sha256"]}
                state["request_errors"].append(error)
                record["error"], page["error"] = error, error
                page["recoverable"] = True
            except Exception as exc:
                state["semantic_failure"] = True
                error = {"endpoint": ENDPOINT, "request": request, "request_key": request_key,
                         "attempt": attempt, "error_type": type(exc).__name__, "error": str(exc),
                         "wire_sha256": record["wire_sha256"]}
                state["request_errors"].append(error)
                record["error"], page["error"] = error, error
                page["recoverable"] = False
            append_json(files["raw"], record)
            append_json(files["audit"], page)
            checkpoint(files, state)
            if recovered:
                break
            await client.reset()
            if not retry or attempt == MAX_ATTEMPTS:
                raise SemanticError("Tick acquisition stopped; preserved unresolved error/checkpoint")
            await asyncio.sleep(min(2 ** attempt, 16))
        if state["success_pages"] % 20 == 0:
            print(f"{symbol}/{date}: {state['success_pages']} pages, {len(prices)} known seconds", flush=True)
    pages = lines(files["audit"])
    lineage = recovery_lineage(state["request_errors"], pages)
    epochs = sorted(prices)
    gaps = missing_ranges(epochs, start, end)
    if files["clean"].exists():
        with files["clean"].open(newline="") as file:
            saved = list(csv.reader(file))
        if saved != [["epoch", "quote"], *[[str(stamp), prices[stamp]] for stamp in epochs]]:
            raise SemanticError("Finalized normalized tick CSV changed")
    else:
        with files["clean"].open("x", newline="") as file:
            writer = csv.writer(file); writer.writerow(("epoch", "quote"))
            writer.writerows((stamp, prices[stamp]) for stamp in epochs)
    manifest = {"symbol": symbol, "date": date, "start_epoch": start, "end_exclusive_epoch": end,
                "start_utc": utc(start), "end_exclusive_utc": utc(end), "cadence_seconds": CADENCE,
                "endpoint": ENDPOINT, "config_sha256": config_hash,
                "collector_file": "scripts/collect_spike_ticks.py", "collector_sha256": digest(Path(__file__)),
                "clean_file": portable(files["clean"]), "clean_sha256": digest(files["clean"]),
                "raw_pages_file": portable(files["raw"]), "raw_pages_sha256": digest(files["raw"]),
                "page_audit_file": portable(files["audit"]), "page_audit_sha256": digest(files["audit"]),
                "rows": len(epochs), "expected_grid_rows": 86400,
                "missing_seconds": sum(item["missing_seconds"] for item in gaps), "gaps": gaps,
                "first_epoch": epochs[0] if epochs else None, "last_epoch": epochs[-1] if epochs else None,
                "gap_free": not gaps, "equal_duplicates_removed": sum(page.get("equal_duplicates_removed", 0) for page in pages),
                "fills_or_interpolations": False, "authentication_used": False, "safety": SAFETY,
                "features_labels_or_tick_outcomes_computed": False,
                "request_errors": state["request_errors"], "recovered_retry_lineage": lineage,
                "successful_page_count": state["success_pages"], "normalization_valid": True,
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(files["manifest"], manifest, immutable=True)
    print(f"{symbol}/{date}: DONE rows={len(epochs)} missing={manifest['missing_seconds']}", flush=True)
    return manifest


async def bootstrap(client, output):
    request = {"active_symbols": "brief", "req_id": 1}
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        record = {"endpoint": ENDPOINT, "request": request, "attempt": attempt, "response_wire": None,
                  "wire_sha256": None, "received_at_utc": datetime.now(timezone.utc).isoformat(), "safety": SAFETY}
        try:
            raw = await client.request(request)
            wire = raw.decode() if isinstance(raw, bytes) else raw
            record.update(response_wire=wire, wire_sha256=hashlib.sha256(wire.encode()).hexdigest())
            value = json.loads(wire)
            if value.get("req_id") != 1:
                raise SemanticError("Public active-symbol response ID mismatch")
            if value.get("error"):
                if value["error"].get("code") == "RateLimit":
                    raise RateLimitError(json.dumps(value["error"]))
                raise SemanticError("Public active-symbol semantic error")
            available = {(item.get("underlying_symbol") or item.get("symbol")) for item in value["active_symbols"]}
            if not set(SYMBOLS) <= available:
                raise SemanticError("Declared native600 identifiers unavailable")
            record["accepted"] = True
            append_json(output / "active_symbols_wire.jsonl", record)
            return {"request": request, "endpoint": ENDPOINT, "request_errors": errors,
                    "successful_wire_sha256": record["wire_sha256"], "exact_same_request_recovered": True}
        except (RateLimitError, OSError, asyncio.TimeoutError, ConnectionClosed, InvalidStatus) as exc:
            record.update(accepted=False, recoverable=True, error_type=type(exc).__name__, error=str(exc))
            errors.append({key: record[key] for key in ("endpoint", "request", "attempt", "error_type", "error")})
            append_json(output / "active_symbols_wire.jsonl", record)
            await client.reset()
            if attempt == MAX_ATTEMPTS:
                raise SemanticError("Public symbol probe failed; raw errors preserved") from exc
            await asyncio.sleep(min(2 ** attempt, 16))
        except Exception as exc:
            record.update(accepted=False, recoverable=False, error_type=type(exc).__name__, error=str(exc))
            append_json(output / "active_symbols_wire.jsonl", record)
            raise


async def run(args):
    safety_guard()
    output = args.output_dir.resolve()
    portable(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"version": 1, "endpoint": ENDPOINT, "symbols": args.symbols, "dates": args.dates,
              "page_size": args.page_size, "requested_delay_seconds": args.delay,
              "effective_min_request_spacing_seconds": max(args.delay, .5), "cadence_seconds": CADENCE,
              "collector_sha256": digest(Path(__file__)), "safety": SAFETY}
    config_path = output / "acquisition_config.json"
    if config_path.exists():
        if not args.resume or read_json(config_path) != config:
            raise SemanticError("Existing acquisition requires identical immutable config and --resume")
    else:
        write_json(config_path, config, immutable=True)
    config_hash = digest(config_path)
    summary = {"config": config, "config_sha256": config_hash, "safety": SAFETY,
               "completed_sources": {symbol: {} for symbol in args.symbols}, "errors": [],
               "features_labels_or_tick_outcomes_computed": False, "authentication_used": False,
               "account_or_order_requests": False, "status": "RUNNING"}
    client = PublicClient(args.delay)
    try:
        summary["public_symbol_probe"] = await bootstrap(client, output)
        for symbol in args.symbols:
            for date in args.dates:
                manifest = await collect_day(client, output, symbol, date, config_hash, args.page_size, args.resume)
                manifest_path = files_for(output, symbol, date)["manifest"]
                summary["completed_sources"][symbol][date] = {
                    "manifest_file": portable(manifest_path), "manifest_sha256": digest(manifest_path),
                    "clean_file": manifest["clean_file"], "clean_sha256": manifest["clean_sha256"],
                    "rows": manifest["rows"], "missing_seconds": manifest["missing_seconds"]}
                write_json(output / "acquisition_summary.json", summary)
        summary["status"] = "COMPLETE"
    except Exception as exc:
        summary["errors"].append({"error_type": type(exc).__name__, "error": str(exc)})
        summary["status"] = "FAILED_PRESERVED"
        raise
    finally:
        await client.reset()
        write_json(output / "acquisition_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=list(SYMBOLS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=MAX_PAGE)
    parser.add_argument("--delay", type=float, default=.35)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.dates)) != len(args.dates) or len(set(args.symbols)) != len(args.symbols):
        parser.error("Dates and symbols must be unique")
    if not 1 <= args.page_size <= MAX_PAGE or not math.isfinite(args.delay) or args.delay < 0:
        parser.error("Verified page size1..1000 and finite nonnegative delay required")
    try:
        for date in args.dates:
            if date_bounds(date)[1] > int(time.time()):
                raise ValueError("Only fully closed UTC days can be requested")
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
