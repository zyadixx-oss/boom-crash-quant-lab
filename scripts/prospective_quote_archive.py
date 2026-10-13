#!/usr/bin/env python3
"""Bounded, unauthenticated public-feed archive; never CFD fills or profit.

Default: 60 seconds. Maximum: 900 seconds. Resume verifies sealed sessions and
starts a separate bounded session; it never patches missing historical sides.
An unsealed session is preserved and requires review rather than silent repair.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import ssl
import sys
import time

import certifi
import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake

PARSER_PATH = Path(__file__).with_name("probe_public_quote_contract.py")
_spec = importlib.util.spec_from_file_location("_prospective_public_quote_parser", PARSER_PATH)
_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parser)

ENDPOINT = _parser.ENDPOINT
SYMBOLS = ("BOOM500", "CRASH500")
SAFETY = dict.fromkeys((*_parser.SAFETY, *(key.lower() for key in _parser.SAFETY)), False)
SCOPE = "PUBLIC_FEED_OBSERVATION_ONLY"
LIMITATIONS = {
    "authentication_used": False,
    "account_or_order_requests": False,
    "fills_or_interpolations": False,
    "cfd_price_mapping_verified": False,
    "historical_costs_verified": False,
    "cost_verified": False,
    "execution_verified": False,
    "actual_fills_verified": False,
    "profit_tested": False,
    "profitability_claim_allowed": False,
    "transport_timing_is_order_execution_latency": False,
    "whole_requested_window_coverage_verified": False,
}
STREAM_FILES = ("requests.jsonl", "raw.jsonl", "observations.jsonl", "events.jsonl")
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


class ArchiveError(ValueError):
    pass


class SystemClock:
    monotonic = staticmethod(time.monotonic)
    monotonic_ns = staticmethod(time.monotonic_ns)

    @staticmethod
    def utc():
        return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def safety_guard():
    for name in SAFETY:
        if os.environ.get(name, "false").strip().lower() != "false":
            raise ArchiveError(f"Public-only archive requires {name}=false")
    return dict(SAFETY)


def strict_json(data: str):
    def invalid(value):
        raise ArchiveError(f"Invalid JSON constant: {value}")
    return json.loads(data, object_pairs_hook=_parser.unique_object, parse_constant=invalid)


def write_json(path: Path, value, *, replace=False):
    data = encoded(value)
    if not replace:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path: Path, value):
    with path.open("ab") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def lines(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ArchiveError(f"Expected a regular evidence file: {path.name}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ArchiveError(f"Unsealed/truncated JSONL preserved: {path.name}")
    return [strict_json(row) for row in raw.decode("utf-8").splitlines()]


def validate_request(request):
    _parser.validate_request(request)
    if request["ticks"] not in SYMBOLS or request["req_id"] != SYMBOLS.index(request["ticks"]) + 1:
        raise ArchiveError("Only BOOM500/CRASH500 public subscriptions are allowed")


def parse_tick(raw):
    row = _parser.validate_response(raw)
    if row["symbol"] not in SYMBOLS or row["req_id"] != SYMBOLS.index(row["symbol"]) + 1:
        raise ArchiveError("Undeclared public response symbol")
    return row


def configuration(duration=60, receive_timeout=10, max_reconnects=3):
    for name, value, low, high in (("duration", duration, 1, 900),
                                    ("receive_timeout", receive_timeout, 1, 30),
                                    ("max_reconnects", max_reconnects, 0, 3)):
        if type(value) is not int or not low <= value <= high:
            raise ArchiveError(f"{name} must be an integer in [{low}, {high}]")
    return {
        "schema_version": 1, "scope": SCOPE, "endpoint": ENDPOINT,
        "symbols": list(SYMBOLS), "duration_seconds": duration,
        "receive_timeout_seconds": receive_timeout, "max_reconnects": max_reconnects,
        "max_frames": 4 * duration * len(SYMBOLS) + 32,
        "max_payload_bytes": MAX_PAYLOAD_BYTES,
        "max_single_frame_bytes": 65536,
        "safety": SAFETY, "limitations": LIMITATIONS,
        "archive_source_sha256": digest(Path(__file__).read_bytes()),
        "parser_source_sha256": digest(PARSER_PATH.read_bytes()),
        "resume_semantics": "verify_sealed_sessions_then_start_a_distinct_bounded_session",
        "gap_definition": "unknown integer epoch seconds strictly between distinct public ticks",
    }


@contextmanager
def archive_lock(output: Path):
    safety_guard()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ArchiveError("Archive root must not be a symlink")
    path = output / ".archive.lock"
    if path.is_symlink():
        raise ArchiveError("Archive lock must not be a symlink")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError("Archive already has an active kernel lock") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()))
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def owner_status(output: Path):
    path = Path(output) / ".archive.lock"
    if not path.exists():
        return {"lock_held": False, "pid": None, "pid_exists": False}
    if path.is_symlink():
        raise ArchiveError("Archive lock must not be a symlink")
    with path.open("r+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                pid = int(stream.read().strip())
                os.kill(pid, 0)
                alive = True
            except (ValueError, ProcessLookupError, PermissionError):
                pid, alive = None, False
            return {"lock_held": True, "pid": pid, "pid_exists": alive}
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return {"lock_held": False, "pid": None, "pid_exists": False}


def blank_stats():
    return {symbol: {"messages": 0, "distinct_epochs": 0, "duplicate_epochs": 0,
                     "with_bid_ask": 0, "unknown_bid_ask": 0,
                     "first_epoch": None, "last_epoch": None,
                     "missing_epoch_seconds_between_observed_ticks": 0,
                     "gap_intervals": [], "observed_public_spread_min": None,
                     "observed_public_spread_max": None} for symbol in SYMBOLS}


def accumulate(row, context, stats, seen):
    symbol, epoch = row["symbol"], row["epoch"]
    current, known = stats[symbol], seen[symbol]
    values = tuple(None if row[key] is None else Decimal(row[key]) for key in ("quote", "bid", "ask"))
    gap = 0
    if epoch in known:
        if known[epoch] != values:
            raise ArchiveError("Conflicting public quotes at an observed integer epoch")
        action = "duplicate_epoch"
    else:
        last = current["last_epoch"]
        if last is not None and epoch < last:
            raise ArchiveError("Out-of-order public epoch")
        action = "distinct_epoch"
        if last is not None and epoch > last + 1:
            gap = epoch - last - 1
            current["gap_intervals"].append({"start_epoch": last + 1,
                                            "end_exclusive_epoch": epoch,
                                            "unknown_seconds": gap})
        known[epoch] = values
        current["first_epoch"] = epoch if current["first_epoch"] is None else current["first_epoch"]
        current["last_epoch"] = epoch
        current["distinct_epochs"] += 1
        current["missing_epoch_seconds_between_observed_ticks"] += gap
        complete = row["bid_ask_status"] == "PRESENT"
        current["with_bid_ask"] += int(complete)
        current["unknown_bid_ask"] += int(not complete)
        if complete:
            spread = Decimal(row["public_feed_spread_price_units"])
            for field, chooser in (("observed_public_spread_min", min), ("observed_public_spread_max", max)):
                prior = current[field]
                current[field] = str(spread if prior is None else chooser(Decimal(prior), spread))
    current["messages"] += 1
    current["duplicate_epochs"] += int(action == "duplicate_epoch")
    return {**context, **row, "action": action, "gap_before_seconds": gap}


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def raw_payload(record):
    if record["payload_type"] == "text" and record["binary_hex"] is None and isinstance(record["text"], str):
        payload = record["text"]
        data = payload.encode("utf-8")
    elif record["payload_type"] == "binary" and record["text"] is None:
        payload = bytes.fromhex(record["binary_hex"])
        data = payload
    else:
        raise ArchiveError("Invalid retained payload representation")
    if digest(data) != record["payload_sha256"]:
        raise ArchiveError("Retained raw payload hash mismatch")
    return payload, len(data)


def context_for(record):
    return {key: record[key] for key in ("sequence", "connection_id", "received_utc", "received_monotonic_ns")}


def reproduce(raw_rows):
    stats, seen, observations = blank_stats(), {s: {} for s in SYMBOLS}, []
    semantic_failure, payload_bytes, previous_ns = None, 0, None
    for ordinal, raw in enumerate(raw_rows, 1):
        if type(raw["sequence"]) is not int or raw["sequence"] != ordinal:
            raise ArchiveError("Raw sequence is not contiguous")
        if type(raw["connection_id"]) is not int or raw["connection_id"] < 1:
            raise ArchiveError("Invalid raw connection identity")
        ns = raw["received_monotonic_ns"]
        if type(ns) is not int or ns < 0 or previous_ns is not None and ns < previous_ns:
            raise ArchiveError("Nonmonotonic retained receive time")
        previous_ns = ns
        stamp = datetime.fromisoformat(raw["received_utc"])
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ArchiveError("Retained receipt time must be UTC")
        payload, size = raw_payload(raw)
        payload_bytes += size
        try:
            row = parse_tick(payload)
            observations.append(accumulate(row, context_for(raw), stats, seen))
        except (ValueError, TypeError, KeyError) as exc:
            semantic_failure = {"sequence": ordinal, **error_record(exc)}
            if ordinal != len(raw_rows):
                raise ArchiveError("Messages followed a nonrecoverable semantic failure") from exc
    return stats, observations, semantic_failure, payload_bytes


def verify_requests(rows, max_connections):
    completed = set()
    pending = None
    prior_connection, next_id = 0, 1
    for row in rows:
        if row["event"] == "send_attempt":
            request = row["request"]
            validate_request(request)
            cid = row["connection_id"]
            if pending is not None and cid <= pending[0]:
                raise ArchiveError("A request followed an unresolved send attempt in the same connection")
            if type(cid) is not int or not 1 <= cid <= max_connections or cid < prior_connection:
                raise ArchiveError("Invalid request connection sequence")
            if cid != prior_connection:
                next_id = 1
            if request["req_id"] != next_id:
                raise ArchiveError("Public subscription order changed")
            prior_connection, next_id = cid, next_id + 1
            if strict_json(row["text"]) != request or digest(row["text"].encode()) != row["text_sha256"]:
                raise ArchiveError("Sent public request/hash mismatch")
            pending = (cid, request["req_id"])
        elif row["event"] == "send_completed":
            if pending != (row["connection_id"], row["req_id"]):
                raise ArchiveError("Unmatched completed send")
            completed.add(pending)
            pending = None
        else:
            raise ArchiveError("Undeclared request event")
    return completed


def verify_session(directory: Path, config, config_sha):
    if directory.is_symlink():
        raise ArchiveError("Session must not be a symlink")
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ArchiveError(f"Unsealed session preserved; review required: {directory.name}")
    receipt = strict_json(receipt_path.read_text())
    declaration_path = directory / "declaration.json"
    declaration = strict_json(declaration_path.read_text())
    if declaration["config_sha256"] != config_sha or declaration["session_id"] != directory.name:
        raise ArchiveError("Session declaration/config identity mismatch")
    if (declaration["duration_seconds"] != config["duration_seconds"] or declaration["safety"] != SAFETY or
            declaration["scope"] != SCOPE or declaration["limitations"] != LIMITATIONS):
        raise ArchiveError("Session declaration bounds/safety/scope changed")
    if receipt["config_sha256"] != config_sha or receipt["session_id"] != directory.name:
        raise ArchiveError("Session receipt/config identity mismatch")
    if receipt["safety"] != SAFETY or receipt["limitations"] != LIMITATIONS or receipt["scope"] != SCOPE:
        raise ArchiveError("Session safety/scope claim changed")
    expected = {"declaration.json", *STREAM_FILES}
    if set(receipt["files"]) != expected or {p.name for p in directory.iterdir()} != expected | {"receipt.json"}:
        raise ArchiveError("Session evidence file set changed")
    for name, saved in receipt["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ArchiveError("Evidence must be a regular file")
        data = path.read_bytes()
        if saved != {"sha256": digest(data), "bytes": len(data)}:
            raise ArchiveError(f"Evidence hash/size mismatch: {name}")
    requests = lines(directory / "requests.jsonl")
    completed_sends = verify_requests(requests, config["max_reconnects"] + 1)
    raw_rows = lines(directory / "raw.jsonl")
    stats, observations, failure, size = reproduce(raw_rows)
    if observations != lines(directory / "observations.jsonl"):
        raise ArchiveError("Observations do not reproduce from retained raw messages")
    if stats != receipt["counts"] or failure != receipt["semantic_failure"]:
        raise ArchiveError("Receipt counts/semantic failure do not reproduce")
    if size != receipt["payload_bytes"] or len(raw_rows) != receipt["received_frames"]:
        raise ArchiveError("Receipt raw frame/byte counts differ")
    if len(raw_rows) > config["max_frames"] or size > config["max_payload_bytes"] + config["max_single_frame_bytes"]:
        raise ArchiveError("Retained raw data exceeds the bounded capture envelope")
    for record in raw_rows:
        if record["connection_id"] > config["max_reconnects"] + 1:
            raise ArchiveError("Raw connection exceeds the declared reconnect bound")
    if any((row["connection_id"], row["req_id"]) not in completed_sends for row in observations):
        raise ArchiveError("Observed quote has no completed public subscription")
    events = lines(directory / "events.jsonl")
    if not events or events[-1].get("event") != "session_finished":
        raise ArchiveError("Session completion event missing")
    attempts = [row["connection_id"] for row in events if row["event"] == "connection_attempt"]
    if attempts != list(range(1, receipt["connections_attempted"] + 1)) or len(attempts) > config["max_reconnects"] + 1:
        raise ArchiveError("Connection attempt count/sequence differs from its bound or receipt")
    close_errors = [row["error"] for row in events if row["event"] == "connection_close_error"]
    if close_errors != receipt["connection_close_errors"] or events[-1]["status"] != receipt["status"]:
        raise ArchiveError("Connection closure/completion receipt differs from retained events")
    if receipt["status"] not in ("COMPLETED", "FAILED_PRESERVED"):
        raise ArchiveError("Unknown archive session status")
    if receipt["status"] == "COMPLETED" and (failure is not None or receipt["error"] is not None or close_errors or
            any(stats[s]["distinct_epochs"] == 0 for s in SYMBOLS)):
        raise ArchiveError("Incomplete or failed observations claimed as completed")
    return {"session_id": directory.name, "receipt_sha256": digest(receipt_path.read_bytes()),
            "status": receipt["status"], "received_frames": len(raw_rows)}


def verify_archive(output: Path, *, require_current_config=None):
    output = Path(output)
    if not output.is_dir() or not (output / "declaration.json").is_file():
        raise ArchiveError("An existing declared archive is required for offline verification")
    if owner_status(output)["lock_held"]:
        raise ArchiveError("Cannot verify a mutable session while its kernel lock is held")
    with archive_lock(output):
        return _verify_archive_unlocked(output, require_current_config=require_current_config)


def _verify_archive_unlocked(output, *, require_current_config=None):
    declaration_path = output / "declaration.json"
    index_path = output / "receipts_index.json"
    if declaration_path.is_symlink() or index_path.is_symlink():
        raise ArchiveError("Archive metadata must not be symlinks")
    declaration = strict_json(declaration_path.read_text())
    config = declaration["config"]
    # Current-source/config pins deliberately prevent silent resumes after code changes.
    validated = configuration(config["duration_seconds"], config["receive_timeout_seconds"], config["max_reconnects"])
    if config != validated or require_current_config is not None and config != require_current_config:
        raise ArchiveError("Pinned archive configuration/source differs from the current tool")
    index = strict_json(index_path.read_text())
    config_sha = digest(declaration_path.read_bytes())
    if index["declaration_sha256"] != config_sha:
        raise ArchiveError("Archive declaration hash mismatch")
    for name, key in (("archive_source.py", "archive_source_sha256"), ("parser_source.py", "parser_source_sha256")):
        path = output / name
        if path.is_symlink() or digest(path.read_bytes()) != config[key]:
            raise ArchiveError("Pinned source snapshot hash mismatch")
    sessions = sorted(output.glob("session_*"))
    if [p.name for p in sessions] != [f"session_{i:06d}" for i in range(1, len(sessions) + 1)]:
        raise ArchiveError("Archive session sequence changed")
    verified = [verify_session(path, config, config_sha) for path in sessions]
    if verified != index["sessions"]:
        raise ArchiveError("Receipt index does not match sealed sessions")
    return {"status": "VERIFIED", "scope": SCOPE, "sessions": verified,
            "declaration_sha256": config_sha, "safety": SAFETY, "limitations": LIMITATIONS}


def _prepare(output, config, resume, clock):
    declaration = output / "declaration.json"
    if declaration.exists():
        if not resume:
            raise ArchiveError("Archive exists; use --resume to verify and add a separate bounded session")
        checked = _verify_archive_unlocked(output, require_current_config=config)
        return checked["declaration_sha256"], checked["sessions"]
    if {p.name for p in output.iterdir()} - {".archive.lock"}:
        raise ArchiveError("Unrecognized partial archive preserved; use a new output after review")
    if resume:
        raise ArchiveError("--resume requires an existing sealed archive")
    write_json(declaration, {"created_utc": clock.utc(), "config": config})
    (output / "archive_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "parser_source.py").write_bytes(PARSER_PATH.read_bytes())
    pin = digest(declaration.read_bytes())
    write_json(output / "receipts_index.json", {"declaration_sha256": pin, "sessions": []})
    return pin, []


async def archive(output, *, duration=60, receive_timeout=10, max_reconnects=3,
                  resume=False, connector=None, clock=None, sleep=None):
    safety_guard()
    config = configuration(duration, receive_timeout, max_reconnects)
    output, clock = Path(output), clock or SystemClock()
    sleep = sleep or asyncio.sleep
    with archive_lock(output):
        pin, previous_sessions = _prepare(output, config, resume, clock)
        session_id = f"session_{len(previous_sessions) + 1:06d}"
        directory = output / session_id
        directory.mkdir(exist_ok=False)
        started = clock.monotonic()
        deadline = started + duration
        write_json(directory / "declaration.json", {
            "session_id": session_id, "config_sha256": pin, "started_utc": clock.utc(),
            "duration_seconds": duration, "safety": SAFETY, "scope": SCOPE,
            "python_version": sys.version, "platform": platform.platform(),
            "websockets_version": importlib.metadata.version("websockets"),
            "certifi_version": importlib.metadata.version("certifi"),
            "limitations": LIMITATIONS,
        })
        for name in STREAM_FILES:
            (directory / name).touch(exist_ok=False)
        stats, seen = blank_stats(), {symbol: {} for symbol in SYMBOLS}
        sequence, payload_bytes, connection_id = 0, 0, 0
        error, semantic_failure, ws = None, None, None
        close_errors = []
        connect = connector or websockets.connect

        def event(name, **fields):
            append_json(directory / "events.jsonl", {"event": name, "utc": clock.utc(),
                        "connection_id": connection_id, **fields})

        async def close_socket():
            nonlocal ws
            if ws is not None:
                try:
                    await asyncio.wait_for(ws.close(), timeout=5)
                    event("connection_closed")
                except Exception as exc:
                    close_error = error_record(exc)
                    close_errors.append(close_error)
                    event("connection_close_error", error=close_error)
                finally:
                    ws = None

        try:
            while clock.monotonic() < deadline:
                safety_guard()
                if sequence >= config["max_frames"] or payload_bytes >= config["max_payload_bytes"]:
                    raise ArchiveError("Declared frame/payload bound reached")
                if ws is None:
                    connection_id += 1
                    event("connection_attempt")
                    try:
                        safety_guard()
                        ws = await asyncio.wait_for(connect(ENDPOINT,
                            ssl=ssl.create_default_context(cafile=certifi.where()), open_timeout=20,
                            close_timeout=5, ping_interval=20, ping_timeout=20,
                            max_size=config["max_single_frame_bytes"], max_queue=16),
                            timeout=min(20, max(.001, deadline - clock.monotonic())))
                        event("connection_opened")
                        for req_id, symbol in enumerate(SYMBOLS, 1):
                            request = {"ticks": symbol, "subscribe": 1, "req_id": req_id}
                            validate_request(request)
                            safety_guard()
                            text = encoded(request).decode().rstrip("\n")
                            append_json(directory / "requests.jsonl", {"event": "send_attempt", "utc": clock.utc(),
                                "connection_id": connection_id, "request": request, "text": text,
                                "text_sha256": digest(text.encode())})
                            await asyncio.wait_for(ws.send(text), timeout=min(receive_timeout,
                                max(.001, deadline - clock.monotonic())))
                            append_json(directory / "requests.jsonl", {"event": "send_completed", "utc": clock.utc(),
                                "connection_id": connection_id, "req_id": req_id})
                    except (OSError, TimeoutError, ConnectionClosed, InvalidHandshake) as exc:
                        event("transport_error", error=error_record(exc))
                        await close_socket()
                        if connection_id > max_reconnects:
                            raise ArchiveError("Transport reconnect limit exhausted") from exc
                        await sleep(min(2 ** connection_id, max(0, deadline - clock.monotonic())))
                        continue
                remaining = deadline - clock.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(receive_timeout, remaining))
                except (OSError, TimeoutError, ConnectionClosed, InvalidHandshake) as exc:
                    if clock.monotonic() >= deadline:
                        event("bounded_deadline_reached")
                        break
                    event("transport_error", error=error_record(exc))
                    await close_socket()
                    if connection_id > max_reconnects:
                        raise ArchiveError("Transport reconnect limit exhausted") from exc
                    await sleep(min(2 ** connection_id, max(0, deadline - clock.monotonic())))
                    continue
                if not isinstance(raw, (str, bytes)):
                    raise ArchiveError("Unexpected WebSocket payload type")
                sequence += 1
                raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
                record = {"sequence": sequence, "connection_id": connection_id,
                          "received_utc": clock.utc(), "received_monotonic_ns": clock.monotonic_ns(),
                          "payload_type": "text" if isinstance(raw, str) else "binary",
                          "text": raw if isinstance(raw, str) else None,
                          "binary_hex": raw.hex() if isinstance(raw, bytes) else None,
                          "payload_sha256": digest(raw_bytes)}
                # Preserve and fsync exact decoded payload bytes BEFORE attempting JSON/schema parsing.
                append_json(directory / "raw.jsonl", record)
                payload_bytes += len(raw_bytes)
                try:
                    row = parse_tick(raw)
                    observation = accumulate(row, context_for(record), stats, seen)
                except (ValueError, TypeError, KeyError) as exc:
                    semantic_failure = {"sequence": sequence, **error_record(exc)}
                    raise
                append_json(directory / "observations.jsonl", observation)
            if any(stats[symbol]["distinct_epochs"] == 0 for symbol in SYMBOLS):
                raise ArchiveError("No valid quote retained for one or more declared symbols")
        except asyncio.CancelledError as exc:
            error = error_record(exc)
            event("interrupted", error=error)
        except Exception as exc:
            error = error_record(exc)
            event("failed_preserved", error=error)
        finally:
            await close_socket()
        if error is None and close_errors:
            error = error_record(ArchiveError("One or more socket closures could not be verified"))
        status = "COMPLETED" if error is None else "FAILED_PRESERVED"
        event("session_finished", status=status)
        receipt = {"session_id": session_id, "config_sha256": pin, "scope": SCOPE,
                   "status": status, "finished_utc": clock.utc(),
                   "elapsed_seconds": clock.monotonic() - started,
                   "connections_attempted": connection_id, "received_frames": sequence,
                   "payload_bytes": payload_bytes, "counts": stats, "error": error,
                   "connection_close_errors": close_errors,
                   "semantic_failure": semantic_failure, "safety": SAFETY,
                   "limitations": LIMITATIONS,
                   "files": {name: {"sha256": digest((directory / name).read_bytes()),
                                    "bytes": (directory / name).stat().st_size}
                             for name in ("declaration.json", *STREAM_FILES)}}
        write_json(directory / "receipt.json", receipt)
        verified = verify_session(directory, config, pin)
        write_json(output / "receipts_index.json", {"declaration_sha256": pin,
                   "sessions": [*previous_sessions, verified]}, replace=True)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--receive-timeout", type=int, default=10)
    parser.add_argument("--max-reconnects", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Offline integrity/derivation verification only")
    args = parser.parse_args()
    try:
        if args.verify:
            if args.resume:
                parser.error("--verify and --resume are mutually exclusive")
            result = verify_archive(args.output)
        else:
            result = asyncio.run(archive(args.output, duration=args.duration,
                receive_timeout=args.receive_timeout, max_reconnects=args.max_reconnects,
                resume=args.resume))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] in ("COMPLETED", "VERIFIED") else 1
    except (ArchiveError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAILED_PRESERVED", "error": error_record(exc),
                          "safety": SAFETY, "limitations": LIMITATIONS}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
