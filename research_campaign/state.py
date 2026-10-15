"""Atomic, resumable research state and exclusive execution ownership."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SAFETY = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)
INTERVAL_SECONDS = 900
TARGET_CYCLES = 32


def guard() -> dict:
    for name in (*SAFETY, *(name.lower() for name in SAFETY)):
        if os.environ.get(name, "false").strip().lower() != "false":
            raise RuntimeError(f"Research requires {name}=false")
    return dict(SAFETY)


def utc(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat() if epoch is not None else datetime.now(timezone.utc).isoformat()


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if hasattr(value, "item"):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def encoded(value) -> bytes:
    return (json.dumps(finite_json(value), sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic_json(path: Path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Publish a complete file without replacing an existing immutable
            # record; an interrupted write can only leave an unused temp file.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    def bad(constant):
        raise ValueError(f"Invalid JSON constant: {constant}")
    return json.loads(Path(path).read_text(), parse_constant=bad)


@contextmanager
def campaign_lock(directory: Path):
    guard()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Campaign is already running; no duplicate cycle") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()))
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def live_owner(directory: Path) -> dict:
    """Check an actual held kernel lock and PID, never infer life from state alone."""
    path = directory / "runner.lock"
    if not path.exists():
        return {"live": False, "pid": None, "lock_held": False}
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            try:
                pid = int(stream.read().strip())
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError, PermissionError):
                return {"live": False, "pid": None, "lock_held": True}
            return {"live": True, "pid": pid, "lock_held": True}
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return {"live": False, "pid": None, "lock_held": False}


def next_due(state: dict) -> float:
    completed = state["completed_cycles"]
    if not 0 <= completed <= TARGET_CYCLES:
        raise ValueError("Invalid actual cycle count")
    scheduled = state["started_epoch"] + completed * INTERVAL_SECONDS
    if completed and state.get("cycles"):
        last_start = datetime.fromisoformat(state["cycles"][-1]["started_at"]).timestamp()
        scheduled = max(scheduled, last_start + INTERVAL_SECONDS)
    return scheduled


def validate_state(state: dict):
    guard()
    if state["safety"] != SAFETY or state["target_cycles"] != TARGET_CYCLES or state["interval_seconds"] != INTERVAL_SECONDS:
        raise ValueError("Campaign safety or schedule changed")
    cycles = state.get("cycles", [])
    actual = [r for r in cycles if r.get("status") == "COMPLETED"]
    if state["completed_cycles"] != len(actual):
        raise ValueError("Cycle count lacks completed execution records")
    if [r["number"] for r in actual] != list(range(1, len(actual) + 1)):
        raise ValueError("Completed cycle records are not contiguous")
    if state.get("final_accessed") and state["completed_cycles"] != TARGET_CYCLES:
        raise ValueError("Final holdout opened before development completed")
    return state
