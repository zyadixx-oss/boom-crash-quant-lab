"""Export quotes from an already running MT5 terminal; never submit trades.

Windows only. No credentials are accepted, no terminal is installed, and no
login, order, deal-history, or account-changing API is called. A demo connection
is required by default. Existing live-account quotes require an explicit flag
and the same disabled-trading protections. Quote history is neither real fills
nor a historical contract/fee schedule.

Example (symbol names must be verified in the existing terminal)::

    python scripts/export_mt5_quotes.py --terminal-path C:\\MT5\\terminal64.exe \
      --symbol BOOM500 "Boom 500 Index" --symbol CRASH500 "Crash 500 Index" \
      --window 2025-11-01T00:00:00Z 2025-11-01T00:01:00Z \
      --output C:\\research\\quotes-001

The end of each window is exclusive. At most two windows, each <= 24 hours,
are accepted. Exit 2 records empty/partial diagnostic evidence; it is not a
successful coverage result. Even exit 0 never certifies complete coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid

SCHEMA_VERSION = 1
TOOL_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
SAFETY = {name: False for name in (
    "LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES",
    "live_trading", "ready_for_live", "live_allowed", "opened_trades",
)}
RAW_FIELDS = ("time", "bid", "ask", "last", "volume", "time_msc", "flags", "volume_real")
ALLOWED_CALLS = frozenset({"initialize", "account_info", "terminal_info", "symbol_info",
                           "copy_ticks_range", "last_error", "shutdown"})
MAX_WINDOW_SECONDS = 86400
MAX_ROWS = 200000
TIMEOUT_CODES = {-10005, 4403}
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class ExportError(ValueError):
    """A safe fixed message, never an exception repr from the terminal."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def epoch_ms(value: datetime) -> int:
    delta = value - EPOCH
    return delta.days * 86400000 + delta.seconds * 1000 + delta.microseconds // 1000


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|\+00:00)", value
    ):
        raise ExportError("Dates must be explicit UTC ISO timestamps (Z or +00:00; millisecond precision).")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ExportError("Invalid UTC date.") from None
    return result


def strict_safety(safety, environment) -> None:
    if not isinstance(safety, dict) or set(safety) != set(SAFETY) or any(v is not False for v in safety.values()):
        raise ExportError("All eight safety flags must be boolean false.")
    for key in SAFETY:
        if key in environment and environment[key] != "false":
            raise ExportError("Safety environment flags, when present, must be the literal false.")


def validate_request(request: dict, now: datetime) -> dict:
    if set(request) != {"terminal_path", "output", "symbols", "windows", "allow_existing_live_market_data", "timeout_ms", "max_call_seconds"}:
        raise ExportError("Unexpected request fields; credentials and account-changing options are prohibited.")
    if type(request["allow_existing_live_market_data"]) is not bool:
        raise ExportError("Live-market-data permission must be boolean.")
    if type(request["timeout_ms"]) is not int or not 1000 <= request["timeout_ms"] <= 60000:
        raise ExportError("Initialization timeout must be 1000..60000 ms.")
    if type(request["max_call_seconds"]) is not int or not 1 <= request["max_call_seconds"] <= 120:
        raise ExportError("Call observation limit must be 1..120 seconds.")
    windows = request["windows"]
    if not isinstance(windows, list) or not 1 <= len(windows) <= 2:
        raise ExportError("Require one or two bounded UTC windows.")
    intervals = []
    for pair in windows:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ExportError("Invalid window.")
        start, end = map(parse_utc, pair)
        if start < EPOCH or not timedelta(0) < end - start <= timedelta(seconds=MAX_WINDOW_SECONDS) or end > now:
            raise ExportError("Each window must be past-only, positive and at most 24 hours.")
        intervals.append((start, end))
    ordered = sorted(intervals)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ExportError("Windows must not overlap.")
    symbols = request["symbols"]
    if not isinstance(symbols, dict) or not symbols or not set(symbols) <= {"BOOM500", "CRASH500"}:
        raise ExportError("Only explicit BOOM500/CRASH500 symbol mappings are supported.")
    for canonical, source in symbols.items():
        side = "Boom" if canonical == "BOOM500" else "Crash"
        if not isinstance(source, str) or not re.fullmatch(side + r"\s*500\s*Index(?:[._-][A-Za-z0-9]+)?", source, re.IGNORECASE):
            raise ExportError("Source symbol must explicitly identify the requested Boom/Crash 500 Index.")
    terminal = Path(request["terminal_path"])
    if terminal.name.lower() not in {"terminal64.exe", "terminal.exe"} or not terminal.is_file() or terminal.is_symlink():
        raise ExportError("Require the path of an existing MT5 terminal executable.")
    output = Path(request["output"])
    if output.is_symlink() or output.exists() or not output.parent.is_dir():
        raise ExportError("Output must be new, with an existing parent directory.")
    for part in [output.parent, *output.parent.parents]:
        if part.is_symlink():
            raise ExportError("Symlinked output ancestors are prohibited.")
    output = output.resolve()
    for protected in (ROOT / "data", ROOT / "research_campaign", ROOT / "frontend" / "public" / "campaign", ROOT / "docs"):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ExportError("Output must be outside frozen campaign and publication paths.")
    return {**request, "terminal_path": terminal.resolve(), "output": output, "intervals": intervals}


def existing_windows_terminal(executable: Path) -> bool:
    """Match the running executable path using Windows process APIs, no launch."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return False
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    expected = os.path.normcase(str(executable))
    try:
        success = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            if entry.szExeFile.lower() == executable.name.lower():
                handle = kernel.OpenProcess(0x1000, False, entry.th32ProcessID)
                if handle:
                    try:
                        buf, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
                        if kernel.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) and os.path.normcase(buf.value) == expected:
                            return True
                    finally:
                        kernel.CloseHandle(handle)
            success = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return False


class ReadOnlyCalls:
    def __init__(self, backend):
        self.backend = backend
        self.calls = Counter()

    def call(self, name: str, *args, **kwargs):
        if name not in ALLOWED_CALLS:
            raise ExportError("Prohibited terminal API.")
        self.calls[name] += 1
        try:
            return getattr(self.backend, name)(*args, **kwargs)
        except Exception:
            raise ExportError("Terminal API call failed; backend details withheld.") from None


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def protection(api: ReadOnlyCalls, allow_live: bool, prior=None):
    account = api.call("account_info")
    terminal = api.call("terminal_info")
    if account is None or terminal is None:
        raise ExportError("Existing connected account/terminal information is unavailable.")
    mode = field(account, "trade_mode")
    if type(mode) is not int or mode not in ({0, 2} if allow_live else {0}):
        raise ExportError("Require DEMO account; existing live quotes need explicit permission.")
    server, company = field(account, "server"), field(account, "company")
    if not isinstance(server, str) or not re.fullmatch(r"Deriv(?:SVG|BVI|FX|V|Mauritius|Cayman)?(?:[- ._][A-Za-z0-9 ._()-]+)?", server, re.IGNORECASE):
        raise ExportError("Existing account server is not an identified Deriv server.")
    if not isinstance(company, str) or not re.fullmatch(r"Deriv(?:[ .(][A-Za-z0-9 .(),-]*)?", company, re.IGNORECASE):
        raise ExportError("Existing broker company is not identified as Deriv.")
    if field(terminal, "connected") is not True or field(terminal, "trade_allowed") is not False:
        raise ExportError("Require connected terminal with automated trading disabled.")
    api_disabled = field(terminal, "tradeapi_disabled", "UNAVAILABLE")
    if api_disabled != "UNAVAILABLE" and api_disabled is not True:
        raise ExportError("Require external Python trading API disabled when its state is available.")
    # Compare nonpersonal provenance only. A switch between accounts on the same
    # server/mode cannot be identified without reading personal account fields.
    identity = (server, company, mode)
    if prior is not None and identity != prior:
        raise ExportError("Account/server changed during quote export.")
    provenance = {"server": server, "broker_company": company,
                  "account_mode": "DEMO" if mode == 0 else "EXISTING_LIVE_MARKET_DATA_ONLY",
                  "automated_trading_enabled": False,
                  "external_python_trading_disabled": api_disabled if type(api_disabled) is bool else None,
                  "external_python_trading_state_available": type(api_disabled) is bool}
    return identity, provenance


def finite(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ExportError("Invalid numeric " + label + ".")
    value = float(value)
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ExportError("Invalid numeric " + label + ".")
    return value


def integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        raise ExportError("Invalid integer " + label + ".")
    return int(value)


def current_spec(api, source):
    value = api.call("symbol_info", source)
    if value is None or field(value, "name") != source or field(value, "custom") is not False:
        raise ExportError("Mapped source symbol is missing, mismatched or custom.")
    required = ("point", "trade_contract_size", "trade_tick_size", "volume_min", "volume_max", "volume_step")
    result = {key: finite(field(value, key), key, positive=True) for key in required}
    result["digits"] = integer(field(value, "digits"), "digits")
    if result["digits"] > 16 or result["volume_max"] < result["volume_min"]:
        raise ExportError("Invalid current symbol specification.")
    for key in ("trade_tick_value", "trade_tick_value_profit", "trade_tick_value_loss", "swap_long", "swap_short"):
        v = field(value, key)
        result[key] = None if v is None else finite(v, key, nonnegative=not key.startswith("swap_"))
    for key in ("swap_mode", "swap_rollover3days", "trade_calc_mode", "trade_exemode", "trade_stops_level", "trade_freeze_level"):
        v = field(value, key)
        result[key] = None if v is None else integer(v, key)
    for key in ("currency_base", "currency_profit", "currency_margin"):
        v = field(value, key)
        if v is not None and (not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9]{0,12}", v)):
            raise ExportError("Invalid current specification currency.")
        result[key] = v
    return {"source_symbol": source, "scope": "CURRENT_SNAPSHOT_NOT_HISTORICAL",
            "historical_applicability_verified": False, "values": result}


def error_code(api):
    value = api.call("last_error")
    if not isinstance(value, (tuple, list)) or len(value) < 1 or type(value[0]) is not int:
        raise ExportError("Terminal error status unavailable.")
    # Error descriptions can include paths/identifiers; retain only numeric code.
    return value[0]


def normalize_ticks(value, start, end):
    if value is None:
        return []
    try:
        count = len(value)
    except Exception:
        raise ExportError("Invalid tick-array container.") from None
    if count > MAX_ROWS:
        raise ExportError("Tick-array row limit exceeded.")
    start_ms, end_ms = epoch_ms(start), epoch_ms(end)
    rows = []
    previous = None
    for raw in value:
        names = set(raw) if isinstance(raw, dict) else set(getattr(getattr(raw, "dtype", None), "names", ()) or ())
        if names != set(RAW_FIELDS):
            raise ExportError("Unexpected tick schema; order/deal rows and extra fields are prohibited.")
        get = raw.__getitem__
        row = {key: integer(get(key), key) if key in {"time", "time_msc", "volume", "flags"} else finite(get(key), key, nonnegative=True) for key in RAW_FIELDS}
        if row["bid"] <= 0 or row["ask"] <= 0 or row["ask"] < row["bid"]:
            raise ExportError("Nonpositive or crossed Bid/Ask.")
        if not start_ms <= row["time_msc"] < end_ms or row["time"] != row["time_msc"] // 1000:
            raise ExportError("Tick timestamp is outside the requested half-open UTC window or inconsistent.")
        if previous is not None and row["time_msc"] < previous:
            raise ExportError("Tick order goes backwards; no sorting or filtering is allowed.")
        if row["flags"] > 2**32 - 1:
            raise ExportError("Invalid tick flags.")
        previous = row["time_msc"]
        rows.append(row)
    return rows


def preserve_expected_numeric_ticks(value):
    """Retain bad quote evidence without serializing unexpected/private fields.

    Nonnumeric values in expected numeric fields are redacted, never coerced.
    Their positions remain as empty CSV cells and the redaction count is public.
    Numeric invalid values (zero, crossed sides, NaN, etc.) remain as received.
    """
    if value is None:
        return [], {"reported_received_count": 0, "raw_rows_preserved": 0,
                    "raw_rows_omitted_by_limit": 0, "redacted_or_missing_numeric_fields": 0,
                    "unexpected_or_missing_schema_rows": 0}
    try:
        count = len(value)
    except Exception:
        return [], {"reported_received_count": None, "raw_rows_preserved": 0,
                    "raw_rows_omitted_by_limit": None, "redacted_or_missing_numeric_fields": 0,
                    "unexpected_or_missing_schema_rows": 0}
    saved, redactions, bad_schema = [], 0, 0
    for i, raw in enumerate(value):
        if i >= MAX_ROWS:
            break
        names = set(raw) if isinstance(raw, dict) else set(getattr(getattr(raw, "dtype", None), "names", ()) or ())
        bad_schema += names != set(RAW_FIELDS)
        row = {}
        for key in RAW_FIELDS:
            try:
                v = raw[key] if key in names else None
            except Exception:
                v = None
            if isinstance(v, bool):
                row[key] = v
            elif isinstance(v, numbers.Integral):
                row[key] = int(v)
            elif isinstance(v, numbers.Real):
                row[key] = float(v)
            else:
                row[key] = None
                redactions += 1
        saved.append(row)
    return saved, {"reported_received_count": count, "raw_rows_preserved": len(saved),
                   "raw_rows_omitted_by_limit": max(0, count-len(saved)),
                   "redacted_or_missing_numeric_fields": redactions,
                   "unexpected_or_missing_schema_rows": bad_schema}


@contextmanager
def reserved_output(output: Path):
    lock = output.with_name(output.name + ".lock")
    token = uuid.uuid4().hex
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise ExportError("Another export or stale lock exists; inspect it before retrying.") from None
    pending = None
    published = False
    try:
        if output.exists() or output.is_symlink():
            raise ExportError("Output already exists.")
        # Reserve destination exclusively, so a later rename cannot overwrite it.
        output.mkdir()
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=output))
        yield pending
        for source in sorted(pending.iterdir(), key=lambda p: (p.name == "manifest.json", p.name)):
            destination = output / source.name
            if destination.exists() or destination.is_symlink():
                raise ExportError("Output changed before publication.")
            source.rename(destination)
        pending.rmdir()
        pending = None
        published = True
    finally:
        if not published:
            # Only files in this newly reserved directory belong to this invocation.
            if pending is not None and pending.exists():
                shutil.rmtree(pending)
            try:
                output.rmdir()
            except OSError:
                pass
        try:
            if lock.read_text(encoding="ascii") == token:
                lock.unlink()
        except FileNotFoundError:
            pass


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def export_quotes(request, backend, *, platform_name=None, process_checker=None,
                  safety=None, environment=None, now=None, monotonic=time.monotonic):
    """Injectable backend for offline tests; CLI always uses the actual platform."""
    safety = SAFETY.copy() if safety is None else safety
    environment = os.environ if environment is None else environment
    strict_safety(safety, environment)
    if (sys.platform if platform_name is None else platform_name) != "win32":
        raise ExportError("Windows MT5 terminal required. On macOS use the desktop/MQL5 quote exporter; no connection attempted.")
    now = datetime.now(UTC) if now is None else now
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ExportError("Clock must be UTC-aware.")
    validated = validate_request(request, now)
    helper = Path(__file__).resolve()
    helper_hash = digest(helper)
    running = existing_windows_terminal if process_checker is None else process_checker
    if running(validated["terminal_path"]) is not True:
        raise ExportError("The specified MT5 executable must already be running; this tool does not launch it.")
    api = ReadOnlyCalls(backend)
    initialized = False
    attachment_attempted = False
    results = []
    capture_started = utc_text(datetime.now(UTC))
    with reserved_output(validated["output"]) as directory:
        try:
            # No login/password/server argument. Never call initialize without an
            # explicit path already observed in the process list.
            attachment_attempted = True
            initialized = api.call("initialize", str(validated["terminal_path"]), timeout=validated["timeout_ms"]) is True
            if not initialized:
                raise ExportError("Could not attach to the existing terminal; no login attempted.")
            identity, provenance = protection(api, validated["allow_existing_live_market_data"])
            try:
                copy_all = getattr(backend, "COPY_TICKS_ALL")
            except Exception:
                raise ExportError("COPY_TICKS_ALL constant is unavailable.") from None
            if type(copy_all) is not int:
                raise ExportError("Invalid COPY_TICKS_ALL constant.")
            for canonical, source in validated["symbols"].items():
                protection(api, validated["allow_existing_live_market_data"], identity)
                spec = current_spec(api, source)
                protection(api, validated["allow_existing_live_market_data"], identity)
                for i, (start, end) in enumerate(validated["intervals"], 1):
                    strict_safety(safety, environment)
                    protection(api, validated["allow_existing_live_market_data"], identity)
                    before = monotonic()
                    # Request 1ms before the exclusive end and reject any returned
                    # out-of-window row. Effective server/Python boundary precision
                    # remains unverified, so this never certifies full coverage.
                    raw = api.call("copy_ticks_range", source, start, end - timedelta(milliseconds=1), copy_all)
                    elapsed = monotonic() - before
                    code = error_code(api)
                    protection(api, validated["allow_existing_live_market_data"], identity)
                    strict_safety(safety, environment)
                    preserved, preservation = preserve_expected_numeric_ticks(raw)
                    validation_error = None
                    try:
                        rows = normalize_ticks(raw, start, end)
                    except ExportError as error:
                        rows = preserved
                        validation_error = str(error)
                    except Exception:
                        rows = preserved
                        validation_error = "Invalid tick-array values; backend details withheld."
                    if validation_error is not None:
                        status = "FAILED_PRESERVED"
                    elif raw is None:
                        status = "ERROR_NO_DATA"
                    elif code in TIMEOUT_CODES or elapsed > validated["max_call_seconds"]:
                        status = "PARTIAL_TIMEOUT_NOT_VERIFIED"
                    elif code != 1:
                        status = "PARTIAL_BACKEND_ERROR_NOT_VERIFIED"
                    elif not rows:
                        status = "EMPTY_NOT_VERIFIED"
                    else:
                        status = "QUOTES_RECORDED_COVERAGE_NOT_VERIFIED"
                    stem = f"{canonical}-window-{i}"
                    raw_path = directory / (stem + "-raw.csv")
                    quotes_path = directory / (stem + "-bid-ask.csv")
                    write_csv(raw_path, RAW_FIELDS, preserved)
                    quote_fields = ("canonical_symbol", "source_symbol", "time_msc", "utc", "bid", "ask", "spread_price", "flags")
                    write_csv(quotes_path, quote_fields, (
                        {"canonical_symbol": canonical, "source_symbol": source, "time_msc": row["time_msc"],
                         "utc": utc_text(EPOCH + timedelta(milliseconds=row["time_msc"])),
                         "bid": row["bid"], "ask": row["ask"], "spread_price": row["ask"] - row["bid"], "flags": row["flags"]}
                        for row in (rows if validation_error is None else [])
                    ))
                    intervals = [b["time_msc"] - a["time_msc"] for a, b in zip(rows, rows[1:])] if validation_error is None else []
                    first = rows[0]["time_msc"] if rows and type(rows[0]["time_msc"]) is int else None
                    last = rows[-1]["time_msc"] if rows and type(rows[-1]["time_msc"]) is int else None
                    results.append({"canonical_symbol": canonical, "source_symbol": source,
                                    "requested_start_utc": utc_text(start), "requested_end_utc_exclusive": utc_text(end),
                                    "api_end_utc_inclusive": utc_text(end - timedelta(milliseconds=1)),
                                    "received_count": preservation["reported_received_count"], "received_first_time_msc": first,
                                    "received_last_time_msc": last,
                                    "status": status, "last_error_code": code, "call_elapsed_seconds": elapsed,
                                    "validation_error": validation_error,
                                    "raw_preservation": preservation,
                                    "bid_ask_rows_published": validation_error is None,
                                    "same_millisecond_adjacent_pairs": sum(delta == 0 for delta in intervals),
                                    "tick_flag_counts": dict(sorted(Counter(str(row["flags"]) for row in rows).items())) if validation_error is None else {},
                                    "gap_diagnostics": {"threshold_ms": 5000, "intervals_above_threshold": sum(delta > 5000 for delta in intervals),
                                                        "max_inter_tick_ms": max(intervals, default=None),
                                                        "leading_unobserved_ms": first - epoch_ms(start) if first is not None and validation_error is None else None,
                                                        "trailing_unobserved_ms": epoch_ms(end) - 1 - last if last is not None and validation_error is None else None,
                                                        "available": validation_error is None,
                                                        "meaning": "Observed spacing only; no missing-tick count or continuity certification."},
                                    "current_symbol_specification": spec,
                                    "files": [{"name": path.name, "sha256": digest(path), "bytes": path.stat().st_size} for path in (raw_path, quotes_path)]})
            protection(api, validated["allow_existing_live_market_data"], identity)
        finally:
            if attachment_attempted:
                api.call("shutdown")
        strict_safety(safety, environment)
        if digest(helper) != helper_hash:
            raise ExportError("Exporter source changed during capture.")
        for result in results:
            for saved in result["files"]:
                path = directory / saved["name"]
                if path.is_symlink() or digest(path) != saved["sha256"] or path.stat().st_size != saved["bytes"]:
                    raise ExportError("Quote output changed before publication.")
        ok = bool(results) and all(r["status"] == "QUOTES_RECORDED_COVERAGE_NOT_VERIFIED" for r in results)
        manifest = {"schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION, "tool_sha256": helper_hash,
                    "scope": "MT5_SERVER_QUOTE_EXPORT_ONLY", "capture_started_utc": capture_started,
                    "capture_finished_utc": utc_text(datetime.now(UTC)),
                    "provenance": provenance, "safety": safety.copy(), "symbol_mapping": validated["symbols"],
                    "account_login_name_balance_recorded": False, "login_or_account_creation_attempted": False,
                    "existing_live_market_data_explicitly_allowed": validated["allow_existing_live_market_data"],
                    "coverage_complete_verified": False, "historical_costs_verified": False, "execution_verified": False,
                    "python_server_boundary_precision_verified": False,
                    "same_server_same_mode_account_switch_detection": False,
                    "campaign_state_accessed_or_modified": False, "new_backtests": 0, "orders_sent": 0,
                    "allowed_api_calls": sorted(ALLOWED_CALLS), "api_call_counts": dict(sorted(api.calls.items())),
                    "export_status": ("FAILED_PRESERVED" if any(r["status"] == "FAILED_PRESERVED" for r in results) else
                                      "QUOTES_RECORDED_COVERAGE_NOT_VERIFIED" if ok else "PARTIAL_OR_EMPTY_NOT_VERIFIED"),
                    "native_call_timeout_limit": "Observation only; does not interrupt or cancel a blocking native call.",
                    "raw_tick_policy": "Preserve received row order and multiplicity, including identical time_msc and identical rows.",
                    "results": results}
        with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with (directory / "manifest.sha256").open("x", encoding="ascii") as stream:
            stream.write(digest(directory / "manifest.json") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return manifest, 0 if ok else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terminal-path", required=True)
    parser.add_argument("--output", required=True, help="New output directory; parent must exist.")
    parser.add_argument("--symbol", action="append", nargs=2, metavar=("CANONICAL", "MT5_SOURCE"), required=True)
    parser.add_argument("--window", action="append", nargs=2, metavar=("START_UTC", "END_UTC_EXCLUSIVE"), required=True)
    parser.add_argument("--allow-existing-live-market-data", action="store_true", help="Read quotes only from an already connected live account; trading protections remain mandatory.")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--max-call-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        strict_safety(SAFETY, os.environ)
        if sys.platform != "win32":
            raise ExportError("Windows MT5 terminal required. On macOS use the desktop/MQL5 quote exporter; no connection attempted.")
        if len({s[0] for s in args.symbol}) != len(args.symbol):
            raise ExportError("Duplicate canonical symbol mappings are prohibited.")
        request = {"terminal_path": args.terminal_path, "output": args.output, "symbols": dict(args.symbol),
                   "windows": args.window, "allow_existing_live_market_data": args.allow_existing_live_market_data,
                   "timeout_ms": args.timeout_ms, "max_call_seconds": args.max_call_seconds}
        validate_request(request, datetime.now(UTC))
        backend = importlib.import_module("MetaTrader5")
        _, code = export_quotes(request, backend)
        print("Quote evidence recorded; coverage, historical fees and real fills remain unverified.")
        return code
    except ExportError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        print("Quote export failed; implementation details withheld. No trading action was requested.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
