"""Authentic, closed Deriv M1 data and strict independent UTC frame aggregation.

Only the unauthenticated public endpoint is reachable. Wire responses, canonical
snapshots and manifests are immutable; ``latest.json`` is an atomic cache pointer.
No account/order requests, interpolations, substituted instruments or CFD costs.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import re
import ssl
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import pandas as pd
import websockets

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
COLLECTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SAFETY = {k: False for k in ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")}
TARGETS = tuple(f"{family}{number}" for family in ("BOOM", "CRASH")
                for number in (300, 500, 600, 900, 1000))
FRAMES = {"M1": 60, "M2": 120, "M3": 180, "M5": 300, "M10": 600,
          "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
FIELDS = ["epoch", "open", "high", "low", "close"]
DOCS = ["https://developers.deriv.com/docs/options/ws-public/",
        "https://developers.deriv.com/docs/data/active-symbols/",
        "https://developers.deriv.com/docs/data/ticks-history/",
        "https://developers.deriv.com/docs/data/contracts-for/"]
_REQUEST_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def _throttle_public_request() -> None:
    # A shared per-event-loop limiter prevents each concurrent symbol believing
    # it has the whole public quota. API RateLimit responses still back off.
    loop = asyncio.get_running_loop()
    state = _REQUEST_LIMITS.setdefault(loop, {"lock": asyncio.Lock(), "next": 0.0})
    async with state["lock"]:
        delay = state["next"] - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        state["next"] = loop.time() + .4


def _utc(epoch: float | int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enforce_safety() -> None:
    if any(value is not False for value in SAFETY.values()):
        raise RuntimeError("Every collector safety flag must remain false")
    for key in (*SAFETY, "ready_for_live", "live_allowed", "opened_trades"):
        if os.environ.get(key, "false").strip().lower() not in {"false", "0", "no", "off"}:
            raise RuntimeError(f"Research requires {key}=false")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Immutable artifact differs: {path}")
        return
    # Exclusive creation prevents concurrent campaigns silently replacing evidence.
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise RuntimeError(f"Concurrent immutable artifact differs: {path}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def normalize(rows: list[dict], start_epoch: int, cutoff_epoch: int) -> tuple[pd.DataFrame, dict]:
    """Validate raw OHLC; conflicting duplicates or malformed prices fail closed.

    Off-grid/open/out-of-window rows are excluded and counted; missing minutes are
    represented as ranges in the audit, never fabricated rows.
    """
    if start_epoch % 60 or cutoff_epoch % 60 or start_epoch >= cutoff_epoch:
        raise ValueError("M1 interval must be aligned and nonempty")
    seen: dict[int, dict] = {}
    excluded: dict[str, int] = {}
    equal_duplicates = 0
    for raw in rows:
        try:
            numeric_epoch = float(raw["epoch"])
            if not math.isfinite(numeric_epoch) or numeric_epoch != int(numeric_epoch):
                raise ValueError("nonintegral epoch")
            epoch = int(numeric_epoch)
            row = {"epoch": epoch, **{key: float(raw[key]) for key in FIELDS[1:]}}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Malformed candle: {raw!r}") from exc
        if not all(math.isfinite(row[k]) and row[k] > 0 for k in FIELDS[1:]):
            raise ValueError(f"Invalid finite positive OHLC at {epoch}")
        if not row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]:
            raise ValueError(f"Invalid OHLC ordering at {epoch}")
        reason = ("off_grid" if epoch % 60 else "before_start" if epoch < start_epoch
                  else "not_closed_before_cutoff" if epoch + 60 > cutoff_epoch else None)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        if epoch in seen:
            if seen[epoch] != row:
                raise ValueError(f"Conflicting duplicate at {epoch}")
            equal_duplicates += 1
        seen[epoch] = row
    epochs = sorted(seen)
    frame = pd.DataFrame([seen[e] for e in epochs], columns=FIELDS)
    frame = frame.astype({"epoch": "int64", **{key: "float64" for key in FIELDS[1:]}})
    missing = []
    boundaries = [start_epoch - 60, *epochs, cutoff_epoch]
    for before, after in zip(boundaries, boundaries[1:]):
        if after - before > 60:
            missing.append({"start_epoch": before + 60, "end_exclusive_epoch": after,
                            "missing_minutes": (after - before) // 60 - 1})
    audit = {"raw_rows": len(rows), "normalized_rows": len(frame),
             "equal_duplicates_removed": equal_duplicates, "excluded_counts": excluded,
             "expected_minutes": (cutoff_epoch - start_epoch) // 60,
             "missing_minutes": sum(r["missing_minutes"] for r in missing),
             "missing_ranges": missing, "gap_free": not missing,
             "first_epoch": epochs[0] if epochs else None, "last_epoch": epochs[-1] if epochs else None,
             "reliable_start_epoch": epochs[0] if epochs else None,
             "reliable_cutoff_exclusive_epoch": epochs[-1] + 60 if epochs else None,
             "internal_missing_minutes": sum((after - before) // 60 - 1
                                             for before, after in zip(epochs, epochs[1:])),
             "missing_prefix_minutes": (epochs[0] - start_epoch) // 60 if epochs else (cutoff_epoch - start_epoch) // 60,
             "missing_suffix_minutes": (cutoff_epoch - epochs[-1] - 60) // 60 if epochs else 0,
             "start_epoch": start_epoch, "cutoff_exclusive_epoch": cutoff_epoch,
             "fills_or_interpolations": False, "finite_positive_ohlc": True}
    return frame, audit


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Vector validation avoids rebuilding millions of Python dicts per frame."""
    clean = frame[FIELDS].copy()
    if clean.empty:
        return clean.astype({"epoch": "int64", **{k: "float64" for k in FIELDS[1:]}})
    epochs = clean.epoch.to_numpy(dtype="float64")
    if not np.isfinite(epochs).all() or np.any(epochs != np.floor(epochs)) or np.any(epochs % 60):
        raise ValueError("Input requires canonical integral UTC M1 epochs")
    prices = clean[FIELDS[1:]].to_numpy(dtype="float64")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid finite positive OHLC")
    opening, high, low, close = prices.T
    if np.any(low > np.minimum(opening, close)) or np.any(high < np.maximum(opening, close)):
        raise ValueError("Invalid OHLC ordering")
    clean = clean.astype({"epoch": "int64", **{k: "float64" for k in FIELDS[1:]}})
    if clean.epoch.duplicated().any():
        raise ValueError("Input requires canonical unique M1 rows")
    return clean.sort_values("epoch", kind="stable").reset_index(drop=True)


def load_m1(path: str | Path) -> pd.DataFrame:
    return _canonical_frame(pd.read_csv(path, usecols=FIELDS, float_precision="round_trip"))


def resample(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Return exact closed independent bars; reject partial groups and input errors.

    ``epoch`` denotes the bar open; ``close_epoch`` the information availability.
    No forward fill. Dropped incomplete groups and internal discontinuities are
    recorded in attrs so callers can report unavailable frames explicitly.
    """
    if timeframe not in FRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    seconds = FRAMES[timeframe]
    if frame.empty:
        result = pd.DataFrame(columns=FIELDS + ["close_epoch"])
        result.attrs.update(timeframe=timeframe, seconds=seconds, dropped_incomplete_bars=0,
                            complete_bars=0, gaps_preserved=True)
        return result
    clean = _canonical_frame(frame)
    missing_minutes = int(np.maximum(0, np.diff(clean.epoch.to_numpy()) // 60 - 1).sum())
    clean["bucket"] = clean.epoch // seconds * seconds
    grouped = clean.groupby("bucket", sort=True)
    agg = grouped.agg(epoch=("epoch", "first"), last=("epoch", "last"),
                      count=("epoch", "size"), open=("open", "first"),
                      high=("high", "max"), low=("low", "min"), close=("close", "last"))
    valid = ((agg["count"] == seconds // 60) & (agg["epoch"] == agg.index)
             & (agg["last"] == agg.index + seconds - 60))
    result = agg.loc[valid, FIELDS].reset_index(drop=True)
    result["close_epoch"] = result.epoch + seconds
    result.attrs.update(timeframe=timeframe, seconds=seconds,
                        dropped_incomplete_bars=int((~valid).sum()), complete_bars=len(result),
                        gaps_preserved=True, input_missing_minutes=missing_minutes)
    return result


def _validate_request(payload: dict) -> None:
    names = {"active_symbols", "ticks_history", "contracts_for"} & payload.keys()
    allowed = {"active_symbols", "ticks_history", "contracts_for", "count", "start", "end",
               "style", "granularity", "req_id"}
    if len(names) != 1 or payload.keys() - allowed:
        raise ValueError("Only allowlisted unauthenticated market-data requests are permitted")
    if "ticks_history" in payload and (payload.get("style") != "candles" or payload.get("granularity") != 60):
        raise ValueError("Campaign collector only requests real M1 candles")


class PublicDataClient:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.ws = None
        self.sequence = 0
        self.errors: list[dict] = []

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def request(self, payload: dict) -> tuple[dict, dict]:
        enforce_safety()
        self.sequence += 1
        payload = dict(payload, req_id=self.sequence)
        _validate_request(payload)
        last_error = None
        for attempt in range(4):
            try:
                if self.ws is None:
                    self.ws = await websockets.connect(ENDPOINT, open_timeout=20,
                        ssl=ssl.create_default_context(cafile=certifi.where()),
                        ping_interval=20, ping_timeout=20, max_size=12_000_000)
                await _throttle_public_request()
                await self.ws.send(json.dumps(payload))
                while True:
                    wire = await asyncio.wait_for(self.ws.recv(), 35)
                    response = json.loads(wire)
                    if response.get("req_id") == self.sequence:
                        break
                raw = wire.encode() if isinstance(wire, str) else wire
                digest = hashlib.sha256(raw).hexdigest()
                wire_path = self.output_dir / "wire" / f"{digest}.json.gz"
                _immutable(wire_path, gzip.compress(raw, mtime=0))
                provenance = {"request": payload, "endpoint": ENDPOINT,
                              "collector_sha256": COLLECTOR_SHA256,
                              "received_at_utc": _utc(time.time()), "wire_sha256": digest,
                              "wire_file": str(wire_path.resolve()), "file_sha256": sha256(wire_path),
                              "safety": SAFETY, "authentication_used": False}
                if response.get("error"):
                    error_path = self.output_dir / "errors" / f"{uuid.uuid4().hex}.json"
                    _immutable(error_path, _json_bytes({**provenance, "error": response["error"]}))
                    self.errors.append({"error": response["error"], "attempt": attempt + 1,
                                        "receipt_file": str(error_path.resolve()),
                                        "receipt_sha256": sha256(error_path)})
                    if response["error"].get("code") == "RateLimit" and attempt < 3:
                        await asyncio.sleep(15 * (attempt + 1))
                        continue
                    raise RuntimeError(json.dumps(response["error"], sort_keys=True))
                return response, provenance
            except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
                last_error = exc
                self.errors.append({"error": f"{type(exc).__name__}: {exc}", "attempt": attempt + 1,
                                    "request": payload, "no_wire_response_received": True})
                await self.close()
                if attempt == 3:
                    raise
                await asyncio.sleep(attempt + 1)
        raise RuntimeError(str(last_error))


def resolve_catalogue(catalogue: list[dict]) -> dict[str, dict]:
    resolved = {}
    for target in TARGETS:
        family, number = re.fullmatch(r"(BOOM|CRASH)(\d+)", target).groups()
        expected_name = f"{family.title()} {number} Index".casefold()
        matches = [row for row in catalogue
                   if str(row.get("underlying_symbol_name", row.get("display_name", ""))).casefold() == expected_name]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous real symbol name: {target}")
        if not matches:
            resolved[target] = {"symbol": target, "status": "UNAVAILABLE", "api_symbol": None}
            continue
        row = matches[0]
        api_symbol = row.get("underlying_symbol") or row.get("symbol")
        if not isinstance(api_symbol, str) or not api_symbol.isalnum():
            raise ValueError(f"Invalid public symbol identifier: {target}")
        resolved[target] = {"symbol": target, "status": "AVAILABLE", "api_symbol": api_symbol,
                            "display_name": row.get("underlying_symbol_name", row.get("display_name")),
                            "catalogue_entry": row}
    return resolved


async def discover_symbols(output_dir: str | Path | None = None) -> dict:
    """Verify all ten names from a fresh official catalogue, retaining wire bytes."""
    enforce_safety()
    directory = Path(output_dir or "data/research_campaign").resolve()
    client = PublicDataClient(directory)
    try:
        response, wire = await client.request({"active_symbols": "brief"})
    finally:
        await client.close()
    resolved = resolve_catalogue(response.get("active_symbols", []))
    value = {"targets": resolved, "catalogue": wire, "documentation": DOCS, "safety": SAFETY}
    # Identical catalogue wire bytes may recur at a later receipt time. Address
    # the complete receipt, not only its wire body, to preserve both observations.
    content = _json_bytes(value)
    digest = hashlib.sha256(content).hexdigest()
    _immutable(directory / "catalogues" / f"{digest}.json", content)
    return value


def _verified_page(directory: Path, payload: dict) -> tuple[dict, dict] | None:
    key = hashlib.sha256(_json_bytes(payload)).hexdigest()
    pointer = directory / "page_cache" / f"{key}.json"
    if not pointer.exists():
        return None
    provenance = json.loads(pointer.read_text())
    path = Path(provenance["wire_file"])
    if sha256(path) != provenance["file_sha256"]:
        raise ValueError(f"Cached wire hash mismatch: {path}")
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != provenance["wire_sha256"]:
        raise ValueError(f"Cached decompressed wire hash mismatch: {path}")
    response = json.loads(raw)
    if {k: v for k, v in provenance["request"].items() if k != "req_id"} != payload:
        raise ValueError("Cached request identity mismatch")
    return response, {**provenance, "cache_hit": True}


def _source_history(prior: dict | None) -> tuple[dict[str, int], int, list[dict]]:
    """Reconstruct cumulative source exclusions from pinned immutable ancestry.

    A cache-only update reads already canonical CSV rows, so its own normalization
    has no raw exclusions. That must not erase the original wire observations.
    Older manifests can have reset summary fields; use each node's incremental
    ``cache_quality`` and deduplicate provider-boundary receipts across versions.
    """
    nodes, seen = [], set()
    node = prior
    while node:
        nodes.append(node)
        parent_path = node.get("parent_manifest")
        if not parent_path:
            break
        path = Path(parent_path).resolve()
        if path in seen:
            raise ValueError("Cached source manifest ancestry cycle")
        seen.add(path)
        if sha256(path) != node.get("parent_manifest_sha256"):
            raise ValueError(f"Cached source ancestor manifest hash mismatch: {path}")
        node = json.loads(path.read_text())
    excluded: dict[str, int] = {}
    duplicates = 0
    boundaries = {}
    for item in reversed(nodes):
        quality = item.get("cache_quality", {})
        for reason, count in quality.get("excluded_counts", {}).items():
            excluded[reason] = excluded.get(reason, 0) + int(count)
        duplicates += int(quality.get("equal_duplicates_removed", 0))
        for boundary in item.get("provider_history_boundaries", []):
            key = hashlib.sha256(_json_bytes(boundary)).hexdigest()
            boundaries[key] = boundary
    return excluded, duplicates, list(boundaries.values())


async def _fetch_symbol(directory: Path, target: dict, start: int, cutoff: int,
                        page_size: int, delay: float) -> dict:
    symbol = target["symbol"]
    client = PublicDataClient(directory)
    pointer = directory / symbol / "latest.json"
    rows, prior, pages = [], None, []
    if pointer.exists():
        prior = json.loads(pointer.read_text())
        if sha256(prior["manifest_file"]) != prior["manifest_sha256"]:
            raise ValueError(f"Cached manifest hash mismatch: {symbol}")
        original = json.loads(Path(prior["manifest_file"]).read_text())
        if original != {k: v for k, v in prior.items() if k not in {"manifest_file", "manifest_sha256"}}:
            raise ValueError(f"Cached pointer metadata differs from immutable manifest: {symbol}")
        if prior["api_symbol"] != target["api_symbol"]:
            raise ValueError("Catalogue symbol changed; explicit cache migration required")
        if sha256(prior["file"]) != prior["sha256"]:
            raise ValueError(f"Cached normalized file hash mismatch: {symbol}")
        rows = load_m1(prior["file"]).to_dict("records")
    # Keep all previously observed authentic rows; latest requested interval is
    # separately sliced and audited. A longer first collection is never discarded.
    ranges = [(start, cutoff)]
    if prior:
        old_start = prior["cache_quality"]["start_epoch"]
        old_cutoff = prior["cache_quality"]["cutoff_exclusive_epoch"]
        ranges = []
        if start < old_start:
            ranges.append((start, min(old_start, cutoff)))
        if cutoff > old_cutoff:
            ranges.append((max(start, old_cutoff), cutoff))
        # Historical provider gaps stay explicit; they do not trigger repeated
        # identical network calls every 15 minutes or become fabricated prices.
    contract = None
    boundaries = []
    try:
        if prior and prior.get("contract_metadata"):
            contract = prior["contract_metadata"]
        else:
            metadata, meta_wire = await client.request({"contracts_for": target["api_symbol"]})
            contract = {"available": metadata.get("contracts_for", {}).get("available", []),
                        "provenance": meta_wire, "costs_verified": False,
                        "cfd_mapping_verified": False, "historical_spread_slippage_commission": None,
                        "limitation": "Options contract catalogue is not a CFD execution/cost specification"}
        for range_start, range_cutoff in ranges:
            end = range_cutoff - 1
            while end >= range_start:
                payload = {"ticks_history": target["api_symbol"], "style": "candles", "granularity": 60,
                           "count": page_size, "start": range_start, "end": end}
                cached = _verified_page(directory, payload)
                response, wire = cached if cached else await client.request(payload)
                if not cached:
                    key = hashlib.sha256(_json_bytes(payload)).hexdigest()
                    _immutable(directory / "page_cache" / f"{key}.json", _json_bytes(wire))
                batch = response.get("candles", [])
                audit_page = {**wire, "returned_rows": len(batch)}
                pages.append(audit_page)
                if not batch:
                    audit_page["provider_empty_before_requested_start"] = True
                    break
                oldest = min(int(row["epoch"]) for row in batch)
                newest = max(int(row["epoch"]) for row in batch)
                audit_page.update(oldest_epoch=oldest, newest_epoch=newest)
                if oldest > end or newest > end:
                    # The public endpoint can silently return current prices
                    # when asked beyond retained history. Preserve the wire but
                    # never ingest those off-window candles as historical data.
                    boundary = {"reason": "provider_returned_candles_newer_than_requested_end",
                                "requested_start": range_start, "requested_end": end,
                                "oldest_returned_epoch": oldest, "newest_returned_epoch": newest,
                                "wire_sha256": wire["wire_sha256"]}
                    boundaries.append(boundary)
                    audit_page["excluded_provider_boundary_response"] = True
                    break
                rows.extend(batch)
                end = oldest - 1
                if not cached:
                    await asyncio.sleep(delay)
    finally:
        await client.close()
    # Preserve complete previous data while providing a reproducible current slice.
    all_start = min(start, min((int(row["epoch"]) // 60 * 60 for row in rows), default=start))
    all_cutoff = max(cutoff, prior["cutoff_exclusive_epoch"] if prior else cutoff)
    clean, cache_quality = normalize(rows, all_start, all_cutoff)
    requested, quality = normalize(clean.to_dict("records"), start, cutoff)
    source_excluded, source_duplicates, source_boundaries = _source_history(prior)
    for reason, count in cache_quality["excluded_counts"].items():
        source_excluded[reason] = source_excluded.get(reason, 0) + count
    source_duplicates += cache_quality["equal_duplicates_removed"]
    boundary_index = {hashlib.sha256(_json_bytes(item)).hexdigest(): item
                      for item in [*source_boundaries, *boundaries]}
    source_boundaries = list(boundary_index.values())
    quality["raw_source_excluded_counts"] = source_excluded
    quality["raw_source_equal_duplicates_removed"] = source_duplicates
    csv_content = clean.to_csv(index=False, lineterminator="\n").encode()
    cache_sha = hashlib.sha256(csv_content).hexdigest()
    cache_path = directory / symbol / "snapshots" / f"{cache_sha}.csv"
    _immutable(cache_path, csv_content)
    slice_content = requested.to_csv(index=False, lineterminator="\n").encode()
    slice_sha = hashlib.sha256(slice_content).hexdigest()
    slice_path = directory / symbol / "snapshots" / f"{slice_sha}.csv"
    _immutable(slice_path, slice_content)
    result = {"symbol": symbol, "api_symbol": target["api_symbol"], "display_name": target["display_name"],
              "status": "READY" if len(requested) else "INSUFFICIENT_DATA",
              "file": str(cache_path), "sha256": cache_sha, "rows": len(clean),
              "requested_file": str(slice_path), "requested_sha256": slice_sha, "requested_rows": len(requested),
              "requested_start_epoch": start, "cutoff_exclusive_epoch": cutoff,
              "requested_start_utc": _utc(start), "cutoff_exclusive_utc": _utc(cutoff),
              "quality": quality, "cache_quality": cache_quality,
              "pages": pages, "new_network_pages": sum(not p.get("cache_hit", False) for p in pages),
              "provider_history_boundaries": source_boundaries,
              "current_attempt_provider_boundaries": boundaries,
              "provider_limited": bool(source_boundaries),
              "source_history_excluded_counts": source_excluded,
              "source_history_equal_duplicates_removed": source_duplicates,
              "request_errors": client.errors,
              "collector_sha256": COLLECTOR_SHA256,
              "parent_manifest": prior.get("manifest_file") if prior else None,
              "parent_manifest_sha256": prior.get("manifest_sha256") if prior else None,
              "contract_metadata": contract, "safety": SAFETY, "documentation": DOCS,
              "retrieved_at_utc": _utc(time.time()), "authentication_used": False}
    manifest_path = directory / symbol / "manifests" / f"{cutoff}_{uuid.uuid4().hex}.json"
    _immutable(manifest_path, _json_bytes(result))
    result.update(manifest_file=str(manifest_path), manifest_sha256=sha256(manifest_path))
    _atomic_json(pointer, result)
    return result


async def update_history(output_dir: str | Path, days: int = 365, cutoff_epoch: int | None = None,
                         symbols: list[str] | None = None, page_size: int = 5000,
                         concurrency: int = 2, delay: float = .12) -> dict:
    """Fetch greatest requested real interval; unavailable/provider-limited data explicit.

    ``symbols`` uses canonical TARGETS names. Each result's ``requested_file`` is
    the exact requested closed M1 window; ``file`` additionally preserves older
    cache rows. Network failures remain ERROR, never synthetic replacements.
    """
    enforce_safety()
    if not isinstance(days, int) or days < 1 or not 1 <= page_size <= 5000 or concurrency < 1 or delay < .08:
        raise ValueError("days>=1, page_size 1..5000, concurrency>=1, delay>=.08 required")
    latest = int(time.time()) // 60 * 60
    cutoff = latest if cutoff_epoch is None else cutoff_epoch
    if not isinstance(cutoff, int) or cutoff % 60 or cutoff > latest:
        raise ValueError("cutoff must be a past or current closed UTC minute boundary")
    selected = list(TARGETS) if symbols is None else symbols
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(TARGETS):
        raise ValueError("Only unique independently verified ten Boom/Crash targets are allowed")
    directory = Path(output_dir).resolve()
    discovery = await discover_symbols(directory)
    semaphore = asyncio.Semaphore(concurrency)
    async def one(symbol):
        target = discovery["targets"][symbol]
        if target["status"] != "AVAILABLE":
            return target
        async with semaphore:
            try:
                return await _fetch_symbol(directory, target, cutoff - days * 86400, cutoff, page_size, delay)
            except Exception as exc:
                return {**target, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "safety": SAFETY}
    results = await asyncio.gather(*(one(symbol) for symbol in selected))
    summary = {"source": "Official Deriv public M1 candles", "endpoint": ENDPOINT,
               "requested_days": days, "cutoff_exclusive_epoch": cutoff,
               "start_epoch": cutoff - days * 86400, "symbols": {r["symbol"]: r for r in results},
               "catalogue": discovery["catalogue"], "safety": SAFETY,
               "completed_at_utc": _utc(time.time()), "documentation": DOCS,
               "collector_sha256": COLLECTOR_SHA256,
               "cost_model_verified": False, "real_orders_sent": 0}
    path = directory / "runs" / f"{cutoff}_{uuid.uuid4().hex}.json"
    _immutable(path, _json_bytes(summary))
    summary.update(manifest_file=str(path), manifest_sha256=sha256(path))
    _atomic_json(directory / "latest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/research_campaign"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cutoff-epoch", type=int)
    parser.add_argument("--symbols", nargs="+", choices=TARGETS)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    result = asyncio.run(update_history(args.output_dir, args.days, args.cutoff_epoch,
                                        args.symbols, concurrency=args.concurrency))
    print(json.dumps({"manifest_file": result["manifest_file"],
                      "symbols": {k: {"status": v["status"], "rows": v.get("requested_rows", 0),
                                       "missing": v.get("quality", {}).get("missing_minutes"),
                                       "error": v.get("error")} for k, v in result["symbols"].items()}}, indent=2))


if __name__ == "__main__":
    main()
