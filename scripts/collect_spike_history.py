"""Collect and audit authentic Deriv M1 history through the public data channel.

Python 3.12+; pip install -r requirements-research-lock.txt
No credentials, authorization, orders, balance queries, or synthetic data fills.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import certifi
import websockets

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SAFETY = {"LIVE_TRADING": False, "READY_FOR_LIVE": False, "LIVE_ALLOWED": False, "OPENED_TRADES": False}
ALLOWED_REQUESTS = {"active_symbols", "ticks_history"}
FIELDS = ["epoch", "open", "high", "low", "close"]
DOCS = ["https://developers.deriv.com/docs/options/ws-public/",
        "https://developers.deriv.com/docs/data/ticks-history/"]


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def enforce_safety():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.research.spike_hunter import assert_offline

    assert_offline()
    for flag in SAFETY:
        value = os.environ.get(flag, "false").strip().lower()
        if value not in {"false", "0", "no", "off"}:
            raise RuntimeError(f"Research collector requires {flag}=false")


async def connect():
    return await websockets.connect(ENDPOINT, ssl=ssl.create_default_context(cafile=certifi.where()),
                                    open_timeout=20, ping_interval=20, ping_timeout=20, max_size=8_000_000)


async def request(ws, payload):
    if len(ALLOWED_REQUESTS & payload.keys()) != 1:
        raise ValueError("Only public active_symbols and ticks_history requests are permitted")
    if payload.keys() - {"active_symbols", "ticks_history", "count", "end", "style", "granularity", "req_id"}:
        raise ValueError("Unexpected public-data request fields")
    forbidden = {"authorize", "buy", "sell", "proposal", "balance", "new_account_real"} & payload.keys()
    if forbidden:
        raise ValueError("Account or trading requests are prohibited")
    await ws.send(json.dumps(payload))
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=35))
        if message.get("req_id") != payload["req_id"]:
            continue
        if message.get("error"):
            raise RuntimeError(json.dumps(message["error"]))
        return message


def normalize(raw_path, clean_path, raw_manifest):
    with raw_path.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    seen = {}
    excluded = []
    duplicates = 0
    for record in records:
        epoch = int(record["epoch"])
        row = {"epoch": epoch, **{key: float(record[key]) for key in FIELDS[1:]}}
        if epoch % 60:
            excluded.append({"epoch": epoch, "utc": utc(epoch), "reason": "off_utc_minute_grid", "raw": row})
            continue
        if not all(math.isfinite(row[key]) and row[key] > 0 for key in FIELDS[1:]):
            raise ValueError(f"Invalid price at {epoch}")
        if not row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]:
            raise ValueError(f"Invalid OHLC ordering at {epoch}")
        if epoch in seen:
            if seen[epoch] != row:
                raise ValueError(f"Conflicting duplicate at {epoch}")
            duplicates += 1
            continue
        seen[epoch] = row
    epochs = sorted(seen)
    if not epochs:
        raise ValueError("No valid normalized candles")
    gaps = [{"previous_epoch": before, "next_epoch": after,
             "previous_utc": utc(before), "next_utc": utc(after),
             "missing_bars": (after - before) // 60 - 1,
             "missing_epochs": list(range(before + 60, after, 60))}
            for before, after in zip(epochs, epochs[1:]) if after - before != 60]
    write_csv(clean_path, (seen[epoch] for epoch in epochs))
    return {"symbol": raw_manifest["symbol"], "source": "Official Deriv public WebSocket",
            "endpoint": ENDPOINT, "raw_file": str(raw_path), "raw_sha256": sha256(raw_path),
            "raw_rows": len(records), "normalized_file": str(clean_path),
            "normalized_sha256": sha256(clean_path), "normalized_rows": len(seen),
            "equal_duplicates_removed": duplicates, "excluded_rows": excluded, "gaps": gaps,
            "first_epoch": epochs[0], "last_epoch": epochs[-1], "first_utc": utc(epochs[0]),
            "last_utc": utc(epochs[-1]), "requested_start_epoch": raw_manifest["requested_start_epoch"],
            "requested_cutoff_exclusive_epoch": raw_manifest["requested_cutoff_exclusive_epoch"],
            "safety": SAFETY, "authentication_used": False, "fills_or_interpolations": False,
            "normalization_valid": True, "gap_free": not gaps, "finite_positive_ohlc": True,
            "canonical_utc_minute_grid": True, "retrieved_at_utc": raw_manifest["retrieved_at_utc"],
            "request_errors": raw_manifest["request_errors"], "page_count": raw_manifest["page_count"],
            "documentation": DOCS}


async def fetch(symbol, output_dir, days, cutoff, delay, page_size):
    raw_path = output_dir / f"{symbol.lower()}_m1_{days}d.csv"
    clean_path = output_dir / f"{symbol.lower()}_m1_{days}d_clean.csv"
    if raw_path.exists() or clean_path.exists():
        raise FileExistsError(f"Use a new output directory to preserve existing data: {raw_path}")
    start = cutoff - days * 86400
    rows = {}
    pages = []
    errors = []
    duplicate_rows = 0
    request_id = 1
    end = cutoff - 1
    ws = await connect()
    try:
        active = await request(ws, {"active_symbols": "brief", "req_id": request_id})
        matched = [row for row in active.get("active_symbols", [])
                   if (row.get("underlying_symbol") or row.get("symbol")) == symbol]
        if not matched:
            raise RuntimeError(f"Symbol unavailable: {symbol}")
        write_json(output_dir / f"{symbol}_active_symbol.json", matched[0])
        request_id += 1
        while end >= start:
            payload = {"ticks_history": symbol, "count": page_size, "end": end,
                       "style": "candles", "granularity": 60, "req_id": request_id}
            for attempt in range(4):
                try:
                    response = await request(ws, payload)
                    break
                except Exception as exc:
                    errors.append({"end": end, "attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}"})
                    await ws.close()
                    if attempt == 3:
                        raise
                    await asyncio.sleep(1 + attempt)
                    ws = await connect()
            batch = response.get("candles", [])
            if not batch:
                errors.append({"end": end, "error": "No candles returned before requested start"})
                break
            oldest = min(int(row["epoch"]) for row in batch)
            newest = max(int(row["epoch"]) for row in batch)
            if oldest > end:
                raise RuntimeError("Pagination did not move backward")
            for record in batch:
                epoch = int(record["epoch"])
                if start <= epoch < cutoff:
                    row = {"epoch": epoch, **{key: float(record[key]) for key in FIELDS[1:]}}
                    if epoch in rows:
                        if rows[epoch] != row:
                            raise RuntimeError(f"Conflicting OHLC overlap at {epoch}")
                        duplicate_rows += 1
                    rows[epoch] = row
            pages.append({"request": payload, "returned_rows": len(batch), "oldest_epoch": oldest,
                          "newest_epoch": newest,
                          "response_sha256": hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()})
            if len(pages) == 1 or len(pages) % 20 == 0:
                print(f"{symbol}: {len(pages)} pages, {len(rows)} rows, oldest {utc(oldest)}", flush=True)
            if len(pages) % 50 == 0:
                write_csv(output_dir / f"{symbol.lower()}_m1_checkpoint.csv", (rows[epoch] for epoch in sorted(rows)))
            end = oldest - 1
            request_id += 1
            await asyncio.sleep(delay)
    finally:
        await ws.close()
    if not rows:
        raise RuntimeError(f"No historical data returned for {symbol}")
    data = [rows[epoch] for epoch in sorted(rows)]
    write_csv(raw_path, data)
    manifest = {"symbol": symbol, "api_symbol": symbol, "source": "Official Deriv public WebSocket ticks_history",
                "endpoint": ENDPOINT, "authentication_used": False, "safety": SAFETY,
                "retrieved_at_utc": utc(time.time()), "granularity_seconds": 60, "requested_days": days,
                "requested_start_epoch": start, "requested_cutoff_exclusive_epoch": cutoff,
                "requested_start_utc": utc(start), "requested_cutoff_exclusive_utc": utc(cutoff),
                "rows": len(data), "first_epoch": data[0]["epoch"], "last_epoch": data[-1]["epoch"],
                "first_utc": utc(data[0]["epoch"]), "last_utc": utc(data[-1]["epoch"]),
                "file": str(raw_path), "sha256": sha256(raw_path), "page_count": len(pages),
                "equal_pagination_duplicates_removed": duplicate_rows, "request_errors": errors,
                "expected_grid_rows": days * 1440, "documentation": DOCS}
    write_json(output_dir / f"{symbol}_manifest.json", manifest)
    write_json(output_dir / f"{symbol}_page_audit.json", pages)
    normalized = normalize(raw_path, clean_path, manifest)
    write_json(output_dir / f"{symbol}_normalization_manifest.json", normalized)
    print(f"{symbol}: DONE {normalized['normalized_rows']} valid M1 rows, {len(normalized['gaps'])} gaps", flush=True)
    return normalized


async def run(args):
    enforce_safety()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_closed_cutoff = int(time.time()) // 60 * 60
    cutoff = latest_closed_cutoff if args.cutoff_epoch is None else args.cutoff_epoch
    if cutoff % 60 or cutoff > latest_closed_cutoff:
        raise ValueError("cutoff-epoch must be a closed UTC minute boundary in the past")
    config = {"endpoint": ENDPOINT, "days": args.days, "symbols": args.symbols,
              "requested_start_epoch": cutoff - args.days * 86400,
              "cutoff_exclusive_epoch": cutoff, "safety": SAFETY}
    write_json(output_dir / "acquisition_config.json", config)
    results = await asyncio.gather(*(fetch(symbol, output_dir, args.days, cutoff, args.delay, args.page_size)
                                     for symbol in args.symbols), return_exceptions=True)
    summary = {"config": config, "successful_symbols": [], "errors": []}
    for symbol, result in zip(args.symbols, results):
        if isinstance(result, Exception):
            summary["errors"].append({"symbol": symbol, "error": f"{type(result).__name__}: {result}"})
        else:
            summary["successful_symbols"].append({"symbol": symbol, "normalized_rows": result["normalized_rows"],
                                                  "normalized_sha256": result["normalized_sha256"]})
    write_json(output_dir / "acquisition_summary.json", summary)
    if summary["errors"]:
        raise RuntimeError(json.dumps(summary["errors"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--symbols", nargs="+", default=["BOOM500", "CRASH500"])
    parser.add_argument("--cutoff-epoch", type=int, help="Exclusive UTC minute boundary; freezes reproducible window")
    parser.add_argument("--delay", type=float, default=.18, help="Seconds between requests per symbol")
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()
    if args.days < 1 or args.delay < .08 or not 1 <= args.page_size <= 5000:
        parser.error("days >= 1, delay >= 0.08, and page-size in 1..5000 are required")
    if len(set(args.symbols)) != len(args.symbols):
        parser.error("symbols must be unique")
    if not all(symbol.isalnum() and len(symbol) <= 30 for symbol in args.symbols):
        parser.error("symbols must be alphanumeric public symbol identifiers")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
