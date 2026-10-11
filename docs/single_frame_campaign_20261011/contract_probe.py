"""Bounded public-only CFD-page and four historical-tick availability probes.

This does not estimate historical costs or replay any strategy. No proposal,
account, authentication, subscription, order or execution API is reachable here.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
from urllib.request import Request, urlopen, build_opener, HTTPSHandler, HTTPRedirectHandler

import certifi
import websockets

from research_campaign.source_audit import audit_catalogue
from research_campaign.state import guard

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
SPEC_URL = "https://deriv.com/trading-specifications"
WINDOWS = (1761955200, 1785542400)  # 2025-11-01 / 2026-08-01 00:00:00 UTC.
TARGETS = ("BOOM500", "CRASH500")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def save_json(path, value):
    save(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def validate_request(request, symbols):
    if set(request) != {"ticks_history", "style", "count", "start", "end", "req_id"}:
        raise ValueError("Only the exact public historical-tick request is permitted")
    if request["ticks_history"] not in symbols or request["style"] != "ticks" or request["count"] != 4:
        raise ValueError("Only declared four-second actual-symbol probes are permitted")
    if any(type(request[k]) is not int for k in ("start", "end", "req_id", "count")):
        raise ValueError("Exact integer request parameters required")
    if request["start"] not in WINDOWS or request["end"] != request["start"] + 3:
        raise ValueError("Probe windows are fixed before measurement")


def extract_specs(html):
    """Decode embedded JSON strings, never execute website JavaScript."""
    target_names = {f"{family} {n} Index" for family in ("Boom", "Crash")
                    for n in (300, 500, 600, 900, 1000)}
    decoder, rows = json.JSONDecoder(), {}
    for body in re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S):
        body = body.strip().rstrip(";")
        prefix = "self.__next_f.push("
        if not body.startswith(prefix) or not body.endswith(")"):
            continue
        flight = json.loads(body[len(prefix):-1])
        if len(flight) < 2 or not isinstance(flight[1], str):
            continue
        for match in re.finditer(r'\{"name":"((?:Boom|Crash) (?:300|500|600|900|1000) Index)"', flight[1]):
            row, _ = decoder.raw_decode(flight[1][match.start():])
            if row["name"] in rows and rows[row["name"]] != row:
                raise ValueError("Conflicting public specification rows")
            rows[row["name"]] = row
    if set(rows) != target_names:
        raise ValueError("All ten target specification rows must be independently present")
    required = {"name", "contract-size", "base-currency", "min-size", "min-spread",
                "swap-long-pts", "swap-short-pts", "trading-hours-gmt"}
    if any(not required.issubset(row) for row in rows.values()):
        raise ValueError("Missing public specification fields")
    return [rows[name] for name in sorted(rows)]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirects are not allowed by this fixed-source probe")


def capture_specs(directory):
    context = ssl.create_default_context(cafile=certifi.where())
    opener = build_opener(HTTPSHandler(context=context), NoRedirect())
    started = utc()
    with opener.open(Request(SPEC_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
        raw = response.read()
        receipt = {"url": SPEC_URL, "resolved_url": response.url, "status": response.status,
                   "requested_at_utc": started, "received_at_utc": utc(),
                   "http_date": response.headers.get("Date"), "sha256": sha(raw),
                   "authentication_used": False, "historical_costs_verified": False,
                   "safety": guard()}
    save(directory / "cfd_specifications_response.html", raw)
    save_json(directory / "cfd_specifications_receipt.json", receipt)
    rows = extract_specs(raw.decode())
    save_json(directory / "current_cfd_specification_rows.json", {
        "source_response_sha256": receipt["sha256"], "retrieved_at_utc": receipt["received_at_utc"],
        "rows": rows, "historical_applicability": "NOT TESTED", "account_applicability": "NOT TESTED",
        "point_value_and_live_execution_verified": False, "safety": guard()})
    return receipt


def summarize_ticks(response, request):
    if response.get("req_id") != request["req_id"] or response.get("echo_req") != request:
        raise ValueError("Original tick response/request identity mismatch")
    if response.get("error"):
        return {"status": "API_ERROR", "error": response["error"], "response_keys": sorted(response)}
    history = response.get("history", {})
    epochs, prices = history.get("times", []), history.get("prices", [])
    if len(epochs) != len(prices) or any(type(epoch) is not int for epoch in epochs):
        raise ValueError("Invalid original tick arrays")
    values = [float(value) for value in prices]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Invalid original tick price")
    expected = list(range(request["start"], request["end"] + 1))
    return {"status": "EXACT_WINDOW_AVAILABLE" if epochs == expected else "PARTIAL_OR_OFF_WINDOW",
            "returned_ticks": len(epochs), "first_epoch": epochs[0] if epochs else None,
            "last_epoch": epochs[-1] if epochs else None,
            "missing_requested_seconds": sorted(set(expected) - set(epochs)),
            "outside_requested_seconds": sorted(set(epochs) - set(expected)),
            "duplicate_seconds": len(epochs) - len(set(epochs)),
            "response_keys": sorted(response), "history_keys": sorted(history),
            "historical_bid_ask_present": {"bid", "ask"}.issubset(history),
            "full_history_complete": False, "strategy_execution_replayed": False,
            "actual_cfd_fill_verified": False}


async def probe(directory, symbols):
    output = []
    context = ssl.create_default_context(cafile=certifi.where())
    async with websockets.connect(ENDPOINT, ssl=context, open_timeout=20, ping_interval=20,
                                  ping_timeout=20, max_size=1000000) as connection:
        for number, (symbol, start) in enumerate(((symbol, start) for symbol in TARGETS for start in WINDOWS), 1):
            request = {"ticks_history": symbols[symbol], "style": "ticks", "count": 4,
                       "start": start, "end": start + 3, "req_id": number}
            validate_request(request, set(symbols[target] for target in TARGETS))
            save_json(directory / f"tick_{number}_intent.json", {"request": request, "endpoint": ENDPOINT,
                "issued_before_send_at_utc": utc(), "canonical_symbol": symbol, "safety": guard(),
                "authentication_used": False})
            await connection.send(json.dumps(request))
            save_json(directory / f"tick_{number}_sent.json", {"sent_at_utc": utc(), "request": request,
                                                               "safety": guard()})
            wire = await asyncio.wait_for(connection.recv(), timeout=35)
            raw = wire.encode() if isinstance(wire, str) else wire
            save(directory / f"tick_{number}_response.wire", raw)
            receipt = {"request": request, "canonical_symbol": symbol, "endpoint": ENDPOINT,
                       "received_at_utc": utc(), "response_sha256": sha(raw),
                       "authentication_used": False, "safety": guard()}
            save_json(directory / f"tick_{number}_receipt.json", receipt)
            measured = {**receipt, **summarize_ticks(json.loads(raw), request)}
            save_json(directory / f"tick_{number}_summary.json", measured)
            output.append(measured)
            if measured["status"] == "API_ERROR":
                break  # No quota retry while the independent collector is active.
            if number < 4:
                await asyncio.sleep(2)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalogue-receipt", type=Path, required=True)
    args = parser.parse_args()
    guard()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.catalogue_receipt.read_text())
    catalogue = original.get("catalogue", original)
    symbols = audit_catalogue(catalogue)
    if any(target not in symbols for target in TARGETS):
        raise ValueError("Actual catalogue binding required for both target symbols")
    declaration = {"started_at_utc": utc(), "source_sha256": sha(Path(__file__).read_bytes()),
                   "catalogue_receipt_sha256": sha(args.catalogue_receipt.read_bytes()),
                   "symbols": {target: symbols[target] for target in TARGETS}, "windows": WINDOWS,
                   "ticks_per_probe": 4, "maximum_tick_requests": 4, "spacing_seconds": 2,
                   "specifications_url": SPEC_URL, "safety": guard(),
                   "accounts_authentication_orders_proposals_used": False}
    save_json(directory / "declaration.json", declaration)
    errors = []
    try:
        specifications = capture_specs(directory)
    except Exception as error:
        specifications = None
        errors.append({"stage": "specifications", "type": type(error).__name__, "message": str(error)})
    try:
        ticks = asyncio.run(probe(directory, symbols))
    except Exception as error:
        ticks = [json.loads(path.read_text()) for path in sorted(directory.glob("tick_*_summary.json"))]
        errors.append({"stage": "ticks", "type": type(error).__name__, "message": str(error)})
    result = {"finished_at_utc": utc(), "declaration_sha256": sha((directory / "declaration.json").read_bytes()),
              "current_specifications": specifications, "tick_probes": ticks, "errors": errors,
              "actual_tick_requests_sent": len(list(directory.glob("tick_*_sent.json"))),
              "actual_tick_responses_saved": len(list(directory.glob("tick_*_response.wire"))),
              "safety": guard(), "historical_cfd_costs_verified": False,
              "realistic_cfd_execution_verified": False, "strategy_profit_tested": False,
              "files": {p.name: sha(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}}
    save_json(directory / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
