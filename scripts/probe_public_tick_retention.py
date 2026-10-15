#!/usr/bin/env python3
"""Four fixed public-history availability requests; no return or payoff analysis.

The source collector remains immutable. A successful four-second response only
establishes availability of that window, never completeness of a year of ticks.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from collect_spike_ticks import ENDPOINT, ROOT, SAFETY, PublicClient, safety_guard, tick_rows, validate_request

STARTS = ("2025-10-09T11:08:00+00:00", "2026-01-01T00:00:00+00:00")
SYMBOLS = ("BOOM600", "CRASH600")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(data)


def requests():
    result = []
    for symbol in SYMBOLS:
        for start_text in STARTS:
            start = int(datetime.fromisoformat(start_text).timestamp())
            request = {"ticks_history": symbol, "start": start, "end": start + 3,
                       "count": 4, "style": "ticks", "req_id": len(result) + 1}
            validate_request(request)
            result.append(request)
    return result


def parse_wire(wire):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value}")

    value = json.loads(wire, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Public response must be an object")
    return value


def describe(wire, request):
    rows = tick_rows(parse_wire(wire), request)
    epochs = [row[0] for row in rows]
    if len(set(epochs)) != len(epochs):
        raise ValueError("Repeated seconds cannot establish exact window completeness")
    expected = list(range(request["start"], request["end"] + 1))
    return {"status": "COMPLETE_WINDOW" if epochs == expected else "INCOMPLETE_WINDOW",
            "observed_rows": len(rows), "expected_rows": 4,
            "epochs": epochs, "missing_epochs": sorted(set(expected) - set(epochs)),
            "first_epoch": epochs[0] if epochs else None,
            "last_epoch": epochs[-1] if epochs else None}


async def run(output):
    safety_guard()
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Availability artifacts must stay inside the repository")
    output.mkdir(parents=True, exist_ok=False)
    declaration = {"schema": 1, "declared_at_utc": utc(), "endpoint": ENDPOINT,
                   "purpose": "Bounded old-tick availability only; no returns, features, labels or payoff",
                   "requests": requests(), "maximum_requests": 4, "retries": 0,
                   "coverage_claim": "Only the four requested four-second windows",
                   "full_continuous_tick_coverage": "NOT TESTED",
                   "safety": safety_guard(),
                   "source_sha256": {name: sha((ROOT / "scripts" / name).read_bytes()) for name in
                                     ("probe_public_tick_retention.py", "collect_spike_ticks.py")}}
    declaration_sha = save(output / "declaration.json", declaration)
    client = PublicClient()
    records = []
    try:
        for request in declaration["requests"]:
            safety_guard()
            prefix = f"request_{request['req_id']}"
            # Intent is flushed before opening/sending on the socket.
            intent = {"request": request, "attempted_at_utc": utc(), "declaration_sha256": declaration_sha,
                      "safety": safety_guard()}
            intent_sha = save(output / f"{prefix}_intent.json", intent)
            record = {"request": request, "intent_sha256": intent_sha,
                      "attempted_at_utc": intent["attempted_at_utc"], "safety": dict(SAFETY)}
            try:
                wire = await client.request(request)
                data = wire.encode("utf-8") if isinstance(wire, str) else bytes(wire)
                with (output / f"{prefix}_response.wire").open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                record["wire_sha256"] = sha(data)
                record["wire_bytes"] = len(data)
                if not isinstance(wire, str):
                    raise ValueError("Expected a text WebSocket response")
                record.update(describe(wire, request))
            except Exception as exc:
                record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
                await client.reset()
            record["finished_at_utc"] = utc()
            record["record_sha256"] = save(output / f"{prefix}_result.json", record)
            records.append(record)
            print(json.dumps({key: record.get(key) for key in
                              ("request", "status", "observed_rows", "first_epoch", "last_epoch",
                               "error_type", "error")}), flush=True)
    finally:
        await client.reset()
    summary = {"schema": 1, "finished_at_utc": utc(), "declaration_sha256": declaration_sha,
               "records": records, "safety": safety_guard(),
               "complete_windows": sum(row["status"] == "COMPLETE_WINDOW" for row in records),
               "full_continuous_tick_coverage": "NOT TESTED", "strategy_payoff": "NOT TESTED",
               "cash_profit": "NOT TESTED"}
    result_sha = save(output / "results.json", summary)
    print(json.dumps({"results_sha256": result_sha, "complete_windows": summary["complete_windows"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output_dir))
