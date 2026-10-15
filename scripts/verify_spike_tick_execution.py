#!/usr/bin/env python3
"""Independent stdlib audit of declared primary tick quote-path diagnosis.

No collection, orders, model fitting, research engine or metrics imports.
Historical execution must be authorized separately after results are ready.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SYMBOLS = ("BOOM600", "CRASH600")
DATES = ("2026-04-11", "2026-04-26", "2026-05-12", "2026-05-28", "2026-06-13", "2026-06-29",
         "2026-07-15", "2026-07-31", "2026-08-16", "2026-09-01", "2026-09-17", "2026-10-03")
CONFIG = {"stop_atr": 2.0, "max_hold_minutes": 15, "entry_delay_minutes": 1,
          "round_trip_cost_atr": 0.1, "max_gap_seconds": 1, "stop_latency_ticks": 1}
ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
KEYS = {"ticks_history", "start", "end", "style", "count", "req_id"}
RECOVERABLE = {"RateLimitError", "OSError", "TimeoutError", "ConnectionClosed", "ConnectionClosedError",
               "ConnectionClosedOK", "InvalidStatus", "InvalidStatusCode", "gaierror", "ConnectionResetError",
               "BrokenPipeError", "SSLError"}
TIMES = {"signal_time", "entry_time", "exit_time", "planned_end", "missing_time", "nominal_entry_time", "trigger_time"}
NUMBERS = {"entry", "exit", "atr", "gross_R", "net_R", "holding_minutes", "trigger_quote"}
BOOLS = {"ambiguous", "censored"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_path(value):
    path = (ROOT / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Nonportable source path: {value}")
    return path


def timestamp(value, whole_second=False):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    seconds = stamp.timestamp()
    if whole_second and not seconds.is_integer():
        raise ValueError("Whole UTC second required")
    return int(seconds) if whole_second else seconds


def date_bounds(date):
    start = timestamp(date + "T00:00:00+00:00", True)
    return start, start + 86400


def date_of(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d")


def number(value, optional=False):
    if optional and value in (None, "", "NaT"):
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a quote/return")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Finite quote/return required")
    return result


def boolean(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError("Canonical saved boolean required")


def integer(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Integer required")
    return value


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.safety_values_checked = 0
        self.max_numeric_error = {}
        self.inputs = {}

    def equal(self, label, actual, expected):
        self.checks += 1
        if isinstance(actual, float) and isinstance(expected, (float, int)) and not isinstance(expected, bool):
            ok = math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9)
            key = label.rsplit("/", 1)[-1]
            self.max_numeric_error[key] = max(self.max_numeric_error.get(key, 0), abs(actual - expected))
        else:
            ok = actual == expected
        if not ok:
            self.failures.append({"check": label, "actual": actual, "expected": expected})

    def require(self, label, condition):
        self.equal(label, condition is True, True)
        if condition is not True:
            raise ValueError(label)

    def safety(self, payload, label):
        if isinstance(payload, dict):
            for key, item in payload.items():
                if key in FLAGS:
                    self.safety_values_checked += 1
                    self.require(f"{label}/{key}", item is False)
                self.safety(item, f"{label}/{key}")
        elif isinstance(payload, list):
            for i, item in enumerate(payload):
                self.safety(item, f"{label}/{i}")

    def pin(self, path, expected=None):
        fingerprint = digest(path)
        self.inputs[str(Path(path).resolve())] = fingerprint
        if expected is not None:
            self.require(f"hash/{path}", fingerprint == expected)
        return fingerprint


def gaps(epochs, start, end):
    result, cursor = [], start
    for stamp in epochs:
        if cursor < stamp:
            result.append({"start_epoch": cursor, "end_exclusive_epoch": stamp, "missing_seconds": stamp-cursor})
        cursor = stamp + 1
    if cursor < end:
        result.append({"start_epoch": cursor, "end_exclusive_epoch": end, "missing_seconds": end-cursor})
    return result


def request_key(request):
    payload = {"endpoint": ENDPOINT, "request": request}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_wire(wire):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON wire key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON wire constant: "+value)
    return json.loads(wire, parse_float=Decimal, parse_constant=invalid_constant, object_pairs_hook=pairs)


def read_ticks(path):
    ticks, previous = {}, None
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["epoch", "quote"]:
            raise ValueError("Exact epoch/quote schema required")
        for row in reader:
            stamp = int(row["epoch"])
            if str(stamp) != row["epoch"] or previous is not None and stamp <= previous:
                raise ValueError("Strict increasing integer tick seconds required")
            quote = Decimal(row["quote"])
            if not quote.is_finite() or quote <= 0:
                raise ValueError("Positive finite tick quote required")
            ticks[stamp], previous = quote, stamp
    return ticks


def float_quotes(quotes):
    result = {}
    for stamp, quote in quotes.items():
        value = float(quote)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Normalized tick quote cannot become a nonfinite or nonpositive float")
        result[stamp] = value
    return result


def audit_source(value, declaration, audit):
    """Reconstruct every accepted raw price and every absent UTC second."""
    label = f"source/{value['symbol']}/{value['date']}"
    start, end = date_bounds(value["date"])
    for key, expected in (("start_epoch", start), ("end_exclusive_epoch", end), ("endpoint", ENDPOINT),
                          ("cadence_seconds", 1), ("expected_grid_rows", 86400), ("normalization_valid", True),
                          ("fills_or_interpolations", False), ("authentication_used", False),
                          ("features_labels_or_tick_outcomes_computed", False)):
        audit.require(f"{label}/{key}", value[key] == expected and type(value[key]) is type(expected))
    audit.safety(value, label)
    for name in ("clean", "raw_pages", "page_audit"):
        audit.pin(source_path(value[name+"_file"]), value[name+"_sha256"])
    raw_path, page_path = (source_path(value[name+"_file"]) for name in ("raw_pages", "page_audit"))
    raw, pages = ([json.loads(line) for line in path.read_text().splitlines()] for path in (raw_path, page_path))
    audit.require(label+"/attempt_count", len(raw) == len(pages) and bool(raw))
    cursor, successes, reconstructed, canonical_quotes, failures, duplicate_total = end-1, {}, {}, {}, [], 0
    oldest_attempt, newest_attempt = None, None
    for ordinal, (record, page) in enumerate(zip(raw, pages, strict=True)):
        tag = f"{label}/attempt/{ordinal}"
        request = record["request"]
        audit.require(tag+"/exact_request_schema", set(request) == KEYS)
        for key in ("start", "end", "count", "req_id"):
            integer(request[key])
        expected = {"ticks_history": value["symbol"], "start": start, "end": cursor,
                    "style": "ticks", "count": declaration["page_size"], "req_id": 1000+len(successes)}
        audit.require(tag+"/bounded_page_chain", request == expected and page["request"] == request)
        audit.require(tag+"/endpoint", record["endpoint"] == page["endpoint"] == ENDPOINT)
        key = request_key(request)
        audit.require(tag+"/request_hash", record["request_key"] == page["request_key"] == key)
        audit.require(tag+"/attempt_metadata", 1 <= integer(record["attempt"]) <= 4 and page["attempt"] == record["attempt"])
        attempted = timestamp(record["received_at_utc"])
        oldest_attempt = attempted if oldest_attempt is None else min(oldest_attempt, attempted)
        newest_attempt = attempted if newest_attempt is None else max(newest_attempt, attempted)
        audit.require(tag+"/after_declaration", attempted >= timestamp(declaration["run_utc"]))
        audit.require(tag+"/before_finalization", attempted <= timestamp(value["completed_at_utc"]))
        wire = record["response_wire"]
        fingerprint = hashlib.sha256(wire.encode()).hexdigest() if wire is not None else None
        audit.require(tag+"/wire_hash", fingerprint == record["wire_sha256"] == page["wire_sha256"])
        audit.require(tag+"/accepted_boolean", type(page["accepted"]) is bool)
        if not page["accepted"]:
            error = page["error"]
            audit.require(tag+"/recoverable", page.get("recoverable") is True and error["error_type"] in RECOVERABLE)
            audit.require(tag+"/error_preservation", record.get("error") == error)
            for field, expected_error in (("endpoint", ENDPOINT), ("request", request), ("request_key", key),
                                          ("attempt", record["attempt"]), ("wire_sha256", fingerprint)):
                audit.require(tag+"/error_"+field, error[field] == expected_error)
            if error["error_type"] == "RateLimitError":
                response = strict_wire(wire) if wire is not None else {}
                audit.require(tag+"/rate_limit_proof", response.get("req_id") == request["req_id"] and
                              response.get("error", {}).get("code") == "RateLimit")
            else:
                audit.require(tag+"/transport_has_no_response", wire is None and fingerprint is None)
            failures.append(error)
            continue
        audit.require(tag+"/successful_wire", wire is not None and key not in successes and "error" not in record)
        response = strict_wire(wire)
        audit.require(tag+"/successful_response", not response.get("error") and response.get("msg_type") == "history" and
                      response.get("req_id") == request["req_id"])
        times, prices = response["history"]["times"], response["history"]["prices"]
        audit.require(tag+"/array_count", isinstance(times, list) and isinstance(prices, list) and
                      len(times) == len(prices) == integer(page["rows"]) and len(times) <= request["count"])
        previous, duplicates = None, 0
        for stamp, price in zip(times, prices, strict=True):
            integer(stamp)
            quote = Decimal(str(price)) if not isinstance(price, bool) else Decimal("NaN")
            audit.require(tag+"/raw_tick", start <= stamp <= cursor and (previous is None or stamp >= previous) and
                          quote.is_finite() and quote > 0)
            if stamp in reconstructed:
                audit.require(tag+"/equal_duplicate", reconstructed[stamp] == quote)
                duplicates += 1
            else:
                canonical_quotes[stamp] = format(quote,"f")
            reconstructed[stamp], previous = quote, stamp
        audit.require(tag+"/duplicate_count", duplicates == page["equal_duplicates_removed"])
        duplicate_total += duplicates
        audit.require(tag+"/page_bounds", page["oldest_epoch"] == (times[0] if times else None) and
                      page["newest_epoch"] == (times[-1] if times else None))
        audit.require(tag+"/empty_terminal", bool(times) or ordinal == len(pages)-1)
        successes[key] = (ordinal, page)
        cursor = times[0]-1 if times else start-1
    audit.require(label+"/reached_start", cursor < start)
    audit.require(label+"/all_errors_preserved", failures == value["request_errors"])
    lineage = value["recovered_retry_lineage"]
    audit.require(label+"/retry_count", len(lineage) == len(failures))
    for index, (error, recovery) in enumerate(zip(failures, lineage, strict=True)):
        audit.require(label+f"/retry/{index}/matched", error["request_key"] in successes)
        ordinal, page = successes[error["request_key"]]
        expected = {"error_index": index, "successful_page_audit_index": ordinal, "request_key": error["request_key"],
                    "successful_wire_sha256": page["wire_sha256"], "endpoint": ENDPOINT,
                    "start_epoch": error["request"]["start"], "end_epoch": error["request"]["end"]}
        audit.require(label+f"/retry/{index}/lineage", recovery == expected and page["request"] == error["request"])
    clean = read_ticks(source_path(value["clean_file"]))
    audit.require(label+"/raw_clean_exact_quotes", clean == dict(sorted(reconstructed.items())))
    with source_path(value["clean_file"]).open(newline="") as stream:
        saved_quotes = [(int(row["epoch"]),row["quote"]) for row in csv.DictReader(stream)]
    audit.require(label+"/canonical_quote_strings", saved_quotes == sorted(canonical_quotes.items()))
    audit.require(label+"/nonempty", bool(clean))
    epochs = list(clean)
    audit.require(label+"/clean_bounds", start <= epochs[0] <= epochs[-1] < end)
    unknown = gaps(epochs, start, end)
    for field, expected in (("rows", len(clean)), ("first_epoch", epochs[0]), ("last_epoch", epochs[-1]),
                            ("missing_seconds", 86400-len(clean)), ("gaps", unknown), ("gap_free", not unknown),
                            ("equal_duplicates_removed", duplicate_total), ("successful_page_count", len(successes))):
        audit.require(label+"/"+field, value[field] == expected and type(value[field]) is type(expected))
    checkpoint_path = source_path(value["raw_pages_file"]).with_name(f"{value['symbol'].lower()}_{value['date']}_checkpoint.json")
    audit.pin(checkpoint_path)
    checkpoint = read_json(checkpoint_path)
    for field, expected in (("config_sha256",value["config_sha256"]),("symbol",value["symbol"]),("date",value["date"]),
                            ("start_epoch",start),("end_exclusive_epoch",end),("endpoint",ENDPOINT),
                            ("next_end",cursor),("page_size",declaration["page_size"]),("success_pages",len(successes)),
                            ("request_errors",failures),("semantic_failure",False),("raw_sha256",value["raw_pages_sha256"]),
                            ("audit_sha256",value["page_audit_sha256"])):
        audit.require(label+"/checkpoint/"+field, checkpoint[field] == expected)
    return float_quotes(clean), {
        "symbol": value["symbol"], "date": value["date"], "rows": len(clean), "missing_seconds": 86400-len(clean),
        "pages": len(successes), "attempts": len(pages), "recovered_errors": len(failures),
        "equal_duplicates_removed": duplicate_total, "first_attempt_started_utc": oldest_attempt,
        "last_attempt_started_utc": newest_attempt, "clean_sha256": value["clean_sha256"]}


def read_m1(path):
    result, previous = {}, None
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["epoch"])
            values = tuple(number(row[k]) for k in ("open", "high", "low", "close"))
            o, h, low, c = values
            if stamp % 60 or previous is not None and stamp <= previous or not 0 < low <= min(o,c) <= max(o,c) <= h:
                raise ValueError("Invalid sorted M1 OHLC")
            result[stamp], previous = values, stamp
    return result


def causal_atr(minutes, signal):
    ranges = []
    for close in range(signal-13*300, signal+1, 300):
        bars = [minutes[t] for t in range(close-300, close, 60)]
        previous = minutes[close-360][3]
        high, low = max(b[1] for b in bars), min(b[2] for b in bars)
        ranges.append(max(high-low, abs(high-previous), abs(low-previous)))
    return math.fsum(ranges)/14


def reconcile(ticks, minutes, date, audit):
    start, end = date_bounds(date)
    complete, maximum = 0, None
    for opening in range(start, end, 60):
        quotes = [ticks.get(t) for t in range(opening, opening+60)]
        if any(q is None for q in quotes):
            continue
        audit.require(f"reconcile/{date}/{opening}/reference", opening in minutes)
        observed = (quotes[0], max(quotes), min(quotes), quotes[-1])
        reference = minutes[opening]
        for name, a, b in zip(("open","high","low","close"), observed, reference, strict=True):
            audit.require(f"reconcile/{date}/{opening}/{name}", abs(a-b) <= 1e-8+1e-12*abs(b))
            maximum = max(maximum or 0, abs(a-b))
        complete += 1
    return {"complete_minutes": complete, "unknown_minutes": 1440-complete, "max_abs_difference": maximum,
            "price_match": True, "coverage_is_separately_audited": True}


def read_signals(path):
    result = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            item = dict(row)
            item["signal_time"] = timestamp(row["signal_time"], True)
            item["atr"] = number(row["atr"])
            item["side"] = int(row["side"])
            if item["signal_time"] % 60 or item["atr"] <= 0 or item["side"] not in (-1,1):
                raise ValueError("Invalid frozen issuance")
            if item["signal_time"] % 1800:
                raise ValueError("Saved issuance must use the declared UTC00/30 clock")
            for key in ("signal_close", "score"):
                if key in row:
                    item[key] = number(row[key])
            result.append(item)
    if len({r["signal_time"] for r in result}) != len(result):
        raise ValueError("Frozen issuance must be unique")
    return result


def scalar_replay(ticks, signals, start, end):
    """Independent second-grid walk: no searchsorted, interpolation or engine."""
    rows, busy = [], start
    accounting = dict.fromkeys(("issued", "filled", "completed", "censored", "missing_entry", "missing_path",
                               "outside_partition", "purged", "overlap_skipped", "ambiguous"), 0)
    accounting["issued"] = len(signals)
    for _, signal in sorted(enumerate(signals), key=lambda pair: (pair[1]["signal_time"], pair[0])):
        issue, atr, side = signal["signal_time"], signal["atr"], signal["side"]
        nominal, planned = issue+60, issue+960
        if not start <= issue < end:
            accounting["outside_partition"] += 1
            continue
        if issue+31*60 > end or planned+1 > end:
            accounting["purged"] += 1
            continue
        if nominal < busy:
            accounting["overlap_skipped"] += 1
            continue
        entry_time = nominal+1
        if entry_time not in ticks:
            accounting["missing_entry"] += 1
            busy = planned
            continue
        entry = ticks[entry_time]
        risk = 2*atr
        stop = entry-side*risk
        if not math.isfinite(stop) or stop <= 0:
            raise ValueError("Positive finite stop required")
        row = {"signal_time": issue, "nominal_entry_time": nominal, "entry_time": entry_time,
               "exit_time": planned, "entry": entry, "exit": None, "atr": atr,
               "gross_R": None, "net_R": None, "reason": "censored_missing_path", "ambiguous": False,
               "holding_minutes": (planned-entry_time)/60, "censored": False,
               "variant": signal.get("variant", ""), "planned_end": planned, "missing_time": None,
               "trigger_time": None, "trigger_quote": None, "side": side}
        accounting["filled"] += 1
        for stamp in range(entry_time+1, planned+2):
            if stamp not in ticks:
                row.update(censored=True, missing_time=stamp)
                break
            price = ticks[stamp]
            if stamp == planned+1:
                row.update(exit=price, exit_time=stamp, reason="time")
                break
            if side*(price-stop) <= 0:
                row.update(trigger_time=stamp, trigger_quote=price)
                if stamp+1 not in ticks:
                    row.update(censored=True, missing_time=stamp+1)
                else:
                    row.update(exit=ticks[stamp+1], exit_time=stamp+1, reason="sl")
                break
        if row["censored"]:
            accounting["censored"] += 1
            accounting["missing_path"] += 1
            busy = planned
        else:
            row["gross_R"] = side*(row["exit"]-entry)/risk
            row["net_R"] = row["gross_R"]-.05
            accounting["completed"] += 1
            busy = row["exit_time"]
        row["holding_minutes"] = (row["exit_time"]-entry_time)/60
        rows.append(row)
    return rows, accounting


def read_ledger(path):
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            for field in TIMES:
                row[field] = timestamp(row[field], True) if row[field] not in ("", "NaT") else None
            for field in NUMBERS:
                row[field] = number(row[field], optional=True)
            for field in BOOLS:
                row[field] = boolean(row[field])
            row["side"] = int(row["side"])
            rows.append(row)
    return rows


def summary(rows):
    known = [r for r in rows if not r["censored"] and r["net_R"] is not None]
    values, gross = [r["net_R"] for r in known], [r["gross_R"] for r in known]
    wins, losses = [v for v in values if v > 0], [v for v in values if v < 0]
    n = len(known)
    mean = statistics.fmean(values) if n else None
    equity = peak = 1.
    drawdown, ruined = 0., False
    for row in sorted(known, key=lambda r: (r["exit_time"], r["entry_time"])):
        factor = 1+.0025*row["net_R"]
        ruined = ruined or factor <= 0
        equity = 0. if ruined else equity*factor
        peak = max(peak, equity)
        drawdown = max(drawdown, 1-equity/peak)
    days = {day: [r["net_R"] for r in known if date_of(r["signal_time"]) == day] for day in DATES}
    residual = math.fsum((math.fsum(v)-mean*len(v))**2 for v in days.values()) if n else None
    mean_gross = statistics.fmean(gross) if n else None
    return {"trades": len(rows), "completed": n, "censored": sum(r["censored"] for r in rows),
            "invalid_uncensored": len(rows)-sum(r["censored"] for r in rows)-n,
            "wins": len(wins), "losses": len(losses), "flat_trades": sum(v == 0 for v in values),
            "win_rate": len(wins)/n if n else None, "mean_net_R": mean,
            "median_net_R": statistics.median(values) if n else None, "sum_net_R": math.fsum(values),
            "mean_gross_R": mean_gross, "profit_factor": math.fsum(wins)/-math.fsum(losses) if losses else None,
            "average_win_R": statistics.fmean(wins) if wins else None,
            "average_loss_R": statistics.fmean(losses) if losses else None,
            "active_days": sum(bool(v) for v in days.values()), "calendar_days": 12,
            "span_calendar_days": (date_bounds(DATES[-1])[1]-date_bounds(DATES[0])[0])//86400,
            "median_holding_minutes": statistics.median(r["holding_minutes"] for r in known) if n else None,
            "ambiguous": sum(r["ambiguous"] for r in rows),
            "break_even_cost_atr": mean_gross*2 if mean_gross is not None else None,
            "cluster_se": math.sqrt(12/11*residual)/n if n else None, "selection_score": None,
            "closed_trade_return": equity-1, "closed_trade_max_drawdown": drawdown, "equity_ruin": ruined,
            "illustrative_risk_fraction": .0025, "drawdown_basis": "closed trades only; fixed fractional risk illustration"}


def stop_summary(rows):
    stopped = [r for r in rows if r["reason"] == "sl" and not r["censored"]]
    if not stopped:
        return {"completed_stops": 0, "mean_trigger_overshoot_R": None, "mean_successor_move_R": None}
    overshoot = [-r["side"]*(r["trigger_quote"]-(r["entry"]-r["side"]*2*r["atr"]))/(2*r["atr"]) for r in stopped]
    moves = [r["side"]*(r["exit"]-r["trigger_quote"])/(2*r["atr"]) for r in stopped]
    return {"completed_stops": len(stopped), "mean_trigger_overshoot_R": statistics.fmean(overshoot),
            "median_trigger_overshoot_R": statistics.median(overshoot), "max_trigger_overshoot_R": max(overshoot),
            "mean_successor_move_R": statistics.fmean(moves)}


def audit_bootstrap(summary_path, payload, declaration, audit):
    """Only declared descriptive labels/count algebra; no bootstrap regeneration."""
    for field, expected in (("bootstrap_repeats", 9999), ("bootstrap_seed", 20261005),
                            ("inference_kind", "descriptive_iid_scheduled_day_bootstrap_not_discovery"),
                            ("no_contiguous_weekly_claim", True), ("selection_score", None)):
        audit.equal(summary_path+"/"+field, payload[field], expected)
    audit.require(summary_path+"/valid_replicates", 0 <= payload["mean_valid_replicates"] <= 9999 and
                  0 <= payload["pf_undefined_replicates"] <= 9999)


def frozen_metadata(study, audit):
    declaration = read_json(study/"declaration.json")
    sources = read_json(study/"tick_sources.json")
    result = read_json(study/"results.json")
    old = ROOT/"docs/spike_nonlinear_20261005"
    selection, old_declaration = read_json(old/"selection.json"), read_json(old/"declaration.json")
    for name, payload in (("declaration",declaration),("sources",sources),("results",result),("selection6",selection),("declaration6",old_declaration)):
        audit.safety(payload, name)
        audit.require(name+"/safety", payload["safety"] == dict.fromkeys(FLAGS,False))
    decl_sha = audit.pin(study/"declaration.json", (study/"declaration.sha256").read_text().strip())
    sources_sha = audit.pin(study/"tick_sources.json", (study/"tick_sources.sha256").read_text().strip())
    audit.pin(study/"results.json")
    audit.require("frozen/linkage", sources["declaration_sha256"] == result["declaration_sha256"] == decl_sha and
                  result["tick_sources_sha256"] == sources_sha)
    old_sha = audit.pin(old/"selection.json", declaration["round6_selection_sha256"])
    audit.require("round6/selection_pin", old_sha == (old/"selection.sha256").read_text().strip())
    old_decl_sha = audit.pin(old/"declaration.json", selection["declaration_sha256"])
    audit.require("round6/declaration_pin", old_decl_sha == (old/"declaration.sha256").read_text().strip())
    for field in old_declaration:
        if field != "stage":
            audit.require("round6/declaration_copy/"+field, selection[field] == old_declaration[field])
    for mapping in (declaration["code_hashes"], declaration["round6_source_files"], selection["code_hashes"], sources["acquisition_files"]):
        for path, fingerprint in mapping.items():
            audit.pin(source_path(path), fingerprint)
    for pairs in selection["sources"].values():
        for value in pairs.values():
            for filekey, shakey in (("path","sha256"),("manifest","manifest_sha256"),("raw_file","raw_sha256"),("page_audit","page_audit_sha256")):
                audit.pin(source_path(value[filekey]), value[shakey])
    for name, payload in (("declaration",declaration),("result",result)):
        audit.require(name+"/dates", payload["dates"] == list(DATES))
        audit.require(name+"/config", payload["config"] == CONFIG)
    audit.require("declaration/symbols", declaration["symbols"] == list(SYMBOLS))
    audit.require("declaration/prepayoff", declaration["tick_outcomes_evaluated"] is False and declaration["raw_ticks_collected"] is False)
    audit.require("sources/prepayoff", sources["tick_outcomes_evaluated"] is False)
    audit.require("declaration/page_size", declaration["page_size"] == 1000 and declaration["endpoint"] == ENDPOINT)
    for field, expected in (("bootstrap_repeats",9999),("modeled_cost_not_measured",True),("max_opportunities_per_model",576),
                            ("study_kind","adaptive_historical_sampled_execution_diagnosis_not_profit_validation")):
        audit.require("declaration/"+field,declaration[field] == expected)
    audit.require("chronology/stages", declaration["stage"] == "tick_execution_preacquisition" and
                  sources["stage"] == "tick_sources_frozen_before_payoff" and result["stage"] == "sampled_tick_execution_diagnosis")
    audit.require("chronology/frozen_before_execution", timestamp(declaration["run_utc"]) <= timestamp(sources["frozen_utc"]) <= timestamp(result["run_utc"]))
    audit.require("chronology/round6_frozen_first", timestamp(selection["frozen_utc"]) <= timestamp(declaration["run_utc"]))
    for field, expected in (("actual_money_profit","NOT TESTED"),("prospective_paper","NOT TESTED"),
                            ("goal_achieved",False),("maximum_possible_per_model",576),("no_model_refit",True)):
        audit.require("result/"+field, result[field] == expected)
    audit.require("result/symbols", sorted(result["symbols"]) == sorted(SYMBOLS))
    return declaration, sources, result, selection


def audit_acquisition(sources, declaration, audit):
    paths = list(sources["acquisition_files"])
    configs = [p for p in paths if p.endswith("/acquisition_config.json")]
    summaries = [p for p in paths if p.endswith("/acquisition_summary.json")]
    audit.require("acquisition/files", len(configs) == len(summaries) == 1)
    config, summary_payload = read_json(source_path(configs[0])), read_json(source_path(summaries[0]))
    config_sha = digest(source_path(configs[0]))
    for payload, name in ((config,"config"),(summary_payload,"summary")):
        audit.safety(payload, "acquisition/"+name)
    for field, expected in (("symbols",list(SYMBOLS)),("dates",list(DATES)),("endpoint",ENDPOINT),
                            ("page_size",1000),("cadence_seconds",1),("collector_sha256",declaration["code_hashes"]["scripts/collect_spike_ticks.py"])):
        audit.require("acquisition/config/"+field, config[field] == expected)
    audit.require("acquisition/minspacing", config["effective_min_request_spacing_seconds"] == max(config["requested_delay_seconds"],.5))
    audit.require("acquisition/config_linkage", summary_payload["config"] == config and summary_payload["config_sha256"] == config_sha)
    audit.require("acquisition/complete", summary_payload["status"] == "COMPLETE" and not summary_payload["errors"])
    for field in ("authentication_used","account_or_order_requests","features_labels_or_tick_outcomes_computed"):
        audit.require("acquisition/"+field, summary_payload[field] is False)
    for symbol in SYMBOLS:
        audit.require(symbol+"/source_dates", sorted(sources["sources"][symbol]) == sorted(DATES))
        audit.require(symbol+"/summary_dates", sorted(summary_payload["completed_sources"][symbol]) == sorted(DATES))
        for date, value in sources["sources"][symbol].items():
            audit.require(symbol+"/"+date+"/config", value["config_sha256"] == config_sha)
            item = summary_payload["completed_sources"][symbol][date]
            expected = {"manifest_file":value["manifest"],"manifest_sha256":value["manifest_sha256"],
                        "clean_file":value["clean_file"],"clean_sha256":value["clean_sha256"],
                        "rows":value["rows"],"missing_seconds":value["missing_seconds"]}
            audit.require(symbol+"/"+date+"/summary_linkage", item == expected)
    # This file is not separately in the source freeze: the pinned summary anchors its successful wire.
    probe_path = source_path(configs[0]).parent/"active_symbols_wire.jsonl"
    audit.pin(probe_path)
    probe = summary_payload["public_symbol_probe"]
    records = [json.loads(line) for line in probe_path.read_text().splitlines()]
    failures, successes, current_failures = [], [], []
    for ordinal, record in enumerate(records):
        label = f"active_symbols/{ordinal}"
        audit.safety(record, label)
        audit.require(label+"/whitelist", record["request"] == {"active_symbols":"brief","req_id":1} and record["endpoint"] == ENDPOINT)
        audit.require(label+"/after_declaration", timestamp(record["received_at_utc"]) >= timestamp(declaration["run_utc"]))
        wire = record["response_wire"]
        audit.require(label+"/wire_hash", record["wire_sha256"] == (hashlib.sha256(wire.encode()).hexdigest() if wire is not None else None))
        if record["accepted"]:
            response = strict_wire(wire)
            audit.require(label+"/response", response.get("req_id") == 1 and not response.get("error"))
            available = {item.get("underlying_symbol") or item.get("symbol") for item in response["active_symbols"]}
            audit.require(label+"/native_symbols", set(SYMBOLS) <= available)
            successes.append(record)
            last_group_failures = current_failures
            current_failures = []
        else:
            audit.require(label+"/recoverable", record.get("recoverable") is True and record["error_type"] in RECOVERABLE)
            if record["error_type"] == "RateLimitError":
                response = strict_wire(wire) if wire is not None else {}
                audit.require(label+"/RateLimit", response.get("req_id") == 1 and response.get("error",{}).get("code") == "RateLimit")
            else:
                audit.require(label+"/transport_has_no_response",wire is None and record["wire_sha256"] is None)
            error = {key:record[key] for key in ("endpoint","request","attempt","error_type","error")}
            failures.append(error);current_failures.append(error)
    audit.require("active_symbols/summary", bool(successes) and successes[-1]["wire_sha256"] == probe["successful_wire_sha256"] and
                  not current_failures and last_group_failures == probe["request_errors"] and probe["request"] == {"active_symbols":"brief","req_id":1} and
                  probe["endpoint"] == ENDPOINT and probe["exact_same_request_recovered"] is True)


def verify(study, audit):
    declaration, sources, result, selection = frozen_metadata(study, audit)
    audit_acquisition(sources, declaration, audit)
    groups, source_reports = [], []
    source_files = []
    for symbol in SYMBOLS:
        for date in DATES:
            value = sources["sources"][symbol][date]
            audit.require(f"{symbol}/{date}/identity",value["symbol"] == symbol and value["date"] == date)
            source_files.extend(value[key] for key in ("manifest","clean_file","raw_pages_file","page_audit_file"))
    audit.require("sources/no_aliases",len(source_files) == len(set(source_files)) == 96)
    for symbol in SYMBOLS:
        m1_source = selection["sources"][symbol]["fresh"]
        minutes = read_m1(source_path(m1_source["path"]))
        tick_days = {}
        for date in DATES:
            value = sources["sources"][symbol][date]
            audit.pin(source_path(value["manifest"]), value["manifest_sha256"])
            manifest = read_json(source_path(value["manifest"]))
            audit.require(f"{symbol}/{date}/manifest_copy", {k:value[k] for k in manifest} == manifest)
            audit.require(f"{symbol}/{date}/collector_pin", value["collector_sha256"] == declaration["code_hashes"]["scripts/collect_spike_ticks.py"])
            audit.require(f"{symbol}/{date}/freeze_after_complete", timestamp(value["completed_at_utc"]) <= timestamp(sources["frozen_utc"]))
            ticks, report = audit_source(value, declaration, audit)
            reconciliation = reconcile(ticks, minutes, date, audit)
            for field, expected in reconciliation.items():
                audit.equal(f"{symbol}/{date}/reconciliation/{field}", value["reconciliation"][field], expected)
            tick_days[date] = ticks
            source_reports.append({**report, "reconciliation":reconciliation})
        models = result["symbols"][symbol]["models"]
        expected_keys = {f"{family}_{mode}" for family in ("CLOCK","BOOST44","RIDGE44","RIDGE19") for mode in ("SPIKE","DRIFT")}
        audit.require(symbol+"/model_family", set(models) == set(declaration["sampled_signals"][symbol]) == expected_keys)
        reconstructed_groups = {}
        for key, saved in declaration["sampled_signals"][symbol].items():
            label = symbol+"/"+key
            before = len(audit.failures)
            signals_path = source_path(saved["path"])
            audit.pin(signals_path, saved["sha256"])
            signals = read_signals(signals_path)
            audit.require(label+"/issued_count", len(signals) == saved["rows"] and len(signals) <= 576)
            mode = key.rsplit("_",1)[1]
            side = (1 if symbol == "BOOM600" else -1)*(1 if mode == "SPIKE" else -1)
            variant = key if key.startswith("CLOCK_") else "NONLINEAR_"+key
            for i, signal in enumerate(signals):
                audit.require(f"{label}/issuance/{i}/date_side_variant", date_of(signal["signal_time"]) in DATES and
                              signal["signal_time"] % 1800 == 0 and signal["side"] == side and signal["variant"] == variant)
                audit.equal(f"{label}/issuance/{i}/causal_M5ATR14", signal["atr"], causal_atr(minutes,signal["signal_time"]))
                audit.equal(f"{label}/issuance/{i}/closed_signal_price", signal["signal_close"], minutes[signal["signal_time"]-60][3])
            row = models[key]
            if not key.startswith("CLOCK_"):
                frozen = selection["symbols"][symbol]["models"][key]
                audit.require(label+"/development_binding", row["development_eligible"] == saved["development_eligible"] == frozen["development_eligible"])
                fit = frozen["final_model"]
                audit.require(label+"/training_before_fresh", timestamp(fit["training_end"],True) < m1_source["first_epoch"])
                audit.require(label+"/training_purged", timestamp(fit["training_latest_issue"],True)+31*60 <= timestamp(fit["training_end"],True) and
                              timestamp(fit["training_latest_planned_end"],True) <= timestamp(fit["training_end"],True))
                cutoff = fit["estimator"]["threshold"]
                for i, issued_signal in enumerate(signals):
                    audit.require(f"{label}/issuance/{i}/saved_score_cutoff", issued_signal["score"] >= cutoff and issued_signal["score"] > 0)
                original_path = ROOT/"docs/spike_nonlinear_20261005"/f"{symbol}_later_temporal180_{key}_signals.csv"
                original = [r for r in read_signals(original_path) if date_of(r["signal_time"]) in DATES]
                audit.require(label+"/all_frozen_sampled_issuance", len(original) == len(signals))
                for i, (observed, expected) in enumerate(zip(signals,original,strict=True)):
                    audit.require(f"{label}/issuance/{i}/fields", set(observed) == set(expected))
                    for field in expected:
                        audit.equal(f"{label}/issuance/{i}/frozen/{field}",observed[field],expected[field])
            reconstructed = []
            saved_audits = row["audits"]
            audit.require(label+"/twelve_audits", len(saved_audits) == 12)
            for date, recorded in zip(DATES,saved_audits,strict=True):
                part = [s for s in signals if date_of(s["signal_time"]) == date]
                audit.require(label+"/"+date+"/opportunity_bound", len(part) <= 48)
                rows, accounting = scalar_replay(tick_days[date],part,*date_bounds(date))
                audit.require(label+"/"+date+"/audit_date", recorded["date"] == date)
                for field, count in accounting.items():
                    audit.equal(label+"/"+date+"/accounting/"+field,recorded[field],count)
                for field, expected in (("purge_minutes",31),("take_profit",None),("config",CONFIG),
                                        ("entry_rule","first_observed_quote_strictly_after_nominal_order_time"),
                                        ("stop_rule","next_observed_quote_strictly_after_trigger"),
                                        ("expiry_rule","first_observed_quote_strictly_after_nominal_expiry"),
                                        ("expiry_anchor","nominal_entry_time_not_actual_entry_time"),
                                        ("deadline_priority","stop_trigger_at_expiry_precedes_timeout"),
                                        ("partition_exit_allowance_seconds",1),("fill_interpretation","quoteproxy_notbrokerfills")):
                    audit.equal(label+"/"+date+"/policy/"+field,recorded[field],expected)
                audit.safety(recorded,label+"/"+date)
                audit.require(label+"/"+date+"/accounting_identity", accounting["issued"] == sum(accounting[f] for f in
                              ("filled","missing_entry","outside_partition","purged","overlap_skipped")) and
                              accounting["filled"] == accounting["completed"]+accounting["censored"] == len(rows))
                reconstructed.extend(rows)
            ledger_path = study/f"{symbol}_{key}_ticks_trades.csv"
            audit.pin(ledger_path)
            observed = read_ledger(ledger_path)
            audit.require(label+"/ledger_count", len(observed) == len(reconstructed))
            for i, (actual, expected) in enumerate(zip(observed,reconstructed,strict=True)):
                audit.require(f"{label}/ledger/{i}/fields", set(actual) == set(expected))
                for field, value in expected.items():
                    audit.equal(f"{label}/ledger/{i}/{field}",actual[field],value)
            independent = summary(reconstructed)
            for field, expected in independent.items():
                audit.equal(label+"/summary/"+field,row["metrics"][field],expected)
            audit_bootstrap(label+"/summary",row["metrics"],declaration,audit)
            for field, expected in stop_summary(reconstructed).items():
                audit.equal(label+"/stops/"+field,row["stop_diagnostics"][field],expected)
            for field in ("historical_candidate","user_target_observed"):
                audit.require(label+"/rejected/"+field,row[field] is False)
            reasons = {"diagnostic_sample_max576_below1000","reused_adaptive_history_not_strategy_oos","measured_costs_fills_not_tested"}
            if not key.startswith("CLOCK_") and not row["development_eligible"]:
                reasons.add("round6_development_rejected")
            audit.require(label+"/rejection_reasons", set(row["rejection_reasons"]) == reasons)
            reconstructed_groups[key] = reconstructed
            groups.append({"symbol":symbol,"model":key,"saved_issuance":len(signals),"trades":len(reconstructed),
                           "completed":independent["completed"],"censored":independent["censored"],
                           "profit_factor":independent["profit_factor"],"mean_net_R":independent["mean_net_R"],
                           "ledger_sha256":digest(ledger_path),"mismatches":len(audit.failures)-before})
        for key, rows in reconstructed_groups.items():
            if key.startswith("CLOCK_"):
                continue
            clock = summary(reconstructed_groups["CLOCK_"+key.rsplit("_",1)[1]])
            own = summary(rows)
            delta = own["mean_net_R"]-clock["mean_net_R"] if own["completed"] and clock["completed"] else None
            audit.equal(symbol+"/"+key+"/clock_mean_difference",models[key]["clock_comparison"]["mean_R_difference"],delta)
    for path, expected in tuple(audit.inputs.items()):
        audit.require("unchanged_during_audit/"+path,digest(path) == expected)
    audit.require("all_groups_audited",len(groups) == 16)
    return groups, source_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/spike_tick_execution_20261005")
    args = parser.parse_args()
    audit, groups, sources = Audit(), [], []
    for flag in FLAGS:
        audit.require("environment/"+flag,os.environ.get(flag,"false").lower() == "false")
    target = args.output/"independent_audit.json"
    if target.exists():
        raise ValueError("Refusing independent audit overwrite; preserve prior attempt explicitly")
    try:
        groups, sources = verify(args.output.resolve(),audit)
    except Exception as exc:
        audit.failures.append({"check":"fatal_exception","type":type(exc).__name__,"message":str(exc)})
    payload = {"stage":"independent_primary_tick_execution_audit","run_utc":datetime.now(timezone.utc).isoformat(),
               "passed":not audit.failures,"checks":audit.checks,"errors":audit.failures,
               "safety":dict.fromkeys(FLAGS,False),"saved_false_values_checked":audit.safety_values_checked,
               "groups":groups,"sources":sources,"primary_trade_paths":sum(v["trades"] for v in groups),
               "max_numeric_error":audit.max_numeric_error,"input_sha256":audit.inputs,
               "verifier_file":"scripts/verify_spike_tick_execution.py","verifier_sha256":digest(Path(__file__)),
               "scope":{"implementation":"Independent stdlib raw/Decimal/grid and scalar one-second payoff arithmetic",
                        "included":["all24 raw-normalized public tick sources and exact retry/page-chain provenance",
                                    "1-second coverage and complete60-tick M1 OHLC reconciliation",
                                    "frozen code/models/source/selection/issuance hashes and chronology",
                                    "all16 saved primary issuance-to-ledger accounting including purges/overlap/missing entry/censoring",
                                    "causal M5ATR14, strict-next entry/stop/expiry, quote R and holding time",
                                    "point summaries, stop overshoot/successor movement and closed-trade risk arithmetic",
                                    "saved primary clock mean differences and explicit diagnostic-only rejection"],
                        "excluded":["regenerated ML features/fitting/model scores and complete eligible-clock discovery",
                                    "bootstrap regeneration or discovery/inferential validity",
                                    "trigger sensitivity, OHLC stress/barrier and cost-sensitivity ledgers or matched inference",
                                    "actual broker execution, costs, monetary profit and prospective OOS claims"],
                        "attempt_time_caveat":"received_at_utc is recorded before the request; no actual send pacing assertion"}}
    with target.open("x") as stream:
        json.dump(payload,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
    print(json.dumps({"passed":payload["passed"],"checks":audit.checks,"groups":len(groups),"sources":len(sources),
                      "paths":payload["primary_trade_paths"],"errors":len(audit.failures),"audit":str(target)}),flush=True)
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
