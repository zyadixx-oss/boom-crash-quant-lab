"""Independent data-only reconstruction from retained Deriv wire responses.

No strategy, indicator, return, network, account or execution code is imported.
The effective interval depends only on the first/last authentic canonical M1
row. Internal gaps are retained; a convenient gap-free segment is never chosen.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ENDPOINT = "wss://api.derivws.com/trading/v1/options/ws/public"
FIELDS = ["epoch", "open", "high", "low", "close"]
SAFETY = dict.fromkeys(("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"), False)


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(raw):
    def bad(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(raw, parse_constant=bad)


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"Boolean {name}")
    try:
        numeric = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name}") from error
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValueError(f"Nonintegral {name}")
    return int(numeric)


def _receipt(receipt, *, kind, api_symbol=None):
    safety = receipt.get("safety", {})
    if (receipt.get("endpoint") != ENDPOINT or set(safety) != set(SAFETY)
            or any(value is not False for value in safety.values())
            or receipt.get("authentication_used") is not False):
        raise ValueError("Original unauthenticated public-only receipt required")
    path = Path(receipt["wire_file"])
    if _hash(path) != receipt["file_sha256"]:
        raise ValueError("Original compressed wire hash mismatch")
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != receipt["wire_sha256"]:
        raise ValueError("Original decoded wire hash mismatch")
    response = _json(raw)
    request = receipt["request"]
    if not isinstance(response, dict) or response.get("error"):
        raise ValueError("Error response cannot establish historical prices")
    if response.get("req_id") != request.get("req_id") or "req_id" not in request:
        raise ValueError("Original response/request identity mismatch")
    if "echo_req" in response and response["echo_req"] != request:
        raise ValueError("Original echoed request differs from receipt")
    if kind == "catalogue":
        if set(request) - {"active_symbols", "req_id"} or request.get("active_symbols") != "brief":
            raise ValueError("Only allowlisted public catalogue request permitted")
        if not isinstance(response.get("active_symbols"), list):
            raise ValueError("Missing original public catalogue")
    else:
        allowed = {"ticks_history", "style", "granularity", "count", "start", "end", "req_id"}
        if (set(request) - allowed or request.get("ticks_history") != api_symbol
                or request.get("style") != "candles" or request.get("granularity") != 60):
            raise ValueError("Original M1 page uses wrong symbol or nonallowlisted request")
        if not 1 <= _integer(request.get("count"), "count") <= 5000:
            raise ValueError("Invalid original page count")
        if _integer(request.get("start"), "start") > _integer(request.get("end"), "end"):
            raise ValueError("Invalid original request interval")
        if not isinstance(response.get("candles"), list):
            raise ValueError("Missing original candle array")
    return response


def audit_catalogue(receipt):
    """Resolve exact ten names independently from the original catalogue wire."""
    response = _receipt(receipt, kind="catalogue")
    resolved = {}
    for family in ("BOOM", "CRASH"):
        for number in (300, 500, 600, 900, 1000):
            name = f"{family.title()} {number} Index".casefold()
            matches = [row for row in response["active_symbols"]
                       if str(row.get("underlying_symbol_name", row.get("display_name", ""))).casefold() == name]
            if len(matches) > 1:
                raise ValueError("Ambiguous original catalogue symbol name")
            if matches:
                symbol = matches[0].get("underlying_symbol") or matches[0].get("symbol")
                if not isinstance(symbol, str) or not symbol.isalnum():
                    raise ValueError("Invalid original catalogue identifier")
                resolved[f"{family}{number}"] = symbol
    return resolved


def _manifest_chain(row):
    """Pinned ancestors are necessary when a resumed fetch inherits old prices."""
    chain, seen = [row], set()
    current = row
    while current.get("parent_manifest"):
        path = Path(current["parent_manifest"])
        if str(path.resolve()) in seen:
            raise ValueError("Cyclic source manifest ancestry")
        seen.add(str(path.resolve()))
        if _hash(path) != current.get("parent_manifest_sha256"):
            raise ValueError("Original parent manifest hash mismatch")
        parent = _json(path.read_bytes())
        if parent.get("symbol") != row["symbol"] or parent.get("api_symbol") != row["api_symbol"]:
            raise ValueError("Original parent symbol identity mismatch")
        chain.append(parent)
        current = parent
    return list(reversed(chain))


def _prices(batch):
    values = []
    for candle in batch:
        epoch = _integer(candle.get("epoch"), "candle epoch")
        try:
            if any(isinstance(candle[name], bool) for name in FIELDS[1:]):
                raise ValueError("Boolean original price")
            prices = [float(candle[name]) for name in FIELDS[1:]]
        except (ValueError, TypeError, OverflowError, KeyError) as error:
            raise ValueError("Malformed original OHLC") from error
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise ValueError("Invalid original finite positive OHLC")
        opening, high, low, close = prices
        if low > min(opening, close) or high < max(opening, close):
            raise ValueError("Invalid original OHLC envelope")
        values.append((epoch, *prices))
    return pd.DataFrame(values, columns=FIELDS).astype({"epoch": "int64", **{k: "float64" for k in FIELDS[1:]}})


def _gaps(epochs, start, end):
    boundaries = np.r_[start - 60, epochs, end].astype(np.int64)
    gaps = []
    for before, after in zip(boundaries[:-1], boundaries[1:]):
        if after - before > 60:
            gaps.append({"start_epoch": int(before + 60), "end_exclusive_epoch": int(after),
                         "missing_minutes": int((after - before) // 60 - 1)})
    return gaps


def audit_source_row(row, *, catalogue_provenance=None):
    """Reconstruct one requested CSV, preserving missing and invalid evidence.

    Malformed prices, contradictory duplicates, changed bytes or wrong identity
    raise rather than return a usable dataset. Off-grid rows remain counted and
    only aligned, closed, actually received rows can enter the canonical CSV.
    """
    canonical = row.get("symbol", "")
    if not re.fullmatch(r"(?:BOOM|CRASH)(?:300|500|600|900|1000)", canonical):
        raise ValueError("Only the ten named Boom/Crash targets are allowed")
    api_symbol = row.get("api_symbol")
    if not isinstance(api_symbol, str) or not api_symbol.isalnum():
        raise ValueError("Invalid actual API symbol")
    if catalogue_provenance is not None and audit_catalogue(catalogue_provenance).get(canonical) != api_symbol:
        raise ValueError("Target does not match independently resolved public catalogue")
    start = _integer(row["requested_start_epoch"], "requested start")
    end = _integer(row["cutoff_exclusive_epoch"], "requested cutoff")
    if start % 60 or end % 60 or start >= end:
        raise ValueError("Requested interval must be aligned nonempty M1")
    path = Path(row["requested_file"])
    if _hash(path) != row["requested_sha256"]:
        raise ValueError("Original normalized source hash mismatch")
    saved = pd.read_csv(path, float_precision="round_trip")
    if list(saved.columns) != FIELDS:
        raise ValueError("Canonical CSV field identity mismatch")
    saved_values = _prices(saved.to_dict("records"))
    saved_epoch = saved_values.epoch.to_numpy()
    if (len(saved_epoch) and (np.any(np.diff(saved_epoch) <= 0) or np.any(saved_epoch % 60)
                             or saved_epoch[0] < start or saved_epoch[-1] + 60 > end)):
        raise ValueError("Saved source is not unique closed chronological canonical M1")
    chunks, seen_pages, off_grid = [], set(), []
    original_rows = boundary_rows = equal_duplicates = references = 0
    excluded = {"off_grid": 0, "before_requested_start": 0,
                "not_closed_before_requested_cutoff": 0, "not_closed_at_original_receipt": 0}
    chain = _manifest_chain(row)
    for source in chain:
        original_cutoff = _integer(source["cutoff_exclusive_epoch"], "original receipt cutoff")
        for page in source.get("pages", []):
            references += 1
            response = _receipt(page, kind="candles", api_symbol=api_symbol)
            page_key = (page["wire_sha256"], original_cutoff,
                        json.dumps(page["request"], sort_keys=True, separators=(",", ":")))
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            batch = _prices(response["candles"])
            original_rows += len(batch)
            if page.get("excluded_provider_boundary_response"):
                if batch.empty or not (batch.epoch > _integer(page["request"]["end"], "request end")).any():
                    raise ValueError("Allegedly excluded provider boundary is inside its requested window")
                boundary_rows += len(batch)
                continue
            if len(batch) and (batch.epoch > _integer(page["request"]["end"], "request end")).any():
                raise ValueError("Unmarked provider response exceeds its requested end")
            for label, mask in (
                ("off_grid", batch.epoch % 60 != 0),
                ("before_requested_start", batch.epoch < start),
                ("not_closed_before_requested_cutoff", batch.epoch + 60 > end),
                ("not_closed_at_original_receipt", batch.epoch + 60 > original_cutoff),
            ):
                excluded[label] += int(mask.sum())
            off_grid.extend(int(value) for value in batch.loc[batch.epoch % 60 != 0, "epoch"])
            keep = ((batch.epoch % 60 == 0) & (batch.epoch >= start)
                    & (batch.epoch + 60 <= min(end, original_cutoff)))
            chunks.append(batch.loc[keep])
    if not seen_pages:
        raise ValueError("Full original page lineage required")
    received = (pd.concat(chunks, ignore_index=True) if chunks else
                pd.DataFrame(columns=FIELDS).astype({"epoch": "int64", **{k: "float64" for k in FIELDS[1:]}}))
    if received.epoch.duplicated().any():
        duplicates = received.loc[received.epoch.duplicated(keep=False)]
        if (duplicates.groupby("epoch")[FIELDS[1:]].nunique(dropna=False) > 1).any().any():
            raise ValueError("Conflicting original candle duplicates")
        equal_duplicates = int(received.epoch.duplicated().sum())
    received = received.drop_duplicates("epoch").sort_values("epoch", kind="stable").reset_index(drop=True)
    if (not np.array_equal(received.epoch.to_numpy(), saved_epoch)
            or not np.array_equal(received[FIELDS[1:]].to_numpy(), saved_values[FIELDS[1:]].to_numpy())):
        raise ValueError("Normalized CSV differs from canonical original received prices")
    effective_start = int(saved_epoch[0]) if len(saved_epoch) else None
    effective_end = int(saved_epoch[-1] + 60) if len(saved_epoch) else None
    internal = _gaps(saved_epoch, effective_start, effective_end) if len(saved_epoch) else []
    requested = _gaps(saved_epoch, start, end)
    internal_off_grid = sum(effective_start <= value < effective_end for value in off_grid) if len(saved_epoch) else 0
    return {"status": "PASS", "source_file_sha256": row["requested_sha256"],
            "catalogue_binding_checked": catalogue_provenance is not None,
            "manifest_chain_length": len(chain), "original_page_references": references,
            "verified_unique_pages": len(seen_pages), "original_received_rows": original_rows,
            "excluded_provider_boundary_rows": boundary_rows, "excluded_counts": excluded,
            "equal_original_duplicates_removed": equal_duplicates, "canonical_rows": len(saved_epoch),
            "effective_start_epoch": effective_start, "effective_end_exclusive_epoch": effective_end,
            "requested_start_epoch": start, "requested_end_exclusive_epoch": end,
            "leading_boundary_loss_minutes": (effective_start - start) // 60 if len(saved_epoch) else (end - start) // 60,
            "trailing_boundary_loss_minutes": (end - effective_end) // 60 if len(saved_epoch) else 0,
            "requested_missing_minutes": sum(gap["missing_minutes"] for gap in requested),
            "requested_missing_ranges": requested,
            "internal_missing_minutes": sum(gap["missing_minutes"] for gap in internal),
            "internal_missing_ranges": internal, "internal_invalid_rows": internal_off_grid,
            "off_grid_inside_effective_interval": internal_off_grid,
            "off_grid_outside_effective_interval": len(off_grid) - internal_off_grid,
            "effective_interval_rule": "first canonical received M1 through last canonical received M1 plus 60 seconds",
            "gap_free_segment_selected": False, "interpolations": False, "performance_accessed": False,
            "safety": dict(SAFETY)}
