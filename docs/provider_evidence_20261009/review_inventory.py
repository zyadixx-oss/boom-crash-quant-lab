"""Reproduce a descriptive inventory of the already observed public capture.

This is not a new predictive test, cost model, historical backtest or proof that
all provider endpoints lack additional fields. No network/account/order calls.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/public_quote_contract_20261007/capture01/received_text_messages.jsonl"
SOURCE_SHA = "afc224883af67d1730ec09de99d3655ecabe7ea4cf6d63fd0a44cdbc9b59b298"
FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")


def main(output):
    for flag in FLAGS:
        if os.environ.get(flag, "false").lower() not in ("false", "0", "no", "off", ""):
            raise ValueError("Live gate must be false: " + flag)
    payload = SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError("Exact previously reviewed capture bytes required")
    groups = defaultdict(list)
    for line in payload.decode().splitlines():
        envelope = json.loads(line)
        text = envelope["text"]
        if envelope["payload_type"] != "text" or hashlib.sha256(text.encode()).hexdigest() != envelope["payload_sha256"]:
            raise ValueError("Original text-message integrity changed")
        obj = json.loads(text, parse_float=Decimal)
        if obj["msg_type"] != "tick":
            raise ValueError("Exact saved tick population required")
        tick = obj["tick"]
        if tick["id"] != obj["subscription"]["id"]:
            raise ValueError("Observed tick/subscription identity changed")
        groups[tick["symbol"]].append(tick)
    if set(groups) != {"BOOM500", "CRASH500", "BOOM600", "CRASH600"} or any(len(rows) != 60 for rows in groups.values()):
        raise ValueError("All240 messages from the exact four-symbol capture required")
    symbols = {}
    half = Decimal("0.00000675")
    for symbol, rows in sorted(groups.items()):
        ratios, matches, keys = [], 0, set()
        for tick in rows:
            keys.add(tuple(sorted(tick)))
            quote, bid, ask = (Decimal(tick[name]) for name in ("quote", "bid", "ask"))
            if not 0 < bid <= quote <= ask:
                raise ValueError("Original public-side relationship changed")
            step = Decimal(10) ** (-tick["pip_size"])
            ratios.append((ask - bid) / quote)
            matches += ((quote * (1 - half)).quantize(step, rounding=ROUND_HALF_EVEN) == bid
                and (quote * (1 + half)).quantize(step, rounding=ROUND_HALF_EVEN) == ask)
        epochs = [tick["epoch"] for tick in rows]
        if epochs != list(range(epochs[0], epochs[0] + 60)):
            raise ValueError("Original per-symbol contiguous UTC clock changed")
        symbols[symbol] = {"messages": len(rows), "observed_tick_key_sets": [list(k) for k in sorted(keys)],
            "distinct_subscription_ids": len({tick["id"] for tick in rows}),
            "first_epoch": epochs[0], "last_epoch": epochs[-1],
            "spread_over_quote_min": str(min(ratios)), "spread_over_quote_max": str(max(ratios)),
            "posthoc_fixed_half_markup_exact_matches": matches}
    report = {"stage": "descriptive_saved_public_field_inventory", "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": SOURCE_SHA,
        "review_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "messages": 240, "symbols": symbols, "posthoc_fixed_half_markup": str(half),
        "rounding_probe_chosen_after_capture_observed": True,
        "rounding_probe_is_predictive_test": False, "all_public_endpoints_exhaustively_reviewed": False,
        "new_predictor_established": False, "universal_profit_impossibility_proven": False,
        "executable_CFD_mapping_verified": False, "historical_costs_verified": False,
        "profit_tested_by_this_inventory": False, "QUALIFIED": False,
        "safety": dict.fromkeys(FLAGS, False)}
    path = Path(output).resolve()
    path.relative_to(Path(__file__).resolve().parent)
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path.relative_to(ROOT)), "messages": 240,
        "exact_rule_matches": {s: v["posthoc_fixed_half_markup_exact_matches"] for s, v in symbols.items()},
        "new_predictor_established": False, "profit_tested": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).with_name("inventory.json")))
    main(parser.parse_args().output)
