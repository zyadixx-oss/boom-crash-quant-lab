"""Independent stdlib review of the one captured 240-message artifact, not CFD fills."""
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parent
CAPTURE = BASE / "capture01"
SYMBOLS = ["BOOM500", "CRASH500", "BOOM600", "CRASH600"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    result = json.loads((CAPTURE / "result.json").read_text())
    declaration = json.loads((CAPTURE / "declaration.json").read_text())
    assert result["status"] == "PASS" and result["connection_closed"] is True
    assert result["error"] is None and result["close_error"] is None
    assert digest(CAPTURE / "result.json") == (CAPTURE / "result.sha256").read_text().split()[0]
    for name, expected in result["artifact_sha256"].items():
        assert Path(name).name == name
        assert digest(CAPTURE / name) == expected
    for document in (declaration, result):
        assert all(document["safety"][name] is False for name in (
            "LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"))
        assert document["authentication_used"] is False
        assert document["executable_cfd_verified"] is False and document["profit_tested"] is False
    requests = [json.loads(line) for line in (CAPTURE / "requests.jsonl").read_text().splitlines()]
    assert len(requests) == 8
    for index, symbol in enumerate(SYMBOLS, 1):
        sent, completed = requests[2 * (index - 1):2 * index]
        expected = {"ticks": symbol, "subscribe": 1, "req_id": index}
        assert sent["event"] == "send_attempt" and sent["request"] == expected
        assert json.loads(sent["text"]) == expected
        assert hashlib.sha256(sent["text"].encode()).hexdigest() == sent["text_sha256"]
        assert completed["event"] == "send_completed" and completed["req_id"] == index
    raw = [json.loads(line) for line in (CAPTURE / "received_text_messages.jsonl").read_text().splitlines()]
    observed = [json.loads(line) for line in (CAPTURE / "observations.jsonl").read_text().splitlines()]
    assert len(raw) == len(observed) == result["received_messages"] == 240
    data = defaultdict(list)
    for index, (packet, saved) in enumerate(zip(raw, observed), 1):
        assert packet["frame_number"] == saved["frame_number"] == index
        assert packet["payload_type"] == "text" and saved["action"] == "sampled"
        assert hashlib.sha256(packet["text"].encode()).hexdigest() == packet["payload_sha256"]
        message = json.loads(packet["text"], parse_float=Decimal)
        assert message["msg_type"] == "tick" and "error" not in message
        tick = message["tick"]; symbol = tick["symbol"]
        assert symbol in SYMBOLS and message["req_id"] == SYMBOLS.index(symbol) + 1
        bid, quote, ask = (Decimal(tick[k]) for k in ("bid", "quote", "ask"))
        assert 0 < bid <= quote <= ask and all(x.is_finite() for x in (bid, quote, ask))
        assert saved["symbol"] == symbol and saved["epoch"] == tick["epoch"]
        for key, value in (("bid", bid), ("quote", quote), ("ask", ask),
                           ("public_feed_spread_price_units", ask - bid)):
            assert Decimal(saved[key]) == value
        data[symbol].append((tick["epoch"], ask - bid))
    summaries = {}
    for symbol in SYMBOLS:
        values = data[symbol]; epochs = [row[0] for row in values]
        assert len(values) == len(set(epochs)) == 60
        assert epochs == list(range(epochs[0], epochs[0] + 60))
        spreads = [row[1] for row in values]
        stats = {"observations": 60, "minimum": str(min(spreads)),
                 "median": str(median(spreads)), "maximum": str(max(spreads))}
        saved = result["counts"][symbol]
        assert saved["distinct_epochs"] == saved["with_bid_ask"] == 60
        assert saved["missing_seconds_within_observed_span"] == 0
        for name, value in stats.items():
            assert saved["public_feed_spread_summary_price_units"][name] == value
        summaries[symbol] = stats
    return {"status": "PASS", "raw_text_messages": 240, "public_subscriptions": 4,
            "source_sha256": digest(Path(__file__)), "capture_result_sha256": digest(CAPTURE / "result.json"),
            "summaries_price_units": summaries, "all_four_flags_false": True,
            "scope": "specific saved public-feed requests/messages/schema/spread arithmetic only",
            "executable_CFD_or_historical_cost_or_profit_verified": False}


if __name__ == "__main__":
    report = review()
    print(json.dumps(report, indent=2))
