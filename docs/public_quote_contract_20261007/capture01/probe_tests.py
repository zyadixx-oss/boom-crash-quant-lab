"""Public probe safety/schema tests use fabricated text messages only."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("public_quote_probe", REPO / "scripts/probe_public_quote_contract.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def safety(monkeypatch):
    for name in probe.SAFETY:
        monkeypatch.setenv(name, "false")


def message(symbol="BOOM500", epoch=100, **overrides):
    tick = {"symbol": symbol, "epoch": epoch, "quote": 100.001,
            "bid": 100.0, "ask": 100.002}
    tick.update(overrides)
    return json.dumps({"msg_type": "tick", "req_id": probe.SYMBOLS.index(symbol) + 1, "tick": tick})


class Socket:
    def __init__(self, messages, send_error=None, close_error=None):
        self.messages = list(messages)
        self.sent = []
        self.closed = False
        self.send_error = send_error
        self.close_error = close_error

    async def send(self, text):
        if self.send_error:
            raise self.send_error
        self.sent.append(json.loads(text))

    async def recv(self):
        if not self.messages:
            raise TimeoutError("Synthetic receive timeout")
        value = self.messages.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def run(tmp_path, messages, **kwargs):
    socket = Socket(messages, kwargs.pop("send_error", None), kwargs.pop("close_error", None))

    async def connect(endpoint, **options):
        assert endpoint == probe.ENDPOINT
        assert options["max_size"] == 65536
        assert options["max_queue"] == 16
        return socket

    output = tmp_path / "run"
    result = asyncio.run(probe.probe(output, samples=kwargs.pop("samples", 1), connector=connect, **kwargs))
    return result, socket, output


@pytest.mark.parametrize("flag", probe.SAFETY)
def test_any_enabled_gate_blocks_before_artifact_or_network(tmp_path, monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    with pytest.raises(probe.ContractError, match=flag):
        asyncio.run(probe.probe(tmp_path / "run"))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("payload", [
    {"authorize": "fake"}, {"buy": "fake"}, {"balance": 1},
    {"ticks": "BOOM500", "subscribe": 1, "req_id": 1, "authorize": "fake"},
    {"ticks": "BOOM500", "subscribe": 0, "req_id": 1},
    {"ticks": "BOOM500", "subscribe": True, "req_id": 1},
    {"ticks": "BOOM500", "subscribe": 1, "req_id": True},
    {"ticks": "BOOM600", "subscribe": 1, "req_id": 1},
    {"ticks": "R_100", "subscribe": 1, "req_id": 1},
])
def test_exact_allowlist_rejects_other_calls_and_bool_aliases(payload):
    with pytest.raises(probe.ContractError):
        probe.validate_request(payload)


def test_decimal_sides_preserve_schema_and_order():
    row = probe.validate_response(message())
    assert row["quote"] == "100.001"
    assert row["public_feed_spread_price_units"] == "0.002"
    assert row["quote_within_bid_ask"] is True


@pytest.mark.parametrize("missing", ["bid", "ask", "both"])
def test_missing_or_null_sides_are_unknown(missing):
    obj = json.loads(message())
    if missing in ("bid", "both"):
        del obj["tick"]["bid"]
    if missing in ("ask", "both"):
        obj["tick"]["ask"] = None
    row = probe.validate_response(json.dumps(obj))
    assert row["bid_ask_status"] == "UNKNOWN"
    assert row["quote_within_bid_ask"] is None
    assert row["public_feed_spread_price_units"] is None


@pytest.mark.parametrize("change", [
    {"quote": True}, {"quote": "100.001"}, {"quote": None}, {"quote": float("nan")},
    {"ask": float("inf")}, {"bid": -1}, {"epoch": True}, {"epoch": 100.1},
    {"epoch": 0}, {"bid": 101}, {"ask": 99}, {"quote": 101},
])
def test_malformed_quotes_fail_closed(change):
    with pytest.raises(probe.ContractError):
        probe.validate_response(message(**change))


def test_duplicate_json_key_and_response_id_mismatch_rejected():
    with pytest.raises(probe.ContractError, match="Duplicate"):
        probe.validate_response('{"msg_type":"tick","msg_type":"history"}')
    obj = json.loads(message())
    obj["req_id"] = 2
    with pytest.raises(probe.ContractError, match="mismatch"):
        probe.validate_response(json.dumps(obj))


def test_success_is_bounded_pinned_and_socket_closed(tmp_path):
    msgs = [message(s, epoch=e) for e in (100, 101) for s in probe.SYMBOLS]
    result, socket, output = run(tmp_path, msgs, samples=2)
    assert result["status"] == "PASS"
    assert socket.closed and result["connection_closed"]
    assert result["received_messages"] == 8
    assert all(x["distinct_epochs"] == x["with_bid_ask"] == 2 for x in result["counts"].values())
    assert len(socket.sent) == 4
    for request in socket.sent:
        probe.validate_request(request)
    for filename, expected in result["artifact_sha256"].items():
        assert probe.sha((output / filename).read_bytes()) == expected
    raw = [json.loads(line) for line in (output / "received_text_messages.jsonl").read_text().splitlines()]
    assert [entry["text"] for entry in raw] == msgs
    assert all(not result[key] for key in ("executable_cfd_verified", "actual_fills_verified",
                                           "broker_costs_verified", "profit_tested"))


def test_equal_duplicates_do_not_reach_sample_target(tmp_path):
    msgs = [message(), message()] + [message(s) for s in probe.SYMBOLS[1:]]
    result, socket, _ = run(tmp_path, msgs)
    assert result["status"] == "PASS"
    assert result["counts"]["BOOM500"]["duplicate_epochs"] == 1
    assert result["counts"]["BOOM500"]["distinct_epochs"] == 1
    assert socket.closed


def test_unknown_optional_fields_counted_without_fabricating_spread(tmp_path):
    result, _, _ = run(tmp_path, [message(s, bid=None) for s in probe.SYMBOLS])
    assert result["status"] == "PASS"
    assert all(x["unknown_bid_ask"] == 1 and x["with_bid_ask"] == 0 for x in result["counts"].values())


@pytest.mark.parametrize("bad", [
    message(quote=101), "{bad json", '{"msg_type":"tick","error":{"code":"Oops"}}', b"binary",
])
def test_raw_failures_are_retained_and_connection_closed(tmp_path, bad):
    result, socket, output = run(tmp_path, [bad])
    assert result["status"] == "FAIL" and result["error"]
    assert socket.closed
    record = json.loads((output / "received_text_messages.jsonl").read_text())
    assert record["payload_sha256"] == probe.sha(bad.encode() if isinstance(bad, str) else bad)
    assert (output / "result.sha256").exists()


@pytest.mark.parametrize("second", [message(quote=100.002), message(epoch=99)])
def test_conflicting_or_out_of_order_epoch_fails(tmp_path, second):
    result, socket, _ = run(tmp_path, [message(), second], samples=2)
    assert result["status"] == "FAIL"
    assert socket.closed
    assert result["received_messages"] == 2


def test_receive_timeout_preserves_partial_sample_and_closes(tmp_path):
    result, socket, _ = run(tmp_path, [message()], samples=2)
    assert result["status"] == "FAIL"
    assert result["error"]["type"] == "TimeoutError"
    assert result["counts"]["BOOM500"]["distinct_epochs"] == 1
    assert socket.closed


def test_send_failure_records_attempt_and_closes(tmp_path):
    result, socket, output = run(tmp_path, [], send_error=OSError("synthetic send failure"))
    assert result["status"] == "FAIL"
    assert socket.closed
    records = [json.loads(x) for x in (output / "requests.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["event"] == "send_attempt"


def test_close_failure_cannot_be_pass(tmp_path):
    result, socket, _ = run(tmp_path, [message(s) for s in probe.SYMBOLS], close_error=OSError("close"))
    assert result["status"] == "FAIL"
    assert result["close_error"] and not result["connection_closed"]
    assert socket.closed


def test_existing_run_cannot_be_overwritten(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "result.json").write_text("preserved")
    with pytest.raises(FileExistsError):
        asyncio.run(probe.probe(output))
    assert (output / "result.json").read_text() == "preserved"


@pytest.mark.parametrize("kwargs", [{"samples": 121}, {"samples": True}, {"duration": 301},
                                    {"receive_timeout": 31}])
def test_unbounded_configuration_rejected_before_artifacts(tmp_path, kwargs):
    with pytest.raises(probe.ContractError):
        asyncio.run(probe.probe(tmp_path / "run", **kwargs))
    assert not (tmp_path / "run").exists()


def test_frame_cap_terminates_duplicate_flood(tmp_path):
    result, socket, _ = run(tmp_path, [message()] * 60)
    assert result["status"] == "FAIL"
    assert result["received_messages"] == 40
    assert "cap" in result["error"]["message"]
    assert socket.closed


def test_overall_deadline_bounds_endless_receive(tmp_path, monkeypatch):
    # Advance only the probe clock; asyncio's own monotonic clock is untouched.
    class Clock:
        value = 0
        def monotonic(self):
            self.value += 0.3
            return self.value
    monkeypatch.setattr(probe, "time", Clock())
    result, socket, _ = run(tmp_path, [], duration=1)
    assert result["status"] == "FAIL"
    assert result["error"]["type"] == "TimeoutError"
    assert "deadline" in result["error"]["message"]
    assert socket.closed


@pytest.mark.parametrize("changes", [{"ask": None, "bid": 101}, {"bid": None, "ask": 99}])
def test_partial_side_still_must_respect_known_quote_order(changes):
    with pytest.raises(probe.ContractError, match="ordering"):
        probe.validate_response(message(**changes))


def test_spread_summary_uses_distinct_epochs_and_preserves_units(tmp_path):
    msgs = [message(s, epoch=100) for s in probe.SYMBOLS]
    msgs += [message(s, epoch=101, bid=99.999, ask=100.003) for s in probe.SYMBOLS]
    result, _, _ = run(tmp_path, msgs, samples=2)
    assert result["status"] == "PASS"
    summary = result["counts"]["BOOM500"]["public_feed_spread_summary_price_units"]
    assert summary["observations"] == 2
    assert (summary["minimum"], summary["median"], summary["maximum"]) == ("0.002", "0.003", "0.004")
