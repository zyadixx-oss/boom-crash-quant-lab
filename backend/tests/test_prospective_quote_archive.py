"""Fabricated transport/schema/integrity tests only; no market-data requests."""
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/prospective_quote_archive.py"
spec = importlib.util.spec_from_file_location("prospective_quote_archive_tests", SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1_000_000_000)

    def utc(self):
        return datetime.fromtimestamp(1791800000 + self.now, timezone.utc).isoformat()

    async def sleep(self, seconds):
        self.now += seconds


def tick(symbol="BOOM500", epoch=100, **changes):
    value = {"symbol": symbol, "epoch": epoch, "quote": 100.001,
             "bid": 100.0, "ask": 100.002}
    value.update(changes)
    req_id = archive._parser.SYMBOLS.index(symbol) + 1
    return json.dumps({"msg_type": "tick", "req_id": req_id, "tick": value})


class Socket:
    def __init__(self, clock, events, deadline, *, step=1, fail_send=False, fail_close=False):
        self.clock, self.events, self.deadline = clock, list(events), deadline
        self.step, self.fail_send, self.fail_close = step, fail_send, fail_close
        self.sent, self.closed = [], False

    async def send(self, text):
        value = json.loads(text)
        archive.validate_request(value)
        self.sent.append(value)
        if self.fail_send:
            raise OSError("Fabricated interrupted send")

    async def recv(self):
        if not self.events:
            self.clock.now = max(self.clock.now, self.deadline)
            raise TimeoutError("Fabricated bounded deadline")
        self.clock.now += self.step
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self):
        self.closed = True
        if self.fail_close:
            raise OSError("Fabricated close warning")


class Connector:
    def __init__(self, events):
        self.events, self.calls = list(events), []

    async def __call__(self, endpoint, **options):
        assert endpoint == archive.ENDPOINT
        assert options["ssl"].check_hostname is True
        assert options["max_size"] == 65536
        assert options["max_queue"] == 16
        self.calls.append((endpoint, options))
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


@pytest.fixture(autouse=True)
def safe(monkeypatch):
    for name in archive.SAFETY:
        monkeypatch.setenv(name, "false")


def run(output, events=None, *, duration=4, resume=False, clock=None,
        sockets=None, max_reconnects=3, step=1):
    clock = clock or Clock()
    if events is None:
        events = [tick(), tick("CRASH500"), tick(epoch=101), tick("CRASH500", 101)]
    sockets = sockets or [Socket(clock, events, duration, step=step)]
    connector = Connector(sockets)
    result = asyncio.run(archive.archive(output, duration=duration, resume=resume,
        max_reconnects=max_reconnects, connector=connector, clock=clock, sleep=clock.sleep))
    return result, connector, sockets


def read(path):
    return json.loads(path.read_text())


def reseal(output, name):
    """A stronger corruption test: update retained hashes, leaving derived truth wrong."""
    session = output / "session_000001"
    receipt = read(session / "receipt.json")
    data = (session / name).read_bytes()
    receipt["files"][name] = {"sha256": archive.digest(data), "bytes": len(data)}
    (session / "receipt.json").write_bytes(archive.encoded(receipt))
    index = read(output / "receipts_index.json")
    index["sessions"][0]["receipt_sha256"] = archive.digest((session / "receipt.json").read_bytes())
    (output / "receipts_index.json").write_bytes(archive.encoded(index))


@pytest.mark.parametrize("name", list(archive.SAFETY))
@pytest.mark.parametrize("value", ["true", "1", "yes", ""])
def test_all_eight_gates_block_before_files_or_network(tmp_path, monkeypatch, name, value):
    output = tmp_path / "blocked"
    monkeypatch.setenv(name, value)
    with pytest.raises(archive.ArchiveError, match=name):
        run(output)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("duration", 0), ("duration", 901), ("duration", True), ("duration", 60.0),
    ("receive_timeout", 0), ("receive_timeout", 31),
    ("max_reconnects", -1), ("max_reconnects", 4),
])
def test_invalid_bounds_fail_before_files_or_network(tmp_path, field, value):
    with pytest.raises(archive.ArchiveError):
        asyncio.run(archive.archive(tmp_path / "blocked", **{field: value}))
    assert not (tmp_path / "blocked").exists()


def test_default_is_60_seconds_and_maximum_is_900():
    assert archive.configuration()["duration_seconds"] == 60
    assert archive.configuration(900)["duration_seconds"] == 900


@pytest.mark.parametrize("payload", [
    {"authorize": "fake"}, {"buy": "fake"}, {"balance": 1},
    {"ticks": "BOOM600", "subscribe": 1, "req_id": 3},
    {"ticks": "BOOM500", "subscribe": 0, "req_id": 1},
    {"ticks": "BOOM500", "subscribe": 1, "req_id": True},
    {"ticks": "BOOM500", "subscribe": 1, "req_id": 2},
    {"ticks": "BOOM500", "subscribe": 1, "req_id": 1, "authorize": "fake"},
])
def test_public_requests_are_exact_allowlist_only(payload):
    with pytest.raises(ValueError):
        archive.validate_request(payload)


def test_archive_preserves_raw_and_reproduces_all_derived_records(tmp_path):
    result, connector, sockets = run(tmp_path)
    assert result["status"] == "COMPLETED"
    assert result["received_frames"] == 4 and result["elapsed_seconds"] == 4
    assert len(connector.calls) == 1 and sockets[0].closed
    assert sockets[0].sent == [{"ticks": s, "subscribe": 1, "req_id": i} for i, s in enumerate(archive.SYMBOLS, 1)]
    assert all(x["distinct_epochs"] == x["with_bid_ask"] == 2 for x in result["counts"].values())
    raw = archive.lines(tmp_path / "session_000001/raw.jsonl")
    assert raw[0]["text"] == tick()
    assert raw[0]["payload_sha256"] == archive.digest(tick().encode())
    checked = archive.verify_archive(tmp_path)
    assert checked["status"] == "VERIFIED" and len(checked["sessions"]) == 1
    assert all(value is False for value in checked["safety"].values())
    assert all(value is False for value in checked["limitations"].values())
    assert archive.owner_status(tmp_path)["lock_held"] is False


def test_exact_raw_is_fsynced_before_parser_runs(tmp_path, monkeypatch):
    original = archive.parse_tick
    observed = []

    def inspect(raw):
        saved = archive.lines(tmp_path / "session_000001/raw.jsonl")
        assert any(row["text"] == raw for row in saved)
        observed.append(raw)
        return original(raw)

    monkeypatch.setattr(archive, "parse_tick", inspect)
    result, _, _ = run(tmp_path)
    assert result["status"] == "COMPLETED" and len(observed) >= 4


@pytest.mark.parametrize("bad", [
    "{malformed", '{"msg_type":"tick","msg_type":"tick"}',
    tick(quote=float("nan")), tick(bid=101), tick(ask=99), tick(epoch=True),
    tick("BOOM600"), b"not a text frame",
])
def test_semantic_failure_keeps_raw_and_never_reconnects(tmp_path, bad):
    result, connector, _ = run(tmp_path, [bad])
    assert result["status"] == "FAILED_PRESERVED"
    assert len(connector.calls) == 1 and result["semantic_failure"]["sequence"] == 1
    raw = archive.lines(tmp_path / "session_000001/raw.jsonl")
    assert len(raw) == 1 and not archive.lines(tmp_path / "session_000001/observations.jsonl")
    assert archive.verify_archive(tmp_path)["sessions"][0]["status"] == "FAILED_PRESERVED"


def test_null_missing_sides_remain_unknown(tmp_path):
    result, _, _ = run(tmp_path, [tick(bid=None), tick("CRASH500", ask=None)])
    assert result["status"] == "COMPLETED"
    for value in result["counts"].values():
        assert value["unknown_bid_ask"] == 1 and value["with_bid_ask"] == 0
        assert value["observed_public_spread_min"] is None
    observed = archive.lines(tmp_path / "session_000001/observations.jsonl")
    assert all(row["public_feed_spread_price_units"] is None for row in observed)


def test_reconnect_retains_separate_connection_identity_and_unknown_gap(tmp_path):
    clock = Clock()
    one = Socket(clock, [tick(), tick("CRASH500"), OSError("Fabricated disconnect")], 8)
    two = Socket(clock, [tick(epoch=104), tick("CRASH500", 104), tick(epoch=105)], 8)
    result, connector, _ = run(tmp_path, duration=8, clock=clock, sockets=[one, two])
    assert result["status"] == "COMPLETED" and len(connector.calls) == 2
    assert one.closed and two.closed
    assert {row["connection_id"] for row in archive.lines(tmp_path / "session_000001/raw.jsonl")} == {1, 2}
    for count in result["counts"].values():
        assert count["missing_epoch_seconds_between_observed_ticks"] == 3
        assert count["gap_intervals"] == [{"start_epoch": 101, "end_exclusive_epoch": 104, "unknown_seconds": 3}]
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_failed_send_is_retained_before_new_connection_requests(tmp_path):
    clock = Clock()
    one = Socket(clock, [], 8, fail_send=True)
    two = Socket(clock, [tick(), tick("CRASH500")], 8)
    result, _, _ = run(tmp_path, duration=8, clock=clock, sockets=[one, two])
    assert result["status"] == "COMPLETED"
    requests = archive.lines(tmp_path / "session_000001/requests.jsonl")
    assert requests[0]["connection_id"] == 1 and requests[0]["event"] == "send_attempt"
    assert requests[1]["connection_id"] == 2 and requests[1]["event"] == "send_attempt"
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_transport_attempt_limit_is_real_and_no_data_is_invented(tmp_path):
    clock = Clock()
    result, connector, _ = run(tmp_path, duration=8, clock=clock,
        sockets=[OSError("Fabricated unavailable provider")] * 2, max_reconnects=1)
    assert result["status"] == "FAILED_PRESERVED" and len(connector.calls) == 2
    assert result["received_frames"] == 0 and result["semantic_failure"] is None
    assert all(x["first_epoch"] is None and x["distinct_epochs"] == 0 for x in result["counts"].values())
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_identical_duplicate_is_counted_but_conflict_is_fatal(tmp_path):
    result, _, _ = run(tmp_path, [tick(), tick("CRASH500"), tick(), tick(quote=100.0015)])
    assert result["status"] == "FAILED_PRESERVED"
    assert result["counts"]["BOOM500"]["distinct_epochs"] == 1
    assert result["counts"]["BOOM500"]["duplicate_epochs"] == 1
    assert result["semantic_failure"]["sequence"] == 4
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_out_of_order_epoch_is_fatal(tmp_path):
    result, _, _ = run(tmp_path, [tick(epoch=101), tick("CRASH500"), tick(epoch=100)])
    assert result["status"] == "FAILED_PRESERVED"
    assert "Out-of-order" in result["semantic_failure"]["message"]


def test_frame_cap_cannot_be_extended_by_busy_duplicate_stream(tmp_path):
    result, _, _ = run(tmp_path, [tick(), tick("CRASH500")] * 30, duration=1, step=0)
    assert result["status"] == "FAILED_PRESERVED" and result["received_frames"] == 40
    assert "bound reached" in result["error"]["message"]
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_payload_cap_stops_without_parsing_or_inventing_more_quotes(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "MAX_PAYLOAD_BYTES", 1)
    result, _, _ = run(tmp_path)
    assert result["status"] == "FAILED_PRESERVED" and result["received_frames"] == 1
    assert "bound reached" in result["error"]["message"]
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_cancellation_seals_a_failed_session_for_review(tmp_path):
    result, _, sockets = run(tmp_path, [tick(), asyncio.CancelledError()])
    assert result["status"] == "FAILED_PRESERVED" and result["error"]["type"] == "CancelledError"
    assert sockets[0].closed and archive.verify_archive(tmp_path)["status"] == "VERIFIED"


def test_kernel_lock_blocks_a_real_second_process(tmp_path):
    with archive.archive_lock(tmp_path):
        owner = archive.owner_status(tmp_path)
        assert owner["lock_held"] is True and owner["pid_exists"] is True
        code = "import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('a',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.archive_lock(Path(sys.argv[2])).__enter__()"
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(tmp_path)], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and "active kernel lock" in result.stderr
    assert archive.owner_status(tmp_path)["lock_held"] is False


def test_resume_verifies_old_bytes_and_adds_distinct_sealed_session(tmp_path):
    run(tmp_path)
    first = {p.name: p.read_bytes() for p in (tmp_path / "session_000001").iterdir()}
    result, _, _ = run(tmp_path, resume=True)
    assert result["session_id"] == "session_000002" and result["elapsed_seconds"] == 4
    assert {p.name: p.read_bytes() for p in (tmp_path / "session_000001").iterdir()} == first
    assert len(archive.verify_archive(tmp_path)["sessions"]) == 2


def test_existing_archive_requires_explicit_resume(tmp_path):
    run(tmp_path)
    with pytest.raises(archive.ArchiveError, match="--resume"):
        run(tmp_path)


def test_resume_rejects_changed_configuration_without_new_session(tmp_path):
    run(tmp_path)
    with pytest.raises(archive.ArchiveError, match="configuration/source"):
        run(tmp_path, duration=5, resume=True)
    assert not (tmp_path / "session_000002").exists()


@pytest.mark.parametrize("name", ["raw.jsonl", "observations.jsonl", "requests.jsonl", "events.jsonl", "declaration.json"])
def test_tampered_evidence_is_rejected_before_resume(tmp_path, name):
    run(tmp_path)
    path = tmp_path / "session_000001" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(archive.ArchiveError, match="hash/size"):
        run(tmp_path, resume=True)
    assert not (tmp_path / "session_000002").exists()


def test_rehashing_false_observation_does_not_bypass_reproduction(tmp_path):
    run(tmp_path)
    path = tmp_path / "session_000001/observations.jsonl"
    rows = archive.lines(path)
    rows[0]["gap_before_seconds"] = 999
    path.write_bytes(b"".join(archive.encoded(row) for row in rows))
    reseal(tmp_path, "observations.jsonl")
    with pytest.raises(archive.ArchiveError, match="reproduce"):
        archive.verify_archive(tmp_path)


def test_rehashing_illegal_request_does_not_bypass_allowlist(tmp_path):
    run(tmp_path)
    path = tmp_path / "session_000001/requests.jsonl"
    rows = archive.lines(path)
    rows[0]["request"]["authorize"] = "fabricated-prohibited"
    path.write_bytes(b"".join(archive.encoded(row) for row in rows))
    reseal(tmp_path, "requests.jsonl")
    with pytest.raises(ValueError, match="exact public ticks request"):
        archive.verify_archive(tmp_path)


def test_truncated_and_unsealed_session_is_preserved_and_blocks_resume(tmp_path):
    run(tmp_path)
    partial = tmp_path / "session_000002"
    partial.mkdir()
    (partial / "raw.jsonl").write_bytes(b'{"interrupted"')
    with pytest.raises(archive.ArchiveError, match="Unsealed session preserved"):
        run(tmp_path, resume=True)
    assert (partial / "raw.jsonl").read_bytes() == b'{"interrupted"'


def test_missing_archive_verify_does_not_create_directory(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(archive.ArchiveError, match="existing declared archive"):
        archive.verify_archive(missing)
    assert not missing.exists()


def test_pinned_parser_source_tamper_fails_offline(tmp_path):
    run(tmp_path)
    (tmp_path / "parser_source.py").write_text("# altered snapshot\n")
    with pytest.raises(archive.ArchiveError, match="source snapshot"):
        archive.verify_archive(tmp_path)


def test_close_failure_is_retained_and_blocks_complete_status(tmp_path):
    clock = Clock()
    socket = Socket(clock, [tick(), tick("CRASH500")], 4, fail_close=True)
    result, _, _ = run(tmp_path, clock=clock, sockets=[socket])
    assert result["status"] == "FAILED_PRESERVED"
    assert len(result["connection_close_errors"]) == 1
    events = archive.lines(tmp_path / "session_000001/events.jsonl")
    assert any(row["event"] == "connection_close_error" for row in events)
    assert all(value is False for value in result["limitations"].values())
    assert archive.verify_archive(tmp_path)["status"] == "VERIFIED"
