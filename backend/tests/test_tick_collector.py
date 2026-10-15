"""Engineering checks use fabricated quotes only; never public/history data."""
import asyncio
import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tick_collector", REPO / "scripts/collect_spike_ticks.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
DATE = "2026-04-11"
START, END = collector.date_bounds(DATE)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    for flag in collector.SAFETY:
        monkeypatch.setenv(flag, "false")

    async def no_sleep(_):
        pass

    monkeypatch.setattr(collector.asyncio, "sleep", no_sleep)
    return tmp_path


def payload(end=END-1, req_id=1000, count=2):
    return {"ticks_history": "BOOM600", "count": count, "start": START,
            "end": end, "style": "ticks", "req_id": req_id}


def response(request, epochs, prices=None):
    return {"msg_type": "history", "req_id": request["req_id"],
            "history": {"times": epochs, "prices": prices if prices is not None else [100.001] * len(epochs)}}


class FakeClient:
    def __init__(self, epochs=None, events=None, full=False):
        self.epochs = epochs or [START, START+1, END-1]
        self.events = list(events or [])
        self.requests = []
        self.resets = 0
        self.full = full

    async def request(self, request):
        self.requests.append(dict(request))
        collector.validate_request(request)
        if self.events:
            event = self.events.pop(0)
            if isinstance(event, BaseException):
                raise event
            if callable(event):
                return json.dumps(event(request))
        epochs = (list(range(max(request["start"], request["end"]-request["count"]+1), request["end"]+1))
                  if self.full else [t for t in self.epochs if request["start"] <= t <= request["end"]][-request["count"]:])
        return json.dumps(response(request, epochs))

    async def reset(self):
        self.resets += 1


def run(client, directory, resume=False, config="fixed", page_size=2):
    return asyncio.run(collector.collect_day(client, directory, "BOOM600", DATE, config,
                                            page_size=page_size, resume=resume))


@pytest.mark.parametrize("flag", list(collector.SAFETY))
def test_all_four_flags_fail_closed(flag, sandbox, monkeypatch):
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match=flag):
        collector.safety_guard()


@pytest.mark.parametrize("change", [
    {"authorize": "prohibited"}, {"adjust_start_time": 1}, {"subscribe": 1},
    {"count": 1001}, {"count": True}, {"style": "candles"}, {"ticks_history": "BOOM500"}, {"end": START-1},
])
def test_public_request_whitelist_and_verified_limit(change):
    request = {**payload(), **change}
    with pytest.raises(collector.SemanticError):
        collector.validate_request(request)


def test_start_is_mandatory_and_active_only_request_is_permitted():
    request = payload(); del request["start"]
    with pytest.raises(collector.SemanticError):
        collector.validate_request(request)
    collector.validate_request({"active_symbols": "brief", "req_id": 1})


@pytest.mark.parametrize("epochs,prices", [
    ([END], [100.]), ([START-1], [100.]), ([START+1, START], [100., 100.]),
    ([START], []), ([START+.5], [100.]), ([START], ["NaN"]), ([START], [True]), ([START], [-1.]),
])
def test_tick_arrays_reject_adjusted_unordered_or_invalid_data(epochs, prices):
    with pytest.raises(collector.SemanticError):
        collector.tick_rows(response(payload(), epochs, prices), payload())


def test_rate_limit_error_is_retryable_even_with_error_message_type():
    value = {"msg_type": "ticks_history", "req_id": 1000,
             "error": {"code": "RateLimit", "message": "slow down"}}
    with pytest.raises(collector.RateLimitError):
        collector.tick_rows(value, payload())


def test_unmatched_response_and_other_server_errors_rejected():
    value = response(payload(), [START]); value["req_id"] = 999
    with pytest.raises(collector.SemanticError):
        collector.tick_rows(value, payload())
    with pytest.raises(collector.SemanticError):
        collector.tick_rows({"req_id": 1000, "error": {"code": "NoHistory", "message": "none"}}, payload())


def test_equal_duplicates_preserved_in_raw_and_removed_only_in_clean():
    values = {}
    assert collector.insert_rows(values, [(START, "100.001"), (START, "100.0010")]) == 1
    assert values == {START: "100.001"}
    with pytest.raises(collector.SemanticError, match="Unequal duplicate"):
        collector.insert_rows(values, [(START, "100.002")])


def test_missing_one_second_and_boundary_gaps_remain_unknown():
    assert collector.missing_ranges([2, 4], 0, 6) == [
        {"start_epoch": 0, "end_exclusive_epoch": 2, "missing_seconds": 2},
        {"start_epoch": 3, "end_exclusive_epoch": 4, "missing_seconds": 1},
        {"start_epoch": 5, "end_exclusive_epoch": 6, "missing_seconds": 1}]


def test_day_pagination_preserves_gaps_wire_hashes_and_portable_manifest(sandbox):
    client = FakeClient(); manifest = run(client, sandbox)
    assert [x["end"] for x in client.requests] == [END-1, START]
    assert all(x["start"] == START for x in client.requests)
    assert manifest["rows"] == 3 and manifest["missing_seconds"] == 86397
    assert manifest["cadence_seconds"] == 1 and manifest["gap_free"] is False
    assert manifest["fills_or_interpolations"] is False and manifest["safety"] == collector.SAFETY
    files = collector.files_for(sandbox, "BOOM600", DATE)
    wire_pages = collector.lines(files["raw"])
    assert all(collector.hashlib.sha256(x["response_wire"].encode()).hexdigest() == x["wire_sha256"] for x in wire_pages)
    assert not Path(manifest["clean_file"]).is_absolute()
    with files["clean"].open() as stream:
        assert [int(x["epoch"]) for x in csv.DictReader(stream)] == [START, START+1, END-1]


def test_complete_fabricated_day_has_86400_known_seconds(sandbox):
    manifest = run(FakeClient(full=True), sandbox, page_size=1000)
    assert manifest["rows"] == 86400 and manifest["missing_seconds"] == 0
    assert manifest["gap_free"] is True and manifest["successful_page_count"] == 87


def test_rate_limit_and_transport_retries_keep_exact_requests_and_lineage(sandbox):
    def limited(request):
        return {"msg_type": "ticks_history", "req_id": request["req_id"],
                "error": {"code": "RateLimit", "message": "slow down"}}
    client = FakeClient(events=[limited, OSError("transport interrupted")])
    manifest = run(client, sandbox)
    assert client.requests[0] == client.requests[1] == client.requests[2]
    assert len(manifest["request_errors"]) == len(manifest["recovered_retry_lineage"]) == 2
    pages = collector.lines(collector.files_for(sandbox, "BOOM600", DATE)["audit"])
    assert [x["accepted"] for x in pages[:3]] == [False, False, True]
    assert all(x["successful_page_audit_index"] == 2 for x in manifest["recovered_retry_lineage"])


def test_semantic_out_of_bounds_failure_preserved_and_cannot_resume(sandbox):
    client = FakeClient(events=[lambda request: response(request, [END])])
    with pytest.raises(collector.SemanticError):
        run(client, sandbox)
    files = collector.files_for(sandbox, "BOOM600", DATE)
    assert len(client.requests) == 1 and collector.read_json(files["checkpoint"])["semantic_failure"] is True
    assert collector.lines(files["raw"])[0]["response_wire"] and not files["manifest"].exists()
    with pytest.raises(collector.SemanticError, match="semantic failure"):
        run(FakeClient(), sandbox, resume=True)


def test_transport_retry_exhaustion_preserved_without_manifest(sandbox):
    client = FakeClient(events=[OSError("down")] * 4)
    with pytest.raises(collector.SemanticError, match="stopped"):
        run(client, sandbox)
    files = collector.files_for(sandbox, "BOOM600", DATE)
    assert len(collector.lines(files["raw"])) == 4 and not files["manifest"].exists()
    resumed = FakeClient(); manifest = run(resumed, sandbox, resume=True)
    assert resumed.requests[0] == client.requests[0]
    assert len(manifest["recovered_retry_lineage"]) == 4


def test_interrupted_checkpoint_resumes_only_identical_verified_bytes(sandbox):
    client = FakeClient(events=[None, asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        run(client, sandbox)
    files = collector.files_for(sandbox, "BOOM600", DATE); original = files["raw"].read_bytes()
    with pytest.raises(collector.SemanticError, match="config_sha256"):
        run(FakeClient(), sandbox, resume=True, config="changed")
    resumed = FakeClient(); manifest = run(resumed, sandbox, resume=True)
    assert resumed.requests[0]["end"] == START
    assert files["raw"].read_bytes().startswith(original) and manifest["rows"] == 3


def test_resume_rejects_raw_tampering(sandbox):
    client = FakeClient(events=[None, asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        run(client, sandbox)
    path = collector.files_for(sandbox, "BOOM600", DATE)["raw"]
    path.write_text(path.read_text()+"\n")
    with pytest.raises(collector.SemanticError, match="bytes differ"):
        run(FakeClient(), sandbox, resume=True)


def test_resume_rejects_changed_page_size_even_with_same_config_identifier(sandbox):
    client = FakeClient(events=[None, asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        run(client, sandbox)
    with pytest.raises(collector.SemanticError, match="page_size"):
        run(FakeClient(), sandbox, resume=True, page_size=1)


def test_resume_rejects_checkpoint_error_removal(sandbox):
    client = FakeClient(events=[OSError("down")] * 4)
    with pytest.raises(collector.SemanticError):
        run(client, sandbox)
    files = collector.files_for(sandbox, "BOOM600", DATE)
    state = collector.read_json(files["checkpoint"]); state["request_errors"] = []
    collector.write_json(files["checkpoint"], state)
    with pytest.raises(collector.SemanticError, match="retained request errors"):
        run(FakeClient(), sandbox, resume=True)


def test_completed_day_immutable_resume_skips_calls_and_detects_clean_tampering(sandbox):
    manifest = run(FakeClient(), sandbox)
    with pytest.raises(FileExistsError):
        run(FakeClient(), sandbox)
    idle = FakeClient(); assert run(idle, sandbox, resume=True) == manifest and idle.requests == []
    path = collector.files_for(sandbox, "BOOM600", DATE)["clean"]; path.write_text(path.read_text()+"\n")
    with pytest.raises(collector.SemanticError, match="bytes changed"):
        run(FakeClient(), sandbox, resume=True)


def test_empty_termination_marks_whole_day_unknown_without_filling(sandbox):
    client = FakeClient(events=[lambda request: response(request, [])])
    manifest = run(client, sandbox)
    assert manifest["rows"] == 0 and manifest["missing_seconds"] == 86400 and not manifest["gap_free"]
    assert manifest["gaps"] == [{"start_epoch": START, "end_exclusive_epoch": END, "missing_seconds": 86400}]


def test_unmatched_error_cannot_be_declared_recovered():
    with pytest.raises(collector.SemanticError, match="no matching"):
        collector.recovery_lineage([{"request_key": "not-successful"}], [])


def test_symbol_probe_retry_preserves_wire_and_never_authorizes(sandbox):
    class SymbolClient(FakeClient):
        async def request(self, request):
            self.requests.append(dict(request)); collector.validate_request(request)
            if len(self.requests) == 1:
                return json.dumps({"req_id": 1, "error": {"code": "RateLimit", "message": "slow"}})
            return json.dumps({"req_id": 1, "active_symbols": [{"underlying_symbol": s} for s in collector.SYMBOLS]})
    client = SymbolClient(); lineage = asyncio.run(collector.bootstrap(client, sandbox))
    assert client.requests == [{"active_symbols": "brief", "req_id": 1}] * 2
    assert len(lineage["request_errors"]) == 1 and lineage["successful_wire_sha256"]
    assert len(collector.lines(sandbox / "active_symbols_wire.jsonl")) == 2


def test_canonical_date_and_portable_path_contract(sandbox):
    with pytest.raises(ValueError):
        collector.date_bounds("2026-4-11")
    with pytest.raises(ValueError):
        collector.portable(sandbox.parent / "outside")
