"""Small deterministic transport fixtures are unit tests, never market substitutes."""
import asyncio
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research_campaign import data


def candle(epoch, value=100.0):
    return dict(epoch=epoch, open=value, high=value + 2, low=value - 2, close=value + 1)


def replace_fixture_wire(tmp_path, response, provenance):
    raw = json.dumps(response).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "wire" / f"{digest}.json.gz"
    data._immutable(path, gzip.compress(raw, mtime=0))
    return {**provenance, "wire_sha256": digest, "wire_file": str(path),
            "file_sha256": data.sha256(path)}


def test_normalization_audits_boundaries_gaps_and_equal_duplicates():
    rows = [candle(120), candle(0), candle(120), candle(61), candle(240)]
    frame, quality = data.normalize(rows, 0, 240)
    assert frame.epoch.tolist() == [0, 120]
    assert quality["equal_duplicates_removed"] == 1
    assert quality["excluded_counts"] == {"off_grid": 1, "not_closed_before_cutoff": 1}
    assert quality["missing_ranges"] == [
        {"start_epoch": 60, "end_exclusive_epoch": 120, "missing_minutes": 1},
        {"start_epoch": 180, "end_exclusive_epoch": 240, "missing_minutes": 1}]
    assert quality["expected_minutes"] == 4
    assert quality["reliable_start_epoch"] == 0
    assert quality["reliable_cutoff_exclusive_epoch"] == 180
    assert quality["internal_missing_minutes"] == 1
    assert quality["missing_prefix_minutes"] == 0
    assert quality["missing_suffix_minutes"] == 1


@pytest.mark.parametrize("bad", [
    dict(close=float("nan")), dict(low=0), dict(open=110), dict(epoch=1.5), dict(epoch=float("inf"))])
def test_malformed_prices_or_epochs_fail_closed(bad):
    with pytest.raises(ValueError):
        data.normalize([{**candle(0), **bad}], 0, 60)


def test_conflicting_duplicate_fails():
    with pytest.raises(ValueError, match="Conflicting"):
        data.normalize([candle(0), candle(0, 101)], 0, 60)


def test_empty_history_is_explicit_missing_interval():
    frame, audit = data.normalize([], 0, 600)
    assert frame.empty
    assert audit["missing_minutes"] == 10


@pytest.mark.parametrize("name,seconds", data.FRAMES.items())
def test_all_requested_single_frames_have_exact_availability(name, seconds):
    frame = pd.DataFrame([candle(e, 100 + e / 60) for e in range(0, seconds * 2, 60)])
    bars = data.resample(frame, name)
    assert bars.epoch.tolist() == [0, seconds]
    assert bars.close_epoch.tolist() == [seconds, seconds * 2]
    assert bars.open.iloc[0] == 100
    assert bars.close.iloc[0] == 100 + seconds / 60
    assert bars.attrs["dropped_incomplete_bars"] == 0


def test_resample_rejects_partial_edges_and_gap_without_filling():
    frame = pd.DataFrame([candle(e) for e in [60, 120, 180, 240, 300, 360, 480, 540, 600, 660]])
    bars = data.resample(frame, "M5")
    assert bars.empty
    assert bars.attrs["dropped_incomplete_bars"] == 3
    assert bars.attrs["input_missing_minutes"] == 1
    frame = pd.DataFrame([candle(e) for e in range(0, 900, 60) if e != 420])
    bars = data.resample(frame, "M5")
    assert bars.epoch.tolist() == [0, 600]
    assert bars.attrs["dropped_incomplete_bars"] == 1


def test_resample_rejects_noncanonical_input():
    with pytest.raises(ValueError, match="canonical"):
        data.resample(pd.DataFrame([candle(0), candle(0)]), "M1")
    with pytest.raises(ValueError, match="Unsupported"):
        data.resample(pd.DataFrame(), "M4")


@pytest.mark.parametrize("timeframe", data.FRAMES)
def test_closed_bar_prefix_is_unchanged_by_unseen_future_minutes(timeframe):
    seconds = data.FRAMES[timeframe]
    rows = [candle(e, 100 + e / 100000) for e in range(0, seconds * 3, 60)]
    prefix = pd.DataFrame(rows[:seconds // 60 + max(1, seconds // 120)])
    available_at = int(prefix.epoch.max()) + 60
    known = data.resample(prefix, timeframe)
    future = pd.DataFrame(rows)
    future.loc[future.epoch >= available_at, ["open", "high", "low", "close"]] *= 10
    full = data.resample(future, timeframe)
    pd.testing.assert_frame_equal(known.reset_index(drop=True),
                                  full[full.close_epoch <= available_at].reset_index(drop=True))


def test_resolve_actual_named_symbols_independently_and_no_guessing():
    catalogue = [dict(underlying_symbol=f"BOOM{n}" + ("N" if n == 300 else ""),
                      underlying_symbol_name=f"Boom {n} Index") for n in (300, 500)]
    targets = data.resolve_catalogue(catalogue)
    assert len(targets) == 10
    assert targets["BOOM300"]["api_symbol"] == "BOOM300N"
    assert targets["CRASH300"]["status"] == "UNAVAILABLE"
    with pytest.raises(ValueError, match="Ambiguous"):
        data.resolve_catalogue(catalogue + catalogue[:1])


@pytest.mark.parametrize("payload", [
    {"buy": "x"}, {"authorize": "secret"}, {"ticks_history": "BOOM500", "proposal": 1},
    {"active_symbols": "brief", "contracts_for": "BOOM500"},
    {"ticks_history": "BOOM500", "style": "ticks"},
    {"ticks_history": "BOOM500", "style": "candles", "granularity": 300}])
def test_public_transport_rejects_account_order_and_wrong_data_requests(payload):
    with pytest.raises(ValueError):
        data._validate_request(payload)


def test_all_environment_safety_aliases_are_enforced(monkeypatch):
    for key in (*data.SAFETY, "ready_for_live", "live_allowed", "opened_trades"):
        monkeypatch.setenv(key, "true")
        with pytest.raises(RuntimeError, match=key):
            data.enforce_safety()
        monkeypatch.delenv(key)


def test_runtime_safety_dictionary_cannot_enable_a_gate(monkeypatch):
    monkeypatch.setitem(data.SAFETY, "LIVE_TRADING", True)
    with pytest.raises(RuntimeError, match="must remain false"):
        data.enforce_safety()


def test_csv_float_round_trip_preserves_exact_cache_identity(tmp_path):
    rows = [candle(0, 14545.183000000001)]
    frame, _ = data.normalize(rows, 0, 60)
    path = tmp_path / "cache.csv"
    frame.to_csv(path, index=False)
    loaded = data.load_m1(path)
    assert loaded.to_dict("records") == frame.to_dict("records")


def install_fake_transport(monkeypatch, tmp_path):
    calls = []
    async def request(self, payload):
        calls.append(dict(payload))
        if "active_symbols" in payload:
            value = {"active_symbols": [dict(underlying_symbol="BOOM300N", underlying_symbol_name="Boom 300 Index")]}
        elif "contracts_for" in payload:
            value = {"contracts_for": {"available": [{"contract_type": "MULTUP"}]}}
        else:
            epochs = list(range(payload["start"], (payload["end"] // 60) * 60 + 1, 60))[-payload["count"]:]
            value = {"candles": [candle(e) for e in epochs]}
        raw = json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = tmp_path / "wire" / f"{digest}.json.gz"
        data._immutable(path, gzip.compress(raw, mtime=0))
        wire = dict(request=dict(payload, req_id=1), wire_file=str(path),
                    wire_sha256=digest, file_sha256=data.sha256(path))
        return value, wire
    monkeypatch.setattr(data.PublicDataClient, "request", request)
    return calls


def test_actual_collection_contract_paginates_caches_and_only_updates_new_minutes(monkeypatch, tmp_path):
    calls = install_fake_transport(monkeypatch, tmp_path)
    first = asyncio.run(data.update_history(tmp_path, days=1, cutoff_epoch=86400,
                                            symbols=["BOOM300"], page_size=1000, delay=.08))
    item = first["symbols"]["BOOM300"]
    assert item["api_symbol"] == "BOOM300N"
    assert item["requested_rows"] == 1440
    assert item["quality"]["missing_minutes"] == 0
    assert len([r for r in calls if "ticks_history" in r]) == 2
    old_file, old_hash = item["file"], item["sha256"]
    calls.clear()
    second = asyncio.run(data.update_history(tmp_path, days=1, cutoff_epoch=86400,
                                             symbols=["BOOM300"], page_size=1000, delay=.08))
    assert not any("ticks_history" in r for r in calls)
    assert second["symbols"]["BOOM300"]["requested_sha256"] == item["requested_sha256"]
    calls.clear()
    third = asyncio.run(data.update_history(tmp_path, days=1, cutoff_epoch=87000,
                                            symbols=["BOOM300"], page_size=1000, delay=.08))
    new = third["symbols"]["BOOM300"]
    history_calls = [r for r in calls if "ticks_history" in r]
    assert len(history_calls) == 1
    assert history_calls[0]["start"] == 86400
    assert new["rows"] == 1450
    assert new["requested_rows"] == 1440
    assert data.sha256(old_file) == old_hash
    assert new["contract_metadata"]["costs_verified"] is False
    assert all(v is False for v in third["safety"].values())
    assert third["real_orders_sent"] == 0


def test_verified_page_cannot_accept_tampered_wire(monkeypatch, tmp_path):
    install_fake_transport(monkeypatch, tmp_path)
    result = asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], delay=.08))
    item = result["symbols"]["BOOM300"]
    page = item["pages"][0]
    Path(page["wire_file"]).write_bytes(b"not-the-wire")
    payload = {k: v for k, v in page["request"].items() if k != "req_id"}
    with pytest.raises(ValueError, match="hash mismatch"):
        data._verified_page(tmp_path, payload)


def test_bad_snapshot_is_error_never_replaced(monkeypatch, tmp_path):
    install_fake_transport(monkeypatch, tmp_path)
    first = asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], delay=.08))
    item = first["symbols"]["BOOM300"]
    Path(item["file"]).write_text("tampered")
    second = asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], delay=.08))
    assert second["symbols"]["BOOM300"]["status"] == "ERROR"
    assert "hash mismatch" in second["symbols"]["BOOM300"]["error"]


def test_repeated_identical_catalogue_wire_has_distinct_timestamped_receipts(monkeypatch, tmp_path):
    install_fake_transport(monkeypatch, tmp_path)
    request = data.PublicDataClient.request
    receipt_number = 0
    async def timestamped(self, payload):
        nonlocal receipt_number
        response, wire = await request(self, payload)
        receipt_number += 1
        wire["received_at_utc"] = f"receipt-{receipt_number}"
        return response, wire
    monkeypatch.setattr(data.PublicDataClient, "request", timestamped)
    one = asyncio.run(data.discover_symbols(tmp_path))
    two = asyncio.run(data.discover_symbols(tmp_path))
    assert one["catalogue"]["wire_sha256"] == two["catalogue"]["wire_sha256"]
    assert len(list((tmp_path / "catalogues").glob("*.json"))) == 2


def test_input_validation(monkeypatch, tmp_path):
    for kwargs in (dict(days=0), dict(page_size=5001), dict(symbols=["BTCUSD"]), dict(symbols=[]),
                   dict(cutoff_epoch=61), dict(concurrency=0), dict(delay=0)):
        with pytest.raises(ValueError):
            asyncio.run(data.update_history(tmp_path, **kwargs))


def test_provider_off_window_response_is_preserved_excluded_and_explicit(monkeypatch, tmp_path):
    install_fake_transport(monkeypatch, tmp_path)
    request = data.PublicDataClient.request
    async def history_boundary(self, payload):
        response, wire = await request(self, payload)
        if "ticks_history" in payload and payload["end"] < 50000:
            response = {"candles": [candle(86400)]}
            wire = replace_fixture_wire(tmp_path, response, wire)
        return response, wire
    monkeypatch.setattr(data.PublicDataClient, "request", history_boundary)
    result = asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], page_size=1000, delay=.08))
    item = result["symbols"]["BOOM300"]
    assert item["status"] == "READY"
    assert item["requested_rows"] == 1000
    assert item["provider_limited"] is True
    assert item["quality"]["missing_minutes"] == 440
    assert data.load_m1(item["requested_file"]).epoch.max() == 86340
    assert item["pages"][-1]["excluded_provider_boundary_response"] is True


def test_mutated_cache_pointer_cannot_silently_change_coverage(monkeypatch, tmp_path):
    install_fake_transport(monkeypatch, tmp_path)
    asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], delay=.08))
    path = tmp_path / "BOOM300" / "latest.json"
    value = json.loads(path.read_text())
    value["cache_quality"]["cutoff_exclusive_epoch"] += 600
    path.write_text(json.dumps(value))
    result = asyncio.run(data.update_history(tmp_path, 1, 87000, ["BOOM300"], delay=.08))
    assert result["symbols"]["BOOM300"]["status"] == "ERROR"
    assert "pointer metadata differs" in result["symbols"]["BOOM300"]["error"]


def test_actual_transport_preserves_rate_limit_error_and_retries(monkeypatch, tmp_path):
    sleeps = []
    class FakeWebsocket:
        def __init__(self):
            self.payload = None
            self.responses = 0
        async def send(self, payload):
            self.payload = json.loads(payload)
        async def recv(self):
            self.responses += 1
            if self.responses == 1:
                return json.dumps({"req_id": self.payload["req_id"], "error": {"code": "RateLimit", "message": "quota"}})
            return json.dumps({"req_id": self.payload["req_id"], "active_symbols": []})
        async def close(self):
            pass
    ws = FakeWebsocket()
    async def connect(*args, **kwargs):
        return ws
    async def sleep(seconds):
        sleeps.append(seconds)
    async def throttle():
        pass
    monkeypatch.setattr(data.websockets, "connect", connect)
    monkeypatch.setattr(data.asyncio, "sleep", sleep)
    monkeypatch.setattr(data, "_throttle_public_request", throttle)
    async def run():
        client = data.PublicDataClient(tmp_path)
        response, wire = await client.request({"active_symbols": "brief"})
        await client.close()
        return response, wire
    response, wire = asyncio.run(run())
    assert response["active_symbols"] == []
    assert sleeps == [15]
    assert len(list((tmp_path / "errors").glob("*.json"))) == 1
    assert wire["authentication_used"] is False


def test_source_history_recovers_legacy_clone_reset_and_preserves_boundary(tmp_path):
    boundary = {"wire_sha256": "boundary-wire", "reason": "out_of_window"}
    original = {"cache_quality": {"excluded_counts": {"off_grid": 2}, "equal_duplicates_removed": 3},
                "provider_history_boundaries": [boundary]}
    path = tmp_path / "original.json"
    data._immutable(path, data._json_bytes(original))
    legacy_clone = {"cache_quality": {"excluded_counts": {}, "equal_duplicates_removed": 0},
                    "quality": {"raw_source_excluded_counts": {}},
                    "provider_history_boundaries": [],
                    "parent_manifest": str(path), "parent_manifest_sha256": data.sha256(path)}
    clone_path = tmp_path / "clone.json"
    data._immutable(clone_path, data._json_bytes(legacy_clone))
    latest = {"cache_quality": {"excluded_counts": {"off_grid": 1}, "equal_duplicates_removed": 4},
              "provider_history_boundaries": [boundary],
              "parent_manifest": str(clone_path), "parent_manifest_sha256": data.sha256(clone_path)}
    excluded, duplicates, boundaries = data._source_history(latest)
    assert excluded == {"off_grid": 3}
    assert duplicates == 7
    assert boundaries == [boundary]
    path.write_text("tampered")
    with pytest.raises(ValueError, match="ancestor manifest hash mismatch"):
        data._source_history(latest)


def test_cache_only_refresh_carries_exclusion_and_provider_limit_history(monkeypatch, tmp_path):
    install_fake_transport(monkeypatch, tmp_path)
    request = data.PublicDataClient.request
    async def history_boundary_and_offgrid(self, payload):
        response, wire = await request(self, payload)
        if "ticks_history" in payload:
            if payload["end"] < 50000:
                response = {"candles": [candle(86400)]}
            else:
                response["candles"].append(candle(86341))
            wire = replace_fixture_wire(tmp_path, response, wire)
        return response, wire
    monkeypatch.setattr(data.PublicDataClient, "request", history_boundary_and_offgrid)
    first = asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], page_size=1000, delay=.08))
    before = first["symbols"]["BOOM300"]
    assert before["quality"]["raw_source_excluded_counts"] == {"off_grid": 1}
    assert before["provider_limited"] is True
    second = asyncio.run(data.update_history(tmp_path, 1, 86400, ["BOOM300"], page_size=1000, delay=.08))
    after = second["symbols"]["BOOM300"]
    assert after["pages"] == []
    assert after["current_attempt_provider_boundaries"] == []
    assert after["quality"]["raw_source_excluded_counts"] == {"off_grid": 1}
    assert after["source_history_excluded_counts"] == {"off_grid": 1}
    assert after["provider_history_boundaries"] == before["provider_history_boundaries"]
    assert after["provider_limited"] is True
    assert after["requested_sha256"] == before["requested_sha256"]
