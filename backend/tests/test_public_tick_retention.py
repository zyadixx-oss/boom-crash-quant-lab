"""Synthetic messages verify the bounded availability probe, without networking."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("tick_retention_probe", REPO / "scripts/probe_public_tick_retention.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def wire(request, times=None):
    times = list(range(request["start"], request["end"] + 1)) if times is None else times
    return json.dumps({"req_id": request["req_id"], "msg_type": "history",
                       "history": {"times": times, "prices": [100] * len(times)}})


def test_exact_fixed_four_request_contract():
    requests = probe.requests()
    assert len(requests) == 4
    assert [row["req_id"] for row in requests] == [1, 2, 3, 4]
    assert [row["ticks_history"] for row in requests] == ["BOOM600"] * 2 + ["CRASH600"] * 2
    for row in requests:
        assert row["end"] == row["start"] + 3
        assert row["count"] == 4
        assert row["style"] == "ticks"
        assert set(row) == {"ticks_history", "start", "end", "count", "style", "req_id"}


def test_complete_window_describes_times_without_quote_values():
    request = probe.requests()[0]
    result = probe.describe(wire(request), request)
    assert result["status"] == "COMPLETE_WINDOW"
    assert result["observed_rows"] == 4
    assert result["missing_epochs"] == []
    assert "prices" not in result


@pytest.mark.parametrize("offsets", [[], [0], [0, 2, 3]])
def test_incomplete_windows_stay_unknown(offsets):
    request = probe.requests()[0]
    result = probe.describe(wire(request, [request["start"] + i for i in offsets]), request)
    assert result["status"] == "INCOMPLETE_WINDOW"
    assert result["observed_rows"] == len(offsets)


@pytest.mark.parametrize("offsets", [[-1], [4], [0, 0]])
def test_outside_or_duplicate_epochs_rejected(offsets):
    request = probe.requests()[0]
    with pytest.raises(ValueError):
        probe.describe(wire(request, [request["start"] + i for i in offsets]), request)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '[]'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        probe.parse_wire(text)


@pytest.mark.parametrize("flag", probe.SAFETY)
def test_enabled_gate_blocks_before_output_or_network(tmp_path, monkeypatch, flag):
    for name in probe.SAFETY:
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match=flag):
        asyncio.run(probe.run(tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_intents_before_network_raw_before_parse_and_immutable_output(tmp_path, monkeypatch):
    for name in probe.SAFETY:
        monkeypatch.setenv(name, "false")
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("probe_public_tick_retention.py", "collect_spike_ticks.py"):
        (scripts / name).write_bytes((REPO / "scripts" / name).read_bytes())
    output = tmp_path / "observation"

    class Client:
        async def request(self, request):
            declaration = json.loads((output / "declaration.json").read_text())
            assert declaration["requests"] == probe.requests()
            assert (output / f"request_{request['req_id']}_intent.json").exists()
            return "invalid_json" if request["req_id"] == 2 else wire(request)

        async def reset(self):
            pass

    monkeypatch.setattr(probe, "PublicClient", Client)
    asyncio.run(probe.run(output))
    summary = json.loads((output / "results.json").read_text())
    assert summary["complete_windows"] == 3
    assert summary["full_continuous_tick_coverage"] == "NOT TESTED"
    assert summary["strategy_payoff"] == "NOT TESTED"
    assert summary["records"][1]["status"] == "ERROR"
    assert (output / "request_2_response.wire").read_text() == "invalid_json"
    assert all(value is False for value in summary["safety"].values())
    with pytest.raises(FileExistsError):
        asyncio.run(probe.run(output))
