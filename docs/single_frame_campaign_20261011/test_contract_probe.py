"""Synthetic parser/allowlist fixtures only; never historical market evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).with_name("contract_probe.py")
module_spec = importlib.util.spec_from_file_location("contract_probe", path)
p = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(p)


def request():
    return {"ticks_history": "B500", "style": "ticks", "count": 4,
            "start": p.WINDOWS[0], "end": p.WINDOWS[0] + 3, "req_id": 1}


def test_exact_declared_public_request():
    p.validate_request(request(), {"B500"})


@pytest.mark.parametrize("changed", [{"authorize": "fixture"}, {"buy": "fixture"},
    {"proposal": 1}, {"count": 5}, {"style": "candles"}, {"ticks_history": "BTC"},
    {"start": p.WINDOWS[0] + 1}, {"req_id": True}])
def test_unapproved_requests_are_not_sent(changed):
    with pytest.raises(ValueError):
        p.validate_request({**request(), **changed}, {"B500"})


def test_tick_availability_does_not_claim_full_execution_or_costs():
    r = request()
    result = p.summarize_ticks({"req_id": 1, "echo_req": r,
        "history": {"times": list(range(r["start"], r["end"] + 1)), "prices": [100] * 4}}, r)
    assert result["status"] == "EXACT_WINDOW_AVAILABLE"
    assert result["historical_bid_ask_present"] is False
    assert result["full_history_complete"] is result["actual_cfd_fill_verified"] is False


def test_offwindow_ticks_preserved_as_unavailable():
    r = request()
    result = p.summarize_ticks({"req_id": 1, "echo_req": r,
        "history": {"times": [r["end"] + 100], "prices": [100]}}, r)
    assert result["status"] == "PARTIAL_OR_OFF_WINDOW"
    assert len(result["missing_requested_seconds"]) == 4


def test_echo_mismatch_fails_closed():
    r = request()
    with pytest.raises(ValueError, match="identity"):
        p.summarize_ticks({"req_id": 1, "echo_req": {**r, "count": 5}}, r)


def test_specs_json_is_decoded_without_executing_scripts():
    rows = [{"name": f"{family} {number} Index", "contract-size": "1", "base-currency": "USD",
             "min-size": ".2", "min-spread": ".1", "swap-long-pts": "-1",
             "swap-short-pts": "-1", "trading-hours-gmt": "24/7"}
            for family in ("Boom", "Crash") for number in (300, 500, 600, 900, 1000)]
    html = "<script>arbitraryFunctionMustNeverRun()</script><script>self.__next_f.push(" + json.dumps([1, json.dumps(rows, separators=(",", ":"))]) + ")</script>"
    assert len(p.extract_specs(html)) == 10
    with pytest.raises(ValueError, match="All ten"):
        p.extract_specs("<script>unrelated()</script>")


def test_original_raw_file_cannot_be_overwritten(tmp_path):
    target = tmp_path / "wire"
    p.save(target, b"original")
    with pytest.raises(FileExistsError):
        p.save(target, b"changed")
    assert target.read_bytes() == b"original"
