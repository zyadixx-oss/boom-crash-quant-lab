import json
from pathlib import Path
import pytest

from research_campaign.state import (
    SAFETY, atomic_json, campaign_lock, encoded, guard, key, live_owner,
    next_due, read_json, validate_state,
)


def test_safety_rejects_true(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(RuntimeError):
        guard()


@pytest.mark.parametrize("name", ["live_trading", "ready_for_live", "live_allowed", "opened_trades"])
def test_lowercase_live_override_is_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(RuntimeError):
        guard()


def test_interrupted_exclusive_write_never_publishes_partial_record(tmp_path, monkeypatch):
    from research_campaign import state as module
    def interrupted(_):
        raise OSError("simulated power loss before publish")
    monkeypatch.setattr(module.os, "fsync", interrupted)
    target = tmp_path / "declaration.json"
    with pytest.raises(OSError):
        atomic_json(target, {"immutable": True}, exclusive=True)
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_roundtrip_and_exclusive(tmp_path):
    target = tmp_path / "state.json"
    atomic_json(target, {"عينة": [1, None]})
    assert read_json(target) == {"عينة": [1, None]}
    with pytest.raises(FileExistsError):
        atomic_json(target, {}, exclusive=True)
    assert not list(tmp_path.glob("*.tmp"))


def test_kernel_lock_proves_actual_owner(tmp_path):
    assert live_owner(tmp_path)["live"] is False
    with campaign_lock(tmp_path):
        assert live_owner(tmp_path)["live"] is True
        with pytest.raises(RuntimeError):
            with campaign_lock(tmp_path):
                pass
    assert live_owner(tmp_path)["live"] is False


def state(completed=0):
    return {"safety": SAFETY, "target_cycles": 32, "interval_seconds": 900,
            "completed_cycles": completed, "started_epoch": 1000, "cycles": []}


def test_schedule_uses_actual_completions():
    assert next_due(state()) == 1000
    assert next_due(state(2)) == 2800
    with pytest.raises(ValueError):
        validate_state(state(2))


def test_missed_slots_are_not_compressed_into_back_to_back_cycles():
    value = state(1)
    value["cycles"] = [{"started_at": "1970-01-01T02:00:00+00:00"}]
    assert next_due(value) == 7200 + 900


def test_final_cannot_open_early():
    value = state()
    value["final_accessed"] = True
    with pytest.raises(ValueError):
        validate_state(value)


def test_nan_not_a_json_performance_number():
    assert json.loads(encoded({"PF": float("nan")}))["PF"] is None
    assert key({"x": 1, "y": 2}) == key({"y": 2, "x": 1})
