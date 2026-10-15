"""Synthetic boundary fixtures only; these are never historical performance."""
from dataclasses import replace
import gzip
import json
import numpy as np
import pandas as pd
import pytest

from research_campaign import campaign as c
from research_campaign.engine import CostModel, SplitPlan
from research_campaign.state import SAFETY, atomic_json, encoded
from research_campaign.strategies import candidates_for_round


def bars():
    n = 3000
    epochs = 1728000000 + np.arange(n) * 60
    close = 100 + np.sin(np.arange(n) / 7) + np.arange(n) * .0002
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"epoch": epochs, "open": open_, "close": close,
                         "high": np.maximum(open_, close) + .08,
                         "low": np.minimum(open_, close) - .08})


def declaration(frame):
    split = SplitPlan.from_calendar(int(frame.epoch.iloc[0]), int(frame.epoch.iloc[-1] + 60))
    return {"split": split.to_dict(), "costs": CostModel().to_dict(), "source_pins": {},
            "final_bootstrap_repetitions": 99,
            "datasets": {"BOOM500": {"requested_sha256": "unit_fixture_not_real_history",
                         "requested_file": "unit_fixture", "manifest_file": "unit_fixture",
                         "api_symbol": "BOOM500", "quality": {"missing_minutes": 0}}}}


def test_search_never_constructs_final_indicators():
    frame = bars()
    d = declaration(frame)
    split = SplitPlan.from_dict(d["split"])
    prepared = c.prepared_contexts(frame, split, "M1", "development")
    assert (prepared["context"].bars.epoch + 60 <= split.validation_end).all()
    assert (prepared["training_context"].bars.epoch + 60 <= split.train_end).all()


def test_changed_final_prices_cannot_change_development_measurement(tmp_path):
    frame = bars()
    d = declaration(frame)
    mutated = frame.copy()
    end = d["split"]["validation_end"]
    mask = mutated.epoch >= end
    for col in ("open", "high", "low", "close"):
        mutated.loc[mask, col] *= 1000
    spec = candidates_for_round(0)[1]
    a, _, _ = c.measure(tmp_path / "a", d, "BOOM500", "M1", spec, frame)
    b, _, _ = c.measure(tmp_path / "b", d, "BOOM500", "M1", spec, mutated)
    assert a["train"]["metrics"] == b["train"]["metrics"]
    assert a["validation"]["metrics"] == b["validation"]["metrics"]
    assert a["train"]["trades"] == b["train"]["trades"]
    assert a["validation"]["trades"] == b["validation"]["trades"]


def test_identical_scientific_candidate_is_reused(tmp_path, monkeypatch):
    frame = bars()
    d = declaration(frame)
    first = candidates_for_round(0)[0]
    duplicate = candidates_for_round(2)[0]  # ICT does not consume threshold.
    assert first.scientific_id == duplicate.scientific_id
    a, artifact, hit = c.measure(tmp_path, d, "BOOM500", "M1", first, frame)
    assert hit is False
    monkeypatch.setattr(c, "run_backtest", lambda *a, **kw: pytest.fail("Repeated identical test"))
    b, repeated, hit = c.measure(tmp_path, d, "BOOM500", "M1", duplicate, frame)
    assert hit is True and artifact == repeated and encoded(a) == encoded(b)


def test_artifact_tampering_is_not_a_cache_hit(tmp_path):
    artifact = c.save_artifact(tmp_path, "test", {"source": "synthetic_unit_test"})
    (tmp_path / artifact["path"]).write_bytes(gzip.compress(b"{}"))
    with pytest.raises(ValueError):
        c.read_artifact(tmp_path, artifact)


def test_resume_after_artifact_commit_before_cache_pointer(tmp_path, monkeypatch):
    frame = bars()
    d = declaration(frame)
    spec = candidates_for_round(0)[1]
    original, artifact, _ = c.measure(tmp_path, d, "BOOM500", "M1", spec, frame)
    next((tmp_path / "cache").glob("*.json")).unlink()
    monkeypatch.setattr(c, "run_backtest", lambda *a, **kw: pytest.fail("Crash recovery repeated measurement"))
    restored, recovered, hit = c.measure(tmp_path, d, "BOOM500", "M1", spec, frame)
    assert hit and recovered == artifact and encoded(restored) == encoded(original)
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_resume_rebuilds_winner_from_durable_cycle_row():
    summary = {"symbol": "BOOM500", "timeframe": "M5", "family": "crt",
               "validation_score": .1, "strategy_id": "a"}
    state = {"winners": {}}
    assert c.consider_winner(state, summary)
    assert not c.consider_winner(state, summary)
    assert c.consider_winner(state, {**summary, "strategy_id": "b", "validation_score": .2})
    assert state["winners"]["BOOM500/M5/crt"]["strategy_id"] == "b"


def test_resume_committed_cycle_preserves_actual_completion_time(tmp_path, monkeypatch):
    artifact = c.save_artifact(tmp_path, "synthetic-recovery", {"fixture": True})
    summary = {"symbol": "BOOM500", "timeframe": "M5", "family": "crt", "artifact": artifact,
               "validation_score": .1, "strategy_id": "a"}
    cycle = {"number": 1, "status": "COMPLETED", "started_at": "2026-10-11T00:00:00Z",
             "completed_at": "2026-10-11T00:02:00Z", "tested_count": 1, "notes": [],
             "coverage": {f"{s}/{f}": {"status": "TESTED"} for s in c.TARGETS for f in c.FRAMES},
             "results": [summary]}
    path = tmp_path / "cycles/01.json"
    atomic_json(path, cycle)
    original = path.read_bytes()
    state = {"completed_cycles": 0, "started_epoch": 1, "winners": {}, "coverage": {}, "cycles": []}
    monkeypatch.setattr(c, "frozen", lambda *a: {})
    monkeypatch.setattr(c, "_save_state", lambda *a: None)
    monkeypatch.setattr(c, "load_m1", lambda *a: pytest.fail("Committed cycle reloaded prices"))
    monkeypatch.setattr(c, "update_history", lambda *a, **kw: pytest.fail("Committed cycle requested new market data"))
    c.execute_cycle(tmp_path, tmp_path / "public", state)
    assert path.read_bytes() == original
    assert state["completed_cycles"] == 1
    assert state["cycles"][0]["completed_at"] == cycle["completed_at"]
    assert state["winners"]["BOOM500/M5/crt"] == summary


def test_finalization_refuses_incomplete_schedule(tmp_path):
    with pytest.raises(ValueError):
        c.finalize(tmp_path, tmp_path / "public", {"completed_cycles": 31})


def test_finalization_waits_full_eight_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(c.time, "time", lambda: 1000 + 31 * 900)
    with pytest.raises(ValueError, match="eight-hour"):
        c.finalize(tmp_path, tmp_path / "public", {"completed_cycles": 32, "started_epoch": 1000})


def test_recover_committed_final_family_without_remeasurement(tmp_path, monkeypatch):
    selection = {"source": "pinned", "selection": {"BOOM500/M5/crt": {"status": "TESTED"}},
                 "walk_forward": {}, "final_data_accessed": False}
    atomic_json(tmp_path / "selection_frozen.json", selection)
    selection_sha = c.digest(tmp_path / "selection_frozen.json")
    artifact = c.save_artifact(tmp_path, "final-family-" + selection_sha, {
        "selection_sha256": selection_sha, "safety": SAFETY, "finalized_at": "2026-10-12T00:00:00+00:00",
        "results": [{"selection_group": "BOOM500/M5/crt", "qualified": False}]})
    state = {"completed_cycles": 32, "started_epoch": 1, "declaration_sha256": "pinned"}
    monkeypatch.setattr(c, "frozen", lambda *a: {})
    monkeypatch.setattr(c, "_save_state", lambda *a: None)
    monkeypatch.setattr(c, "measure", lambda *a, **kw: pytest.fail("Repeated committed final evaluation"))
    c.finalize(tmp_path, tmp_path / "public", state)
    assert state["status"] == "COMPLETED" and state["final_accessed"]
    assert state["final_artifact"] == artifact
    assert state["completed_at"] == "2026-10-12T00:00:00+00:00"


def test_unknown_source_cannot_initialize_actual_campaign(tmp_path):
    source = tmp_path / "fake.json"
    atomic_json(source, {"source": "random prices", "safety": SAFETY, "symbols": {}})
    with pytest.raises(ValueError):
        c.initialize(tmp_path / "campaign", source)


def initialization_fixture(tmp_path):
    """Synthetic wire fixture, used only for transaction-boundary testing."""
    from research_campaign.tests.test_source_audit import source, catalogue
    raw_dir = tmp_path / "synthetic_raw"
    raw_dir.mkdir()
    row = {**source(raw_dir), "status": "READY"}
    manifest = tmp_path / "initial_fixture.json"
    atomic_json(manifest, {"source": "Official Deriv public M1 candles", "endpoint": c.ENDPOINT,
                          "catalogue": catalogue(raw_dir), "safety": SAFETY,
                          "symbols": {"BOOM500": row}, "start_epoch": 0,
                          "cutoff_exclusive_epoch": 600, "fixture_only": True})
    return manifest


def orphan_initialization(tmp_path, monkeypatch):
    manifest = initialization_fixture(tmp_path)
    directory = tmp_path / "campaign"
    original = c.atomic_json

    def crash_on_initial_state(path, *args, **kwargs):
        if path.name == "state.json":
            raise RuntimeError("Synthetic interruption after declaration commit")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(c, "atomic_json", crash_on_initial_state)
        with pytest.raises(RuntimeError, match="after declaration commit"):
            c.initialize(directory, manifest)
    assert (directory / "declaration.json").exists()
    assert not (directory / "state.json").exists()
    return directory, manifest


def test_initialization_recovers_exact_orphan_declaration(tmp_path, monkeypatch):
    directory, manifest = orphan_initialization(tmp_path, monkeypatch)
    committed_bytes = (directory / "declaration.json").read_bytes()
    recovered = c.initialize(directory, manifest)
    assert (directory / "declaration.json").read_bytes() == committed_bytes
    assert recovered["completed_cycles"] == 0 and recovered["cycles"] == []
    assert recovered["final_accessed"] is False
    assert recovered["declaration_sha256"] == c.digest(directory / "declaration.json")
    assert c.read_json(directory / "declaration.json")["datasets"]["BOOM500"]["study_quality"]["catalogue_binding_checked"] is True
    with pytest.raises(ValueError, match="cannot be initialized again"):
        c.initialize(directory, manifest)


@pytest.mark.parametrize("field,replacement", [
    ("runtime", {"python": "changed"}), ("source_pins", {}),
    ("candidate_rounds", []), ("target_cycles", 31), ("source_manifest_sha256", "changed"),
    ("costs", {}), ("splits", {}),
])
def test_orphan_initialization_cannot_change_scientific_plan(tmp_path, monkeypatch, field, replacement):
    directory, manifest = orphan_initialization(tmp_path, monkeypatch)
    committed = c.read_json(directory / "declaration.json")
    committed[field] = replacement
    atomic_json(directory / "declaration.json", committed)
    before = (directory / "declaration.json").read_bytes()
    with pytest.raises(ValueError, match="differs from pinned inputs or scientific plan"):
        c.initialize(directory, manifest)
    assert (directory / "declaration.json").read_bytes() == before
    assert not (directory / "state.json").exists()
