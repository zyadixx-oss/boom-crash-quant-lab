"""Freeze, provenance and integrated accounting checks using synthetic inputs."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_zone_study as study
from app.research.zone_replay import ZoneReplayConfig, replay_zones, zone_trades


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for flag in study.SAFETY_FLAGS:
        monkeypatch.setenv(flag, "false")


@pytest.fixture
def frozen_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "audited_sources", lambda: {"symbols": {"BOOM600": []}})
    versions = {"module.py": "a" * 64}
    monkeypatch.setattr(study, "code_hashes", lambda extra=(): versions.copy())
    return tmp_path, versions


def test_chronological_partitions_and_protocol_preserve_user_contract():
    p = study.partitions()
    assert p["development"][1] == pd.Timestamp("2026-02-12T11:08Z")
    assert p["final_test"] == (p["development"][1], study.OLD_END)
    assert p["wf1"][1] == p["wf2"][0] and p["wf2"][1] == p["wf3"][0]
    assert p["wf3"][1] == p["final_test"][0]
    assert p["later180"] == (study.OLD_END, study.LATER_END)
    assert study.protocol()["config"] == asdict(ZoneReplayConfig())
    assert study.protocol()["target"] == {"profit_factor": 1.5, "completed_per_symbol_variant": 1000, "active_days": 60}
    assert study.protocol()["fresh_out_of_sample"] is False


def test_freeze_is_exclusive_and_requires_exact_bytes(frozen_sandbox):
    root, _ = frozen_sandbox
    artifact = study.freeze("frozen/declaration.json")
    declaration, path = study.frozen(artifact["path"], artifact["sha256"])
    assert declaration["historical_prices_decoded_for_this_study"] is False
    assert all(v is False for v in declaration["safety"].values())
    with pytest.raises(ValueError): study.freeze(artifact["path"])
    with pytest.raises(ValueError): study.frozen(artifact["path"], "b" * 64)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError): study.frozen(artifact["path"], artifact["sha256"])


def test_postfreeze_source_or_code_change_cannot_silently_run(frozen_sandbox):
    _, versions = frozen_sandbox
    artifact = study.freeze("frozen/declaration.json")
    versions["module.py"] = "b" * 64
    with pytest.raises(ValueError, match="code"):
        study.frozen(artifact["path"], artifact["sha256"])


def test_artifacts_cannot_escape_repository(frozen_sandbox):
    with pytest.raises(ValueError): study.relative_path("../outside.json")


def test_execution_failure_is_saved_and_cannot_be_overwritten(frozen_sandbox, monkeypatch):
    root, _ = frozen_sandbox
    artifact = study.freeze("frozen/declaration.json")
    monkeypatch.setattr(study, "load_symbol", lambda _: (_ for _ in ()).throw(ValueError("synthetic source changed")))
    with pytest.raises(ValueError, match="synthetic source changed"):
        study.execute(artifact["path"], artifact["sha256"], "frozen")
    failure = json.loads((root / "frozen/execution_failure.json").read_text())
    assert failure["QUALIFIED"] is False and failure["error"] == "synthetic source changed"
    saved = (root / "frozen/execution_failure.json").read_bytes()
    with pytest.raises(FileExistsError): study.execute(artifact["path"], artifact["sha256"], "frozen")
    assert (root / "frozen/execution_failure.json").read_bytes() == saved


def test_pinned_daily_decode_preserves_complete_missing_and_run_population(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    start = pd.Timestamp("2026-01-01T00:00Z")
    monkeypatch.setattr(study, "DATA_START", start)
    monkeypatch.setattr(study, "DATA_END", start + pd.Timedelta(minutes=3))
    epoch = start.value // 10**9 + np.arange(180)
    table = pd.DataFrame({"epoch": epoch, "quote": 100 + np.arange(180) / 1000}).drop(index=65)
    path = tmp_path / "synthetic.csv"
    table.to_csv(path, index=False)
    source = {"date": "2026-01-01", "path": path.name, "sha256": study.digest(path), "rows": 179}
    dataset, ticks = study.load_symbol([source])
    assert len(ticks) == 179 and len(dataset.m1) == 3
    assert dataset.summary["missing_seconds"] == 1
    assert dataset.coverage.minute_valid.tolist() == [True, False, True]
    assert dataset.coverage.run_id.iloc[0] == 0 and dataset.coverage.run_id.iloc[2] == 1
    assert dataset.m1.iloc[1].isna().all()
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"): study.load_symbol([source])


def test_zero_trade_inference_stays_unknown_instead_of_creating_perfect_PF(monkeypatch):
    monkeypatch.setattr(study, "BOOTSTRAP_REPEATS", 19)
    start = pd.Timestamp("2026-01-01T00:00Z")
    end = start + pd.Timedelta(days=10)
    ticks = pd.DataFrame({"quote": []}, index=pd.DatetimeIndex([], tz="UTC"))
    events, audit = replay_zones(ticks, [], ZoneReplayConfig(), start, end)
    ledger = zone_trades(events)
    row = study.metric_row("BOOM600", "CRT_RETEST", "final_test", (start, end), ledger, ledger, audit, {})
    assert row["metrics"]["completed"] == 0
    assert row["metrics"]["profit_factor"] is None
    assert row["profit_factor_inference"]["profit_factor_ci95"] == [None, None]
    assert row["day_inference"]["p"] == 1.
    assert row["weekly_inference"]["weekly_p"] == 1.
    assert row["QUALIFIED"] is False


def test_result_json_does_not_emit_nonstandard_nan():
    value = study.json_value({"a": np.float64(np.nan), "b": pd.NaT, "c": pd.NA, "d": np.int64(2)})
    assert json.dumps(value, allow_nan=False) == '{"a": null, "b": null, "c": null, "d": 2}'
