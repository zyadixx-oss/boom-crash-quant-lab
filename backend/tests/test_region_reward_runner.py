import importlib.util
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

P = Path(__file__).resolve().parents[2] / "scripts/run_region_reward_regions.py"
S = importlib.util.spec_from_file_location("region_reward_runner", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "FOLDER", tmp_path / "docs")
    monkeypatch.setattr(M, "code_hashes", lambda extra: {"example.py": "a" * 64})
    monkeypatch.setattr(M, "inputs_manifest", lambda: {})
    monkeypatch.setattr(M, "audited_sources", lambda: {"symbols": {}, "audit_sha256": "b" * 64})


def test_declaration_is_exclusive_and_verifies_exact_bytes(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    artifact = M.freeze()
    d = M.frozen(artifact["sha256"])
    assert d["new_historical_targets_or_scores_computed"] is False
    assert not any(d["safety"].values()) and d["QUALIFIED"] is False
    with pytest.raises(ValueError): M.freeze()
    with pytest.raises(ValueError): M.frozen("0" * 64)


@pytest.mark.parametrize("change", ["code", "sources", "specification", "sidecar"])
def test_changed_frozen_inputs_refused(monkeypatch, tmp_path, change):
    isolate(monkeypatch, tmp_path)
    a = M.freeze()
    if change == "code": monkeypatch.setattr(M, "code_hashes", lambda extra: {"example.py": "c" * 64})
    elif change == "sources": monkeypatch.setattr(M, "audited_sources", lambda: {"symbols": {"unexpected": []}})
    elif change == "specification": monkeypatch.setattr(M, "spec", lambda: {})
    else: (M.FOLDER / "declaration.sha256").write_text("d" * 64)
    with pytest.raises(ValueError): M.frozen(a["sha256"])


def test_each_fold_is_trained_before_its_evaluations_without_final_refit():
    f = M.folds()
    assert set(f) == {"fit40", "fit50", "fit60", "fit70"}
    for contract in f.values():
        assert contract["train"][0] < contract["train"][1]
        for start, end in contract["evaluate"].values():
            assert contract["train"][1] <= start < end
    assert set(f["fit70"]["evaluate"]) == {"final_test", "later180"}


def test_incomplete_multiplicity_family_is_refused():
    with pytest.raises(ValueError): M.attach_gates([])


def test_live_env_refuses_declaration(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(RuntimeError): M.freeze()


def write_cache(tmp_path, name, frame):
    p = tmp_path / name
    payload = gzip.compress(frame.to_csv(index=False).encode(), mtime=0)
    p.write_bytes(payload)
    return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}


def cache_fixture(tmp_path):
    index = pd.date_range("2026-01-01T00:00Z", periods=24, freq="5min").as_unit("ns")
    f = pd.DataFrame(np.ones((len(index), 44)), columns=M.FEATURE_NAMES)
    f.insert(0, "m5_open", index);f["feature_valid"] = True;f["atr"] = 2.;f["close"] = 100.
    h = f.copy();h["atr"] = .02
    mask = (index + pd.Timedelta(minutes=5)).minute % 30 == 0
    available = pd.DataFrame({"m5_open": index[mask], "raw_feature_valid": True,
        "transformed_feature_valid": True, "common_feature_valid": True,
        "raw_execution_atr": 2., "transformed_feature_atr": .02,
        "run_id": pd.array([0, pd.NA, 1, 1], dtype="Int64")})
    for pos, stamp in enumerate(index[mask]):
        if pos == 1:
            f.loc[f.m5_open.eq(stamp), "feature_valid"] = False
            h.loc[h.m5_open.eq(stamp), "feature_valid"] = False
            available.loc[pos, ["raw_feature_valid", "transformed_feature_valid", "common_feature_valid"]] = False
    features = {"RAW44": write_cache(tmp_path, "raw.csv.gz", f), "HYBRID44": write_cache(tmp_path, "hybrid.csv.gz", h),
                "availability": write_cache(tmp_path, "availability.csv.gz", available)}
    p = tmp_path / "model.json";p.write_text(json.dumps({"status": "FITTED", "models": {}, "model_metadata": {}}))
    return {"features": features, "timed_fit": {"path": "model.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}


def test_cached_features_and_nullable_clock_runs_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path);manifest = cache_fixture(tmp_path)
    inputs, fit = M.load_inputs(manifest)
    assert len(inputs.raw) == 24 and inputs.availability.run_id.notna().sum() == 3
    assert inputs.availability.raw_execution_atr.eq(2).all()
    assert inputs.availability.transformed_feature_atr.eq(.02).all()
    assert inputs.raw.feature_valid.sum() == 23


def test_cached_feature_tampering_is_refused_before_decode(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path);manifest = cache_fixture(tmp_path)
    with (tmp_path / "raw.csv.gz").open("ab") as f:f.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"): M.load_inputs(manifest)


def test_mismatched_saved_common_availability_cannot_enable_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path);manifest = cache_fixture(tmp_path)
    p = tmp_path / "availability.csv.gz";f = pd.read_csv(p);f.loc[1, "common_feature_valid"] = True
    manifest["features"]["availability"] = write_cache(tmp_path, p.name, f)
    with pytest.raises(ValueError, match="availability differs"): M.load_inputs(manifest)


def test_quote_loader_retains_missing_seconds_and_original_prices(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    epoch = pd.Timestamp("2026-01-01T00:00Z").value // 10**9
    p = tmp_path / "ticks.csv"
    pd.DataFrame({"epoch": epoch + np.array([0, 1, 4, 5]), "quote": [100., 101., 99., 100.]}).to_csv(p, index=False)
    source = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": 4, "date": "2026-01-01"}
    quotes = M.load_quotes([source])
    np.testing.assert_array_equal(np.diff(quotes.index.asi8) // 10**9, [1, 3, 1])
    np.testing.assert_array_equal(quotes.quote, [100, 101, 99, 100])


def test_artifact_cannot_be_written_outside_declared_root(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path / "inside")
    with pytest.raises(ValueError): M.save(tmp_path / "outside.json", {})
    assert not (tmp_path / "outside.json").exists()


def gate_rows():
    rows = []
    for symbol in M.SYMBOLS:
        for part in ("wf1", "wf2", "wf3", "walk_forward_combined", "final_test", "later180"):
            for family in M.FAMILIES:
                day = {"p": .0001, "mean_net_R_ci95": [.1, .3], "baseline_difference_ci95": [.05, .2]}
                week = {"weekly_p": .0001, "weekly_mean_net_R_ci95": [.1, .3], "weekly_difference_ci95": [.05, .2]}
                rows.append({"symbol": symbol, "partition": part, "variant": family, "metrics": {},
                    "replay_audit": {"unknown": 0}, "day_inference": day, "weekly_inference": week,
                    "profit_factor_inference": {}, "thirds": [], "double_cost_metrics": {},
                    "target_comparisons": {ref: {"day": day.copy(), "week": week.copy()} for ref in M.references(family) if ref != "CLOCK"} if family in M.NEW else {}})
    return rows


@pytest.mark.parametrize("change", ["day_advantage", "week_advantage", "undefined", "nan", "unknown"])
def test_extra_required_reference_can_veto_otherwise_accepted_gate(monkeypatch, change):
    # Even a successful old CLOCK gate and significant conjunction p cannot
    # substitute for a positive interval or known outcomes in another reference.
    monkeypatch.setattr(M, "historical_zone_gate", lambda **kw: {
        "historical_criteria_passed": True, "historical_rejection_reasons": [],
        "development_eligible": True, "qualified": False})
    rows = gate_rows()
    target = next(r for r in rows if r["symbol"] == M.SYMBOLS[0] and r["partition"] == "later180" and r["variant"] == "RAW_REGION")
    ref = target["target_comparisons"]["RAW_TIMED"]
    if change == "day_advantage": ref["day"]["baseline_difference_ci95"] = [0., .2]
    elif change == "week_advantage": ref["week"]["weekly_difference_ci95"] = [-.1, .2]
    elif change == "undefined": ref["day"]["mean_net_R_ci95"] = [None, None]
    elif change == "nan": ref["week"]["weekly_mean_net_R_ci95"] = [float("nan"), .2]
    else:
        next(r for r in rows if r["symbol"] == M.SYMBOLS[0] and r["partition"] == "later180" and r["variant"] == "RAW_TIMED")["replay_audit"]["unknown"] = 1
    M.attach_gates(rows)
    assert target["holm_p"] < .05
    assert not target["qualification"]["historical_criteria_passed"]
    assert any("required_reference_RAW_TIMED" in x for x in target["qualification"]["historical_rejection_reasons"])


def test_extra_reference_unknowns_also_block_development(monkeypatch):
    captured = []
    def gate(**kw):
        captured.append(kw)
        return {"historical_criteria_passed": True, "historical_rejection_reasons": []}
    monkeypatch.setattr(M, "historical_zone_gate", gate)
    rows = gate_rows()
    for r in rows:
        if r["variant"] in ("RAW_TIMED", "HYBRID_TIMED"):
            r["replay_audit"]["unknown"] = 2
    M.attach_gates(rows)
    assert len(captured) == 8
    assert all(v["baseline_unknown_regions"] == 2 and v["validation"]["control_unknown_regions"] == 2
        and all(f["control_unknown_regions"] == 2 for f in v["folds"]) for v in captured)


def test_missing_required_target_reference_refuses_gate():
    rows = gate_rows()
    target = next(r for r in rows if r["partition"] == "later180" and r["variant"] == "HYBRID_REGION")
    del target["target_comparisons"]["RAW_REGION"]
    with pytest.raises(ValueError, match="required target references"): M.attach_gates(rows)
