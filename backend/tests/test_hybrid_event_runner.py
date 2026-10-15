import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[2] / "scripts/run_hybrid_event_regions.py"
S = importlib.util.spec_from_file_location("hybrid_event_runner", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "FOLDER", tmp_path / "docs")
    monkeypatch.setattr(M, "code_hashes", lambda extra: {"example.py": "a" * 64})
    monkeypatch.setattr(M, "audited_sources", lambda: {"symbols": {}, "audit_sha256": "b" * 64})


def test_declaration_is_exclusive_and_verifies_exact_bytes(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    artifact = M.freeze()
    d = M.frozen(artifact["sha256"])
    assert d["new_historical_features_or_labels_computed"] is False
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
