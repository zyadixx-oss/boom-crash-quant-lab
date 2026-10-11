"""Synthetic local fixtures only; never open the real campaign final data.

The saved final family and WF folds are produced by the original campaign
finalizer. Tests invoke the original numerical replay without monkeypatching.
"""
import gzip
import importlib.util
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pytest

from research_campaign import campaign as campaign
from research_campaign.data import FRAMES, load_m1
from research_campaign.engine import CostModel, SplitPlan
from research_campaign.state import SAFETY, atomic_json, digest, encoded, read_json, utc
from research_campaign.strategies import candidates_for_round

module_spec = importlib.util.spec_from_file_location("operational_final_replay", Path(__file__).resolve().parents[2] / "scripts/replay_campaign_final.py")
helper = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(helper)


@pytest.fixture
def completed(tmp_path):
    directory = tmp_path / "synthetic_campaign"
    directory.mkdir()
    n = 1800
    epochs = 1728000000 + np.arange(n) * 60
    close = 100 + np.sin(np.arange(n) / 7) + np.arange(n) * .0002
    opening = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({"epoch": epochs, "open": opening,
        "high": np.maximum(opening, close) + .08, "low": np.minimum(opening, close) - .08, "close": close})
    source = directory / "SYNTHETIC_UNIT_NOT_MARKET.csv"
    frame.to_csv(source, index=False, float_format="%.17g")
    frame = load_m1(source)
    manifest = directory / "SYNTHETIC_UNIT_MANIFEST.json"
    atomic_json(manifest, {"source": "SYNTHETIC TEST ONLY; NO HISTORICAL PERFORMANCE EVIDENCE"})
    split = SplitPlan.from_calendar(int(epochs[0]), int(epochs[-1] + 60))
    row = {"requested_file": str(source), "requested_sha256": digest(source), "requested_rows": n,
           "manifest_file": str(manifest), "api_symbol": "BOOM500",
           "study_quality": {"internal_missing_minutes": 0, "internal_invalid_rows": 0}}
    declaration = {"split": split.to_dict(), "splits": {"BOOM500": split.to_dict()},
        "costs": CostModel().to_dict(), "source_pins": campaign.source_pins(),
        "source_manifest": str(manifest), "source_manifest_sha256": digest(manifest),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "timeframes": dict(FRAMES), "candidate_rounds": [[s.to_dict() for s in candidates_for_round(i)] for i in range(32)],
        "final_bootstrap_repetitions": 99, "datasets": {"BOOM500": row},
        "symbol_catalogue": {"BOOM500": {"display_name": "SYNTHETIC UNIT ONLY", "api_symbol": "BOOM500"}}}
    atomic_json(directory / "declaration.json", declaration)
    winners = {}
    for spec in candidates_for_round(0)[2:4]:
        saved, artifact, _ = campaign.measure(directory, declaration, "BOOM500", "M1", spec, frame)
        winners[f"BOOM500/M1/{spec.family}"] = {"symbol": "BOOM500", "timeframe": "M1", "family": spec.family,
            "status": "TESTED", "spec": spec.to_dict(), "strategy_id": spec.id, "tested_at": saved["tested_at"],
            "validation_score": saved["validation_score"], "artifact": artifact}
    folds = campaign.development_walk_forward(directory, declaration, winners)
    atomic_json(directory / "selection_frozen.json", {"frozen_at": utc(), "selection": winners,
        "walk_forward": folds, "source": digest(directory / "declaration.json"), "final_data_accessed": False})
    cycles = []
    for i in range(1, 33):
        summary = {"number": i, "status": "COMPLETED", "started_at": utc(i * 900),
            "completed_at": utc(i * 900 + 1), "tested_count": 0, "notes": ["SYNTHETIC TEST FIXTURE"]}
        cycles.append(summary)
        atomic_json(directory / "cycles" / f"{i:02d}.json", {**summary, "results": []})
    state = {"id": "SYNTHETIC_TEST_NOT_RESEARCH", "declaration_sha256": digest(directory / "declaration.json"),
        "target_cycles": 32, "interval_seconds": 900, "safety": dict(SAFETY), "completed_cycles": 32,
        "cycles": cycles, "current_cycle": None, "started_epoch": 1, "started_at": utc(1),
        "deadline_at": utc(28801), "final_accessed": False, "status": "RUNNING", "winners": winners, "coverage": {}}
    atomic_json(directory / "state.json", state)
    campaign.finalize(directory, tmp_path / "synthetic_public", state)
    assert read_json(directory / "state.json")["status"] == "COMPLETED"
    return directory, tmp_path / "operational_receipts", source


def family(completed):
    directory, _, _ = completed
    state = read_json(directory / "state.json")
    return campaign.read_artifact(directory, state["final_artifact"])


def rewrite_family(completed, changed):
    directory, _, _ = completed
    state = read_json(directory / "state.json")
    path = directory / state["final_artifact"]["path"]
    path.write_bytes(gzip.compress(encoded(changed), mtime=0))
    state["final_artifact"]["sha256"] = digest(path)
    atomic_json(directory / "state.json", state)


@pytest.mark.parametrize("change", ["status", "count", "sealed", "elapsed", "safety"])
def test_actual_gate_denies_before_any_final_file_open_or_receipt_write(completed, change):
    directory, output, _ = completed
    state = read_json(directory / "state.json")
    if change == "status": state["status"] = "RUNNING"
    if change == "count":
        state["completed_cycles"] = 31
        state["cycles"] = state["cycles"][:31]
    if change == "sealed": state["final_accessed"] = False
    if change == "elapsed": state["started_epoch"] = time.time()
    if change == "safety": state["safety"]["LIVE_TRADING"] = 0
    state["final_artifact"]["path"] = "MUST_NEVER_BE_OPENED.json.gz"
    atomic_json(directory / "state.json", state)
    with pytest.raises(ValueError): helper.run_final(directory, output)
    assert not output.exists()


def test_complete_originals_holm_and_all_actual_saved_folds(completed):
    directory, output, source = completed
    state_bytes = (directory / "state.json").read_bytes()
    source_sha = digest(source)
    report = helper.run_final(directory, output, include_walk_forward=True)
    assert report["status"] == "PASS"
    assert report["expected_final_measurements"] == 2
    assert report["expected_walk_forward_folds"] == 6
    assert report["measured_now"] == report["attempted_now"] == 8
    assert report["reused_verified_receipts"] == 0
    assert report["holm"]["status"] == "PASS"
    assert report["selected_final_and_WF_reproduction_complete"] is True
    assert report["all_development_search_measurements_replayed"] is False
    assert report["phase_counts"]["final"]["verified_total"] == 2
    assert report["phase_counts"]["walk_forward"]["verified_total"] == 6
    assert (directory / "state.json").read_bytes() == state_bytes
    assert digest(source) == source_sha


def test_default_scope_never_claims_wf_or_complete_search_reproduction(completed):
    directory, output, _ = completed
    report = helper.run_final(directory, output)
    assert report["status"] == "PASS" and report["measured_now"] == 2
    assert report["all_final_measurements_replayed"] is True
    assert report["selected_final_and_WF_reproduction_complete"] is False
    assert report["phase_counts"]["walk_forward"]["verified_total"] == 0


def test_explicit_limit_resume_counts_only_actual_new_computations(completed):
    directory, output, _ = completed
    first = helper.run_final(directory, output, max_new=1, include_walk_forward=True)
    assert first["status"] == "PARTIAL" and first["measured_now"] == 1
    assert first["holm"]["status"] == "NOT_RUN"
    second = helper.run_final(directory, output, max_new=1, include_walk_forward=True)
    assert second["measured_now"] == 1 and second["reused_verified_receipts"] == 1
    assert second["holm"]["status"] == "PASS"
    third = helper.run_final(directory, output, include_walk_forward=True)
    assert third["status"] == "PASS" and third["measured_now"] == 6
    assert third["reused_verified_receipts"] == 2
    fourth = helper.run_final(directory, output, include_walk_forward=True)
    assert fourth["attempted_now"] == fourth["measured_now"] == 0
    assert fourth["reused_verified_receipts"] == fourth["completed_total_verified_this_invocation"] == 8


def test_orphan_committed_receipts_are_recovered_without_rerun(completed):
    directory, output, _ = completed
    helper.run_final(directory, output)
    receipts = {p.name: p.read_bytes() for p in (output / "receipts").glob("*.json")}
    (output / "receipts_index.json").unlink()
    resumed = helper.run_final(directory, output)
    assert resumed["measured_now"] == resumed["attempted_now"] == 0
    assert resumed["reused_verified_receipts"] == resumed["recovered_orphan_receipts"] == 2
    assert receipts == {p.name: p.read_bytes() for p in (output / "receipts").glob("*.json")}


def test_changed_receipt_bytes_are_rejected_before_reuse(completed):
    directory, output, _ = completed
    helper.run_final(directory, output)
    receipt = next((output / "receipts").glob("*.json"))
    value = read_json(receipt)
    value["report"]["matched_trades"] += 1
    atomic_json(receipt, value)
    report = helper.run_final(directory, output)
    assert report["status"] == "FAIL"
    assert any("receipt path/hash changed" in e["message"] for e in report["errors"])


def test_receipt_index_for_other_helper_version_is_rejected(completed):
    directory, output, _ = completed
    helper.run_final(directory, output)
    index = read_json(output / "receipts_index.json")
    index["campaign"]["helper_sha256"] = "0" * 64
    atomic_json(output / "receipts_index.json", index)
    with pytest.raises(ValueError, match="another helper version"): helper.run_final(directory, output)


def test_source_change_cannot_reuse_previously_valid_receipts(completed):
    directory, output, source = completed
    helper.run_final(directory, output)
    with source.open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="price source changed"): helper.run_final(directory, output)


def test_family_cannot_omit_one_selected_policy(completed):
    directory, output, _ = completed
    changed = family(completed)
    changed["original_measurements"].pop(next(iter(changed["original_measurements"])))
    rewrite_family(completed, changed)
    with pytest.raises(ValueError, match="coverage"): helper.run_final(directory, output)
    assert not output.exists()


def test_real_numerical_mismatch_fails_without_success_receipt(completed):
    directory, output, _ = completed
    changed = family(completed)
    item = changed["original_measurements"][sorted(changed["original_measurements"])[0]]
    saved = campaign.read_artifact(directory, item)
    saved["final"]["metrics"]["net_profit"] += 123
    path = directory / item["path"]
    path.write_bytes(gzip.compress(encoded(saved), mtime=0))
    item["sha256"] = digest(path)
    rewrite_family(completed, changed)
    report = helper.run_final(directory, output)
    assert report["status"] == "FAIL" and report["attempted_now"] == 1
    assert report["measured_now"] == report["completed_total_verified_this_invocation"] == 0
    assert not list((output / "receipts").glob("*.json"))
    assert any(e["type"] == "ReplayMismatch" for e in report["errors"])


def test_holm_gate_mismatch_is_not_hidden_by_valid_original_replays(completed):
    directory, output, _ = completed
    changed = family(completed)
    changed["results"][0]["gates"]["positive_net_profit"] = not changed["results"][0]["gates"]["positive_net_profit"]
    rewrite_family(completed, changed)
    report = helper.run_final(directory, output)
    assert report["status"] == "FAIL" and report["measured_now"] == 2
    assert report["selected_final_and_WF_reproduction_complete"] is False
    assert any("Holm/gates/stability" in e["message"] for e in report["errors"])


def test_wf_saved_artifact_tamper_prevents_complete_reproduction(completed):
    directory, output, _ = completed
    selection = read_json(directory / "selection_frozen.json")
    artifact = selection["walk_forward"][sorted(selection["walk_forward"])[0]][0]["artifact"]
    with (directory / artifact["path"]).open("ab") as stream: stream.write(b"tamper")
    report = helper.run_final(directory, output, include_walk_forward=True)
    assert report["status"] == "FAIL" and report["measured_now"] == 2
    assert report["attempted_now"] == 3
    assert report["selected_final_and_WF_reproduction_complete"] is False


def test_operational_receipts_cannot_be_written_inside_frozen_campaign(completed):
    directory, _, _ = completed
    with pytest.raises(ValueError, match="outside"): helper.run_final(directory, directory / "new_receipts")
