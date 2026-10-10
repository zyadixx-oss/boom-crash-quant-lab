from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

FOLDER = Path(__file__).resolve().parent
sys.path[:0] = [str(FOLDER), str(FOLDER.parents[1]), str(FOLDER.parents[1] / "backend")]
specification = importlib.util.spec_from_file_location("nonlinear_region_runner", FOLDER / "runner.py")
M = importlib.util.module_from_spec(specification)
specification.loader.exec_module(M)
from stable_gate import stable_positive_gate
from app.research.zone_qualification import historical_zone_gate


def metric():
    return {"completed": 1500, "active_days": 100, "censored": 0, "invalid_uncensored": 0,
        "unknown_regions": 0, "control_unknown_regions": 0, "selection_score": .01,
        "profit_factor": 1.02, "mean_net_R": .01, "closed_trade_max_drawdown": .05, "equity_ruin": False}


def arguments():
    return {"metrics": {**metric(), "mean_net_R_ci95": [.001, .02], "baseline_difference_ci95": [.001, .02]},
        "pf": {"day_profit_factor_ci95": [1.001, 1.05], "weekly_profit_factor_ci95": [1.001, 1.05]},
        "weekly": {"weekly_mean_net_R_ci95": [.001, .02], "weekly_difference_ci95": [.001, .02]},
        "folds": [metric() for _ in range(3)], "validation": metric(),
        "thirds": [metric() for _ in range(3)], "doubled_cost": metric(),
        "unknown_regions": 0, "baseline_unknown_regions": 0, "holm_p": .001}


def test_user_revised_target_preserves_every_gate_except_old_point_floor():
    args = arguments(); old_args = deepcopy(args)
    assert historical_zone_gate(**args)["historical_rejection_reasons"] == ["PF_below_1.5_or_unknown"]
    gate = stable_positive_gate(**args)
    assert gate["historical_criteria_passed"]
    assert gate["qualified"] is False and gate["fresh_out_of_sample"] is False
    assert "PF_1_5_lower_CI_supported" not in gate
    assert args == old_args


@pytest.mark.parametrize("field,value", [("profit_factor", 1.), ("profit_factor", .999), ("profit_factor", None),
    ("profit_factor", float("nan")), ("profit_factor", True), ("mean_net_R", 0.), ("mean_net_R", -.001),
    ("mean_net_R", None), ("mean_net_R", float("inf"))])
def test_nonpositive_or_undefined_is_not_positive_profit(field, value):
    args = arguments(); args["metrics"][field] = value
    assert not stable_positive_gate(**args)["historical_criteria_passed"]


@pytest.mark.parametrize("mutation", ["unknown", "reference_unknown", "week_PF", "day_mean", "cost", "third", "n", "days", "development", "holm", "drawdown"])
def test_revised_point_target_cannot_remove_existing_rejections(mutation):
    args = arguments()
    if mutation == "unknown": args["unknown_regions"] = 1
    elif mutation == "reference_unknown": args["baseline_unknown_regions"] = 1
    elif mutation == "week_PF": args["pf"]["weekly_profit_factor_ci95"] = [.999, 1.1]
    elif mutation == "day_mean": args["metrics"]["mean_net_R_ci95"] = [0., .02]
    elif mutation == "cost": args["doubled_cost"]["mean_net_R"] = 0.
    elif mutation == "third": args["thirds"][1]["completed"] = 199
    elif mutation == "n": args["metrics"]["completed"] = 999
    elif mutation == "days": args["metrics"]["active_days"] = 59
    elif mutation == "development": args["folds"][0]["mean_net_R"] = 0.
    elif mutation == "holm": args["holm_p"] = .05
    else: args["metrics"]["closed_trade_max_drawdown"] = .1001
    old = historical_zone_gate(**args); new = stable_positive_gate(**args)
    assert not new["historical_criteria_passed"]
    assert new["historical_rejection_reasons"] == [r for r in old["historical_rejection_reasons"] if r != "PF_below_1.5_or_unknown"]


def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "FOLDER", tmp_path / "study")
    monkeypatch.setattr(M, "code_hashes", lambda extra: {"code": "a" * 64})
    monkeypatch.setattr(M, "inputs_manifest", lambda: {})
    monkeypatch.setattr(M, "audited_sources", lambda: {})


def test_freeze_exclusive_exact_bytes_and_no_new_scores(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    artifact = M.freeze(); d = M.frozen(artifact["sha256"])
    assert d["new_historical_targets_or_scores_computed"] is False
    assert not any(d["safety"].values())
    with pytest.raises(ValueError): M.freeze()
    with pytest.raises(ValueError): M.frozen("0" * 64)


@pytest.mark.parametrize("change", ["code", "sources", "inputs", "runtime", "specification", "sidecar"])
def test_freeze_rejects_changed_lineage(monkeypatch, tmp_path, change):
    isolate(monkeypatch, tmp_path); artifact = M.freeze()
    names = {"code": "code_hashes", "sources": "audited_sources", "inputs": "inputs_manifest", "runtime": "runtime", "specification": "spec"}
    if change == "sidecar": (M.FOLDER / "declaration.sha256").write_text("b" * 64)
    else: monkeypatch.setattr(M, names[change], lambda *args: {"changed": True})
    with pytest.raises(ValueError): M.frozen(artifact["sha256"])


@pytest.mark.parametrize("flag", ["LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES"])
def test_live_flag_refuses_freeze(monkeypatch, tmp_path, flag):
    isolate(monkeypatch, tmp_path); monkeypatch.setenv(flag, "true")
    with pytest.raises(RuntimeError): M.freeze()


def test_unknowns_disclosed_and_no_parameter_or_split_search():
    s = M.spec()
    assert s["known_unknown_references_preclude_historical_qualification"] is True
    assert s["known_parent_CLOCK_unknowns_each_symbol"]["later180"] == 8
    assert s["parameters"]["n_trees"] == 100
    assert s["point_criterion"] == "finite PF > 1 AND finite mean_net_R > 0"
    assert s["fresh_out_of_sample"] is False
    for fold in M.folds().values():
        for start, end in fold["evaluate"].values():
            assert fold["train"][1] <= start < end
    assert set(M.folds()["fit70"]["evaluate"]) == {"final_test", "later180"}


def test_exact_reference_copy_preserves_numbers_and_source_identity():
    parent = {"symbol": "BOOM600", "variant": "RAW_REGION", "partition": "later180", "metrics": {"profit_factor": 1.01},
        "qualification": {"old": True}, "holm_p": .9, "target_comparisons": {"old": {}}, "required_reference_checks": {}}
    copied = M.reference_row(parent)
    assert copied["metrics"] == parent["metrics"]
    copied["metrics"]["profit_factor"] = 2.
    assert parent["metrics"]["profit_factor"] == 1.01
    assert "qualification" not in copied and copied["reference_origin"]["result_sha256"] == M.PARENT_RESULT


def rows():
    out = []
    for symbol in M.SYMBOLS:
        for family in M.FAMILIES:
            for part in M.PARTS:
                day = {"p": .0001, "mean_net_R_ci95": [.001, .02], "baseline_difference_ci95": [.001, .02]}
                week = {"weekly_p": .0001, "weekly_mean_net_R_ci95": [.001, .02], "weekly_difference_ci95": [.001, .02]}
                out.append({"symbol": symbol, "variant": family, "partition": part, "metrics": metric(),
                    "day_inference": day, "weekly_inference": week, "replay_audit": {"unknown": 0},
                    "profit_factor_inference": arguments()["pf"], "thirds": [{"metrics": metric()} for _ in range(3)],
                    "double_cost_metrics": metric(), "target_comparisons": {
                        ref: {"day": day.copy(), "week": week.copy()} for ref in M.references(family) if ref != "CLOCK"} if family in M.NEW else {}})
    return out


def target(data, family="HYBRID_BOOST_REGION", part="later180"):
    return next(r for r in data if r["symbol"] == "CRASH600" and r["variant"] == family and r["partition"] == part)


def test_joint_eight_conjunction_family_at_revised_point_target():
    data = rows(); M.attach_gates(data)
    held = [r for r in data if "qualification" in r]
    assert len(held) == 8 and all(r["qualification"]["historical_criteria_passed"] for r in held)
    assert all(r["holm_p"] == .0008 for r in held)


@pytest.mark.parametrize("change", ["unknown", "day_mean", "week_advantage", "union_missing", "reference_missing", "unlisted_cell"])
def test_required_references_and_exact_family_cannot_be_bypassed(change):
    data = rows(); item = target(data)
    if change == "unknown": target(data, "HYBRID_REGION")["replay_audit"]["unknown"] = 1
    elif change == "day_mean": item["target_comparisons"]["HYBRID_REGION"]["day"]["mean_net_R_ci95"] = [None, None]
    elif change == "week_advantage": item["target_comparisons"]["RAW_BOOST_REGION"]["week"]["weekly_difference_ci95"] = [0., .02]
    elif change == "union_missing": target(data, part="walk_forward_combined")["target_comparisons"].pop("HYBRID_REGION")
    elif change == "reference_missing": item["target_comparisons"].pop("RAW_BOOST_REGION")
    else: data[0]["variant"] = "unlisted"
    if change in ("union_missing", "reference_missing", "unlisted_cell"):
        with pytest.raises(ValueError): M.attach_gates(data)
    else:
        M.attach_gates(data)
        assert not item["qualification"]["historical_criteria_passed"]


def test_required_reference_unknowns_block_development():
    data = rows(); target(data, "CLOCK", "wf2")["replay_audit"]["unknown"] = 1
    M.attach_gates(data)
    assert target(data)["qualification"]["development_eligible"] is False


def test_pinned_reads_and_exclusive_writes_stay_inside_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    saved = M.save(tmp_path / "a.json", {"a": 1})
    assert json.loads(M.read_pinned(saved)[1]) == {"a": 1}
    with pytest.raises(FileExistsError): M.save(tmp_path / "a.json", {})
    with pytest.raises(ValueError): M.save(tmp_path.parent / "outside.json", {})
    (tmp_path / "a.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"): M.read_pinned(saved)
    with pytest.raises(ValueError): M.read_pinned({"path": "../outside.json", "sha256": "0" * 64})


def test_saved_planned_horizon_units_are_normalized_without_changing_times(monkeypatch):
    issues = pd.DatetimeIndex(["2026-01-01T00:00Z"]).as_unit("ns")
    expected = pd.DataFrame({"net_R": [.1], "planned_end": issues + pd.Timedelta(minutes=30, seconds=1)}, index=issues)
    saved = expected.reset_index(names="signal_time")
    saved["planned_end"] = saved.planned_end.dt.as_unit("us").astype(str)
    monkeypatch.setattr(M, "spike_clock_signals", lambda *args: SimpleNamespace(signals=pd.DataFrame()))
    monkeypatch.setattr(M, "region_training_labels", lambda *args: expected)
    monkeypatch.setattr(M, "read_table", lambda a: saved.copy() if a == "labels" else pd.DataFrame())
    actual = M.cached_labels(None, "BOOM600", (None, None), {"labels": "labels", "events": "events"})
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_names=False)
