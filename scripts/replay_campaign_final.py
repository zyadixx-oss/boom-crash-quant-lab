"""Resume numerical replay of the committed final family only.

Operational helper outside the frozen campaign. It calls the original replay
implementation without modifying formulas, measurement caches or campaign state.
Saved receipts prove earlier replays of this exact helper/source version; they
are reported separately from computations performed by the current invocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_campaign import replay as original
from research_campaign.campaign import _quality_ok
from research_campaign.data import FRAMES, load_m1, resample
from research_campaign.engine import CostModel, SplitPlan, finalize_multiplicity, run_backtest
from research_campaign.state import (INTERVAL_SECONDS, TARGET_CYCLES, SAFETY,
    atomic_json, digest, encoded, guard, read_json, utc, validate_state)
from research_campaign.strategies import StrategySpec, fit_strategy, generate_signals, prepare_context

HEAVY_FIELDS = {"trades", "equity", "dispositions", "monthly"}


def _slim(result):
    return {k: v for k, v in result.items() if k not in HEAVY_FIELDS}


def _false_safety(value):
    return isinstance(value, dict) and set(value) == set(SAFETY) and all(v is False for v in value.values())


def _terminal(directory):
    """Reject a sealed or unfinished campaign before opening any final file."""
    guard()
    state = read_json(directory / "state.json")
    validate_state(state)
    if (state.get("status") != "COMPLETED" or state.get("completed_cycles") != TARGET_CYCLES
            or type(state.get("completed_cycles")) is not int
            or state.get("final_accessed") is not True or not _false_safety(state.get("safety"))
            or time.time() < state["started_epoch"] + TARGET_CYCLES * INTERVAL_SECONDS):
        raise ValueError("Final-only replay denied: require COMPLETED, 32 actual cycles, eight hours and final_accessed=true")
    if not state.get("final_artifact") or not state.get("selection_sha256"):
        raise ValueError("Terminal campaign lacks committed final family or selection")
    return state


def _environment(directory, expected=None):
    state = _terminal(directory)
    declaration = read_json(directory / "declaration.json")
    authoritative, verification = original._verified_environment(directory, declaration)
    if state != authoritative:
        raise ValueError("Authoritative campaign state changed during verification")
    selected_path = directory / "selection_frozen.json"
    if digest(selected_path) != state["selection_sha256"]:
        raise ValueError("Frozen final selection hash changed")
    selection = read_json(selected_path)
    if (selection.get("source") != state["declaration_sha256"]
            or selection.get("final_data_accessed") is not False):
        raise ValueError("Selection was not frozen before final access")
    identity = {"campaign_directory": str(directory), "declaration_sha256": state["declaration_sha256"],
                "selection_sha256": state["selection_sha256"], "final_artifact": state["final_artifact"],
                "source_manifest_sha256": declaration["source_manifest_sha256"],
                "source_pins": declaration["source_pins"], "runtime": declaration["runtime"],
                "helper_sha256": digest(Path(__file__)), "helper_schema_version": 1}
    if expected is not None and encoded(identity) != encoded(expected):
        raise ValueError("Final replay campaign, helper version or frozen environment changed")
    return state, declaration, selection, identity, verification


def _plan(directory, state, declaration, selection):
    family = original._read_artifact(directory, state["final_artifact"])
    selected = {g: v for g, v in selection["selection"].items() if v.get("status") == "TESTED"}
    results = family.get("results", [])
    groups = [r.get("selection_group") for r in results]
    measurements = family.get("original_measurements", {})
    if (family.get("selected_before_access") is not True or not _false_safety(family.get("safety"))
            or family.get("selection_sha256") != state["selection_sha256"]
            or family.get("family_size") != len(results) or len(groups) != len(set(groups))
            or set(groups) != set(selected) or set(measurements) != set(selected)):
        raise ValueError("Committed final-family coverage differs from frozen TESTED selection")
    if datetime.fromisoformat(selection["frozen_at"]) > datetime.fromisoformat(family["finalized_at"]):
        raise ValueError("Final-family timestamp precedes frozen selection")
    plan, folds = [], []
    expected = {r["selection_group"]: _slim(r) for r in results}
    artifacts = set()
    for group in sorted(selected):
        winner = selected[group]
        spec = StrategySpec.from_dict(winner["spec"])
        expected_group = f'{winner["symbol"]}/{winner["timeframe"]}/{spec.family}'
        if group != expected_group or winner.get("strategy_id") != spec.id:
            raise ValueError("Selected final policy identity mismatch")
        artifact = measurements[group]
        artifact_key = (artifact["path"], artifact["sha256"])
        if artifact_key in artifacts:
            raise ValueError("Two final selection groups share one original measurement")
        artifacts.add(artifact_key)
        row = declaration["datasets"][winner["symbol"]]
        common = {"selection_group": group, "symbol": winner["symbol"],
                     "timeframe": winner["timeframe"], "strategy_id": spec.id,
                     "source_file": row["requested_file"], "source_sha256": row["requested_sha256"]}
        plan.append({**common, "phase": "final", "artifact": artifact})
        saved_folds = selection["walk_forward"].get(group, [])
        if len(saved_folds) != 3:
            raise ValueError("Selected policy lacks three frozen development folds")
        for number, fold in enumerate(saved_folds, 1):
            if fold.get("label") != f"wf{number}" or fold.get("status") not in {"TESTED", "INSUFFICIENT_DATA"}:
                raise ValueError("Frozen development-fold identity/status mismatch")
            if fold["status"] == "TESTED" and not fold.get("artifact"):
                raise ValueError("TESTED development fold lacks its actual saved artifact")
            folds.append({**common, "phase": "walk_forward", "fold_number": number,
                          "artifact": fold.get("artifact"), "fold_status": fold["status"]})
    # Only small expected summaries survive this function. The large family
    # trade/equity ledgers are released before any numerical reconstruction.
    return plan, folds, expected


def _verify_sources(plan):
    seen = set()
    for item in plan:
        pair = (item["source_file"], item["source_sha256"])
        if pair not in seen:
            if digest(Path(pair[0])) != pair[1]:
                raise ValueError("Final replay original price source changed")
            seen.add(pair)


@contextmanager
def _output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".replay.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another final replay owns this receipt directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_receipt(receipt, identity):
    report = receipt["report"]
    expected = identity["measurement"]
    verification = report.get("verification", {})
    if (receipt.get("schema_version") != 1 or receipt.get("status") != "PASS"
            or encoded(receipt.get("identity")) != encoded(identity)
            or receipt.get("report_sha256") != sha256(encoded(report)).hexdigest()
            or report.get("status") != "PASS" or report.get("phase") != expected["phase"]
            or report.get("matched_measurements") != 1 or report.get("cache_used") is not False
            or report.get("artifact") != expected["artifact"]
            or report.get("selection_sha256") != identity["campaign"]["selection_sha256"]
            or report.get("symbol") != expected["symbol"] or report.get("timeframe") != expected["timeframe"]
            or report.get("strategy_id") != expected["strategy_id"]
            or report.get("source_sha256") != expected["source_sha256"]
            or verification.get("declaration_sha256") != identity["campaign"]["declaration_sha256"]
            or verification.get("source_manifest_sha256") != identity["campaign"]["source_manifest_sha256"]
            or verification.get("runtime") != identity["campaign"]["runtime"]
            or verification.get("replay_code_sha256") != identity["campaign"]["source_pins"]["replay.py"]
            or verification.get("replay_code_declared") is not True
            or not _false_safety(receipt.get("safety")) or not _false_safety(report.get("safety"))):
        raise ValueError("Saved receipt differs from exact helper version, measurement or verified source")
    return receipt


def _receipt(output, pointer, identity):
    relative = Path(pointer["path"])
    path = (output / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(output) or digest(path) != pointer["sha256"]:
        raise ValueError("Saved final replay receipt path/hash changed")
    return _validate_receipt(read_json(path), identity)


def _walk_forward(directory, declaration, selection, item, campaign_identity):
    _, _, _, _, verification = _environment(directory, campaign_identity)
    row = declaration["datasets"][item["symbol"]]
    if digest(Path(row["requested_file"])) != row["requested_sha256"]:
        raise ValueError("Walk-forward original price source changed")
    timeframe = item["timeframe"]
    bars = resample(load_m1(row["requested_file"]), timeframe)
    split = SplitPlan.from_dict(declaration.get("splits", {}).get(item["symbol"], declaration["split"]))
    span = split.validation_end - split.start
    fraction = (.4, .6, .8)[item["fold_number"] - 1]
    train_end = split.start + int(span * fraction)
    fold_end = split.start + int(span * (fraction + .2))
    training = prepare_context(bars.loc[bars.epoch + FRAMES[timeframe] <= train_end], timeframe)
    fold = selection["walk_forward"][item["selection_group"]][item["fold_number"] - 1]
    measured = None
    if len(training.bars) < 100:
        if fold != {"label": f'wf{item["fold_number"]}', "status": "INSUFFICIENT_DATA"}:
            raise ValueError("Insufficient development fold differs from original frozen fold")
    else:
        if item["fold_status"] != "TESTED":
            raise ValueError("Frozen fold falsely reports insufficient training data")
        spec = StrategySpec.from_dict(selection["selection"][item["selection_group"]]["spec"])
        prefix = prepare_context(bars.loc[bars.epoch + FRAMES[timeframe] <= fold_end], timeframe)
        fit = fit_strategy(training, spec, item["symbol"])
        signals = generate_signals(prefix, spec, item["symbol"], fit)
        # Exact production development_walk_forward defaults and provenance.
        measured = run_backtest(prefix, signals, spec, item["symbol"], timeframe, train_end, fold_end,
                                partition="walk_forward", costs=CostModel.from_dict(declaration["costs"]),
                                data_quality_ok=_quality_ok(row), provenance={"real_deriv_data": True})
        saved = original._read_artifact(directory, item["artifact"])
        original._compare(saved, measured, "complete development-fold numerical replay")
        original._compare(fold["metrics"], measured["metrics"], "frozen development-fold summary")
        from research_campaign.engine import risk_adjusted_score
        original._compare(fold["validation_score"], risk_adjusted_score(measured), "frozen fold validation score")
    _environment(directory, campaign_identity)
    if digest(Path(row["requested_file"])) != row["requested_sha256"]:
        raise ValueError("Walk-forward source changed while reconstructing")
    return {"status": "PASS", "phase": "walk_forward", "artifact": item["artifact"],
            "symbol": item["symbol"], "timeframe": timeframe, "strategy_id": item["strategy_id"],
            "source_sha256": item["source_sha256"], "selection_sha256": campaign_identity["selection_sha256"],
            "verification": verification, "matched_measurements": 1,
            "matched_partitions": int(measured is not None), "backtests_run": int(measured is not None),
            "matched_trades": len(measured["trades"]) if measured else 0,
            "matched_equity_points": len(measured["equity"]) if measured else 0,
            "matched_dispositions": len(measured["dispositions"]) if measured else 0,
            "cache_used": False, "replayed_at": utc(), "safety": guard(),
            "scope": "full saved development-fold replay after policy selection; not independent OOS",
            "real_execution_costs": "NOT_VERIFIED_BY_REPLAY"}


def _holm(directory, originals, expected, selection, selection_sha256):
    results = []
    for item in originals:
        saved = original._read_artifact(directory, item["artifact"])
        if saved["identity"]["phase"] != "final" or saved["spec"]["id"] != item["strategy_id"]:
            raise ValueError("Final original measurement does not match selected phase/policy")
        value = _slim(saved["final"])
        value.update(measured_at=saved["tested_at"], selection_group=item["selection_group"], selection_sha256=selection_sha256,
                     development_walk_forward=selection["walk_forward"][item["selection_group"]])
        results.append(value)
    results = finalize_multiplicity(results)
    for result in results:
        folds = result["development_walk_forward"]
        stable = len(folds) == 3 and all(f.get("status") == "TESTED" and
            f["metrics"]["net_profit"] > 0 and f["metrics"]["profit_factor"] is not None and
            f["metrics"]["profit_factor"] >= 1.10 for f in folds)
        result["gates"]["development_walk_forward_stability"] = stable
        result["qualified"] = all(result["gates"].values())
        result["rejection_reasons"] = [k for k, v in result["gates"].items() if not v]
        original._compare(expected[result["selection_group"]], result, "final family Holm/gates/stability")
    return {"status": "PASS", "policies_compared": len(results), "calculated_this_invocation": True,
            "scope": "joint Holm plus complete slim final-family summaries; WF source ledgers require --include-walk-forward"}


def run_final(directory, output, max_new=None, include_walk_forward=False):
    """Replay final originals; valid old receipts do not count as new work."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("Operational receipts must stay outside the frozen campaign directory")
    if max_new is not None and (type(max_new) is not int or max_new < 0):
        raise ValueError("max_new must be a nonnegative integer or None")
    state, declaration, selection, campaign_identity, _ = _environment(directory)
    originals, folds, expected = _plan(directory, state, declaration, selection)
    plan = originals + (folds if include_walk_forward else [])
    _verify_sources(originals)
    started = utc()
    with _output_lock(output):
        index_path = output / "receipts_index.json"
        index = read_json(index_path) if index_path.exists() else {
            "schema_version": 1, "campaign": campaign_identity, "receipts": {}}
        if index.get("schema_version") != 1 or encoded(index.get("campaign")) != encoded(campaign_identity):
            raise ValueError("Receipt index belongs to another helper version or final campaign")
        attempted = measured = reused = recovered = 0
        completed = []
        errors = []
        expected_keys = {sha256(encoded({"campaign": campaign_identity, "measurement": item})).hexdigest()
                         for item in originals + folds}
        phase_counts = {phase: {"measured_now": 0, "backtests_now": 0, "reused": 0, "verified_total": 0}
                        for phase in ("final", "walk_forward")}
        holm = {"status": "NOT_RUN", "reason": "final originals not yet completely verified"}
        try:
            for item in plan:
                identity = {"campaign": campaign_identity, "measurement": item}
                name = sha256(encoded(identity)).hexdigest()
                path = output / "receipts" / f"{name}.json"
                pointer = index["receipts"].get(name)
                if pointer is not None or path.exists():
                    orphan = pointer is None
                    pointer = pointer or {"path": str(path.relative_to(output)), "sha256": digest(path)}
                    _receipt(output, pointer, identity)
                    if item["artifact"] is not None:
                        original._read_artifact(directory, item["artifact"])
                    reused += 1
                    phase_counts[item["phase"]]["reused"] += 1
                    recovered += int(orphan)
                    if orphan:
                        index["receipts"][name] = pointer
                        atomic_json(index_path, index)
                elif max_new is None or attempted < max_new:
                    attempted += 1
                    report = (original.replay_measurement(directory, item["artifact"], declaration,
                                                         final_access_allowed=True) if item["phase"] == "final" else
                              _walk_forward(directory, declaration, selection, item, campaign_identity))
                    if digest(Path(__file__)) != campaign_identity["helper_sha256"]:
                        raise ValueError("Operational helper changed while replaying")
                    receipt = {"schema_version": 1, "status": "PASS", "identity": identity,
                               "completed_at": utc(), "report": report,
                               "report_sha256": sha256(encoded(report)).hexdigest(), "safety": guard()}
                    _validate_receipt(receipt, identity)
                    atomic_json(path, receipt, exclusive=True)
                    pointer = {"path": str(path.relative_to(output)), "sha256": digest(path)}
                    _receipt(output, pointer, identity)
                    index["receipts"][name] = pointer
                    atomic_json(index_path, index)
                    measured += 1
                    phase_counts[item["phase"]]["measured_now"] += 1
                    phase_counts[item["phase"]]["backtests_now"] += report.get("backtests_run", report["matched_partitions"])
                else:
                    continue
                completed.append({"selection_group": item["selection_group"], "receipt": pointer})
                phase_counts[item["phase"]]["verified_total"] += 1
            if set(index["receipts"]) - expected_keys:
                raise ValueError("Receipt index contains undeclared final measurements")
            if phase_counts["final"]["verified_total"] == len(originals):
                holm = _holm(directory, originals, expected, selection, state["selection_sha256"])
            _environment(directory, campaign_identity)
            if digest(directory / state["final_artifact"]["path"]) != state["final_artifact"]["sha256"]:
                raise ValueError("Final-family artifact changed during replay")
            _verify_sources(originals)
        except Exception as error:
            errors.append({"type": type(error).__name__, "message": str(error)})
        status = "FAIL" if errors else "PASS" if len(completed) == len(plan) else "PARTIAL"
        summary = {"schema_version": 1, "status": status, "started_at": started, "completed_at": utc(),
                   "scope": "final originals plus joint Holm; selected development WF folds only when explicitly requested; no development-search inventory",
                   "campaign": campaign_identity, "expected_final_measurements": len(originals),
                   "expected_walk_forward_folds": len(folds), "include_walk_forward": include_walk_forward,
                   "requested_measurements_and_folds": len(plan), "phase_counts": phase_counts, "holm": holm,
                   "attempted_now": attempted, "measured_now": measured,
                   "backtests_now": sum(v["backtests_now"] for v in phase_counts.values()),
                   "reused_verified_receipts": reused, "recovered_orphan_receipts": recovered,
                   "completed_total_verified_this_invocation": len(completed),
                   "remaining_requested_measurements_and_folds": len(plan) - len(completed),
                   "remaining_final_measurements": len(originals) - phase_counts["final"]["verified_total"],
                   "remaining_walk_forward_folds": len(folds) - phase_counts["walk_forward"]["verified_total"],
                   "all_final_measurements_replayed": not errors and phase_counts["final"]["verified_total"] == len(originals),
                   "selected_final_and_WF_reproduction_complete": not errors and holm["status"] == "PASS" and
                       include_walk_forward and phase_counts["walk_forward"]["verified_total"] == len(folds),
                   "all_development_search_measurements_replayed": False,
                   "max_new": max_new, "receipts": completed, "errors": errors, "safety": guard(),
                   "formulas": "original frozen replay_measurement, not an independent mathematical implementation",
                   "real_execution_costs": "NOT_VERIFIED_BY_REPLAY"}
        atomic_json(output / f"run-{time.time_ns()}.json", summary, exclusive=True)
        atomic_json(output / "latest.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "data/single_frame_campaign_20261011")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/final_replay_20261011")
    parser.add_argument("--max-new", type=int, help="Explicit cap on new computations; old receipts are still verified")
    parser.add_argument("--include-walk-forward", action="store_true", help="Also numerically replay all three saved development folds of each selected policy")
    args = parser.parse_args()
    try:
        report = run_final(args.state_dir, args.output_dir, args.max_new, args.include_walk_forward)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "REJECTED_PRECONDITIONS", "error": str(error), "safety": guard()}))
        raise SystemExit(1)
    print(json.dumps({k: v for k, v in report.items() if k != "receipts"}, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
