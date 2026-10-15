"""Resumable 32-cycle single-frame search; final data stays sealed during tuning."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from .data import TARGETS, FRAMES, ENDPOINT, load_m1, resample, update_history
from .engine import CostModel, SplitPlan, evaluate_final, finalize_multiplicity, risk_adjusted_score, run_backtest
from .state import (INTERVAL_SECONDS, TARGET_CYCLES, SAFETY, atomic_json, campaign_lock,
                    digest, encoded, guard, key, live_owner, next_due, read_json, utc, validate_state)
from .strategies import StrategySpec, candidates_for_round, fit_strategy, generate_signals, prepare_context

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "data/single_frame_campaign_20261011"
DEFAULT_PUBLIC = ROOT / "frontend/public/campaign"
CODE_FILES = ("campaign.py", "data.py", "engine.py", "strategies.py", "state.py", "reporting.py", "source_audit.py", "replay.py")


def split_for(declaration, symbol):
    return SplitPlan.from_dict(declaration.get("splits", {}).get(symbol, declaration["split"]))


def source_pins():
    return {name: digest(Path(__file__).with_name(name)) for name in CODE_FILES}


def verify_receipt(receipt):
    if receipt.get("endpoint") != ENDPOINT or receipt.get("safety") != SAFETY or receipt.get("authentication_used") is not False:
        raise ValueError("Real public-only source receipt required")
    path = Path(receipt["wire_file"])
    if digest(path) != receipt["file_sha256"]:
        raise ValueError("Raw public wire file changed")
    import hashlib
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != receipt["wire_sha256"]:
        raise ValueError("Decoded wire hash mismatch")
    response = json.loads(raw)
    if response.get("req_id") != receipt["request"].get("req_id"):
        raise ValueError("Raw public response request identity mismatch")
    if response.get("error"):
        raise ValueError("Error response cannot establish real history")
    return response


def read_artifact(directory, artifact):
    path = directory / artifact["path"]
    if digest(path) != artifact["sha256"]:
        raise ValueError("Saved research artifact changed")
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def save_artifact(directory, name, value):
    path = directory / "artifacts" / f"{name}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(encoded(value), mtime=0)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Immutable artifact collision")
    else:
        descriptor, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        temporary = Path(temp_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return {"path": str(path.relative_to(directory)), "sha256": digest(path)}


def initialize(directory, initial_manifest):
    guard()
    if (directory / "state.json").exists():
        raise ValueError("Existing campaign cannot be initialized again")
    manifest = read_json(initial_manifest)
    if manifest.get("source") != "Official Deriv public M1 candles" or manifest.get("endpoint") != ENDPOINT:
        raise ValueError("Only official public Deriv history may initialize this campaign")
    verify_receipt(manifest["catalogue"]["provenance"] if "provenance" in manifest["catalogue"] else manifest["catalogue"])
    if manifest.get("safety") != SAFETY or not any(v.get("status") == "READY" for v in manifest["symbols"].values()):
        raise ValueError("Need actual public history before starting the campaign clock")
    ready = {}
    for symbol, row in manifest["symbols"].items():
        if row.get("status") == "READY":
            if digest(Path(row["requested_file"])) != row["requested_sha256"]:
                raise ValueError("Initial source snapshot hash mismatch")
            # A cache-only update may retain its original wire receipts in
            # pinned ancestors; the independent audit reconstructs the chain.
            for page in row.get("pages", []):
                response = verify_receipt(page)
                if page["request"].get("ticks_history") != row["api_symbol"] or "candles" not in response:
                    raise ValueError("Original candle page does not match independently discovered symbol")
            from .source_audit import audit_source_row
            catalogue = manifest["catalogue"].get("provenance", manifest["catalogue"])
            audit = audit_source_row(row, catalogue_provenance=catalogue)
            ready[symbol] = {**row, "study_quality": audit}
    split = SplitPlan.from_calendar(manifest["start_epoch"], manifest["cutoff_exclusive_epoch"])
    splits = {s: SplitPlan.from_calendar(r["study_quality"]["effective_start_epoch"],
                                       r["study_quality"]["effective_end_exclusive_epoch"]).to_dict()
              for s, r in ready.items()}
    declaration = {
        "schema_version": 1, "declared_at": utc(), "source_manifest": str(Path(initial_manifest).resolve()),
        "source_manifest_sha256": digest(Path(initial_manifest)), "datasets": ready,
        "symbol_catalogue": manifest["symbols"], "split": split.to_dict(), "splits": splits,
        "candidate_rounds": [[s.to_dict() for s in candidates_for_round(r)] for r in range(TARGET_CYCLES)],
        "timeframes": dict(FRAMES), "source_pins": source_pins(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "costs": CostModel().to_dict(), "safety": guard(),
        "final_bootstrap_repetitions": 99999,
        "fresh_out_of_sample": False, "history_exposure": "Adaptive history; previously exposed intervals remain disclosed",
        "final_access_before_selection": False, "selection": "validation-only return/drawdown score with sample penalty",
        "effective_calendars": "per-symbol observed reliable endpoints, shared across that symbol's frames; no internal gaps discarded",
        "walk_forward": "three expanding development folds for each validation-selected symbol/frame/family before final access",
        "final_hypothesis_family": "all validation-selected symbol/frame/family policies, joint Holm correction",
        "cycle_interval_seconds": INTERVAL_SECONDS, "target_cycles": TARGET_CYCLES,
        "new_updates": "cached and inventoried every cycle; never move frozen split or input data",
        "trailing_stop": False, "live_execution": False,
    }
    declaration_path = directory / "declaration.json"
    if declaration_path.exists():
        # A crash may commit the immutable declaration before the initial state.
        # Reconstruct and validate every scientific field from the same pinned
        # source, runtime and current code; retain the original declaration time.
        committed = read_json(declaration_path)
        expected_fields = {k: v for k, v in declaration.items() if k != "declared_at"}
        committed_fields = {k: v for k, v in committed.items() if k != "declared_at"}
        if (not isinstance(committed.get("declared_at"), str)
                or encoded(committed_fields) != encoded(expected_fields)):
            raise ValueError("Uncommitted initialization declaration differs from pinned inputs or scientific plan")
        declaration = committed
    else:
        atomic_json(declaration_path, declaration, exclusive=True)
    start = time.time()
    state = {
        "id": "single-frame-" + key(declaration)[:12], "declaration_sha256": digest(directory / "declaration.json"),
        "started_epoch": start, "started_at": utc(start), "deadline_at": utc(start + TARGET_CYCLES * INTERVAL_SECONDS),
        "target_cycles": TARGET_CYCLES, "interval_seconds": INTERVAL_SECONDS, "completed_cycles": 0,
        "cycles": [], "status": "RUNNING", "winners": {}, "coverage": {}, "final_accessed": False,
        "safety": guard(), "updated_at": utc(), "runner_pid": os.getpid(),
    }
    atomic_json(directory / "state.json", state, exclusive=True)
    return state


def frozen(directory, state):
    validate_state(state)
    path = directory / "declaration.json"
    if digest(path) != state["declaration_sha256"]:
        raise ValueError("Campaign declaration changed")
    declaration = read_json(path)
    if declaration["source_pins"] != source_pins():
        raise ValueError("Frozen campaign code changed; preserve this run rather than silently reinterpret it")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    if declaration["runtime"] != runtime:
        raise ValueError("Frozen numerical runtime changed")
    return declaration


def status(directory):
    if not (directory / "state.json").exists():
        return {"status": "NOT_STARTED", "completed_cycles": 0, "target_cycles": TARGET_CYCLES,
                "owner": live_owner(directory), "safety": guard()}
    state = read_json(directory / "state.json")
    validate_state(state)
    return {k: state.get(k) for k in ("id", "status", "completed_cycles", "target_cycles", "started_at",
                                     "deadline_at", "updated_at", "final_accessed", "failure")} | {
        "next_cycle_at": utc(next_due(state)) if state["completed_cycles"] < TARGET_CYCLES else None,
        "owner": live_owner(directory), "safety": guard(),
    }


def _save_state(directory, state, public):
    state["updated_at"] = utc()
    state["runner_pid"] = os.getpid()
    validate_state(state)
    atomic_json(directory / "state.json", state)
    from .reporting import publish_snapshot
    publish_snapshot(directory, public, state)


def _source_frame(row, timeframe):
    source = Path(row["requested_file"])
    if digest(source) != row["requested_sha256"]:
        raise ValueError("Frozen real source changed")
    return resample(load_m1(source), timeframe)


def _quality_ok(row):
    if "study_quality" in row:
        q = row["study_quality"]
        return q["internal_missing_minutes"] == 0 and q.get("internal_invalid_rows", 0) == 0
    q = row.get("quality", {})
    excluded = q.get("raw_source_excluded_counts", q.get("excluded_counts", {}))
    return q.get("missing_minutes", 1) == 0 and not excluded.get("off_grid", 0) and not q.get("conflicting_duplicates", 0)


def prepared_contexts(bars, split, timeframe, phase):
    train = bars.loc[bars["epoch"] + FRAMES[timeframe] <= split.train_end].copy()
    end = split.validation_end if phase == "development" else split.end
    prefix = bars.loc[bars["epoch"] + FRAMES[timeframe] <= end].copy()
    n = len(train) + (len(prefix) - len(train)) // 2
    return {"train": train, "training_context": prepare_context(train, timeframe),
            "context": prepare_context(prefix, timeframe),
            "prefix_context": prepare_context(prefix.iloc[:n], timeframe), "n": n}


def measure(directory, declaration, symbol, timeframe, spec, bars, phase="development", prepared=None):
    split = split_for(declaration, symbol)
    identity = {"symbol": symbol, "timeframe": timeframe, "scientific_spec": spec.scientific_id,
                "input_sha256": declaration["datasets"][symbol]["requested_sha256"],
                "split": split.to_dict(), "phase": phase, "source_pins": declaration["source_pins"]}
    cache_key = key(identity)
    cache_path = directory / "cache" / f"{cache_key}.json"
    if cache_path.exists():
        artifact = read_json(cache_path)
        result = read_artifact(directory, artifact)
        if result["identity"] != identity:
            raise ValueError("Cached scientific identity mismatch")
        return result, artifact, True
    orphan = directory / "artifacts" / f"{cache_key}.json.gz"
    if orphan.exists():
        artifact = {"path": str(orphan.relative_to(directory)), "sha256": digest(orphan)}
        result = read_artifact(directory, artifact)
        if result.get("identity") != identity:
            raise ValueError("Unpointed artifact scientific identity mismatch")
        atomic_json(cache_path, artifact, exclusive=True)
        return result, artifact, True
    row = declaration["datasets"][symbol]
    train = bars.loc[bars["epoch"] + FRAMES[timeframe] <= split.train_end]
    if len(train) < 100:
        result = {"identity": identity, "spec": spec.to_dict(), "symbol": symbol, "timeframe": timeframe,
                  "status": "INSUFFICIENT_DATA", "reason": "fewer_than_100_closed_training_bars", "tested_at": utc()}
    else:
        prepared = prepared or prepared_contexts(bars, split, timeframe, phase)
        training_context = prepared["training_context"]
        fit = fit_strategy(training_context, spec, symbol)
        # The search process never even constructs indicator arrays on final bars.
        context = prepared["context"]
        signals = generate_signals(context, spec, symbol, fit)
        n = prepared["n"]
        prefix_context = prepared["prefix_context"]
        causal = np.array_equal(signals[:n], generate_signals(prefix_context, spec, symbol, fit))
        if not causal:
            raise ValueError("Actual-history prefix invariance failed")
        provenance = {"real_deriv_data": True, "source_file": row["requested_file"],
                      "source_sha256": row["requested_sha256"], "source_manifest": row["manifest_file"],
                      "api_symbol": row["api_symbol"], "fresh_oos": False,
                      "actual_prefix_invariance_checked": True}
        common = {"costs": CostModel.from_dict(declaration["costs"]), "data_quality_ok": _quality_ok(row),
                  "provenance": provenance}
        result = {"identity": identity, "spec": spec.to_dict(), "symbol": symbol, "timeframe": timeframe,
                  "status": "TESTED", "tested_at": utc(), "fit": fit.to_dict() if fit else None}
        result["signal_fingerprint"] = {
            "sha256": hashlib.sha256(signals.astype(np.int8).tobytes()).hexdigest(),
            "length": len(signals), "encoding": "signed-int8-chronological",
            "epoch_sha256": hashlib.sha256(context.bars["epoch"].to_numpy(dtype="<i8").tobytes()).hexdigest(),
            "last_available_close_epoch": int(context.bars["epoch"].iloc[-1] + FRAMES[timeframe]),
        }
        phases = ("train", "validation") if phase == "development" else ("final",)
        for part in phases:
            start, finish = split.bounds(part)
            measured = run_backtest(context, signals, spec, symbol, timeframe, start, finish, partition=part,
                                    bootstrap_repetitions=declaration["final_bootstrap_repetitions"], **common)
            measured["reproducibility"].update(signal_causality_verified=causal, raw_source_hash_verified=True)
            measured["gates"] = evaluate_final(measured)
            measured["qualified"] = all(measured["gates"].values())
            measured["rejection_reasons"] = [k for k, value in measured["gates"].items() if not value]
            result[part] = measured
        if phase == "development":
            result["validation_score"] = risk_adjusted_score(result["validation"])
    artifact = save_artifact(directory, cache_key, result)
    atomic_json(cache_path, artifact, exclusive=True)
    return result, artifact, False


def consider_winner(state, summary):
    group = f'{summary["symbol"]}/{summary["timeframe"]}/{summary["family"]}'
    old = state["winners"].get(group)
    if old is None or (summary["validation_score"] is not None and (
            old.get("validation_score") is None or summary["validation_score"] > old["validation_score"])):
        state["winners"][group] = summary
        return True
    return False


def execute_cycle(directory, public, state, *, update=True):
    declaration = frozen(directory, state)
    index = state["completed_cycles"]
    if index >= TARGET_CYCLES:
        return
    if time.time() < next_due(state):
        return
    cycle_file = directory / "cycles" / f"{index + 1:02d}.json"
    cycle = read_json(cycle_file) if cycle_file.exists() else {
        "number": index + 1, "status": "RUNNING", "started_at": utc(), "due_at": utc(next_due(state)),
        "tested_count": 0, "cache_hits": 0, "coverage": {}, "results": [],
    }
    if cycle["status"] == "COMPLETED":
        # The cycle can commit before its state counter. Preserve the original
        # completion receipt instead of giving a completed run a new timestamp.
        if (cycle["number"] != index + 1 or not cycle.get("completed_at")
                or len(cycle["coverage"]) != len(TARGETS) * len(FRAMES)):
            raise ValueError("Incomplete committed cycle cannot be recovered")
        for summary in cycle["results"]:
            read_artifact(directory, summary["artifact"])
            consider_winner(state, summary)
        state["coverage"].update(cycle["coverage"])
        state["cycles"].append({k: cycle[k] for k in ("number", "status", "started_at", "completed_at", "tested_count", "notes")})
        state["completed_cycles"] += 1
        state["current_cycle"] = None
        state["status"] = "RUNNING"
        _save_state(directory, state, public)
        return
    atomic_json(cycle_file, cycle)
    state["current_cycle"] = index + 1
    state["status"] = "RUNNING"
    _save_state(directory, state, public)
    if update and "market_update" not in cycle:
        try:
            update_result = asyncio.run(update_history(directory / "history", days=declaration_source_days(declaration)))
            cycle["market_update"] = {"manifest_file": update_result["manifest_file"],
                                      "manifest_sha256": update_result["manifest_sha256"],
                                      "statuses": {s: r["status"] for s, r in update_result["symbols"].items()}}
        except Exception as error:
            cycle["market_update"] = {"status": "ERROR", "error": f"{type(error).__name__}: {error}"}
        atomic_json(cycle_file, cycle)
    specs = [StrategySpec.from_dict(v) for v in declaration["candidate_rounds"][index]]
    for summary in cycle["results"]:
        read_artifact(directory, summary["artifact"])
        consider_winner(state, summary)
    done = {(r["symbol"], r["timeframe"], r.get("attempted_strategy_id", r["strategy_id"])) for r in cycle["results"]}
    for symbol in TARGETS:
        row = declaration["datasets"].get(symbol)
        split = split_for(declaration, symbol)
        m1 = None
        if row is not None:
            if digest(Path(row["requested_file"])) != row["requested_sha256"]:
                raise ValueError("Frozen real source changed")
            m1 = load_m1(row["requested_file"])
        for timeframe in FRAMES:
            cell = f"{symbol}/{timeframe}"
            if row is None:
                cycle["coverage"][cell] = {"status": "INSUFFICIENT_DATA", "reason": "no_ready_real_source"}
                state["coverage"][cell] = cycle["coverage"][cell]
                continue
            bars = resample(m1, timeframe)
            prepared = prepared_contexts(bars, split, timeframe, "development") if len(bars) else None
            cycle["coverage"][cell] = {"status": "TESTED", "rows": len(bars),
                                      "dropped_incomplete": bars.attrs.get("dropped_incomplete_bars")}
            state["coverage"][cell] = cycle["coverage"][cell]
            for spec in specs:
                if (symbol, timeframe, spec.id) in done:
                    continue
                result, artifact, hit = measure(directory, declaration, symbol, timeframe, spec, bars, prepared=prepared)
                actual_spec = StrategySpec.from_dict(result["spec"])
                summary = {"symbol": symbol, "timeframe": timeframe, "strategy_id": actual_spec.id,
                           "attempted_strategy_id": spec.id,
                           "family": spec.family, "status": result["status"], "artifact": artifact,
                           "validation_score": result.get("validation_score"), "tested_at": result["tested_at"],
                           "cache_hit": hit, "spec": actual_spec.to_dict(),
                           "training_metrics": result.get("train", {}).get("metrics"),
                           "validation_metrics": result.get("validation", {}).get("metrics")}
                cycle["results"].append(summary)
                cycle["tested_count"] += int(result["status"] == "TESTED" and not hit)
                cycle["cache_hits"] += int(hit)
                if consider_winner(state, summary):
                    from .reporting import publish_detail
                    summary["details_url"] = publish_detail(public, artifact, result)
                # Artifact and cycle checkpoint precede the state checkpoint.
                atomic_json(cycle_file, cycle)
            _save_state(directory, state, public)
            print(json.dumps({"cycle": index + 1, "cell": cell, "candidates": cycle["tested_count"],
                              "completed_cycles": state["completed_cycles"]}), flush=True)
    if len(cycle["coverage"]) != len(TARGETS) * len(FRAMES):
        raise ValueError("Cycle has not assessed every symbol/frame")
    cycle.update(status="COMPLETED", completed_at=utc(), notes=[
        "Training/validation search only; final holdout remains sealed",
        "Point-price execution and assumed costs do not verify CFD profits",
    ])
    atomic_json(cycle_file, cycle)
    state["cycles"].append({k: cycle[k] for k in ("number", "status", "started_at", "completed_at", "tested_count", "notes")})
    state["completed_cycles"] += 1
    state["current_cycle"] = None
    _save_state(directory, state, public)


def declaration_source_days(declaration):
    split = declaration["split"]
    return max(1, (split["end"] - split["start"]) // 86400)


def development_walk_forward(directory, declaration, selected):
    rows = {}
    previous_cell, contexts = None, []
    for group, winner in selected.items():
        if winner["status"] != "TESTED":
            rows[group] = []
            continue
        symbol, timeframe = winner["symbol"], winner["timeframe"]
        split = split_for(declaration, symbol)
        span = split.validation_end - split.start
        spec = StrategySpec.from_dict(winner["spec"])
        cell = (symbol, timeframe)
        if cell != previous_cell:
            bars = _source_frame(declaration["datasets"][symbol], timeframe)
            contexts = []
            for fraction in (.4, .6, .8):
                train_end = split.start + int(span * fraction)
                fold_end = split.start + int(span * (fraction + .2))
                train = bars.loc[bars["epoch"] + FRAMES[timeframe] <= train_end]
                prefix = bars.loc[bars["epoch"] + FRAMES[timeframe] <= fold_end]
                contexts.append((train_end, fold_end, prepare_context(train, timeframe),
                                 prepare_context(prefix, timeframe)))
            previous_cell = cell
        folds = []
        for number, (train_end, fold_end, training_ctx, ctx) in enumerate(contexts, 1):
            if len(training_ctx.bars) < 100:
                folds.append({"label": f"wf{number}", "status": "INSUFFICIENT_DATA"})
                continue
            fit = fit_strategy(training_ctx, spec, symbol)
            signals = generate_signals(ctx, spec, symbol, fit)
            measured = run_backtest(ctx, signals, spec, symbol, timeframe, train_end, fold_end,
                                    partition="walk_forward", costs=CostModel.from_dict(declaration["costs"]),
                                    data_quality_ok=_quality_ok(declaration["datasets"][symbol]),
                                    provenance={"real_deriv_data": True})
            artifact = save_artifact(directory, key({"group": group, "fold": number,
                                                     "selected_id": spec.id, "declaration": digest(directory / "declaration.json")}), measured)
            folds.append({"label": f"wf{number}", "status": "TESTED", "metrics": measured["metrics"],
                          "validation_score": risk_adjusted_score(measured), "artifact": artifact})
        rows[group] = folds
    return rows


def finalize(directory, public, state):
    if state["completed_cycles"] != TARGET_CYCLES:
        raise ValueError("Cannot finalize before all 32 actual cycles")
    if time.time() < state["started_epoch"] + TARGET_CYCLES * INTERVAL_SECONDS:
        raise ValueError("The full eight-hour research window has not elapsed")
    declaration = frozen(directory, state)
    selected_path = directory / "selection_frozen.json"
    if not selected_path.exists():
        state["status"] = "DEVELOPMENT_WALK_FORWARD"
        _save_state(directory, state, public)
        walk_forward = development_walk_forward(directory, declaration, state["winners"])
        atomic_json(selected_path, {"frozen_at": utc(), "selection": state["winners"], "walk_forward": walk_forward,
                                   "source": state["declaration_sha256"], "final_data_accessed": False}, exclusive=True)
    selected = read_json(selected_path)
    if selected["source"] != state["declaration_sha256"]:
        raise ValueError("Selection lineage mismatch")
    state["selection_sha256"] = digest(selected_path)
    committed = directory / "artifacts" / ("final-family-" + state["selection_sha256"] + ".json.gz")
    if committed.exists():
        artifact = {"path": str(committed.relative_to(directory)), "sha256": digest(committed)}
        saved = read_artifact(directory, artifact)
        groups = {g for g, r in selected["selection"].items() if r["status"] == "TESTED"}
        if (saved["selection_sha256"] != state["selection_sha256"] or saved["safety"] != SAFETY
                or {r["selection_group"] for r in saved["results"]} != groups):
            raise ValueError("Committed final family does not match frozen selection")
        state.update(status="COMPLETED", final_accessed=True, completed_at=saved["finalized_at"],
                     final_artifact=artifact, qualified_policies=sum(r["qualified"] for r in saved["results"]))
        _save_state(directory, state, public)
        return
    state["final_accessed"] = True
    state["status"] = "FINAL_EVALUATION"
    _save_state(directory, state, public)
    results = []
    originals = {}
    previous_cell, prepared = None, None
    for group, winner in selected["selection"].items():
        if winner["status"] != "TESTED":
            continue
        symbol, timeframe = winner["symbol"], winner["timeframe"]
        cell = (symbol, timeframe)
        if cell != previous_cell:
            bars = _source_frame(declaration["datasets"][symbol], timeframe)
            prepared = prepared_contexts(bars, split_for(declaration, symbol), timeframe, "final")
            previous_cell = cell
        result, artifact, _ = measure(directory, declaration, symbol, timeframe,
                                      StrategySpec.from_dict(winner["spec"]), bars, phase="final", prepared=prepared)
        if result["status"] == "TESTED":
            final = result["final"]
            final["measured_at"] = result["tested_at"]
            final["selection_group"] = group
            final["selection_sha256"] = state["selection_sha256"]
            final["development_walk_forward"] = selected["walk_forward"].get(group, [])
            results.append(final)
            originals[group] = artifact
    results = finalize_multiplicity(results)
    for result in results:
        folds = result["development_walk_forward"]
        stable = len(folds) == 3 and all(f.get("status") == "TESTED" and
                    f["metrics"]["net_profit"] > 0 and f["metrics"]["profit_factor"] is not None and
                    f["metrics"]["profit_factor"] >= 1.10 for f in folds)
        result["gates"]["development_walk_forward_stability"] = stable
        result["qualified"] = all(result["gates"].values())
        result["rejection_reasons"] = [k for k, v in result["gates"].items() if not v]
    final_artifact = save_artifact(directory, "final-family-" + state["selection_sha256"], {
        "selected_before_access": True, "selection_sha256": state["selection_sha256"],
        "family_size": len(results), "results": results, "original_measurements": originals,
        "safety": guard(), "finalized_at": utc()})
    state.update(status="COMPLETED", completed_at=utc(), final_artifact=final_artifact,
                 qualified_policies=sum(r["qualified"] for r in results))
    _save_state(directory, state, public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "finalize", "verify", "replay"))
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--initial-manifest", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--replay-limit", type=int, help="Explicitly limit numerical replay; omitted replays every recorded measurement")
    args = parser.parse_args()
    guard()
    directory, public = args.state_dir.resolve(), args.public_dir.resolve()
    if args.command == "status":
        print(json.dumps(status(directory), indent=2))
        return
    if args.command in {"verify", "replay"}:
        state = read_json(directory / "state.json")
        declaration = frozen(directory, state)
        if args.command == "replay":
            from .replay import replay_campaign
            report = replay_campaign(directory, state, declaration, limit=args.replay_limit)
            destination = directory / "replays" / f"replay-{time.time_ns()}.json"
            atomic_json(destination, report, exclusive=True)
            print(json.dumps({k: v for k, v in report.items() if k != "reports"} | {"report_file": str(destination)}, indent=2))
            if report["status"] == "FAIL":
                raise SystemExit(1)
        else:
            checked = 0
            for cycle in state["cycles"]:
                record = read_json(directory / "cycles" / f'{cycle["number"]:02d}.json')
                for row in record["results"]:
                    read_artifact(directory, row["artifact"])
                    checked += 1
            if state.get("final_artifact"):
                read_artifact(directory, state["final_artifact"])
            print(json.dumps({"status": "PASS", "scope": "immutable artifact hashes; use replay for numerical recomputation",
                              "artifacts_checked": checked, "actual_cycles": state["completed_cycles"],
                              "final_accessed": state["final_accessed"], "safety": guard()}))
        return
    with campaign_lock(directory):
        if not (directory / "state.json").exists():
            if args.initial_manifest is None:
                raise ValueError("Initial real-data acquisition manifest required")
            state = initialize(directory, args.initial_manifest)
        else:
            state = read_json(directory / "state.json")
        frozen(directory, state)
        try:
            while state["status"] != "COMPLETED":
                if args.command == "finalize" or state["completed_cycles"] == TARGET_CYCLES:
                    remaining = state["started_epoch"] + TARGET_CYCLES * INTERVAL_SECONDS - time.time()
                    if remaining > 0 and args.command != "finalize":
                        if state["status"] != "WAITING_FINAL_WINDOW":
                            state["status"] = "WAITING_FINAL_WINDOW"
                            _save_state(directory, state, public)
                        if args.once:
                            break
                        time.sleep(min(30, remaining))
                        continue
                    finalize(directory, public, state)
                    break
                if time.time() >= next_due(state):
                    execute_cycle(directory, public, state)
                    if args.once:
                        break
                else:
                    if args.once:
                        break
                    # No accelerated fake cycles: real wall time controls the next run.
                    time.sleep(min(30, max(.1, next_due(state) - time.time())))
            print(json.dumps(status(directory), indent=2), flush=True)
        except Exception as error:
            state.update(status="FAILED", failure={"at": utc(), "type": type(error).__name__,
                                                   "message": str(error), "traceback": traceback.format_exc()})
            _save_state(directory, state, public)
            raise


if __name__ == "__main__":
    main()
