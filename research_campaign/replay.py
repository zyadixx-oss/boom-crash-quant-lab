"""Numerically rerun immutable campaign measurements without their result cache.

This verifies reproducibility with the frozen production formulas.  It is not a
second mathematical implementation, a source-authenticity oracle, or evidence
of verified CFD costs/fills.  No new final data is opened during development.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from .data import FRAMES, load_m1, resample
from .engine import (CostModel, SplitPlan, evaluate_final, risk_adjusted_score,
                     run_backtest)
from .state import (INTERVAL_SECONDS, TARGET_CYCLES, digest, encoded, guard,
                    read_json, utc, validate_state)
from .strategies import (StrategySpec, fit_strategy, generate_signals,
                         prepare_context)

REQUIRED_CODE_PINS = {"campaign.py", "data.py", "engine.py", "strategies.py",
                      "state.py", "reporting.py", "source_audit.py"}


class ReplayMismatch(ValueError):
    """Saved numerical evidence does not match its actual frozen-source replay."""


def _runtime() -> dict[str, str]:
    return {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}


def _verified_environment(directory: Path, declaration: dict) -> tuple[dict, dict]:
    guard()
    state=read_json(directory/"state.json")
    validate_state(state)
    declared=directory/"declaration.json"
    if digest(declared)!=state["declaration_sha256"] or encoded(read_json(declared))!=encoded(declaration):
        raise ValueError("Replay declaration differs from authoritative frozen state")
    pins=declaration["source_pins"]
    if not REQUIRED_CODE_PINS.issubset(pins):
        raise ValueError("Replay requires every declared numerical/source code pin")
    package=Path(__file__).resolve().parent
    for name, expected in pins.items():
        if Path(name).name!=name or digest(package/name)!=expected:
            raise ValueError(f"Frozen replay code changed: {name}")
    if declaration["runtime"]!=_runtime():
        raise ValueError("Frozen replay numerical runtime changed")
    source_manifest=Path(declaration["source_manifest"])
    if digest(source_manifest)!=declaration["source_manifest_sha256"]:
        raise ValueError("Frozen replay source manifest changed")
    return state,{"declaration_sha256":state["declaration_sha256"],
                  "source_manifest_sha256":declaration["source_manifest_sha256"],
                  "code_files_rehashed":len(pins),"runtime":_runtime(),
                  "replay_code_sha256":digest(Path(__file__)),
                  "replay_code_declared":pins.get("replay.py")==digest(Path(__file__))}


def _read_artifact(directory: Path, artifact: dict) -> dict:
    relative=Path(artifact["path"])
    path=(directory/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Measurement artifact must remain inside its campaign directory")
    if digest(path)!=artifact["sha256"]:
        raise ValueError("Immutable replay measurement artifact changed")
    def bad(value):
        raise ValueError(f"Invalid saved JSON constant {value}")
    with gzip.open(path,"rt") as stream:
        result=json.load(stream,parse_constant=bad)
    if not isinstance(result,dict):
        raise ValueError("Measurement artifact must contain an object")
    return result


def _first_difference(saved: Any, replayed: Any, path="$") -> str:
    if isinstance(saved,dict) and isinstance(replayed,dict):
        if set(saved)!=set(replayed):
            return f"{path}: differing keys {sorted(set(saved)^set(replayed))}"
        for key in saved:
            if saved[key]!=replayed[key]:
                return _first_difference(saved[key],replayed[key],f"{path}.{key}")
    elif isinstance(saved,list) and isinstance(replayed,list):
        if len(saved)!=len(replayed):
            return f"{path}: lengths {len(saved)} != {len(replayed)}"
        for i,(a,b) in enumerate(zip(saved,replayed)):
            if a!=b:
                return _first_difference(a,b,f"{path}[{i}]")
    return f"{path}: {str(saved)[:160]} != {str(replayed)[:160]}"


def _compare(saved: Any, replayed: Any, label: str) -> None:
    saved_bytes,replayed_bytes=encoded(saved),encoded(replayed)
    if saved_bytes!=replayed_bytes:
        # Normalize JSON-equivalent tuples/numpy scalars before locating the
        # first *numerical* mismatch in a saved JSON artifact.
        a,b=json.loads(saved_bytes),json.loads(replayed_bytes)
        raise ReplayMismatch(f"{label}: {_first_difference(a,b)}")


def _split(declaration: dict, symbol: str) -> SplitPlan:
    return SplitPlan.from_dict(declaration.get("splits",{}).get(symbol,declaration["split"]))


def _quality_ok(row: dict) -> bool:
    if "study_quality" in row:
        q=row["study_quality"]
        return q["internal_missing_minutes"]==0 and q.get("internal_invalid_rows",0)==0
    q=row.get("quality",{})
    excluded=q.get("raw_source_excluded_counts",q.get("excluded_counts",{}))
    return q.get("missing_minutes",1)==0 and not excluded.get("off_grid",0) and not q.get("conflicting_duplicates",0)


def _authorize_final(directory: Path, state: dict, declaration: dict,
                     symbol: str, timeframe: str, spec: StrategySpec,
                     final_access_allowed: bool) -> str:
    if not final_access_allowed or state.get("final_accessed") is not True:
        raise ValueError("Final replay denied: authoritative holdout remains sealed")
    if state["completed_cycles"]!=TARGET_CYCLES or time.time()<state["started_epoch"]+TARGET_CYCLES*INTERVAL_SECONDS:
        raise ValueError("Final replay denied before 32 completed cycles and eight hours")
    path=directory/"selection_frozen.json"
    if digest(path)!=state.get("selection_sha256"):
        raise ValueError("Final replay selection hash changed or is missing")
    selection=read_json(path)
    if selection.get("source")!=state["declaration_sha256"] or selection.get("final_data_accessed") is not False:
        raise ValueError("Final replay selection was not pinned before final access")
    group=f"{symbol}/{timeframe}/{spec.family}"
    winner=selection.get("selection",{}).get(group)
    if not winner or winner.get("status")!="TESTED" or winner.get("strategy_id")!=spec.id:
        raise ValueError("Final replay policy was not selected before holdout access")
    _compare(winner["spec"],spec.to_dict(),"frozen final selection spec")
    return state["selection_sha256"]


def _fingerprint(context, signals: np.ndarray, timeframe: str) -> dict:
    epochs=context.bars.epoch.to_numpy(dtype="<i8")
    return {"sha256":sha256(signals.astype(np.int8).tobytes()).hexdigest(),
            "epoch_sha256":sha256(epochs.tobytes()).hexdigest(),"length":len(signals),
            "last_available_close_epoch":int(epochs[-1]+FRAMES[timeframe]) if len(epochs) else None,
            "encoding":"signed-int8-chronological"}


def replay_measurement(directory, artifact, declaration, final_access_allowed=False) -> dict:
    """Actually reconstruct and rerun one pinned campaign.measure artifact.

    Every saved numerical result field is compared exactly, including the full
    virtual trade/equity/disposition ledgers, references and uncertainty.  Only
    the outer measurement's tested_at runtime timestamp is excluded.  Legacy
    artifacts without vector fingerprints get disposition-complete in-period
    signal checks, explicitly distinguished from a full-vector hash comparison.
    Verification failures raise ValueError/ReplayMismatch rather than PASS.
    """
    directory=Path(directory).resolve()
    state,verification=_verified_environment(directory,declaration)
    saved=_read_artifact(directory,artifact)
    symbol=saved["symbol"]; timeframe=saved["timeframe"]
    spec=StrategySpec.from_dict(saved["spec"])
    if timeframe not in declaration["timeframes"] or timeframe not in FRAMES:
        raise ValueError("Undeclared replay timeframe")
    declared_specs=[v for row in declaration["candidate_rounds"] for v in row]
    if not any(v.get("id")==spec.id and encoded(v)==encoded(spec.to_dict()) for v in declared_specs):
        raise ValueError("Replay strategy was not predeclared")
    row=declaration["datasets"][symbol]
    split=_split(declaration,symbol)
    phase=saved["identity"]["phase"]
    if phase not in {"development","final"}:
        raise ValueError("Unknown saved measurement phase")
    selection_sha=None
    if phase=="final":
        selection_sha=_authorize_final(directory,state,declaration,symbol,timeframe,spec,final_access_allowed)
    identity={"symbol":symbol,"timeframe":timeframe,"scientific_spec":spec.scientific_id,
              "input_sha256":row["requested_sha256"],"split":split.to_dict(),
              "phase":phase,"source_pins":declaration["source_pins"]}
    _compare(saved["identity"],identity,"measurement identity")
    source=Path(row["requested_file"])
    if digest(source)!=row["requested_sha256"]:
        raise ValueError("Frozen replay price source changed")
    # Only the authorized prefix is resampled/processed.  Reading the immutable
    # CSV for hashing does not construct indicators or measure final outcomes.
    m1=load_m1(source)
    prefix_end=split.validation_end if phase=="development" else split.end
    m1=m1.loc[m1.epoch+60<=prefix_end]
    bars=resample(m1,timeframe)
    train=bars.loc[bars.epoch+FRAMES[timeframe]<=split.train_end].copy()
    base={"artifact":dict(artifact),"symbol":symbol,"timeframe":timeframe,
          "strategy_id":spec.id,"phase":phase,"source_sha256":row["requested_sha256"],
          "selection_sha256":selection_sha,"verification":verification,
          "replayed_at":utc(),"cache_used":False,"safety":guard(),
          "scope":"fresh numerical rerun using frozen production formulas; not an independent formula or source-authenticity audit",
          "real_execution_costs":"NOT_VERIFIED_BY_REPLAY"}
    if len(train)<100:
        _compare(saved["status"],"INSUFFICIENT_DATA","insufficient status")
        _compare(saved["reason"],"fewer_than_100_closed_training_bars","insufficient reason")
        _verified_environment(directory,declaration)
        if digest(source)!=row["requested_sha256"]:
            raise ValueError("Source changed during insufficient-data replay")
        return {**base,"status":"PASS","matched_measurements":1,"matched_partitions":0,
                "matched_trades":0,"matched_equity_points":0,"matched_dispositions":0,
                "matched_signal_epochs":0,"full_signal_fingerprint_matched":False,
                "insufficient_training_rows_verified":len(train)}
    _compare(saved["status"],"TESTED","tested status")
    training_context=prepare_context(train,timeframe)
    fit=fit_strategy(training_context,spec,symbol)
    _compare(saved["fit"],fit.to_dict() if fit else None,"training-only fit")
    context=prepare_context(bars,timeframe)
    signals=generate_signals(context,spec,symbol,fit)
    n=len(train)+(len(bars)-len(train))//2
    prefix_context=prepare_context(bars.iloc[:n],timeframe)
    if not np.array_equal(signals[:n],generate_signals(prefix_context,spec,symbol,fit)):
        raise ReplayMismatch("Reconstructed signal failed actual-prefix invariance")
    fingerprint=_fingerprint(context,signals,timeframe)
    full_fingerprint="signal_fingerprint" in saved
    if full_fingerprint:
        _compare(saved["signal_fingerprint"],fingerprint,"full chronological signal fingerprint")
    provenance={"real_deriv_data":True,"source_file":row["requested_file"],
                "source_sha256":row["requested_sha256"],"source_manifest":row["manifest_file"],
                "api_symbol":row["api_symbol"],"fresh_oos":False,
                "actual_prefix_invariance_checked":True}
    counts={"matched_partitions":0,"matched_trades":0,"matched_equity_points":0,
            "matched_dispositions":0,"matched_signal_epochs":0}
    phases=("train","validation") if phase=="development" else ("final",)
    availability=bars.epoch.to_numpy(dtype=np.int64)+FRAMES[timeframe]
    replayed_parts={}
    for part in phases:
        start,end=split.bounds(part)
        selected=availability[(signals!=0)&(availability>=start)&(availability<end)].tolist()
        _compare([v["signal_epoch"] for v in saved[part]["dispositions"]],selected,f"{part} all effective signal epochs")
        measured=run_backtest(context,signals,spec,symbol,timeframe,start,end,partition=part,
                             costs=CostModel.from_dict(declaration["costs"]),data_quality_ok=_quality_ok(row),
                             provenance=provenance,bootstrap_repetitions=declaration["final_bootstrap_repetitions"])
        measured["reproducibility"].update(signal_causality_verified=True,raw_source_hash_verified=True)
        measured["gates"]=evaluate_final(measured)
        measured["qualified"]=all(measured["gates"].values())
        measured["rejection_reasons"]=[k for k,v in measured["gates"].items() if not v]
        _compare(saved[part],measured,f"{part} complete numerical replay")
        replayed_parts[part]=measured
        counts["matched_partitions"]+=1;counts["matched_trades"]+=len(measured["trades"])
        counts["matched_equity_points"]+=len(measured["equity"])
        counts["matched_dispositions"]+=len(measured["dispositions"])
        counts["matched_signal_epochs"]+=len(selected)
    if phase=="development":
        _compare(saved["validation_score"],risk_adjusted_score(replayed_parts["validation"]),"validation-only ranking score")
    # Fail if any pinned source/code/runtime changed while numerical work ran.
    _verified_environment(directory,declaration)
    if digest(source)!=row["requested_sha256"]:
        raise ValueError("Source changed during numerical replay")
    return {**base,"status":"PASS","matched_measurements":1,**counts,
            "full_signal_fingerprint_matched":full_fingerprint,"reconstructed_signal_fingerprint":fingerprint,
            "signal_check_scope":"full vector plus all effective disposition signals" if full_fingerprint else "legacy: all effective disposition signals; no saved full-vector fingerprint"}


def replay_campaign(directory, state, declaration, limit=None) -> dict:
    """Replay unique recorded measurement artifacts; a limit is explicit scope.

    Durable completed/current cycle rows inventory development measurements.
    Completed final-family original_measurements inventory final measurements.
    No cache entry substitutes for a numerical rerun, and no final-only artifact
    is accessed by a campaign whose actual state still seals the holdout.
    """
    directory=Path(directory).resolve()
    actual,_=_verified_environment(directory,declaration)
    validate_state(state)
    if state["declaration_sha256"]!=actual["declaration_sha256"]:
        raise ValueError("Replay campaign state belongs to another declaration")
    if limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit<0):
        raise ValueError("Replay limit must be a nonnegative integer or None")
    numbers=[v["number"] for v in state["cycles"]]
    if state.get("current_cycle") and state["current_cycle"] not in numbers:
        numbers.append(state["current_cycle"])
    artifacts=[];seen=set();duplicates=0
    def add(artifact):
        nonlocal duplicates
        ident=(artifact["path"],artifact["sha256"])
        if ident in seen:
            duplicates+=1;return
        seen.add(ident);artifacts.append(artifact)
    for number in numbers:
        cycle=read_json(directory/"cycles"/f"{number:02d}.json")
        if cycle["number"]!=number:
            raise ValueError("Replay durable cycle identity mismatch")
        for row in cycle.get("results",[]):
            add(row["artifact"])
    if state.get("final_artifact"):
        if actual.get("final_accessed") is not True:
            raise ValueError("Final family exists while actual holdout is sealed")
        family=_read_artifact(directory,state["final_artifact"])
        for artifact in family.get("original_measurements",{}).values():
            add(artifact)
    selected=artifacts if limit is None else artifacts[:limit]
    reports=[];errors=[]
    for artifact in selected:
        try:
            reports.append(replay_measurement(directory,artifact,declaration,
                                             final_access_allowed=actual.get("final_accessed") is True))
        except (ValueError,KeyError,OSError) as error:
            errors.append({"artifact":artifact,"error":f"{type(error).__name__}: {error}"})
    keys=("matched_measurements","matched_partitions","matched_trades","matched_equity_points",
          "matched_dispositions","matched_signal_epochs")
    return {"status":"FAIL" if errors else "PASS" if reports else "NOT_RUN",
            "completed_at":utc(),"available_unique_measurements":len(artifacts),"requested_limit":limit,
            "attempted_measurements":len(selected),"duplicate_artifacts_skipped":duplicates,
            "all_available_replayed":len(reports)==len(artifacts) and not errors and bool(artifacts),
            **{key:sum(r[key] for r in reports) for key in keys},
            "full_signal_fingerprints_matched":sum(r["full_signal_fingerprint_matched"] for r in reports),
            "reports":reports,"errors":errors,"safety":guard(),
            "cache_used":False,"real_execution_costs":"NOT_VERIFIED_BY_REPLAY"}
