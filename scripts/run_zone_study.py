#!/usr/bin/env python3
"""Freeze and execute the fixed pre-issued-zone study on audited public ticks.

Freeze reads only source-audit metadata and code bytes. Execution verifies each
daily CSV before decoding those same bytes, retains missing seconds, and reports
all fixed hypotheses. Neither this runner nor a favorable result permits live
execution or promotes adaptive known history to fresh out-of-sample evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.research.learned_metrics import profit_factor_inference
from app.research.payoff_metrics import paired_inference, summarize
from app.research.spike_hunter import SAFETY_FLAGS, assert_offline
from app.research.zone_dataset import build_zone_dataset
from app.research.zone_replay import ZoneReplayConfig, replay_zones, zone_trades
from scripts.run_spike_timed_study import thirds, weekly_inference


SYMBOLS = ("BOOM600", "CRASH600")
VARIANTS = ("CRT_RETEST", "FIB_RETRACE", "TREND_RETEST", "CONTEXT_GEOMETRIC")
CONTROL = "CONTEXT_GEOMETRIC"
DATA_START = pd.Timestamp("2025-10-09T00:00:00Z")
DATA_END = pd.Timestamp("2026-10-05T00:00:00Z")
OLD_START = pd.Timestamp("2025-10-09T11:08:00Z")
OLD_END = pd.Timestamp("2026-04-07T11:08:00Z")
LATER_END = pd.Timestamp("2026-10-04T11:08:00Z")
BOOTSTRAP_REPEATS = 9999
BOOTSTRAP_SEED = 20261008
SOURCE_AUDIT = "docs/jump_representation_20261007/independent_source_audit.json"
SOURCE_AUDIT_SHA256 = "34471138144552407390938f1ef58d7513d7fe556074265b674cabfb9245116a"
SOURCE_DIRECTORY = "data/jump_representation_ticks_20261007"
DEFAULT_DECLARATION = "docs/zone_study_20261008/declaration.json"


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def relative_path(value: str | Path) -> tuple[Path, str]:
    path = (ROOT / value).resolve()
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Study artifacts and pins must remain within the repository") from error
    return path, relative


def json_value(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported study JSON value: {type(value).__name__}")


def save_exclusive(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return {"path": relative_path(path)[1], "sha256": digest(path)}


def partitions() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    duration = OLD_END - OLD_START
    cut = {n: OLD_START + duration * n / 10 for n in (4, 5, 6, 7)}
    return {
        "development": (OLD_START, cut[7]),
        "wf1": (cut[4], cut[5]), "wf2": (cut[5], cut[6]),
        "wf3": (cut[6], cut[7]), "final_test": (cut[7], OLD_END),
        "later180": (OLD_END, LATER_END),
    }


def protocol() -> dict:
    return json_value({
        "version": 1, "symbols": SYMBOLS, "variants": VARIANTS,
        "baseline_variant": CONTROL,
        "baseline_interpretation": "same-context fixed geometric-region control; not market-entry clock",
        "data_start": DATA_START, "data_end_exclusive": DATA_END,
        "partitions": partitions(), "walk_forward_combined": [partitions()["wf1"][0], partitions()["wf3"][1]],
        "config": asdict(ZoneReplayConfig()), "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED, "double_cost_atr": 2 * ZoneReplayConfig().cost_atr,
        "fixed_rules_no_parameter_fitting": True, "all_variants_reported_no_selection": True,
        "known_history_adaptive": True, "fresh_out_of_sample": False,
        "holdout_interpretation": "chronological withheld partition inside previously exposed adaptive history",
        "target": {"profit_factor": 1.5, "completed_per_symbol_variant": 1000, "active_days": 60},
        "qualification_requires_independent_economic_audit": True,
        "broker_costs": "NOT TESTED", "cash_profit": "NOT TESTED", "prospective_paper": "NOT TESTED",
    })


def code_hashes(extra_paths=()) -> dict[str, str]:
    names = {str(path.relative_to(ROOT)) for path in (ROOT / "backend/app/research").glob("*.py")}
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / "backend/tests").glob("test_zone*.py"))
    names.update({"backend/app/config.py", "scripts/run_zone_study.py", "scripts/run_spike_timed_study.py",
                  "scripts/run_spike_payoff_study.py"})
    names.update(relative_path(path)[1] for path in extra_paths)
    return {name: digest(relative_path(name)[0]) for name in sorted(names)}


def audited_sources() -> dict:
    path = relative_path(SOURCE_AUDIT)[0]
    if digest(path) != SOURCE_AUDIT_SHA256:
        raise ValueError("Independent source audit bytes changed")
    audit = json.loads(path.read_text())
    dates = pd.date_range(DATA_START, DATA_END, freq="D", inclusive="left")
    expected = {(symbol, stamp.date().isoformat()) for symbol in SYMBOLS for stamp in dates}
    if (audit.get("passed") is not True or audit.get("completeness_passed") is not True
            or audit.get("integrity_of_available_sources") != "PASS"
            or audit.get("errors") != [] or audit.get("expected_sources") != len(expected)
            or audit.get("available_manifests") != len(expected) or audit.get("missing_manifests") != 0
            or audit.get("unexpected_manifest_files") != [] or audit.get("missing_source_identities") != []
            or any(audit.get("safety", {}).get(flag) is not False for flag in SAFETY_FLAGS)):
        raise ValueError("Independent source audit did not pass the declared full population")
    entries = audit.get("sources", [])
    if len(entries) != len(expected) or {(r.get("symbol"), r.get("date")) for r in entries} != expected:
        raise ValueError("Independent audit source identities differ from the complete planned calendar")
    pins = audit.get("input_sha256", {})
    sources = {symbol: [] for symbol in SYMBOLS}
    for row in sorted(entries, key=lambda r: (r["symbol"], r["date"])):
        name = f"{SOURCE_DIRECTORY}/{row['symbol'].lower()}_{row['date']}_ticks_clean.csv"
        fingerprint = pins.get(name)
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("A required clean source lacks a valid audited SHA256")
        count, missing = row.get("rows"), row.get("missing_seconds")
        if (type(count) is not int or type(missing) is not int or not 0 <= count <= 86400
                or count + missing != 86400):
            raise ValueError("Audited source counts do not preserve the full daily population")
        sources[row["symbol"]].append({"date": row["date"], "path": name,
                                        "sha256": fingerprint, "rows": count, "missing_seconds": missing})
    return {"audit_path": SOURCE_AUDIT, "audit_sha256": SOURCE_AUDIT_SHA256, "symbols": sources}


def freeze(declaration_path: str | Path, extra_paths=()) -> dict:
    safety = assert_offline()
    path, _ = relative_path(declaration_path)
    if any((path.parent / name).exists() for name in
           (path.name, "declaration.sha256", "execution_started.json", "results.json", "execution_failure.json")):
        raise ValueError("Refusing to replace a frozen or started study")
    declaration = {
        "stage": "frozen_zone_study_declaration", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "safety": safety, "protocol": protocol(), "sources": audited_sources(),
        "extra_pin_paths": sorted({relative_path(name)[1] for name in extra_paths}),
        "code_sha256": code_hashes(extra_paths),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "historical_prices_decoded_for_this_study": False, "QUALIFIED": False,
    }
    artifact = save_exclusive(path, declaration)
    with (path.parent / "declaration.sha256").open("x") as target:
        target.write(artifact["sha256"] + "\n")
    return artifact


def frozen(declaration_path: str | Path, expected_sha256: str) -> tuple[dict, Path]:
    assert_offline()
    path, _ = relative_path(declaration_path)
    if len(expected_sha256) != 64 or digest(path) != expected_sha256:
        raise ValueError("Exact frozen declaration SHA256 is required")
    if (path.parent / "declaration.sha256").read_text().strip() != expected_sha256:
        raise ValueError("Frozen declaration sidecar differs")
    declaration = json.loads(path.read_text())
    if (declaration.get("stage") != "frozen_zone_study_declaration"
            or declaration.get("protocol") != protocol()
            or declaration.get("sources") != audited_sources()
            or declaration.get("code_sha256") != code_hashes(declaration.get("extra_pin_paths", []))
            or declaration.get("QUALIFIED") is not False
            or any(declaration.get("safety", {}).get(flag) is not False for flag in SAFETY_FLAGS)):
        raise ValueError("Frozen protocol, code, safety or source metadata changed")
    expected_runtime = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    if declaration.get("runtime") != expected_runtime:
        raise ValueError("Frozen numerical runtime changed")
    return declaration, path


def load_symbol(sources: list[dict]) -> tuple[object, pd.DataFrame]:
    """Decode pinned bytes once into day chunks and two compact full-size arrays."""
    expected = sum(row["rows"] for row in sources)
    timestamps = np.empty(expected, dtype=np.int64)
    quotes = np.empty(expected, dtype=np.float64)
    offset = 0

    def chunks():
        nonlocal offset
        for source in sources:
            payload = relative_path(source["path"])[0].read_bytes()
            if hashlib.sha256(payload).hexdigest() != source["sha256"]:
                raise ValueError(f"Audited source bytes changed: {source['path']}")
            table = pd.read_csv(io.BytesIO(payload), dtype={"epoch": np.int64, "quote": np.float64})
            if list(table.columns) != ["epoch", "quote"] or len(table) != source["rows"]:
                raise ValueError("Clean daily source schema/count differs from audit")
            epoch = table.epoch.to_numpy()
            day_start = pd.Timestamp(source["date"], tz="UTC").value // 1_000_000_000
            if np.any(epoch < day_start) or np.any(epoch >= day_start + 86400):
                raise ValueError("Source timestamps leave the declared UTC day")
            end = offset + len(table)
            timestamps[offset:end] = epoch * 1_000_000_000
            quotes[offset:end] = table.quote.to_numpy()
            frame = pd.DataFrame({"quote": quotes[offset:end]},
                                 index=pd.DatetimeIndex(timestamps[offset:end], tz="UTC"), copy=False)
            offset = end
            yield frame
    dataset = build_zone_dataset(chunks(), start=DATA_START, end=DATA_END)
    if offset != expected or dataset.summary["observed_seconds"] != expected:
        raise ValueError("Decoded full source population differs from the audit")
    ticks = pd.DataFrame({"quote": quotes}, index=pd.DatetimeIndex(timestamps, tz="UTC"), copy=False)
    return dataset, ticks


def save_csv(path: Path, frame: pd.DataFrame) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        frame.to_csv(target, index=False)
    return {"path": relative_path(path)[1], "sha256": digest(path)}


def metric_row(symbol, variant, partition, bounds, ledger, baseline, audit, artifacts):
    start, end = bounds
    cfg = ZoneReplayConfig()
    doubled = ledger.copy()
    doubled["net_R"] = pd.to_numeric(doubled.net_R, errors="raise") - cfg.cost_atr / cfg.stop_atr
    return {
        "symbol": symbol, "variant": variant, "partition": partition,
        "start": start, "end_exclusive": end,
        "metrics": summarize(ledger, start, end, stop_atr=cfg.stop_atr),
        "day_inference": paired_inference(ledger, baseline, start, end,
                                           repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED),
        "weekly_inference": weekly_inference(ledger, baseline, start, end,
                                               BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED),
        "profit_factor_inference": profit_factor_inference(ledger, start, end,
            repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED, block_days=7),
        "thirds": thirds(ledger, start, end, cfg.stop_atr),
        "double_cost_metrics": summarize(doubled, start, end, stop_atr=cfg.stop_atr),
        "double_cost_atr": 2 * cfg.cost_atr, "replay_audit": audit,
        "baseline_variant": CONTROL, "artifacts": artifacts,
        "QUALIFIED": False, "independent_economic_audit": "PENDING",
    }


def attach_qualifications(rows):
    from app.research.zone_qualification import historical_zone_gate, holm_adjust
    indexed = {(r["symbol"], r["variant"], r["partition"]): r for r in rows}
    heldout = [r for r in rows if r["variant"] != CONTROL and r["partition"] in ("final_test", "later180")]
    adjusted = holm_adjust([max(r["day_inference"]["p"], r["weekly_inference"]["weekly_p"]) for r in heldout])
    def development_metrics(symbol, variant, part):
        candidate = indexed[(symbol, variant, part)]
        control = indexed[(symbol, CONTROL, part)]
        return {**candidate["metrics"], "unknown_regions": candidate["replay_audit"]["unknown"],
                "control_unknown_regions": control["replay_audit"]["unknown"]}
    for row, corrected in zip(heldout, adjusted, strict=True):
        symbol, variant, part = row["symbol"], row["variant"], row["partition"]
        control = indexed[(symbol, CONTROL, part)]
        row["qualification"] = historical_zone_gate(
            metrics={**row["metrics"], **row["day_inference"]},
            pf=row["profit_factor_inference"], weekly=row["weekly_inference"],
            folds=[development_metrics(symbol, variant, fold) for fold in ("wf1", "wf2", "wf3")],
            validation=development_metrics(symbol, variant, "walk_forward_combined"),
            thirds=[third["metrics"] for third in row["thirds"]],
            doubled_cost=row["double_cost_metrics"], unknown_regions=row["replay_audit"]["unknown"],
            baseline_unknown_regions=control["replay_audit"]["unknown"], holm_p=corrected,
        )
        row["holm_p"] = corrected


def evaluate_symbol(symbol, dataset, ticks, output):
    from app.research.zone_rules import zone_candidates
    candidates = zone_candidates(dataset, "boom" if symbol.startswith("BOOM") else "crash")
    if not set(candidates.variant.unique()).issubset(VARIANTS):
        raise ValueError("Candidate variants differ from the frozen hypothesis family")
    if candidates.duplicated(["variant", "signal_time"]).any():
        raise ValueError("Candidate issuance is duplicated")
    if candidates.eligible.isna().any() or not pd.api.types.is_bool_dtype(candidates.eligible.dtype):
        raise ValueError("Candidate eligibility must be a known boolean")
    candidate_artifact = save_csv(output / f"{symbol}_candidates.csv", candidates)
    ledgers, audits, saved, rows = {}, {}, {}, []
    for part, bounds in partitions().items():
        start, end = bounds
        for variant in VARIANTS:
            selected = candidates.loc[candidates.eligible & candidates.variant.eq(variant)
                                      & candidates.signal_time.ge(start) & candidates.signal_time.lt(end)]
            zones = selected.to_dict("records")
            prefix = output / f"{symbol}_{variant}_{part}"
            zone_artifact = save_exclusive(prefix.with_name(prefix.name + "_zones.json"), zones)
            events, audit = replay_zones(ticks, zones, ZoneReplayConfig(), start, end)
            ledger = zone_trades(events)
            ledgers[(part, variant)], audits[(part, variant)] = ledger, audit
            saved[(part, variant)] = {
                "zones": zone_artifact,
                "events": save_csv(prefix.with_name(prefix.name + "_events.csv"), events),
                "ledger": save_csv(prefix.with_name(prefix.name + "_ledger.csv"), ledger),
            }
        for variant in VARIANTS:
            row = metric_row(symbol, variant, part, bounds, ledgers[(part, variant)],
                             ledgers[(part, CONTROL)], audits[(part, variant)], saved[(part, variant)])
            rows.append(row)
            print(f"{symbol}/{variant}/{part}: completed={row['metrics']['completed']} "
                  f"PF={row['metrics']['profit_factor']}", flush=True)
    bounds = (partitions()["wf1"][0], partitions()["wf3"][1])
    combined = {variant: pd.concat([ledgers[(fold, variant)] for fold in ("wf1", "wf2", "wf3")],
                                   ignore_index=True) for variant in VARIANTS}
    for variant in VARIANTS:
        audit = {"unknown": sum(audits[(fold, variant)]["unknown"] for fold in ("wf1", "wf2", "wf3")),
                 "component_folds": {fold: audits[(fold, variant)] for fold in ("wf1", "wf2", "wf3")},
                 "independent_replays_at_frozen_partition_boundaries": True}
        artifacts = {"ledger": save_csv(output / f"{symbol}_{variant}_walk_forward_combined_ledger.csv",
                                         combined[variant]),
                     "folds": {fold: saved[(fold, variant)] for fold in ("wf1", "wf2", "wf3")}}
        rows.append(metric_row(symbol, variant, "walk_forward_combined", bounds,
                               combined[variant], combined[CONTROL], audit, artifacts))
    return rows, {"dataset_summary": dataset.summary, "candidate_artifact": candidate_artifact,
                  "candidate_audit": {variant: {
                      "planned_rows": int(candidates.variant.eq(variant).sum()),
                      "eligible_rows": int((candidates.variant.eq(variant) & candidates.eligible).sum()),
                      "reasons": candidates.loc[candidates.variant.eq(variant), "reason"].value_counts().to_dict(),
                  } for variant in VARIANTS}}


def execute(declaration_path: str | Path, expected_sha256: str, output_path: str | Path) -> dict:
    declaration, path = frozen(declaration_path, expected_sha256)
    output, _ = relative_path(output_path)
    if output != path.parent:
        raise ValueError("Execution output must be the frozen declaration directory")
    save_exclusive(output / "execution_started.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": expected_sha256, "safety": assert_offline(),
    })
    try:
        results = {"stage": "fixed_zone_historical_study", "declaration_sha256": expected_sha256,
                   "safety": assert_offline(), "known_history_adaptive": True,
                   "QUALIFIED": False, "economic_audit": "PENDING", "symbols": {}, "rows": []}
        for symbol in SYMBOLS:
            dataset, ticks = load_symbol(declaration["sources"]["symbols"][symbol])
            print(f"{symbol}: {dataset.summary['observed_seconds']} audited native observations loaded", flush=True)
            rows, info = evaluate_symbol(symbol, dataset, ticks, output)
            results["symbols"][symbol] = info
            results["rows"].extend(rows)
            del ticks, dataset
        attach_qualifications(results["rows"])
        # Recheck immutable metadata/code after the complete calculation. Each
        # decoded CSV was independently byte-checked immediately before use.
        frozen(path, expected_sha256)
        results["completed_utc"] = datetime.now(timezone.utc).isoformat()
        artifact = save_exclusive(output / "results.json", results)
        with (output / "results.sha256").open("x") as target:
            target.write(artifact["sha256"] + "\n")
        return artifact
    except Exception as error:
        save_exclusive(output / "execution_failure.json", {
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "declaration_sha256": expected_sha256, "error_type": type(error).__name__,
            "error": str(error), "safety": {flag: False for flag in SAFETY_FLAGS}, "QUALIFIED": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", metavar="DECLARATION")
    mode.add_argument("--execute", metavar="DECLARATION_SHA256")
    parser.add_argument("--declaration", default=DEFAULT_DECLARATION)
    parser.add_argument("--output", default=str(Path(DEFAULT_DECLARATION).parent))
    parser.add_argument("--pin", action="append", default=[], help="Additional protocol/document byte pin at freeze")
    args = parser.parse_args()
    if args.execute and args.pin:
        parser.error("Additional pins can only be specified before freezing")
    artifact = freeze(args.freeze, args.pin) if args.freeze else execute(args.declaration, args.execute, args.output)
    print(json.dumps(artifact), flush=True)


if __name__ == "__main__":
    main()
