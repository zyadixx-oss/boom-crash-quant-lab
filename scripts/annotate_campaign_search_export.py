"""Label the completed campaign's mixed-scope search export without changing it.

All original cells and their CSV bytes are retained. The added labels identify
validation metrics separately from final/development selection dispositions.
This verifies recorded evidence; it does not recompute or change any outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORE_FILES = ("campaign.py", "data.py", "engine.py", "strategies.py", "state.py",
              "reporting.py", "source_audit.py", "replay.py")
LIVE_FLAGS = ("LIVE_TRADING", "READY_FOR_LIVE", "LIVE_ALLOWED", "OPENED_TRADES")
SYMBOLS = {f"{side}{number}" for side in ("BOOM", "CRASH") for number in (300, 500, 600, 900, 1000)}
FRAMES = {"M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4"}
COLUMNS = ["cycle", "symbol", "timeframe", "family", "strategy_id", "attempted_strategy_id",
           "status", "cache_hit", "selection_disposition", "reasons", "settings_json",
           "validation_score", "completed_trades", "profit_factor", "net_profit",
           "max_drawdown", "artifact"]
LABEL_COLUMNS = ["metrics_partition", "disposition_partition"]
FINAL_GATES = {"final_partition", "minimum_100_completed", "maximum_drawdown_20pct", "profit_factor_1_10",
               "positive_net_profit", "outperforms_buy_and_hold", "outperforms_native_reference", "data_quality",
               "future_leak_checked", "reproducible_real_source", "actual_contract_costs_verified",
               "original_tick_execution_verified", "statistical_uncertainty", "multiplicity_adjusted",
               "all_live_gates_false", "development_walk_forward_stability"}


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def check_safety(value):
    if not isinstance(value, dict) or not set(LIVE_FLAGS).issubset(value):
        raise ValueError("Required false live flags missing")
    if any(flag is not False for flag in value.values()):
        raise ValueError("Every live flag must be boolean false")


def check_complete_state(state):
    check_safety(state["safety"])
    if (state.get("status") != "COMPLETED" or state.get("completed_cycles") != 32
            or state.get("target_cycles") != 32 or state.get("interval_seconds") != 900
            or state.get("final_accessed") is not True or state.get("current_cycle") is not None):
        raise ValueError("Completed safe 32-cycle campaign required")
    cycles = state.get("cycles", [])
    if len(cycles) != 32:
        raise ValueError("All 32 actual completion records required")
    def stamp(text):
        value = datetime.fromisoformat(text)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timezone-aware actual timestamps required")
        return value.timestamp()
    started = state["started_epoch"]
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ValueError("Finite campaign start required")
    if abs(stamp(state["started_at"]) - started) > .00001:
        raise ValueError("Campaign start fields disagree")
    now = datetime.now(timezone.utc).timestamp()
    completion = stamp(state["completed_at"])
    if stamp(state["deadline_at"]) != started + 28800 or not started + 28800 <= completion <= now:
        raise ValueError("Full elapsed eight-hour window required")
    previous_start, previous_end = None, started
    for number, cycle in enumerate(cycles, 1):
        if cycle.get("number") != number or cycle.get("status") != "COMPLETED":
            raise ValueError("Contiguous completed cycles required")
        start, end = stamp(cycle["started_at"]), stamp(cycle["completed_at"])
        if (start < previous_end or end < start or end > completion
                or (previous_start is not None and start - previous_start < 900)):
            raise ValueError("Actual cycle timestamps violate cadence")
        previous_start, previous_end = start, end


def canonical_spec(values):
    from research_campaign.strategies import StrategySpec
    canonical = StrategySpec.from_dict(values).to_dict()
    if canonical != values:
        raise ValueError("Noncanonical strategy settings or labels")
    return canonical


def validate_record(record, attempted, catalogue):
    actual = canonical_spec(record["spec"])
    if (record["family"] != actual["family"] or record["strategy_id"] != actual["id"]
            or record["attempted_strategy_id"] != attempted["id"]
            or actual != catalogue[actual["round_index"]][actual["family"]]
            or actual["scientific_id"] != attempted["scientific_id"]
            or type(record["cache_hit"]) is not bool
            or (not record["cache_hit"] and actual != attempted)):
        raise ValueError("Journal settings/cache identity differs from frozen catalogue")
    if record["status"] not in {"TESTED", "INSUFFICIENT_DATA"}:
        raise ValueError("Unexpected journal measurement status")
    metrics = record.get("validation_metrics")
    if record["status"] == "INSUFFICIENT_DATA":
        if metrics is not None or record.get("validation_score") is not None:
            raise ValueError("Insufficient measurement cannot contain validation metrics")
        return
    required = {"completed_trades", "profit_factor", "net_profit", "max_drawdown"}
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise ValueError("Tested validation metric fields missing")
    if type(metrics["completed_trades"]) is not int or metrics["completed_trades"] < 0:
        raise ValueError("Nonnegative completed trade count required")
    values = [metrics["net_profit"], metrics["max_drawdown"], record["validation_score"]]
    pf = metrics["profit_factor"]
    if pf is not None:
        values.append(pf)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Finite real validation metrics required")
    if not 0 <= metrics["max_drawdown"] <= 1 or (pf is not None and pf < 0):
        raise ValueError("Invalid validation PF/drawdown")


def contained(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Evidence path escapes campaign directory")
    return path


def iter_family(path, *, chunk_size=1024 * 1024):
    """Parse one original result at a time, then yield the small outer metadata."""
    decoder = json.JSONDecoder()
    marker = re.compile(r'\n  "results":\s*\[')
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        buffer = ""
        while not (match := marker.search(buffer)):
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("Missing top-level final results array")
            buffer += chunk
        prefix, buffer = buffer[:match.end() - 1], buffer[match.end():]
        need_comma = False
        count = 0
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                buffer = stream.read(chunk_size)
                if not buffer:
                    raise ValueError("Truncated final family")
                continue
            if buffer.startswith("]"):
                meta = json.loads(prefix + "[]" + buffer[1:] + stream.read())
                if meta["family_size"] != count:
                    raise ValueError("Final family count mismatch")
                meta.pop("results")
                yield "metadata", meta
                return
            if need_comma:
                if not buffer.startswith(","):
                    raise ValueError("Invalid final result separator")
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    result, stop = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk
            if not isinstance(result, dict):
                raise ValueError("Invalid final result object")
            buffer = buffer[stop:]
            count += 1
            need_comma = True
            yield "result", result
            del result


def final_dispositions(path, selection_sha, policies, verify_original):
    rows, meta = {}, None
    for kind, result in iter_family(path):
        if kind == "metadata":
            meta = result
            continue
        group = result["selection_group"]
        check_safety(result["safety"])
        gates = result["gates"]
        if (group in rows or group not in policies or result["partition"] != "final"
                or result["selection_sha256"] != selection_sha
                or set(gates) != FINAL_GATES
                or any(type(value) is not bool for value in gates.values())
                or type(result["qualified"]) is not bool
                or result["qualified"] != all(gates.values())
                or len(result["rejection_reasons"]) != len(set(result["rejection_reasons"]))
                or set(result["rejection_reasons"]) != {key for key, value in gates.items() if not value}):
            raise ValueError("Final disposition/selection lineage mismatch")
        policy = policies[group]
        if (canonical_spec(result["strategy"]) != policy["spec"]
                or result["strategy_id"] != policy["strategy_id"]
                or result["symbol"] != policy["symbol"] or result["timeframe"] != policy["timeframe"]
                or result["start"] != policy["start"] or result["end"] != policy["end"]
                or result["costs"] != policy["costs"]):
            raise ValueError("Final result differs from selected policy or frozen calendar/costs")
        identity = {key: result[key] for key in ("strategy", "symbol", "timeframe", "start", "end", "partition", "costs")}
        if result["id"] != sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            raise ValueError("Final result identity digest mismatch")
        rows[group] = {"qualified": result["qualified"], "rejection_reasons": result["rejection_reasons"]}
    if (meta is None or meta["selection_sha256"] != selection_sha
            or meta["selected_before_access"] is not True or set(rows) != set(policies)
            or set(meta["original_measurements"]) != set(policies)):
        raise ValueError("Final family does not exactly cover frozen selection")
    check_safety(meta["safety"])
    for group, artifact in meta["original_measurements"].items():
        if artifact["path"] != policies[group]["original_path"]:
            raise ValueError("Original final artifact path differs from scientific identity")
        verify_original(artifact)
    return rows


def expected_row(number, record, winners, finals):
    row = {column: record.get(column) for column in COLUMNS}
    group = f'{record["symbol"]}/{record["timeframe"]}/{record["family"]}'
    selected = winners.get(group, {}).get("strategy_id") == record["strategy_id"]
    final = finals.get(group) if selected else None
    if record["status"] != "TESTED":
        disposition, reasons = "INSUFFICIENT_DATA", ["INSUFFICIENT_CLOSED_TRAINING_BARS"]
        phase = "development_selection"
    elif final:
        disposition = "QUALIFIED" if final["qualified"] else "REJECTED_FINAL"
        reasons, phase = final["rejection_reasons"], "final_selection"
    elif selected:
        raise ValueError("Selected tested policy lacks its committed final disposition")
    else:
        disposition, reasons = "NOT_SELECTED", ["NOT_BEST_VALIDATION_SCORE_WITHIN_FAMILY"]
        phase = "development_selection"
    row.update(cycle=number, artifact=record["artifact"]["path"],
               selection_disposition=disposition, reasons=";".join(reasons),
               settings_json=json.dumps(record["spec"], sort_keys=True, ensure_ascii=False))
    for column in ("completed_trades", "profit_factor", "net_profit", "max_drawdown"):
        row[column] = (record.get("validation_metrics") or {}).get(column)
    return ["" if row[column] is None else str(row[column]) for column in COLUMNS], phase


class RawCSVLines:
    """Give csv.reader text lines while retaining each original record's bytes."""
    def __init__(self, stream):
        self.stream, self.pending = stream, []

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.stream)
        self.pending.append(line)
        return line.decode("utf-8")

    def take(self):
        record = b"".join(self.pending)
        self.pending.clear()
        return record


def append_labels(raw, labels):
    if raw.endswith(b"\r\n"):
        body, ending = raw[:-2], b"\r\n"
    elif raw.endswith(b"\n"):
        body, ending = raw[:-1], b"\n"
    else:
        body, ending = raw, b""
    return body + b"," + ",".join(labels).encode("ascii") + ending


def validate_and_annotate(source, destination, expected, *, required_count):
    counts, rows = Counter(), 0
    with source.open("rb") as stream:
        tracked = RawCSVLines(stream)
        reader = csv.reader(tracked, strict=True)
        if next(reader, None) != COLUMNS:
            raise ValueError("Unexpected original search CSV columns")
        destination.write(append_labels(tracked.take(), LABEL_COLUMNS))
        for rows, (values, phase) in enumerate(expected, 1):
            actual = next(reader, None)
            raw = tracked.take()
            if actual != values:
                field = next((COLUMNS[i] for i, (a, b) in enumerate(zip(actual or [], values)) if a != b), "row_count_or_width")
                raise ValueError(f"Search CSV differs from journal evidence at row {rows}, field {field}")
            destination.write(append_labels(raw, ["validation", phase]))
            counts[phase] += 1
            counts["cache_hits" if values[COLUMNS.index("cache_hit")] == "True" else "new_measurements"] += 1
            counts[values[COLUMNS.index("selection_disposition")]] += 1
        if next(reader, None) is not None or rows != required_count:
            raise ValueError("Search CSV row count differs from complete journal inventory")
    return {"rows": rows, **dict(counts)}


def annotate(source, output, receipt, campaign_dir, *, root=ROOT):
    source, output, receipt, base = map(lambda p: Path(p).resolve(), (source, output, receipt, campaign_dir))
    if output == receipt or output == source or receipt == source:
        raise ValueError("Distinct original, output and receipt paths required")
    if any(path.is_relative_to(protected) for path in (output, receipt)
           for protected in (base, (root / "research_campaign").resolve())):
        raise ValueError("Annotated outputs must stay outside frozen campaign/code directories")
    if output.exists() or receipt.exists():
        raise FileExistsError("Output and receipt are exclusive; existing evidence will not be replaced")
    for name in (*LIVE_FLAGS, *(name.lower() for name in LIVE_FLAGS)):
        if os.environ.get(name, "false").strip().lower() != "false":
            raise ValueError(f"Research requires {name}=false")
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"Frozen evidence hash mismatch: {path.name}")
        pins[str(path)] = value
        return value

    pin(source)
    script_sha = pin(Path(__file__))
    state_path = base / "state.json"
    pin(state_path)
    state = read_json(state_path)
    check_complete_state(state)
    declaration_path = base / "declaration.json"
    declaration_sha = pin(declaration_path, state["declaration_sha256"])
    declaration = read_json(declaration_path)
    check_safety(declaration["safety"])
    if (set(declaration["datasets"]) != SYMBOLS or set(declaration["timeframes"]) != FRAMES
            or len(declaration["candidate_rounds"]) != 32 or set(declaration["source_pins"]) != set(CORE_FILES)):
        raise ValueError("Declared 10-symbol/10-frame/32-round scope required")
    families = {spec["family"] for spec in declaration["candidate_rounds"][0]}
    if len(families) != 13:
        raise ValueError("Declared thirteen-family scope required")
    for number, round_specs in enumerate(declaration["candidate_rounds"]):
        if (len(round_specs) != 13 or {r["family"] for r in round_specs} != families
                or any(r["round_index"] != number or r["trailing_stop"] is not False for r in round_specs)):
            raise ValueError("Frozen parameter catalogue changed")
    for name, expected in declaration["source_pins"].items():
        pin(root / "research_campaign" / name, expected)
        pin(base / "frozen_source" / name, expected)
    catalogue = [{spec["family"]: canonical_spec(spec) for spec in specs}
                 for specs in declaration["candidate_rounds"]]
    pin(Path(declaration["source_manifest"]), declaration["source_manifest_sha256"])
    for dataset in declaration["datasets"].values():
        pin(Path(dataset["requested_file"]), dataset["requested_sha256"])
    selection_path = base / "selection_frozen.json"
    selection_sha = pin(selection_path, state["selection_sha256"])
    selected = read_json(selection_path)
    winners = selected["selection"]
    expected_groups = {f"{symbol}/{frame}/{family}" for symbol in SYMBOLS for frame in FRAMES for family in families}
    if (selected["source"] != declaration_sha or selected["final_data_accessed"] is not False
            or winners != state["winners"] or set(winners) != expected_groups):
        raise ValueError("Frozen prefinal selection lineage mismatch")
    policies = {}
    for group, winner in winners.items():
        spec = canonical_spec(winner["spec"])
        if (spec != catalogue[spec["round_index"]][winner["family"]]
                or winner["strategy_id"] != spec["id"]
                or group != f'{winner["symbol"]}/{winner["timeframe"]}/{spec["family"]}'):
            raise ValueError("Selected policy differs from frozen catalogue")
        if winner["status"] == "TESTED":
            split = declaration.get("splits", {}).get(winner["symbol"], declaration["split"])
            identity = {"symbol": winner["symbol"], "timeframe": winner["timeframe"],
                        "scientific_spec": spec["scientific_id"],
                        "input_sha256": declaration["datasets"][winner["symbol"]]["requested_sha256"],
                        "split": split, "phase": "final", "source_pins": declaration["source_pins"]}
            key = sha256((json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()).hexdigest()
            policies[group] = {"spec": spec, "strategy_id": spec["id"], "symbol": winner["symbol"],
                               "timeframe": winner["timeframe"], "start": split["validation_end"], "end": split["end"],
                               "costs": declaration["costs"], "original_path": f"artifacts/{key}.json.gz"}
        elif winner["status"] != "INSUFFICIENT_DATA":
            raise ValueError("Unknown selected policy status")
    final_path = contained(base, state["final_artifact"]["path"])
    pin(final_path, state["final_artifact"]["sha256"])
    finals = final_dispositions(final_path, selection_sha, policies,
                                lambda artifact: pin(contained(base, artifact["path"]), artifact["sha256"]))
    journals = []
    for cycle in state["cycles"]:
        path = base / "cycles" / f'{cycle["number"]:02d}.json'
        pin(path)
        record = read_json(path)
        for field in ("number", "status", "started_at", "completed_at", "tested_count"):
            if record[field] != cycle[field]:
                raise ValueError("Cycle journal differs from terminal state")
        groups = [f'{r["symbol"]}/{r["timeframe"]}/{r["family"]}' for r in record["results"]]
        attempted_catalogue = catalogue[cycle["number"] - 1]
        if len(groups) != 1300 or set(groups) != expected_groups:
            raise ValueError("Cycle journal lacks all 1300 actual candidates")
        for row in record["results"]:
            validate_record(row, attempted_catalogue[row["family"]], catalogue)
        if (record["cache_hits"] != sum(row["cache_hit"] for row in record["results"])
                or record["tested_count"] != sum(row["status"] == "TESTED" and not row["cache_hit"] for row in record["results"])):
            raise ValueError("Journal new/cache counters disagree")
        journals.append(path)

    def expectations():
        for path in journals:
            cycle = read_json(path)
            for row in cycle["results"]:
                yield expected_row(cycle["number"], row, winners, finals)

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="search-labels.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary)
    receipt_temp = None
    try:
        with os.fdopen(descriptor, "wb") as destination:
            counts = validate_and_annotate(source, destination, expectations(), required_count=41600)
            destination.flush()
            os.fsync(destination.fileno())
        for path, expected in pins.items():
            if digest(Path(path)) != expected:
                raise ValueError("Input evidence changed during export validation")
        data = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "VERIFIED_JOURNAL_EXPORT_WITH_EXPLICIT_PHASE_LABELS",
                "campaign_id": state["id"], "completed_cycles": 32,
                "source": {"path": str(source), "sha256": pins[str(source)], "bytes": source.stat().st_size},
                "output": {"path": str(output), "sha256": digest(temporary), "bytes": temporary.stat().st_size},
                "original_columns": COLUMNS, "appended_columns": LABEL_COLUMNS,
                "metrics_partition": "validation", "counts": counts, "source_and_evidence_pins_before_after": pins,
                "script_sha256": script_sha, "safety": state["safety"],
                "original_source_bytes_unchanged": True, "original_record_bytes_retained_before_added_labels": True,
                "scope": "All 41600 original CSV rows exactly matched 32 journals, validation metrics, settings and frozen-selection/final dispositions; no backtests, formulas, costs or outcomes changed. This is distinct from the 1300-row OOS results export.",
                "limitations": ["Final dispositions accompany validation metrics; appended labels make this explicit.",
                                "CSV metrics are matched to recorded journals; this does not independently re-read development artifact contents or numerically recompute the development search.",
                                "Historical data was previously exposed; actual CFD costs and fills remain unverified."]}
        fd, name = tempfile.mkstemp(prefix="search-label-receipt.", suffix=".tmp", dir=receipt.parent)
        receipt_temp = Path(name)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        try:
            os.link(receipt_temp, receipt)
        except BaseException:
            # Only remove the new file we just published; the original is untouched.
            output.unlink()
            raise
        return data
    finally:
        temporary.unlink(missing_ok=True)
        if receipt_temp is not None:
            receipt_temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "frontend/public/campaign/search.csv")
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "data/single_frame_campaign_20261011")
    parser.add_argument("--output", type=Path, default=ROOT / "work/search_phase_labelled_20261012.csv")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = args.receipt or args.output.with_suffix(".receipt.json")
    result = annotate(args.source, args.output, receipt, args.campaign_dir)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "source": result["source"], "output": result["output"], "receipt": str(receipt.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
