"""Tiny synthetic CSVs only: phase confusion must fail before publication."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gzip
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "operational_search_export", Path(__file__).resolve().parents[2] / "scripts/annotate_campaign_search_export.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
SAFE = {name: False for name in helper.LIVE_FLAGS}


@pytest.fixture
def small_rows():
    selected = {"symbol": "BOOM500", "timeframe": "M5", "family": "crt", "strategy_id": "crt.r00",
                "attempted_strategy_id": "crt.r00", "status": "TESTED", "cache_hit": False,
                "spec": {"id": "crt.r00", "label": 'Arabic عربي, quote " and\nnewline', "threshold": .8},
                "artifact": {"path": "artifacts/synthetic\nquoted-path.json.gz"}, "validation_score": .2,
                "validation_metrics": {"completed_trades": 17, "profit_factor": 1.2,
                                       "net_profit": 12.5, "max_drawdown": .03}}
    loser = deepcopy(selected)
    loser.update(strategy_id="crt.r01", attempted_strategy_id="crt.r01", cache_hit=True)
    loser["spec"]["id"] = "crt.r01"
    insufficient = deepcopy(selected)
    insufficient.update(family="pre_spike", strategy_id="pre_spike.r00", status="INSUFFICIENT_DATA",
                        validation_metrics=None, validation_score=None)
    insufficient["spec"]["id"] = "pre_spike.r00"
    group = "BOOM500/M5/crt"
    winners = {group: {"strategy_id": "crt.r00"}}
    finals = {group: {"qualified": False, "rejection_reasons": ["actual_contract_costs_verified", "minimum_100_completed"]}}
    records = [selected, loser, insufficient]
    expected = [helper.expected_row(i, row, winners, finals) for i, row in enumerate(records, 1)]
    return records, winners, finals, expected


def csv_bytes(expected, *, line_ending="\r\n"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator=line_ending)
    writer.writerow(helper.COLUMNS)
    writer.writerows(row for row, _ in expected)
    return stream.getvalue().encode()


def verify(tmp_path, raw, expected, *, required_count=None):
    source = tmp_path / "source.csv"
    source.write_bytes(raw)
    before = helper.digest(source)
    output = io.BytesIO()
    counts = helper.validate_and_annotate(source, output, iter(expected), required_count=required_count or len(expected))
    assert helper.digest(source) == before
    return output.getvalue(), counts


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_labels_preserve_original_records_and_validation_values(tmp_path, small_rows, line_ending):
    _, _, _, expected = small_rows
    original = csv_bytes(expected, line_ending=line_ending)
    labelled, counts = verify(tmp_path, original, expected)
    rows = list(csv.reader(io.StringIO(labelled.decode(), newline="")))
    assert rows[0] == helper.COLUMNS + helper.LABEL_COLUMNS
    assert rows[1][:-2] == expected[0][0]
    assert rows[1][-2:] == ["validation", "final_selection"]
    assert rows[2][-2:] == ["validation", "development_selection"]
    assert rows[3][-2:] == ["validation", "development_selection"]
    # Raw prefixes, quotes, Unicode and embedded newlines survive byte for byte.
    with io.BytesIO(original) as stream:
        tracker = helper.RawCSVLines(stream)
        reader = csv.reader(tracker)
        original_records = []
        for _ in reader:
            original_records.append(tracker.take())
    assert labelled == b"".join(helper.append_labels(record, labels) for record, labels in zip(
        original_records, [helper.LABEL_COLUMNS, ["validation", "final_selection"],
                           ["validation", "development_selection"], ["validation", "development_selection"]]))
    assert counts == {"rows": 3, "final_selection": 1, "new_measurements": 2,
                      "REJECTED_FINAL": 1, "development_selection": 2, "cache_hits": 1,
                      "NOT_SELECTED": 1, "INSUFFICIENT_DATA": 1}


@pytest.mark.parametrize("field,replacement", [
    ("completed_trades", "99"), ("profit_factor", "8.0"), ("net_profit", "900.0"),
    ("max_drawdown", "0.9"), ("validation_score", "44.0"),
    ("settings_json", '{"threshold":1.2}'), ("reasons", "different_final_reason"),
    ("selection_disposition", "QUALIFIED"), ("artifact", "artifacts/other.json.gz"),
    ("attempted_strategy_id", "crt.r99"), ("cache_hit", "FalseFalse"), ("cycle", "32"),
])
def test_validation_or_final_evidence_mismatch_rejected(tmp_path, small_rows, field, replacement):
    _, _, _, expected = small_rows
    tampered = deepcopy(expected)
    tampered[0][0][helper.COLUMNS.index(field)] = replacement
    with pytest.raises(ValueError, match="differs from journal evidence"):
        verify(tmp_path, csv_bytes(tampered), expected)


@pytest.mark.parametrize("change", ["missing", "extra", "reordered", "wrong_header", "wrong_count"])
def test_inventory_boundaries_rejected(tmp_path, small_rows, change):
    _, _, _, expected = small_rows
    rows = deepcopy(expected)
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(rows[-1])
    elif change == "reordered":
        rows.reverse()
    raw = csv_bytes(rows)
    if change == "wrong_header":
        raw = raw.replace(b"validation_score", b"oos_score", 1)
    with pytest.raises(ValueError):
        verify(tmp_path, raw, expected, required_count=4 if change == "wrong_count" else 3)


def test_missing_final_disposition_is_not_guessed(small_rows):
    records, winners, _, _ = small_rows
    with pytest.raises(ValueError, match="lacks its committed final"):
        helper.expected_row(1, records[0], winners, {})


def family_file(tmp_path, results, *, selection_sha="synthetic-sha", **overrides):
    original = tmp_path / "synthetic-original.json.gz"
    original.write_bytes(b"synthetic artifact for hash/lineage unit test only")
    groups = {r["selection_group"]: {"path": original.name, "sha256": helper.digest(original)} for r in results}
    body = {"family_size": len(results), "results": results, "safety": SAFE,
            "selection_sha256": selection_sha, "selected_before_access": True,
            "original_measurements": groups, **overrides}
    path = tmp_path / "family.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
    return path


def final_result():
    from research_campaign.strategies import candidates_for_round
    spec = next(r for r in candidates_for_round(0) if r.family == "crt").to_dict()
    identity = {"strategy": spec, "symbol": "BOOM500", "timeframe": "M5", "start": 1728000000,
                "end": 1729000000, "partition": "final", "costs": {"scenario": "synthetic unit"}}
    gates = dict.fromkeys(sorted(helper.FINAL_GATES), True)
    gates.update(actual_contract_costs_verified=False, minimum_100_completed=False)
    return {**identity, "id": helper.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "strategy_id": spec["id"], "selection_group": "BOOM500/M5/crt", "selection_sha256": "synthetic-sha",
            "safety": SAFE, "qualified": False, "gates": gates,
            "rejection_reasons": ["minimum_100_completed", "actual_contract_costs_verified"],
            "irrelevant_ledger": [{"note": 'عربي "quoted"\ntext'}]}


def final_policy(result):
    return {"spec": result["strategy"], "strategy_id": result["strategy_id"], "symbol": result["symbol"],
            "timeframe": result["timeframe"], "start": result["start"], "end": result["end"],
            "costs": result["costs"], "original_path": "synthetic-original.json.gz"}


def original_verifier(tmp_path):
    def verify(artifact):
        assert helper.digest(tmp_path / artifact["path"]) == artifact["sha256"]
    return verify


def test_final_stream_tiny_chunks_and_reason_order(tmp_path):
    result = final_result()
    path = family_file(tmp_path, [result])
    parsed = list(helper.iter_family(path, chunk_size=7))
    assert parsed[0] == ("result", result)
    assert parsed[-1][1]["family_size"] == 1
    assert helper.final_dispositions(path, "synthetic-sha", {result["selection_group"]: final_policy(result)}, original_verifier(tmp_path)) == {
        result["selection_group"]: {"qualified": False, "rejection_reasons": ["minimum_100_completed", "actual_contract_costs_verified"]}}


@pytest.mark.parametrize("change", ["wrong_selection", "wrong_phase", "unsafe", "reasons", "duplicate", "missing_group", "wrong_family_count", "empty_gates", "extra_gate", "changed_policy", "changed_id", "missing_original", "original_path"])
def test_final_stream_rejects_wrong_lineage(tmp_path, change):
    result = final_result()
    results, extra = [result], {}
    policies = {result["selection_group"]: deepcopy(final_policy(result))}
    if change == "wrong_selection":
        result["selection_sha256"] = "other"
    elif change == "wrong_phase":
        result["partition"] = "validation"
    elif change == "unsafe":
        result["safety"] = {**SAFE, "live_allowed": True}
    elif change == "reasons":
        result["rejection_reasons"] = ["missing"]
    elif change == "duplicate":
        results.append(deepcopy(result))
    elif change == "missing_group":
        policies["CRASH500/M5/crt"] = final_policy(result)
    elif change == "wrong_family_count":
        extra["family_size"] = 2
    elif change == "empty_gates":
        result.update(gates={}, qualified=True, rejection_reasons=[])
    elif change == "extra_gate":
        result["gates"]["invented"] = True
    elif change == "changed_policy":
        result["strategy"]["stop_atr"] = 999
    elif change == "changed_id":
        result["id"] = "arbitrary"
    elif change == "original_path":
        policies[result["selection_group"]]["original_path"] = "different.json.gz"
    path = family_file(tmp_path, results, **extra)
    if change == "missing_original":
        (tmp_path / "synthetic-original.json.gz").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        helper.final_dispositions(path, "synthetic-sha", policies, original_verifier(tmp_path))


def completed_state():
    stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
    start = 1728000000
    return {"status": "COMPLETED", "completed_cycles": 32, "target_cycles": 32,
            "interval_seconds": 900, "final_accessed": True, "current_cycle": None,
            "safety": SAFE, "started_epoch": start, "started_at": stamp(start),
            "deadline_at": stamp(start + 28800), "completed_at": stamp(start + 28800),
            "cycles": [{"number": i + 1, "status": "COMPLETED", "started_at": stamp(start + i * 900),
                        "completed_at": stamp(start + i * 900 + 10)} for i in range(32)]}


@pytest.mark.parametrize("change", ["none", "running", "missing_cycle", "false_count", "short_cadence", "short_window", "live_flag", "numeric_false", "future", "naive", "overlap", "late_cycle_end", "early_first", "nonfinite_start"])
def test_completed_state_and_safe_schedule_required(change):
    state = completed_state()
    if change == "none":
        helper.check_complete_state(state)
        return
    if change == "running":
        state["status"] = "RUNNING"
    elif change == "missing_cycle":
        state["cycles"].pop()
    elif change == "false_count":
        state["completed_cycles"] = 31
    elif change == "short_cadence":
        state["cycles"][1]["started_at"] = state["cycles"][0]["started_at"]
    elif change == "short_window":
        state["completed_at"] = state["cycles"][-1]["completed_at"]
    elif change == "future":
        state["completed_at"] = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + 86400, timezone.utc).isoformat()
    elif change == "naive":
        state["cycles"][0]["started_at"] = state["cycles"][0]["started_at"].replace("+00:00", "")
    elif change == "overlap":
        state["cycles"][0]["completed_at"] = state["cycles"][1]["completed_at"]
    elif change == "late_cycle_end":
        state["cycles"][-1]["completed_at"] = datetime.fromtimestamp(state["started_epoch"] + 30000, timezone.utc).isoformat()
    elif change == "early_first":
        state["cycles"][0]["started_at"] = datetime.fromtimestamp(state["started_epoch"] - 1, timezone.utc).isoformat()
    elif change == "nonfinite_start":
        state["started_epoch"] = float("nan")
    else:
        state["safety"] = {**SAFE, "ready_for_live": True if change == "live_flag" else 0}
    with pytest.raises(ValueError):
        helper.check_complete_state(state)


def test_original_and_existing_evidence_cannot_be_overwritten(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_bytes(b"raw evidence")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="Distinct"):
        helper.annotate(source, source, tmp_path / "receipt.json", tmp_path / "campaign")
    output = tmp_path / "existing.csv"
    output.write_bytes(b"old output")
    with pytest.raises(FileExistsError):
        helper.annotate(source, output, tmp_path / "receipt.json", tmp_path / "campaign")
    assert source.read_bytes() == before and output.read_bytes() == b"old output"


def test_frozen_directory_and_unsafe_environment_rejected(tmp_path, monkeypatch):
    source = tmp_path / "raw.csv"
    source.write_bytes(b"raw evidence")
    base = tmp_path / "campaign"
    with pytest.raises(ValueError, match="outside frozen"):
        helper.annotate(source, base / "new.csv", tmp_path / "receipt.json", base)
    monkeypatch.setenv("LIVE_ALLOWED", "true")
    with pytest.raises(ValueError, match="LIVE_ALLOWED=false"):
        helper.annotate(source, tmp_path / "new.csv", tmp_path / "receipt.json", base)
    assert not (tmp_path / "new.csv").exists()


@pytest.fixture
def catalogue():
    from research_campaign.strategies import candidates_for_round
    return [{s.family: s.to_dict() for s in candidates_for_round(i)} for i in range(32)]


def canonical_record(catalogue):
    spec = catalogue[0]["ict_liquidity_mss"]
    return {"family": spec["family"], "strategy_id": spec["id"], "attempted_strategy_id": spec["id"],
            "spec": deepcopy(spec), "cache_hit": False, "status": "TESTED", "validation_score": .2,
            "validation_metrics": {"completed_trades": 0, "profit_factor": None, "net_profit": 0.0, "max_drawdown": 0.0}}


def test_canonical_record_and_scientific_cache_alias(catalogue):
    record = canonical_record(catalogue)
    helper.validate_record(record, catalogue[0][record["family"]], catalogue)
    attempted = catalogue[2][record["family"]]
    record.update(cache_hit=True, attempted_strategy_id=attempted["id"])
    helper.validate_record(record, attempted, catalogue)


@pytest.mark.parametrize("change", ["stop", "target", "hold", "family", "label", "extra_spec_field", "wrong_status", "missing_metric", "nan_metric", "numeric_count", "bad_dd", "false_cache", "nan_score"])
def test_full_settings_and_metric_schema_required(catalogue, change):
    record = canonical_record(catalogue)
    if change in {"stop", "target", "hold", "family", "label", "extra_spec_field"}:
        field, value = {"stop": ("stop_atr", 999), "target": ("target_atr", 999), "hold": ("max_hold_bars", 99),
                        "family": ("family", "crt"), "label": ("label", "changed"), "extra_spec_field": ("invented", 1)}[change]
        record["spec"][field] = value
    elif change == "wrong_status":
        record["status"] = "BROKEN"
    elif change == "missing_metric":
        record["validation_metrics"].pop("profit_factor")
    elif change == "nan_metric":
        record["validation_metrics"]["net_profit"] = float("nan")
    elif change == "numeric_count":
        record["validation_metrics"]["completed_trades"] = True
    elif change == "bad_dd":
        record["validation_metrics"]["max_drawdown"] = 1.1
    elif change == "false_cache":
        attempted = catalogue[2][record["family"]]
        record["attempted_strategy_id"] = attempted["id"]
        with pytest.raises(ValueError):
            helper.validate_record(record, attempted, catalogue)
        return
    elif change == "nan_score":
        record["validation_score"] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        helper.validate_record(record, catalogue[0][record["family"]], catalogue)
