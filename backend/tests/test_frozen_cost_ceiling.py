"""Synthetic arithmetic/lineage cases; no historical files read by these tests."""
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cost_ceiling", ROOT / "scripts/diagnose_frozen_cost_ceiling.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key in module.SAFETY:
        monkeypatch.setenv(key, "false")


def test_variable_nonnegative_costs_cannot_increase_profit_factor():
    value = module.cost_ceiling([3, 1, -2, -1, 0], [0.2, 2, 0.1, 0, 0.3])
    assert value["gross"]["profit_factor"] == pytest.approx(4 / 3)
    assert value["net"]["profit_factor"] == pytest.approx(2.8 / 4.4)
    assert value["target_1_5_impossible_by_cost_reduction_alone"] is True


def test_zero_cost_identity_and_exact_target_boundary():
    row = module.cost_ceiling([3, -2], [0, 0])
    assert row["gross"] == row["net"]
    assert row["finite_zero_cost_ceiling"] == 1.5
    assert row["target_1_5_impossible_by_cost_reduction_alone"] is False


@pytest.mark.parametrize("gross", [[], [0], [1, 3]])
def test_no_gross_losses_is_undefined_not_a_finite_rejection(gross):
    row = module.cost_ceiling(gross, [0] * len(gross))
    assert row["finite_zero_cost_ceiling"] is None
    assert row["target_1_5_impossible_by_cost_reduction_alone"] is False


@pytest.mark.parametrize("gross,cost", [([1], [-0.1]), ([1], []), ([float("nan")], [0]),
                                       ([1], [float("inf")]), ([True], [0]), ([1], [True])])
def test_rebates_unknowns_bool_and_wrong_shapes_are_refused(gross, cost):
    with pytest.raises(ValueError):
        module.cost_ceiling(gross, cost)


def test_bound_holds_when_cost_changes_wins_to_losses():
    for costs in ([0, 0, 0], [.1, .2, .5], [1, 2, 3], [10, 10, 10]):
        row = module.cost_ceiling([2, .1, -1], costs)
        assert row["net"]["gains_R"] <= row["gross"]["gains_R"]
        assert row["net"]["losses_R"] >= row["gross"]["losses_R"]
        assert row["net"]["profit_factor"] <= row["finite_zero_cost_ceiling"]


@pytest.mark.parametrize("key", module.SAFETY)
def test_true_gate_is_refused(monkeypatch, key):
    monkeypatch.setenv(key, "true")
    with pytest.raises(ValueError, match=key):
        module.guard()


def test_metadata_safety_requires_literal_false():
    with pytest.raises(ValueError):
        module.safety({"nested": [{"LIVE_ALLOWED": 0}]})


def test_pin_refuses_tampering_path_escape_and_symlink(tmp_path):
    source = tmp_path / "source"; source.write_text("fixed")
    old = module.digest(source)
    assert module.pin(tmp_path, "source", old, {}) == source
    source.write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        module.pin(tmp_path, "source", old, {})
    for name in ("../source", str(source)):
        with pytest.raises(ValueError):
            module.safe_file(tmp_path, name)
    (tmp_path / "link").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        module.safe_file(tmp_path, "link")


START = 1775520000  # 2026-04-07 00:00 UTC


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def rows(symbol="BOOM600", family="BOOST44", mode="SPIKE"):
    side = (1 if symbol == "BOOM600" else -1) * (1 if mode == "SPIKE" else -1)
    result = []
    for i, gross in enumerate([-1.0, 2.0]):
        signal = START + i * 1800
        values = [stamp(signal), stamp(signal + 60), stamp(signal + 960), 100,
                  100 + side * 2 * gross, 1, gross, gross - .05, "time", "False", 15,
                  "False", f"NONLINEAR_{family}_{mode}", stamp(signal + 960), ""]
        result.append(dict(zip(module.FIELDS, values)))
    return result


def write_csv(path, data, fieldnames=None):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or module.FIELDS)
        writer.writeheader(); writer.writerows(data)


def test_ledger_recomputes_cash_difference_and_counts_entry_days(tmp_path):
    path = tmp_path / "ledger.csv"; write_csv(path, rows())
    gross, net, days = module.read_ledger(path, "BOOM600", "BOOST44", "SPIKE", START, START + 86400)
    assert gross == [-1, 2] and net == [-1.05, 1.95] and days == ["2026-04-07"]


@pytest.mark.parametrize("field,value", [("censored", "True"), ("missing_time", "2026-04-07T01:00:00+00:00"),
                                         ("gross_R", 99), ("net_R", 99), ("atr", 0),
                                         ("signal_time", stamp(START + 1)), ("variant", "wrong")])
def test_invalid_ledger_cannot_silently_shrink_complete_sample(tmp_path, field, value):
    data = rows(); data[0][field] = value
    path = tmp_path / "ledger.csv"; write_csv(path, data)
    with pytest.raises(ValueError):
        module.read_ledger(path, "BOOM600", "BOOST44", "SPIKE", START, START + 86400)


def test_duplicate_or_ragged_ledger_is_rejected(tmp_path):
    path = tmp_path / "ledger.csv"; data = rows(); write_csv(path, [data[0], data[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        module.read_ledger(path, "BOOM600", "BOOST44", "SPIKE", START, START + 86400)
    write_csv(path, data)
    with path.open("a") as stream:
        stream.write("too,few,columns\n")
    with pytest.raises(ValueError, match="Ragged"):
        module.read_ledger(path, "BOOM600", "BOOST44", "SPIKE", START, START + 86400)


def test_all_twelve_models_end_to_end_and_frozen_ledger_tamper(tmp_path, monkeypatch):
    study = tmp_path / module.STUDY; study.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    helper = tmp_path / "scripts/verify_spike_nonlinear.py"; helper.write_text("synthetic hash anchor\n")
    def save(name, value):
        p = study / name; p.write_text(json.dumps(value)); return module.digest(p)
    result = {"code_hashes": {}, "safety": module.SAFETY, "symbols": {}}
    result["declaration_sha256"] = save("declaration.json", {"code_hashes": {}})
    result["selection_sha256"] = save("selection.json", {"code_hashes": {}})
    audit = {"status": "PASS", "pass": True, "error_count": 0, "errors": [], "helper_sha256": {},
             "script_sha256": module.digest(helper), "selection_sha256": result["selection_sha256"], "ledgers": []}
    for symbol in module.SYMBOLS:
        cohort = {"start": stamp(START), "end": stamp(START + 86400), "models": {}}
        result["symbols"][symbol] = {"cohorts": {module.COHORT: cohort}}
        for family in module.FAMILIES:
            for mode in module.MODES:
                prefix = study / f"{symbol}_{module.COHORT}_{family}_{mode}"
                ledger = Path(str(prefix) + "_trades.csv"); write_csv(ledger, rows(symbol, family, mode))
                signals = Path(str(prefix) + "_signals.csv"); signals.write_text("already audited synthetic signals")
                metrics = {"completed": 2, "active_days": 1, "mean_net_R": .45, "sum_net_R": .9,
                           "profit_factor": 1.95 / 1.05}
                zero = {**metrics, "mean_net_R": .5, "sum_net_R": 1, "profit_factor": 2,
                        "fill_mode": "adverse_extreme", "entry_delay_minutes": 1, "round_trip_cost_atr": 0}
                execution = {k: 0 for k in ("censored", "missing_entry", "outside_partition", "purged", "overlap_skipped", "ambiguous")}
                execution.update(config=module.PRIMARY, issued=2, filled=2, completed=2)
                cohort["models"][f"{family}_{mode}"] = {"config": module.PRIMARY, "audit": execution,
                    "metrics": metrics, "sensitivities": [zero], "development_eligible": False}
                audit["ledgers"].append({"symbol": symbol, "family": family, "mode": mode,
                    "cohort": module.COHORT, "trades": 2, "completed": 2, "issued_signals": 2,
                    "censored": 0, "purged": 0, "missing_entries": 0, "mismatches": 0,
                    "ledger_sha256": module.digest(ledger), "signals_sha256": module.digest(signals),
                    "profit_factor": metrics["profit_factor"], "mean_net_R": .45})
    result_hash = save("results.json", result); audit["results_sha256"] = result_hash
    monkeypatch.setattr(module, "RESULT_SHA", result_hash)
    monkeypatch.setattr(module, "AUDIT_SHA", save("independent_audit.json", audit))
    measured = module.verify(tmp_path)
    assert measured["status"] == "PASS" and len(measured["rows"]) == 12
    assert measured["total_primary_records"] == 24
    assert not measured["all_fixed_models_below_target_even_without_cost"]
    assert all(not row["historical_candidate"] for row in measured["rows"])
    ledger.write_text(ledger.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="Changed frozen input"):
        module.verify(tmp_path)
