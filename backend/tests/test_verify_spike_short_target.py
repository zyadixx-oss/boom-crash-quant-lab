"""Synthetic independent-auditor checks; never decode historical files."""
from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import struct

import pytest

from scripts import verify_spike_short_target as v


def time_text(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(sep=" ")


def descriptor(tmp_path, monkeypatch, name, rows, columns, *, indexed=False):
    monkeypatch.setattr(v, "ROOT", tmp_path)
    path = tmp_path / name
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(["signal_time"] if indexed else []) + columns)
        writer.writeheader(); writer.writerows(rows)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows), "columns": columns, "local_ignored_artifact": True}


def ledger_raw(rows):
    result = []
    for row in rows:
        result.append({k: time_text(val) if k in ("signal_time", "entry_time", "exit_time", "planned_end", "missing_time") and val is not None
                       else "" if val is None else val for k, val in row.items()})
    return result


@pytest.mark.parametrize("bad", ["", ".", "..", "/tmp/a", "a/../b", "a//b", "./a", "a\\b", "a/.", 3, None])
def test_paths_reject_aliases_and_traversal(bad):
    with pytest.raises(ValueError):
        v.relative_path(bad)


def test_path_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(v, "ROOT", root)
    with pytest.raises(ValueError, match="symlink"):
        v.relative_path("escape/file")


@pytest.mark.parametrize(("actual", "expected"), [(True, 1), (False, 0), (1, True), (0, False), (1., 1)])
def test_metadata_boolean_integer_alias_refused(actual, expected):
    with pytest.raises(ValueError):
        v.compare(v.Audit(), "typed", actual, expected)


def test_nonfinite_comparison_fails_even_when_both_nan():
    audit = v.Audit(); audit.equal("nan", float("nan"), float("nan"))
    assert audit.failures


def test_little_endian_hash_contract():
    assert v.hash_doubles([1.]) == hashlib.sha256(bytes.fromhex("000000000000f03f")).hexdigest()
    assert v.hash_times([1]) == hashlib.sha256(struct.pack("<q", 1_000_000_000)).hexdigest()


@pytest.mark.parametrize("key,value", [("sha256", "0"*64), ("rows", 2), ("columns", ["wrong"]), ("local_ignored_artifact", 1)])
def test_artifact_tamper_refused(tmp_path, monkeypatch, key, value):
    desc = descriptor(tmp_path, monkeypatch, "artifact.csv", [{"x": 1}], ["x"])
    desc[key] = value
    audit = v.Audit()
    try:
        v.csv_rows(desc, audit)
    except ValueError:
        return
    assert audit.failures


def test_artifact_good(tmp_path, monkeypatch):
    desc = descriptor(tmp_path, monkeypatch, "artifact.csv", [{"x": 1}], ["x"])
    audit = v.Audit()
    assert v.csv_rows(desc, audit) == [{"x": "1"}]
    assert not audit.failures


def inputs_and_signals():
    t = 1800
    row = {name: 1. for name in v.NAMES}; row.update(atr=1., close=100.)
    return {t: row}, v.expected_signals({t: row}, "BOOM600", "SPIKE")


@pytest.mark.parametrize(("symbol", "mode", "side"), [("BOOM600", "SPIKE", 1), ("BOOM600", "DRIFT", -1), ("CRASH600", "SPIKE", -1), ("CRASH600", "DRIFT", 1)])
def test_native_mode_sides(symbol, mode, side):
    inputs, _ = inputs_and_signals()
    assert v.expected_signals(inputs, symbol, mode)[0]["side"] == side


@pytest.mark.parametrize(("symbol", "mode"), [("BOOM500", "SPIKE"), ("BOOM600", "SELL")])
def test_undeclared_symbols_directions_refused(symbol, mode):
    with pytest.raises(ValueError):
        v.expected_signals({}, symbol, mode)


def test_short_consumes_one_entry_bar_and_next_open_is_irrelevant():
    _, signals = inputs_and_signals()
    minutes = {1860: (100., 101., 99., 100.8), 1920: (500., 1000., 1., 999.)}
    rows, counts = v.replay(minutes, signals, 1800, 4000)
    assert counts["completed"] == 1
    assert rows[0]["entry_time"] == 1860 and rows[0]["exit_time"] == 1920
    assert rows[0]["exit"] == 100.8 and rows[0]["net_R"] == pytest.approx(.35)
    assert rows[0]["holding_minutes"] == 1.


@pytest.mark.parametrize(("side", "bar", "expected"), [(1, (100., 120., 94., 110.), 94.), (-1, (100., 108., 70., 90.), 108.)])
def test_adverse_minute_extreme_not_barrier_fill(side, bar, expected):
    _, signals = inputs_and_signals(); signals[0]["side"] = side
    rows, _ = v.replay({1860: bar}, signals, 1800, 4000)
    assert rows[0]["reason"] == "sl" and rows[0]["exit"] == expected
    assert rows[0]["net_R"] < -1.


def test_common31minute_purge_even_for_one_minute_horizon():
    _, signals = inputs_and_signals()
    rows, counts = v.replay({1860: (100., 101., 99., 100.)}, signals, 1800, 1920)
    assert rows == [] and counts["purged"] == 1


def test_missing_entry_is_not_a_zero_reward():
    _, signals = inputs_and_signals()
    rows, counts = v.replay({}, signals, 1800, 4000)
    assert rows == [] and counts["missing_entry"] == 1


def test_missing_long_future_censored_but_short_known():
    _, signals = inputs_and_signals()
    minutes = {1860: (100., 101., 99., 100.5)}
    short, _ = v.replay(minutes, signals, 1800, 4000)
    long, counts = v.replay(minutes, signals, 1800, 4000, v.TARGET_CONFIGS["LONG44"])
    assert not short[0]["censored"]
    assert long[0]["censored"] and long[0]["net_R"] is None and long[0]["missing_time"] == 1920
    assert counts["censored"] == 1
    assert v.common_targets({"SHORT44": short, "LONG44": long}) == {}


def test_signal_duplicate_refused():
    _, signals = inputs_and_signals()
    with pytest.raises(ValueError, match="sorted and unique"):
        v.replay({}, signals*2, 1800, 4000)


def test_unknown_path_reserves_occupancy():
    _, signals = inputs_and_signals()
    signals += [{**signals[0], "signal_time": 2100}]
    minutes = {1860: (100., 101., 99., 100.5), 2160: (100., 101., 99., 100.5)}
    rows, counts = v.replay(minutes, signals, 1800, 5000, v.TARGET_CONFIGS["LONG44"])
    assert len(rows) == 1 and counts["overlap_skipped"] == 1


@pytest.mark.parametrize("field", ["entry", "exit", "gross_R", "net_R", "reason", "censored", "signal_time", "variant"])
def test_saved_ledger_tampering_detected(tmp_path, monkeypatch, field):
    _, signals = inputs_and_signals()
    rows, _ = v.replay({1860: (100., 101., 99., 100.5)}, signals, 1800, 4000)
    raw = ledger_raw(rows)
    raw[0][field] = "sl" if field == "reason" else "True" if field == "censored" else time_text(0) if field == "signal_time" else "changed" if field == "variant" else 777
    desc = descriptor(tmp_path, monkeypatch, "ledger.csv", raw, list(raw[0]))
    audit = v.Audit(); v.audit_ledger(desc, rows, audit, "ledger")
    assert audit.failures


def test_saved_ledger_matches_scalar(tmp_path, monkeypatch):
    _, signals = inputs_and_signals()
    rows, _ = v.replay({1860: (100., 101., 99., 100.5)}, signals, 1800, 4000)
    raw = ledger_raw(rows)
    desc = descriptor(tmp_path, monkeypatch, "ledger.csv", raw, list(raw[0]))
    audit = v.Audit(); v.audit_ledger(desc, rows, audit, "ledger")
    assert not audit.failures


def test_saved_signal_exact_side_and_score(tmp_path, monkeypatch):
    _, signals = inputs_and_signals()
    raw = [{**signals[0], "signal_time": time_text(1800), "score": ""}]
    desc = descriptor(tmp_path, monkeypatch, "signals.csv", raw, list(raw[0]))
    audit = v.Audit(); v.audit_signals(desc, signals, audit, "signals")
    assert not audit.failures


def model_fixture(n=1000, family="SHORT44"):
    first, end = 0, (n+1)*1800
    inputs, common = {}, {}
    for i in range(n):
        t = i*1800
        inputs[t] = {name: float(j) for j, name in enumerate(v.NAMES)}
        common[t] = ({"net_R": .2, "planned_end": t+120}, {"net_R": .4, "planned_end": t+960})
    estimator = {"version": 1, "feature_names": list(v.NAMES), "fit_rows": n, "penalty": .1, "quantile": .75,
        "score_kind": "continuous_uncalibrated_score", "objective": "mean_squared_error_plus_penalty_times_squared_coefficients",
        "scaler_fitted_on": "training_rows_only", "target_clipped": False, "features_clipped": False,
        "means": list(map(float, range(44))), "std": [1.]*44, "coefs": [0.]*44,
        "intercept": .2 if family == "SHORT44" else .4, "threshold": .2 if family == "SHORT44" else .4}
    model = {"family": family, "status": "TESTED", "target_name": "SHORT1" if family == "SHORT44" else "LONG15",
        "feature_names": list(v.NAMES), "training_completed_labels": n, "target_horizon_minutes": 1 if family == "SHORT44" else 15,
        "execution_horizon_minutes": 1, "label_purge_minutes": 31, "training_start": time_text(first), "training_end": time_text(end),
        "estimator": estimator, "serialized_estimator_sha256": v.canonical_hash(estimator),
        "shared_scaler_sha256": v.canonical_hash({key: estimator[key] for key in ("means", "std")})}
    return model, inputs, common, first, end


@pytest.mark.parametrize("family", v.FAMILIES)
def test_independent_model_scaler_intercept_q75(family):
    model, inputs, common, start, end = model_fixture(family=family)
    audit = v.Audit(); v.validate_model(model, inputs, common, start, end, audit, "fit")
    assert not audit.failures


@pytest.mark.parametrize(("field", "value"), [("version", True), ("version", 2), ("fit_rows", True),
    ("penalty", .2), ("quantile", .5), ("scaler_fitted_on", "full_history"), ("target_clipped", True),
    ("features_clipped", True), ("objective", "sum_squared_error"), ("score_kind", "probability"),
    ("intercept", .9), ("threshold", .7), ("means", [0.]*44), ("std", [2.]*44)])
def test_rehashed_semantic_or_training_tamper_refused(field, value):
    model, inputs, common, start, end = model_fixture()
    model["estimator"][field] = value
    model["serialized_estimator_sha256"] = v.canonical_hash(model["estimator"])
    model["shared_scaler_sha256"] = v.canonical_hash({key: model["estimator"][key] for key in ("means", "std")})
    audit = v.Audit()
    try:
        v.validate_model(model, inputs, common, start, end, audit, "fit")
    except ValueError:
        return
    assert audit.failures


@pytest.mark.parametrize("field", ["target_horizon_minutes", "execution_horizon_minutes", "label_purge_minutes", "training_completed_labels"])
def test_top_model_boolean_alias_refused(field):
    model, inputs, common, start, end = model_fixture()
    model[field] = True
    with pytest.raises(ValueError):
        v.validate_model(model, inputs, common, start, end, v.Audit(), "fit")


def test_empty_model_remains_not_tested_and_no_issuance():
    model, inputs, common, start, end = model_fixture(n=0)
    model.update(status="NOT TESTED", estimator=None, serialized_estimator_sha256=None, shared_scaler_sha256=None)
    audit = v.Audit(); v.validate_model(model, inputs, common, start, end, audit, "fit")
    assert not audit.failures and v.expected_signals(inputs, "BOOM600", "SPIKE", model) == []


def test_training_targets_intersection_excludes_unknowns():
    labels = {"SHORT44": [{"signal_time": 0, "censored": False, "net_R": .2}, {"signal_time": 1, "censored": False, "net_R": .3}],
              "LONG44": [{"signal_time": 0, "censored": True, "net_R": None}, {"signal_time": 1, "censored": False, "net_R": .4}]}
    assert list(v.common_targets(labels)) == [1]


@pytest.mark.parametrize("unknown", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_targets_remain_outside_shared_training(unknown):
    labels = {"SHORT44": [{"signal_time": 1, "censored": False, "net_R": unknown}],
              "LONG44": [{"signal_time": 1, "censored": False, "net_R": .4}]}
    assert v.common_targets(labels) == {}


def training_context_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "ROOT", tmp_path)
    start, n = 1800, 1001
    end = start+(n+1)*1800
    inputs = {start+i*1800: {**{name: float(j) for j, name in enumerate(v.NAMES)}, "atr": 1., "close": 100.}
              for i in range(n)}
    minutes = {t: (100., 101.2, 99.5, 100.2) for t in range(start, end, 60)}
    for t in inputs:
        minutes[t+15*60] = (100., 101.2, 99.5, 101.)
    del minutes[start+120]  # SHORT complete; corresponding LONG unknown.
    signals = v.expected_signals(inputs, "BOOM600", "SPIKE")
    labels, audits, artifacts = {}, {}, {}
    for family in v.FAMILIES:
        labels[family], counts = v.replay(minutes, signals, start, end, v.TARGET_CONFIGS[family])
        audits[family] = {**counts, "config": v.TARGET_CONFIGS[family], "purge_minutes": 31,
            "take_profit": None, "safety": dict.fromkeys(v.FLAGS, False),
            "fill_interpretation": "hypothetical_ohlc_proxy_or_extreme_stress_not_measured_execution",
            "training_targets_not_strategy_evidence": True,
            "target_horizon_minutes": v.TARGET_CONFIGS[family]["max_hold_minutes"], "common_purge_minutes": 31}
        raw = ledger_raw(labels[family])
        artifacts[family] = descriptor(tmp_path, monkeypatch, family+".csv", raw, list(raw[0]))
    common = v.common_targets(labels)
    assert len(common) == 1000
    matrix_rows = [{"signal_time": time_text(t), **{name: inputs[t][name] for name in v.NAMES},
        "short_net_R": pair[0]["net_R"], "long_net_R": pair[1]["net_R"],
        "short_planned_end": time_text(pair[0]["planned_end"]), "long_planned_end": time_text(pair[1]["planned_end"])}
        for t, pair in common.items()]
    matrix = descriptor(tmp_path, monkeypatch, "matrix.csv", matrix_rows,
        [*v.NAMES, "short_net_R", "long_net_R", "short_planned_end", "long_planned_end"], indexed=True)
    identity = {"common_training_rows": len(common), "feature_names": list(v.NAMES),
        "byte_hash_schema": {"matrix_and_targets": "C_row_major_little_endian_float64", "issuance_times": "little_endian_int64_UTC_nanoseconds"},
        "matrix_native_sha256": v.hash_doubles(inputs[t][name] for t in common for name in v.NAMES),
        "issue_timestamps_native_sha256": v.hash_times(common),
        "short_target_native_sha256": v.hash_doubles(r[0]["net_R"] for r in common.values()),
        "long_target_native_sha256": v.hash_doubles(r[1]["net_R"] for r in common.values()),
        "training_latest_issue": time_text(max(common)),
        "training_latest_long_planned_end": time_text(max(r[1]["planned_end"] for r in common.values()))}
    models = {}
    for family in v.FAMILIES:
        model, _, _, _, _ = model_fixture(family=family)
        y = statistics.fmean(pair[0 if family == "SHORT44" else 1]["net_R"] for pair in common.values())
        model["estimator"].update(intercept=y, threshold=max(0., y))
        model.update(training_start=time_text(start), training_end=time_text(end), training_identity=identity,
            artifacts={"matrix_and_both_targets": matrix}, serialized_estimator_sha256=v.canonical_hash(model["estimator"]))
        models[family] = model
    context = {"identity": identity, "models": models, "target_artifacts": artifacts, "target_audits": audits}
    return context, inputs, minutes, start, end


def test_full_dual_target_matrix_and_common_scaler_audit(tmp_path, monkeypatch):
    context, inputs, minutes, start, end = training_context_fixture(tmp_path, monkeypatch)
    audit = v.Audit()
    labels, rows = v.audit_training(context, inputs, minutes, "BOOM600", "SPIKE", start, end, audit, "training")
    assert labels == 2002 and rows == 1000
    assert not audit.failures


@pytest.mark.parametrize("kind", ["native_x", "native_clock", "native_short", "native_long", "schema", "common_membership", "family_swap"])
def test_dual_target_matrix_identity_tamper_detected(tmp_path, monkeypatch, kind):
    context, inputs, minutes, start, end = training_context_fixture(tmp_path, monkeypatch)
    if kind == "family_swap":
        context["models"]["SHORT44"], context["models"]["LONG44"] = context["models"]["LONG44"], context["models"]["SHORT44"]
    elif kind == "common_membership":
        context["identity"]["common_training_rows"] = 1001
    elif kind == "schema":
        context["identity"]["byte_hash_schema"]["issuance_times"] = "microseconds"
    else:
        field = {"native_x": "matrix_native_sha256", "native_clock": "issue_timestamps_native_sha256",
                 "native_short": "short_target_native_sha256", "native_long": "long_target_native_sha256"}[kind]
        context["identity"][field] = "0"*64
    audit = v.Audit()
    v.audit_training(context, inputs, minutes, "BOOM600", "SPIKE", start, end, audit, "training")
    assert audit.failures


def inference_fixture():
    metric = {"mean_net_R": .2, "completed": 1200, "active_days": 100, "calendar_days": 180,
              "censored": 0, "invalid_uncensored": 0}
    model = {"metrics": metric, "audit": {"missing_entry": 0}}
    reference = {"metrics": {**metric, "mean_net_R": .1}, "audit": {"missing_entry": 0}}
    day = {"baseline_mean_net_R": .1, "baseline_difference": .1, "completed": 1200, "baseline_completed": 1200,
        "active_days": 100, "baseline_active_days": 100, "calendar_days": 180, "bootstrap_repeats": 9999,
        "bootstrap_seed": 20261005, "paired_utc_day_resampling": True, "mean_p": .001, "difference_p": .002,
        "p": .002, "bootstrap_mean_valid_replicates": 9999, "bootstrap_difference_valid_replicates": 9999}
    week = {"weekly_block_days": 7, "weekly_bootstrap_repeats": 9999, "weekly_bootstrap_seed": 20261005,
            "weekly_p": .003, "weekly_valid_mean_replicates": 9999, "weekly_valid_difference_replicates": 9999}
    test = {"day": day, "weekly": week, "policy_inference_available": True, "p": .003,
        "policy_unknowns_forbid_promotion": False, "day_mean_undefined_replicates": 0,
        "day_difference_undefined_replicates": 0, "weekly_mean_undefined_replicates": 0, "weekly_difference_undefined_replicates": 0}
    return test, model, reference


def test_valid_reference_inference_accounting():
    test, model, ref = inference_fixture()
    audit = v.Audit(); v.audit_inference(test, model, ref, 0, 180*86400, audit, "comparison")
    assert not audit.failures


@pytest.mark.parametrize("kind", ["missing_reference", "censored_reference", "day_mean_draw", "day_diff_draw", "weekly_mean_draw", "weekly_diff_draw"])
def test_unknown_reference_or_draw_cannot_claim_inference(kind):
    test, model, ref = inference_fixture()
    if kind == "missing_reference": ref["audit"]["missing_entry"] = 1
    elif kind == "censored_reference": ref["metrics"]["censored"] = 1
    elif kind.startswith("day_"): test["day"]["bootstrap_mean_valid_replicates" if "mean" in kind else "bootstrap_difference_valid_replicates"] = 9998
    else: test["weekly"]["weekly_valid_mean_replicates" if "mean" in kind else "weekly_valid_difference_replicates"] = 9998
    audit = v.Audit(); v.audit_inference(test, model, ref, 0, 180*86400, audit, "comparison")
    assert audit.failures


@pytest.mark.parametrize("count", [9999., True, -1, 10000])
def test_inference_draw_counts_exact_integer_and_bounded(count):
    test, model, ref = inference_fixture()
    test["day"]["bootstrap_mean_valid_replicates"] = count
    with pytest.raises(ValueError, match="exact_draw_count"):
        v.audit_inference(test, model, ref, 0, 180*86400, v.Audit(), "comparison")


def positive_gate_fixture():
    test, _, _ = inference_fixture()
    test["day"].update(mean_net_R_ci95=[.1, .3], baseline_difference_ci95=[.05, .15])
    test["weekly"].update(weekly_mean_net_R_ci95=[.1, .3], weekly_difference_ci95=[.05, .15])
    return {"family": "SHORT44", "metrics": {"completed": 1200, "profit_factor": 1.6, "mean_net_R": .2,
        "active_days": 100, "censored": 0, "invalid_uncensored": 0, "day_undefined_replicates": 0,
        "weekly_undefined_replicates": 0, "day_profit_factor_ci95": [1.1, 2.1],
        "weekly_profit_factor_ci95": [1.1, 2.1], "holm_p": .012, "equity_ruin": False, "closed_trade_max_drawdown": .05},
        "audit": {"missing_entry": 0}, "development_eligible": True,
        "thirds": [{"metrics": {"completed": 400, "mean_net_R": .2}} for _ in range(3)],
        "doubled_cost": {"mean_net_R": .15}, "comparisons": {"LONG44": deepcopy(test), "CLOCK": deepcopy(test)},
        "user_target_observed": True, "historical_candidate": False, "conditional_economic_gates_pass": True,
        "conditional_lower_pf_at_least1_5": False, "supports_expected_pf_1_5": False, "reference_only": False,
        "rejection_reasons": ["adaptive_known_history_requires_new_evidence", "broker_costs_and_fills_not_tested"],
        "actual_money_profit": "NOT TESTED", "prospective_validation": "NOT TESTED"}


def test_even_all_economic_gates_keep_known_history_non_candidate():
    row = positive_gate_fixture(); audit = v.Audit()
    v.audit_gate(row, audit, "gate")
    assert not audit.failures
    row["historical_candidate"] = True
    audit = v.Audit()
    try:
        v.audit_gate(row, audit, "gate")
    except ValueError:
        pass
    assert audit.failures


@pytest.mark.parametrize("kind", ["mean", "sample", "days", "third_sample", "unknown_reference"])
def test_economic_gate_failures_are_detected(kind):
    row = positive_gate_fixture()
    if kind == "mean": row["metrics"]["mean_net_R"] = .099
    elif kind == "sample": row["metrics"]["completed"] = 999
    elif kind == "days": row["metrics"]["active_days"] = 59
    elif kind == "third_sample": row["thirds"][0]["metrics"]["completed"] = 199
    else: row["comparisons"]["LONG44"]["policy_inference_available"] = False
    audit = v.Audit()
    try:
        v.audit_gate(row, audit, "gate")
    except ValueError:
        pass
    assert audit.failures


def test_holm_keeps_all_four_even_empty_or_weak():
    assert v.v6.holm_values([.001, .01, 1., 1.]) == [.004, .03, 1., 1.]


def test_chronological_thirds_each_apply_original31minute_planned_purge():
    start, end = 0, 3*86400
    paths = []
    for day in range(3):
        for t in (day*86400+1800, (day+1)*86400-1800):
            paths.append({"signal_time": t, "entry_time": t+60, "exit_time": t+120,
                "entry": 100., "exit": 100.5, "atr": 1., "gross_R": .25, "net_R": .2,
                "reason": "time", "ambiguous": False, "censored": False, "holding_minutes": 1.,
                "planned_end": t+120, "missing_time": None, "variant": "CLOCK_SPIKE"})
    thirds = [{"start": time_text(day*86400), "end": time_text((day+1)*86400),
               "metrics": v.v3.mixed_summary([paths[day*2]], day*86400, (day+1)*86400, 2., .0025)} for day in range(3)]
    audit = v.Audit(); v.audit_thirds({"thirds": thirds}, paths, start, end, audit, "thirds")
    assert not audit.failures
    assert all(third["metrics"]["completed"] == 1 for third in thirds)


def test_third_purge_boundary_is_inclusive_when_exact31minutes_remain():
    start, end = 0, 3*86400
    t = 86400-31*60
    row = {"signal_time": t, "entry_time": t+60, "exit_time": t+120,
        "entry": 100., "exit": 100.5, "atr": 1., "gross_R": .25, "net_R": .2,
        "reason": "time", "ambiguous": False, "censored": False, "holding_minutes": 1.,
        "planned_end": t+120, "missing_time": None, "variant": "CLOCK_SPIKE"}
    thirds = [{"start": time_text(day*86400), "end": time_text((day+1)*86400),
        "metrics": v.v3.mixed_summary([row] if day == 0 else [], day*86400, (day+1)*86400, 2., .0025)} for day in range(3)]
    audit = v.Audit(); v.audit_thirds({"thirds": thirds}, [row], start, end, audit, "thirds")
    assert not audit.failures


@pytest.mark.parametrize("family", ["LONG44", "CLOCK"])
def test_favorable_references_never_candidate(family):
    row = {"family": family, "historical_candidate": True, "supports_expected_pf_1_5": True,
           "reference_only": True, "rejection_reasons": ["fixed_diagnostic_reference_only"]}
    audit = v.Audit(); v.audit_gate(row, audit, "gate")
    assert audit.failures


def test_standard_library_only_imports():
    source = Path(v.__file__).read_text()
    assert "from app." not in source and "import numpy" not in source and "import pandas" not in source
    assert "run_spike" not in source
