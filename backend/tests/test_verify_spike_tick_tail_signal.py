"""Synthetic-only independent M5 economic audit checks; no historical reads."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import verify_spike_tick_tail_signal as a

START = a.v7.date_bounds(a.DATES[0])[0]


def signal(t=START, side=1):
    return {"signal_time": t, "atr": 1., "side": side, "variant": "CLOCK_SPIKE",
            "signal_close": 100., "prior_age_seconds": 600.}


def quotes(start=START, end=START + 5000):
    return dict.fromkeys(range(start, end), 100.)


def saved_inference(undefined=0, interval=None):
    return {"ci95": [-1., 1.] if interval is None and undefined < 9999 else interval,
            "undefined_draws": undefined, "all_draws_defined": undefined == 0,
            "conditional_on_defined_draws": undefined > 0}


def csv_file(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    return path


def test_m5_signals_accept_minute05_clock_and_omit_clock_score(tmp_path):
    raw = {**signal(START + 300), "signal_time": a.iso(START + 300), "score": ""}
    columns = ["signal_time", "atr", "side", "variant", "signal_close", "prior_age_seconds", "score"]
    found = a.read_signals(csv_file(tmp_path / "s.csv", columns, [raw]))
    assert found == [signal(START + 300)]
    assert "score" not in found[0]
    expected = json.dumps([{**signal(START + 300), "signal_time": a.iso(START + 300)}],
                          sort_keys=True, separators=(",", ":")).encode()
    assert a.issuance_hash(found) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("fault", ["off_clock", "duplicate", "reversed", "naive", "nan", "zero_age"])
def test_signal_parser_refuses_unsafe_saved_schedule(tmp_path, fault):
    raws = [{**signal(t), "signal_time": a.iso(t), "score": ""} for t in (START, START + 300)]
    if fault == "off_clock": raws[0]["signal_time"] = a.iso(START + 60)
    if fault == "duplicate": raws[1]["signal_time"] = raws[0]["signal_time"]
    if fault == "reversed": raws.reverse()
    if fault == "naive": raws[0]["signal_time"] = "2026-04-11T00:00:00"
    if fault == "nan": raws[0]["atr"] = "nan"
    if fault == "zero_age": raws[0]["prior_age_seconds"] = 0
    with pytest.raises(ValueError):
        a.read_signals(csv_file(tmp_path / "s.csv", list(raws[0]), raws))


def test_two_overlapping_labels_are_distinct_from_one_open_strategy():
    issued = [signal(), signal(START + 300)]
    overlapping, target = a.replay_partition(quotes(), issued, START, START + 5000, overlapping=True)
    strategy, portfolio = a.replay_partition(quotes(), issued, START, START + 5000)
    assert len(overlapping) == target["completed"] == 2
    assert len(strategy) == portfolio["completed"] == portfolio["overlap_skipped"] == 1
    assert all(r["net_R"] == -.05 for r in overlapping)


@pytest.mark.parametrize("side,trigger,successor,expected", [(1, 95., 94., -3.05), (-1, 105., 106., -3.05)])
def test_irreversible_jump_stop_uses_successor_quote_and_overshoot(side, trigger, successor, expected):
    ticks = quotes()
    ticks[START + 62], ticks[START + 63] = trigger, successor
    ticks[START + 64] = 100.
    rows, counts = a.replay_partition(ticks, [signal(side=side)], START, START + 5000)
    assert counts["completed"] == 1
    row = rows[0]
    assert row["reason"] == "sl" and row["trigger_time"] == START + 62 and row["exit_time"] == START + 63
    assert row["net_R"] == expected


def test_deadline_crossing_has_priority_and_timeout_is_strict_next_second():
    ticks = quotes()
    ticks[START + 960], ticks[START + 961] = 98., 97.
    rows, _ = a.replay_partition(ticks, [signal()], START, START + 5000)
    assert rows[0]["reason"] == "sl" and rows[0]["net_R"] == -1.55
    ticks[START + 960] = 100.
    rows, _ = a.replay_partition(ticks, [signal()], START, START + 5000)
    assert rows[0]["reason"] == "time" and rows[0]["exit_time"] == START + 961


def test_missing_entry_and_gap_do_not_become_zero_payoffs():
    ticks = quotes()
    del ticks[START + 61]
    rows, counts = a.replay_partition(ticks, [signal(), signal(START + 300)], START, START + 5000)
    assert not rows and counts["missing_entry"] == 1 and counts["overlap_skipped"] == 1
    ticks = quotes()
    del ticks[START + 62]
    rows, counts = a.replay_partition(ticks, [signal(), signal(START + 300)], START, START + 5000)
    assert counts["censored"] == counts["overlap_skipped"] == 1
    assert rows[0]["net_R"] is None and rows[0]["missing_time"] == START + 62


def test_missing_quote_after_known_exit_is_irrelevant_and_releases_occupancy():
    ticks = quotes()
    ticks[START + 62] = 97.
    del ticks[START + 64]
    rows, counts = a.replay_partition(ticks, [signal(), signal(START + 300)], START, START + 5000)
    assert counts["completed"] == 2 and counts["censored"] == 0
    assert rows[0]["exit_time"] == START + 63


def test_nominal_equal_known_exit_is_accepted():
    ticks = quotes()
    ticks[START + 359] = 97.
    rows, counts = a.replay_partition(ticks, [signal(), signal(START + 300)], START, START + 5000)
    assert rows[0]["exit_time"] == START + 360
    assert counts["completed"] == 2 and counts["overlap_skipped"] == 0


def test_planned_fold_purge_ignores_early_stop():
    ticks = quotes()
    ticks[START + 62] = 95.
    rows, counts = a.replay_partition(ticks, [signal()], START, START + 1800, overlapping=True)
    assert not rows and counts["purged"] == 1
    assert not a.planned_eligible([signal()], START, START + 1800)


def test_partition_end_clips_boundary_day_and_observed_days_include_zero_activity():
    next_day = a.v7.date_bounds(a.DATES[1])[0]
    days = a.observed_days(START + 300, next_day + 600)
    assert days == list(a.DATES[:2])
    assert a.day_bounds(START + 300, next_day + 600)[-1][2] == next_day + 600
    assert a.day_sums([], days) == [[0., 0, 0., 0.], [0., 0, 0., 0.]]
    assert not a.planned_eligible([signal(next_day)], START + 300, next_day + 600)


def test_point_summary_uses_pooled_paths_and_source_days_with_no_signals():
    rows = [{"signal_time": START, "censored": False, "net_R": 2.},
            {"signal_time": START + 300, "censored": False, "net_R": -.5},
            {"signal_time": START + 600, "censored": True, "net_R": None}]
    point = a.point_metrics(rows, {"issued": 4, "censored": 1, "missing_entry": 1}, list(a.DATES[:2]))
    assert point["completed"] == 2 and point["mean_net_R"] == .75 and point["profit_factor"] == 4.
    assert point["active_days"] == 1 and point["observed_day_clusters"] == 2
    assert point["day_sums_return_count_gain_loss"] == [[1.5, 2, 2., .5], [0., 0, 0., 0.]]


def test_all_wins_has_null_point_pf_instead_of_infinite_profit_claim():
    p = a.point_metrics([{"signal_time": START, "censored": False, "net_R": 1.}], {}, [a.DATES[0]])
    assert p["profit_factor"] is None and p["day_ratio_cluster_se"] is None


def test_paired_point_difference_pools_each_policy_before_subtracting():
    left = [{"signal_time": START, "censored": False, "net_R": 2.}]
    right = [{"signal_time": START, "censored": False, "net_R": -1.},
             {"signal_time": START + 300, "censored": False, "net_R": 1.}]
    saved = {**saved_inference(), "mean_difference_R": 2., "observed_days": [a.DATES[0]],
             "observed_day_clusters": 1, "bootstrap_repeats": 9999, "discovery_claim": False,
             "interpretation": "pooled_path_mean_difference_paired_observed_day_draws"}
    audit = a.Audit()
    a.paired_policy(saved, left, right, [a.DATES[0]], audit, "paired")
    assert not audit.failures


def test_disjoint_active_days_do_not_force_all_paired_draws_undefined():
    left = [{"signal_time": START, "censored": False, "net_R": 1.}]
    right = [{"signal_time": a.v7.date_bounds(a.DATES[1])[0], "censored": False, "net_R": 0.}]
    saved = {**saved_inference(5000), "mean_difference_R": 1., "observed_days": list(a.DATES[:2]),
             "observed_day_clusters": 2, "bootstrap_repeats": 9999, "discovery_claim": False,
             "interpretation": "pooled_path_mean_difference_paired_observed_day_draws"}
    audit = a.Audit()
    a.paired_policy(saved, left, right, list(a.DATES[:2]), audit, "paired")
    assert not audit.failures


@pytest.mark.parametrize("unknown,denominators", [(0, [1, 1]), (9999, [0, 0]), (150, [1, 0])])
def test_inference_metadata_count_and_null_policy(unknown, denominators):
    audit = a.Audit()
    found = a.inference_policy(saved_inference(unknown), [[0, d] for d in denominators], 1, audit, "mean")
    assert found is (unknown == 0) and not audit.failures


@pytest.mark.parametrize("fault", ["bool_count", "contradictory_flag", "nonfinite", "reversed"])
def test_inference_metadata_rejects_contradictions(fault):
    saved = saved_inference()
    if fault == "bool_count": saved["undefined_draws"] = False
    if fault == "contradictory_flag": saved["all_draws_defined"] = False
    if fault == "nonfinite": saved["ci95"] = [float("nan"), 1.]
    if fault == "reversed": saved["ci95"] = [1., -1.]
    audit = a.Audit()
    try:
        a.inference_policy(saved, [[0, 1]], 1, audit, "mean")
    except ValueError:
        pass
    assert audit.failures


@pytest.mark.parametrize("fault", ["too_few", "undefined", "missing", "censored", "invalid"])
def test_age_gate_blocks_whole_policy_rejection_on_insufficient_or_unknown_paths(fault):
    values = dict(completed=100, all_defined=True, missing=0, censored=0, invalid=0, interval=[-1., 0.])
    if fault == "too_few": values["completed"] = 99
    if fault == "undefined": values["all_defined"] = False
    if fault in ("missing", "censored", "invalid"): values[fault] = 1
    gate = a.age_gate(**values)
    assert gate == {"status": "INSUFFICIENT_EVIDENCE", "whole_eligible_policy_rejection_allowed": False}


def test_age_rejection_is_bounded_fixed_overlapping_label_mean_only():
    assert a.age_gate(100, True, 0, 0, 0, [-1., 0.])["status"] == "REJECT_POSITIVE_FIXED_OVERLAPPING_LABEL_MEAN"
    assert a.age_gate(100, True, 0, 0, 0, [-1., .01])["status"] == "NOT_REJECTED_POSITIVE_MEAN"
    with pytest.raises(ValueError): a.age_gate(True, True, 0, 0, 0, [-1., 0.])


def test_exact_ordered_native_matrix_and_target_hash():
    inputs = {START: {"x": 1., "y": 2.}, START + 300: {"x": 3., "y": 4.}}
    labels = [{"signal_time": START + 300, "net_R": -.5, "censored": False},
              {"signal_time": START, "net_R": .25, "censored": False}]
    matrix, target, ordered = a.matrix_hash(("x", "y"), inputs, labels)
    x = struct.pack("=dddd", 1., 2., 3., 4.)
    y = struct.pack("=dd", .25, -.5)
    t = struct.pack("=qq", START * 1_000_000_000, (START + 300) * 1_000_000_000)
    assert matrix == hashlib.sha256(b'["x", "y"]' + x + y + t).hexdigest()
    assert target == hashlib.sha256(t + y).hexdigest() and ordered[0]["signal_time"] == START
    assert a.matrix_hash(("y", "x"), inputs, labels)[0] != matrix
    assert a.matrix_hash(("x", "y"), inputs, [{**labels[0], "net_R": -.6}, labels[1]])[1] != target
    with pytest.raises(ValueError): a.matrix_hash(("x", "y"), inputs, [labels[0], labels[0]])


def training_fixture(n=1000):
    inputs, labels = {}, []
    for i in range(n):
        t = a.v7.date_bounds(a.DATES[i // 250])[0] + i % 250 * 300
        inputs[t] = {name: .125 for name in a.NAMES46}
        labels.append({"signal_time": t, "net_R": -.25, "censored": False})
    end = a.v7.date_bounds(a.DATES[(n - 1) // 250])[1]
    matrix, target, _ = a.matrix_hash(a.NAMES44, inputs, labels)
    model = {"family": "RIDGE44", "feature_names": list(a.NAMES44), "training_start": a.iso(START), "training_end": a.iso(end),
             "training_completed_labels": n, "training_censored_labels": 0, "training_invalid_labels": 0,
             "training_latest_issue": a.iso(labels[-1]["signal_time"]), "matrix_and_target_sha256": matrix,
             "common_timestamp_target_sha256": target, "training_labels_may_overlap": True,
             "statistically_independent_labels": False, "counts_toward_profit_sample_target": False, "label_purge_minutes": 31}
    if n < 1000:
        model.update(status="NOT TESTED", estimator=None, serialized_estimator_sha256=None)
    else:
        estimator = {"version": 1, "feature_names": list(a.NAMES44), "means": [.125] * 44,
                     "std": [1.] * 44, "coefs": [0.] * 44, "intercept": -.25, "threshold": 0.,
                     "fit_rows": n, "penalty": .1, "quantile": .75, "score_kind": "continuous_uncalibrated_score",
                     "scaler_fitted_on": "training_rows_only", "target_clipped": False, "features_clipped": False}
        model.update(status="TESTED", estimator=estimator, serialized_estimator_sha256=a.canonical_hash(estimator))
    return model, inputs, labels, end


@pytest.mark.parametrize("n", [999, 1000])
def test_training_floor_is_model_fit_only_and_hashes_match(n):
    model, inputs, labels, end = training_fixture(n)
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "fit")
    assert not audit.failures
    assert model["status"] == ("NOT TESTED" if n == 999 else "TESTED")
    assert model["counts_toward_profit_sample_target"] is False


@pytest.mark.parametrize("fault", ["target", "matrix", "scaler", "threshold", "feature_order", "promotion"])
def test_saved_fit_tampering_is_detected_without_refitting(fault):
    model, inputs, labels, end = training_fixture()
    if fault == "target": labels[0]["net_R"] = -.3
    if fault == "matrix": inputs[labels[0]["signal_time"]][a.NAMES44[0]] = 99.
    if fault == "scaler": model["estimator"]["means"][0] = 1.
    if fault == "threshold": model["estimator"]["threshold"] = .01
    if fault == "feature_order": model["feature_names"].reverse()
    if fault == "promotion": model["counts_toward_profit_sample_target"] = True
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "fit")
    assert audit.failures


def test_tiny_positive_scale_around_saved_mean_is_not_replaced_by_one():
    model, inputs, labels, end = training_fixture()
    saved_mean = .12500000000000003
    model["estimator"]["means"][0] = saved_mean
    model["estimator"]["std"][0] = abs(.125 - saved_mean)
    model["serialized_estimator_sha256"] = a.canonical_hash(model["estimator"])
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "fit")
    assert not audit.failures and model["estimator"]["std"][0] < 1e-14


def test_future_training_target_cannot_cross_fold_end():
    model, inputs, labels, end = training_fixture()
    with pytest.raises(ValueError):
        a.validate_model(model, inputs, labels, START, labels[-1]["signal_time"] + 100, a.Audit(), "fit")


def test_prediction_threshold_equality_and_positive_floor():
    inputs = {START: {"x": 1., "atr": 1., "close": 100., "prior_age_seconds": 600},
              START + 300: {"x": 0., "atr": 1., "close": 100., "prior_age_seconds": 601}}
    estimator = {"feature_names": ["x"], "means": [0.], "std": [1.], "coefs": [1.], "intercept": 0., "threshold": 1.}
    model = {"status": "TESTED", "estimator": estimator}
    issued = a.expected_signals(inputs, "CRASH600", "DRIFT", "RIDGE44", model)
    assert len(issued) == 1 and issued[0]["score"] == 1. and issued[0]["side"] == 1
    estimator["threshold"] = 0.
    assert len(a.expected_signals(inputs, "CRASH600", "DRIFT", "RIDGE44", model)) == 1
    estimator["std"] = [0.]
    with pytest.raises(ValueError): a.prediction(inputs[START], estimator)


def test_near_threshold_issuance_is_strict_without_tolerance_inflation():
    import math
    threshold = 1.
    inputs = {START: {"x": math.nextafter(threshold, 0.), "atr": 1., "close": 100., "prior_age_seconds": 1},
              START + 300: {"x": threshold, "atr": 1., "close": 100., "prior_age_seconds": 301},
              START + 600: {"x": math.nextafter(threshold, 2.), "atr": 1., "close": 100., "prior_age_seconds": 601}}
    model = {"status": "TESTED", "estimator": {"feature_names": ["x"], "means": [0.], "std": [1.],
             "coefs": [1.], "intercept": 0., "threshold": threshold}}
    signals = a.expected_signals(inputs, "BOOM600", "SPIKE", "RIDGE44", model)
    assert [s["signal_time"] for s in signals] == [START + 300, START + 600]


def input_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "source_path", lambda path: Path(path))
    minute = {t: (100., 101., 99., 100.) for t in range(START - 14400, START + 900, 60)}
    tail, raws = {}, []
    for i in range(3):
        t, age = START + i * 300, 1 + i * 300
        tail[t] = a.v8.Row(t, True, 0., False, age, .001, "lt_N" if age < 600 else "ge_N")
        row = {"m5_open_time": a.iso(t - 300), "signal_time": a.iso(t), **dict.fromkeys(a.NAMES44, .125),
               "tail_age_log1p": __import__("math").log1p(age / 600), "prior_tail_mark": .001,
               "atr": 2., "close": 100., "prior_age_seconds": age,
               **dict.fromkeys(("original_feature_valid", "original44_all_finite", "multiframe_feature_valid",
                                "tail_feature_valid", "common_available", "feature_valid"), True)}
        for name, seconds in (("h4", 14400), ("h1", 3600), ("m15", 900), ("m1", 60)):
            row[name + "_closed_at"], row[name + "_row_valid"] = a.iso(t // seconds * seconds), True
        raws.append(row)
    path = csv_file(tmp_path / "inputs.csv", list(raws[0]), raws)
    value = {"inputs": {"path": str(path), "sha256": a.digest(path), "rows": 3, "local_ignored_artifact": True},
             "feature_end_exclusive": a.iso(START + 900), "suffix_calculations": False, "detector_refit": False,
             "clock": "every closed UTC M5", "eligible_common_clock_rows": 3, "canonical_boundary_day_quotes_decoded": True,
             "last_m1_close": a.iso(START + 900), "tick_last_used": a.iso(START + 899)}
    return minute, tail, raws, value, path


def test_streamed_saved_common_matrix_has_causal_atr_and_exact_prior_state(tmp_path, monkeypatch):
    minute, tail, _, value, _ = input_fixture(tmp_path, monkeypatch)
    audit = a.Audit()
    rows = a.read_inputs(value, START + 900, minute, tail, audit, "prefix")
    assert not audit.failures and len(rows) == 3 and rows[START]["atr"] == 2.


@pytest.mark.parametrize("fault", ["current_event_age", "future_context", "mask", "grid_gap", "changed_atr"])
def test_streamed_inputs_detect_causal_and_availability_corruption(tmp_path, monkeypatch, fault):
    minute, tail, raws, value, path = input_fixture(tmp_path, monkeypatch)
    if fault == "current_event_age": raws[0]["prior_age_seconds"] = 0
    if fault == "future_context": raws[0]["h4_closed_at"] = a.iso(START + 14400)
    if fault == "mask": raws[0]["common_available"] = False
    if fault == "grid_gap": raws.pop(1)
    if fault == "changed_atr": raws[0]["atr"] = 9.
    csv_file(path, list(raws[0]), raws); value["inputs"]["sha256"] = a.digest(path)
    audit = a.Audit()
    try: a.read_inputs(value, START + 900, minute, tail, audit, "prefix")
    except ValueError: pass
    assert audit.failures


def test_unknown_tail_row_stays_unavailable_in_both_families_and_clock(tmp_path, monkeypatch):
    minute, tail, raws, value, path = input_fixture(tmp_path, monkeypatch)
    tail.pop(START + 300)
    for key in ("prior_age_seconds", "tail_age_log1p", "prior_tail_mark"): raws[1][key] = ""
    for key in ("tail_feature_valid", "common_available", "feature_valid"): raws[1][key] = False
    csv_file(path, list(raws[0]), raws); value["inputs"]["sha256"] = a.digest(path)
    value["eligible_common_clock_rows"] = 2
    audit = a.Audit()
    rows = a.read_inputs(value, START + 900, minute, tail, audit, "prefix")
    assert not audit.failures and START + 300 not in rows
    assert len(a.expected_signals(rows, "BOOM600", "SPIKE")) == 2


def test_full_grid_fingerprint_detects_invalid_to_valid_prefix_change(tmp_path, monkeypatch):
    minute, tail, raws, value, path = input_fixture(tmp_path, monkeypatch)
    fingerprints = {}
    raws[1]["original_feature_valid"] = False
    for key in ("multiframe_feature_valid", "common_available", "feature_valid"): raws[1][key] = False
    csv_file(path, list(raws[0]), raws); value["inputs"]["sha256"] = a.digest(path)
    value["eligible_common_clock_rows"] = 2
    first = a.Audit()
    a.read_inputs(value, START + 900, minute, tail, first, "earlier", fingerprints)
    assert not first.failures
    for key in ("original_feature_valid", "multiframe_feature_valid", "common_available", "feature_valid"): raws[1][key] = True
    csv_file(path, list(raws[0]), raws); value["inputs"]["sha256"] = a.digest(path)
    value["eligible_common_clock_rows"] = 3
    second = a.Audit()
    a.read_inputs(value, START + 900, minute, tail, second, "later", fingerprints)
    assert any("full_grid_prefix_consistency" in error["check"] for error in second.failures)


def test_detector_inadequate_symbols_have_no_models_or_candidates():
    development = {"status": "NOT TESTED", "reason": "inadequate_fixed_round8_detector",
                   "final_models": {}, "candidates": {}, "selected_candidate": None}
    final = {"status": "NOT TESTED", "reason": "inadequate_fixed_round8_detector", "models": {}}
    audit = a.Audit()
    a.audit_untested_symbol(development, final, audit, "unknown")
    assert not audit.failures
    final["models"] = {"SPIKE": {}}
    a.audit_untested_symbol(development, final, audit, "unsafe")
    assert audit.failures


def test_untested_symbols_metrics_csv_remain_untested(tmp_path):
    path = csv_file(tmp_path / "metrics.csv", ["symbol", "status"],
                    [{"symbol": s, "status": "NOT TESTED"} for s in a.SYMBOLS])
    value = {"symbols": {s: {"status": "NOT TESTED"} for s in a.SYMBOLS}}
    audit = a.Audit()
    a.verify_metrics_csv(path, value, False, audit)
    assert not audit.failures


def test_mixed_status_metric_table_accepts_integral_float_csv_counts(tmp_path):
    metrics = dict(completed=73, censored=0, missing_entry=0, profit_factor=1.2,
                   mean_net_R=.05, active_days=4, observed_day_clusters=4)
    rows = []
    for mode in a.MODES:
        for family in ("CLOCK", *a.FAMILIES):
            rows.append({"symbol": a.SYMBOLS[0], "mode": mode, "family": family, "status": "TESTED",
                         **{k: str(float(v)) if type(v) is int else v for k, v in metrics.items()},
                         "historical_gate": False, "live_candidate": False})
    rows.append(dict.fromkeys(rows[0], ""))
    rows[-1].update(symbol=a.SYMBOLS[1], status="NOT TESTED")
    path = csv_file(tmp_path / "metrics.csv", list(rows[0]), rows)
    value = {"symbols": {a.SYMBOLS[0]: {"status": "TESTED", "models": {
        mode: {"reports": {family: {"status": "TESTED", "metrics": metrics}
                          for family in ("CLOCK", *a.FAMILIES)}} for mode in a.MODES}},
        a.SYMBOLS[1]: {"status": "NOT TESTED"}}}
    audit = a.Audit()
    a.verify_metrics_csv(path, value, False, audit)
    assert not audit.failures
    with pytest.raises(ValueError): a.csv_integer("73.5")
    with pytest.raises(ValueError): a.csv_integer("nan")


def test_nonfinite_mismatch_can_be_preserved_in_json_failure_artifact():
    audit = a.Audit()
    audit.equal("bad", float("nan"), 1.)
    assert audit.failures and json.dumps(audit.failures, allow_nan=False)


def test_missing_m1_slot_counts_in_grid_but_cannot_supply_context_or_atr():
    minute = {t: (100., 101., 99., 100.) for t in range(START - 4500, START + 300, 60)}
    missing = START - 60
    del minute[missing]
    before = a.m1_prefix_inventory(minute, missing + 59)
    at_close = a.m1_prefix_inventory(minute, missing + 60)
    assert before == {"closed_grid_slots": 74, "available_closed_candles": 74,
                      "unavailable_closed_slots": 0, "last_grid_close": missing}
    assert at_close == {"closed_grid_slots": 75, "available_closed_candles": 74,
                        "unavailable_closed_slots": 1, "last_grid_close": START}
    assert missing not in minute
    with pytest.raises(KeyError): a.v7.causal_atr(minute, START)
    assert not all(t in minute for t in range(START - 300, START, 60))


def test_closed_grid_inventory_clips_seconds_and_preserves_terminal_unknown_slot():
    minute = {START: (100., 101., 99., 100.), START + 120: (100., 101., 99., 100.)}
    assert a.m1_prefix_inventory(minute, START + 119) == {
        "closed_grid_slots": 1, "available_closed_candles": 1,
        "unavailable_closed_slots": 0, "last_grid_close": START + 60}
    assert a.m1_prefix_inventory(minute, START + 120) == {
        "closed_grid_slots": 2, "available_closed_candles": 1,
        "unavailable_closed_slots": 1, "last_grid_close": START + 120}
    assert a.m1_prefix_inventory(minute, START + 600)["closed_grid_slots"] == 3
    assert a.m1_prefix_inventory(minute, START - 60)["last_grid_close"] is None
    with pytest.raises(ValueError): a.m1_prefix_inventory(minute, True)


def test_pooled_comparison_requires_partial_wrapper_for_failed_fold_fit():
    days = [a.DATES[0]]
    saved = {**saved_inference(9999), "mean_difference_R": None, "observed_days": days,
             "observed_day_clusters": 1, "bootstrap_repeats": 9999, "discovery_claim": False,
             "interpretation": "pooled_path_mean_difference_paired_observed_day_draws"}
    comparisons = {key: {"status": "NOT TESTED", "partial_descriptive_only": deepcopy(saved)}
                   for key in ("RIDGE46_minus_RIDGE44", "RIDGE44_minus_CLOCK", "RIDGE46_minus_CLOCK")}
    audit = a.Audit()
    a.audit_comparisons(comparisons, dict.fromkeys(("CLOCK", *a.FAMILIES), []), days, audit, "pooled",
                       {"CLOCK": "TESTED", "RIDGE44": "PARTIALLY TESTED", "RIDGE46": "NOT TESTED"})
    assert not audit.failures
    comparisons["RIDGE44_minus_CLOCK"] = {**saved, "status": "TESTED"}
    a.audit_comparisons(comparisons, dict.fromkeys(("CLOCK", *a.FAMILIES), []), days, audit, "bad",
                       {"CLOCK": "TESTED", "RIDGE44": "PARTIALLY TESTED", "RIDGE46": "NOT TESTED"})
    assert audit.failures


def test_audit_refuses_bool_counter_alias_and_true_live_flag():
    audit = a.Audit(); audit.equal("counter", True, 1)
    assert audit.failures
    with pytest.raises(ValueError): audit.safety({"LIVE_TRADING": True}, "unsafe")


@pytest.mark.parametrize("flag", a.FLAGS)
def test_cli_live_environment_is_blocked_before_any_artifact_read(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(sys, "argv", ["verify_spike_tick_tail_signal.py"])
    with pytest.raises(ValueError): a.main()
