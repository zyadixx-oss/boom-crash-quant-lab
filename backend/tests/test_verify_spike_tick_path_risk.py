"""Synthetic-only challenges for the independent fixed native-path audit."""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import verify_spike_tick_path_risk as a

START = a.v7.date_bounds(a.DATES[0])[0]


def price_stream(increments, side=1, start=START):
    total, quotes = 0., [100.]
    for increment in increments:
        total += increment * side
        quotes.append(100. * math.exp(total))
    return list(range(start, start + len(quotes))), quotes


def signal(t=START, side=1):
    return {"signal_time": t, "atr": 1., "side": side, "variant": "CLOCK_SPIKE",
            "signal_close": 100., a.PATH_FEATURE: .5}


def tick_quotes(end=START + 5000):
    return dict.fromkeys(range(START, end), 100.)


def csv_file(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("count", [1, 2, 600, 601, 602])
def test_independent_feature_requires_exactly601_prior_quotes(count):
    times, quotes = price_stream([0.] * (count - 1))
    observed = a.prior_adverse_window(times, quotes, .001, 1)
    assert list(observed) == [t + 1 for t in times]
    assert sum(v is not None for v in observed.values()) == max(0, count - 600)
    assert all(v == 0. for v in observed.values() if v is not None)
    if count >= 601:
        assert observed[START + 600] is None
        assert observed[START + 601] == 0.


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("native_return,expected", [(.002, 0.), (-.002, 4.)])
def test_hand_calculated_native_favorable_and_adverse_semivariance(side, native_return, expected):
    times, quotes = price_stream([native_return] * 600, side)
    risk = a.prior_adverse_window(times, quotes, .001, side)
    assert risk[times[-1] + 1] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("side", [1, -1])
def test_mixed600_returns_and_scale_are_fixed_without_strategy_side_selection(side):
    increments = [-.001, .002, -.003, 0.] * 150
    times, quotes = price_stream(increments, side)
    risk = a.prior_adverse_window(times, quotes, .002, side)
    expected = ((.001 / .002) ** 2 + (.003 / .002) ** 2) / 4
    assert risk[times[-1] + 1] == pytest.approx(expected, abs=1e-10)
    larger_scale = a.prior_adverse_window(times, quotes, .004, side)
    assert larger_scale[times[-1] + 1] == pytest.approx(expected / 4, abs=1e-10)
    assert times == list(range(START, START + 601))


def test_first_adverse_increment_exits_at_exact600_increment_boundary():
    times, quotes = price_stream([-.01] + [0.] * 601)
    risk = a.prior_adverse_window(times, quotes, .001, 1)
    assert risk[times[600] + 1] == pytest.approx(100 / 600, abs=1e-10)
    assert risk[times[601] + 1] == 0.
    assert risk[times[602] + 1] == 0.


def test_current_future_quotes_and_missing_current_do_not_rewrite_prior_feature():
    times, quotes = price_stream([-.001, .002] * 700)
    target = START + 900
    original = a.prior_adverse_window(times, quotes, .001, 1)
    changed = [q if t < target else 1000. + t - target for t, q in zip(times, quotes, strict=True)]
    mutated = a.prior_adverse_window(times, changed, .001, 1)
    assert {t: v for t, v in original.items() if t <= target} == {t: v for t, v in mutated.items() if t <= target}
    prefix = [(t, q) for t, q in zip(times, quotes, strict=True) if t < target]
    assert a.prior_adverse_window([t for t, _ in prefix], [q for _, q in prefix], .001, 1) == {
        t: v for t, v in original.items() if t <= target}
    without_current = [(t, q) for t, q in zip(times, quotes, strict=True) if t != target]
    missing = a.prior_adverse_window([t for t, _ in without_current], [q for _, q in without_current], .001, 1)
    assert missing[target] == original[target]
    assert target + 1 not in missing
    assert missing[target + 2] is None


@pytest.mark.parametrize("gap_seconds", [1, 2, 30, 86400])
def test_missing_second_requires_new601_contiguous_quotes_without_stale_carry(gap_seconds):
    times, quotes = price_stream([-.001] * 1400)
    times = [t if i < 700 else t + gap_seconds for i, t in enumerate(times)]
    first_after = times[700]
    risk = a.prior_adverse_window(times, quotes, .001, 1)
    assert risk[times[699] + 1] == pytest.approx(1., abs=1e-10)
    assert all(risk[first_after + i + 1] is None for i in range(600))
    assert risk[first_after + 601] == pytest.approx(1., abs=1e-10)
    assert times[699] + 2 not in risk


def test_contiguous_midnight_and_clock_only_view_keep_exact_prior_history():
    times, quotes = price_stream([-.001] * 1000, start=START - 600)
    full = a.prior_adverse_window(times, quotes, .001, 1)
    assert full[START + 1] == pytest.approx(1., abs=1e-10)
    clock = a.prior_adverse_window(times, quotes, .001, 1, clock_only=True)
    assert clock == {t: v for t, v in full.items() if t % 300 == 0}
    assert START in clock and clock[START] is None
    assert clock[START + 300] == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize("fault", ["duplicate", "reversed", "fractional", "bool_time", "unequal", "zero_quote",
                                   "nan_quote", "infinite_quote", "bool_quote", "zero_scale", "nan_scale",
                                   "bool_scale", "wrong_side", "bool_side", "wrong_window", "bool_window"])
def test_prior_window_refuses_invalid_arrays_parameters_and_changed_window(fault):
    times, quotes = price_stream([0.] * 600)
    scale, side, window = .001, 1, 600
    if fault == "duplicate": times[10] = times[9]
    if fault == "reversed": times.reverse()
    if fault == "fractional": times[10] += .5
    if fault == "bool_time": times[10] = True
    if fault == "unequal": quotes.pop()
    if fault == "zero_quote": quotes[10] = 0.
    if fault == "nan_quote": quotes[10] = math.nan
    if fault == "infinite_quote": quotes[10] = math.inf
    if fault == "bool_quote": quotes[10] = True
    if fault == "zero_scale": scale = 0.
    if fault == "nan_scale": scale = math.nan
    if fault == "bool_scale": scale = True
    if fault == "wrong_side": side = 2
    if fault == "bool_side": side = True
    if fault == "wrong_window": window = 599
    if fault == "bool_window": window = True
    with pytest.raises((TypeError, ValueError)):
        a.prior_adverse_window(times, quotes, scale, side, window)


def test_extreme_positive_quotes_use_finite_log_fallback_without_squared_scale_underflow():
    times = list(range(START, START + 601))
    quotes = ([1e300, 1e-300] * 301)[:601]
    risk = a.prior_adverse_window(times, quotes, 1., 1)
    expected = (math.log(1e-300) - math.log(1e300)) ** 2 / 2
    assert risk[times[-1] + 1] == pytest.approx(expected)
    with pytest.raises(ValueError):
        a.prior_adverse_window(times, quotes, 1e-300, 1)


def test_finite_individual_squares_cannot_escape_as_overflowing_rolling_sum():
    times = list(range(START, START + 601))
    quotes = ([100., 99.] * 301)[:601]
    scale = 3e-156
    assert math.isfinite((math.log(.99) / scale) ** 2)
    with pytest.raises(ValueError, match="rolling arithmetic.*finite"):
        a.prior_adverse_window(times, quotes, scale, 1)


def test_native_feature_stays_same_for_spike_and_drift_while_issuance_sides_change():
    inputs = {START: {a.PATH_FEATURE: .5, "atr": 1., "close": 100.}}
    estimator = {"feature_names": [a.PATH_FEATURE], "means": [0.], "std": [1.], "coefs": [1.],
                 "intercept": 0., "threshold": .5}
    model = {"status": "TESTED", "estimator": estimator}
    for symbol, native_side in (("BOOM600", 1), ("CRASH600", -1)):
        spike = a.expected_signals(inputs, symbol, "SPIKE", "RIDGE45", model)
        drift = a.expected_signals(inputs, symbol, "DRIFT", "RIDGE45", model)
        assert len(spike) == len(drift) == 1
        assert spike[0][a.PATH_FEATURE] == drift[0][a.PATH_FEATURE] == .5
        assert spike[0]["score"] == drift[0]["score"] == .5
        assert spike[0]["side"] == native_side == -drift[0]["side"]


def test_m5_clock_signal_csv_preserves_zero_risk_and_omits_clock_score(tmp_path):
    raw = {**signal(START + 300), "signal_time": a.v8.iso(START + 300), a.PATH_FEATURE: 0., "score": ""}
    found = a.read_signals(csv_file(tmp_path / "signals.csv", list(raw), [raw]))
    assert found == [{**signal(START + 300), a.PATH_FEATURE: 0.}]
    assert "score" not in found[0]


@pytest.mark.parametrize("fault", ["duplicate", "reversed", "off_clock", "naive", "fractional_side", "negative_risk", "nonfinite_risk"])
def test_saved_issuance_uniqueness_clock_side_and_risk_are_strict(tmp_path, fault):
    raws = [{**signal(t), "signal_time": a.iso(t), "score": ""} for t in (START, START + 300)]
    if fault == "duplicate": raws[1]["signal_time"] = raws[0]["signal_time"]
    if fault == "reversed": raws.reverse()
    if fault == "off_clock": raws[0]["signal_time"] = a.iso(START + 60)
    if fault == "naive": raws[0]["signal_time"] = "2026-04-11T00:00:00"
    if fault == "fractional_side": raws[0]["side"] = "1.5"
    if fault == "negative_risk": raws[0][a.PATH_FEATURE] = -1e-12
    if fault == "nonfinite_risk": raws[0][a.PATH_FEATURE] = math.inf
    with pytest.raises(ValueError):
        a.read_signals(csv_file(tmp_path / "invalid_signals.csv", list(raws[0]), raws))


def test_saved_native45_schema_is44_plus_one_path_feature_without_tail_age_or_mark():
    assert tuple(a.NAMES45) == (*a.NAMES44, a.PATH_FEATURE)
    assert len(a.NAMES44) == 44 and len(a.NAMES45) == 45
    assert "tail_age_log1p" not in a.NAMES45 and "prior_tail_mark" not in a.NAMES45


def test_singleton_training_labels_and_one_open_strategy_do_not_share_occupancy():
    issued = [signal(), signal(START + 300)]
    individual, labels = a.v9.replay_partition(tick_quotes(), issued, START, START + 5000, overlapping=True)
    joint, strategy = a.v9.replay_partition(tick_quotes(), issued, START, START + 5000)
    assert len(individual) == labels["completed"] == 2
    assert len(joint) == strategy["completed"] == strategy["overlap_skipped"] == 1
    assert all(row["net_R"] == -.05 for row in individual)


def test_planned31minute_purge_is_preserved_for_both_singleton_and_joint_paths():
    ticks = tick_quotes()
    ticks[START + 62] = 95.
    for overlapping in (False, True):
        rows, counts = a.v9.replay_partition(ticks, [signal()], START, START + 1800, overlapping=overlapping)
        assert not rows and counts["purged"] == 1


def test_missing_entry_reserves_one_open_occupancy_but_does_not_create_zero_target():
    ticks = tick_quotes()
    del ticks[START + 61]
    issued = [signal(), signal(START + 300)]
    individual, target = a.v9.replay_partition(ticks, issued, START, START + 5000, overlapping=True)
    joint, strategy = a.v9.replay_partition(ticks, issued, START, START + 5000)
    assert len(individual) == target["completed"] == target["missing_entry"] == 1
    assert not joint and strategy["missing_entry"] == strategy["overlap_skipped"] == 1


def input_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "source_path", lambda value: Path(value))
    minute = {t: (100., 101., 99., 100.) for t in range(START - 14400, START + 900, 60)}
    risk, raws = {}, []
    for i in range(3):
        t = START + i * 300
        risk[t] = i / 2
        row = {"m5_open_time": a.iso(t - 300), "signal_time": a.iso(t),
               **dict.fromkeys(a.NAMES44, .125), a.PATH_FEATURE: risk[t], "atr": 2., "close": 100.,
               **dict.fromkeys(("original_feature_valid", "original44_all_finite", "multiframe_feature_valid",
                               "tick_path_feature_valid", "common_available", "feature_valid"), True)}
        for name, seconds in (("h4", 14400), ("h1", 3600), ("m15", 900), ("m1", 60)):
            row[name + "_closed_at"], row[name + "_row_valid"] = a.iso(t // seconds * seconds), True
        raws.append(row)
    path = csv_file(tmp_path / "inputs.csv", list(raws[0]), raws)
    inventory = a.v9.m1_prefix_inventory(minute, START + 900)
    value = {"inputs": {"path": str(path), "sha256": a.digest(path), "rows": 3, "local_ignored_artifact": True},
             "feature_end_exclusive": a.iso(START + 900), "suffix_calculations": False, "scale_refit": False,
             "age_or_mark_availability_used": False, "clock": "every closed UTC M5",
             "eligible_common_clock_rows": 3, "canonical_boundary_day_quotes_decoded": True,
             "m1_closed_rows_used": inventory["closed_grid_slots"],
             "m1_available_closed_rows_used": inventory["available_closed_candles"],
             "m1_unknown_closed_rows_retained": inventory["unavailable_closed_slots"],
             "last_m1_close": a.iso(inventory["last_grid_close"])}
    return minute, risk, raws, value, path


def test_common45_matrix_uses_causal_atr_exact_risk_and_original_grid(tmp_path, monkeypatch):
    minute, risk, _, value, _ = input_fixture(tmp_path, monkeypatch)
    audit = a.Audit()
    rows = a.read_inputs(value, START + 900, minute, risk, audit, "prefix")
    assert not audit.failures and len(rows) == 3
    assert rows[START][a.PATH_FEATURE] == 0. and rows[START]["atr"] == 2.
    assert len(a.expected_signals(rows, "BOOM600", "SPIKE")) == 3


@pytest.mark.parametrize("missing", ["risk", "original_flag", "original_number"])
def test_unavailable_risk_or_original44_row_is_unavailable_to_both_models_and_clock(tmp_path, monkeypatch, missing):
    minute, risk, raws, value, path = input_fixture(tmp_path, monkeypatch)
    row = raws[1]
    if missing == "risk":
        risk.pop(START + 300)
        row[a.PATH_FEATURE], row["tick_path_feature_valid"] = "", False
    else:
        row["original_feature_valid"], row["multiframe_feature_valid"] = False, False
        if missing == "original_number":
            row[a.NAMES44[0]], row["original44_all_finite"] = "", False
    row["common_available"], row["feature_valid"] = False, False
    csv_file(path, list(raws[0]), raws)
    value["inputs"]["sha256"], value["eligible_common_clock_rows"] = a.digest(path), 2
    audit = a.Audit()
    rows = a.read_inputs(value, START + 900, minute, risk, audit, "prefix")
    assert not audit.failures and START + 300 not in rows
    assert len(a.expected_signals(rows, "CRASH600", "DRIFT")) == 2


@pytest.mark.parametrize("fault", ["changed_risk", "future_context", "common_mask", "grid_gap", "changed_atr", "age_column"])
def test_saved_inputs_refuse_changed_path_context_mask_clock_or_extra_age(tmp_path, monkeypatch, fault):
    minute, risk, raws, value, path = input_fixture(tmp_path, monkeypatch)
    if fault == "changed_risk": raws[0][a.PATH_FEATURE] = 99.
    if fault == "future_context": raws[0]["h4_closed_at"] = a.iso(START + 14400)
    if fault == "common_mask": raws[0]["common_available"] = False
    if fault == "grid_gap": raws.pop(1)
    if fault == "changed_atr": raws[0]["atr"] = 9.
    if fault == "age_column":
        for row in raws: row["tail_age_log1p"] = 0.
    csv_file(path, list(raws[0]), raws)
    value["inputs"]["sha256"] = a.digest(path)
    audit = a.Audit()
    try:
        a.read_inputs(value, START + 900, minute, risk, audit, "fault")
    except ValueError:
        pass
    assert audit.failures


def test_full_grid_fingerprint_detects_prefix_availability_change_without_regenerating44(tmp_path, monkeypatch):
    minute, risk, raws, value, path = input_fixture(tmp_path, monkeypatch)
    fingerprints = {}
    for name in ("original_feature_valid", "multiframe_feature_valid", "common_available", "feature_valid"):
        raws[1][name] = False
    csv_file(path, list(raws[0]), raws)
    value["inputs"]["sha256"], value["eligible_common_clock_rows"] = a.digest(path), 2
    first = a.Audit()
    a.read_inputs(value, START + 900, minute, risk, first, "first", fingerprints)
    assert not first.failures
    for name in ("original_feature_valid", "multiframe_feature_valid", "common_available", "feature_valid"):
        raws[1][name] = True
    csv_file(path, list(raws[0]), raws)
    value["inputs"]["sha256"], value["eligible_common_clock_rows"] = a.digest(path), 3
    second = a.Audit()
    a.read_inputs(value, START + 900, minute, risk, second, "second", fingerprints)
    assert any("full_grid_prefix_consistency" in failure["check"] for failure in second.failures)


def test_missing_source_minute_remains_unknown_grid_slot_and_cannot_be_rescued_by_known_risk(tmp_path, monkeypatch):
    minute, risk, raws, value, path = input_fixture(tmp_path, monkeypatch)
    del minute[START - 60]
    for row in raws:
        t = a.stamp(row["signal_time"])
        for name in ("original_feature_valid", "multiframe_feature_valid", "common_available", "feature_valid"):
            row[name] = False
        for name, seconds in (("h4", 14400), ("h1", 3600), ("m15", 900), ("m1", 60)):
            close = t // seconds * seconds
            row[name + "_row_valid"] = all(s in minute for s in range(close - seconds, close, 60))
    csv_file(path, list(raws[0]), raws)
    value["inputs"]["sha256"], value["eligible_common_clock_rows"] = a.digest(path), 0
    inventory = a.v9.m1_prefix_inventory(minute, START + 900)
    value["m1_available_closed_rows_used"] = inventory["available_closed_candles"]
    value["m1_unknown_closed_rows_retained"] = inventory["unavailable_closed_slots"]
    audit = a.Audit()
    rows = a.read_inputs(value, START + 900, minute, risk, audit, "missing")
    assert not audit.failures and not rows
    assert value["m1_unknown_closed_rows_retained"] == 1
    assert value["m1_closed_rows_used"] == value["m1_available_closed_rows_used"] + 1
    assert all(row["tick_path_feature_valid"] for row in raws)


def native_hashes(names, inputs, labels):
    known = sorted((r for r in labels if not r["censored"] and r["net_R"] is not None), key=lambda r: r["signal_time"])
    x = b"".join(struct.pack("=d", inputs[r["signal_time"]][name]) for r in known for name in names)
    y = b"".join(struct.pack("=d", r["net_R"]) for r in known)
    clock = b"".join(struct.pack("=q", r["signal_time"] * 1_000_000_000) for r in known)
    return hashlib.sha256(json.dumps(list(names)).encode() + x + y + clock).hexdigest(), hashlib.sha256(clock + y).hexdigest()


def training_fixture(tmp_path, monkeypatch, n=1000, family="RIDGE45", target=-.25):
    monkeypatch.setattr(a, "source_path", lambda value: Path(value))
    inputs, labels = {}, []
    for i in range(n):
        t = a.v7.date_bounds(a.DATES[i // 250])[0] + i % 250 * 300
        inputs[t] = dict.fromkeys(a.NAMES45, .125)
        labels.append({"signal_time": t, "net_R": target, "censored": False})
    end = a.v7.date_bounds(a.DATES[(n - 1) // 250])[1]
    names = a.NAMES44 if family == "RIDGE44" else a.NAMES45
    matrix, common = native_hashes(names, inputs, labels)
    rows = [{"signal_time": a.iso(r["signal_time"]), **inputs[r["signal_time"]], "net_R": r["net_R"]} for r in labels]
    path = csv_file(tmp_path / "matrix.csv", ["signal_time", *a.NAMES45, "net_R"], rows)
    model = {"family": family, "feature_names": list(names), "training_start": a.iso(START), "training_end": a.iso(end),
             "training_completed_labels": n, "training_censored_labels": 0, "training_invalid_labels": 0,
             "training_latest_issue": a.iso(labels[-1]["signal_time"]), "matrix_and_target_sha256": matrix,
             "common_timestamp_target_sha256": common, "training_labels_may_overlap": True,
             "statistically_independent_labels": False, "counts_toward_profit_sample_target": False,
             "label_purge_minutes": 31, "training_matrix": {"path": str(path), "sha256": a.digest(path),
                                                           "rows": n, "local_ignored_artifact": True}}
    if n < 1000:
        model.update(status="NOT TESTED", estimator=None, serialized_estimator_sha256=None)
    else:
        estimator = {"version": 1, "feature_names": list(names), "means": [.125] * len(names),
                     "std": [1.] * len(names), "coefs": [0.] * len(names), "intercept": target,
                     "threshold": max(0., target), "fit_rows": n, "penalty": .1, "quantile": .75,
                     "score_kind": "continuous_uncalibrated_score", "scaler_fitted_on": "training_rows_only",
                     "features_clipped": False, "target_clipped": False}
        model.update(status="TESTED", estimator=estimator, serialized_estimator_sha256=a.canonical_hash(estimator))
    return model, inputs, labels, end, path


@pytest.mark.parametrize("family", ["RIDGE44", "RIDGE45"])
@pytest.mark.parametrize("n", [999, 1000])
def test_exact_native_training_identity_and1000_floor_do_not_count_labels_as_strategy_paths(tmp_path, monkeypatch, family, n):
    model, inputs, labels, end, _ = training_fixture(tmp_path, monkeypatch, n, family)
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "fit")
    assert not audit.failures
    assert model["status"] == ("TESTED" if n == 1000 else "NOT TESTED")
    assert model["counts_toward_profit_sample_target"] is False


@pytest.mark.parametrize("fault", ["target", "matrix", "scaler", "threshold", "feature_order", "promotion"])
def test_saved45_training_model_tampering_is_detected_without_fitting(tmp_path, monkeypatch, fault):
    model, inputs, labels, end, _ = training_fixture(tmp_path, monkeypatch)
    if fault == "target": labels[0]["net_R"] = -.3
    if fault == "matrix": inputs[labels[0]["signal_time"]][a.PATH_FEATURE] = 99.
    if fault == "scaler": model["estimator"]["means"][-1] = 1.
    if fault == "threshold": model["estimator"]["threshold"] = .01
    if fault == "feature_order": model["feature_names"].reverse()
    if fault == "promotion": model["counts_toward_profit_sample_target"] = True
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "tamper")
    assert audit.failures


def test_tiny_positive_saved_scale_is_preserved_and_coefficient_dimensions_are_strict(tmp_path, monkeypatch):
    model, inputs, labels, end, _ = training_fixture(tmp_path, monkeypatch)
    saved_mean = .12500000000000003
    model["estimator"]["means"][-1], model["estimator"]["std"][-1] = saved_mean, abs(.125 - saved_mean)
    model["serialized_estimator_sha256"] = a.canonical_hash(model["estimator"])
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "tiny_scale")
    assert not audit.failures
    broken = deepcopy(model["estimator"])
    broken["coefs"].pop()
    with pytest.raises(ValueError): a.prediction(inputs[labels[0]["signal_time"]], broken)
    broken = deepcopy(model["estimator"])
    broken["std"][-1] = 0.
    with pytest.raises(ValueError): a.prediction(inputs[labels[0]["signal_time"]], broken)


def test_saved_training_matrix_requires_unique_ordered_timestamp_target_and45_column_identity(tmp_path, monkeypatch):
    model, inputs, labels, _, path = training_fixture(tmp_path, monkeypatch, n=999)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][a.PATH_FEATURE] = "9."
    csv_file(path, ["signal_time", *a.NAMES45, "net_R"], rows)
    model["training_matrix"]["sha256"] = a.digest(path)
    audit = a.Audit()
    a.audit_training_matrix(model["training_matrix"], inputs, labels, audit, "matrix")
    assert audit.failures
    rows.reverse()
    csv_file(path, ["signal_time", *a.NAMES45, "net_R"], rows)
    model["training_matrix"]["sha256"] = a.digest(path)
    audit = a.Audit()
    a.audit_training_matrix(model["training_matrix"], inputs, labels, audit, "order")
    assert audit.failures


def test_training_guard_uses_original_partition_end_even_for_early_completed_labels(tmp_path, monkeypatch):
    model, inputs, labels, _, _ = training_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        a.validate_model(model, inputs, labels, START, labels[-1]["signal_time"] + 100, a.Audit(), "future")


def test_saved45_training_q75_is_linear_interpolation_of_frozen_scaled_scores(tmp_path, monkeypatch):
    model, inputs, labels, end, path = training_fixture(tmp_path, monkeypatch, target=.25)
    name = a.NAMES44[0]
    for i, row in enumerate(labels):
        inputs[row["signal_time"]][name] = (-3., -1., 1., 3.)[i % 4]
    model["matrix_and_target_sha256"], model["common_timestamp_target_sha256"] = native_hashes(a.NAMES45, inputs, labels)
    rows = [{"signal_time": a.iso(r["signal_time"]), **inputs[r["signal_time"]], "net_R": r["net_R"]} for r in labels]
    csv_file(path, ["signal_time", *a.NAMES45, "net_R"], rows)
    model["training_matrix"]["sha256"] = a.digest(path)
    estimator = model["estimator"]
    estimator["means"][0], estimator["std"][0], estimator["coefs"][0], estimator["threshold"] = 0., math.sqrt(5.), math.sqrt(5.), 1.75
    model["serialized_estimator_sha256"] = a.canonical_hash(estimator)
    audit = a.Audit()
    a.validate_model(model, inputs, labels, START, end, audit, "q75")
    assert not audit.failures


def test_scalar_score_threshold_has_no_tolerance_inflation_and_zero_is_not_issued():
    threshold = 1.
    inputs = {START + i * 300: {a.PATH_FEATURE: value, "atr": 1., "close": 100.}
              for i, value in enumerate((math.nextafter(threshold, 0.), threshold, math.nextafter(threshold, 2.), 0.))}
    model = {"status": "TESTED", "estimator": {"feature_names": [a.PATH_FEATURE], "means": [0.], "std": [1.],
             "coefs": [1.], "intercept": 0., "threshold": threshold}}
    chosen = a.expected_signals(inputs, "BOOM600", "SPIKE", "RIDGE45", model)
    assert [row["signal_time"] for row in chosen] == [START + 300, START + 600]
    model["estimator"]["threshold"] = 0.
    assert len(a.expected_signals(inputs, "BOOM600", "SPIKE", "RIDGE45", model)) == 3


def expansion_fixture(positive):
    reports = {family: {"status": "TESTED", "metrics": {
        "mean_inference": {"all_draws_defined": True}, "day_mean_ci95": [.05, .3],
        "censored": 0, "missing_entry": 0, "invalid_uncensored": 0}}
        for family in ("CLOCK", *a.FAMILIES)}
    comparisons = {key: {"status": "TESTED", "all_draws_defined": True, "ci95": [.01, .2]}
                   for key in ("RIDGE45_minus_RIDGE44", "RIDGE44_minus_CLOCK", "RIDGE45_minus_CLOCK")}
    value = {"positive_absolute_and_incremental_day_bounds": positive, "expansion_authorized": False,
             "historical_gate": False, "live_candidate": False,
             "scope": "known_history_economic_prerequisite_only_not_stability_or_profit_discovery"}
    return value, reports, comparisons


def test_positive_descriptive_expansion_bound_is_never_authorization_or_strategy_promotion():
    value, reports, comparisons = expansion_fixture(True)
    audit = a.Audit()
    a.audit_expansion(value, reports, comparisons, audit, "expansion")
    assert not audit.failures and value["expansion_authorized"] is False
    value["live_candidate"] = True
    a.audit_expansion(value, reports, comparisons, audit, "unsafe")
    assert audit.failures


@pytest.mark.parametrize("family", ["CLOCK", "RIDGE44", "RIDGE45"])
@pytest.mark.parametrize("unknown", ["censored", "missing_entry", "invalid_uncensored"])
def test_unknown_path_in_any_comparison_policy_blocks_positive_expansion_prerequisite(family, unknown):
    value, reports, comparisons = expansion_fixture(False)
    reports[family]["metrics"][unknown] = 1
    audit = a.Audit()
    a.audit_expansion(value, reports, comparisons, audit, "unknown")
    assert not audit.failures


@pytest.mark.parametrize("fault", ["undefined_absolute", "nonpositive_absolute", "undefined_increment", "nonpositive_increment", "untested_increment"])
def test_expansion_prerequisite_requires_positive_defined_absolute_and_incremental_bounds(fault):
    value, reports, comparisons = expansion_fixture(False)
    if fault == "undefined_absolute": reports["RIDGE45"]["metrics"]["mean_inference"]["all_draws_defined"] = False
    if fault == "nonpositive_absolute": reports["RIDGE45"]["metrics"]["day_mean_ci95"][0] = 0.
    if fault == "undefined_increment": comparisons["RIDGE45_minus_RIDGE44"]["all_draws_defined"] = False
    if fault == "nonpositive_increment": comparisons["RIDGE45_minus_CLOCK"]["ci95"][0] = 0.
    if fault == "untested_increment": comparisons["RIDGE45_minus_CLOCK"]["status"] = "NOT TESTED"
    audit = a.Audit()
    a.audit_expansion(value, reports, comparisons, audit, "incomplete")
    assert not audit.failures


@pytest.mark.parametrize("absolute", [False, True])
def test_inherited_absolute_audit_pin_matches_new_relative_lineage_without_reading_artifact(absolute):
    relative = "backend/tests/synthetic_path_risk_anchor.csv"
    fingerprint = "a" * 64
    old_name = str(a.ROOT / relative) if absolute else relative
    audit = a.Audit()
    a.audit_inherited_inputs({relative: fingerprint}, {old_name: fingerprint}, audit, "lineage")
    assert not audit.failures


@pytest.mark.parametrize("fault", ["omitted", "changed_hash", "duplicate_alias", "outside_repository"])
def test_inherited_lineage_rejects_missing_changed_duplicate_or_escaped_anchor(tmp_path, fault):
    relative = "backend/tests/synthetic_path_risk_anchor.csv"
    fingerprint = "a" * 64
    files, old = {relative: fingerprint}, {str(a.ROOT / relative): fingerprint}
    if fault == "omitted": files.clear()
    if fault == "changed_hash": files[relative] = "b" * 64
    if fault == "duplicate_alias": old[relative] = fingerprint
    if fault == "outside_repository": old = {str(tmp_path / "outside.csv"): fingerprint}
    audit = a.Audit()
    if fault == "outside_repository":
        with pytest.raises(ValueError): a.audit_inherited_inputs(files, old, audit, "unsafe")
    else:
        try:
            a.audit_inherited_inputs(files, old, audit, "unsafe")
        except ValueError:
            pass
        assert audit.failures


def test_unique_legacy_declaration_anchor_survives_repository_relocation_without_external_reads():
    anchor = "docs/spike_tick_age_payoff_20261005/declaration.json"
    relative = "backend/tests/synthetic_path_risk_anchor.csv"
    prefix = "/legacy/synthetic/boom-crash-quant-lab"
    files = {anchor: "a" * 64, relative: "b" * 64}
    old = {prefix + "/" + key: value for key, value in files.items()}
    audit = a.Audit()
    a.audit_inherited_inputs(files, old, audit, "relocated")
    assert not audit.failures


@pytest.mark.parametrize("fault", ["foreign_prefix", "prefix_collision", "traversal", "duplicate_alias"])
def test_legacy_anchor_does_not_accept_foreign_prefix_traversal_or_duplicate_relative_identity(fault):
    anchor = "docs/spike_tick_age_payoff_20261005/declaration.json"
    relative = "backend/tests/synthetic_path_risk_anchor.csv"
    prefix = "/legacy/synthetic/boom-crash-quant-lab"
    files = {anchor: "a" * 64, relative: "b" * 64}
    old = {prefix + "/" + anchor: files[anchor]}
    if fault == "foreign_prefix": old["/foreign/repository/" + relative] = files[relative]
    if fault == "prefix_collision": old[prefix + "-unrelated/" + relative] = files[relative]
    if fault == "traversal": old[prefix + "/backend/../" + relative] = files[relative]
    if fault == "duplicate_alias":
        old[prefix + "/" + relative] = files[relative]
        old[relative] = files[relative]
    audit = a.Audit()
    with pytest.raises(ValueError): a.audit_inherited_inputs(files, old, audit, "unsafe_legacy")
    assert audit.failures


@pytest.mark.parametrize("flag", a.FLAGS)
def test_any_true_live_flag_blocks_cli_before_actual_metadata_or_prices(monkeypatch, flag):
    for name in a.FLAGS:
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(sys, "argv", ["verify_spike_tick_path_risk.py"])
    with pytest.raises(ValueError): a.main()
