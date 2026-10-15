"""Synthetic independent conditional-region audit; no historical quote IO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import verify_region_reward_study as M
from scripts import verify_zone_study as Z


def example(symbol="BOOM600"):
    start = pd.Timestamp("2026-01-01T00:00Z")
    times = np.arange(M.epoch(start), M.epoch(start) + 4000, dtype=np.int64)
    prices = np.full(len(times), 100.)
    side = 1 if symbol == "BOOM600" else -1
    signals = pd.DataFrame({"signal_time": [start], "atr": [1.], "side": [side], "signal_close": [100.]})
    return times, prices, signals, (start, start + pd.Timedelta(seconds=4000))


@pytest.mark.parametrize("symbol,lo,hi,inv", [("BOOM600", 99.45, 99.55, 99.2), ("CRASH600", 100.45, 100.55, 100.8)])
def test_original_geometry_and_fixed_native_side(symbol, lo, hi, inv):
    _, _, signals, _ = example(symbol)
    z = M.region(signals.iloc[0], symbol, "RAW_REGION")
    assert (z["zone_low"], z["zone_high"], z["invalidation"]) == pytest.approx((lo, hi, inv))
    assert z["zone_id"] == f"{symbol}:RAW_REGION:2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize("change", ["side", "bool_side", "clock", "atr", "close", "geometry", "symbol"])
def test_malformed_source_region_rejected(change):
    _, _, signals, _ = example(); signal = signals.iloc[0].to_dict(); symbol = "BOOM600"
    if change == "side": signal["side"] = -1
    elif change == "bool_side": signal["side"] = True
    elif change == "clock": signal["signal_time"] += pd.Timedelta(seconds=1)
    elif change == "atr": signal["atr"] = np.nan
    elif change == "close": signal["signal_close"] = 0
    elif change == "geometry": signal["atr"] = 1000
    else: symbol = "BOOM500"
    with pytest.raises(ValueError): M.region(signal, symbol, "RAW_REGION")


@pytest.mark.parametrize("symbol", ["BOOM600", "CRASH600"])
def test_conditional_reward_uses_actual_next_quote_and_uncapped_stop(symbol):
    times, prices, signals, bounds = example(symbol)
    side = int(signals.side.iloc[0]); prices[61] = 100 - side * .5
    prices[62:100] = 100 + side * 3
    prices[100] = 100 + side * .9
    prices[101] = 100 - side * 1
    events = M.oracle_events(times, prices, signals, symbol, "CLOCK", bounds, M.FIXED_CONFIG)
    event = events[0]
    assert event["status"] == "completed" and event["reason"] == "sl"
    assert event["entry_time"] == times[62] and event["entry"] == prices[62]
    assert event["gross_R"] == -2 and event["net_R"] == -2.05
    labels = M.conditional_labels(signals.signal_time, events)
    assert labels.completed.tolist() == [True] and labels.net_R.tolist() == [-2.05]
    assert labels.planned_end.iloc[0] == bounds[0] + pd.Timedelta(minutes=30, seconds=1)


@pytest.mark.parametrize("state", ["expired", "invalidated", "waiting_gap", "entry_gap", "path_gap", "overlap_skipped", "completed"])
def test_all_clock_categories_keep_distinct_nan_semantics(state):
    _, _, signals, _ = example(); issue = M.epoch(signals.signal_time.iloc[0])
    event = {"signal_time": issue, "status": state, "censored": state in M.UNKNOWN,
             "entry_time": issue + 62 if state in ("completed", "path_gap") else None,
             "net_R": 0. if state == "completed" else None}
    labels = M.conditional_labels(signals.signal_time, [event])
    assert labels[["completed", "known_nonfill", "unknown_path", "exposure_skipped"]].to_numpy().sum() == 1
    assert labels.completed.iloc[0] == (state == "completed")
    assert labels.known_nonfill.iloc[0] == (state in M.NONFILL)
    assert labels.unknown_path.iloc[0] == (state in M.UNKNOWN)
    assert labels.exposure_skipped.iloc[0] == (state == "overlap_skipped")
    assert (labels.net_R.iloc[0] == 0) if state == "completed" else pd.isna(labels.net_R.iloc[0])


@pytest.mark.parametrize("change", ["wrong_clock", "zero_nonfill", "finite_unknown", "missing_completed", "censor", "unfilled_completed", "purged"])
def test_invalid_independent_label_dispositions_refused(change):
    _, _, signals, _ = example(); issue = M.epoch(signals.signal_time.iloc[0])
    event = {"signal_time": issue, "status": "completed", "censored": False, "entry_time": issue + 62, "net_R": .2}
    if change == "wrong_clock": event["signal_time"] += 1800
    elif change == "zero_nonfill": event.update(status="expired", net_R=0, entry_time=None)
    elif change == "finite_unknown": event.update(status="path_gap", censored=True)
    elif change == "missing_completed": event["net_R"] = None
    elif change == "censor": event["censored"] = True
    elif change == "unfilled_completed": event["entry_time"] = None
    else: event["status"] = "purged"
    with pytest.raises(ValueError): M.conditional_labels(signals.signal_time, [event])


def labels_fixture():
    _, _, signals, bounds = example()
    issues = pd.date_range(bounds[0], periods=4, freq="30min")
    events = [{"signal_time": M.epoch(t), "status": state, "censored": state in M.UNKNOWN,
               "entry_time": M.epoch(t)+62 if state == "completed" else None,
               "net_R": .3 if state == "completed" else None}
              for t, state in zip(issues, ("completed", "expired", "waiting_gap", "overlap_skipped"), strict=True)]
    labels = M.conditional_labels(issues, events)
    return labels, bounds[0] + pd.Timedelta(hours=3)


def test_saved_labels_equal_independent_full_population():
    labels, end = labels_fixture(); audit = M.Audit()
    M.check_labels(audit, labels.copy(), labels, end)
    assert not audit.errors


@pytest.mark.parametrize("change", ["zero", "category", "status", "horizon", "clock"])
def test_saved_label_tampering_is_detected(change):
    labels, end = labels_fixture(); actual = labels.copy(); audit = M.Audit()
    if change == "zero": actual.loc[1, "net_R"] = 0
    elif change == "category": actual.loc[1, "known_nonfill"] = False
    elif change == "status": actual.loc[1, "status"] = "completed"
    elif change == "horizon": actual["planned_end"] += pd.Timedelta(seconds=1)
    else: actual["signal_time"] += pd.Timedelta(minutes=30)
    M.check_labels(audit, actual, labels, end)
    assert audit.errors


@pytest.mark.parametrize("value", [1, "true", None, np.nan])
def test_unknown_or_nonboolean_categories_are_never_coerced(value):
    with pytest.raises(ValueError): M.booleans(pd.Series([value]), "category")


def test_target_hash_binds_order_times_and_actual_reward():
    issues = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    y = np.array([.2, -.5, 1.]); before = M.target_hash(issues, y)
    assert before == M.target_hash(issues, y.copy())
    assert before != M.target_hash(issues[::-1], y)
    assert before != M.target_hash(issues, y + .01)
    assert before != M.target_hash(issues + pd.Timedelta(seconds=1), y)


def model_fixture(n=1000):
    issues = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    y = np.linspace(-.2, .8, n); x = np.c_[np.arange(n), np.ones(n)]
    labels = pd.DataFrame({"signal_time": issues, "completed": True, "net_R": y,
        "known_nonfill": False, "unknown_path": False, "exposure_skipped": False, "status": "completed"})
    raw = pd.DataFrame(x, columns=["x", "constant"]); hybrid = raw * 3 + 7
    names = list(raw.columns); digest = M.target_hash(issues, y); matrices, models = {}, {}
    for arm, features in (("RAW_REGION", raw), ("HYBRID_REGION", hybrid)):
        model = M.independent_ridge(features, y)
        models[arm] = {**{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()},
            "feature_names": names, "fit_rows": n, "penalty": .1, "quantile": .75, "version": 1}
        h = hashlib.sha256(digest.encode());h.update(np.asarray(features, dtype=">f8").tobytes());matrices[arm] = h.hexdigest()
    end = issues[-1] + pd.Timedelta(minutes=31)
    meta = {"symbol": "BOOM600", "native_side": 1, "training_start": issues[0].isoformat(), "training_end": end.isoformat(),
        "completed": n, "known_nonfill": 0, "unknown_path": 0, "exposure_skipped": 0, "opportunities": n,
        "feature_names": names, "penalty": .1, "quantile": .75, "safety": M.safety(),
        "target": "conditional_completed_original_quote_CLOCK_region_net_R", "training_is_independent_trade_sample": False,
        "label_selection_is_conditional_on_CLOCK_fill": True, "target_sha256": digest,
        "matrix_sha256": matrices, "model_sha256": {k: M.fingerprint(v) for k, v in models.items()}}
    meta["metadata_sha256"] = M.fingerprint(meta)
    state = {"completed": n, "known_nonfill": 0, "unknown_path": 0, "exposure_skipped": 0,
        "training_dispositions": {"completed": n}, "safety": M.safety(), "QUALIFIED": False,
        "status": "FITTED", "error": None, "models": models, "model_metadata": meta}
    return state, labels, raw, hybrid, np.arange(n), names, "BOOM600", [issues[0], end]


def test_independently_solved_paired_models_and_all_hashes_match():
    args = model_fixture(); audit = M.Audit(); models = M.check_model(audit, *args)
    assert not audit.errors and set(models) == set(M.NEW)
    assert models["RAW_REGION"]["std"][1] == 1
    assert models["HYBRID_REGION"]["std"][1] == 1
    assert models["RAW_REGION"]["means"][0] != models["HYBRID_REGION"]["means"][0]


@pytest.mark.parametrize("change", ["coef", "mean", "threshold", "target_hash", "matrix_hash", "model_hash", "metadata_hash", "bounds", "rows"])
def test_model_arithmetic_or_identity_tampering_is_detected(change):
    args = list(model_fixture()); state = args[0]; audit = M.Audit()
    if change == "coef": state["models"]["RAW_REGION"]["coefs"][0] += .01
    elif change == "mean": state["models"]["HYBRID_REGION"]["means"][0] += .01
    elif change == "threshold": state["models"]["RAW_REGION"]["threshold"] += .01
    elif change == "target_hash": state["model_metadata"]["target_sha256"] = "0" * 64
    elif change == "matrix_hash": state["model_metadata"]["matrix_sha256"]["RAW_REGION"] = "0" * 64
    elif change == "model_hash": state["model_metadata"]["model_sha256"]["HYBRID_REGION"] = "0" * 64
    elif change == "metadata_hash": state["model_metadata"]["metadata_sha256"] = "0" * 64
    elif change == "bounds": state["model_metadata"]["training_end"] = "2026-03-01T00:00Z"
    else: state["models"]["RAW_REGION"]["fit_rows"] += 1
    M.check_model(audit, *args)
    assert audit.errors


def test_insufficient_training_cannot_make_oracles_from_nonfills():
    args = list(model_fixture(999)); state = args[0];state.update(status="NOT_FIT_INSUFFICIENT_REGION_LABELS", models=None, model_metadata=None)
    audit = M.Audit(); assert M.check_model(audit, *args) == {} and not audit.errors
    state["models"] = {"RAW_REGION": {}}
    audit = M.Audit();M.check_model(audit, *args);assert audit.errors


def test_evaluation_score_uses_frozen_scaler_cutoff_positive_floor_and_no_labels():
    features = pd.DataFrame({"x": [-1., 0., 1., 2.]})
    model = {"intercept": 0., "means": [0.], "std": [1.], "coefs": [1.], "threshold": 1.}
    scores, selected = M.scores_and_selection(features, np.arange(4), ["x"], model)
    np.testing.assert_array_equal(scores, [-1, 0, 1, 2]);np.testing.assert_array_equal(selected, [False, False, True, True])
    model["threshold"] = 0
    assert M.scores_and_selection(features, np.arange(4), ["x"], model)[1].tolist() == [False, False, True, True]


def test_frozen_reference_renaming_alone_preserves_identity():
    actual = pd.DataFrame({"signal_time": ["2026-01-01T00:00Z"], "variant": ["RAW_TIMED"], "zone_id": ["new"], "net_R": [.1]})
    previous = actual.copy();previous["variant"] = "RAW44";previous["zone_id"] = "old"
    audit = M.Audit();M.check_reference(audit, actual, previous, "reference");assert not audit.errors
    actual.loc[0, "net_R"] += 1e-8
    audit = M.Audit();M.check_reference(audit, actual, previous, "reference");assert audit.errors


@pytest.mark.parametrize("week", [False, True])
def test_paired_reference_inference_detects_ci_and_p_tampering(week):
    keys = ("weekly_mean_net_R_ci95", "weekly_difference_ci95", "weekly_p") if week else ("mean_net_R_ci95", "baseline_difference_ci95", "p")
    values = ([.1, .3], [.01, .2], .005)
    saved = dict(zip(keys, values, strict=True)); audit = M.Audit()
    assert M.compare_inference(audit, saved, values, week, "reference") == .005
    assert not audit.errors
    saved[keys[1]] = [0, .2];saved[keys[2]] = .001
    M.compare_inference(audit, saved, values, week, "reference");assert len(audit.errors) == 2


def test_joint_holm_handles_order_ties_and_no_candidate_pooling():
    np.testing.assert_allclose(M.holm([.01, .03, .01, .8]), [.04, .06, .04, .8])
    np.testing.assert_array_equal(M.holm([1.] * 8), np.ones(8))


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-.1], [1.1], [[.2]]])
def test_undefined_or_invalid_holm_probabilities_refused(values):
    with pytest.raises(ValueError): M.holm(values)


def test_numpy_decoder_preserves_original_prices_and_missing_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(Z, "ROOT", tmp_path)
    start = M.epoch(pd.Timestamp("2026-01-01T00:00Z")); p = tmp_path / "quotes.csv"
    p.write_text(f"epoch,quote\n{start},100\n{start+1},101\n{start+4},98\n")
    source = {"path": p.name, "rows": 3, "date": "2026-01-01", "sha256": M.sha(p)}
    times, prices = M.decode_sources(M.Audit(), [source])
    np.testing.assert_array_equal(times-start, [0, 1, 4]);np.testing.assert_array_equal(prices, [100, 101, 98])


@pytest.mark.parametrize("change", ["pin", "schema", "count", "date", "order", "price"])
def test_source_tampering_or_invalid_daily_contract_refused(monkeypatch, tmp_path, change):
    monkeypatch.setattr(Z, "ROOT", tmp_path)
    start = M.epoch(pd.Timestamp("2026-01-01T00:00Z"));p = tmp_path / "quotes.csv"
    p.write_text(f"epoch,quote\n{start},100\n{start+1},101\n")
    source = {"path": p.name, "rows": 2, "date": "2026-01-01", "sha256": M.sha(p)}
    if change == "pin": p.write_text(p.read_text() + "changed")
    elif change == "schema": p.write_text(p.read_text().replace("epoch,quote", "time,price"))
    elif change == "count": source["rows"] = 3
    elif change == "date": source["date"] = "2026-01-02"
    elif change == "order": p.write_text(f"epoch,quote\n{start+1},100\n{start},101\n")
    else: p.write_text(p.read_text().replace(",101", ",-1"))
    if change != "pin": source["sha256"] = M.sha(p)
    with pytest.raises(ValueError): M.decode_sources(M.Audit(), [source])


@pytest.mark.parametrize("flag", M.FLAGS)
def test_every_live_flag_refuses_before_audit_io(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match="Live flag"): M.verify(M.Audit())


@pytest.mark.parametrize("values", [["2026-01-01T00:00"], ["2026-01-01T00:00:00.001Z"], ["2026-01-01T00:00Z"]*2, [None]])
def test_unqualified_fractional_duplicate_and_missing_timestamps_refused(values):
    with pytest.raises((ValueError, TypeError)): M.timestamps(pd.Series(values))


def test_auditor_imports_no_production_learning_or_replay_modules():
    source = Path(M.__file__).read_text()
    assert "from app." not in source and "import app." not in source
    assert "from scripts.region_reward_learning" not in source
    assert "from scripts.run_" not in source
