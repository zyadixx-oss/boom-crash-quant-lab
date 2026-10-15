"""Independent oracle fixtures, including gaps and empty statistical cells."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.verify_hybrid_event_study import (Audit, clock_rows, compare_record,
    independent_ridge, independent_timed, inference_oracle, saved_inference)

CFG = {"stop_atr": 2., "max_hold_minutes": 15, "entry_delay_minutes": 1,
       "round_trip_cost_atr": .1, "max_gap_seconds": 1, "stop_latency_ticks": 1}


def example():
    start = pd.Timestamp("2026-01-01T00:00Z").value // 10**9
    times = np.arange(start, start + 2000, dtype=np.int64)
    return times, np.full(len(times), 100.), {"signal_time": pd.Timestamp(start, unit="s", tz="UTC"),
        "side": 1, "atr": 1., "variant": "HYBRID44_SPIKE"}, start


@pytest.mark.parametrize("side", [1, -1])
def test_next_second_timed_stop_has_uncapped_fill_and_raw_r(side):
    times, prices, signal, start = example(); signal["side"] = side
    prices[100] = 100 - side * 2
    prices[101] = 100 - side * 3
    path, busy, status = independent_timed(times, prices, signal, CFG, start, start + 2000, start)
    assert status == "completed" and path["reason"] == "sl"
    assert path["entry_time"] == start + 61
    assert path["trigger_time"] == start + 100 and path["exit_time"] == start + 101
    assert path["gross_R"] == -1.5 and path["net_R"] == -1.55
    assert busy == path["exit_time"]


def test_stop_at_nominal_expiry_precedes_timeout():
    times, prices, signal, start = example()
    prices[960], prices[961] = 98, 99
    path, _, _ = independent_timed(times, prices, signal, CFG, start, start + 2000, start)
    assert path["reason"] == "sl" and path["trigger_time"] == start + 960
    assert path["exit_time"] == start + 961 and path["net_R"] == -.55


@pytest.mark.parametrize("missing", [61, 80, 101, 961])
def test_missing_entry_path_stopfill_and_expiry_remain_unknown(missing):
    times, prices, signal, start = example()
    if missing == 101: prices[100] = 98
    mask = times != start + missing
    path, busy, status = independent_timed(times[mask], prices[mask], signal, CFG, start, start + 2000, start)
    if missing == 61:
        assert path is None and status == "missing_entry"
    else:
        assert status == "censored" and path["net_R"] is None
        assert path["missing_time"] == start + missing
    assert busy == start + 960


def test_completed_stop_ignores_later_gap():
    times, prices, signal, start = example(); prices[100] = 98
    mask = times != start + 800
    path, _, status = independent_timed(times[mask], prices[mask], signal, CFG, start, start + 2000, start)
    assert status == "completed" and path["reason"] == "sl" and not path["censored"]


def test_partition_purge_is_planned_and_overlap_is_before_entry():
    times, prices, signal, start = example()
    assert independent_timed(times, prices, signal, CFG, start, start + 1000, start)[2] == "purged"
    assert independent_timed(times, prices, signal, CFG, start, start + 2000, start + 62)[2] == "overlap_skipped"


def test_augmented_ridge_matches_analytic_one_feature_solution():
    x = np.arange(1200, dtype=float).reshape(-1, 1)
    z = (x[:, 0] - x.mean()) / x.std()
    model = independent_ridge(x, 1 + .3 * z)
    assert model["coefs"][0] == pytest.approx(.3 / 1.1, abs=1e-14)
    assert model["intercept"] == pytest.approx(1, abs=1e-14)
    assert model["threshold"] == pytest.approx(np.quantile(1 + (.3 / 1.1) * z, .75), abs=1e-14)


def test_constant_column_and_negative_score_floor():
    model = independent_ridge(np.ones((1200, 2)), np.full(1200, -.2))
    np.testing.assert_array_equal(model["std"], [1, 1])
    np.testing.assert_allclose(model["coefs"], 0, atol=1e-14)
    assert model["threshold"] == 0


def test_shared_clock_requires_both_masks_and_full_planned_purge():
    raw = pd.DataFrame({"m5_open": pd.date_range("2026-01-01", periods=24, freq="5min", tz="UTC"),
                        "feature_valid": True})
    hybrid = raw.copy(); hybrid.loc[5, "feature_valid"] = False
    positions, issues = clock_rows(raw, hybrid, "2026-01-01T00:00Z", "2026-01-01T02:00Z")
    assert positions.tolist() == [11]
    assert issues[0] == pd.Timestamp("2026-01-01T01:00Z")


@pytest.mark.parametrize("week", [False, True])
def test_paired_constant_outcomes_and_empty_resample_policy(week):
    times = ["2026-01-01T00:00Z", "2026-01-02T00:00Z"]
    model = pd.DataFrame({"entry_time": times, "net_R": [2., 2.], "censored": False})
    base = model.copy(); base.net_R = 1.
    ci, delta, p = inference_oracle(model, base, "2026-01-01T00:00Z", "2026-01-03T00:00Z", 999, 4, week)
    assert ci == [2., 2.] and delta == [1., 1.] and p == .001
    ci, delta, p = inference_oracle(model.iloc[:0], base, "2026-01-01T00:00Z", "2026-01-03T00:00Z", 999, 4, week)
    assert ci == [None, None] and delta == [None, None] and p == 1


def test_region_and_timed_timestamp_fields_compare_as_epochs():
    audit = Audit()
    stamp = "2026-01-01T00:00:00Z"; number = pd.Timestamp(stamp).value // 10**9
    actual = {n: stamp for n in ("activation_time", "wait_end", "maximum_end", "nominal_entry_time", "planned_end")}
    compare_record(audit, actual, {n: number for n in actual}, "time schema")
    assert not audit.errors and audit.checks == 5


@pytest.mark.parametrize("week", [False, True])
def test_only_declared_descriptive_union_omission_can_be_supplemented(week):
    row = {"partition": "walk_forward_combined", "variant": "HYBRID44"}
    assert saved_inference(row, week, True) is None
    row["representation_comparison"] = {"day": {"p": .2}, "week": {"weekly_p": .3}}
    assert saved_inference(row, week, True) == row["representation_comparison"]["week" if week else "day"]


@pytest.mark.parametrize("part", ["final_test", "later180"])
def test_missing_heldout_comparison_cannot_be_skipped(part):
    with pytest.raises(ValueError): saved_inference({"partition": part, "variant": "HYBRID44"}, False, True)
