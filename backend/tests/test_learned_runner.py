"""Frozen learned-study issuance, label boundaries and evidence gates.

These checks use synthetic rows and ledgers only. They never open acquisition
files or compute historical payoff outcomes.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("learned_runner_tests", ROOT / "scripts/run_spike_learned_study.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def input_rows(n=180):
    times = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC", name="signal_time")
    phase = np.arange(n)
    rows = pd.DataFrame({"first": np.sin(phase / 8), "second": np.cos(phase / 11),
                         "atr": 2., "close": 100.}, index=times)
    target = .2 + .3 * rows["first"].to_numpy() - .1 * rows["second"].to_numpy()
    labels = pd.DataFrame({"signal_time": times, "entry_time": times + pd.Timedelta(minutes=1),
                           "exit_time": times + pd.Timedelta(minutes=2),
                           "planned_end": times + pd.Timedelta(minutes=16),
                           "net_R": target, "censored": False})
    return rows, labels, ["first", "second"]


def model(threshold=.2, intercept=.3):
    return {"version": 1, "feature_names": ["first"], "means": [0.], "std": [1.],
            "coefs": [1.], "intercept": intercept, "threshold": threshold}


def direction_candidate():
    fold = {"metrics": {"completed": 150, "censored": 0, "invalid_uncensored": 0,
                        "mean_net_R": .1, "profit_factor": 1.2}}
    folds = [{**deepcopy(fold), "training": training, "test": test}
             for training, test in (("train40", "wf1"), ("train50", "wf2"), ("train60", "wf3"))]
    return {"walk_forward": folds,
            "validation": {"completed": 500, "active_days": 30, "censored": 0,
                           "invalid_uncensored": 0, "selection_score": .05}}


def successful_row():
    metrics = {"completed": 1000, "profit_factor": 1.6, "censored": 0,
               "invalid_uncensored": 0, "active_days": 60,
               "day_profit_factor_ci95": [1.3, 1.9],
               "weekly_profit_factor_ci95": [1.2, 2.],
               "day_undefined_replicates": 0, "weekly_undefined_replicates": 0,
               "mean_net_R_ci95": [.02, .3], "weekly_mean_net_R_ci95": [.01, .4],
               "baseline_difference_ci95": [.03, .2], "weekly_difference_ci95": [.02, .3],
               "holm_p": .01, "equity_ruin": False, "closed_trade_max_drawdown": .09}
    return {"metrics": metrics, "audit": {"missing_entry": 0},
            "thirds": [{"metrics": {"completed": 300, "mean_net_R": .1}} for _ in range(3)],
            "sensitivities": [{"fill_mode": "adverse_extreme", "entry_delay_minutes": 1,
                               "round_trip_cost_atr": .20, "mean_net_R": .03}]}


def test_prepare_selects_exact_clock_closes_without_shifting_invalid_rows(monkeypatch):
    minute_index = pd.date_range("2026-01-01", periods=120, freq="min", tz="UTC")
    minute_frame = pd.DataFrame(index=minute_index)
    opening_index = pd.date_range(minute_index[0], periods=24, freq="5min")
    features = pd.DataFrame({"feature_valid": True, "first": np.arange(24),
                             "atr": 2., "close": 100.}, index=opening_index)
    # The 00:30 opportunity is invalid; an eligible 00:35 close must not replace it.
    features.loc[pd.Timestamp("2026-01-01T00:25Z"), "feature_valid"] = False
    monkeypatch.setattr(RUNNER, "load_m1", lambda path: (minute_frame, {"fixture": True}))
    monkeypatch.setattr(RUNNER, "causal_inputs", lambda m1, direction: (features, ["first"]))
    _, rows, names, audit, _ = RUNNER.prepare(Path("fixture-not-read.csv"), "BOOM500")
    assert names == ["first"]
    assert rows.index.tolist() == [pd.Timestamp("2026-01-01T01:00Z"),
                                   pd.Timestamp("2026-01-01T01:30Z"),
                                   pd.Timestamp("2026-01-01T02:00Z")]
    assert rows["first"].tolist() == [11, 17, 23]
    assert audit["eligible_clock_rows"] == 3
    assert rows.index.name == "signal_time"
    assert features.index.name is None


@pytest.mark.parametrize("symbol, mode, side", [
    ("BOOM500", "SPIKE", 1), ("BOOM500", "DRIFT", -1),
    ("CRASH500", "SPIKE", -1), ("CRASH500", "DRIFT", 1),
])
def test_issuer_uses_only_current_inputs_and_preserves_direction(symbol, mode, side, monkeypatch):
    rows, _, _ = input_rows(8)
    # Such columns must have no bearing on score or clock issuance.
    rows["future_net_R"] = [np.nan, 999, -999, np.inf, 1, 2, 3, 4]
    def forbidden(*args, **kwargs):
        raise AssertionError("Issuance must not replay future outcomes")
    monkeypatch.setattr(RUNNER, "replay_timed", forbidden)
    before = rows.copy(deep=True)
    result = RUNNER.issued(rows, symbol, mode)
    assert len(result) == len(rows)
    assert [r["signal_time"] for r in result] == rows.index.tolist()
    assert {r["side"] for r in result} == {side}
    assert {r["atr"] for r in result} == {2.}
    assert {r["signal_close"] for r in result} == {100.}
    assert {r["variant"] for r in result} == {f"CLOCK_{mode}"}
    pd.testing.assert_frame_equal(rows, before)


def test_frozen_threshold_is_not_refit_on_prediction_distribution():
    rows, _, _ = input_rows(5)
    rows["first"] = [-.51, -.5, -.25, -.249, 1.]
    frozen = model(threshold=.25, intercept=.5)
    original = deepcopy(frozen)
    result = RUNNER.issued(rows, "BOOM500", "SPIKE", frozen)
    assert [r["signal_time"] for r in result] == [rows.index[2], rows.index[3], rows.index[4]]
    assert [r["score"] for r in result] == pytest.approx([.25, .251, 1.5])
    assert frozen == original
    # Future opportunities may change their own scores, never the past cutoff.
    shifted = rows.copy()
    shifted.iloc[-1, shifted.columns.get_loc("first")] = 10_000
    again = RUNNER.issued(shifted, "BOOM500", "SPIKE", frozen)
    assert result[:-1] == again[:-1]
    assert frozen == original


def test_zero_and_negative_scores_or_empty_input_never_force_signals():
    rows, _, _ = input_rows(3)
    rows["first"] = [-1., -.3, -.4]
    assert RUNNER.issued(rows, "BOOM500", "SPIKE", model(threshold=0)) == []
    assert RUNNER.issued(rows.iloc[:0], "BOOM500", "SPIKE", model(threshold=0)) == []
    assert RUNNER.issued(rows.iloc[:0], "BOOM500", "SPIKE") == []
    with pytest.raises(ValueError, match="mode"):
        RUNNER.issued(rows, "BOOM500", "UNKNOWN")


def test_fit_at_common31_minute_purge_ignores_early_exit_and_heldout_mutation():
    rows, labels, names = input_rows(180)
    start = rows.index[0]
    end = rows.index[120] + pd.Timedelta(minutes=30)
    # Row120 would exit well before end, but its issue+31min exceeds end.
    assert labels.loc[120, "exit_time"] < end
    assert labels.loc[120, "planned_end"] < end
    fitted = RUNNER.fit_at(rows, labels, names, start, end)
    assert fitted["fit_rows"] == fitted["training_completed_labels"] == 120
    assert fitted["training_latest_issue"] == rows.index[119]
    assert fitted["training_latest_planned_end"] == rows.index[119] + pd.Timedelta(minutes=16)
    assert fitted["label_purge_minutes"] == 31
    mutated_rows, mutated_labels = rows.copy(), labels.copy()
    future = mutated_rows.index >= rows.index[120]
    mutated_rows.loc[future, names] = 9999
    mutated_labels.loc[mutated_labels.signal_time >= rows.index[120], "net_R"] = -9999
    # Censoring status and early exits in the heldout interval also cannot enter training.
    mutated_labels.loc[mutated_labels.signal_time >= rows.index[120], "censored"] = True
    mutated_labels.loc[mutated_labels.signal_time >= rows.index[120], "exit_time"] = rows.index[120]
    again = RUNNER.fit_at(mutated_rows, mutated_labels, names, start, end)
    assert fitted == again


def test_fit_at_boundary_equality_and_unknown_training_labels():
    rows, labels, names = input_rows(180)
    start = rows.index[0]
    end = rows.index[120] + pd.Timedelta(minutes=31)
    labels.loc[5, "censored"] = True
    labels.loc[6, "net_R"] = np.nan
    fitted = RUNNER.fit_at(rows, labels, names, start, end)
    assert fitted["fit_rows"] == fitted["training_completed_labels"] == 119
    assert fitted["training_censored_labels"] == 1
    assert fitted["training_invalid_labels"] == 1
    assert fitted["training_latest_issue"] == rows.index[120]
    assert fitted["training_latest_planned_end"] <= end
    expected = rows.loc[rows.index[:121].difference(rows.index[[5, 6]]), names]
    assert fitted["means"] == pytest.approx(expected.mean().to_numpy())


def test_fit_at_cannot_replace_missing_training_labels_with_future_samples():
    rows, labels, names = input_rows(180)
    start, end = rows.index[0], rows.index[99] + pd.Timedelta(minutes=31)
    labels.loc[10, "censored"] = True
    with pytest.raises(ValueError, match="100"):
        RUNNER.fit_at(rows, labels, names, start, end)


def test_development_eligibility_requires_three_complete_positive_folds():
    value = direction_candidate()
    assert RUNNER.development_eligible(value)
    value["walk_forward"] = []
    assert not RUNNER.development_eligible(value)
    value = direction_candidate()
    value["walk_forward"].pop()
    assert not RUNNER.development_eligible(value)
    value = direction_candidate()
    value["walk_forward"][0]["test"] = "wf3"
    assert not RUNNER.development_eligible(value)
    value = direction_candidate()
    value["walk_forward"] = value["walk_forward"][::-1]
    assert not RUNNER.development_eligible(value)


@pytest.mark.parametrize("field, bad", [("completed", 99), ("censored", 1),
                                        ("invalid_uncensored", 1), ("mean_net_R", 0),
                                        ("mean_net_R", None), ("profit_factor", 1),
                                        ("profit_factor", None)])
def test_failed_validation_fold_cannot_be_redeemed_by_other_successes(field, bad):
    value = direction_candidate()
    value["walk_forward"][1]["metrics"][field] = bad
    value["cross_symbol_transfer"] = {"completed": 10000, "profit_factor": 10}
    assert not RUNNER.development_eligible(value)


@pytest.mark.parametrize("field, bad", [("completed", 499), ("active_days", 29),
                                        ("censored", 1), ("invalid_uncensored", 1),
                                        ("selection_score", 0), ("selection_score", None)])
def test_aggregate_development_requirements_are_mandatory(field, bad):
    value = direction_candidate()
    value["validation"][field] = bad
    assert not RUNNER.development_eligible(value)


def test_gate_distinguishes_observed_pf_from_uncertainty_supported_pf_target():
    row = successful_row()
    RUNNER.gate(row, eligible=True)
    assert row["user_target_observed"] and row["historical_candidate"]
    assert not row["supports_expected_pf_1_5"]
    assert row["actual_money_profit"] == row["prospective_validation"] == "NOT TESTED"
    row = successful_row()
    row["metrics"].update(day_profit_factor_ci95=[1.5, 2.], weekly_profit_factor_ci95=[1.5, 2.])
    RUNNER.gate(row, eligible=True)
    assert row["supports_expected_pf_1_5"]


@pytest.mark.parametrize("completed, pf", [(999, 100), (1000, 1.49999), (1000, None)])
def test_user_target_needs_large_sample_and_defined_pf(completed, pf):
    row = successful_row()
    row["metrics"].update(completed=completed, profit_factor=pf)
    RUNNER.gate(row, eligible=True)
    assert not row["user_target_observed"]
    assert not row["historical_candidate"] and not row["supports_expected_pf_1_5"]
    assert "user_pf1_5_n1000_not_met" in row["rejection_reasons"]


def test_rejected_development_cannot_be_promoted_by_transfer_success():
    row = successful_row()
    row["metrics"]["profit_factor"] = 10
    RUNNER.gate(row, eligible=False)
    assert row["user_target_observed"]
    assert not row["historical_candidate"]
    assert "development_rejected" in row["rejection_reasons"]


@pytest.mark.parametrize("interval", ["day_profit_factor_ci95", "weekly_profit_factor_ci95"])
@pytest.mark.parametrize("lower", [None, 1., .9])
def test_pf_uncertainty_requires_both_intervals_strictly_above_one(interval, lower):
    row = successful_row()
    row["metrics"][interval] = [lower, 2.]
    RUNNER.gate(row, eligible=True)
    assert not row["historical_candidate"]
    assert "pf_ci_not_above1" in row["rejection_reasons"]


@pytest.mark.parametrize("field", ["day_undefined_replicates", "weekly_undefined_replicates"])
def test_undefined_pf_resamples_prevent_promotion_even_if_finite_ci_looks_good(field):
    row = successful_row()
    row["metrics"][field] = 1
    RUNNER.gate(row, eligible=True)
    assert not row["historical_candidate"]
    assert not row["supports_expected_pf_1_5"]


@pytest.mark.parametrize("field", ["mean_net_R_ci95", "weekly_mean_net_R_ci95",
                                   "baseline_difference_ci95", "weekly_difference_ci95"])
def test_mean_and_clock_advantage_need_positive_lower_bounds(field):
    row = successful_row()
    row["metrics"][field] = [0., .3]
    RUNNER.gate(row, eligible=True)
    assert not row["historical_candidate"]


def test_uncensored_missing_entry_or_doubled_cost_loss_reject_candidate():
    row = successful_row()
    row["audit"]["missing_entry"] = 1
    row["sensitivities"][0]["mean_net_R"] = 0.
    RUNNER.gate(row, eligible=True)
    assert not row["historical_candidate"]
    assert "missing_or_censored_outcomes" in row["rejection_reasons"]
    assert "doubled_cost_not_positive" in row["rejection_reasons"]
