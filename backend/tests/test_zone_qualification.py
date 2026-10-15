from copy import deepcopy

import pytest

from app.research.zone_qualification import historical_zone_gate, holm_adjust


def qualifying_inputs():
    m = dict(completed=1200, active_days=100, profit_factor=1.6, censored=0,
             invalid_uncensored=0, mean_net_R=.2, selection_score=.1,
             closed_trade_max_drawdown=.08, equity_ruin=False,
             unknown_regions=0, control_unknown_regions=0,
             mean_net_R_ci95=[.1, .3], baseline_difference_ci95=[.05, .2])
    return dict(metrics=m, pf={"day_profit_factor_ci95": [1.2, 1.9],
                              "weekly_profit_factor_ci95": [1.1, 2.]},
                weekly={"weekly_mean_net_R_ci95": [.08, .3],
                        "weekly_difference_ci95": [.03, .2]},
                folds=[deepcopy(m) for _ in range(3)], validation=deepcopy(m),
                thirds=[deepcopy(m) for _ in range(3)], doubled_cost=deepcopy(m),
                unknown_regions=0, baseline_unknown_regions=0, holm_p=.01)


def test_strong_historical_result_never_becomes_fresh_oos_or_live():
    result = historical_zone_gate(**qualifying_inputs())
    assert result["historical_criteria_passed"]
    assert not result["qualified"] and not result["fresh_out_of_sample"]
    assert not result["PF_1_5_lower_CI_supported"]


@pytest.mark.parametrize("field,value,reason", [
    ("completed", 999, "fewer_than_1000_completed"),
    ("active_days", 59, "fewer_than_60_active_days"),
    ("profit_factor", 1.4999999999999, "PF_below_1.5_or_unknown"),
    ("profit_factor", None, "PF_below_1.5_or_unknown"),
    ("closed_trade_max_drawdown", .101, "drawdown_or_ruin"),
    ("censored", 1, "incomplete_payoff"),
])
def test_each_single_symbol_gate_cannot_be_rescued_by_other_results(field, value, reason):
    args = qualifying_inputs()
    args["metrics"][field] = value
    result = historical_zone_gate(**args)
    assert reason in result["historical_rejection_reasons"]
    assert not result["historical_criteria_passed"]


def test_failing_early_fold_is_not_rescued_by_heldout_or_combined_profits():
    args = qualifying_inputs()
    args["folds"][0]["mean_net_R"] = -.001
    assert not historical_zone_gate(**args)["development_eligible"]


def test_unfilled_unknown_development_region_cannot_be_removed_from_gate():
    args = qualifying_inputs()
    args["folds"][1]["unknown_regions"] = 1
    assert not historical_zone_gate(**args)["development_eligible"]


@pytest.mark.parametrize("key", ["unknown_regions", "baseline_unknown_regions"])
def test_missing_region_cannot_vanish_by_only_reporting_filled_trades(key):
    args = qualifying_inputs()
    args[key] = 1
    assert "unknown_region_outcomes" in historical_zone_gate(**args)["historical_rejection_reasons"]


def test_exact_target_and_ci_boundary():
    args = qualifying_inputs()
    args["metrics"]["profit_factor"] = 1.5
    for key in args["pf"]:
        args["pf"][key] = [1.5, 1.9]
    result = historical_zone_gate(**args)
    assert result["historical_criteria_passed"] and result["PF_1_5_lower_CI_supported"]


def test_holm_family_is_not_six_separate_single_tests():
    assert holm_adjust([.04, .003, .01]) == pytest.approx([.04, .009, .02])
    assert holm_adjust([.01] * 12) == pytest.approx([.12] * 12)
    assert holm_adjust([]) == []
