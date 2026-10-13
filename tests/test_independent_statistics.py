"""Artificial daily vectors only; no market prices, ledgers or strategy evaluation."""
import json

import pytest

from independent_profit_research.statistics import (
    ALPHA, CLAIM_IDS, Claim, DailyRecords, analyze_daily, development_gate,
    final_statistical_gate, holm12, stability_gate,
)


SHA = '0123456789abcdef'*4


def records(r=None, **kwargs):
    r = [1.]*30 if r is None else r
    n = len(r)
    values = dict(daily_R=r, daily_quote=[2*x for x in r], clock_daily_R=[0.]*n,
                  completed_counts=[10]*n, unknown_counts=[0]*n, clock_unknown_counts=[0]*n,
                  coverage_complete=True, clock_coverage_complete=True)
    values.update(kwargs)
    return DailyRecords(**values)


def analyze(value, **kwargs):
    return analyze_daily(value, declaration_sha256=SHA, development_eligible=True, **kwargs)


def test_daily_vectors_must_have_same_nonempty_calendar_length():
    with pytest.raises(ValueError): records([], daily_quote=[])
    with pytest.raises(ValueError): records(clock_daily_R=[0.]*29)


def test_counts_and_coverage_fail_closed():
    for field in ('completed_counts', 'unknown_counts', 'clock_unknown_counts'):
        for bad in (-1, True, 1.5):
            with pytest.raises(ValueError): records(**{field: [bad]*30})
    with pytest.raises(ValueError): records(coverage_complete='true')
    with pytest.raises(ValueError): records(clock_coverage_complete=1)


def test_nonfinite_or_unknown_values_are_not_zero_imputed():
    for field in ('daily_R', 'daily_quote', 'clock_daily_R'):
        for bad in (float('nan'), float('inf'), None, True):
            with pytest.raises(ValueError): records(**{field: [bad]*30})


def test_daily_snapshot_does_not_follow_mutable_input():
    original = [1.]*30
    value = records(original)
    original[0] = -100
    assert value.daily_R == (1.,)*30


def test_payoff_without_completed_trade_is_rejected():
    with pytest.raises(ValueError): records(completed_counts=[0]*30)
    with pytest.raises(ValueError): records(daily_gain_R=[1.]*30)
    with pytest.raises(ValueError): records(daily_gain_R=[2.]*30, daily_loss_R=[0.]*30)
    with pytest.raises(ValueError): records(daily_gain_R=[-1.]*30, daily_loss_R=[0.]*30)


def test_constant_positive_daily_vector_has_exact_closed_form():
    result = analyze(records(clock_daily_R=[.5]*30))
    assert result['testable'] and result['draws'] == 99999
    assert result['daily_mean_R'] == 1 and result['observed_sum_R'] == 30
    assert result['observed_mean_R_per_completed_trade'] == .1
    assert result['ci95_daily_mean_R'] == [1., 1.]
    assert result['ci95_daily_mean_quote'] == [2., 2.]
    assert result['ci95_daily_mean_excess_R'] == [.5, .5]
    assert result['p_positive_R'] == result['p_excess_R_vs_clock'] == .00001
    assert result['complete_nonoverlapping_blocks'] == 10
    assert result['cash_net'] is None and result['profit_qualified'] is False
    assert result['median_trade_R'] is None
    assert all(result[key] is False for key in ('LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES',
                                               'live_trading', 'ready_for_live', 'live_allowed', 'opened_trades'))


def test_constant_zero_daily_vector_is_not_positive():
    result = analyze(records([0.]*30))
    assert result['ci95_daily_mean_R'] == [0., 0.]
    assert result['p_positive_R'] == result['p_excess_R_vs_clock'] == 1
    assert not final_statistical_gate(result, holm_positive_p=1, holm_excess_p=1)['conditional_statistical_pass']


def test_constant_negative_daily_vector_has_p_one():
    result = analyze(records([-1.]*30))
    assert result['ci95_daily_mean_R'] == [-1., -1.]
    assert result['p_positive_R'] == result['p_excess_R_vs_clock'] == 1


def test_unknown_candidate_or_clock_blocks_both_claims_without_dropping_day():
    for field in ('unknown_counts', 'clock_unknown_counts'):
        unknown = [0]*30
        unknown[12] = 1
        result = analyze(records(**{field: unknown}))
        assert result['calendar_days'] == 30 and result['observed_sum_R'] == 30
        assert not result['testable'] and result['draws'] == 0
        assert result['p_positive_R'] == result['p_excess_R_vs_clock'] == 1
        assert result['ci95_daily_mean_R'] is None and result['ci95_daily_mean_excess_R'] is None
        assert result['cash_net'] is None


def test_source_coverage_is_explicit_for_both_series():
    for field in ('coverage_complete', 'clock_coverage_complete'):
        result = analyze(records(**{field: False}))
        assert not result['testable'] and result['p_positive_R'] == 1


def test_final_sample_thresholds_are_independent():
    counts = [10]*30
    counts[-1] = 9
    result = analyze(records(completed_counts=counts))
    assert not result['sample_gates']['completed_at_least_300']
    r = [1.]*19+[0.]*11
    result = analyze(records(r, completed_counts=[16]*19+[0]*11))
    assert result['completed_trades'] == 304 and result['active_days'] == 19
    assert not result['testable']
    result = analyze(records([1.]*29, completed_counts=[11]*29))
    assert result['completed_trades'] == 319 and not result['testable']
    assert result['complete_nonoverlapping_blocks'] == 9


def test_twenty_active_days_include_all_ten_known_zero_days():
    result = analyze(records([1.]*20+[0.]*10, completed_counts=[15]*20+[0]*10))
    assert result['testable'] and result['active_days'] == 20 and result['calendar_days'] == 30
    assert result['daily_mean_R'] == 20/30
    assert result['stability']['five_day_block_sum_R'] == [5., 5., 5., 5., 0., 0.]
    assert not result['stability']['conditional_stability_pass']


def test_seed_is_full_sha_and_paired_difference_keeps_correlation():
    r = [float(i+1) for i in range(30)]
    value = records(r, clock_daily_R=[x-1 for x in r])
    first, second = analyze(value), analyze(value)
    assert first == second
    assert first['seed_integer_decimal'] == str(int(SHA, 16))
    assert first['rng'] == 'numpy.PCG64' and first['quantile_method'] == 'linear'
    assert first['paired_candidate_clock_resamples'] is True
    assert first['ci95_daily_mean_excess_R'] == [1., 1.]
    assert first['ci95_daily_mean_R'][0] < first['ci95_daily_mean_R'][1]


def test_undefined_pf_remains_null_and_is_not_a_mean_return_gate():
    result = analyze(records())
    assert result['profit_factor_observed'] is None and result['profit_factor_ci95'] is None
    loss = [2.]+[0.]*29
    result = analyze(records([-1.]+[1.]*29, daily_gain_R=[1.]*30, daily_loss_R=loss))
    assert result['testable'] and result['profit_factor_observed'] == 15
    assert result['undefined_pf_draws'] > 0 and result['profit_factor_ci95'] is None
    assert result['ci95_daily_mean_R'] is not None
    assert json.loads(json.dumps(result, allow_nan=False))['profit_factor_ci95'] is None


def test_pf_below_one_point_one_can_pass_partial_statistics_without_profit_claim():
    result = analyze(records(daily_gain_R=[30.]*30, daily_loss_R=[29.]*30))
    assert 1 < result['profit_factor_observed'] < 1.1
    assert result['undefined_pf_draws'] == 0 and result['profit_factor_ci95'] is not None
    claims = {key: Claim(.00001, True) for key in CLAIM_IDS}
    family = holm12(claims)
    adjusted = family['claims'][CLAIM_IDS[0]]['holm_p_value']
    gate = final_statistical_gate(result, holm_positive_p=adjusted, holm_excess_p=adjusted)
    assert gate['conditional_statistical_pass'] and not gate['profit_qualified']
    assert gate['PF_1_1_is_acceptance_gate'] is False and gate['cash_net'] is None
    assert not gate['costs_verified'] and not gate['execution_verified']
    assert 'MARKED_TO_QUOTE_DRAWDOWN' in gate['remaining_external_gates']


def test_development_is_required_before_inference_by_default():
    result = analyze_daily(records(), declaration_sha256=SHA)
    assert not result['testable'] and result['p_positive_R'] == 1
    assert result['draws'] == 0


def test_declaration_and_eligibility_validation():
    for bad in ('', 'a'*63, 'A'*64, None):
        with pytest.raises(ValueError): analyze_daily(records(), declaration_sha256=bad)
    with pytest.raises(ValueError): analyze_daily(records(), declaration_sha256=SHA, development_eligible=1)


def test_stability_five_of_six_and_positive_block_concentration():
    passing = stability_gate([1.]*25+[-1.]*5)
    assert passing['conditional_stability_pass'] and passing['positive_blocks'] == 5
    assert passing['maximum_positive_block_share'] == .2
    dominant = stability_gate([10.]*5+[1.]*20+[-1.]*5)
    assert dominant['positive_blocks'] == 5 and not dominant['conditional_stability_pass']
    assert dominant['maximum_positive_block_share'] == 50/70
    boundary = stability_gate([4.]*5+[1.]*20+[-1.]*5)
    assert boundary['maximum_positive_block_share'] == .5 and boundary['conditional_stability_pass']
    assert not stability_gate([1.]*31)['conditional_stability_pass']
    assert stability_gate([1.]*31)['five_day_block_sum_R'] == []


def test_all_three_walk_forward_sums_are_declared_and_known():
    passing = development_gate(validation_sum_R=1, walk_forward_sum_R=[1, 1, -10], completed=100, unknown=0)
    assert passing['conditional_eligible'] and not passing['profit_eligible']
    assert passing['cash_net'] is None
    for kwargs in ({'validation_sum_R': 0}, {'walk_forward_sum_R': [1, 0, -1]},
                   {'walk_forward_sum_R': [1, 1, None]}, {'completed': 99}, {'unknown': 1}):
        values = dict(validation_sum_R=1, walk_forward_sum_R=[1, 1, -1], completed=100, unknown=0)
        values.update(kwargs)
        assert not development_gate(**values)['conditional_eligible']
    with pytest.raises(ValueError): development_gate(validation_sum_R=1, walk_forward_sum_R=[1, 1], completed=100, unknown=0)


def test_holm_keeps_untestable_claims_in_the_twelve_member_family():
    claims = dict.fromkeys(CLAIM_IDS, Claim())
    claims[CLAIM_IDS[0]] = Claim(.005, True)
    claims[CLAIM_IDS[1]] = Claim(0, False)
    result = holm12(claims)
    first = result['claims'][CLAIM_IDS[0]]
    assert result['family_size'] == 12 and result['alpha'] == ALPHA
    assert first['holm_p_value'] == .06 and not first['reject_at_0_05']
    assert result['claims'][CLAIM_IDS[1]]['effective_p_value'] == 1
    assert result['claims'][CLAIM_IDS[1]]['holm_p_value'] == 1


def test_holm_exact_catalogue_and_monotone_adjustment():
    claims = {key: Claim((i+1)/10000, True) for i, key in enumerate(CLAIM_IDS)}
    result = holm12(claims)
    values = [result['claims'][key]['holm_p_value'] for key in CLAIM_IDS]
    assert values == sorted(values) and values[0] == pytest.approx(.0012)
    with pytest.raises(ValueError): holm12(dict(list(claims.items())[:-1]))
    renamed = dict(claims)
    renamed['undeclared'] = renamed.pop(CLAIM_IDS[-1])
    with pytest.raises(ValueError): holm12(renamed)
    for bad in (-.1, 1.1, float('nan'), True):
        with pytest.raises(ValueError): Claim(bad, True)
    with pytest.raises(ValueError): Claim(None, True)


def test_final_statistical_gate_requires_both_adjusted_claims():
    # Closed-form positive inference avoids making this a rerun of any market study.
    result = analyze(records())
    assert not final_statistical_gate(result, holm_positive_p=.001, holm_excess_p=.051)['conditional_statistical_pass']
    with pytest.raises(ValueError): final_statistical_gate(result, holm_positive_p=1.1, holm_excess_p=.01)


def test_extra_day_is_not_discarded_to_make_six_profitable_blocks():
    result = analyze(records([1.]*31))
    assert result['calendar_days'] == 31 and result['testable']
    assert not result['stability']['conditional_stability_pass']


def test_aggregation_overflow_is_rejected():
    with pytest.raises(ValueError): analyze(records([1e308]*30))
