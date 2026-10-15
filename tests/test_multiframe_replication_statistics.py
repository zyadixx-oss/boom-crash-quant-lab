"""Synthetic only: no file-backed price, strategy, broker or network fixtures."""
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
import math

import numpy as np
import pytest

from independent_profit_research.statistics import Claim
from independent_multiframe_research import replication_statistics as m

SHA = '1'*64


def calendar(n=219, excluded=(36, 128, 129)):
    first = date(2024, 5, 27)
    return m.Calendar(tuple((first+timedelta(days=i)).isoformat() for i in range(n)),
                      tuple(i not in excluded for i in range(n)))


def case(cal, value=1., *, counts=None, gain_loss=None, unknown=None, usable=True, coverage=True):
    values = [value]*len(cal.utc_days) if isinstance(value, (int, float)) else value
    counts = [1]*len(values) if counts is None else counts
    unknown = [0]*len(values) if unknown is None else unknown
    fields = dict(daily_R=tuple(x if valid else None for x, valid in zip(values, cal.valid_day_mask)),
                  completed_counts=tuple(x if valid else 0 for x, valid in zip(counts, cal.valid_day_mask)),
                  unknown_counts=tuple(x if valid else 0 for x, valid in zip(unknown, cal.valid_day_mask)),
                  whole_case_usable=usable, coverage_complete=coverage)
    if gain_loss is not None:
        fields['daily_gain_R'] = tuple(gain_loss[0] if valid else None for valid in cal.valid_day_mask)
        fields['daily_loss_R'] = tuple(gain_loss[1] if valid else None for valid in cal.valid_day_mask)
    return m.DailyCase(**fields)


def scenarios(cal, candidate=None, sparse=None, dense=None):
    record = m.ScenarioRecords(candidate or case(cal, gain_loss=(1., 0.)), sparse or case(cal, 0), dense or case(cal, -.1))
    return dict.fromkeys(m.SCENARIOS, record)


def analyze(cal=None, *, records=None, draws=79, sha=SHA):
    cal = calendar() if cal is None else cal
    return m.analyze_policy(cal, scenarios(cal) if records is None else records,
                            policy_id=m.POLICIES[0], declaration_sha256=sha, draws=draws)


@pytest.fixture(scope='module')
def constant_full_draws():
    cal = calendar()
    # PF1.05 is allowed: strict positive-R uncertainty replaces a PF floor.
    return analyze(cal, records=scenarios(cal, candidate=case(cal, .05, gain_loss=(1.05, 1))),
                   draws=m.DEFAULT_DRAWS)


def test_full_default_draws_known_constant_and_cash_invariant(constant_full_draws):
    result = constant_full_draws
    assert result['eligible_calendar_days'] == 216 and result['calendar_slots'] == 219
    assert result['requested_draws'] == 99999 and result['production_draws_match']
    for endpoint in result['endpoints'].values():
        assert endpoint['testable'] and endpoint['all_ci95_lower_positive']
        assert endpoint['p_value'] == 1e-5
        assert endpoint['all_stability_pass']
        for scenario in m.SCENARIOS:
            assert set(endpoint['per_scenario_block'][scenario]) == {'3', '7'}
    assert result['case_summaries']['BASE']['candidate']['profit_factor_observed'] == pytest.approx(1.05)
    for s in m.SCENARIOS:
        for b in ('3', '7'):
            assert result['candidate_profit_factor_bootstrap'][s][b]['undefined_draws'] == 0
            assert result['candidate_profit_factor_bootstrap'][s][b]['ci95'] == pytest.approx([1.05, 1.05])
    assert result['cash_net'] is None
    assert all(result[k] is False for k in m.SAFETY)
    assert not result['qualified_cash'] and not result['goal_profit_achieved']
    json.dumps(result, allow_nan=False)


def test_blocks_keep_calendar_masks_and_never_compress():
    cal = calendar(20, (5, 12))
    assert m.eligible_block_starts(cal, 3) == (0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 17)
    assert m.eligible_block_starts(cal, 7) == (13,)
    for block in m.BLOCK_DAYS:
        for start in m.eligible_block_starts(cal, block):
            assert all(cal.valid_day_mask[start:start+block])


def test_inactive_valid_zeros_are_in_mean_not_only_active_days():
    cal = calendar()
    counts = [1 if i % 2 == 0 else 0 for i in range(219)]
    r = [1. if n else 0. for n in counts]
    result = analyze(cal, records=scenarios(cal, candidate=case(cal, r, counts=counts)))
    c = result['case_summaries']['BASE']['candidate']
    assert c['observed_sum_R'] == c['completed']
    assert c['daily_mean_R'] == pytest.approx(c['completed']/216)
    assert c['daily_mean_R'] < 1


def test_paired_indices_produce_exact_constant_excess_and_determinism():
    cal = calendar()
    base = [i % 5 - 2. for i in range(219)]
    record = scenarios(cal, candidate=case(cal, [x+2 for x in base]), sparse=case(cal, base),
                       dense=case(cal, [x-1 for x in base]))
    a, b = analyze(cal, records=record), analyze(cal, records=record)
    assert a == b
    for s in m.SCENARIOS:
        for block in ('3', '7'):
            assert a['endpoints'][m.ENDPOINTS[1]]['per_scenario_block'][s][block]['ci95'] == [2., 2.]
            assert a['endpoints'][m.ENDPOINTS[2]]['per_scenario_block'][s][block]['ci95'] == [3., 3.]
    assert a['bootstrap_streams']['3']['resample_indices_sha256'] != a['bootstrap_streams']['7']['resample_indices_sha256']
    assert m.bootstrap_seed(SHA, 3) != m.bootstrap_seed('2'*64, 3)


@pytest.mark.parametrize('role', ['candidate', 'sparse', 'dense'])
def test_unknown_censors_whole_case_never_prefix_or_other_endpoints(role):
    cal = calendar()
    u = [0]*219; u[200] = 1
    records = scenarios(cal)
    records['BASE'] = replace(records['BASE'], **{role: case(cal, 100., unknown=u)})
    result = analyze(cal, records=records)
    info = result['case_summaries']['BASE'][role]
    assert info['whole_case_censored'] and info['observed_sum_R'] is None and info['daily_mean_R'] is None
    affected = list(m.ENDPOINTS) if role == 'candidate' else [m.ENDPOINTS[1 if role == 'sparse' else 2]]
    for key in affected:
        assert not result['endpoints'][key]['testable'] and result['endpoints'][key]['p_value'] == 1
    if role != 'candidate':
        assert result['endpoints'][m.ENDPOINTS[0]]['testable']
    if role == 'dense':
        assert result['endpoints'][m.ENDPOINTS[1]]['testable']


@pytest.mark.parametrize('field', ['coverage_complete', 'whole_case_usable'])
def test_unusable_case_is_censored_without_unknown(field):
    cal = calendar(); records = scenarios(cal)
    records['STRESS'] = replace(records['STRESS'], candidate=replace(records['STRESS'].candidate, **{field: False}))
    result = analyze(cal, records=records)
    assert all(not x['testable'] and x['p_value'] == 1 for x in result['endpoints'].values())
    assert result['case_summaries']['STRESS']['candidate']['observed_sum_R'] is None


def test_scenarios_counts_cannot_pool_to_sample_floor():
    cal = calendar(); counts = [1 if i % 3 == 0 else 0 for i in range(219)]
    records = scenarios(cal, candidate=case(cal, [float(bool(c)) for c in counts], counts=counts))
    result = analyze(cal, records=records)
    assert result['case_summaries']['BASE']['candidate']['completed'] < 100
    assert sum(result['case_summaries'][s]['candidate']['completed'] for s in m.SCENARIOS) >= 100
    assert all(not x['testable'] for x in result['endpoints'].values())


def test_no_best_scenario_or_block_selection():
    cal = calendar(); records = scenarios(cal)
    records['STRESS'] = replace(records['STRESS'], candidate=case(cal, -1.))
    result = analyze(cal, records=records)
    endpoint = result['endpoints'][m.ENDPOINTS[0]]
    assert endpoint['testable'] and endpoint['p_value'] == 1
    assert not endpoint['all_ci95_lower_positive'] and not endpoint['all_stability_pass']
    fixed = [endpoint['per_scenario_block'][s][str(b)]['p_value'] for s in m.SCENARIOS for b in m.BLOCK_DAYS]
    assert endpoint['p_value'] == max(fixed)


def test_pf_undefined_draw_count_and_no_infinite_ci():
    result = analyze()
    assert set(result['candidate_profit_factor_bootstrap']['BASE']) == {'3', '7'}
    for blocks in result['candidate_profit_factor_bootstrap'].values():
        for value in blocks.values():
            assert value == {'ci95': None, 'undefined_draws': 79}
    assert result['case_summaries']['BASE']['candidate']['profit_factor_observed'] is None


def test_pf_some_undefined_draws_must_censor_entire_interval():
    cal = calendar(); c = case(cal, gain_loss=(1., 0.))
    r, loss = list(c.daily_R), list(c.daily_loss_R)
    r[10], loss[10] = 0., 1.
    c = replace(c, daily_R=r, daily_loss_R=loss)
    result = analyze(cal, records=scenarios(cal, candidate=c), draws=999)
    for blocks in result['candidate_profit_factor_bootstrap'].values():
        for value in blocks.values():
            assert 0 < value['undefined_draws'] < 999 and value['ci95'] is None


def test_no_pf_vectors_is_not_fabricated_pf():
    result = analyze()
    assert result['candidate_profit_factor_bootstrap']['BASE']['3']['undefined_draws'] == 79
    cal = calendar(); no_pf = analyze(cal, records=scenarios(cal, candidate=case(cal, 1., gain_loss=None)))
    assert no_pf['candidate_profit_factor_bootstrap'] == {'BASE': {}, 'STRESS': {}}


def test_fixed_six_thirds_boundaries_masks_and_concentration():
    cal = calendar(); v = tuple(1. if valid else None for valid in cal.valid_day_mask)
    result = m.stability_gate(cal, v)
    assert result['six_boundaries'] == ((0,36),(36,73),(73,109),(109,146),(146,182),(182,219))
    assert result['third_boundaries'] == ((0,73),(73,146),(146,219))
    assert sum(result['six_sum_R']) == 216 and result['pass']
    concentrated = tuple((1000. if i < 36 else 1.) if valid else None for i, valid in enumerate(cal.valid_day_mask))
    assert not m.stability_gate(cal, concentrated)['pass']
    missing = list(v); missing[0] = None
    assert m.stability_gate(cal, missing)['six_sum_R'] is None
    last_era_loss = tuple((1. if i < 182 else -10.) if valid else None
                         for i, valid in enumerate(cal.valid_day_mask))
    s = m.stability_gate(cal, last_era_loss)
    assert s['gates']['five_of_six_fixed_blocks_positive']
    assert not s['gates']['all_three_fixed_thirds_positive'] and not s['pass']


def test_holm33_placeholders_exact_family_and_floor():
    claims = {key: Claim(1e-5, True) for key in m.NEW_CLAIM_IDS}
    result = m.holm33(claims)
    assert len(result['claims']) == 33 and result['old_unmeasured_placeholders'] == 24
    for key in m.OLD_CLAIM_IDS:
        assert result['claims'][key]['effective_p_value'] == 1 and not result['claims'][key]['testable']
    for key in m.NEW_CLAIM_IDS:
        assert result['claims'][key]['holm_p_value'] == pytest.approx(33e-5)
    claims[m.NEW_CLAIM_IDS[0]] = Claim(0., False)
    assert m.holm33(claims)['claims'][m.NEW_CLAIM_IDS[0]]['effective_p_value'] == 1
    assert 'NOT_GLOBAL' in result['scope']


def test_partial_gate_uses_sparse_and_no_cash_promotion(constant_full_draws):
    analyses = dict.fromkeys(m.POLICIES)
    analyses[m.POLICIES[0]] = constant_full_draws
    holm = m.holm33(m.claims_from_analyses(analyses))
    gate = m.conditional_statistical_gate(constant_full_draws, holm)
    assert gate['conditional_statistical_pass'] and not gate['qualified_cash'] and not gate['goal_profit_achieved']
    assert gate['cash_net'] is None and not gate['dense_superiority_implied']
    bad = json.loads(json.dumps(holm)); bad['claims'][m.NEW_CLAIM_IDS[0]]['holm_p_value'] = 0
    with pytest.raises(ValueError, match='adjustments'):
        m.conditional_statistical_gate(constant_full_draws, bad)


def test_dense_censored_does_not_block_fixed_primary_gate(constant_full_draws):
    # Endpoint-wise censoring is separate from absolute/sparse qualification.
    from copy import deepcopy
    result = deepcopy(constant_full_draws)
    dense = result['endpoints'][m.ENDPOINTS[2]]
    dense.update(testable=False, p_value=1., all_ci95_lower_positive=False, all_stability_pass=False)
    analyses = {m.POLICIES[0]: result, m.POLICIES[1]: None, m.POLICIES[2]: None}
    holm = m.holm33(m.claims_from_analyses(analyses))
    assert m.conditional_statistical_gate(result, holm)['conditional_statistical_pass']
    assert holm['claims'][f'V3/CRASH500/{m.POLICIES[0]}/{m.ENDPOINTS[2]}']['effective_p_value'] == 1


def test_safety_constants_cannot_be_mutated():
    with pytest.raises(TypeError): m.SAFETY['LIVE_TRADING'] = True


def test_smaller_synthetic_draws_cannot_pass_production_gate():
    analysis = analyze(draws=999)
    holm = m.holm33(m.claims_from_analyses({m.POLICIES[0]: analysis, m.POLICIES[1]: None, m.POLICIES[2]: None}))
    gate = m.conditional_statistical_gate(analysis, holm)
    assert not gate['gates']['fixed_production_draws_99999'] and not gate['conditional_statistical_pass']
    assert analysis['p_floor'] == .001


@pytest.mark.parametrize('bad', [0, -1, True, 1.5, 100000])
def test_strict_draw_counts(bad):
    with pytest.raises(ValueError): analyze(draws=bad)


@pytest.mark.parametrize('bad', ['', 'a'*63, 'A'*64, 'g'*64, None])
def test_full_canonical_declaration_seed_required(bad):
    with pytest.raises(ValueError): analyze(sha=bad)


def test_strict_inputs_masks_counts_coverage_and_shapes():
    with pytest.raises(ValueError): m.Calendar(('2024-01-01', '2024-01-03'), (True, True))
    with pytest.raises(ValueError): m.Calendar(('2024-01-01',), (1,))
    with pytest.raises(ValueError): m.Calendar(('20240101',), (True,))
    cal = calendar(); c = case(cal)
    for changes in ({'completed_counts': (True,)*219}, {'coverage_complete': 1}, {'daily_R': (np.nan,)*219},
                    {'daily_gain_R': (1.,)*219}, {'unknown_counts': (-1,)*219}, {'daily_R': ('1',)*219}):
        with pytest.raises(ValueError): replace(c, **changes)
    wrong_mask = list(c.daily_R); wrong_mask[36] = 0.
    with pytest.raises(ValueError, match='Excluded'): analyze(cal, records=scenarios(cal, candidate=replace(c, daily_R=wrong_mask)))
    missing = list(c.daily_R); missing[0] = None
    with pytest.raises(ValueError, match='missing'): analyze(cal, records=scenarios(cal, candidate=replace(c, daily_R=missing)))
    with pytest.raises(ValueError): m.eligible_block_starts(cal, 5)
    with pytest.raises(ValueError): m.analyze_policy(cal, {'BASE': scenarios(cal)['BASE']}, policy_id=m.POLICIES[0], declaration_sha256=SHA)
    with pytest.raises(ValueError): m.holm33({})


def test_numeric_overflow_and_gain_loss_reconciliation():
    cal = calendar()
    with pytest.raises(ValueError, match='overflow'): analyze(cal, records=scenarios(cal, candidate=case(cal, 1e308)))
    with pytest.raises(ValueError, match='reconcile'): analyze(cal, records=scenarios(cal, candidate=case(cal, 1., gain_loss=(1., 1.))))
    records = scenarios(cal, candidate=case(cal, 1e306), sparse=case(cal, -1e306))
    with pytest.raises(ValueError): analyze(cal, records=records)


def test_missing_valid_block_length_cannot_choose_other_configuration():
    cal = calendar(219, tuple(i for i in range(219) if i % 6 == 5))
    result = analyze(cal)
    assert m.eligible_block_starts(cal, 3) and not m.eligible_block_starts(cal, 7)
    assert all(not x['testable'] for x in result['endpoints'].values())
    assert result['bootstrap_streams'] == {}


def test_isolated_valid_day_stays_in_mean_but_cannot_disappear_from_bootstrap():
    cal = calendar(219, (1,))
    r = [10000.]+[1.]*218
    result = analyze(cal, records=scenarios(cal, candidate=case(cal, r)))
    assert result['case_summaries']['BASE']['candidate']['daily_mean_R'] == pytest.approx(10217/218)
    for block in ('3', '7'):
        assert result['bootstrap_calendar_support'][block]['uncovered_eligible_day_indices'] == [0]
    for endpoint in result['endpoints'].values():
        assert not endpoint['testable'] and endpoint['p_value'] == 1
        assert endpoint['per_scenario_block'] == {'BASE': {}, 'STRESS': {}}
    assert result['bootstrap_streams'] == {}


def test_short_mask_delimited_span_has_no_seven_day_support():
    cal = calendar(219, (1, 8))
    result = analyze(cal)
    assert result['bootstrap_calendar_support']['3']['uncovered_eligible_day_indices'] == [0]
    assert result['bootstrap_calendar_support']['7']['uncovered_eligible_day_indices'] == [0, 2, 3, 4, 5, 6, 7]
    assert all(not value['testable'] for value in result['endpoints'].values())


def test_sparse_count_floor_is_separate_from_candidate_and_active_days():
    cal = calendar(); valid = [i for i, flag in enumerate(cal.valid_day_mask) if flag]
    for active, per_day in ((79, 1), (40, 2)):
        selected = {valid[k*len(valid)//active] for k in range(active)}
        counts = [per_day if i in selected else 0 for i in range(219)]
        result = analyze(cal, records=scenarios(cal, sparse=case(cal, 0., counts=counts)))
        assert result['endpoints'][m.ENDPOINTS[0]]['testable']
        assert not result['endpoints'][m.ENDPOINTS[1]]['testable']
        assert result['endpoints'][m.ENDPOINTS[1]]['p_value'] == 1


def test_independent_reference_reconstructs_all_paired_bootstrap_math():
    cal = calendar(); n = sum(cal.valid_day_mask); draws = 257
    raw_candidate = [math.sin(i/11)+.25 for i in range(219)]
    raw_sparse = [.15*math.cos(i/7) for i in range(219)]
    raw_dense = [.05*(-1)**i for i in range(219)]
    records = {}
    for scenario, factor in (('BASE', 1.), ('STRESS', .6)):
        values = [factor*r for r in raw_candidate]
        gains = tuple(max(r, 0.) if valid else None for r, valid in zip(values, cal.valid_day_mask))
        losses = tuple(max(-r, 0.) if valid else None for r, valid in zip(values, cal.valid_day_mask))
        candidate = replace(case(cal, values), daily_gain_R=gains, daily_loss_R=losses)
        records[scenario] = m.ScenarioRecords(candidate, case(cal, raw_sparse), case(cal, raw_dense))
    result = analyze(cal, records=records, draws=draws)
    raw_p_values = {endpoint: [] for endpoint in m.ENDPOINTS}
    for block in (3, 7):
        # Independently reconstruct starts/labels/index draws, not private helpers.
        starts = [i for i in range(219-block+1) if all(cal.valid_day_mask[i+j] for j in range(block))]
        label = SHA+'\nMFR_V3/PAIRED_BOOTSTRAP/BLOCK_DAYS='+str(block)
        seed = int(hashlib.sha256(label.encode('ascii')).hexdigest(), 16)
        rng = np.random.Generator(np.random.PCG64(seed))
        choices = rng.integers(0, len(starts), size=(draws, math.ceil(n/block)))
        indices = [[i for choice in row for i in range(starts[choice], starts[choice]+block)][:n] for row in choices]
        for scenario in m.SCENARIOS:
            c, sparse, dense = records[scenario].candidate, records[scenario].sparse, records[scenario].dense
            vectors = (c.daily_R, tuple(None if not flag else a-b for a,b,flag in zip(c.daily_R,sparse.daily_R,cal.valid_day_mask)),
                       tuple(None if not flag else a-b for a,b,flag in zip(c.daily_R,dense.daily_R,cal.valid_day_mask)))
            for endpoint, vector in zip(m.ENDPOINTS, vectors):
                observed = math.fsum(x for x, valid in zip(vector, cal.valid_day_mask) if valid)/n
                boot = [math.fsum(vector[i] for i in row)/n for row in indices]
                centered = [math.fsum(vector[i]-observed for i in row)/n for row in indices]
                p = (1+sum(x >= observed for x in centered))/(draws+1)
                ordered = sorted(boot)
                def percentile(q):
                    position = (draws-1)*q; low = math.floor(position); high = math.ceil(position)
                    return ordered[low]+(ordered[high]-ordered[low])*(position-low)
                actual = result['endpoints'][endpoint]['per_scenario_block'][scenario][str(block)]
                assert actual['p_value'] == p
                assert actual['ci95'] == pytest.approx([percentile(.025), percentile(.975)], abs=1e-12)
                raw_p_values[endpoint].append(p)
            pf = [math.fsum(c.daily_gain_R[i] for i in row)/math.fsum(c.daily_loss_R[i] for i in row) for row in indices]
            actual_pf = result['candidate_profit_factor_bootstrap'][scenario][str(block)]
            assert actual_pf['undefined_draws'] == 0
            assert actual_pf['ci95'] == pytest.approx(np.quantile(pf, [.025, .975], method='linear'), abs=1e-12)
    for endpoint in m.ENDPOINTS:
        assert result['endpoints'][endpoint]['p_value'] == max(raw_p_values[endpoint])


def test_unselected_policies_and_mixed_declarations():
    assert all(not c.testable for c in m.claims_from_analyses(dict.fromkeys(m.POLICIES)).values())
    a, b = analyze(), analyze(sha='2'*64)
    b['policy_id'] = m.POLICIES[1]
    with pytest.raises(ValueError, match='different frozen'):
        m.claims_from_analyses({m.POLICIES[0]: a, m.POLICIES[1]: b, m.POLICIES[2]: None})
