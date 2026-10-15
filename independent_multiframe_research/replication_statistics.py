"""Pure conditional replication inference on explicit UTC daily records.

No market, account or file access. Pilot exclusions are masks, never compressed
calendar days or zero-payoff observations. UNKNOWN censors the entire case.
The separate old studies and their Claim semantics remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from numbers import Real
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from independent_profit_research.statistics import Claim, CLAIM_IDS as V1_CLAIM_IDS

POLICIES = ('CRT_SWEEP_RECLAIM', 'SQUEEZE_BREAKOUT', 'PULLBACK_CONTINUATION')
SCENARIOS = ('BASE', 'STRESS')
ENDPOINTS = ('POSITIVE_DAILY_R', 'EXCESS_DAILY_R_VS_SPARSE_CLOCK',
             'EXCESS_DAILY_R_VS_DENSE_CLOCK_DIAGNOSTIC')
BLOCK_DAYS = (3, 7)
DEFAULT_DRAWS = 99999
SAFETY = MappingProxyType(dict.fromkeys(('LIVE_TRADING', 'READY_FOR_LIVE', 'LIVE_ALLOWED', 'OPENED_TRADES',
                       'live_trading', 'ready_for_live', 'live_allowed', 'opened_trades'), False))
NEW_CLAIM_IDS = tuple(f'V3/CRASH500/{p}/{e}' for p in POLICIES for e in ENDPOINTS)
OLD_CLAIM_IDS = tuple('V1/'+key for key in V1_CLAIM_IDS) + tuple(
    f'V2/{symbol}/{policy}/{endpoint}' for symbol in ('BOOM500', 'CRASH500')
    for policy in POLICIES for endpoint in ('POSITIVE_DAILY_R', 'EXCESS_DAILY_R_VS_CLOCK'))
_MAX_COUNT = 2**53-1


def _number(value, label, *, nonnegative=False, nullable=False):
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label+' must be finite numeric')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(label+' must be finite numeric') from None
    if not math.isfinite(result) or nonnegative and result < 0:
        raise ValueError(label+' must be finite and in range')
    return result


def _count(value, label):
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError(label+' must be a nonnegative exact safe integer')
    return value


def _sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError):
        raise ValueError('Daily aggregation overflow') from None
    if not math.isfinite(result):
        raise ValueError('Daily aggregation overflow')
    return result


def _difference(a, b):
    return _number(a-b, 'Daily difference')


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Calendar:
    utc_days: Sequence[str]
    valid_day_mask: Sequence[bool]

    def __post_init__(self):
        days, mask = tuple(self.utc_days), tuple(self.valid_day_mask)
        if not days or len(days) != len(mask) or any(type(x) is not bool for x in mask):
            raise ValueError('Explicit equal nonempty UTC dates and boolean mask required')
        parsed = []
        for text in days:
            if type(text) is not str:
                raise ValueError('Canonical UTC date strings required')
            try:
                stamp = date.fromisoformat(text)
            except ValueError:
                raise ValueError('Canonical UTC date strings required') from None
            if stamp.isoformat() != text:
                raise ValueError('Canonical UTC date strings required')
            parsed.append(stamp)
        if any(b-a != timedelta(days=1) for a, b in zip(parsed, parsed[1:])):
            raise ValueError('UTC calendar must remain consecutive; never compress exclusions')
        object.__setattr__(self, 'utc_days', days)
        object.__setattr__(self, 'valid_day_mask', mask)


@dataclass(frozen=True)
class DailyCase:
    daily_R: Sequence[float | None]
    completed_counts: Sequence[int]
    unknown_counts: Sequence[int]
    coverage_complete: bool
    whole_case_usable: bool
    daily_gain_R: Sequence[float | None] | None = None
    daily_loss_R: Sequence[float | None] | None = None

    def __post_init__(self):
        if type(self.coverage_complete) is not bool or type(self.whole_case_usable) is not bool:
            raise ValueError('Explicit boolean case coverage/usability required')
        r = tuple(_number(x, 'daily_R', nullable=True) for x in self.daily_R)
        completed = tuple(_count(x, 'completed_counts') for x in self.completed_counts)
        unknown = tuple(_count(x, 'unknown_counts') for x in self.unknown_counts)
        if not r or len(completed) != len(r) or len(unknown) != len(r):
            raise ValueError('Equal nonempty case vectors required')
        if sum(completed) > _MAX_COUNT or sum(unknown) > _MAX_COUNT:
            raise ValueError('Count aggregation outside exact range')
        if (self.daily_gain_R is None) != (self.daily_loss_R is None):
            raise ValueError('Both daily gain and loss vectors are required for PF')
        for name, values in (('daily_R', r), ('completed_counts', completed), ('unknown_counts', unknown)):
            object.__setattr__(self, name, values)
        if self.daily_gain_R is not None:
            for name in ('daily_gain_R', 'daily_loss_R'):
                values = tuple(_number(x, name, nullable=True, nonnegative=True) for x in getattr(self, name))
                if len(values) != len(r):
                    raise ValueError('Gain/loss calendar length differs')
                object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ScenarioRecords:
    candidate: DailyCase
    sparse: DailyCase
    dense: DailyCase

    def __post_init__(self):
        if any(not isinstance(getattr(self, role), DailyCase) for role in ('candidate', 'sparse', 'dense')):
            raise ValueError('Three explicit DailyCase records required')


def eligible_block_starts(calendar: Calendar, block_days: int):
    if not isinstance(calendar, Calendar) or type(block_days) is not int or block_days not in BLOCK_DAYS:
        raise ValueError('Only the fixed3/7day calendar block configurations are permitted')
    return tuple(i for i in range(len(calendar.utc_days)-block_days+1)
                 if all(calendar.valid_day_mask[i:i+block_days]))


def _block_support(calendar, block_days):
    starts = eligible_block_starts(calendar, block_days)
    covered = {i for start in starts for i in range(start, start+block_days)}
    missing = [i for i, valid in enumerate(calendar.valid_day_mask) if valid and i not in covered]
    return {'allowed_start_indices': list(starts), 'uncovered_eligible_day_indices': missing,
            'every_eligible_day_has_resampling_support': bool(starts) and not missing}


def bootstrap_seed(declaration_sha256: str, block_days: int):
    if type(declaration_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', declaration_sha256):
        raise ValueError('Full canonical premeasurement declaration SHA256 required')
    if type(block_days) is not int or block_days not in BLOCK_DAYS:
        raise ValueError('Fixed bootstrap block label required')
    label = f'MFR_V3/PAIRED_BOOTSTRAP/BLOCK_DAYS={block_days}'
    return int(hashlib.sha256((declaration_sha256+'\n'+label).encode('ascii')).hexdigest(), 16)


def _validate_case(calendar, case):
    if len(case.daily_R) != len(calendar.utc_days):
        raise ValueError('Case must align with the entire consecutive calendar')
    unknown_total = sum(case.unknown_counts)
    for i, valid in enumerate(calendar.valid_day_mask):
        r, count, unknown = case.daily_R[i], case.completed_counts[i], case.unknown_counts[i]
        gain = None if case.daily_gain_R is None else case.daily_gain_R[i]
        loss = None if case.daily_loss_R is None else case.daily_loss_R[i]
        if not valid:
            if r is not None or count or unknown or gain is not None or loss is not None:
                raise ValueError('Excluded pilot days require null payoff and zero counts, never zero returns')
            continue
        if r is not None and count == 0 and r != 0:
            raise ValueError('Observed payoff without completed trades')
        if count == 0 and any(x is not None and x != 0 for x in (gain, loss)):
            raise ValueError('Observed gain/loss without completed trades')
        if (gain is None) != (loss is None):
            raise ValueError('Paired gain/loss cells required')
        if gain is not None and r is not None and not math.isclose(_difference(gain, loss), r, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Daily gain-loss must reconcile to daily R')
        if case.coverage_complete and case.whole_case_usable and not unknown_total:
            if r is None or case.daily_gain_R is not None and gain is None:
                raise ValueError('Claimed usable complete case has missing in-scope payoff')


def _boundaries(n, groups):
    return tuple((j*n//groups, (j+1)*n//groups) for j in range(groups))


def _cell_sums(calendar, values, groups):
    return [_sum(values[i] for i in range(a, b) if calendar.valid_day_mask[i])
            for a, b in _boundaries(len(calendar.utc_days), groups)]


def stability_gate(calendar: Calendar, values: Sequence[float | None]):
    values = tuple(values)
    if len(values) != len(calendar.utc_days):
        raise ValueError('Stability requires whole calendar vectors')
    for x, valid in zip(values, calendar.valid_day_mask):
        if not valid and x is not None:
            raise ValueError('Excluded stability payoff must be null')
        if x is not None:
            _number(x, 'stability R')
    known = all(x is not None for x, valid in zip(values, calendar.valid_day_mask) if valid)
    six = _cell_sums(calendar, values, 6) if known else None
    thirds = _cell_sums(calendar, values, 3) if known else None
    positive = [] if six is None else [x for x in six if x > 0]
    positive_sum = _sum(positive)
    share = max(positive)/positive_sum if positive_sum > 0 else None
    gates = {'all_in_scope_payoffs_known': known,
             'five_of_six_fixed_blocks_positive': len(positive) >= 5,
             'all_three_fixed_thirds_positive': thirds is not None and all(x > 0 for x in thirds),
             'maximum_positive_block_share_at_most_half': share is not None and share <= .5}
    return {'pass': all(gates.values()), 'gates': gates,
            'six_boundaries': _boundaries(len(values), 6), 'third_boundaries': _boundaries(len(values), 3),
            'six_sum_R': six, 'third_sum_R': thirds, 'positive_block_share': share}


def _case_info(calendar, case, role):
    _validate_case(calendar, case)
    completed = sum(case.completed_counts)
    unknown = sum(case.unknown_counts)
    active = sum(c > 0 for c in case.completed_counts)
    usable = case.coverage_complete and case.whole_case_usable and unknown == 0
    values = case.daily_R if usable else tuple(None for _ in case.daily_R)
    counts_six = _cell_sums(calendar, case.completed_counts, 6)
    counts_thirds = _cell_sums(calendar, case.completed_counts, 3)
    count_floor, active_floor = (80, 60) if role == 'sparse' else (100, 40)
    gates = {'whole_case_usable_known_coverage': usable, 'completed_floor': completed >= count_floor,
             'active_days_floor': active >= active_floor, 'at_least_five_trades_each_six_block': all(x >= 5 for x in counts_six)}
    if role == 'candidate':
        gates['at_least_twenty_trades_each_third'] = all(x >= 20 for x in counts_thirds)
    total = _sum(x for x, valid in zip(values, calendar.valid_day_mask) if valid) if usable else None
    eligible = sum(calendar.valid_day_mask)
    mean = total/eligible if total is not None and eligible else None
    gains = losses = pf = None
    if usable and case.daily_gain_R is not None:
        gains = _sum(x for x, valid in zip(case.daily_gain_R, calendar.valid_day_mask) if valid)
        losses = _sum(x for x, valid in zip(case.daily_loss_R, calendar.valid_day_mask) if valid)
        if losses > 0:
            pf = gains/losses
            if not math.isfinite(pf):
                pf = None
    return {'usable': usable, 'sample_pass': all(gates.values()), 'sample_gates': gates,
            'completed': completed, 'unknown': unknown, 'active_days': active,
            'completed_six_blocks': counts_six, 'completed_thirds': counts_thirds,
            'observed_sum_R': total, 'daily_mean_R': mean, 'profit_factor_observed': pf,
            'stability': stability_gate(calendar, values), 'values': values,
            'whole_case_censored': not usable}


def _quantile(values):
    with np.errstate(over='raise', invalid='raise'):
        result = np.quantile(values, [.025, .975], method='linear')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite bootstrap quantiles')
    return result.tolist()


def _bootstrap(calendar, scenarios, infos, endpoints, sha, draws):
    n = sum(calendar.valid_day_mask)
    output = {name: {s: {} for s in SCENARIOS} for name in ENDPOINTS}
    pf = {s: {} for s in SCENARIOS}
    streams = {}
    for block in BLOCK_DAYS:
        starts = np.asarray(eligible_block_starts(calendar, block), dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(bootstrap_seed(sha, block)))
        samples = {name: {s: np.empty(draws) for s in SCENARIOS} for name in ENDPOINTS if endpoints[name]['testable']}
        arrays, centers, tails, pfs = {}, {}, {}, {}
        for name in samples:
            arrays[name], centers[name], tails[name] = {}, {}, {}
            for scenario in SCENARIOS:
                values = endpoints[name]['vectors'][scenario]
                array = np.asarray([np.nan if x is None else x for x in values], dtype=np.float64)
                mean = endpoints[name]['means'][scenario]
                with np.errstate(over='raise', invalid='raise'):
                    arrays[name][scenario], centers[name][scenario] = array, array-mean
                tails[name][scenario] = 0
        for scenario in SCENARIOS:
            case = scenarios[scenario].candidate
            if endpoints[ENDPOINTS[0]]['testable'] and case.daily_gain_R is not None:
                pfs[scenario] = np.full(draws, np.nan)
        index_hash = hashlib.sha256()
        for first in range(0, draws, 1024):
            last = min(draws, first+1024)
            chosen = starts[rng.integers(0, len(starts), size=(last-first, (n+block-1)//block))]
            indices = (chosen[:, :, None]+np.arange(block)).reshape(last-first, -1)[:, :n]
            index_hash.update(indices.astype('<i8', copy=False).tobytes())
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                for name in samples:
                    for scenario in SCENARIOS:
                        samples[name][scenario][first:last] = arrays[name][scenario][indices].mean(axis=1)
                        centered = centers[name][scenario][indices].mean(axis=1)
                        tails[name][scenario] += int(np.count_nonzero(centered >= endpoints[name]['means'][scenario]))
                for scenario in pfs:
                    case = scenarios[scenario].candidate
                    gains = np.asarray([np.nan if x is None else x for x in case.daily_gain_R])[indices].sum(axis=1)
                    losses = np.asarray([np.nan if x is None else x for x in case.daily_loss_R])[indices].sum(axis=1)
                    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                        np.divide(gains, losses, out=pfs[scenario][first:last], where=losses > 0)
        for name in samples:
            for scenario in SCENARIOS:
                values = samples[name][scenario]
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite bootstrap distribution')
                output[name][scenario][str(block)] = {'ci95': _quantile(values),
                    'p_value': (1+tails[name][scenario])/(draws+1), 'draws': draws}
        for scenario in pfs:
            undefined = int(np.count_nonzero(~np.isfinite(pfs[scenario])))
            pf[scenario][str(block)] = {'ci95': _quantile(pfs[scenario]) if undefined == 0 else None,
                                       'undefined_draws': undefined}
        streams[str(block)] = {'seed_integer_decimal': str(bootstrap_seed(sha, block)),
            'allowed_start_indices': starts.tolist(), 'resample_indices_sha256': index_hash.hexdigest(),
            'draws': draws, 'sampled_eligible_days_per_draw': n,
            'shared_across_candidates_controls_and_scenarios': True}
    return output, pf, streams


def analyze_policy(calendar: Calendar, scenarios: Mapping[str, ScenarioRecords], *,
                   policy_id: str, declaration_sha256: str, draws: int = DEFAULT_DRAWS):
    """All fixed scenarios/blocks, never selecting the best configuration.

    Smaller explicit draws exist for synthetic tests; production orchestration
    must freeze and enforce DEFAULT_DRAWS. Every receipt reports actual draws.
    Statistical pass is partial: DD, source/replay and cash evidence are external.
    """
    if not isinstance(calendar, Calendar) or not isinstance(scenarios, Mapping) or set(scenarios) != set(SCENARIOS):
        raise ValueError('Explicit calendar and exactly BASE/STRESS required')
    if type(policy_id) is not str or policy_id not in POLICIES or any(not isinstance(x, ScenarioRecords) for x in scenarios.values()):
        raise ValueError('Exactly one declared Crash500 policy and complete scenario records required')
    if type(draws) is not int or not 1 <= draws <= DEFAULT_DRAWS:
        raise ValueError('Draw count must be explicit positive integer no larger than99999')
    bootstrap_seed(declaration_sha256, 3)
    infos = {s: {role: _case_info(calendar, getattr(scenarios[s], role), role)
                 for role in ('candidate', 'sparse', 'dense')} for s in SCENARIOS}
    support = {str(block): _block_support(calendar, block) for block in BLOCK_DAYS}
    blocks_available = all(x['allowed_start_indices'] for x in support.values())
    endpoints = {}
    for endpoint, baseline in zip(ENDPOINTS, (None, 'sparse', 'dense')):
        vectors, means, stability = {}, {}, {}
        gates = {'both_fixed_block_lengths_available': bool(blocks_available),
                 'all_eligible_days_supported_at_both_fixed_lengths': all(
                     x['every_eligible_day_has_resampling_support'] for x in support.values())}
        for scenario in SCENARIOS:
            candidate = infos[scenario]['candidate']
            other = None if baseline is None else infos[scenario][baseline]
            usable = candidate['usable'] and (other is None or other['usable'])
            gates[scenario+'_candidate_sample'] = candidate['sample_pass']
            if other is not None:
                gates[scenario+'_baseline_sample'] = other['sample_pass']
            values = candidate['values'] if baseline is None else tuple(
                _difference(a, b) if usable and valid else None
                for a, b, valid in zip(candidate['values'], other['values'], calendar.valid_day_mask))
            vectors[scenario] = values
            eligible = sum(calendar.valid_day_mask)
            means[scenario] = _sum(x for x, valid in zip(values, calendar.valid_day_mask) if valid)/eligible if usable and eligible else None
            stability[scenario] = stability_gate(calendar, values)
        endpoints[endpoint] = {'testable': all(gates.values()), 'sample_gates': gates,
            'vectors': vectors, 'means': means, 'stability': stability, 'p_value': 1.,
            'per_scenario_block': {s: {} for s in SCENARIOS}, 'all_ci95_lower_positive': False}
    try:
        if any(x['testable'] for x in endpoints.values()):
            distributions, pf, streams = _bootstrap(calendar, scenarios, infos, endpoints, declaration_sha256, draws)
            for name, endpoint in endpoints.items():
                if endpoint['testable']:
                    endpoint['per_scenario_block'] = distributions[name]
                    fixed = [distributions[name][s][str(b)] for s in SCENARIOS for b in BLOCK_DAYS]
                    endpoint['p_value'] = max(x['p_value'] for x in fixed)
                    endpoint['all_ci95_lower_positive'] = all(x['ci95'][0] > 0 for x in fixed)
        else:
            pf, streams = {s: {} for s in SCENARIOS}, {}
    except (FloatingPointError, OverflowError):
        raise ValueError('Bootstrap arithmetic overflow') from None
    for endpoint in endpoints.values():
        endpoint.pop('vectors')
        endpoint['all_stability_pass'] = all(x['pass'] for x in endpoint['stability'].values())
    for scenario in SCENARIOS:
        for info in infos[scenario].values():
            info.pop('values')
    return {'policy_id': policy_id, 'symbol': 'CRASH500', 'declaration_sha256': declaration_sha256,
        'input_sha256': _fingerprint({'calendar': calendar.__dict__, 'scenarios': {
            s: {role: getattr(scenarios[s], role).__dict__ for role in ('candidate', 'sparse', 'dense')} for s in SCENARIOS}}),
        'calendar_slots': len(calendar.utc_days), 'eligible_calendar_days': sum(calendar.valid_day_mask),
        'excluded_day_indices': [i for i, x in enumerate(calendar.valid_day_mask) if not x],
        'daily_estimand': 'MEAN_R_PER_ELIGIBLE_UTC_CALENDAR_DAY_INCLUDING_COVERED_IDLE_ZERO_DAYS',
        'daily_attribution': 'EXIT_UTC_DAY', 'case_summaries': infos, 'endpoints': endpoints,
        'candidate_profit_factor_bootstrap': pf, 'bootstrap_streams': streams,
        'bootstrap_calendar_support': support,
        'prescribed_draws': DEFAULT_DRAWS, 'requested_draws': draws, 'production_draws_match': draws == DEFAULT_DRAWS,
        'block_days': BLOCK_DAYS, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'quantile_method': 'linear',
        'p_floor': 1/(draws+1), 'IUT': 'MAX_P_ACROSS_BOTH_SCENARIOS_AND_FIXED3_7DAY_BLOCKS',
        'ci95_is_simultaneous': False, 'PF_floor_is_gate': False, 'cash_net': None,
        'qualified_cash': False, 'profit_qualified': False, 'goal_profit_achieved': False,
        'costs_verified': False, 'execution_verified': False, **SAFETY}


def holm33(claims: Mapping[str, Claim]):
    if not isinstance(claims, Mapping) or set(claims) != set(NEW_CLAIM_IDS) or any(not isinstance(x, Claim) for x in claims.values()):
        raise ValueError('Exactly nine new fixed policy/endpoint Claim objects required')
    values = {key: Claim() for key in OLD_CLAIM_IDS}
    values.update(claims)
    effective = {key: value.p_value if value.testable else 1. for key, value in values.items()}
    output, running = {}, 0.
    for rank, key in enumerate(sorted(values, key=lambda key: (effective[key], key))):
        running = max(running, min(1., (33-rank)*effective[key]))
        output[key] = {'raw_p_value': values[key].p_value, 'effective_p_value': effective[key],
                       'holm_p_value': running, 'testable': values[key].testable,
                       'reject_at_0_05': values[key].testable and running <= .05}
    return {'family_size': 33, 'old_unmeasured_placeholders': 24, 'alpha': .05, 'claims': output,
            'scope': 'FIXED_THREE_GENERATIONS_NOT_GLOBAL_ALL_ADAPTIVE_RESEARCH_FWER'}


def claims_from_analyses(analyses: Mapping[str, Mapping | None]):
    if not isinstance(analyses, Mapping) or set(analyses) != set(POLICIES):
        raise ValueError('All three declared policy analyses or explicit None placeholders required')
    claims, declarations = {}, set()
    for policy, analysis in analyses.items():
        if analysis is not None and (not isinstance(analysis, Mapping) or analysis.get('policy_id') != policy):
            raise ValueError('Policy analysis identity differs')
        if analysis is not None:
            bootstrap_seed(analysis['declaration_sha256'], 3)
            declarations.add(analysis['declaration_sha256'])
        for endpoint in ENDPOINTS:
            value = None if analysis is None else analysis['endpoints'][endpoint]
            claims[f'V3/CRASH500/{policy}/{endpoint}'] = Claim() if value is None else Claim(value['p_value'], value['testable'])
    if len(declarations) > 1:
        raise ValueError('Cannot pool claims from different frozen declarations')
    return claims


def conditional_statistical_gate(analysis, holm):
    """A partial sparse-comparator gate; dense UNKNOWN never promotes cash."""
    if holm.get('family_size') != 33 or set(holm.get('claims', {})) != set(OLD_CLAIM_IDS+NEW_CLAIM_IDS):
        raise ValueError('Full fixed Holm33 output required')
    policy = analysis['policy_id']
    if policy not in POLICIES:
        raise ValueError('Declared policy required')
    # Verify supplied adjustment against the full family, including p1 history.
    for key in OLD_CLAIM_IDS:
        if holm['claims'][key]['raw_p_value'] is not None or holm['claims'][key]['testable'] is not False:
            raise ValueError('Old unmeasured claims must remain placeholders')
    recomputed = holm33({key: Claim(holm['claims'][key]['raw_p_value'], holm['claims'][key]['testable'])
                         for key in NEW_CLAIM_IDS})
    if recomputed['claims'] != holm['claims']:
        raise ValueError('Holm33 adjustments do not match the complete family')
    gates = {'fixed_production_draws_99999': (
        analysis.get('production_draws_match') is True and analysis.get('requested_draws') == DEFAULT_DRAWS
        and analysis.get('prescribed_draws') == DEFAULT_DRAWS
        and set(analysis.get('bootstrap_streams', {})) == {'3', '7'}
        and all(stream.get('draws') == DEFAULT_DRAWS for stream in analysis['bootstrap_streams'].values()))}
    for name in ENDPOINTS[:2]:
        endpoint = analysis['endpoints'][name]
        claim = holm['claims'][f'V3/CRASH500/{policy}/{name}']
        if claim['raw_p_value'] != endpoint['p_value'] or claim['testable'] != endpoint['testable']:
            raise ValueError('Holm claim differs from the corresponding policy analysis')
        adjusted = _number(claim['holm_p_value'], 'Holm p', nonnegative=True)
        if adjusted > 1:
            raise ValueError('Holm p outside probability range')
        gates[name] = (endpoint['testable'] is True and endpoint['all_ci95_lower_positive'] is True
                       and endpoint['all_stability_pass'] is True and adjusted <= .05
                       and all(mean is not None and mean > 0 for mean in endpoint['means'].values()))
    return {'conditional_statistical_pass': all(gates.values()), 'gates': gates,
        'claim_scope': 'CONDITIONAL_REPLICATION_VS_PREDECLARED_SPARSE_CLOCK',
        'dense_superiority_implied': False, 'PF_floor_is_gate': False,
        'remaining_external_gates': ['DEVELOPMENT_SELECTION_BEFORE_FINAL', 'MARKED_DRAWDOWN',
                                   'SOURCE_AND_ACTUAL_REPRODUCTION_AUDIT', 'HISTORICAL_COST_AND_FILL_EVIDENCE'],
        'qualified_cash': False, 'profit_qualified': False, 'goal_profit_achieved': False,
        'cash_net': None, 'costs_verified': False, 'execution_verified': False, **SAFETY}
