"""Fixed conditional daily inference; no market reads or verified cash-profit claim.

All daily vectors contain observed completed-trade sums, assigned to EXIT UTC
day by the caller. Empty covered calendar days are zero. UNKNOWN is a separate
count and blocks inference; an unknown trade is never assigned a zero payoff.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Mapping, Sequence

import numpy as np


BLOCK_DAYS, DRAWS, ALPHA = 3, 99999, .05
SYMBOLS = ('BOOM500', 'CRASH500')
POLICIES = ('SIDE_LEAD', 'QUOTE_CLOCK_ACCELERATION', 'POST_JUMP_CARRY_30S')
ENDPOINTS = ('POSITIVE_DAILY_R', 'EXCESS_DAILY_R_VS_CLOCK')
CLAIM_IDS = tuple(f'{s}/{p}/{e}' for s in SYMBOLS for p in POLICIES for e in ENDPOINTS)


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label+' must be finite numeric')
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise ValueError(label+' must be finite numeric') from None
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError(label+' outside finite permitted range')
    return value


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label+' must be a nonnegative integer')
    return value


def _sum(values):
    try:
        result = math.fsum(values)
    except OverflowError:
        raise ValueError('Daily aggregation overflow') from None
    if not math.isfinite(result):
        raise ValueError('Daily aggregation must remain finite')
    return result


@dataclass(frozen=True)
class DailyRecords:
    daily_R: Sequence[float]
    daily_quote: Sequence[float]
    clock_daily_R: Sequence[float]
    completed_counts: Sequence[int]
    unknown_counts: Sequence[int]
    clock_unknown_counts: Sequence[int]
    coverage_complete: bool = False
    clock_coverage_complete: bool = False
    daily_gain_R: Sequence[float] | None = None
    daily_loss_R: Sequence[float] | None = None

    def __post_init__(self):
        for key in ('daily_R', 'daily_quote', 'clock_daily_R'):
            object.__setattr__(self, key, tuple(_number(x, key) for x in getattr(self, key)))
        for key in ('completed_counts', 'unknown_counts', 'clock_unknown_counts'):
            object.__setattr__(self, key, tuple(_count(x, key) for x in getattr(self, key)))
        n = len(self.daily_R)
        if n == 0 or any(len(getattr(self, key)) != n for key in
                         ('daily_quote', 'clock_daily_R', 'completed_counts', 'unknown_counts', 'clock_unknown_counts')):
            raise ValueError('Explicit nonempty equal-length covered-calendar vectors required')
        if type(self.coverage_complete) is not bool or type(self.clock_coverage_complete) is not bool:
            raise ValueError('Coverage assertions must be explicit booleans')
        if any(count == 0 and (r != 0 or quote != 0)
               for count, r, quote in zip(self.completed_counts, self.daily_R, self.daily_quote)):
            raise ValueError('Observed payoff without a completed trade')
        if (self.daily_gain_R is None) != (self.daily_loss_R is None):
            raise ValueError('Both daily gain/loss vectors are required for observed PF')
        if self.daily_gain_R is not None:
            for key in ('daily_gain_R', 'daily_loss_R'):
                values = tuple(_number(x, key, nonnegative=True) for x in getattr(self, key))
                if len(values) != n:
                    raise ValueError('Daily gain/loss lengths must match the full calendar')
                object.__setattr__(self, key, values)
            for count, gain, loss, r in zip(self.completed_counts, self.daily_gain_R, self.daily_loss_R, self.daily_R):
                if count == 0 and (gain != 0 or loss != 0) or not math.isclose(gain-loss, r, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Daily gain/loss accounting does not match observed R')


def _fingerprint(records):
    values = {key: getattr(records, key) for key in records.__dataclass_fields__}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stability_gate(daily_R):
    values = tuple(_number(x, 'daily_R') for x in daily_R)
    totals = [_sum(values[i:i+5]) for i in range(0, 30, 5)] if len(values) == 30 else []
    positive = [x for x in totals if x > 0]
    positive_sum = _sum(positive)
    share = max(positive)/positive_sum if positive_sum > 0 else None
    gates = {'exactly_30_days_six_fixed_blocks': len(values) == 30,
             'at_least_five_positive_blocks': len(positive) >= 5,
             'maximum_positive_block_share_at_most_half': share is not None and share <= .5}
    return {'conditional_stability_pass': all(gates.values()), 'gates': gates,
            'five_day_block_sum_R': totals, 'positive_blocks': len(positive),
            'maximum_positive_block_share': share,
            'concentration_denominator': 'SUM_OF_POSITIVE_FIVE_DAY_BLOCK_TOTALS'}


def development_gate(*, validation_sum_R, walk_forward_sum_R, completed, unknown):
    completed, unknown = _count(completed, 'completed'), _count(unknown, 'unknown')
    if len(walk_forward_sum_R) != 3:
        raise ValueError('Exactly three declared walk-forward fold sums required')
    validation = None if validation_sum_R is None else _number(validation_sum_R, 'validation_sum_R')
    folds = [None if x is None else _number(x, 'walk_forward_sum_R') for x in walk_forward_sum_R]
    gates = {'validation_positive': validation is not None and validation > 0,
             'two_of_three_walk_forward_positive': sum(x is not None and x > 0 for x in folds) >= 2,
             'future_development_completed_at_least_100': completed >= 100,
             'no_unknown_development_trades': unknown == 0,
             'all_walk_forward_sums_known': all(x is not None for x in folds)}
    return {'conditional_eligible': all(gates.values()), 'gates': gates,
            'validation_sum_R': validation, 'walk_forward_sum_R': folds,
            'profit_eligible': False, 'cash_net': None, 'costs_verified': False}


def _bootstrap(records, seed, means):
    n = len(records.daily_R)
    block_count = (n+BLOCK_DAYS-1)//BLOCK_DAYS
    rng = np.random.Generator(np.random.PCG64(seed))
    r = np.asarray(records.daily_R, dtype=np.float64)
    quote = np.asarray(records.daily_quote, dtype=np.float64)
    excess = r-np.asarray(records.clock_daily_R, dtype=np.float64)
    centered_r, centered_excess = r-means['R'], excess-means['excess_R']
    draws = {key: np.empty(DRAWS, dtype=np.float64) for key in ('R', 'quote', 'excess_R')}
    positive_tail = excess_tail = 0
    pf_draws = np.full(DRAWS, np.nan) if records.daily_gain_R is not None else None
    gain = None if pf_draws is None else np.asarray(records.daily_gain_R, dtype=np.float64)
    loss = None if pf_draws is None else np.asarray(records.daily_loss_R, dtype=np.float64)
    try:
        with np.errstate(over='raise', invalid='raise'):
            for first in range(0, DRAWS, 4096):
                last = min(DRAWS, first+4096)
                starts = rng.integers(0, n-BLOCK_DAYS+1, size=(last-first, block_count), dtype=np.int64)
                indices = (starts[:, :, None]+np.arange(BLOCK_DAYS)).reshape(last-first, -1)[:, :n]
                draws['R'][first:last] = r[indices].mean(axis=1)
                draws['quote'][first:last] = quote[indices].mean(axis=1)
                draws['excess_R'][first:last] = excess[indices].mean(axis=1)
                positive_tail += int(np.count_nonzero(centered_r[indices].mean(axis=1) >= means['R']))
                excess_tail += int(np.count_nonzero(centered_excess[indices].mean(axis=1) >= means['excess_R']))
                if pf_draws is not None:
                    sampled_gain, sampled_loss = gain[indices].sum(axis=1), loss[indices].sum(axis=1)
                    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                        np.divide(sampled_gain, sampled_loss, out=pf_draws[first:last], where=sampled_loss > 0)
    except FloatingPointError:
        raise ValueError('Bootstrap arithmetic must remain finite') from None
    if any(not np.isfinite(values).all() for values in draws.values()):
        raise ValueError('Bootstrap mean distribution must remain finite')
    result = {f'ci95_daily_mean_{key}': np.quantile(values, [.025, .975], method='linear').tolist()
              for key, values in draws.items()}
    result.update(p_positive_R=(1+positive_tail)/(DRAWS+1),
                  p_excess_R_vs_clock=(1+excess_tail)/(DRAWS+1))
    undefined = None if pf_draws is None else int(np.count_nonzero(~np.isfinite(pf_draws)))
    result.update(profit_factor_ci95=(np.quantile(pf_draws, [.025, .975], method='linear').tolist()
                                     if pf_draws is not None and undefined == 0 else None),
                  undefined_pf_draws=undefined)
    return result


def analyze_daily(records: DailyRecords, *, declaration_sha256: str, development_eligible: bool = False):
    """Observed-only metrics and fixed daily block inference, never cash net.

    Coverage and pre-final development eligibility must be asserted explicitly.
    Any candidate/baseline UNKNOWN, inadequate sample or coverage failure keeps
    both claims in the Holm family at p=1, with no inferential interval.
    """
    if not isinstance(records, DailyRecords):
        raise ValueError('DailyRecords required')
    if not isinstance(declaration_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', declaration_sha256):
        raise ValueError('Canonical full declaration SHA256 required')
    if type(development_eligible) is not bool:
        raise ValueError('Development eligibility must be explicit boolean')
    n, completed = len(records.daily_R), sum(records.completed_counts)
    unknown, clock_unknown = sum(records.unknown_counts), sum(records.clock_unknown_counts)
    active = sum(x > 0 for x in records.completed_counts)
    totals = {key: _sum(values) for key, values in (('R', records.daily_R), ('quote', records.daily_quote))}
    excess = [_number(a-b, 'daily excess R') for a, b in zip(records.daily_R, records.clock_daily_R)]
    means = {key: total/n for key, total in totals.items()}
    means['excess_R'] = _sum(excess)/n
    gates = {'completed_at_least_300': completed >= 300, 'active_days_at_least_20': active >= 20,
             'calendar_days_at_least_30': n >= 30, 'ten_complete_nonoverlapping_three_day_blocks': n//3 >= 10,
             'candidate_coverage_complete': records.coverage_complete,
             'clock_coverage_complete': records.clock_coverage_complete,
             'no_candidate_unknown': unknown == 0, 'no_clock_unknown': clock_unknown == 0,
             'development_eligible_before_final': development_eligible}
    testable = all(gates.values())
    gain = None if records.daily_gain_R is None else _sum(records.daily_gain_R)
    loss = None if records.daily_loss_R is None else _sum(records.daily_loss_R)
    pf = gain/loss if loss is not None and loss > 0 else None
    if pf is not None and not math.isfinite(pf): pf = None
    result = {'scope': 'CONDITIONAL_QUOTE_PAPER_DAILY_OBSERVED_COMPLETES_ONLY',
              'status': 'TESTABLE_CONDITIONAL' if testable else 'INCONCLUSIVE_CONDITIONAL',
              'testable': testable, 'sample_gates': gates, 'daily_input_sha256': _fingerprint(records),
              'declaration_sha256': declaration_sha256, 'seed_integer_decimal': str(int(declaration_sha256, 16)),
              'rng': 'numpy.PCG64', 'numpy_version': np.__version__,
              'block_days': BLOCK_DAYS, 'draws': DRAWS if testable else 0, 'prescribed_draws': DRAWS,
              'block_sampling': 'NONCIRCULAR_MOVING_STARTS_WITH_REPLACEMENT_CONCATENATE_TRUNCATE_TO_N',
              'quantile_method': 'linear', 'paired_candidate_clock_resamples': True,
              'calendar_days': n, 'complete_nonoverlapping_blocks': n//3, 'active_days': active,
              'completed_trades': completed, 'unknown_trades': unknown, 'clock_unknown_trades': clock_unknown,
              'daily_attribution': 'EXIT_UTC_DAY', 'observed_sum_R': totals['R'], 'observed_sum_quote': totals['quote'],
              'daily_mean_R': means['R'], 'daily_mean_quote': means['quote'],
              'daily_mean_excess_R_vs_clock': means['excess_R'],
              'observed_mean_R_per_completed_trade': totals['R']/completed if completed else None,
              'observed_mean_quote_per_completed_trade': totals['quote']/completed if completed else None,
              'median_trade_R': None, 'median_trade_R_scope': 'NOT_RECOVERABLE_FROM_DAILY_SUMS',
              'profit_factor_observed': pf, 'profit_factor_ci95': None, 'undefined_pf_draws': None,
              'ci95_daily_mean_R': None, 'ci95_daily_mean_quote': None, 'ci95_daily_mean_excess_R': None,
              'p_positive_R': 1., 'p_excess_R_vs_clock': 1., 'stability': stability_gate(records.daily_R),
              'cash_net': None, 'costs_verified': False, 'execution_verified': False, 'profit_qualified': False,
              'ci95_is_simultaneous': False,
              'assumption': 'Weakly stationary dependent daily process approximated by fixed3day blocks; no rare-tail guarantee.',
              'live_trading': False, 'ready_for_live': False, 'live_allowed': False, 'opened_trades': False,
              'LIVE_TRADING': False, 'READY_FOR_LIVE': False, 'LIVE_ALLOWED': False, 'OPENED_TRADES': False}
    if testable:
        result.update(_bootstrap(records, int(declaration_sha256, 16), means))
    return result


@dataclass(frozen=True)
class Claim:
    p_value: float | None = None
    testable: bool = False

    def __post_init__(self):
        if type(self.testable) is not bool or self.testable and self.p_value is None:
            raise ValueError('Testable claim requires a finite p-value')
        if self.p_value is not None:
            p = _number(self.p_value, 'p_value', nonnegative=True)
            if p > 1: raise ValueError('p_value exceeds1')
            object.__setattr__(self, 'p_value', p)


def holm12(claims: Mapping[str, Claim]):
    """Require the entire predeclared six-policy/two-endpoint family, including p1 placeholders."""
    if not isinstance(claims, Mapping) or set(claims) != set(CLAIM_IDS) or any(not isinstance(c, Claim) for c in claims.values()):
        raise ValueError('Exactly the twelve named declared claims are required')
    effective = {key: (claim.p_value if claim.testable else 1.) for key, claim in claims.items()}
    adjusted, running = {}, 0.
    for rank, key in enumerate(sorted(CLAIM_IDS, key=lambda key: (effective[key], key))):
        running = max(running, min(1., (12-rank)*effective[key]))
        adjusted[key] = running
    return {'family_size': 12, 'alpha': ALPHA, 'method': 'HOLM_STEP_DOWN_ALL_TWELVE',
            'claims': {key: {'raw_p_value': claims[key].p_value, 'effective_p_value': effective[key],
                             'holm_p_value': adjusted[key], 'testable': claims[key].testable,
                             'reject_at_0_05': claims[key].testable and adjusted[key] <= ALPHA}
                       for key in CLAIM_IDS}}


def final_statistical_gate(analysis, *, holm_positive_p, holm_excess_p):
    """Partial statistical gate only; external stress/DD/source/cost/fill evidence remains required."""
    adjusted = [_number(x, 'Holm p', nonnegative=True) for x in (holm_positive_p, holm_excess_p)]
    if any(x > 1 for x in adjusted): raise ValueError('Holm p exceeds1')
    ci, excess_ci = analysis['ci95_daily_mean_R'], analysis['ci95_daily_mean_excess_R']
    gates = {'conditional_inference_testable': analysis['testable'] is True,
             'observed_R_mean_and_sum_positive': analysis['daily_mean_R'] > 0 and analysis['observed_sum_R'] > 0,
             'daily_R_ci95_lower_positive': ci is not None and ci[0] > 0,
             'paired_clock_excess_ci95_lower_positive': excess_ci is not None and excess_ci[0] > 0,
             'holm_both_claims_pass': all(x <= ALPHA for x in adjusted),
             'five_of_six_blocks_and_concentration_pass': analysis['stability']['conditional_stability_pass'] is True}
    return {'conditional_statistical_pass': all(gates.values()), 'gates': gates,
            'profit_qualified': False, 'cash_net': None, 'costs_verified': False, 'execution_verified': False,
            'remaining_external_gates': ['MARKED_TO_QUOTE_DRAWDOWN', 'DECLARED_STRESS_POSITIVE_AND_DRAWDOWN',
                                         'SOURCE_AND_REPRODUCTION_AUDIT', 'HISTORICAL_CASH_COST_AND_EXECUTION_EVIDENCE'],
            'PF_1_1_is_acceptance_gate': False}
