"""Post-selection descriptive intervals on reused DEVELOPMENT only.

This diagnostic is not a preregistered acceptance test, does not change selection,
and never reads final prices or compares a censored CLOCK to an observed policy.
Low-sample moving-block intervals cannot establish rare-tail or cash-profit safety.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_multiframe_quote_study as study

BLOCK, DRAWS = 3, 99_999


def intervals(daily_r, daily_gain, daily_loss, *, seed, draws=DRAWS):
    vectors = [np.asarray(values, dtype=np.float64) for values in (daily_r, daily_gain, daily_loss)]
    r, gain, loss = vectors
    if (r.ndim != 1 or len(r) < BLOCK or any(v.shape != r.shape for v in vectors) or
            any(not np.isfinite(v).all() for v in vectors) or (gain < 0).any() or (loss < 0).any() or
            not np.allclose(gain-loss, r, rtol=1e-10, atol=1e-12) or
            type(seed) is not int or seed < 0 or type(draws) is not int or draws < 1):
        raise ValueError('Known consistent daily vectors and explicit positive draw count required')
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(r)
    means, pf = np.empty(draws), np.full(draws, np.nan)
    for first in range(0, draws, 4096):
        last = min(draws, first+4096)
        starts = rng.integers(0, n-BLOCK+1, size=(last-first, (n+BLOCK-1)//BLOCK))
        index = (starts[:, :, None]+np.arange(BLOCK)).reshape(last-first, -1)[:, :n]
        means[first:last] = r[index].mean(axis=1)
        sampled_gain, sampled_loss = gain[index].sum(axis=1), loss[index].sum(axis=1)
        np.divide(sampled_gain, sampled_loss, out=pf[first:last], where=sampled_loss > 0)
    if not np.isfinite(means).all():
        raise ValueError('Bootstrap mean overflow')
    undefined = int(np.count_nonzero(~np.isfinite(pf)))
    return {'CI95_daily_mean_R_descriptive': np.quantile(means, [.025, .975], method='linear').tolist(),
            'CI95_PF_descriptive': None if undefined else np.quantile(pf, [.025, .975], method='linear').tolist(),
            'undefined_PF_draws': undefined, 'block_days': BLOCK, 'draws': draws,
            'calendar_days': n, 'method': 'NONCIRCULAR_MOVING_BLOCK_LINEAR_PERCENTILES',
            'seed': str(seed), 'not_a_primary_acceptance_test': True, 'profit_qualified': False}


def main():
    _, digest = study.declaration()
    study.check(study.OUT/'selection.json')
    selection_sha = study.sha(study.OUT/'selection.json')
    selected = study.read(study.OUT/'selection.json')
    if selected != study.selection_from_development(digest):
        raise ValueError('Development selection evidence changed')
    output = {'created_utc': dt.datetime.now(dt.UTC).isoformat(), 'declaration_sha256': digest,
              'selection_sha256': selection_sha, 'diagnostic_registered_after_selection': True,
              'scope': 'REUSED_DEVELOPMENT_ONLY_NOT_FINAL_NOT_SELECTION_EVIDENCE',
              'simultaneous_CI': False, 'p_values_computed': False, 'clock_comparison_computed': False,
              'final_prices_opened': False, 'selection_changed': False, 'cash_net': None,
              'costs_verified': False, 'execution_verified': False, 'profit_qualified': False,
              'source_pins': {}, 'results': {}, 'safety': {k: False for k in study.SAFETY_FIELDS}}
    for symbol in study.SYMBOLS:
        for policy in sorted(study.POLICY_IDS):
            days = []
            for stage in ('WF1', 'WF2', 'WF3', 'VALIDATION'):
                path = study.case_path(symbol, stage, policy, 'BASE')
                case = study.validated_case(path, digest)
                if case['summary']['remaining_censored'] or case['summary']['unknown']:
                    raise ValueError('Unknown candidate development blocks this diagnostic')
                days += case['daily']
                output['source_pins'][str(path.relative_to(ROOT))] = study.sha(path)
            timestamps = [d['day_start_ms'] for d in days]
            if len(days) != 90 or any(b-a != 86_400_000 for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError('Exact disjoint consecutive ninety-day development calendar required')
            result = intervals([d['observed_sum_R'] for d in days],
                               [d['observed_gain_R'] for d in days],
                               [d['observed_loss_R'] for d in days], seed=int(digest, 16))
            result['completed_trades'] = sum(d['completed'] for d in days)
            result['active_days'] = sum(d['completed'] > 0 for d in days)
            output['results'][symbol+'/'+policy] = result
    if study.sha(study.OUT/'selection.json') != selection_sha:
        raise ValueError('Diagnostic must never alter selection')
    output['diagnostic_code_sha256'] = study.sha(Path(__file__))
    path = study.OUT/'development_uncertainty_posthoc.json'
    with path.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output['results'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
