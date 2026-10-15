"""Finite, preregistered multiframe quote study; no broker or account access.

The original study and its source pins stay unchanged. Reused development data
are labelled as such. Failed development candidates never load final prices.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_independent_profit_study as original
from independent_profit_research.data import SAFETY_FIELDS
from independent_profit_research.execution import ExecutionConfig
from independent_profit_research.statistics import (
    Claim, DailyRecords, analyze_daily, development_gate, final_statistical_gate,
)
from independent_profit_research.tape import QuoteTape
from independent_multiframe_research.bars import build_frames
from independent_multiframe_research.policies import CLOCK_POLICY_ID, POLICY_IDS, generate_signals, clock_signals
from independent_multiframe_research.runner import evaluate_signals

OUT = ROOT / 'work/independent_multiframe_study_20261014'
DOCS = ROOT / 'docs/independent_multiframe_research_20261014'
STAGES = dict(original.STAGES)
SYMBOLS = dict(original.SYMBOLS)
SOURCE_FILES = [
    'independent_multiframe_research/' + name + '.py'
    for name in ('bars', 'policies', 'runner')
] + ['independent_multiframe_research/__init__.py',
     'independent_profit_research/__init__.py', 'scripts/run_multiframe_quote_study.py']
ENDPOINTS = ('POSITIVE_DAILY_R', 'EXCESS_DAILY_R_VS_CLOCK')


def sha(path):
    return original.sha(path)


def read(path):
    return json.loads(path.read_text())


def create(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    if path.is_relative_to(OUT):
        index_path = OUT / 'artifact_index.json'
        index = read(index_path) if index_path.exists() else {}
        key = str(path.relative_to(OUT))
        if key in index:
            raise ValueError('Receipt was already registered')
        index[key] = sha(path)
        temporary = OUT / 'artifact_index.new'
        with temporary.open('x') as stream:
            json.dump(index, stream, indent=2, sort_keys=True)
            stream.write('\n')
        temporary.replace(index_path)


def check(path):
    if read(OUT/'artifact_index.json').get(str(path.relative_to(OUT))) != sha(path):
        raise ValueError('Changed or unregistered receipt: ' + str(path))


def claim_ids():
    return tuple(f'{s}/{p}/{e}' for s in SYMBOLS for p in sorted(POLICY_IDS) for e in ENDPOINTS)


def holm24(claims):
    expected = set(claim_ids())
    if set(claims) != expected or any(not isinstance(v, Claim) for v in claims.values()):
        raise ValueError('Exactly twelve declared generation-2 claims are required')
    # Unmeasured generation-1 claims are retained, not silently dropped.
    values = {f'V2/{key}': value for key, value in claims.items()}
    values.update({f'V1/{key}': Claim() for key in original.CLAIM_IDS})
    effective = {key: value.p_value if value.testable else 1.0 for key, value in values.items()}
    ordered = sorted(values, key=lambda key: (effective[key], key))
    result, running = {}, 0.0
    for rank, key in enumerate(ordered):
        value = values[key]
        running = max(running, min(1.0, (24-rank)*effective[key]))
        result[key] = {'raw_p_value': value.p_value, 'effective_p_value': effective[key], 'holm_p_value': running,
                       'testable': value.testable}
    return {'family_size': 24, 'claims': result,
            'scope': 'TWO_DECLARED_GENERATIONS_NOT_ALL_ADAPTIVE_RESEARCH'}


def freeze(validation):
    original.safe()
    _, old_decl_sha = original.declaration()
    original.check_receipt(original.OUT/'final_summary.json')
    if read(original.OUT/'final_summary.json')['results']:
        raise ValueError('Generation-1 final was measured; this holdout cannot be reused')
    if (OUT/'declaration.json').exists():
        raise ValueError('This generation is already frozen')
    report = read(validation)
    if report.get('status') != 'PASS' or report.get('market_strategy_computations') != 0:
        raise ValueError('Actual passing premeasurement checks required')
    pins = {name: sha(ROOT/name) for name in SOURCE_FILES}
    acquisition = read(original.DATA/'acquisition_state.json')
    if acquisition['status'] != 'COMPLETED':
        raise ValueError('Fixed source acquisition has not completed')
    fits = {}
    for symbol in SYMBOLS:
        for end in (30, 50, 70, 90):
            path = original.OUT/'fits'/f'{symbol}_TRAIN0_{end}.json'
            if not path.is_file():
                raise ValueError('Only existing frozen TRAIN fits can be reused')
            original.check_receipt(path)
            original.unit_fit(symbol, end, old_decl_sha)
            fits[str(path.relative_to(ROOT))] = sha(path)
    frozen = DOCS/'PROTOCOL_FROZEN.ar.md'
    with frozen.open('x') as stream:
        stream.write((DOCS/'PROTOCOL_DRAFT.ar.md').read_text().replace(
            'DRAFT / NOT FROZEN / NOT TESTED', 'FROZEN BEFORE V2 MEASUREMENT / NOT TESTED'
        ).replace('مسودة بروتوكول', 'بروتوكول مجمد'))
    obj = {'id': 'multiframe-quote-20261014', 'schema_version': 1,
           'frozen_utc': dt.datetime.now(dt.UTC).isoformat(),
           'original_declaration_sha256': sha(original.OUT/'declaration.json'),
           'original_final_summary_sha256': sha(original.OUT/'final_summary.json'),
           'completed_acquisition_state_sha256': sha(original.DATA/'acquisition_state.json'),
           'acquisition_declaration_sha256': sha(original.DATA/'acquisition_declaration.json'),
           'source_pins': pins, 'reused_training_fit_pins': fits,
           'protocol_path': str(frozen.relative_to(ROOT)), 'protocol_sha256': sha(frozen),
           'premeasurement_validation_path': str(validation.relative_to(ROOT)),
           'premeasurement_validation_sha256': sha(validation),
           'python': platform.python_version(), 'numpy': np.__version__,
           'stages': STAGES, 'symbols': SYMBOLS, 'policies': sorted(POLICY_IDS),
           'claims': claim_ids(), 'holm_family_size': 24,
           'development_previously_exposed': True, 'final_strategy_measured': False,
           'cash_net': None, 'costs_verified': False, 'execution_verified': False,
           **{key: False for key in SAFETY_FIELDS}}
    create(OUT/'declaration.json', obj)
    print('V2 FROZEN; no strategy measurement', flush=True)


def declaration():
    original.safe()
    original.declaration()
    path = OUT/'declaration.json'
    check(path)
    spec = read(path)
    if sha(original.OUT/'declaration.json') != spec['original_declaration_sha256']:
        raise ValueError('Original declaration changed')
    if sha(original.OUT/'final_summary.json') != spec['original_final_summary_sha256']:
        raise ValueError('Original unopened-final evidence changed')
    if sha(original.DATA/'acquisition_state.json') != spec['completed_acquisition_state_sha256']:
        raise ValueError('Completed source metadata changed')
    if sha(original.DATA/'acquisition_declaration.json') != spec['acquisition_declaration_sha256']:
        raise ValueError('Acquisition declaration changed')
    pins = dict(spec['source_pins']) | dict(spec['reused_training_fit_pins'])
    pins[spec['protocol_path']] = spec['protocol_sha256']
    pins[spec['premeasurement_validation_path']] = spec['premeasurement_validation_sha256']
    for name, pinned in pins.items():
        if sha(ROOT/name) != pinned:
            raise ValueError('Frozen generation-2 dependency changed: ' + name)
    if spec['python'] != platform.python_version() or spec['numpy'] != np.__version__:
        raise ValueError('Runtime differs from frozen generation-2 runtime')
    return spec, sha(path)


def case_path(symbol, stage, policy, scenario):
    return OUT/'cases'/f'{symbol}_{stage}_{policy}_{scenario}.json'


def validated_case(path, decl_sha):
    check(path)
    value = read(path)
    identities = [(s, t, p, c) for s in SYMBOLS for t in STAGES
                  for p in (*sorted(POLICY_IDS), 'CLOCK') for c in ('BASE', 'STRESS')
                  if path == case_path(s, t, p, c)]
    if len(identities) != 1:
        raise ValueError('Case is not a declared identity/path')
    symbol, stage, policy, scenario = identities[0]
    if any(value.get(k) != v for k, v in zip(
            ('symbol', 'stage', 'policy', 'scenario'), identities[0])):
        raise ValueError('Cached case identity differs from its requested path')
    start, end, _ = STAGES[stage]
    if (value.get('stage_start_ms') != original.START_MS+start*original.DAY_MS or
            value.get('stage_end_exclusive_ms') != original.START_MS+end*original.DAY_MS or
            value.get('ledger_path') != str((OUT/'ledgers'/(path.stem+'.json.gz')).relative_to(ROOT))):
        raise ValueError('Case ledger or calendar differs from its declared identity')
    if value['declaration_sha256'] != decl_sha or sha(ROOT/value['ledger_path']) != value['ledger_sha256']:
        raise ValueError('Cached case/ledger differs from declared evidence')
    for name, pinned in value['source_files'].items():
        if sha(ROOT/name) != pinned:
            raise ValueError('Case source changed')
    return value


def bounded_signals(signals, start_ms, end_ms):
    """Keep causal prehistory warmup, but consume no decisions beyond the stage embargo."""
    try:
        for signal in signals:
            if signal.decision_ms >= end_ms:
                break
            if signal.decision_ms >= start_ms:
                yield signal
    finally:
        close = getattr(signals, 'close', None)
        if callable(close):
            close()


def calculate(tape, frames, symbol, stage, policy, scenario, unit, decl_sha, pins):
    path = case_path(symbol, stage, policy, scenario)
    if path.exists():
        result = validated_case(path, decl_sha)
        if result['source_files'] != pins or result['U_scaled'] != unit:
            raise ValueError('Cached case has different source or fit')
        print('REUSED', path.stem, flush=True)
        return result
    start, end, _ = STAGES[stage]
    start_ms = original.START_MS + start*original.DAY_MS
    end_ms = original.START_MS + end*original.DAY_MS
    config = ExecutionConfig(250 if scenario == 'BASE' else 2000,
                             250 if scenario == 'BASE' else 2000,
                             5000, (.25 if scenario == 'BASE' else .5)*unit)
    if policy == 'CLOCK':
        signals = clock_signals(frames, tape, spread_unit=unit,
                                native_direction=SYMBOLS[symbol],
                                start_ms=start_ms+1800000, end_ms=end_ms-1800000)
    else:
        signals = generate_signals(frames, tape=tape, policy_id=policy,
                                   native_direction=SYMBOLS[symbol], spread_unit=unit)
    bounded = bounded_signals(signals, start_ms+1800000, end_ms-1800000)
    result = evaluate_signals(tape, bounded, start_ms, end_ms, unit, config,
                              policy_id=CLOCK_POLICY_ID if policy == 'CLOCK' else policy,
                              baseline=policy == 'CLOCK')
    ledger = result.pop('ledger')
    ledger_path = OUT/'ledgers'/(path.stem+'.json.gz')
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ledger, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with ledger_path.open('xb') as stream:
        stream.write(gzip.compress(payload, mtime=0))
    result.update(declaration_sha256=decl_sha, symbol=symbol, stage=stage,
                  policy=policy, scenario=scenario, U_scaled=unit, source_files=pins,
                  ledger_path=str(ledger_path.relative_to(ROOT)), ledger_sha256=sha(ledger_path),
                  development_previously_exposed=True, cash_net=None,
                  costs_verified=False, execution_verified=False, profit_qualified=False)
    create(path, result)
    print('CALCULATED', path.stem, json.dumps(result['summary'], sort_keys=True), flush=True)
    return result


def run_stage(stage):
    spec, decl_sha = declaration()
    if stage == 'FINAL':
        check(OUT/'selection.json')
        selection = read(OUT/'selection.json')
        if selection != selection_from_development(decl_sha):
            raise ValueError('Final selection differs from reproduced development evidence')
        allowed = set(selection['conditionally_eligible'])
    else:
        allowed = {s+'/'+p for s in SYMBOLS for p in POLICY_IDS}
    start, end, train_end = STAGES[stage]
    for symbol in SYMBOLS:
        policies = [p for p in sorted(POLICY_IDS) if symbol+'/'+p in allowed]
        if not policies:
            print('NOT_OPENED_DEVELOPMENT_GATE', symbol, stage, flush=True)
            continue
        # Only a previously pinned TRAIN fit is reused. No fitting on test prices.
        fit_path = original.OUT/'fits'/f'{symbol}_TRAIN0_{train_end}.json'
        original.check_receipt(fit_path)
        fit = read(fit_path)
        if (fit['uses_validation_or_final'] is not False or fit['uses_strategy_payoff'] is not False or
                fit['symbol'] != symbol or fit['train_start_day'] != 0 or
                fit['declaration_sha256'] != spec['original_declaration_sha256'] or
                fit['train_end_day_exclusive'] != train_end):
            raise ValueError('Training fit is not the exact permitted prefix')
        source = original.records(symbol, start, end)
        pins = {name: pinned for row in source for name, pinned in row['files'].items()}
        tape = QuoteTape(q for record in source for q in original.load(record).tape)
        frames = build_frames(tape, original.START_MS+start*original.DAY_MS,
                              original.START_MS+end*original.DAY_MS)
        for scenario in ('BASE', 'STRESS'):
            calculate(tape, frames, symbol, stage, 'CLOCK', scenario, fit['U_scaled'], decl_sha, pins)
            for policy in policies:
                calculate(tape, frames, symbol, stage, policy, scenario, fit['U_scaled'], decl_sha, pins)


def selection_from_development(decl_sha):
    value = {'declaration_sha256': decl_sha, 'conditionally_eligible': [], 'gates': {},
             'development_receipts': {}, 'final_strategy_measurements_before_selection': 0,
             'profit_qualified': False, 'cash_net': None, 'costs_verified': False}
    for symbol in SYMBOLS:
        for policy in sorted(POLICY_IDS):
            summaries = []
            for stage in ('WF1', 'WF2', 'WF3', 'VALIDATION'):
                path = case_path(symbol, stage, policy, 'BASE')
                case = validated_case(path, decl_sha)
                value['development_receipts'][str(path.relative_to(ROOT))] = sha(path)
                summaries.append(case['summary'])
            gate = development_gate(validation_sum_R=summaries[3]['sum_R'],
                walk_forward_sum_R=[v['sum_R'] for v in summaries[:3]],
                completed=sum(v['completed'] for v in summaries), unknown=sum(v['unknown'] for v in summaries))
            key = symbol+'/'+policy
            value['gates'][key] = gate
            if gate['conditional_eligible']:
                value['conditionally_eligible'].append(key)
    return value


def select():
    _, decl_sha = declaration()
    value = selection_from_development(decl_sha)
    create(OUT/'selection.json', value)
    print('Selection before final:', value['conditionally_eligible'], flush=True)


def summarize():
    _, decl_sha = declaration()
    check(OUT/'selection.json')
    selected = read(OUT/'selection.json')
    if selected != selection_from_development(decl_sha):
        raise ValueError('Selection evidence changed')
    claims = {key: Claim() for key in claim_ids()}
    analyses, results = {}, {}
    for key in selected['conditionally_eligible']:
        symbol, policy = key.split('/')
        base = validated_case(case_path(symbol, 'FINAL', policy, 'BASE'), decl_sha)
        clock = validated_case(case_path(symbol, 'FINAL', 'CLOCK', 'BASE'), decl_sha)
        daily, control = base['daily'], clock['daily']
        records = DailyRecords([r['observed_sum_R'] for r in daily],
            [r['observed_sum_quote'] for r in daily], [r['observed_sum_R'] for r in control],
            [r['completed'] for r in daily], [r['unknown'] for r in daily],
            [r['unknown'] for r in control], coverage_complete=not base['summary']['remaining_censored'],
            clock_coverage_complete=not clock['summary']['remaining_censored'],
            daily_gain_R=[r['observed_gain_R'] for r in daily], daily_loss_R=[r['observed_loss_R'] for r in daily])
        analysis = analyze_daily(records, declaration_sha256=decl_sha, development_eligible=True)
        analyses[key] = analysis
        claims[key+'/POSITIVE_DAILY_R'] = Claim(analysis['p_positive_R'], analysis['testable'])
        claims[key+'/EXCESS_DAILY_R_VS_CLOCK'] = Claim(analysis['p_excess_R_vs_clock'], analysis['testable'])
    holm = holm24(claims)
    for key, analysis in analyses.items():
        symbol, policy = key.split('/')
        gate = final_statistical_gate(analysis,
            holm_positive_p=holm['claims']['V2/'+key+'/POSITIVE_DAILY_R']['holm_p_value'],
            holm_excess_p=holm['claims']['V2/'+key+'/EXCESS_DAILY_R_VS_CLOCK']['holm_p_value'])
        base = validated_case(case_path(symbol, 'FINAL', policy, 'BASE'), decl_sha)['summary']
        stress = validated_case(case_path(symbol, 'FINAL', policy, 'STRESS'), decl_sha)['summary']
        external = {'base_drawdown': base['marked_drawdown_pct'] is not None and base['marked_drawdown_pct'] <= 20,
                    'stress_positive': stress['sum_R'] is not None and stress['sum_R'] > 0,
                    'stress_drawdown': stress['marked_drawdown_pct'] is not None and stress['marked_drawdown_pct'] <= 20}
        results[key] = {'analysis': analysis, 'statistical_gate': gate, 'external_gates': external,
            'conditional_paper_candidate': gate['conditional_statistical_pass'] and all(external.values()),
            'profit_qualified': False, 'cash_net': None, 'costs_verified': False, 'execution_verified': False}
    create(OUT/'final_summary.json', {'declaration_sha256': decl_sha, 'selection_sha256': sha(OUT/'selection.json'),
        'holm': holm, 'results': results, 'qualified_profit_strategies': 0,
        'unmeasured_final_claims': sorted({s+'/'+p for s in SYMBOLS for p in POLICY_IDS}-set(results)),
        'cash_net': None, 'costs_verified': False, 'execution_verified': False,
        **{key: False for key in SAFETY_FIELDS}})
    print('Summary saved; unverified cash profit remains null', flush=True)


def main():
    original.safe()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.study.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest='command', required=True)
        frozen = sub.add_parser('freeze')
        frozen.add_argument('--validation-receipt', type=Path, required=True)
        run = sub.add_parser('run')
        run.add_argument('--stage', choices=STAGES, required=True)
        sub.add_parser('select')
        sub.add_parser('summarize')
        args = parser.parse_args()
        if args.command == 'freeze':
            freeze(args.validation_receipt.resolve())
        elif args.command == 'run':
            run_stage(args.stage)
        elif args.command == 'select':
            select()
        else:
            summarize()


if __name__ == '__main__':
    main()
