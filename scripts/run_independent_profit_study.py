"""Predeclared staged quote research; never opens a broker position.

Freeze first. Development stages read only their permitted calendar. Selection
is written before final access; a failed policy's final signals are not computed.
Prices are conditional quote units, with unknown fees/cash/fills kept unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from independent_profit_research.data import PRICE_SCALE, SAFETY_FIELDS, load_day
from independent_profit_research.benchmarks import endpoint_reference
from independent_profit_research.execution import ExecutionConfig
from independent_profit_research.policies import POLICY_IDS
from independent_profit_research.runner import evaluate_tape
from independent_profit_research.statistics import (
    CLAIM_IDS, Claim, DailyRecords, analyze_daily, development_gate,
    final_statistical_gate, holm12,
)
from independent_profit_research.tape import QuoteTape
import numpy as np

DATA=ROOT/'data/independent_profit_ticks_20261013'
OUT=ROOT/'work/independent_profit_study_20261013'
DOCS=ROOT/'docs/independent_profit_research_20261013'
DAY_MS=86400000
START_MS=int(datetime.datetime(2025,1,2,tzinfo=datetime.UTC).timestamp())*1000
STAGES={'WF1':(30,50,30),'WF2':(50,70,50),'WF3':(70,90,70),
        'VALIDATION':(90,120,90),'FINAL':(120,150,90)}
SYMBOLS={'BOOM500':1,'CRASH500':-1}
SOURCE_FILES=[f'independent_profit_research/{name}.py' for name in
              ('data','execution','policies','runner','statistics','tape','benchmarks')]+[
    'scripts/verify_mt5_quote_export.py','scripts/run_independent_profit_study.py']

def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):result.update(block)
    return result.hexdigest()

def read(path):
    return json.loads(path.read_text())

def create(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    if path.is_relative_to(OUT):
        index_path=OUT/'artifact_index.json'
        index=read(index_path) if index_path.exists() else {}
        key=str(path.relative_to(OUT))
        if key in index:raise ValueError('Derived receipt is already registered')
        index[key]=sha(path)
        temporary=OUT/'artifact_index.new'
        with temporary.open('x') as stream:
            json.dump(index,stream,indent=2,sort_keys=True);stream.write('\n')
        temporary.replace(index_path)

def check_receipt(path):
    index=read(OUT/'artifact_index.json')
    if index.get(str(path.relative_to(OUT)))!=sha(path):
        raise ValueError('Derived receipt is unregistered or changed')

def safe():
    import os
    for key in SAFETY_FIELDS:
        if os.environ.get(key,'false').strip().lower()!='false':
            raise ValueError('Unsafe research environment: '+key)

def freeze(validation):
    if (OUT/'declaration.json').exists():raise ValueError('Declaration already frozen')
    evidence=read(validation)
    if evidence.get('status')!='PASS' or evidence.get('market_strategy_computations')!=0:
        raise ValueError('An actual passing premeasurement verification receipt is required')
    frozen=DOCS/'PROTOCOL_FROZEN.ar.md'
    if frozen.exists():raise ValueError('Existing protocol snapshot requires inspection')
    acquisition=DATA/'acquisition_declaration.json'
    spec=read(acquisition)
    if spec['split_day_indices']!=[0,90,120,150]:raise ValueError('Unexpected calendar')
    pins={name:sha(ROOT/name) for name in SOURCE_FILES}
    text=(DOCS/'PROTOCOL_DRAFT.ar.md').read_text()
    text=text.replace('مسودة بروتوكول','بروتوكول مجمد')
    text=text.replace('DRAFT / NOT FROZEN / NOT TESTED','FROZEN BEFORE STRATEGY MEASUREMENT / NOT TESTED')
    frozen.write_text(text)
    obj={'schema_version':1,'id':'independent-profit-20261013',
         'frozen_utc':datetime.datetime.now(datetime.UTC).isoformat(),
         'source_pins':pins,
         'acquisition_declaration_sha256':sha(acquisition),
         'protocol_path':str(frozen.relative_to(ROOT)),'protocol_sha256':sha(frozen),
         'premeasurement_validation_path':str(validation.relative_to(ROOT)),
         'premeasurement_validation_sha256':sha(validation),
         'python':platform.python_version(),'numpy':np.__version__,
         'stages':STAGES,'policies':sorted(POLICY_IDS),'symbols':SYMBOLS,
         'price_scale':PRICE_SCALE,'cash_net':None,'costs_verified':False,
         'execution_verified':False,'primary_claims':CLAIM_IDS,
         **{key:False for key in SAFETY_FIELDS}}
    create(OUT/'declaration.json',obj)
    print('Measurement protocol frozen; no strategy computation performed',flush=True)

def declaration():
    path=OUT/'declaration.json';check_receipt(path);spec=read(path)
    for name,pinned in spec['source_pins'].items():
        if sha(ROOT/name)!=pinned:raise ValueError('Frozen code changed: '+name)
    if sha(DATA/'acquisition_declaration.json')!=spec['acquisition_declaration_sha256']:
        raise ValueError('Source acquisition declaration changed')
    if sha(ROOT/spec['protocol_path'])!=spec['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    if sha(ROOT/spec['premeasurement_validation_path'])!=spec['premeasurement_validation_sha256']:
        raise ValueError('Premeasurement verification receipt changed')
    if platform.python_version()!=spec['python'] or np.__version__!=spec['numpy']:
        raise ValueError('Runtime differs from declared runtime')
    return spec,sha(path)

def records(symbol,start,end):
    state=read(DATA/'acquisition_state.json')
    requests=read(DATA/'acquisition_declaration.json')['requests']
    expected={item['ordinal']:item for item in requests}
    if len(expected)!=len(requests):raise ValueError('Duplicate declared request ordinal')
    seen=set();selected=[]
    for record in state['records']:
        ordinal=record.get('ordinal')
        if type(ordinal) is not int or ordinal in seen or ordinal not in expected:
            raise ValueError('Duplicate or unexpected acquired source record')
        seen.add(ordinal);item=expected[ordinal]
        if any(record.get(key)!=value for key,value in item.items()):
            raise ValueError('Acquired source calendar differs from fixed request')
        day=record['day_index']
        if (record['start_ms']!=START_MS+day*DAY_MS or
                record['end_exclusive_ms']!=START_MS+(day+1)*DAY_MS or
                record['partition']!=('development' if day<120 else 'sealed_final')):
            raise ValueError('Source record is outside the declared study calendar')
        if record['symbol']==symbol and start<=day<end:selected.append(record)
    values={r['day_index']:r for r in selected}
    if len(values)!=len(selected):raise ValueError('Duplicate acquired symbol/calendar day')
    if set(values)!=set(range(start,end)):
        raise ValueError(f'Required source shards not yet available: {symbol} {start}:{end}')
    output=[values[i] for i in range(start,end)]
    for record in output:
        folder=DATA/record['partition']/record['prefix']
        exact_files={str((folder/name).relative_to(ROOT)) for name in (
            record['prefix']+'_receipt.tsv',record['prefix']+'_ticks.tsv','independent_verification.json')}
        if set(record['files'])!=exact_files:
            raise ValueError('Source files do not match the fixed request paths')
        if record['verification_status']!='QUOTES_VALIDATED_COVERAGE_NOT_PROVEN':
            raise ValueError('A required source shard failed quality')
        for name,pinned in record['files'].items():
            if sha(ROOT/name)!=pinned:raise ValueError('Retained acquisition file changed')
    return output

def load(record):
    receipt=next(name for name in record['files'] if name.endswith('_receipt.tsv'))
    raw=next(name for name in record['files'] if name.endswith('_ticks.tsv'))
    return load_day(ROOT/receipt,record['files'][receipt],record['files'][raw],
                    record['symbol'],record['start_ms'],record['end_exclusive_ms'])

def unit_fit(symbol,end,decl_sha):
    path=OUT/'fits'/f'{symbol}_TRAIN0_{end}.json'
    training=records(symbol,0,end)
    expected_pins={name:pinned for record in training for name,pinned in record['files'].items()}
    if path.exists():
        check_receipt(path)
        fit=read(path)
        if (fit['declaration_sha256']!=decl_sha or fit['symbol']!=symbol or
                fit['train_start_day']!=0 or fit['train_end_day_exclusive']!=end or
                fit['uses_validation_or_final'] is not False or
                fit['uses_strategy_payoff'] is not False or fit['source_files']!=expected_pins):
            raise ValueError('Cached fit belongs to a different training calendar')
        for name,pinned in fit['source_files'].items():
            if sha(ROOT/name)!=pinned:raise ValueError('Cached fit source changed')
        return fit
    histogram=Counter();pins={};previous_end=0
    # Reuse the same raw-spread histogram, never a fitted future scale or payoff.
    # Each fit still represents exactly TRAIN[0:end], with every source pin.
    earlier=[]
    for old_path in (OUT/'fits').glob(f'{symbol}_TRAIN0_*.json'):
        check_receipt(old_path)
        old=read(old_path)
        if (old.get('declaration_sha256')==decl_sha and
                0<old.get('train_end_day_exclusive',0)<end and 'spread_histogram_scaled' in old):
            earlier.append(old)
    if earlier:
        prior=max(earlier,key=lambda value:value['train_end_day_exclusive'])
        prefix_records=records(symbol,0,prior['train_end_day_exclusive'])
        prefix_pins={name:pinned for record in prefix_records for name,pinned in record['files'].items()}
        if (prior['symbol']!=symbol or prior['train_start_day']!=0 or
                prior['uses_validation_or_final'] is not False or prior['uses_strategy_payoff'] is not False or
                prior['source_files']!=prefix_pins):
            raise ValueError('Earlier fit is not the exact declared training prefix')
        for name,pinned in prior['source_files'].items():
            if sha(ROOT/name)!=pinned:raise ValueError('Earlier training source changed')
        previous_end=prior['train_end_day_exclusive'];pins.update(prior['source_files'])
        histogram.update({value:count for value,count in prior['spread_histogram_scaled']})
    for record in records(symbol,previous_end,end):
        day=load(record)
        for quote in day.tape:
            spread=quote.ask-quote.bid
            if spread>0:histogram[spread]+=1
        pins.update(record['files'])
    n=sum(histogram.values())
    if n==0:raise ValueError('No positive training spread; no fallback scale')
    ranks=((n-1)//2,n//2);found=[];count=0
    for value,total in sorted(histogram.items()):
        for rank in ranks:
            if count<=rank<count+total:found.append(value)
        count+=total
    unit=math.fsum(found)/2
    fit={'declaration_sha256':decl_sha,'symbol':symbol,'train_start_day':0,
         'train_end_day_exclusive':end,'U_scaled':unit,'U_quote_units':unit/PRICE_SCALE,
         'positive_spread_rows':n,'source_files':pins,'uses_strategy_payoff':False,
         'uses_validation_or_final':False,'reused_training_days':previous_end,
         'spread_histogram_scaled':sorted(histogram.items())}
    create(path,fit)
    return fit

def case_paths(symbol,stage,policy,scenario,baseline):
    stem=f'{symbol}_{stage}_{policy}_{scenario}'+('_CLOCK' if baseline else '')
    return OUT/'cases'/(stem+'.json'),OUT/'ledgers'/(stem+'.json.gz')

def validated_case(path,decl_sha):
    check_receipt(path)
    case=read(path)
    if case['declaration_sha256']!=decl_sha:
        raise ValueError('Case declaration changed')
    if sha(ROOT/case['ledger_path'])!=case['ledger_sha256']:
        raise ValueError('Case ledger changed')
    for name,pinned in case['source_files'].items():
        if sha(ROOT/name)!=pinned:raise ValueError('Case source changed')
    return case

def evaluate_case(tape,symbol,stage,policy,scenario,baseline,unit,decl_sha,source_pins):
    path,ledger_path=case_paths(symbol,stage,policy,scenario,baseline)
    if path.exists():
        saved=validated_case(path,decl_sha)
        if saved['declaration_sha256']!=decl_sha or saved['source_files']!=source_pins:
            raise ValueError('Existing case belongs to different source/declaration')
        if sha(ROOT/saved['ledger_path'])!=saved['ledger_sha256']:
            raise ValueError('Cached ledger changed; never silently regenerate')
        print('REUSED',path.stem,flush=True)
        return saved
    latency=250 if scenario=='BASE' else 2000
    slip=(.25 if scenario=='BASE' else .5)*unit
    start,end,_=STAGES[stage]
    result=evaluate_tape(tape,policy,SYMBOLS[symbol],unit,START_MS+start*DAY_MS,
                         START_MS+end*DAY_MS,ExecutionConfig(latency,latency,5000,slip),baseline=baseline)
    ledger=result.pop('ledger')
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(ledger,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    with ledger_path.open('xb') as stream:stream.write(gzip.compress(payload,mtime=0))
    result.update(declaration_sha256=decl_sha,symbol=symbol,stage=stage,policy=policy,
                  scenario=scenario,baseline=baseline,U_scaled=unit,source_files=source_pins,
                  ledger_path=str(ledger_path.relative_to(ROOT)),ledger_sha256=sha(ledger_path),
                  cash_net=None,costs_verified=False,execution_verified=False,profit_qualified=False)
    create(path,result)
    print('CALCULATED',path.stem,json.dumps(result['summary'],sort_keys=True),flush=True)
    return result

def run_stage(stage):
    _,decl_sha=declaration()
    if stage=='FINAL':
        selection_path=OUT/'selection.json'
        check_receipt(selection_path)
        selection=read(selection_path)
        if selection!=selection_from_development(decl_sha):
            raise ValueError('Selection differs from reproduced development gates')
        if selection['declaration_sha256']!=decl_sha:raise ValueError('Selection declaration changed')
        for name,pinned in selection['development_receipts'].items():
            if sha(ROOT/name)!=pinned:raise ValueError('Development selection evidence changed')
        if read(DATA/'acquisition_state.json')['status']!='COMPLETED':
            raise ValueError('Final cannot open until fixed acquisition completes')
        allowed=set(selection['conditionally_eligible'])
    else:allowed={f'{symbol}/{policy}' for symbol in SYMBOLS for policy in POLICY_IDS}
    start,end,train=STAGES[stage]
    for symbol in SYMBOLS:
        policies=[policy for policy in sorted(POLICY_IDS) if f'{symbol}/{policy}' in allowed]
        if not policies:
            print('NOT_OPENED_DEVELOPMENT_GATE',symbol,stage,flush=True);continue
        fit=unit_fit(symbol,train,decl_sha)
        source=records(symbol,start,end)
        pins={name:pinned for record in source for name,pinned in record['files'].items()}
        def quotes():
            for record in source:yield from load(record).tape
        tape=QuoteTape(quotes())
        for scenario in ('BASE','STRESS'):
            benchmark_path=OUT/'benchmarks'/f'{symbol}_{stage}_{scenario}.json'
            if benchmark_path.exists():
                check_receipt(benchmark_path)
            else:
                latency=250 if scenario=='BASE' else 2000
                config=ExecutionConfig(latency,latency,5000,(.25 if scenario=='BASE' else .5)*fit['U_scaled'])
                reference=endpoint_reference(tape,START_MS+start*DAY_MS,START_MS+end*DAY_MS,
                                             SYMBOLS[symbol],config)
                reference.update(declaration_sha256=decl_sha,source_files=pins,
                                 symbol=symbol,stage=stage,scenario=scenario)
                create(benchmark_path,reference)
        for policy in policies:
            for scenario in ('BASE','STRESS'):
                for baseline in (False,True):
                    evaluate_case(tape,symbol,stage,policy,scenario,baseline,fit['U_scaled'],decl_sha,pins)

def selection_from_development(decl_sha):
    output={'declaration_sha256':decl_sha,'conditionally_eligible':[],'gates':{},
            'development_receipts':{},'final_strategy_measurements_before_selection':0,
            'profit_qualified':False,'cash_net':None,'costs_verified':False}
    for symbol in SYMBOLS:
        for policy in sorted(POLICY_IDS):
            values=[]
            for stage in ('WF1','WF2','WF3','VALIDATION'):
                path,_=case_paths(symbol,stage,policy,'BASE',False)
                case=validated_case(path,decl_sha)
                output['development_receipts'][str(path.relative_to(ROOT))]=sha(path)
                values.append(case['summary'])
            gate=development_gate(validation_sum_R=values[3]['sum_R'],
                  walk_forward_sum_R=[v['sum_R'] for v in values[:3]],
                  completed=sum(v['completed'] for v in values),unknown=sum(v['unknown'] for v in values))
            key=f'{symbol}/{policy}';output['gates'][key]=gate
            if gate['conditional_eligible']:output['conditionally_eligible'].append(key)
    return output

def select():
    _,decl_sha=declaration()
    output=selection_from_development(decl_sha)
    create(OUT/'selection.json',output)
    print('Selection frozen before final; conditional candidates',len(output['conditionally_eligible']),flush=True)

def summarize():
    _,decl_sha=declaration();check_receipt(OUT/'selection.json');selection=read(OUT/'selection.json')
    if selection!=selection_from_development(decl_sha):
        raise ValueError('Selection differs from reproduced development gates')
    if selection['declaration_sha256']!=decl_sha:raise ValueError('Selection declaration changed')
    for name,pinned in selection['development_receipts'].items():
        if sha(ROOT/name)!=pinned:raise ValueError('Selection evidence changed')
    claims={key:Claim() for key in CLAIM_IDS};analyses={};cases={}
    for key in selection['conditionally_eligible']:
        symbol,policy=key.split('/')
        path,_=case_paths(symbol,'FINAL',policy,'BASE',False)
        base=validated_case(path,decl_sha)
        clock=validated_case(case_paths(symbol,'FINAL',policy,'BASE',True)[0],decl_sha)
        daily,reference=base['daily'],clock['daily']
        records_=DailyRecords(
            [d['observed_sum_R'] for d in daily],[d['observed_sum_quote'] for d in daily],
            [d['observed_sum_R'] for d in reference],[d['completed'] for d in daily],
            [d['unknown'] for d in daily],[d['unknown'] for d in reference],
            coverage_complete=not base['summary']['remaining_censored'],
            clock_coverage_complete=not clock['summary']['remaining_censored'],
            daily_gain_R=[d['observed_gain_R'] for d in daily],daily_loss_R=[d['observed_loss_R'] for d in daily])
        analysis=analyze_daily(records_,declaration_sha256=decl_sha,development_eligible=True)
        analyses[key]=analysis;cases[key]=base
        claims[key+'/POSITIVE_DAILY_R']=Claim(analysis['p_positive_R'],analysis['testable'])
        claims[key+'/EXCESS_DAILY_R_VS_CLOCK']=Claim(analysis['p_excess_R_vs_clock'],analysis['testable'])
    holm=holm12(claims);results={}
    for key,analysis in analyses.items():
        gate=final_statistical_gate(analysis,
            holm_positive_p=holm['claims'][key+'/POSITIVE_DAILY_R']['holm_p_value'],
            holm_excess_p=holm['claims'][key+'/EXCESS_DAILY_R_VS_CLOCK']['holm_p_value'])
        symbol,policy=key.split('/')
        stress=validated_case(case_paths(symbol,'FINAL',policy,'STRESS',False)[0],decl_sha)['summary']
        base=cases[key]['summary']
        external={'base_conditional_drawdown':base['marked_drawdown_pct'] is not None and base['marked_drawdown_pct']<=20,
                  'stress_positive':stress['sum_R'] is not None and stress['sum_R']>0,
                  'stress_conditional_drawdown':stress['marked_drawdown_pct'] is not None and stress['marked_drawdown_pct']<=20}
        results[key]={'analysis':analysis,'statistical_gate':gate,'external_conditional_gates':external,
            'conditional_paper_candidate':gate['conditional_statistical_pass'] and all(external.values()),
            'profit_qualified':False,'cash_net':None,'costs_verified':False,'execution_verified':False}
    create(OUT/'final_summary.json',{'declaration_sha256':decl_sha,'selection_sha256':sha(OUT/'selection.json'),
        'holm':holm,'results':results,'not_opened_development_gate':sorted(set(f'{s}/{p}' for s in SYMBOLS for p in POLICY_IDS)-set(results)),
        'qualified_profit_strategies':0,'cash_net':None,'costs_verified':False,'execution_verified':False,
        **{key:False for key in SAFETY_FIELDS}})
    print('Final conditional inference saved; cash profit remains unverified',flush=True)

def main():
    safe();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.study.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    frozen=sub.add_parser('freeze');frozen.add_argument('--validation-receipt',type=Path,required=True)
    run=sub.add_parser('run');run.add_argument('--stage',choices=STAGES,required=True)
    sub.add_parser('select');sub.add_parser('summarize')
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.validation_receipt.resolve())
    elif args.command=='run':run_stage(args.stage)
    elif args.command=='select':select()
    else:summarize()

if __name__=='__main__':main()
