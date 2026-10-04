#!/usr/bin/env python3
"""Post-hoc resampling resolution check, with independently computed paired counts.

This reuses the already scalar-audited feature/label implementation and saved
signal timestamps, but independently computes block counts, intervals and the
centered null-tail test. It is not a new out-of-sample test and cannot change
the original frozen study gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values):
    values = values[np.isfinite(values)]
    return np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]


def sensitivity(repo, study_file, repeats, variant, block_hours, chunk_size):
    sys.path.insert(0,str(repo/'backend'))
    from app.research.spike_hunter import assert_offline, features, load_m1, outcomes, partitions, eligible
    assert_offline()
    study = json.loads(study_file.read_text())
    definition = '2.0ATR_15m'
    proof = {'analysis':'POST_HOC_MONTE_CARLO_RESOLUTION_SENSITIVITY',
             'not_fresh_oos':True,'original_research_gate_unchanged':True,
             'definition':definition,'variant':variant,'repeats':repeats,
             'block_hours':block_hours,'seed_base':20261004,'actual_seed':20261004+block_hours,
             'final_comparison_family':study['final_family_size'],
             'study_sha256':hashlib.sha256(study_file.read_bytes()).hexdigest(),
             'safety':{name:False for name in ('LIVE_TRADING','READY_FOR_LIVE','LIVE_ALLOWED','OPENED_TRADES')},
             'symbols':{}}
    for symbol,sym in study['symbols'].items():
        source = repo/'data'/'spike_hunter'/sym['audit']['source_file']
        m1,audit = load_m1(source)
        assert audit['sha256'] == sym['audit']['sha256']
        m5 = features(m1,sym['direction'])
        hit,tts,_ = outcomes(m1,m5,sym['direction'],2,15)
        _,_,valid30 = outcomes(m1,m5,sym['direction'],2,30)
        start,end = partitions(m1)['final_test']
        mask = eligible(m5,valid30,start,end)
        saved = pd.read_csv(study_file.parent/f'{symbol.lower()}_signals.csv')
        saved = saved[saved.variant == variant]
        issue_times = pd.DatetimeIndex(pd.to_datetime(saved.signal_close_utc,utc=True)).as_unit('ns')
        closes = (m5.index+pd.Timedelta(minutes=5)).as_unit('ns')
        indices = np.searchsorted(closes.asi8,issue_times.array.asi8)
        assert np.array_equal(closes.asi8[indices],issue_times.array.asi8)
        selected = indices[mask[indices]]
        stored = sym['definitions'][definition][variant]['final_test']
        assert stored['signals'] == len(selected)
        assert stored['hits'] == int(hit[selected].sum())
        assert stored['opportunities'] == int(mask.sum())
        assert stored['positive_opportunities'] == int(hit[mask].sum())
        origin = start.floor(f'{block_hours}h')
        seconds = block_hours*3600
        nblocks = math.ceil((end-origin).total_seconds()/seconds)
        ids = np.asarray(((closes-origin).total_seconds()//seconds),dtype=int)
        ids = np.clip(ids,0,nblocks-1)
        sn = np.bincount(ids[selected],minlength=nblocks)
        sk = np.bincount(ids[selected[hit[selected]]],minlength=nblocks)
        bn = np.bincount(ids[mask],minlength=nblocks)
        bk = np.bincount(ids[mask & hit],minlength=nblocks)
        precision,base = float(sk.sum()/sn.sum()),float(bk.sum()/bn.sum())
        observed_difference = precision-base
        arrays = {key:[] for key in ('precision','base','lift','difference')}
        rng = np.random.default_rng(20261004+block_hours)
        for offset in range(0,repeats,chunk_size):
            size = min(chunk_size,repeats-offset)
            weights = rng.multinomial(nblocks,np.full(nblocks,1/nblocks),size)
            signal_n,signal_k = weights@sn,weights@sk
            baseline_n,baseline_k = weights@bn,weights@bk
            with np.errstate(divide='ignore',invalid='ignore'):
                bp,bb = signal_k/signal_n,baseline_k/baseline_n
                bl,bd = bp/bb,bp-bb
            bp[signal_n==0] = np.nan
            bl[(signal_n==0) | (baseline_k==0)] = np.nan
            bd[(signal_n==0) | (baseline_n==0)] = np.nan
            for key,value in zip(arrays,(bp,bb,bl,bd)):
                arrays[key].append(value)
        arrays = {key:np.concatenate(values) for key,values in arrays.items()}
        finite = arrays['difference'][np.isfinite(arrays['difference'])]
        tail_count = int(((finite-observed_difference)>=observed_difference).sum())
        pvalue = 1.0 if observed_difference <= 0 else (tail_count+1)/(len(finite)+1)
        record = {'signals':len(selected),'hits':int(sk.sum()),'opportunities':int(bn.sum()),
                  'positive_opportunities':int(bk.sum()),'precision':precision,'base_rate':base,'lift':precision/base,
                  'block_precision_ci95':interval(arrays['precision']),
                  'block_base_rate_ci95':interval(arrays['base']),
                  'block_lift_ci95':interval(arrays['lift']),
                  'block_difference_ci95':interval(arrays['difference']),
                  'centered_tail_exceedances':tail_count,'bootstrap_valid_replicates':len(finite),
                  'bootstrap_one_sided_p':pvalue,
                  'conservative_family_p_times_family':min(1.0,pvalue*study['final_family_size']),
                  'monte_carlo_minimum_p':1/(len(finite)+1),
                  'original_9999_replicate_values_match':None}
        # Verify earlier numbers using the exact original first 9999 draws.
        if repeats>=9999 and study['bootstrap_replicates']==9999 and block_hours==24:
            first_d = arrays['difference'][:9999]
            first_d = first_d[np.isfinite(first_d)]
            first_p = 1.0 if observed_difference<=0 else (1+int(((first_d-observed_difference)>=observed_difference).sum()))/(len(first_d)+1)
            assert first_p == stored['bootstrap_one_sided_p']
            for key,target in [('precision','block_precision_ci95'),('base','block_base_rate_ci95'),('lift','block_lift_ci95'),('difference','block_difference_ci95')]:
                np.testing.assert_allclose(interval(arrays[key][:9999]),stored[target],rtol=1e-12,atol=1e-12)
            record['original_9999_replicate_values_match'] = True
        if tail_count==0:
            # Exact one-sided 95% upper bound for the simulated tail probability.
            upper = 1-.05**(1/len(finite))
            record['zero_exceedance_monte_carlo_tail_p_upper95'] = upper
            record['zero_exceedance_monte_carlo_family_p_upper95'] = min(1.0,upper*study['final_family_size'])
        proof['symbols'][symbol] = record
        print(f'{symbol}: p={pvalue:.8f}, family conservative={record["conservative_family_p_times_family"]:.6f}, liftCI={record["block_lift_ci95"]}',flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=199999)
    parser.add_argument('--variant',default='SR_ALIGNMENT')
    parser.add_argument('--block-hours',type=int,default=24)
    parser.add_argument('--chunk-size',type=int,default=20000)
    args = parser.parse_args()
    if args.repeats<9999 or args.block_hours<1 or args.chunk_size<1:
        parser.error('Use at least 9999 repeats, positive block length and chunk size')
    result = sensitivity(args.repo.resolve(),args.study.resolve(),args.repeats,args.variant,args.block_hours,args.chunk_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
