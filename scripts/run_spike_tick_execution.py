#!/usr/bin/env python3
"""Immutable sampled-day tick execution diagnosis; no fit, orders or promotion."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend')]
from app.research.payoff_ticks import TickExitConfig, replay_ticks
from app.research.payoff_timed import TimedExitConfig, replay_timed
from app.research.payoff_metrics import summarize
from app.research.spike_hunter import assert_offline, load_m1
from scripts.run_spike_nonlinear_study import frozen as frozen6, prepare, issued, SYMBOLS, FAMILIES, MODES

OLD = ROOT / 'docs/spike_nonlinear_20261005'
DATA = ROOT / 'data/spike_tick_execution_20261005'
OUTPUT = ROOT / 'docs/spike_tick_execution_20261005'
DATES = tuple((pd.Timestamp('2026-04-11', tz='UTC') + pd.Timedelta(days=k*175//11)).strftime('%Y-%m-%d') for k in range(12))
CONFIG = TickExitConfig(stop_atr=2., max_hold_minutes=15, entry_delay_minutes=1,
                        round_trip_cost_atr=.1, max_gap_seconds=1, stop_latency_ticks=1)
BOOTSTRAP = 9999
NEW_SOURCES = ('docs/SPIKE_TICK_EXECUTION_PROTOCOL.md', 'scripts/run_spike_tick_execution.py',
               'backend/app/research/payoff_ticks.py', 'backend/tests/test_payoff_ticks.py',
               'scripts/collect_spike_ticks.py', 'backend/tests/test_tick_collector.py',
               'backend/tests/test_spike_tick_execution.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')


def hashes():
    return {p: digest(ROOT / p) for p in NEW_SOURCES}


def day_bounds(date):
    start = pd.Timestamp(date, tz='UTC')
    return start, start + pd.Timedelta(days=1)


def subset(frame):
    result = frame.copy()
    result['signal_time'] = pd.to_datetime(result.signal_time, utc=True)
    return result.loc[result.signal_time.dt.strftime('%Y-%m-%d').isin(DATES)].copy()


def signal_records(frame):
    return frame.to_dict('records')


def declare(output):
    assert_offline()
    if any((output/p).exists() for p in ('declaration.json', 'tick_sources.json', 'results.json')):
        raise ValueError('Refusing to overwrite a declaration or evaluated run')
    selection, fingerprint = frozen6(OLD)
    source_files = {str((OLD/'results.json').relative_to(ROOT)): digest(OLD/'results.json')}
    sampled = {}
    for symbol in SYMBOLS:
        sampled[symbol] = {}
        value = selection['sources'][symbol]['fresh']
        _, rows, _, _, _ = prepare(value, symbol)
        for mode in MODES:
            controls = subset(pd.DataFrame(issued(rows, symbol, mode)))
            path = output/f'{symbol}_CLOCK_{mode}_signals.csv'
            controls.to_csv(path, index=False)
            sampled[symbol][f'CLOCK_{mode}'] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'rows': len(controls)}
            for family in FAMILIES:
                key = f'{family}_{mode}'
                old_path = OLD/f'{symbol}_later_temporal180_{key}_signals.csv'
                source_files[str(old_path.relative_to(ROOT))] = digest(old_path)
                signals = subset(pd.read_csv(old_path))
                if signals.signal_time.duplicated().any():
                    raise ValueError('Saved frozen signals must be unique')
                path = output/f'{symbol}_{key}_signals.csv'
                signals.to_csv(path, index=False)
                sampled[symbol][key] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'rows': len(signals),
                                       'development_eligible': selection['symbols'][symbol]['models'][key]['development_eligible']}
                ledger = OLD/f'{symbol}_later_temporal180_{key}_trades.csv'
                source_files[str(ledger.relative_to(ROOT))] = digest(ledger)
    declaration = {'stage': 'tick_execution_preacquisition', 'run_utc': datetime.now(timezone.utc).isoformat(),
                   'safety': assert_offline(), 'dates': DATES, 'symbols': SYMBOLS, 'config': asdict(CONFIG),
                   'bootstrap_repeats': BOOTSTRAP, 'code_hashes': hashes(), 'round6_selection_sha256': fingerprint,
                   'round6_source_files': source_files, 'sampled_signals': sampled,
                   'tick_outcomes_evaluated': False, 'raw_ticks_collected': False,
                   'endpoint': 'wss://api.derivws.com/trading/v1/options/ws/public', 'page_size': 1000,
                   'modeled_cost_not_measured': True, 'max_opportunities_per_model': 576,
                   'study_kind': 'adaptive_historical_sampled_execution_diagnosis_not_profit_validation'}
    save(output/'declaration.json', declaration)
    (output/'declaration.sha256').write_text(digest(output/'declaration.json')+'\n')
    print('TICK_EXECUTION_DECLARED', digest(output/'declaration.json'), 'no tick outcomes read', flush=True)


def frozen_declaration(output):
    d = json.loads((output/'declaration.json').read_text())
    if digest(output/'declaration.json') != (output/'declaration.sha256').read_text().strip():
        raise ValueError('Tick execution declaration changed')
    _, sha = frozen6(OLD)
    if d['round6_selection_sha256'] != sha or d['code_hashes'] != hashes():
        raise ValueError('Frozen models or new code changed')
    if d['dates'] != list(DATES) or d['symbols'] != list(SYMBOLS) or d['config'] != asdict(CONFIG):
        raise ValueError('Fixed dates/symbols/policy changed')
    for path, fingerprint in d['round6_source_files'].items():
        if digest(ROOT/path) != fingerprint:
            raise ValueError('Frozen round6 saved input changed')
    for models in d['sampled_signals'].values():
        for signal in models.values():
            if digest(ROOT/signal['path']) != signal['sha256']:
                raise ValueError('Frozen sampled issuance changed')
    return d


def load_ticks(value):
    for filekey, shakey in (('clean_file','clean_sha256'), ('raw_pages_file','raw_pages_sha256'),
                            ('page_audit_file','page_audit_sha256')):
        if digest(ROOT/value[filekey]) != value[shakey]:
            raise ValueError('Acquired tick bytes changed')
    frame = pd.read_csv(ROOT/value['clean_file'])
    if list(frame.columns) != ['epoch', 'quote'] or frame.empty:
        raise ValueError('Tick CSV requires epoch/quote')
    epochs = frame.epoch.to_numpy(float)
    prices = frame.quote.to_numpy(float)
    if (not np.isfinite(epochs).all() or not np.equal(epochs, np.floor(epochs)).all() or
            not np.isfinite(prices).all() or (prices <= 0).any() or (np.diff(epochs) <= 0).any()):
        raise ValueError('Ticks require sorted unique seconds and finite positive quotes')
    if (epochs[0] < value['start_epoch'] or epochs[-1] >= value['end_exclusive_epoch'] or
            value['end_exclusive_epoch']-value['start_epoch'] != 86400 or
            len(frame) != value['rows'] or value['expected_grid_rows'] != 86400):
        raise ValueError('Tick day grid/bounds/row count changed')
    missing = 86400-len(frame)
    if value['missing_seconds'] != missing or value['gap_free'] != (missing == 0):
        raise ValueError('Tick gap metadata inconsistent')
    frame.index = pd.to_datetime(frame.pop('epoch'), unit='s', utc=True).astype('datetime64[ns, UTC]')
    return frame


def validate_pages(value, declaration):
    pages = [json.loads(line) for line in (ROOT/value['page_audit_file']).read_text().splitlines()]
    raw = [json.loads(line) for line in (ROOT/value['raw_pages_file']).read_text().splitlines()]
    if len(raw) != len(pages):
        raise ValueError('Tick raw/page attempt counts differ')
    end = value['end_exclusive_epoch']-1
    successes, reconstructed, failures = {}, {}, []
    for ordinal, (page, record) in enumerate(zip(pages, raw, strict=True)):
        request = page['request']
        if (set(request) != {'ticks_history','start','end','style','count','req_id'} or
                request['ticks_history'] != value['symbol'] or request['style'] != 'ticks' or
                request['start'] != value['start_epoch'] or request['end'] != end or
                request['count'] != declaration['page_size'] or record['request'] != request or
                page['endpoint'] != declaration['endpoint'] or record['endpoint'] != declaration['endpoint']):
            raise ValueError('Tick successful-page chain/request is invalid')
        key = hashlib.sha256(json.dumps({'endpoint':declaration['endpoint'],'request':request},
                                       sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if key != record['request_key'] or key != page['request_key']:
            raise ValueError('Tick exact request hash changed')
        wire = record['response_wire']
        if wire is not None:
            fingerprint=hashlib.sha256(wire.encode()).hexdigest()
            if fingerprint != record['wire_sha256'] or fingerprint != page['wire_sha256']:
                raise ValueError('Invalid raw response hash')
        if not page['accepted']:
            if page.get('recoverable') is not True:
                raise ValueError('Unmatched semantic public tick error')
            error=page['error']
            if record.get('error') != error or error['request'] != request or error['request_key'] != key:
                raise ValueError('Failed raw tick request metadata changed')
            if error['error_type'] == 'RateLimitError':
                if wire is None or json.loads(wire).get('error',{}).get('code') != 'RateLimit':
                    raise ValueError('Unproven recovered rate limit')
            elif error['error_type'] not in ('OSError','TimeoutError','ConnectionClosed','ConnectionClosedError',
                                            'ConnectionClosedOK','InvalidStatus','InvalidStatusCode','gaierror',
                                            'ConnectionResetError','BrokenPipeError','SSLError'):
                raise ValueError('Undeclared recovered tick error type')
            failures.append(error)
            continue
        if wire is None or key in successes or not 0 <= page['rows'] <= declaration['page_size']:
            raise ValueError('Invalid successful tick page')
        response=json.loads(wire,parse_float=Decimal)
        if response.get('error') or response.get('msg_type') != 'history' or response.get('req_id') != request['req_id']:
            raise ValueError('Invalid raw successful tick history')
        times,prices=response['history']['times'],response['history']['prices']
        if len(times) != len(prices) or len(times) != page['rows']:
            raise ValueError('Raw tick arrays and page counts differ')
        previous=None
        for stamp,price in zip(times,prices,strict=True):
            if isinstance(price,bool):
                raise ValueError('Invalid boolean tick quote')
            quote=Decimal(str(price))
            if (isinstance(stamp,bool) or not isinstance(stamp,int) or not request['start']<=stamp<=request['end'] or
                    previous is not None and stamp<previous or not quote.is_finite() or quote<=0 or isinstance(price,bool)):
                raise ValueError('Tick response silently adjusted interval or invalid quote')
            if stamp in reconstructed and reconstructed[stamp] != quote:
                raise ValueError('Conflicting raw tick duplicate')
            reconstructed[stamp]=quote;previous=stamp
        successes[key]=(ordinal,page)
        if not times:
            if ordinal != len(pages)-1 or page['oldest_epoch'] is not None or page['newest_epoch'] is not None:
                raise ValueError('Empty tick page can only terminate a day')
            end=value['start_epoch']-1
        else:
            if page['oldest_epoch']!=times[0] or page['newest_epoch']!=times[-1]:
                raise ValueError('Raw tick page bounds changed')
            end=times[0]-1
    if not pages:
        raise ValueError('No public tick pages')
    if end>=value['start_epoch']:
        raise ValueError('Tick pagination did not reach the declared start')
    if failures != value['request_errors']:
        raise ValueError('Preserved tick request failures were omitted or changed')
    lineage=value['recovered_retry_lineage']
    if len(lineage)!=len(value['request_errors']):
        raise ValueError('Tick recovered-error lineage count changed')
    for index,(error,recovery) in enumerate(zip(value['request_errors'],lineage,strict=True)):
        if error['request_key'] not in successes:
            raise ValueError('Unmatched tick request failure')
        ordinal,page=successes[error['request_key']]
        if (error['request'] != page['request'] or error['endpoint'] != declaration['endpoint'] or
                recovery['error_index'] != index or recovery['successful_page_audit_index'] != ordinal or
                recovery['request_key'] != error['request_key'] or
                recovery['successful_wire_sha256'] != page['wire_sha256']):
            raise ValueError('Unmatched tick recovered-error linkage')
    csv=pd.read_csv(ROOT/value['clean_file'],dtype={'epoch':np.int64,'quote':str})
    clean={int(row.epoch):Decimal(row.quote) for row in csv.itertuples(index=False)}
    if len(csv)!=len(clean) or clean!=reconstructed:
        raise ValueError('Clean ticks differ from preserved raw wire quotes')


def reconcile(ticks, m1, date):
    start, end = day_bounds(date)
    bars = ticks.quote.resample('1min').ohlc()
    count = ticks.quote.resample('1min').count()
    complete = bars.loc[count == 60]
    reference = m1.loc[(m1.index >= start) & (m1.index < end), ['open','high','low','close']]
    if not complete.index.isin(reference.index).all():
        raise ValueError('Complete tick minute missing from frozen M1')
    old = reference.loc[complete.index]
    delta = np.abs(complete.to_numpy()-old.to_numpy())
    valid = np.isclose(complete.to_numpy(), old.to_numpy(), rtol=1e-12, atol=1e-8)
    if not valid.all():
        raise ValueError(f'Frozen M1 vs tick complete-minute price mismatch on {date}')
    return {'complete_minutes': len(complete), 'unknown_minutes': 1440-len(complete),
            'max_abs_difference': float(delta.max()) if delta.size else None,
            'price_match': True, 'coverage_is_separately_audited': True}


def freeze_data(output):
    if (output/'tick_sources.json').exists() or (output/'results.json').exists():
        raise ValueError('Refusing tick-source freeze overwrite')
    d = frozen_declaration(output)
    selection, _ = frozen6(OLD)
    config_path, summary_path = DATA/'acquisition_config.json', DATA/'acquisition_summary.json'
    config, summary = json.loads(config_path.read_text()), json.loads(summary_path.read_text())
    config_sha = digest(config_path)
    if (config['symbols'] != list(SYMBOLS) or config['dates'] != list(DATES) or
            config['endpoint'] != d['endpoint'] or config['page_size'] != d['page_size'] or
            config['cadence_seconds'] != 1 or config['collector_sha256'] != d['code_hashes']['scripts/collect_spike_ticks.py'] or
            summary['config_sha256'] != config_sha or summary['config'] != config or summary['status'] != 'COMPLETE' or
            summary['authentication_used'] is not False or summary['account_or_order_requests'] is not False or
            any(config['safety'].get(k) is not False or summary['safety'].get(k) is not False for k in assert_offline())):
        raise ValueError('Tick acquisition configuration/status differs from the declared public run')
    sources = {}
    for symbol in SYMBOLS:
        m1, _ = load_m1(ROOT/selection['sources'][symbol]['fresh']['path'])
        sources[symbol] = {}
        for date in DATES:
            manifest = DATA/f'{symbol}_{date}_manifest.json'
            v = json.loads(manifest.read_text())
            start, end = day_bounds(date)
            if (v['symbol'] != symbol or v['date'] != date or v['start_epoch'] != int(start.timestamp()) or
                    v['end_exclusive_epoch'] != int(end.timestamp()) or v['cadence_seconds'] != 1 or
                    v['endpoint'] != d['endpoint'] or v['normalization_valid'] is not True or
                    v['config_sha256'] != config_sha or v['collector_sha256'] != config['collector_sha256'] or
                    v['fills_or_interpolations'] is not False or v['authentication_used'] is not False or
                    any(v['safety'].get(k) is not False for k in assert_offline())):
                raise ValueError('Tick source violates the declared source contract')
            validate_pages(v, d)
            ticks = load_ticks(v)
            v.update(manifest=str(manifest.relative_to(ROOT)), manifest_sha256=digest(manifest),
                     reconciliation=reconcile(ticks, m1, date))
            sources[symbol][date] = v
    frozen_declaration(output)
    payload = {'stage':'tick_sources_frozen_before_payoff', 'frozen_utc':datetime.now(timezone.utc).isoformat(),
               'declaration_sha256':digest(output/'declaration.json'), 'sources':sources,
               'acquisition_files':{str(config_path.relative_to(ROOT)):config_sha,
                                    str(summary_path.relative_to(ROOT)):digest(summary_path)},
               'safety':assert_offline(), 'tick_outcomes_evaluated':False}
    save(output/'tick_sources.json', payload)
    (output/'tick_sources.sha256').write_text(digest(output/'tick_sources.json')+'\n')
    print('TICK_SOURCES_FROZEN', digest(output/'tick_sources.json'), flush=True)


def frozen_sources(output):
    d = frozen_declaration(output)
    if digest(output/'tick_sources.json') != (output/'tick_sources.sha256').read_text().strip():
        raise ValueError('Frozen tick source declaration changed')
    data = json.loads((output/'tick_sources.json').read_text())
    if data['declaration_sha256'] != digest(output/'declaration.json'):
        raise ValueError('Tick source declaration linkage changed')
    for path,fingerprint in data['acquisition_files'].items():
        if digest(ROOT/path) != fingerprint:
            raise ValueError('Frozen acquisition configuration or summary changed')
    for source in data['sources'].values():
        for v in source.values():
            if digest(ROOT/v['manifest']) != v['manifest_sha256']:
                raise ValueError('Frozen tick manifest changed')
            load_ticks(v)
    return d, data


def daily_arrays(trades):
    known = trades.loc[~trades.censored.astype(bool) & np.isfinite(trades.net_R)].copy()
    known['day'] = pd.to_datetime(known.signal_time, utc=True).dt.strftime('%Y-%m-%d')
    if not known.day.isin(DATES).all():
        raise ValueError('Unscheduled trades in sampled-day analysis')
    counts, sums, gains, losses = (np.zeros(len(DATES)) for _ in range(4))
    for i, day in enumerate(DATES):
        values = known.loc[known.day == day, 'net_R'].to_numpy(float)
        counts[i], sums[i] = len(values), values.sum()
        gains[i], losses[i] = values[values>0].sum(), -values[values<0].sum()
    return counts, sums, gains, losses


def interval(values):
    values = values[np.isfinite(values)]
    return np.quantile(values, [.025,.975]).tolist() if len(values) else [None,None]


def sampled_summary(trades):
    m = summarize(trades, *day_span(), stop_atr=CONFIG.stop_atr)
    n, sums, gains, losses = daily_arrays(trades)
    m['span_calendar_days'], m['calendar_days'] = m['calendar_days'], len(DATES)
    m['calendar_basis'] = 'twelve scheduled sampled UTC days; unsampled days not observed'
    m['cluster_se'] = (float(np.sqrt(12/11*np.sum((sums-m['mean_net_R']*n)**2))/n.sum())
                       if n.sum() else None)
    m['selection_score'] = None
    weights = np.random.default_rng(20261005).multinomial(12, np.full(12,1/12), size=BOOTSTRAP)
    bn, bs, bg, bl = (weights @ x for x in (n,sums,gains,losses))
    means = np.divide(bs,bn,out=np.full(BOOTSTRAP,np.nan),where=bn>0)
    pf = np.divide(bg,bl,out=np.full(BOOTSTRAP,np.nan),where=bl>0)
    m.update(mean_net_R_ci95=interval(means), profit_factor_ci95=interval(pf),
             mean_valid_replicates=int(np.isfinite(means).sum()), pf_undefined_replicates=int((~np.isfinite(pf)).sum()),
             bootstrap_repeats=BOOTSTRAP, bootstrap_seed=20261005,
             inference_kind='descriptive_iid_scheduled_day_bootstrap_not_discovery',
             no_contiguous_weekly_claim=True)
    return m


def day_span():
    return day_bounds(DATES[0])[0], day_bounds(DATES[-1])[1]


def matched_difference(tick, coarse):
    a = tick.loc[~tick.censored & np.isfinite(tick.net_R), ['signal_time','net_R']]
    b = coarse.loc[~coarse.censored & np.isfinite(coarse.net_R), ['signal_time','net_R']]
    merged = a.merge(b,on='signal_time',how='inner',validate='one_to_one',suffixes=('_tick','_coarse'))
    values = merged.net_R_tick-merged.net_R_coarse
    daily = merged.assign(day=pd.to_datetime(merged.signal_time,utc=True).dt.strftime('%Y-%m-%d'), delta=values)
    n, sums = np.zeros(12), np.zeros(12)
    for i, day in enumerate(DATES):
        part = daily.loc[daily.day == day, 'delta']
        n[i], sums[i] = len(part), part.sum()
    w = np.random.default_rng(20261005).multinomial(12,np.full(12,1/12),size=BOOTSTRAP)
    counts, totals = w@n,w@sums
    samples = np.divide(totals,counts,out=np.full(BOOTSTRAP,np.nan),where=counts>0)
    return {'matched_completed':len(merged), 'tick_minus_coarse_mean_R':float(values.mean()) if len(values) else None,
            'difference_ci95':interval(samples), 'unmatched_tick_completed':len(a)-len(merged),
            'unmatched_coarse_completed':len(b)-len(merged),
            'bootstrap_repeats':BOOTSTRAP,'bootstrap_seed':20261005,
            'valid_replicates':int(np.isfinite(samples).sum()),
            'undefined_replicates':int((~np.isfinite(samples)).sum()),'descriptive_only':True,
            'interpretation':'entry/stop/expiry quotes all change; not stop-only causal attribution'}


def sampled_clock_comparison(model, clock):
    n, sums, _, _ = daily_arrays(model)
    bn, bs, _, _ = daily_arrays(clock)
    w = np.random.default_rng(20261005).multinomial(12, np.full(12,1/12), size=BOOTSTRAP)
    count, total, bcount, btotal = w@n, w@sums, w@bn, w@bs
    means = np.divide(total,count,out=np.full(BOOTSTRAP,np.nan),where=count>0)
    base = np.divide(btotal,bcount,out=np.full(BOOTSTRAP,np.nan),where=bcount>0)
    difference = (float(sums.sum()/n.sum()-bs.sum()/bn.sum()) if n.sum() and bn.sum() else None)
    return {'mean_R_difference':difference,'difference_ci95':interval(means-base),
            'valid_replicates':int(np.isfinite(means-base).sum()),
            'comparison':'different issuance exposure; per-completed-path mean, not equal portfolios',
            'descriptive_only':True}


def stop_diagnostics(trades):
    stops = trades.loc[(trades.reason == 'sl') & ~trades.censored].copy()
    if stops.empty:
        return {'completed_stops':0,'mean_trigger_overshoot_R':None,'mean_successor_move_R':None}
    risk = stops.atr * CONFIG.stop_atr
    stop_price = stops.entry - stops.side * risk
    overshoot = -stops.side * (stops.trigger_quote-stop_price) / risk
    successor_move = stops.side * (stops.exit-stops.trigger_quote) / risk
    return {'completed_stops':len(stops),'mean_trigger_overshoot_R':float(overshoot.mean()),
            'median_trigger_overshoot_R':float(overshoot.median()),
            'mean_successor_move_R':float(successor_move.mean()),
            'max_trigger_overshoot_R':float(overshoot.max()),
            'interpretation':'observed indicative quotes, not measured broker slippage'}


def evaluate(output):
    if (output/'results.json').exists():
        raise ValueError('Refusing post-result overwrite')
    d, data = frozen_sources(output)
    selection, _ = frozen6(OLD)
    result = {'stage':'sampled_tick_execution_diagnosis', 'run_utc':datetime.now(timezone.utc).isoformat(),
              'declaration_sha256':digest(output/'declaration.json'), 'tick_sources_sha256':digest(output/'tick_sources.json'),
              'safety':assert_offline(), 'dates':DATES, 'config':asdict(CONFIG), 'symbols':{},
              'actual_money_profit':'NOT TESTED', 'prospective_paper':'NOT TESTED',
              'goal_achieved':False, 'maximum_possible_per_model':576, 'no_model_refit':True}
    flat = []
    for symbol in SYMBOLS:
        m1, _ = load_m1(ROOT/selection['sources'][symbol]['fresh']['path'])
        tick_days = {date:load_ticks(v) for date,v in data['sources'][symbol].items()}
        models, primary_ledgers = {}, {}
        for key, signal in d['sampled_signals'][symbol].items():
            signals = pd.read_csv(ROOT/signal['path'])
            signals['signal_time'] = pd.to_datetime(signals.signal_time,utc=True)
            primary, trigger, stress, barrier, audits = [],[],[],[],[]
            for date in DATES:
                start,end = day_bounds(date)
                records = signal_records(signals.loc[signals.signal_time.dt.strftime('%Y-%m-%d') == date])
                ticks = tick_days[date]
                p,a = replay_ticks(ticks,records,CONFIG,start,end,purge_minutes=31)
                t,_ = replay_ticks(ticks,records,TickExitConfig(**{**asdict(CONFIG),'stop_latency_ticks':0}),start,end,purge_minutes=31)
                c,_ = replay_timed(m1,records,TimedExitConfig(stop_atr=2.,max_hold_minutes=15),start,end,purge_minutes=31)
                b,_ = replay_timed(m1,records,TimedExitConfig(stop_atr=2.,max_hold_minutes=15,fill_mode='barrier_proxy'),start,end,purge_minutes=31)
                primary.append(p);trigger.append(t);stress.append(c);barrier.append(b);audits.append({'date':date,**a})
            p,t,c,b = (pd.concat(x,ignore_index=True) for x in (primary,trigger,stress,barrier))
            row = {'metrics':sampled_summary(p), 'trigger_quote_sensitivity':sampled_summary(t),
                   'ohlc_stress':sampled_summary(c), 'ohlc_barrier':sampled_summary(b),
                   'matched_stress_difference':matched_difference(p,c), 'matched_barrier_difference':matched_difference(p,b),
                   'stop_diagnostics':stop_diagnostics(p),
                   'audits':audits, 'historical_candidate':False, 'user_target_observed':False,
                   'rejection_reasons':['diagnostic_sample_max576_below1000','reused_adaptive_history_not_strategy_oos',
                                        'measured_costs_fills_not_tested'], 'cost_sensitivities':[]}
            if not key.startswith('CLOCK'):
                row['development_eligible']=signal['development_eligible']
                if not row['development_eligible']:
                    row['rejection_reasons'].append('round6_development_rejected')
            for cost in (0.,.05,.10,.20):
                adjusted=p.copy();adjusted['net_R']=adjusted.gross_R-cost/CONFIG.stop_atr
                row['cost_sensitivities'].append({'round_trip_cost_atr':cost,**sampled_summary(adjusted)})
            prefix=output/f'{symbol}_{key}'
            for name,frame in (('ticks',p),('trigger',t),('stress',c),('barrier',b)):
                frame.to_csv(str(prefix)+f'_{name}_trades.csv',index=False)
            models[key]=row
            primary_ledgers[key]=p
            flat.append({'symbol':symbol,'model':key,**row['metrics'],'historical_candidate':False})
            print(symbol,key,'n',row['metrics']['completed'],'PF',row['metrics']['profit_factor'],flush=True)
        for key,row in models.items():
            if not key.startswith('CLOCK'):
                mode=key.rsplit('_',1)[1]
                row['clock_comparison']=sampled_clock_comparison(primary_ledgers[key],primary_ledgers['CLOCK_'+mode])
        result['symbols'][symbol]={'models':models}
    frozen_sources(output)
    save(output/'results.json',result)
    pd.DataFrame(flat).to_csv(output/'metrics.csv',index=False)
    print('TICK_DIAGNOSIS_COMPLETE; profit target remains unproven; all live flagsfalse',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('declare','freeze-data','evaluate'),required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();assert_offline();args.output.mkdir(parents=True,exist_ok=True)
    {'declare':declare,'freeze-data':freeze_data,'evaluate':evaluate}[args.stage](args.output)


if __name__=='__main__':
    main()
