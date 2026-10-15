"""Serial read-only source acquisition; the final partition has no measurement API.

The explicitly pinned local launcher reuses an existing terminal account with
both trading permissions disabled. A monitoring timeout is never a restart.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_mt5_quote_export import verify_export

DATA=ROOT/'data/independent_profit_ticks_20261013'
LAUNCHER=ROOT/'work/independent_profit_pilot_20261013/run_dataset_probe.py'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temporary=path.with_suffix('.new')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def terminal_pids():
    rows=subprocess.run(['ps','-axo','pid=,comm=,args='],capture_output=True,text=True,check=True).stdout.splitlines()
    return [int(line.split(None,1)[0]) for line in rows if
            ('wine64-preloader' in line or 'wine-preloader' in line or
             'terminal64.exe' in line.split(None,2)[1]) and 'terminal64.exe' in line]

def main():
    os.umask(0o077)
    lock=(DATA/'.collection.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    declaration=DATA/'acquisition_declaration.json'
    spec=json.loads(declaration.read_text())
    for relative,pinned in spec['source_pins'].items():
        if digest(ROOT/relative)!=pinned:
            raise RuntimeError('Source acquisition pin changed: '+relative)
    state_path=DATA/'acquisition_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'status':'RUNNING','declaration_sha256':digest(declaration),'records':[],
        'strategy_measurements':0,'final_metrics_opened':False}
    if state['declaration_sha256']!=digest(declaration):
        raise RuntimeError('Acquisition declaration changed')
    state.update(status='RUNNING',owner_pid=os.getpid())
    completed={r['ordinal']:r for r in state['records']}
    for item in spec['requests']:
        folder=DATA/item['partition']/item['prefix']
        if item['ordinal'] in completed:
            old=completed[item['ordinal']]
            if any(digest(ROOT/path)!=pinned for path,pinned in old['files'].items()):
                raise RuntimeError('Retained receipt changed; no automatic repair')
            continue
        if terminal_pids():
            state.update(status='WAITING_EXISTING_TERMINAL',pending=item)
            save(state_path,state)
            raise RuntimeError('A real terminal process is alive; do not start another')
        if folder.exists():
            state.update(status='REQUIRES_INCOMPLETE_PREFIX_INSPECTION',pending=item)
            save(state_path,state)
            raise RuntimeError('Inspect incomplete prefix and source terminal before resuming')
        if shutil.disk_usage(DATA).free<3*1024**3:
            state.update(status='WAITING_DISK_SPACE',pending=item)
            save(state_path,state)
            raise RuntimeError('Disk reserve reached; preserving every retained source')
        partition=DATA/item['partition'];partition.mkdir(exist_ok=True)
        log=DATA/(item['prefix']+'.observer.log')
        state.update(pending=item,updated_utc=datetime.datetime.now(datetime.UTC).isoformat())
        save(state_path,state)
        with log.open('xb') as handle:
            result=subprocess.run([sys.executable,'-u',str(LAUNCHER),
                '--symbol',('Boom' if item['symbol']=='BOOM500' else 'Crash')+' 500 Index',
                '--start',item['start_utc'],'--seconds',str(item['seconds']),
                '--prefix',item['prefix'],'--output-root',str(partition)],
                stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='OBSERVATION_ENDED_CHECK_REAL_PROCESS',observer_exit=result.returncode)
            save(state_path,state)
            raise RuntimeError('Observation ended; process state must be checked before recovery')
        receipt=folder/(item['prefix']+'_receipt.tsv')
        verified=verify_export(receipt)
        verification=folder/'independent_verification.json'
        save(verification,verified)
        raw=folder/(item['prefix']+'_ticks.tsv')
        record={**item,'verification_status':verified['status'],
            'rows':verified['rows'],'gaps_above_5000ms':verified['inter_tick_gap_above_5000ms'],
            'leading_unobserved_ms':verified['leading_unobserved_ms'],
            'trailing_unobserved_ms':verified['trailing_unobserved_ms'],
            'raw_bytes':raw.stat().st_size if raw.exists() else 0,
            'files':{str(file.relative_to(ROOT)):digest(file) for file in
                     [receipt,verification]+([raw] if raw.exists() else [])}}
        state['records'].append(record)
        state.update(pending=None,updated_utc=datetime.datetime.now(datetime.UTC).isoformat())
        save(state_path,state)
        if verified['status']!='QUOTES_VALIDATED_COVERAGE_NOT_PROVEN':
            state['status']='SOURCE_QUALITY_FAILED';save(state_path,state)
            raise RuntimeError('Source quality failed; never substitute or tune a different date')
        print(f"{len(state['records'])}/300 {item['symbol']} day{item['day_index']} rows={verified['rows']} gaps={record['gaps_above_5000ms']}",flush=True)
    state.update(status='COMPLETED',pending=None,updated_utc=datetime.datetime.now(datetime.UTC).isoformat())
    save(state_path,state)
    print('Source acquisition completed; final measurements remain sealed',flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        path=DATA/'acquisition_state.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('owner_pid')==os.getpid() and state.get('status')=='RUNNING':
                state.update(status='RUNNER_ENDED_INSPECT_ERROR',error_type=type(error).__name__,
                             updated_utc=datetime.datetime.now(datetime.UTC).isoformat())
                save(path,state)
        raise
