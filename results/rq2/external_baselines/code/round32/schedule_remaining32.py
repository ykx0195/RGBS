"""Resume-safe scheduling of the four unfinished comparisons at 32 rounds."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/vldb_supplement_20261102')
PY = str(ROOT / 'venv/bin/python')
STATUS = ROOT / 'remaining32_status.json'
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
jobs = [
    {'method': 'gaslite', 'dataset': 'banking77', 'objective': 'soft_threshold',
     'script': 'run_gaslite_adapter_32.py', 'run_subdir': 'gaslite_32'},
    {'method': 'gaslite_flu', 'dataset': 'banking77', 'objective': 'mean_cosine',
     'script': 'run_gaslite_flu_32.py', 'run_subdir': 'gaslite_flu_32'},
    {'method': 'gaslite_flu', 'dataset': 'dogo', 'objective': 'mean_cosine',
     'script': 'run_gaslite_flu_32.py', 'run_subdir': 'gaslite_flu_32'},
    {'method': 'gaslite_flu', 'dataset': 'abcd', 'objective': 'mean_cosine',
     'script': 'run_gaslite_flu_32.py', 'run_subdir': 'gaslite_flu_32'},
]

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def save():
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATUS)

def proc_alive(pid, script):
    try:
        root = Path('/proc') / str(pid)
        cmd = root.joinpath('cmdline').read_bytes().decode().split('\0')
        return root.joinpath('stat').read_text().split()[2] != 'Z' and any(Path(x).name == script for x in cmd if x)
    except FileNotFoundError:
        return False

if STATUS.exists():
    state = json.loads(STATUS.read_text())
    assert state['iterations'] == 32
    state.update(pid=os.getpid(), resumed_at=now())
else:
    for job in jobs:
        job.update(status='queued', attempts=[], n_iter=32,
                   out=str(ROOT / 'runs' / job['run_subdir'] / job['objective'] / job['dataset']))
    state = {'pid': os.getpid(), 'created_at': now(), 'iterations': 32,
             'max_flu_parallel': 3, 'jobs': jobs,
             'policy': 'Restart each unfinished experiment from its original natural seeds; never reuse 100-round slot archives. Other search settings and Test-free selection unchanged.'}
owned = {}
save()
while True:
    for job in state['jobs']:
        if job['status'] in ['completed', 'failed']:
            continue
        report = Path(job['out']) / 'final_report.json'
        if report.exists():
            result = json.loads(report.read_text())
            assert result['config']['n_iter'] == 32
            assert result.get('heldout_loaded') is True
            job.update(status='completed', ended_at=now(), test=result.get('test'))
            item = owned.pop(job['out'], None)
            if item:
                item[0].poll()
                item[1].close()
            print(now(), 'COMPLETE', job['method'], job['dataset'], flush=True)
            continue
        if job['status'] == 'running' and not proc_alive(job['child_pid'], job['script']):
            item = owned.pop(job['out'], None)
            code = None
            if item:
                code = item[0].poll()
                item[1].close()
            job['attempts'].append({'exit_code': code, 'ended_at': now()})
            job.update(status='retry_wait' if len(job['attempts']) < 2 else 'failed',
                       retry_after=time.time() + 120)
    if all(j['status'] in ['completed', 'failed'] for j in state['jobs']):
        state.update(phase='complete' if all(j['status'] == 'completed' for j in state['jobs']) else 'has_failures', ended_at=now())
        save()
        break
    running = [j for j in state['jobs'] if j['status'] == 'running']
    base_running = sum(j['method'] == 'gaslite' for j in running)
    flu_running = sum(j['method'] == 'gaslite_flu' for j in running)
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
    state.update(free_gpu_mib=free, phase='running' if running else 'waiting_gpu_memory')
    for job in state['jobs']:
        if job['status'] not in ['queued', 'retry_wait'] or time.time() < job.get('retry_after', 0):
            continue
        # Reserve future GPU quality-model allocation of the base worker.
        minimum = 6500 if job['method'] == 'gaslite' else 3500 + 3000 * base_running
        job['required_free_mib'] = minimum
        if job['method'] == 'gaslite_flu' and flu_running >= 3:
            continue
        if free < minimum:
            continue
        out = Path(job['out'])
        out.mkdir(parents=True, exist_ok=True)
        logpath = ROOT / 'logs' / f"{job['method']}_32_{job['dataset']}_{job['objective']}_attempt{len(job['attempts'])}.log"
        log = logpath.open('a')
        command = [PY, '-u', str(ROOT / 'code_round32' / job['script']),
                   '--dataset', job['dataset'], '--objective', job['objective']]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned[job['out']] = (proc, log)
        job.update(status='running', started_at=now(), child_pid=proc.pid, log=str(logpath), command=command)
        print(now(), 'START', job['method'], job['dataset'], proc.pid, 'n_iter=32', flush=True)
        save()
        time.sleep(20)
        break
    save()
    time.sleep(10)
