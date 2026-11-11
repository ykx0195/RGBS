"""Run the three materialized datasets at BGE 0.85, 32 rounds, beam 50."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone


ROOT = Path('/root/autodl-tmp/ykx_materialized_t085_32x50_20261031')
CODE = ROOT / 'code/run_global_round_budget.py'
MODEL = Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5')
LM = Path('/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct')
API = Path('/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json')
JOBS = {'banking77': 'cuda', 'abcd': 'cpu', 'dogo': 'cpu'}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(status: dict) -> None:
    p = ROOT / 'queue_status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, p)


def gpu_used() -> int:
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], check=True,
                       capture_output=True, text=True)
    return int(p.stdout.strip().splitlines()[0])


def args_for(dataset: str, device: str) -> list[str]:
    folder = 'multidogo' if dataset == 'dogo' else dataset
    return ['/root/miniconda3/bin/python', '-u', str(CODE),
            '--dataset', dataset, '--representation', 'non_desensitized',
            '--encoder', 'bge', '--threshold', '0.85', '--seed', '42',
            '--data-dir', str(ROOT / 'data' / folder / 'non_desensitized'),
            '--model-dir', str(MODEL), '--lm-dir', str(LM),
            '--api-config', str(API), '--out-dir', str(ROOT / 'runs' / dataset),
            '--device', device, '--steps-per-slot', '32', '--beam-width', '50']


def main() -> None:
    status_path = ROOT / 'queue_status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
    else:
        status = {'started_at_utc': now(), 'server_port': 34868,
                  'protocol': 'Materialized Train/Test; same BGE/quality gate as desensitized 32x50; Train-only global selection; final Test',
                  'jobs': {name: {'status': 'pending', 'device': device, 'attempts': 0}
                           for name, device in JOBS.items()}}
    save(status)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    retry_after: dict[str, float] = {}
    while True:
        for name, (proc, stream) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            stream.close()
            del running[name]
            entry = status['jobs'][name]
            report = ROOT / 'runs' / name / 'final_report.json'
            if rc == 0 and report.is_file():
                obj = json.loads(report.read_text(encoding='utf-8'))
                if obj.get('portfolio_source') != 'global_greedy_swap':
                    raise RuntimeError(f'Wrong portfolio selection in {name}')
                entry.update(status='complete', finished_at_utc=now(),
                             train=obj['metrics']['train'], test=obj['metrics']['test'])
                print(now(), 'COMPLETE', name, entry['test'], flush=True)
            elif entry['attempts'] < 3:
                entry.update(status='retry_wait', exit_code=rc, failed_at_utc=now())
                retry_after[name] = time.monotonic() + 120 * entry['attempts']
                print(now(), 'RETRY_WAIT', name, rc, flush=True)
            else:
                entry.update(status='failed', exit_code=rc, failed_at_utc=now())
                print(now(), 'FAILED', name, rc, flush=True)
            save(status)

        for name, device in JOBS.items():
            if name in running or status['jobs'][name]['status'] in ('complete', 'failed'):
                continue
            if time.monotonic() < retry_after.get(name, 0):
                continue
            if device == 'cuda' and gpu_used() > 13000:
                continue
            output = ROOT / 'runs' / name
            output.mkdir(parents=True, exist_ok=True)
            threads = '4' if device == 'cuda' else '16'
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
                       TOKENIZERS_PARALLELISM='false')
            stream = (output / 'run.log').open('a', encoding='utf-8')
            entry = status['jobs'][name]
            entry.update(status='running', started_at_utc=now(),
                         attempts=entry['attempts'] + 1)
            stream.write(f"\n=== {now()} attempt {entry['attempts']} device={device} ===\n")
            stream.flush()
            proc = subprocess.Popen(args_for(name, device), cwd=ROOT / 'code',
                                    env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            entry['pid'] = proc.pid
            running[name] = (proc, stream)
            save(status)
            print(now(), 'START', name, device, proc.pid, flush=True)
            if device == 'cuda':
                time.sleep(15)

        if not running and all(x['status'] in ('complete', 'failed')
                               for x in status['jobs'].values()):
            status['finished_at_utc'] = now()
            save(status)
            print(now(), 'FINISHED', flush=True)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
