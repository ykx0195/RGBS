"""Wait for the pinned 7B model, smoke it, then run and score all 30 clusters."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/advdec_baseline_20261031')
PYTHON = Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031/venv/bin/python')
WORKER = ROOT/'code'/'advdec_worker.py'
QUALITY = ROOT/'code'/'advdec_quality.py'

def status(stage, **fields):
    payload = {'stage': stage, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **fields}
    path = ROOT/'supervisor_status.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)
    print('ADVDEC_STATUS', stage, fields, flush=True)

def run(argv, log):
    with (ROOT/log).open('a', encoding='utf-8') as output:
        result = subprocess.run([str(PYTHON), '-u', str(argv[0]), *argv[1:]],
                                stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
    if result.returncode:
        tail = (ROOT/log).read_text(encoding='utf-8', errors='replace')[-3000:]
        raise RuntimeError(f'{log} exit={result.returncode}\n{tail}')

try:
    marker = ROOT/'local_model_provenance.json'
    status('waiting_for_verified_model')
    while not marker.exists():
        time.sleep(60)
    provenance = json.loads(marker.read_text(encoding='utf-8'))
    assert provenance['all_match'] and provenance['revision'] == 'a09a35458c702b33eeacc393d103063234e8bc28'
    model = ROOT/'models'/'Qwen2.5-7B-Instruct'
    assert all((model/x['name']).stat().st_size == x['bytes'] for x in provenance['files'])
    status('gpu_smoke_running')
    run([WORKER, '--dataset', 'banking77', '--cluster', '0', '--smoke'], 'smoke_7b.log')
    smoke = json.loads((ROOT/'smoke'/'banking77'/'cluster_00'/'final_cluster.json').read_text(encoding='utf-8'))
    assert smoke['complete_sentence'] and smoke['heldout_loaded'] is False
    assert smoke['gpu_max_allocated_mib'] < 23500
    status('gpu_smoke_passed', smoke_text=smoke['selected']['text'],
           gpu_max_allocated_mib=smoke['gpu_max_allocated_mib'])
    status('formal_gpu_search_running')
    run([WORKER, '--all'], 'formal.log')
    finals = list((ROOT/'runs').glob('*/cluster_*/final_cluster.json'))
    assert len(finals) == 30
    status('formal_gpu_search_complete', clusters=30)
    run([QUALITY], 'quality.log')
    assert json.loads((ROOT/'quality_results.json').read_text(encoding='utf-8'))['complete']
    status('search_and_ppl_complete', clusters=30)
    archive = ROOT/'completed_advdec_search.tar.gz'
    subprocess.run(['tar', '-czf', str(archive), '-C', str(ROOT),
                    'runs', 'smoke', 'quality_results.json', 'experiment_plan.json',
                    'code', 'supervisor_status.json', 'formal.log', 'smoke_7b.log',
                    'quality.log', 'local_model_provenance.json'], check=True)
    status('remote_archive_complete', archive_bytes=archive.stat().st_size)
except Exception as exc:
    status('failed', error_type=type(exc).__name__, error=str(exc)[-3500:])
    raise
