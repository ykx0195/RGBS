"""Resume pinned Qwen shards from a reachable mirror, then SHA-verify every file."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
import threading
import time

ROOT = Path('/root/autodl-tmp/advdec_baseline_20261031')
MODEL = ROOT/'models'/'Qwen2.5-7B-Instruct'
PROVENANCE = ROOT/'mirror_source_provenance.json'
data = json.loads(PROVENANCE.read_text(encoding='utf-8'))
assert data['all_match'] and data['repo'] == 'Qwen/Qwen2.5-7B-Instruct'
revision = data['revision']
assert revision == 'a09a35458c702b33eeacc393d103063234e8bc28'
files = data['files']
weights = [item for item in files if item['name'].endswith('.safetensors')]
assert len(weights) == 4


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def progress(stop):
    while not stop.wait(30):
        rows = []
        for item in weights:
            path = MODEL/(item['name']+'.uploading')
            final = MODEL/item['name']
            done = final.stat().st_size if final.exists() else (path.stat().st_size if path.exists() else 0)
            rows.append((item['name'], done, item['bytes']))
        print('MIRROR_PROGRESS', rows, flush=True)


def fetch(item):
    name = item['name']
    final = MODEL/name
    partial = MODEL/(name+'.uploading')
    if final.exists() and final.stat().st_size == item['bytes'] and sha(final) == item['sha256']:
        print('MIRROR_ALREADY_VERIFIED', name, flush=True)
        return
    assert not final.exists() or final.stat().st_size != item['bytes']
    if partial.exists():
        assert 0 <= partial.stat().st_size <= item['bytes']
    url = f'https://hf-mirror.com/{data["repo"]}/resolve/{revision}/{name}'
    for attempt in range(1, 5):
        current = partial.stat().st_size if partial.exists() else 0
        print('MIRROR_RESUME', name, current, item['bytes'], 'attempt', attempt, flush=True)
        command = ['curl', '--location', '--fail', '--silent', '--show-error',
                   '--retry', '8', '--retry-all-errors', '--retry-delay', '5',
                   '--connect-timeout', '15', '--max-time', '7200',
                   '--continue-at', '-', '--output', str(partial), url]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            print('MIRROR_CURL_RETRY', name, result.returncode, result.stderr[-400:], flush=True)
            time.sleep(10*attempt)
            continue
        if partial.stat().st_size != item['bytes']:
            raise RuntimeError(f'{name} size mismatch: {partial.stat().st_size} != {item["bytes"]}')
        actual = sha(partial)
        if actual != item['sha256']:
            raise RuntimeError(f'{name} sha256 mismatch: {actual} != {item["sha256"]}')
        partial.replace(final)
        print('MIRROR_SHARD_VERIFIED', name, flush=True)
        return
    raise RuntimeError(f'Failed to fetch {name} after retries')


started = time.monotonic()
stop = threading.Event()
reporter = threading.Thread(target=progress, args=(stop,), daemon=True)
reporter.start()
try:
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, item) for item in weights]
        for future in as_completed(futures):
            future.result()
finally:
    stop.set()
    reporter.join(timeout=2)

for item in files:
    path = MODEL/item['name']
    assert path.stat().st_size == item['bytes'], item['name']
    assert sha(path) == item['sha256'], item['name']
marker = ROOT/'local_model_provenance.json'
temporary = marker.with_suffix('.json.tmp')
temporary.write_bytes(PROVENANCE.read_bytes())
temporary.replace(marker)
print('MIRROR_MODEL_COMPLETE', len(files), 'files', round(time.monotonic()-started), 'seconds', flush=True)
