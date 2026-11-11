"""Post-search PPL-only screen with the same local model used for HotFlip/RKGB."""
from pathlib import Path
import hashlib
import json
import sys
import time

import torch

ROOT = Path('/root/autodl-tmp/advdec_baseline_20261031')
HOTFLIP = Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031')
sys.path.insert(0, str(HOTFLIP/'code'))
from hybrid_search import Fluency

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
lm_dir = Path('/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct')
quality = Fluency(lm_dir, 'cuda')
results = []
start = time.monotonic()
for dataset, k in {'banking77': 15, 'abcd': 5, 'dogo': 10}.items():
    for cluster in range(k):
        final = json.loads((ROOT/'runs'/dataset/f'cluster_{cluster:02d}'/'final_cluster.json').read_text(encoding='utf-8'))
        text = final['selected']['text']
        ppl = float(quality.ppl(text))
        results.append({'dataset': dataset, 'method': 'AdvDec_adapted', 'slot': cluster,
                        'text': text, 'ppl': ppl, 'ppl_at_most_150': ppl <= 150.0})
        print('ADVDEC_PPL_PROGRESS', len(results), 30, round(ppl, 2), flush=True)
assert len(results) == 30
weights = {}
for path in sorted(lm_dir.glob('*.safetensors')):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    weights[path.name] = digest.hexdigest()
payload = {'complete': True, 'protocol': 'Qwen2.5-1.5B-Instruct local PPL<=150, no readability hard gate.',
           'results': results, 'model_weight_sha256': weights, 'device': 'cuda', 'api_calls': 0,
           'elapsed_seconds': time.monotonic()-start}
path = ROOT/'quality_results.json'
tmp = path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
tmp.replace(path)
print('ADVDEC_PPL_COMPLETE', sum(x['ppl_at_most_150'] for x in results), flush=True)
