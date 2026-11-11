import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = Path.home() / 'Desktop/results'
DATA = Path.home() / 'Desktop/旧/liman/data'
OLD = RESULTS / '跨编码器迁移_本地CPU_仅总覆盖_20261031'
INPUT = ROOT / 'inputs'
INPUT.mkdir(exist_ok=True)

def rows(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {'api_calls': 0, 'calibration': 'All unordered distinct-source deidentified Validation request pairs; match each dataset BGE cosine>=0.85 pair count; nearest attainable count with conservative tie break.', 'datasets': {}}
for dataset, folder, sizes, k, hits in [
    ('banking77', 'banking77_pii_delexed', [4924,875,1485],15,1109),
    ('abcd','abcd_pii_delexed_v2',[3834,480,476],5,308),
    ('dogo','multidogo_pii_delexed_v2',[3993,535,1094],10,909),
]:
    splitrows = {split: rows(DATA / folder / (split+'.jsonl')) for split in ['train','validation','test']}
    assert [len(splitrows[s]) for s in splitrows] == sizes
    ids = {s: {r['source_id'] for r in rr} for s,rr in splitrows.items()}
    assert all(len(ids[s]) == len(splitrows[s]) for s in ids)
    assert not ids['validation'] & (ids['train'] | ids['test'])
    assert not ids['train'] & ids['test']
    if dataset == 'dogo':
        source = RESULTS / 'MultiDoGO_8309_跨编码器_CPU_20261103/frozen_inputs.json'
        keys = json.loads(source.read_text('utf-8'))['frozen_keys']
    else:
        source = OLD / 'frozen_inputs' / dataset / 'frozen_keys.json'
        keys = json.loads(source.read_text('utf-8'))['key_texts']
    assert len(keys) == k
    payload = {'keys': keys, 'validation':splitrows['validation'], 'test':splitrows['test'], 'expected_fixed085_hits':hits}
    path = INPUT / (dataset+'.json')
    path.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    manifest['datasets'][dataset] = {'k':k,'split_sizes':sizes,'expected_fixed085_hits':hits,'source_keys':str(source),'source_keys_sha256':sha(source),'validation_sha256':sha(DATA/folder/'validation.jsonl'),'test_sha256':sha(DATA/folder/'test.jsonl'),'input_sha256':sha(path),'disjoint_source_ids':True}
(INPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copyfile(OLD/'model_manifest.json',INPUT/'expected_models.json')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
