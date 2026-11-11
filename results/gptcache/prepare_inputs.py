import hashlib, json, shutil, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = Path(r'C:\Users\ykx01\Desktop\results')
DATA = Path(r'C:\Users\ykx01\Desktop\ykx\data')
manifest = {'scope': 'deidentified main portfolios; direct preload; read-only Test; marker payloads, not real answers', 'datasets': {}}
for name, folder, expected in [('banking77', 'banking77', 1109), ('abcd', 'abcd', 308), ('dogo', 'multidogo', 909)]:
    out = ROOT / 'inputs' / name
    out.mkdir(parents=True, exist_ok=True)
    if name == 'dogo':
        src = RESULTS / 'MultiDoGO_8309_模块与重选_CPU_20261103/input/portfolios.json'
        selected = json.loads(src.read_text('utf-8'))[0]['selected']
    else:
        src = RESULTS / '三数据集_脱敏_32x50_20261030/runs' / name / 'final_report.json'
        selected = json.loads(src.read_text('utf-8'))['selected']
    keys = [{'text': x['text'], 'key_id': f'{name}:constructed:{i}', 'category': x.get('intent_label')} for i,x in enumerate(selected)]
    (out / 'constructed.json').write_text(json.dumps(keys, ensure_ascii=False, indent=2), 'utf-8')
    for split in ['train', 'test']:
        shutil.copyfile(DATA / folder / 'desensitized' / (split+'.jsonl'), out / (split+'.jsonl'))
    manifest['datasets'][name] = {'expected_constructed_hits': expected, 'k': len(keys), 'portfolio_source': str(src), 'portfolio_source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
(ROOT / 'inputs' / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
with tarfile.open(ROOT / 'inputs.tar.gz', 'w:gz') as t:
    t.add(ROOT / 'inputs', arcname='inputs')
print(json.dumps({k: {'k': v['k'], 'expected_hits': v['expected_constructed_hits']} for k,v in manifest['datasets'].items()}))
