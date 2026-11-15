"""Verify exported evidence without model inference or API calls."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re

root = Path(__file__).resolve().parents[1]
hashes = json.loads((root / 'ARTIFACT_SHA256.json').read_text(encoding='utf-8'))
for name, expected in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for path in root.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
rows = list(csv.DictReader((root / 'results/main/coverage.csv').open(encoding='utf-8')))
expected = {'banking77': (15,1109,1485), 'abcd': (5,308,476), 'dogo': (10,909,1094)}
for row in rows:
    ds=row['dataset']; k,c,n=expected[ds]
    assert (int(row['budget']),int(row['covered']),int(row['test_total'])) == (k,c,n)
    assert abs(float(row['coverage_percent'])-100*c/n)<1e-9
    frozen=json.loads((root / f'results/main/{ds}/frozen_texts.json').read_text(encoding='utf-8'))
    assert len(frozen)==k, (ds,len(frozen),k)
assert {r['dataset'] for r in rows} == set(expected)
for row in rows:
    report=json.loads((root / row['source_report']).read_text(encoding='utf-8'))
    measured=report['reports'][0]['text_test'] if row['dataset']=='dogo' else report['metrics']['test']
    assert (measured['covered'],measured['total']) == (int(row['covered']),int(row['test_total']))
index=json.loads((root / 'PAPER_RESULTS_INDEX.json').read_text(encoding='utf-8'))
for references in index['results'].values():
    for name in references: assert (root / name).is_file(),name
secret=re.compile(rb'(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{24,}|github_pat_[A-Za-z0-9_]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
for name in hashes:
    assert not secret.search((root / name).read_bytes()), f'Credential pattern in {name}'
for name in ['code/search/run_configurable.py','code/search/run_configurable_resume.py','code/experiments/paired_evaluation/remote_fixed_task_audit.py']:
    assert 'key_file' not in (root / name).read_text(encoding='utf-8'),name
assert len(list((root / 'paper_tables').glob('*.tex'))) == 8
print(f'PASS: {len(hashes)} checksums; syntax; main reports/frozen sets; paper references; credential-pattern and environment-only loading checks.')
