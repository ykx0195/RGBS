"""Verify exported evidence without model inference or API calls."""
from pathlib import Path
import ast
import csv
import hashlib
import json

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
print(f'PASS: {len(hashes)} file checksums, Python syntax, and all three main result counts/frozen sets.')
