"""Independent float64 recomputation, fixed-key identity, and per-query pairing."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(ROOT/'verification.json');assert manifest['passed']
for source,digest in manifest['source_files'].items():assert sha(Path(source))==digest
summary=read(ROOT/'summary.json');checks=[]
for s in summary:
    out=ROOT/s['dataset'];f=read(out/'fixed_keys.json');keys=np.load(f['selected_vectors_source'])
    qp=next(Path(src) for src,digest in manifest['source_files'].items() if Path(src).name=='embeddings.npz' and digest==s['query_vectors_sha256'])
    z=np.load(qp)
    raw=np.einsum('ij,kj->ik',z['raw'].astype(np.float64),keys.astype(np.float64),optimize=False)
    delex=np.einsum('ij,kj->ik',z['delex'].astype(np.float64),keys.astype(np.float64),optimize=False)
    rh=(raw>=.85).any(axis=1);dh=(delex>=.85).any(axis=1)
    with (out/'paired_queries.csv').open(encoding='utf-8-sig',newline='') as stream:rr=list(csv.DictReader(stream))
    assert rh.tolist()==[bool(int(r['original_covered'])) for r in rr]
    assert dh.tolist()==[bool(int(r['deidentified_covered'])) for r in rr]
    assert int(rh.sum())==s['original_covered'] and int(dh.sum())==s['deidentified_covered']
    assert int((dh&~rh).sum())==s['newly_covered'] and int((rh&~dh).sum())==s['lost_coverage']
    assert np.isclose(100*float((dh.astype(int)-rh.astype(int)).mean()),s['change_pp'])
    checks.append({'dataset':s['dataset'],'queries_verified':len(rr),'original_hits':int(rh.sum()),
       'deidentified_hits':int(dh.sum()),'same_per_query_results_float64':True,'passed':True})
result={'passed':True,'source_hashes_unchanged':True,'datasets':checks,'queries_verified':sum(c['queries_verified'] for c in checks),
        'verify_script_sha256':sha(Path(__file__))}
(ROOT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
