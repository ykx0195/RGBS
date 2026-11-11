"""Validate matrix coverage, source splits, file hashes and independent recounts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,csv,hashlib,itertools,json,re,math
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();job=a.job;out=job/'output'
with (out/'collision_results_all_thresholds.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
datasets=['banking77','abcd','multidogo'];models=['minilm','gte','e5','bge'];families=['dataset_fields','presidio'];methods=['generic','typed','typed_numbered','delete','mask_length']
ops=[('fixed',x) for x in [.75,.775,.8,.825,.85,.875,.9]]+[('original_tail',x) for x in [.001,.01]]
expected=set(itertools.product(datasets,models,families,methods,ops))
actual={(r['dataset'],r['model'],r['family'],r['method'],(r['threshold_kind'],float(r['target']))) for r in rows}
assert len(actual)==len(rows)==1080 and actual==expected
for r in rows:
    n=int(r['n_test_records']);den=int(r['n_test_pairs']);b=int(r['original_above_count']);c=int(r['modified_above_count']);up=int(r['upward_count']);down=int(r['downward_count'])
    assert den==n*(n-1)//2 and up-down==c-b==int(r['net_count'])
    assert abs(float(r['delta_pp'])-100*(c-b)/den)<1e-10
    assert float(r['record_jackknife_ci95_low'])<=float(r['net_change'])<=float(r['record_jackknife_ci95_high'])
metadata=0
for d in (out/'embeddings').iterdir():
    for meta in d.glob('*.json'):
        m=json.loads(meta.read_text())
        if 'embedding_sha256' not in m:continue
        arr=meta.with_suffix('.npy');assert sha(arr)==m['embedding_sha256']
        z=np.load(arr,mmap_mode='r');assert z.shape==(m['records'],384) and np.isfinite(z).all();metadata+=1
cohort=json.loads((out/'cohort_audit.json').read_text());recounts=0
for ds in datasets:
    ids=cohort[ds]['test_record_ids'];n=len(ids)
    assert n==cohort[ds]['test_original_records'], 'Check index mapping explicitly if filtering occurs.'
    tri=np.triu_indices(n,1)
    for model in models:
        d=out/'embeddings'/model;x=np.load(d/f'{ds}_original.npy');before=x@x.T>=.85
        for family in families:
            z=np.load(d/f'{ds}_{family}_generic.npy');after=z@z.T>=.85
            row=next(r for r in rows if (r['dataset'],r['model'],r['family'],r['method'],r['threshold_kind'],float(r['target']))==(ds,model,family,'generic','fixed',.85))
            assert int(before[tri].sum())==int(row['original_above_count'])
            assert int(after[tri].sum())==int(row['modified_above_count'])
            # Recompute delete-one record rates directly for a sample of records.
            delta=after.astype(np.int8)-before.astype(np.int8)
            total=int(delta[tri].sum());inc=delta.sum(1)-np.diag(delta)
            for i in [0,n//2,n-1]:
                sub=np.delete(np.delete(delta,i,axis=0),i,axis=1)
                direct=float(sub[np.triu_indices(n-1,1)].mean())
                fast=(total-inc[i])/((n-1)*(n-2)/2)
                assert abs(direct-fast)<1e-12
            recounts+=1
rr=[json.loads(l) for l in (job/'input/request_records.jsonl').open()]
ambiguous=[];test_aligned=0
for r in rr:
    t=r['legacy_delex'];marks=list(re.finditer(r'<([A-Z][A-Z0-9_]*?)_(\d+)>',t));pat='';cur=0
    for m in marks:pat+=re.escape(t[cur:m.start()])+'(.+)';cur=m.end()
    pat+=re.escape(t[cur:]);g=re.fullmatch(pat,r['source_text'],re.S)
    unique=g is not None and [(g.start(i+1),g.end(i+1)) for i in range(len(marks))]==[(s['start'],s['end']) for s in r['spans']]
    if not unique:ambiguous.append({'dataset':r['dataset'],'split':r['split'],'record_id':r['record_id']})
    if r['split']=='test':assert unique;test_aligned+=1
assert all(x['split']=='train' for x in ambiguous)
for ds in datasets:
    tr={r['source_id'] for r in rr if r['dataset']==ds and r['split']=='train'}
    te={r['source_id'] for r in rr if r['dataset']==ds and r['split']=='test'}
    assert not tr&te
report={'matrix_rows':len(rows),'missing_cells':0,'duplicate_cells':0,'verified_embedding_files':metadata,
    'independent_generic_recounts':recounts,'sampled_direct_delete_one_checks':recounts*3,
    'unique_test_field_alignments':test_aligned,'ambiguous_train_field_alignments':ambiguous,
    'alignment_note':'One ambiguous training-only field boundary is unused: calibration encodes original Train text only. All 3055 transformed Test records have unique recoverable boundaries.',
    'record_count_identity':True,'source_disjoint_train_test':True,
    'test_records_retained':{ds:cohort[ds]['test_retained'] for ds in datasets},
    'positive_primary_cells':sum(float(r['delta_pp'])>0 for r in rows if r['threshold_kind']=='fixed' and float(r['target'])==.85),
    'negative_primary_cells':sum(float(r['delta_pp'])<0 for r in rows if r['threshold_kind']=='fixed' and float(r['target'])==.85)}
(out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
# Manifest covers transferable input, code, logs and all output artifacts.
files=[p for folder in ['code','output'] for p in (job/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [job/'input/request_records.jsonl',job/'input/input_manifest.json']
files += list(job.glob('*.log'))
checks={str(p.relative_to(job)):sha(p) for p in sorted(files)}
(job/'checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
