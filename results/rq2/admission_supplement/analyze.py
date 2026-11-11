"""Report ONLY the new strict-pairwise portfolios' completed admission outcomes."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'strict085_admission/server_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(SOURCE/'outputs/independent_verification.json')['passed']
summary=[];query_rows=[];sources={}
for ds in ['banking77','abcd','dogo']:
    p=SOURCE/'outputs'/ds/'placeholder_saved_order';raw=SOURCE/'outputs'/ds/'materialized_saved_order'
    s=read(p/'summary.json');rr=lines(p/'query_results.jsonl');rawrr=lines(raw/'query_results.jsonl')
    assert [(r['source_id'],r['cache_hit']) for r in rr]==[(r['source_id'],r['cache_hit']) for r in rawrr]
    assert len({r['source_id'] for r in rr})==len(rr)
    events=read(p/'admission_events.json');rawevents=read(raw/'admission_events.json')
    assert all(e['written'] and not e['hit'] and e['representation_equal'] for e in events+rawevents)
    mask=np.array([r['cache_hit'] for r in rr]);n=len(mask);rng=np.random.default_rng(20261105)
    boot=[]
    for start in range(0,5000,100):
        idx=rng.integers(0,n,size=(100,n));boot.extend(mask[idx].mean(axis=1)*100)
    low,high=np.quantile(boot,[.025,.975])
    record={'dataset':ds,'submitted':s['submitted_candidates'],'admitted':s['admitted_entries'],
       'test_hits':int(mask.sum()),'test_total':n,'coverage_pct':100*float(mask.mean()),
       'coverage_ci95_low':float(low),'coverage_ci95_high':float(high),
       'max_pairwise_cosine':s['pairwise_max'],'strict_pairwise_constraint':True,
       'materialized_and_placeholder_agree':True}
    assert record['submitted']==record['admitted'] and s['early_hits']==0
    assert record['max_pairwise_cosine']<.85 and record['test_hits']==s['offline_expected_hits']
    summary.append(record)
    for r in rr:query_rows.append({'dataset':ds,'source_id':r['source_id'],'cache_hit':int(r['cache_hit'])})
    for folder in [p,raw]:
        for name in ['summary.json','query_results.jsonl','admission_events.json','entries.json','cache.sqlite']:
            f=folder/name;sources[str(f)]=sha(f)
save(ROOT/'summary.json',summary)
for name,records in [('summary.csv',summary),('query_outcomes.csv',query_rows)]:
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
save(ROOT/'verification.json',{'passed':True,'scope':'new strict-pairwise candidate portfolios only',
   'source_files':sources,'source_verification_sha256':sha(SOURCE/'outputs/independent_verification.json'),
   'unique_queries':len(query_rows),'actual_service_queries_two_conditions':2*len(query_rows),
   'bootstrap_replicates':5000,'bootstrap_seed':20261105,'interval':'95% percentile bootstrap of source-ID-aligned Test records',
   'uncertainty_scope':'conditional on the fixed portfolios; excludes search randomness and service/deployment variation',
   'additional_api_calls':0,'additional_generation_search':0,'script_sha256':sha(Path(__file__))})
print(json.dumps(summary))
