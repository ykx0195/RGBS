"""Fixed newly admitted keys, original/deidentified source-paired queries. No API or encoding."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
RESULTS=Path(r'C:\Users\ykx01\Desktop\results')
DATA=Path(r'C:\Users\ykx01\Desktop\ykx\data')
SERVICE=WORK/'strict085_admission/server_evidence'
PAIRED=RESULTS/'固定文本_原文脱敏配对_20261031'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

protocol={'scope':'only new strict-pairwise portfolios, already sequentially admitted',
 'fixed':'identical cache strings, vectors, order and budget; BGE encoder; cosine >=0.85',
 'variable':'query-side original versus deidentified representation of the same Test records',
 'original_condition':'offline counterfactual: encode original query without deidentification; keys stay deidentified',
 'deidentified_condition':'offline score plus equality check against recorded real-service query outcomes',
 'selection_or_generation':False,'new_encoding':False,'new_api_calls':0,'service_rerun':False,
 'bootstrap':{'replicates':5000,'seed':20261105,'interval':'95% paired record-level percentile bootstrap, same indices in both conditions',
 'scope':'conditional on fixed keys and Test collection; excludes search and deployment variation'},'script_sha256':sha(Path(__file__))}
save(ROOT/'protocol.json',protocol)
assert read(SERVICE/'outputs/independent_verification.json')['passed']
summary=[];source_files={}
for ds,folder,cachefolder in [('banking77','banking77',PAIRED/'banking77'),('abcd','abcd',PAIRED/'ABCD补充/abcd'),('dogo','multidogo',PAIRED/'dogo')]:
    out=ROOT/ds;out.mkdir(exist_ok=True)
    def track(p):source_files[str(p)]=sha(p);return p
    identity=read(track(cachefolder/'embedding_identity.json'))
    assert identity['model_sha256']==read(SERVICE/'outputs/protocol.json')['model_sha256']
    rawpath=track(DATA/folder/'non_desensitized/test.jsonl');delexpath=track(DATA/folder/'desensitized/test.jsonl')
    assert sha(rawpath)==identity['raw_test_sha256'] and sha(delexpath)==identity['delex_test_sha256']
    raw,delex=rows(rawpath),rows(delexpath)
    assert len(raw)==len(delex)
    assert len({r['source_id'] for r in raw})==len(raw)
    for a,b in zip(raw,delex):
        assert (a['source_id'],a['variant_id'],a['split'])==(b['source_id'],b['variant_id'],b['split'])
    with (track(cachefolder/'paired_queries.csv')).open(encoding='utf-8-sig',newline='') as f:cached=list(csv.DictReader(f))
    assert len(cached)==len(raw)
    assert all(c['source_id']==a['source_id'] and c['raw_query']==a['text'] and c['delex_query']==b['text'] for c,a,b in zip(cached,raw,delex))
    # Reuse query arrays only. The archive's 'keys' array belongs to another set and is deliberately not used.
    z=np.load(track(cachefolder/'embeddings.npz'));qraw=z['raw'];qdelex=z['delex']
    keypath=track(SERVICE/'inputs'/ds/'selected_vectors.npy');keys=np.load(keypath)
    freeze=read(track(SERVICE/'inputs'/ds/'train_frozen_selection.json'))
    entries=read(track(SERVICE/'outputs'/ds/'placeholder_saved_order/entries.json'))
    assert [e['text'] for e in entries]==freeze['texts'] and len(keys)==len(entries)
    actual=rows(track(SERVICE/'outputs'/ds/'placeholder_saved_order/query_results.jsonl'))
    assert [r['source_id'] for r in actual]==[r['source_id'] for r in raw]
    assert [r['text'] for r in actual]==[r['text'] for r in delex]
    rs=qraw@keys.T;dscores=qdelex@keys.T
    rh=(rs>=.85).any(axis=1);dh=(dscores>=.85).any(axis=1)
    assert dh.tolist()==[r['cache_hit'] for r in actual]
    n=len(raw);gain=dh&~rh;lost=rh&~dh
    rng=np.random.default_rng(20261105);bo=[]
    for _ in range(50):
        idx=rng.integers(0,n,size=(100,n))
        bo.append(np.column_stack([rh[idx].mean(axis=1),dh[idx].mean(axis=1)])*100)
    bo=np.concatenate(bo);ci=np.quantile(bo[:,1]-bo[:,0],[.025,.975])
    result={'dataset':ds,'k':len(keys),'test_total':n,'original_covered':int(rh.sum()),'original_coverage_pct':100*float(rh.mean()),
      'deidentified_covered':int(dh.sum()),'deidentified_coverage_pct':100*float(dh.mean()),
      'change_pp':100*float(dh.mean()-rh.mean()),'change_ci95_low_pp':float(ci[0]),'change_ci95_high_pp':float(ci[1]),
      'newly_covered':int(gain.sum()),'lost_coverage':int(lost.sum()),'covered_in_both':int((rh&dh).sum()),
      'uncovered_in_both':int((~rh&~dh).sum()),'deidentified_matches_actual_service':True,'key_vectors_sha256':sha(keypath),
      'query_vectors_sha256':sha(cachefolder/'embeddings.npz')}
    assert result['newly_covered']-result['lost_coverage']==result['deidentified_covered']-result['original_covered']
    perquery=[]
    for i,(a,b) in enumerate(zip(raw,delex)):
        state='newly_covered' if gain[i] else 'lost_coverage' if lost[i] else 'both_covered' if rh[i] else 'both_uncovered'
        perquery.append({'source_id':a['source_id'],'variant_id':a['variant_id'],'raw_query':a['text'],'deidentified_query':b['text'],
           'original_covered':int(rh[i]),'deidentified_covered':int(dh[i]),'status':state,
           'original_max_cosine':float(rs[i].max()),'deidentified_max_cosine':float(dscores[i].max()),
           'original_best_key':int(rs[i].argmax()),'deidentified_best_key':int(dscores[i].argmax())})
    for name,rr in [('paired_queries.csv',perquery),('newly_covered_queries.csv',[r for r in perquery if r['status']=='newly_covered']),
                    ('lost_queries.csv',[r for r in perquery if r['status']=='lost_coverage'])]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(perquery[0]));w.writeheader();w.writerows(rr)
    np.savez_compressed(out/'similarities.npz',original=rs,deidentified=dscores)
    save(out/'fixed_keys.json',{'texts':freeze['texts'],'selected_vectors_source':str(keypath),'selected_vectors_sha256':sha(keypath)})
    save(out/'coverage.json',result);summary.append(result);print(json.dumps(result),flush=True)
save(ROOT/'summary.json',summary)
with (ROOT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
save(ROOT/'verification.json',{'passed':True,'source_files':source_files,'source_aligned_queries':sum(r['test_total'] for r in summary),
 'same_keys_in_both_conditions':True,'deidentified_all_equal_actual_service':True,'no_keys_from_old_embedding_archive_used':True,
 'new_api_calls':0,'new_encoding':0,'candidate_reselection':False,'actual_service_rerun':False,'completed_epoch':time.time()})
