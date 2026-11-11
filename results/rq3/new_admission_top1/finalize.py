"""Merge exact-content reused judgments and newly completed two-model judgments."""
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'server_evidence'
SERVICE=ROOT.parent/'strict085_admission/server_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvout(p,rr,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rr[0]));w.writeheader();w.writerows(rr)

pairs=read(ROOT/'all_top1_pairs.json');prep=read(ROOT/'preparation.json')
new=rows(EVIDENCE/'task_judgments.jsonl');manifest=read(EVIDENCE/'task_judge_manifest.json')
assert {x['model'] for x in manifest['providers']}=={'qwen3.8-flash','deepseek-flash'}
byid=defaultdict(dict)
for r in new:
    assert r['label'] in {'SAME_TASK','DIFFERENT_TASK','UNCERTAIN'}
    assert r['provider'] not in byid[r['pair_id']]
    byid[r['pair_id']][r['provider']]=r
assert set(byid)=={r['pair_id'] for r in pairs if not r['reused']}
assert len(new)==2*prep['pending_pairs']
for r in pairs:
    if r['reused']:
        assert len({x['consensus'] for x in r['prior_judgments']})==1
        assert r['consensus'] in {'BOTH_SAME_TASK','BOTH_DIFFERENT_TASK','UNRESOLVED'}
        r['qwen_label']=r['prior_judgments'][0].get('qwen_label','')
        r['deepseek_label']=r['prior_judgments'][0].get('deepseek_label','')
        r['label_origin']='exact_content_reuse'
    else:
        m=byid[r['pair_id']];assert set(m)=={'qwen_api','deepseek_api'}
        a,b=m['qwen_api']['label'],m['deepseek_api']['label']
        r['consensus']='BOTH_DIFFERENT_TASK' if a==b=='DIFFERENT_TASK' else 'BOTH_SAME_TASK' if a==b=='SAME_TASK' else 'UNRESOLVED'
        r['qwen_label']=a;r['deepseek_label']=b
        r['qwen_reason']=m['qwen_api']['reason'];r['deepseek_reason']=m['deepseek_api']['reason'];r['label_origin']='new_two_model_judgment'
save(ROOT/'completed_top1_pairs.json',pairs)
flat=[]
for r in pairs:
    f={k:v for k,v in r.items() if k!='prior_judgments'};f['reuse_provenance']=json.dumps(r['prior_judgments'],ensure_ascii=False);flat.append(f)
fields=list(dict.fromkeys(k for r in flat for k in r));csvout(ROOT/'completed_top1_pairs.csv',flat,fields)
summary=[];queryrows=[];agreement=[];examples=[]
for ds in ['banking77','abcd','dogo']:
    ds_pairs=[r for r in pairs if r['dataset']==ds];counts=Counter(r['consensus'] for r in ds_pairs)
    actual=rows(SERVICE/'outputs'/ds/'placeholder_saved_order/query_results.jsonl');n=len(actual);hits=sum(r['cache_hit'] for r in actual)
    assert hits==len(ds_pairs)
    lookup={r['source_id']:r for r in ds_pairs};assert len(lookup)==hits
    cross=np.zeros(n,dtype=bool);covered=np.array([r['cache_hit'] for r in actual])
    for i,q in enumerate(actual):
        r=lookup.get(q['source_id'])
        if r:
            assert r['query_text']==q['text'] and r['returned_answer']==q['returned_answer']
            cross[i]=r['consensus']=='BOTH_DIFFERENT_TASK'
        queryrows.append({'dataset':ds,'source_id':q['source_id'],'cache_hit':int(q['cache_hit']),
                         'consensus':r['consensus'] if r else 'NO_HIT','different_task_selected':int(cross[i])})
    rng=np.random.default_rng(20261105);allboot=[];hitboot=[]
    for _ in range(50):
        idx=rng.integers(0,n,size=(100,n));different=cross[idx].sum(axis=1);total=covered[idx].sum(axis=1)
        assert np.all(total>0);allboot.extend(100*different/n);hitboot.extend(100*different/total)
    allci=np.quantile(allboot,[.025,.975]);hitci=np.quantile(hitboot,[.025,.975])
    rr={'dataset':ds,'test_total':n,'actual_hits':hits,'same_task_selected':counts['BOTH_SAME_TASK'],
       'different_task_selected':counts['BOTH_DIFFERENT_TASK'],'unresolved_selected':counts['UNRESOLVED'],
       'different_task_pct_all_test':100*counts['BOTH_DIFFERENT_TASK']/n,'different_task_pct_hits':100*counts['BOTH_DIFFERENT_TASK']/hits,
       'different_task_all_test_ci95_low':float(allci[0]),'different_task_all_test_ci95_high':float(allci[1]),
       'different_task_hits_ci95_low':float(hitci[0]),'different_task_hits_ci95_high':float(hitci[1]),
       'reused_pairs':sum(r['reused'] for r in ds_pairs),'newly_judged_pairs':sum(not r['reused'] for r in ds_pairs),
       'missing_judgments':0,'answer_errors_assessed':False}
    assert rr['same_task_selected']+rr['different_task_selected']+rr['unresolved_selected']==hits
    summary.append(rr);print(json.dumps(rr),flush=True)
    fresh=[r for r in ds_pairs if not r['reused']]
    if fresh:
        exact=sum(r['qwen_label']==r['deepseek_label'] for r in fresh)
        agreement.append({'dataset':ds,'new_pairs':len(fresh),'exact_label_agreement':exact,'exact_label_agreement_pct':100*exact/len(fresh),
          'either_uncertain':sum('UNCERTAIN' in [r['qwen_label'],r['deepseek_label']] for r in fresh),
          'same_vs_different_disagreement':sum({r['qwen_label'],r['deepseek_label']}=={'SAME_TASK','DIFFERENT_TASK'} for r in fresh)})
    examples.extend([r for r in ds_pairs if r['consensus']=='BOTH_DIFFERENT_TASK'][:3])
save(ROOT/'summary.json',summary);csvout(ROOT/'summary.csv',summary);csvout(ROOT/'query_level_audit.csv',queryrows)
save(ROOT/'new_judge_agreement.json',agreement);save(ROOT/'illustrative_cross_task_pairs.json',examples)
save(ROOT/'completion.json',{'complete':True,'pairs':len(pairs),'reused':prep['reused_pairs'],'newly_judged':prep['pending_pairs'],
    'new_model_judgments':len(new),'missing':0,'human_gold_available':False,'answer_validity_assessed':False,
    'judge_manifest_sha256':sha(EVIDENCE/'task_judge_manifest.json'),'finalize_script_sha256':sha(Path(__file__))})
