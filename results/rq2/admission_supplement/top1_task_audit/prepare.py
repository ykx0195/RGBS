"""Prepare task audit for actual Top-1 returns of the NEW admitted portfolios only."""
import csv,hashlib,json,shutil,tarfile
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
BASE=Path(r'C:\Users\ykx01\Desktop\results')
SERVICE=WORK/'strict085_admission/server_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def content(key,query):return hashlib.sha256(json.dumps([key,query],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
normalize={'BOTH_DIFFERENT_TASK':'BOTH_DIFFERENT_TASK','BOTH_SAME_TASK':'BOTH_SAME_TASK',
           'UNRESOLVED':'UNRESOLVED','UNRESOLVED_OR_DISAGREEMENT':'UNRESOLVED'}
roots=[BASE/'阈值扩展_080_0825_085_20261031',BASE/'优先补充实验_20261031/baselines',BASE/'联合组件消融_20261031',
       BASE/'CorpusPoisoning_HotFlip对比_20261031/evaluation',BASE/'AdversarialDecoding对比_20261031/evaluation',
       BASE/'固定文本_原文脱敏配对_20261031',BASE/'普通业务请求_固定小集合_脱敏跨任务_20261104/outputs']
paths=set()
for p in roots:
    if p.exists():paths.update(p.rglob('*task_pair_audit.csv'))
paths.add(BASE/'递进主线补充实验_20261101/task_audit_completed_pairs.csv')
cache=defaultdict(list);sourcehashes={}
for p in sorted(paths):
    if not p.exists():continue
    sourcehashes[str(p)]=sha(p)
    with p.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            label=normalize.get(r.get('consensus',''))
            if not label or not r.get('key_text') or not r.get('query_text'):continue
            cache[(r['key_text'],r['query_text'])].append({'consensus':label,'source_csv':str(p),'source_pair_id':r.get('pair_id',''),
              'qwen_label':r.get('qwen_label') or r.get('qwen_api_label') or '',
              'deepseek_label':r.get('deepseek_label') or r.get('deepseek_api_label') or ''})
allpairs=[];pending=[];summary=[];index=0
assert read(SERVICE/'outputs/independent_verification.json')['passed']
for ds in ['banking77','abcd','dogo']:
    out=ROOT/ds;out.mkdir(exist_ok=True)
    p=SERVICE/'outputs'/ds/'placeholder_saved_order'
    entries=read(p/'entries.json');query=rows(p/'query_results.jsonl');scores=np.load(p/'similarities.npz')['actual']
    assert len({e['answer'] for e in entries})==len(entries),'Actual origin must be identifiable from returned answer'
    lookup={e['answer']:j for j,e in enumerate(entries)}
    pairs=[]
    for i,q in enumerate(query):
        if not q['cache_hit']:continue
        j=lookup[q['returned_answer']];entry=entries[j]
        assert scores[i,j]>=.85 and abs(float(scores[i,j])-float(scores[i].max()))<2e-5
        assert abs(float(scores[i,j])-q['max_cosine'])<2e-5
        prior=cache.get((entry['text'],q['text']),[]);labels={r['consensus'] for r in prior}
        reused=len(labels)==1
        item={'pair_id':f'case-{index:06d}','dataset':ds,'test_index':i,'source_id':q['source_id'],'key_index':j,
          'key_text':entry['text'],'query_text':q['text'],'selected_cosine':float(scores[i,j]),
          'returned_answer':q['returned_answer'],'content_sha256':content(entry['text'],q['text']),
          'consensus':next(iter(labels)) if reused else 'CONFLICTING_PRIOR_LABELS' if labels else 'NOT_JUDGED',
          'reused':reused,'prior_judgments':prior}
        allpairs.append(item);pairs.append(item);index+=1
        if not reused:pending.append(item)
    summary.append({'dataset':ds,'test_total':len(query),'actual_top1_hits':len(pairs),'reused':sum(r['reused'] for r in pairs),
       'pending':sum(not r['reused'] for r in pairs),'conflicting':sum(r['consensus']=='CONFLICTING_PRIOR_LABELS' for r in pairs)})
    save(out/'audit_pairs.json',{'pairs':[{'pair_id':r['pair_id'],'key_text':r['key_text'],'query_text':r['query_text']} for r in pairs if not r['reused']]})
save(ROOT/'all_top1_pairs.json',allpairs)
save(ROOT/'preparation.json',{'scope':'actual deidentified-query Top-1 selected entries from NEW admitted portfolios',
 'source_service':str(SERVICE),'source_verification_sha256':sha(SERVICE/'outputs/independent_verification.json'),
 'summary':summary,'total_pairs':len(allpairs),'reused_pairs':len(allpairs)-len(pending),'pending_pairs':len(pending),
 'prior_audit_sources':sourcehashes,'instruction_source':str(BASE/'递进主线补充实验_20261101/code/remote_fixed_task_audit.py'),
 'new_question_generation':False,'answer_validity_assessed':False,'selection_fixed':True,
 'reuse_rule':'exact ordered key/query texts, consistent existing two-model consensus; conflicts rejudged',
 'judge_visible_fields':'opaque case ID and two question texts only; no benchmark, scores, source category, returned answer, or selection state'})
shutil.copy2(BASE/'递进主线补充实验_20261101/code/remote_fixed_task_audit.py',ROOT/'remote_fixed_task_audit.py')
with tarfile.open(ROOT/'audit_inputs.tar.gz','w:gz') as t:
    for ds in ['banking77','abcd','dogo']:t.add(ROOT/ds/'audit_pairs.json',arcname=ds+'/audit_pairs.json')
print(json.dumps(summary));print('TOTAL',len(allpairs),'REUSED',len(allpairs)-len(pending),'PENDING',len(pending))
