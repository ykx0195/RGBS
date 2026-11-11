"""Verify actual selected origins, exact-content reuse, and complete model evidence."""
import ast,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SERVICE=ROOT.parent/'strict085_admission/server_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pairs=read(ROOT/'completed_top1_pairs.json');prep=read(ROOT/'preparation.json');summary=read(ROOT/'summary.json')
assert len(pairs)==2297 and len({r['pair_id'] for r in pairs})==2297
for source,digest in prep['prior_audit_sources'].items():assert sha(Path(source))==digest
instruction=None
for node in ast.parse((ROOT/'remote_fixed_task_audit.py').read_text(encoding='utf-8')).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTRUCTION' for t in node.targets):instruction=ast.literal_eval(node.value)
assert instruction
cache={};files={};usage=Counter()
for p in (ROOT/'server_evidence/judge_cache').rglob('*.json'):
    r=read(p);assert r['instruction_sha256']==hashlib.sha256(instruction.encode()).hexdigest()
    raw=r['raw_response'];assert raw['choices'][0]['finish_reason']=='stop'
    body=json.loads(raw['choices'][0]['message']['content']);items=body.get('items',body.get('results',[])) if isinstance(body,dict) else body
    assert items==r['items']
    visible={v['id']:v for v in r['visible_pairs']}
    assert set(visible)=={i['id'] for i in items}
    for i in items:
        key=(i['id'],r['provider']);assert key not in cache
        cache[key]={'label':i['label'],'pair':visible[i['id']]}
    for k in ['prompt_tokens','completion_tokens','total_tokens']:usage[k]+=(r.get('usage') or {}).get(k,0)
    files[str(p)]=sha(p)
assert len(cache)==2*prep['pending_pairs']
for r in pairs:
    h=hashlib.sha256(json.dumps([r['key_text'],r['query_text']],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert h==r['content_sha256']
    if r['reused']:
        assert {p['consensus'] for p in r['prior_judgments']}=={r['consensus']}
    else:
        labels=[]
        for provider in ['qwen_api','deepseek_api']:
            value=cache[(r['pair_id'],provider)];assert value['pair']['question_a']==r['key_text'] and value['pair']['question_b']==r['query_text']
            labels.append(value['label'])
        expected='BOTH_DIFFERENT_TASK' if labels==['DIFFERENT_TASK']*2 else 'BOTH_SAME_TASK' if labels==['SAME_TASK']*2 else 'UNRESOLVED'
        assert expected==r['consensus']
checks=[]
for s in summary:
    ds=s['dataset'];p=SERVICE/'outputs'/ds/'placeholder_saved_order';entries=read(p/'entries.json')
    actual=rows(p/'query_results.jsonl');scores=np.load(p/'similarities.npz')['actual']
    pp=[r for r in pairs if r['dataset']==ds];lookup={r['test_index']:r for r in pp}
    assert set(lookup)=={i for i,q in enumerate(actual) if q['cache_hit']}
    for i,r in lookup.items():
        q=actual[i];e=entries[r['key_index']]
        assert r['source_id']==q['source_id'] and r['query_text']==q['text']
        assert r['key_text']==e['text'] and r['returned_answer']==e['answer']==q['returned_answer']
        assert scores[i,r['key_index']]>=.85 and abs(float(scores[i,r['key_index']])-float(scores[i].max()))<2e-5
    c=Counter(r['consensus'] for r in pp)
    assert c['BOTH_DIFFERENT_TASK']==s['different_task_selected']
    assert c['BOTH_SAME_TASK']==s['same_task_selected'] and c['UNRESOLVED']==s['unresolved_selected']
    assert sum(c.values())==s['actual_hits']==len(pp)
    assert np.isclose(s['different_task_pct_all_test'],100*c['BOTH_DIFFERENT_TASK']/len(actual))
    assert np.isclose(s['different_task_pct_hits'],100*c['BOTH_DIFFERENT_TASK']/len(pp))
    checks.append({'dataset':ds,'actual_top1_origins_verified':len(pp),'label_partition_complete':True,'passed':True})
result={'passed':True,'checks':checks,'actual_top1_pairs':len(pairs),'new_judgments_verified':len(cache),
  'reused_pairs':prep['reused_pairs'],'source_audit_files_unchanged':True,'new_cache_files_sha256':files,
  'saved_successful_response_usage':dict(usage),'human_review_performed':False,'answer_validity_assessed':False,
  'verify_script_sha256':sha(Path(__file__))}
(ROOT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='new_cache_files_sha256'}))
