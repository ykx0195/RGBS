"""Recover original per-model reasons for unresolved cases, without relabeling."""
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source_cache={}
def source_row(path,key,query):
    if path not in source_cache:
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            source_cache[path]={(r.get('key_text'),r.get('query_text')):r for r in csv.DictReader(f)}
    return source_cache[path].get((key,query),{})
def recover(row):
    a=row.get('qwen_label') or row.get('qwen_api_label') or ''
    b=row.get('deepseek_label') or row.get('deepseek_api_label') or ''
    ar=row.get('qwen_reason') or row.get('qwen_api_reason') or ''
    br=row.get('deepseek_reason') or row.get('deepseek_api_reason') or ''
    if a and b:return a,b,ar,br
    try:ss=json.loads(row.get('source_judgments','[]'))
    except Exception:ss=[]
    mm={r.get('provider'):r for r in ss if r.get('label')}
    if {'qwen_api','deepseek_api'}<=set(mm):
        return mm['qwen_api']['label'],mm['deepseek_api']['label'],mm['qwen_api'].get('reason',''),mm['deepseek_api'].get('reason','')
    return None

uu=[r for r in read(ROOT/'completed_top1_pairs.json') if r['consensus']=='UNRESOLVED']
output=[]
for r in uu:
    detail=recover(r)
    if r['reused']:
        for p in r['prior_judgments']:
            candidate=recover(source_row(p['source_csv'],r['key_text'],r['query_text']))
            if candidate:
                detail=candidate
                if candidate[2] and candidate[3]:break
    assert detail,(r['pair_id'],'missing original labels')
    a,b,ar,br=detail
    assert not (a==b and a in ['SAME_TASK','DIFFERENT_TASK'])
    mechanism='OPPOSITE_DEFINITE_LABELS' if {a,b}=={'SAME_TASK','DIFFERENT_TASK'} else 'AT_LEAST_ONE_UNCERTAIN'
    output.append({'pair_id':r['pair_id'],'dataset':r['dataset'],'key_text':r['key_text'],'query_text':r['query_text'],
       'qwen_label':a,'deepseek_label':b,'qwen_reason':ar,'deepseek_reason':br,'mechanism':mechanism,
       'label_origin':r['label_origin']})
with (ROOT/'unresolved_details.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
mechanisms=Counter(r['mechanism'] for r in output)
bydataset={ds:dict(Counter(r['mechanism'] for r in output if r['dataset']==ds)) for ds in ['banking77','abcd','dogo']}
keygroups=[]
for ds in bydataset:
    for key,n in Counter(r['key_text'] for r in output if r['dataset']==ds).most_common():
        keygroups.append({'dataset':ds,'key_text':key,'unresolved_count':n})
result={'total':len(output),'mechanism_counts':dict(mechanisms),'by_dataset':bydataset,
 'label_pair_counts':{str(k):v for k,v in Counter((r['qwen_label'],r['deepseek_label']) for r in output).items()},
 'no_consensus_changes':True,'no_new_model_calls':True}
save(ROOT/'unresolved_summary.json',result);save(ROOT/'unresolved_by_key.json',keygroups)
print(json.dumps(result));print(json.dumps(keygroups,ensure_ascii=False))
