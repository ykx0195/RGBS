"""Recount frozen control embeddings at a numerical threshold; no fitted threshold."""
from pathlib import Path
import sys, csv, json, hashlib, ast, math
import numpy as np

CODE=Path(r'C:\Users\ykx01\Desktop\ykx\语义缓存脱敏研究\code')
sys.path.insert(0,str(CODE))
import lexical_template_control as t
OUT=Path(r'C:\Users\ykx01\Desktop\results\RQ1_固定阈值控制核对_20261103')
OUT.mkdir(exist_ok=True)
TAU=.85
np.set_printoptions(precision=8)

def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Reuse the existing operators, pair filters, split membership and uncertainty formulas.
# Extract pure NumPy functions without importing GPU/model-loading dependencies.
tree=ast.parse((CODE/'run_matched_nonpii_gpu.py').read_text(encoding='utf-8'))
pure=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in
      ('split_indices','pair_scores','record_jackknife','metrics')]
ns={'np':np,'math':math,'hashlib':hashlib}
exec(compile(ast.Module(body=pure,type_ignores=[]),'<saved-control-functions>','exec'),ns)
rows=[]; hashes={}
base=Path(r'C:\Users\ykx01\Desktop\results\脱敏\matched_nonpii_control_gpu_v1\results')
for d in (*t.DOMAINS,'legal'):
    p=base/d
    records=[json.loads(l) for l in (p/'selected_records.jsonl').open(encoding='utf-8')]
    _,idx=ns['split_indices'](records,d)
    i,j=np.triu_indices(len(idx),1)
    for m in t.MODELS:
        raw=ns['pair_scores'](np.load(p/f'{m}_source_text.npy'),idx)
        for c in t.CONDITIONS:
            a=ns['pair_scores'](np.load(p/f'{m}_pii_{c}.npy'),idx)
            b=ns['pair_scores'](np.load(p/f'{m}_control_{c}.npy'),idx)
            rows.append(dict(domain=d,model=m,method=c,threshold=TAU,
                             **ns['metrics'](raw,a,b,TAU,i,j,len(idx))))
    print('matched',d,flush=True)
save('matched_nonpii_fixed085.csv',rows)

records_by_domain={d:t.load_records(d) for d in t.DOMAINS}
type_map=t.document_types(records_by_domain)
filtered=[]
for d in (*t.DOMAINS,'legal'):
    if d=='legal':
        records=[json.loads(l) for l in t.TAB_RECORDS.open(encoding='utf-8')]
        _,idx=t.tab_split(records);types=None
        def path_for(m,c): return t.TAB_EMBED/f'{m}_{c}.npy'
    else:
        records=records_by_domain[d];_,idx=t.split_nemotron(records)
        types=np.array([type_map[r['uid']] for r in records])[idx]
        def path_for(m,c): return t.EMBED/m/d/f'{c}.npy'
    i,j=np.triu_indices(len(idx),1)
    features=t.VECTOR.transform([t.skeleton(records[k]) for k in idx])
    lex=t.lexical_pair_scores(features,np.arange(len(idx)),i,j)
    universes={'non_pii_lexical_below_0_8':lex<.8,
               'non_pii_lexical_below_0_5':lex<.5}
    if types is not None:universes['different_document_type']=types[i]!=types[j]
    for m in t.MODELS:
        before=t.cosine(path_for(m,'original'),idx)[i,j]>=TAU
        for c in t.CONDITIONS:
            after=t.cosine(path_for(m,c),idx)[i,j]>=TAU
            value=after.astype(np.int8)-before.astype(np.int8)
            for universe,eligible in universes.items():
                lo,hi=t.leave_one_interval(value,i,j,eligible,len(idx))
                glo,ghi=(None,None) if universe!='different_document_type' else t.leave_type_interval(value,i,j,eligible,types)
                filtered.append(dict(domain=d,model=m,method=c,threshold=TAU,
                    universe=universe,n_records=len(idx),n_pairs=int(eligible.sum()),
                    original_above_count=int((eligible&before).sum()),
                    modified_above_count=int((eligible&after).sum()),
                    net_change=float(value[eligible].mean()),
                    record_jackknife_ci95_low=lo,record_jackknife_ci95_high=hi,
                    group_jackknife_ci95_low=glo,group_jackknife_ci95_high=ghi))
    print('filtered',d,flush=True)
save('filtered_pairs_fixed085.csv',filtered)

summary={
    'threshold':TAU,'protocol':'Fixed threshold on previously saved embeddings and evaluation memberships. CPU float32 dot products; no threshold estimation, text generation or re-encoding.',
    'matched_nonpii':{'cells':len(rows),'positive':sum(x['pii_minus_control_net_change']>0 for x in rows),
                     'positive_ci_lower':sum(x['difference_record_jackknife_ci95_low']>0 for x in rows)},
    'filters':{}}
for name in sorted({x['universe'] for x in filtered}):
    r=[x for x in filtered if x['universe']==name]
    key='group_jackknife_ci95_low' if name=='different_document_type' else 'record_jackknife_ci95_low'
    summary['filters'][name]={'cells':len(r),'positive':sum(x['net_change']>0 for x in r),
                             'positive_ci_lower':sum(x[key] is not None and x[key]>0 for x in r)}
for p in [CODE/'run_matched_nonpii_gpu.py',CODE/'lexical_template_control.py',Path(__file__)]:
    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
summary['source_code_sha256']=hashes
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
