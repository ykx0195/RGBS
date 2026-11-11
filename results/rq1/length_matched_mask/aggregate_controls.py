from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import math,json,hashlib,ast,csv,re
from sklearn.feature_extraction.text import HashingVectorizer
OUT=Path(__file__).resolve().parent;BASE=Path(r'C:\Users\ykx01\Desktop\results\脱敏');CODE=Path(r'C:\Users\ykx01\Desktop\ykx\语义缓存脱敏研究\code');NEW=OUT/'outputs'
MODELS=['minilm','gte','e5','bge'];DOMAINS=['healthcare','finance','banking','insurance','legal'];RULES=['generic','typed','typed_numbered','delete','mask_length']
manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
ns={'np':np,'math':math,'hashlib':hashlib}
for file,names in [('run_matched_nonpii_gpu.py',['pair_scores','record_jackknife','metrics']),('lexical_template_control.py',['leave_one_interval'])]:
 tree=ast.parse((CODE/file).read_text(encoding='utf-8'));func=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];exec(compile(ast.Module(body=func,type_ignores=[]),file,'exec'),ns)
def pairs(x):
 m=x@x.T;np.clip(m,-1,1,out=m);return m[np.triu_indices(len(x),1)]
def save(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def skeleton(r):
 raw=r['source_text'];parts=[];cursor=0
 for sp in sorted(r['spans'],key=lambda x:x['start']):
  parts.append(raw[cursor:sp['start']]);cursor=sp['end']
 parts.append(raw[cursor:]);return re.sub(r'[^a-z0-9]+',' ',' '.join(parts).lower()).strip()
vector=HashingVectorizer(analyzer='char',ngram_range=(5,5),n_features=2**20,binary=True,alternate_sign=False,norm='l2')
selection=json.loads((BASE/'nemotron_4domain_embedding_gpu_v1'/'sample_selection.json').read_text(encoding='utf-8'))
matched=[];filtered=[];checks=[]
for panel in ['matched','filtered']:
 for d in DOMAINS:
  name=panel+'_'+d;oldidx=np.array(manifest[name]['eval_indices']);pos=np.load(NEW/f'{name}_retained_positions.npy');idx=oldidx[pos]
  ii,jj=np.triu_indices(len(idx),1)
  if panel=='filtered':
   path=Path(manifest[name]['source_path']);records=[json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]
   if d!='legal':records=[r for r in records if r['record_id'] not in set(selection[d]['excluded_record_ids'])]
   skel=vector.transform([skeleton(records[i]) for i in idx]);lex=np.asarray((skel@skel.T)[ii,jj]).ravel();eligible=lex<.5
  for model in MODELS:
   if panel=='matched':
    base=BASE/'matched_nonpii_control_gpu_v1'/'results'/d
    def cached(field):return np.asarray(np.load(base/f'{model}_{field}.npy',mmap_mode='r')[idx])
    original=cached('source_text')
   else:
    def cached(field):
     p=BASE/'tab_short_legal_gpu_v1'/'results'/f'{model}_{field}.npy' if d=='legal' else BASE/'nemotron_4domain_embedding_gpu_v1'/'embeddings'/model/d/f'{field}.npy'
     return np.asarray(np.load(p,mmap_mode='r')[idx])
    original=cached('original')
   newraw=np.load(NEW/f'{name}_{model}_source_text.npy')
   raw=pairs(original);newscore=pairs(newraw)
   checks.append({'panel':panel,'domain':d,'model':model,'records':len(idx),'embedding_max_abs_diff':float(np.max(np.abs(original-newraw))),'source_gate_disagreements':int(np.sum((raw>=.85)!=(newscore>=.85))),'cosine_min_old_new':float(np.min(np.sum(original*newraw,axis=1)))})
   assert np.min(np.sum(original*newraw,axis=1))>.9999,checks[-1]
   for rule in RULES:
    if panel=='matched':
     a=np.load(NEW/f'{name}_{model}_pii_mask_length.npy') if rule=='mask_length' else cached('pii_'+rule)
     b=np.load(NEW/f'{name}_{model}_control_mask_length.npy') if rule=='mask_length' else cached('control_'+rule)
     met=ns['metrics'](raw,pairs(a),pairs(b),.85,ii,jj,len(idx))
     matched.append({'domain':d,'model':model,'method':rule,'threshold':.85,**met})
    else:
     after=pairs(np.load(NEW/f'{name}_{model}_mask_length.npy') if rule=='mask_length' else cached(rule))>=.85
     before=raw>=.85;value=after.astype(np.int8)-before.astype(np.int8);lo,hi=ns['leave_one_interval'](value,ii,jj,eligible,len(idx))
     filtered.append({'domain':d,'model':model,'method':rule,'threshold':.85,'universe':'non_pii_lexical_below_0_5','n_records':len(idx),'n_pairs':int(eligible.sum()),'original_above_count':int((eligible&before).sum()),'modified_above_count':int((eligible&after).sum()),'net_change':float(value[eligible].mean()),'record_jackknife_ci95_low':lo,'record_jackknife_ci95_high':hi})
  print('AGGREGATED',panel,d,flush=True)
save('matched_nonpii_fixed085.csv',matched);save('filtered_pairs_fixed085.csv',filtered);save('source_reencoding_checks.csv',checks)
summary=[]
for rule in RULES:
 a=[r for r in matched if r['method']==rule];b=[r for r in filtered if r['method']==rule]
 summary.append({'method':rule,'pii_median_pp':float(100*np.median([r['pii_net_change'] for r in a])),'nonpii_median_pp':float(100*np.median([r['control_net_change'] for r in a])),'excess_median_pp':float(100*np.median([r['pii_minus_control_net_change'] for r in a])),'excess_positive':sum(r['pii_minus_control_net_change']>0 for r in a),'filtered_median_pp':float(100*np.median([r['net_change'] for r in b])),'filtered_q1_pp':float(100*np.quantile([r['net_change'] for r in b],.25)),'filtered_q3_pp':float(100*np.quantile([r['net_change'] for r in b],.75)),'filtered_positive':sum(r['net_change']>0 for r in b)})
report={'threshold':.85,'summary':summary,'matched_positive':sum(r['pii_minus_control_net_change']>0 for r in matched),'filtered_positive':sum(r['net_change']>0 for r in filtered),'common_filter':'Original held-out record IDs; retain only records <=256 tokens for source and length-mask variants across all four encoders; recalculate every rule on that common retained cohort.','source_gate_total_disagreements':sum(r['source_gate_disagreements'] for r in checks),'source_embeddings':'Reuse original frozen raw and unchanged-rule embeddings after verifying new source re-encoding; length-mask is newly encoded.'}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
