"""CPU-only post-hoc analyses of frozen Train-selected text portfolios.

No generation search, no GPU, no answer evaluation. Test is never used to
construct, select, or order text portfolios. All inputs are preserved.
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k] = '4'
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv, hashlib, json, sys, time, shutil
from collections import defaultdict, Counter
import numpy as np

BASE = Path(r'C:\Users\ykx01\Desktop\results')
OUT = BASE/'递进主线补充实验_20261101'
DATA = Path(r'C:\Users\ykx01\Desktop\ykx\data')
MODEL = Path(r'C:\Users\ykx01\Desktop\ykx\bge-small-en-v1.5')
CODE = BASE/'newcode/src'
DS = {'banking77':('banking77',15,1109), 'abcd':('abcd',5,308), 'dogo':('multidogo',10,909)}
LONG = BASE/'ykx现行实验/v10_current/multidogo/results/runs/multidogo_riemann_k10_t0p85_s42_dataset_scope_v1'
INPUTS = {}
ENC = None
START = time.monotonic()

def log(*a): print(datetime.now(timezone(timedelta(hours=8))).strftime('%H:%M:%S'), *a, flush=True)
def sha(p):
    p=Path(p)
    if str(p) not in INPUTS:
        INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return INPUTS[str(p)]
def read(p):
    sha(p); return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    sha(p)
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def csvrows(p):
    sha(p)
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def writecsv(p,rr):
    if not rr:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def loadnp(p): sha(p);return np.load(p,allow_pickle=False)
def textsha(texts):return hashlib.sha256(json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def texts_from_selected(selected):return [x['text'] for x in selected]
def encoder():
    global ENC
    if ENC is None:
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        sys.path.insert(0,str(CODE))
        from hybrid_search import BGE
        ENC=BGE(MODEL,'cpu','bge')
    return ENC
def encode_cached(name,texts):
    path=OUT/'vectors'/f'{name}.npy'; identity=path.with_suffix('.identity.json')
    ident={'text_sha256':textsha(texts),'count':len(texts),'model_sha256':sha(MODEL/'model.safetensors'),
           'encoder_code_sha256':sha(CODE/'hybrid_search.py'),'max_seq_length':512,'pooling':'CLS','device':'cpu','dtype':'float32'}
    if path.exists():
        assert read(identity)==ident, name
        return loadnp(path)
    log('ENCODE_CPU',name,len(texts));path.parent.mkdir(parents=True,exist_ok=True)
    chunks=[]
    for start in range(0,len(texts),256):
        chunks.append(encoder().encode(texts[start:start+256],64))
        log('ENCODE_PROGRESS',name,min(start+256,len(texts)),len(texts))
    v=np.concatenate(chunks)
    np.save(path,v);save(identity,ident)
    return v

def greedy_order(mask):
    used=[]; covered=np.zeros(len(mask),bool)
    for _ in range(mask.shape[1]):
        gain=(mask & ~covered[:,None]).sum(axis=0);gain[used]=-1
        j=int(np.argmax(gain));used.append(j);covered|=mask[:,j]
    return used

def natural_selection(vectors,texts,k):
    log('RAW_NATURAL_SELECTION',len(texts),k)
    sim=vectors@vectors.T;mask=sim>=.85
    del sim
    valid=np.zeros(len(texts),bool);seen=set()
    for i,t in enumerate(texts):
        if t not in seen:valid[i]=True;seen.add(t)
    selected=[];covered=np.zeros(len(mask),bool)
    for _ in range(k):
        gains=mask[~covered].sum(axis=0);gains[~valid]=-1;gains[selected]=-1
        j=int(np.argmax(gains));selected.append(j);covered|=mask[:,j]
    greedy=selected[:]
    for _ in range(4):
        changed=False
        for slot in range(k):
            others=selected[:slot]+selected[slot+1:]
            remaining=~mask[:,others].any(axis=1)
            gains=mask[remaining].sum(axis=0);gains[~valid]=-1;gains[others]=-1
            j=int(np.argmax(gains));old=selected[slot]
            if gains[j]>gains[old]:selected[slot]=j;changed=True
        if not changed:break
    return {'greedy_train':greedy,'greedy_train_swap4':selected}

def ci_paired(a,b,source_ids):
    # Cluster bootstrap over source records, paired across representations.
    groups={s:i for i,s in enumerate(dict.fromkeys(source_ids))}
    gi=np.array([groups[s] for s in source_ids]);ng=len(groups)
    numerator=np.bincount(gi,weights=b.astype(int)-a.astype(int),minlength=ng)
    denominator=np.bincount(gi,minlength=ng)
    rng=np.random.default_rng(20261101); vals=[]
    for _ in range(2000):
        idx=rng.integers(0,ng,ng);vals.append(numerator[idx].sum()/denominator[idx].sum()*100)
    return np.quantile(vals,[.025,.975]).tolist()

def overlap_metrics(mask):
    deg=mask.sum(axis=1);sizes=mask.sum(axis=0);n,k=mask.shape
    jacc=[]
    for i in range(k):
        for j in range(i):
            union=(mask[:,i]|mask[:,j]).sum()
            jacc.append(float((mask[:,i]&mask[:,j]).sum()/union) if union else 0.)
    return {'queries':n,'keys':k,'covered':int((deg>0).sum()),'coverage':float((deg>0).mean()),
            'matched_pairs':int(deg.sum()),'queries_with_multiple_matches':int((deg>1).sum()),
            'mean_matches_per_covered_query':float(deg[deg>0].mean()) if np.any(deg) else 0.,
            'sum_individual_minus_union':int(sizes.sum()-(deg>0).sum()),
            'mean_key_jaccard':float(np.mean(jacc)) if jacc else 0.,
            'max_key_coverage':float(sizes.max()/n),'median_key_coverage':float(np.median(sizes)/n)}

def main():
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'code').mkdir(exist_ok=True)
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    plan={'date':'2026-11-01','analysis_type':'Post-hoc analysis of existing frozen Train-selected portfolios',
          'main_expected_counts':{d:n for d,(_,_,n) in DS.items()},'main_threshold':.85,
          'fixed_thresholds':[.75,.775,.8,.825,.85,.875,.9,.925,.95,.975],
          'calibration_targets':[.9,.95],'selection_threshold':.85,
          'budget_protocol':'Train-greedy prefixes within frozen full-K pools; not independent optimization at each K',
          'test_used_for_selection_or_order':False,'new_generation_search':False,'gpu_used':False,'api_calls':0,
          'geometry_stages':'Aligned continuous target, nearest unique natural Train anchors, actual initial pool where available, final text',
          'source_hashes_recorded':True,'audit_reuse':'Exact ordered key text and canonical query text, with original judge protocol'}
    save(OUT/'experiment_plan.json',plan)
    bundles={}; pools={}; test_vectors={}; datasets={}
    # Freeze every selection and Train-only order before reading Test embeddings.
    for ds,(folder,k,expected) in DS.items():
        log('PREPARE',ds)
        dd={}
        for rep,dirname in [('raw','non_desensitized'),('delex','desensitized')]:
            dd[rep]={s:rows(DATA/folder/dirname/f'{s}.jsonl') for s in ['train','test']}
        for split in ['train','test']:
            assert [(r['source_id'],r['variant_id']) for r in dd['raw'][split]]==[(r['source_id'],r['variant_id']) for r in dd['delex'][split]]
        assert not {r['source_id'] for r in dd['raw']['train']}&{r['source_id'] for r in dd['raw']['test']}
        datasets[ds]=dd
        full=read(BASE/f'三数据集_脱敏_32x50_20261030/runs/{ds}/final_report.json')
        raw_path=BASE/f'三数据集_未脱敏_0.85_32x50_20261031/runs/{ds}/final_report.json'; raw_report=read(raw_path)
        assert raw_report['config']['steps_per_slot']==32 and raw_report['config']['beam_width']==50 and raw_report['config']['seed']==42
        assert sha(DATA/folder/'desensitized/train.jsonl')==full['config']['train_sha256']
        assert sha(DATA/folder/'desensitized/test.jsonl')==full['test_sha256']
        assert sha(DATA/folder/'non_desensitized/train.jsonl')==raw_report['config']['train_sha256']
        assert sha(DATA/folder/'non_desensitized/test.jsonl')==raw_report['test_sha256']
        if ds=='dogo':
            mainpath=LONG/'04_final_test/report.json';mainreport=read(mainpath);rep=mainreport['reports'][0]
            maintexts=texts_from_selected(rep['selected']);targets=loadnp(LONG/'01_targets/target_vectors.npy')
            assert mainreport['test_sha256']==full['test_sha256'] and rep['text_test']['covered']==expected
        else:
            mainpath=BASE/f'三数据集_脱敏_32x50_20261030/runs/{ds}/final_report.json';maintexts=texts_from_selected(full['selected'])
            targets=loadnp(BASE/f'优先补充实验_20261031/baselines/{ds}/frozen_target_vectors.npy')
        assert len(maintexts)==k and len(set(maintexts))==k
        save(OUT/ds/'main_frozen_texts.json',{'texts':maintexts,'source':str(mainpath),'source_sha256':sha(mainpath),'expected_test_covered':expected,
             'author_confirmed_search_settings':{'seed':42,'rounds':32,'beam_width':50},
             'machine_verified_source':'Exact saved text set and Test coverage; original source path and hash retained'})
        paired=BASE/('固定文本_原文脱敏配对_20261031/ABCD补充/abcd' if ds=='abcd' else f'固定文本_原文脱敏配对_20261031/{ds}')
        ident=read(paired/'embedding_identity.json');tid=read(BASE/f'优先补充实验_20261031/baselines/{ds}/train_vectors_identity.json')
        assert ident['model_sha256']==tid['model_sha256']==sha(MODEL/'model.safetensors')
        assert ident['code_sha256']==tid['encoder_code_sha256']==sha(CODE/'hybrid_search.py')
        assert ident['raw_test_sha256']==sha(DATA/folder/'non_desensitized/test.jsonl')
        assert ident['delex_test_sha256']==sha(DATA/folder/'desensitized/test.jsonl')
        train_delex=loadnp(BASE/f'优先补充实验_20261031/baselines/{ds}/train_vectors.npy')
        assert tid['train_sha256']==sha(DATA/folder/'desensitized/train.jsonl')
        # Small consistency probe verifies locally loaded encoder against archived vectors.
        probe=[0,len(train_delex)//2,len(train_delex)-1]
        fresh=encode_cached(f'{ds}_consistency_probe',[dd['delex']['train'][i]['text'] for i in probe])
        assert np.max(np.abs(fresh-train_delex[probe]))<2e-5
        train_raw=encode_cached(f'{ds}_raw_train',[r['text'] for r in dd['raw']['train']])
        trains={'raw':train_raw,'delex':train_delex}
        rawindices=natural_selection(train_raw,[r['text'] for r in dd['raw']['train']],k)
        basef=read(BASE/f'优先补充实验_20261031/baselines/{ds}/frozen_baseline_selection.json')
        assert basef['train_sha256']==full['config']['train_sha256']
        pp={}
        for representation in ['raw','delex']:
            train=trains[representation];tr=dd[representation]['train']
            keysets={'RKGB':texts_from_selected(raw_report['selected']) if representation=='raw' else maintexts}
            for method in ['greedy_train','greedy_train_swap4']:
                inds=rawindices[method] if representation=='raw' else [x['train_index'] for x in basef['methods'][method]]
                keysets[method]=[tr[i]['text'] for i in inds]
                if representation=='delex':assert keysets[method]==[x['text'] for x in basef['methods'][method]]
            for method,keytexts in keysets.items():
                kv=encode_cached(f'{ds}_{representation}_{method}_keys',keytexts)
                train_sim=train@kv.T;order=greedy_order(train_sim>=.85)
                item={'texts':keytexts,'vectors':kv,'order':order,'train_sim':train_sim}
                pp[(representation,method)]=item
                save(OUT/ds/f'{representation}_{method}_frozen_order.json',{'texts':keytexts,'order':order,'selection_split':'train','test_used_for_selection':False,
                     'selection_threshold':.85,'pool_size':k,'train_covered':int((train_sim>=.85).any(axis=1).sum())})
        nearest=[]
        for target in targets:
            scores=train_delex@target;scores[nearest]=-np.inf;nearest.append(int(np.argmax(scores)))
        stages={'continuous_targets':targets,'nearest_unique_train_anchors':train_delex[nearest]}
        if ds!='dogo':
            a=read(BASE/f'联合组件消融_20261031/{ds}/frozen_ablation_selection.json')['variants']['without_joint_rewriting']
            anchor_texts=texts_from_selected(a) if isinstance(a,list) else a.get('texts')
            if anchor_texts:stages['actual_initial_natural_pool']=encode_cached(f'{ds}_actual_initial',anchor_texts)
        save(OUT/ds/'stage_definition.json',{'targets_source':str(LONG/'01_targets/target_vectors.npy') if ds=='dogo' else str(BASE/f'优先补充实验_20261031/baselines/{ds}/frozen_target_vectors.npy'),
             'nearest_train_indices':nearest,'nearest_is_actual_initialization':False,'available_stages':list(stages)+['final_text'],
             'scope':'Same encoder, Test split, K and threshold. Diagnostic nearest anchors are not claimed to be actual initial sentences.'})
        stages['final_text']=pp[('delex','RKGB')]['vectors']
        bundles[ds]={'trains':trains,'paired':paired,'mainpath':mainpath,'stages':stages,'raw_expected':raw_report['metrics']['test']['covered']}
        pools[ds]=pp
    save(OUT/'TRAIN_SELECTIONS_FROZEN.json',{'datasets':list(DS),'all_prefix_orders_use_train_only':True,'frozen_at':datetime.now(timezone(timedelta(hours=8))).isoformat()})
    log('ALL_TRAIN_ORDERS_FROZEN')
    fixed=[];budget=[];thresholds=[];calibration=[];overlap=[];perkey=[];pairedresults=[];stage_results=[];natural_nodes=[]
    normal=csvrows(BASE/'优先补充实验_20261031/normal_recall/normal_pair_scores.csv')
    for ds,(folder,k,expected) in DS.items():
        dd=datasets[ds];bundle=bundles[ds];pp=pools[ds]
        ev=loadnp(bundle['paired']/'embeddings.npz'); tv={'raw':ev['raw'],'delex':ev['delex']};test_vectors[ds]=tv
        source_ids=[r['source_id'] for r in dd['delex']['test']];n=len(source_ids)
        for (rep,method),p in pp.items():
            p['test_sim']=tv[rep]@p['vectors'].T
            if method=='RKGB':assert int((p['test_sim']>=.85).any(axis=1).sum())==(expected if rep=='delex' else bundle['raw_expected']), (ds,rep)
            np.savez_compressed(OUT/ds/f'{rep}_{method}_similarities.npz',train=p['train_sim'],test=p['test_sim'])
            for m in range(k+1):
                idx=p['order'][:m];a=(p['test_sim'][:,idx]>=.85).any(axis=1);b=(p['train_sim'][:,idx]>=.85).any(axis=1)
                budget.append({'dataset':ds,'representation':rep,'method':method,'texts':m,'pool_size':k,'threshold':.85,
                     'test_covered':int(a.sum()),'test_total':n,'test_coverage':float(a.mean()),'train_covered':int(b.sum()),'train_total':len(b)})
            mask=p['test_sim']>=.85
            om={'dataset':ds,'representation':rep,'method':method,**overlap_metrics(mask)};overlap.append(om)
            for j in range(k):
                perkey.append({'dataset':ds,'representation':rep,'method':method,'key_index':j,'train_rank':p['order'].index(j)+1,
                    'covered_queries':int(mask[:,j].sum()),'exclusive_queries':int((mask[:,j]&(mask.sum(axis=1)==1)).sum()),'test_total':n,'text':p['texts'][j]})
            for tau in plan['fixed_thresholds']:
                covered=(p['test_sim']>=tau).any(axis=1)
                thresholds.append({'dataset':ds,'representation':rep,'method':method,'threshold':tau,'covered':int(covered.sum()),'total':n,'coverage':float(covered.mean()),'pool_searched_at_threshold':.85})
            normals=[x for x in normal if x['dataset']==folder]
            val=[float(x[rep+'_cosine']) for x in normals if x['split']=='validation'];ntest=np.array([float(x[rep+'_cosine']) for x in normals if x['split']=='test'])
            for target in [.9,.95]:
                rank=max(0,int(np.floor((1-target)*len(val))));tau=float(np.sort(val)[rank])
                hit=(p['test_sim']>=tau).any(axis=1)
                calibration.append({'dataset':ds,'representation':rep,'method':method,'validation_recall_target':target,'validation_pairs':len(val),
                    'threshold':tau,'validation_recall':float((np.array(val)>=tau).mean()),'normal_test_pairs':len(ntest),'normal_test_recall':float((ntest>=tau).mean()),
                    'portfolio_test_covered':int(hit.sum()),'portfolio_test_total':n,'portfolio_test_coverage':float(hit.mean()),'reoptimized_at_calibrated_threshold':False})
        main=pp[('delex','RKGB')];sr=tv['raw']@main['vectors'].T;sd=main['test_sim']
        np.savez_compressed(OUT/ds/'main_fixed_key_paired_similarities.npz',raw=sr,delex=sd)
        for tau in plan['fixed_thresholds']:
            a=(sr>=tau).any(axis=1);b=(sd>=tau).any(axis=1)
            fixed.append({'dataset':ds,'threshold':tau,'k':k,'total':n,'raw_covered':int(a.sum()),'delex_covered':int(b.sum()),
                'gained_queries':int((~a&b).sum()),'lost_queries':int((a&~b).sum()),'delta_pp':float((b.mean()-a.mean())*100),
                'added_pairs':int(((sr<tau)&(sd>=tau)).sum()),'lost_pairs':int(((sr>=tau)&(sd<tau)).sum())})
        for method in ['RKGB','greedy_train','greedy_train_swap4']:
            a=(pp[('raw',method)]['test_sim']>=.85).any(axis=1);b=(pp[('delex',method)]['test_sim']>=.85).any(axis=1)
            ci=ci_paired(a,b,source_ids)
            pairedresults.append({'dataset':ds,'method':method,'threshold':.85,'raw_covered':int(a.sum()),'delex_covered':int(b.sum()),'total':n,
                'delta_pp':float((b.mean()-a.mean())*100),'source_bootstrap_low_pp':ci[0],'source_bootstrap_high_pp':ci[1],
                'scope':'Frozen independently constructed pools; source-paired Test uncertainty only, not search-seed uncertainty.'})
        for name,vec in bundle['stages'].items():
            a=tv['delex']@vec.T
            stage_results.append({'dataset':ds,'stage':name,'k':len(vec),'threshold':.85,'test_covered':int((a>=.85).any(axis=1).sum()),'test_total':n,
                'test_coverage':float((a>=.85).any(axis=1).mean())})
        # All natural Train anchors, aligned by source, for degree-distribution diagnostics.
        degree={}
        for rep in ['raw','delex']:
            mask=tv[rep]@bundle['trains'][rep].T>=.85;degree[rep]=mask.sum(axis=0)
            np.save(OUT/ds/f'{rep}_natural_train_anchor_degrees.npy',degree[rep])
        for i,r in enumerate(dd['delex']['train']):
            natural_nodes.append({'dataset':ds,'train_index':i,'source_id':r['source_id'],'raw_matched_test_queries':int(degree['raw'][i]),
                'delex_matched_test_queries':int(degree['delex'][i]),'delta':int(degree['delex'][i]-degree['raw'][i]),'test_total':n})
        log('ANALYSIS_DONE',ds,'main',expected,n)
    for name,rr in [('representation_comparison.csv',pairedresults),('budget_coverage.csv',budget),('threshold_coverage.csv',thresholds),
                    ('calibrated_coverage.csv',calibration),('fixed_key_paired_comparison.csv',fixed),('neighborhood_summary.csv',overlap),
                    ('per_key_neighborhoods.csv',perkey),('stage_coverage.csv',stage_results),('all_natural_train_anchor_degrees.csv',natural_nodes)]:writecsv(OUT/name,rr)
    save(OUT/'analysis_summary.json',{'representation_comparison':pairedresults,'calibration':calibration,'stages':stage_results,'neighborhood':overlap})
    prepare_audit(pools,datasets,test_vectors)
    plot_results(budget,thresholds,calibration,pairedresults,stage_results)
    make_report(pairedresults,calibration,budget,stage_results)
    save(OUT/'input_manifest.json',INPUTS)
    save(OUT/'completion.json',{'cpu_analyses_complete':True,'elapsed_seconds':time.monotonic()-START,'gpu_used':False,'api_calls':0,
        'test_used_for_selection':False,'new_generation_search':False,'main_test_counts':{x:v[2] for x,v in DS.items()}})
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    log('COMPLETE',str(OUT))

def prepare_audit(pools,datasets,test_vectors):
    cache=defaultdict(list)
    # These share the same two-model task-operation/object/state protocol.
    roots=[BASE/'阈值扩展_080_0825_085_20261031',BASE/'优先补充实验_20261031/baselines',BASE/'联合组件消融_20261031',BASE/'CorpusPoisoning_HotFlip对比_20261031/evaluation',BASE/'AdversarialDecoding对比_20261031/evaluation']
    paths=[]
    for root in roots:
        if root.exists():paths.extend(root.rglob('*task_pair_audit.csv'))
    for path in paths:
        for r in csvrows(path):
            if not all(r.get(k) for k in ['key_text','query_text','consensus']):continue
            cache[(r['key_text'],r['query_text'])].append({'consensus':r['consensus'],'qwen_label':r.get('qwen_label',''),
                'deepseek_label':r.get('deepseek_label',''),'source_csv':str(path),'source_pair_id':r.get('pair_id','')})
    summary=[];pending=[];allpairs=[]
    for ds,(_,k,_) in DS.items():
        p=pools[ds][('delex','RKGB')];sr=test_vectors[ds]['raw']@p['vectors'].T;sd=p['test_sim'];dd=datasets[ds]
        union=(sr>=.85)|(sd>=.85); judged=np.zeros_like(union);diff=np.zeros_like(union);same=np.zeros_like(union)
        for i,j in np.argwhere(union):
            i,j=int(i),int(j);key=p['texts'][j];query=dd['delex']['test'][i]['text'];prior=cache.get((key,query),[])
            labels={r['consensus'] for r in prior};consensus=next(iter(labels)) if len(labels)==1 else 'CONFLICTING_PRIOR_LABELS' if labels else 'NOT_JUDGED'
            ok=consensus in ['BOTH_DIFFERENT_TASK','BOTH_SAME_TASK','UNRESOLVED','UNRESOLVED_OR_DISAGREEMENT']
            # Unknown prior schema is retained as pending rather than guessed.
            if prior and len(labels)==1 and consensus not in ['NOT_JUDGED','CONFLICTING_PRIOR_LABELS']:ok=True
            judged[i,j]=ok;diff[i,j]=consensus=='BOTH_DIFFERENT_TASK';same[i,j]=consensus=='BOTH_SAME_TASK'
            r={'pair_id':f'{ds}:{i}:{j}','content_sha256':textsha([key,query]),'dataset':ds,'test_index':i,'key_index':j,
               'source_id':dd['delex']['test'][i]['source_id'],'key_text':key,'query_text':query,'raw_query_text':dd['raw']['test'][i]['text'],
               'raw_cosine':float(sr[i,j]),'delex_cosine':float(sd[i,j]),'consensus':consensus,'reused':ok,
               'source_judgments':json.dumps(prior,ensure_ascii=False)}
            allpairs.append(r)
            if not ok:pending.append(r)
        mask=sd>=.85;complete=bool(np.all(~mask|judged));lower=int((mask&diff).any(axis=1).sum())
        rr={'dataset':ds,'threshold':.85,'main_total_covered':int(mask.any(axis=1).sum()),'test_total':len(mask),
            'main_matched_pairs':int(mask.sum()),'main_judged_pairs':int((mask&judged).sum()),'main_missing_pairs':int((mask&~judged).sum()),
            'raw_delex_union_pairs':int(union.sum()),'union_judged_pairs':int((union&judged).sum()),'union_missing_pairs':int((union&~judged).sum()),
            'audit_complete':complete,'any_different_task_queries':lower if complete else None,'confirmed_different_task_query_lower_bound':lower,
            'all_matches_different_task_queries':int((mask.any(axis=1)&(~mask|diff).all(axis=1)).sum()) if complete else None}
        summary.append(rr)
        np.savez_compressed(OUT/ds/'audit_membership.npz',judged=judged,different=diff,same=same,raw_similarities=sr,delex_similarities=sd)
    writecsv(OUT/'task_audit_pair_alignment.csv',allpairs);writecsv(OUT/'task_audit_pending_pairs.csv',pending);writecsv(OUT/'task_audit_summary.csv',summary)
    # Adapter-compatible request file; no model calls are made by this program.
    for ds in DS:
        save(OUT/ds/'audit_pairs.json',{'protocol':'Same prior blinded operation/object/state task protocol. Exact content-based cache reuse; only unjudged or conflicting pairs need new judgments.',
             'threshold':.85,'pairs':[r for r in pending if r['dataset']==ds]})
    save(OUT/'audit_status.json',{'datasets':summary,'new_pair_count':len(pending),'required_new_model_judgments':len(pending)*2,'api_calls_made':0})
    log('AUDIT_REUSE',json.dumps(summary,ensure_ascii=False),'PENDING',len(pending))

def plot_results(budget,thresholds,calibration,paired,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,ds in zip(axs,DS):
        for method,color in [('RKGB','#b54d37'),('greedy_train_swap4','#27659a')]:
            for rep,style in [('raw','--'),('delex','-')]:
                rr=[r for r in budget if (r['dataset'],r['method'],r['representation'])==(ds,method,rep)]
                ax.plot([r['texts'] for r in rr],[r['test_coverage']*100 for r in rr],style,color=color,label=method+' '+rep)
        ax.set(title=ds,xlabel='Texts retained from frozen full-K pool',ylim=(0,100));ax.grid(alpha=.2)
    axs[0].set_ylabel('Held-out query coverage (%)');axs[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'budget_coverage.png',dpi=190);fig.savefig(OUT/'budget_coverage.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,ds in zip(axs,DS):
        for rep,color in [('raw','#27659a'),('delex','#b54d37')]:
            rr=[r for r in thresholds if r['dataset']==ds and r['method']=='RKGB' and r['representation']==rep]
            ax.plot([r['threshold'] for r in rr],[r['coverage']*100 for r in rr],'o-',color=color,label=rep)
            for target in [.9,.95]:
                row=next(r for r in calibration if r['dataset']==ds and r['method']=='RKGB' and r['representation']==rep and r['validation_recall_target']==target)
                ax.scatter(row['threshold'],row['portfolio_test_coverage']*100,marker='x' if target==.9 else 'D',s=45,color=color)
        ax.axvline(.85,color='gray',ls=':',lw=1);ax.set(title=ds,xlabel='Matching threshold',ylim=(0,100));ax.grid(alpha=.2)
    axs[0].set_ylabel('Frozen RKGB query coverage (%)');axs[-1].legend();fig.tight_layout()
    fig.savefig(OUT/'threshold_coverage.png',dpi=190);fig.savefig(OUT/'threshold_coverage.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,ds in zip(axs,DS):
        rr=[r for r in stages if r['dataset']==ds];ax.bar(range(len(rr)),[r['test_coverage']*100 for r in rr],color='#27659a')
        ax.set_xticks(range(len(rr)),[r['stage'].replace('_','\n') for r in rr],fontsize=8);ax.set(title=ds,ylim=(0,100));ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Held-out query coverage (%)');fig.tight_layout();fig.savefig(OUT/'stage_coverage.png',dpi=190);fig.savefig(OUT/'stage_coverage.svg');plt.close(fig)

def make_report(paired,calibration,budget,stages):
    audit=read(OUT/'audit_status.json')
    lines=['# 递进主线补充实验','', '日期：2026年11月1日。研究范围为阈值匹配、固定预算覆盖及任务关系。全部新增计算使用本地CPU；未启动文本生成搜索，未使用GPU。本报告为已有冻结产物的事后分析，Test结果此前已经观察。新增选择及排序仅使用Train。','',
      '## 原文与脱敏分别构造的冻结集合','', '| 数据集 | 方法 | 原文覆盖 | 脱敏覆盖 | 差值pp | 来源配对bootstrap区间 |','|---|---|---:|---:|---:|---|']
    for r in paired:lines.append(f"| {r['dataset']} | {r['method']} | {r['raw_covered']}/{r['total']} ({r['raw_covered']/r['total']:.2%}) | {r['delex_covered']}/{r['total']} ({r['delex_covered']/r['total']:.2%}) | {r['delta_pp']:.2f} | [{r['source_bootstrap_low_pp']:.2f}, {r['source_bootstrap_high_pp']:.2f}] |")
    lines+=['','区间仅描述当前冻结集合上来源配对Test的不确定性，不包含搜索随机性。MultiDoGO采用作者指定且已验证的909/1094文本集合；具体来源路径和哈希保存在main_frozen_texts.json。方法和旧组件结果必须按具体文本集合对齐，不能仅凭种子及轮数相同合并。','',
       '## 文本数量与匹配结构','', 'budget_coverage.csv 给出每个冻结完整K集合内部的Train贪心前缀。它不是每个K独立搜索，也不是全局最小文本预算。per_key_neighborhoods.csv给出每键覆盖、独占覆盖；neighborhood_summary.csv给出键间重叠和每查询匹配数。all_natural_train_anchor_degrees.csv比较同一来源Train文本在两种表示下的Test匹配度数，属于描述性结构分析。','',
       '## 同任务匹配目标下的阈值诊断','', '复用已有双模型验证的合成同任务改写。分别在原文和脱敏Validation正例上确定90%与95%召回阈值，固定后评价Test正例和冻结文本覆盖。不同表示的Test召回未被强行调成相等；具体实测值公开。此处没有在新阈值重新搜索，不据此断言适应新阈值的方法无效。','',
       '| 数据集 | 目标召回 | 原文阈值 | 脱敏阈值 | 原文正常Test召回 | 脱敏正常Test召回 | 原文集合覆盖 | 脱敏集合覆盖 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds in DS:
        for target in [.9,.95]:
            rr={r['representation']:r for r in calibration if r['dataset']==ds and r['method']=='RKGB' and r['validation_recall_target']==target};a,b=rr['raw'],rr['delex']
            lines.append(f"| {ds} | {target:.0%} | {a['threshold']:.4f} | {b['threshold']:.4f} | {a['normal_test_recall']:.2%} | {b['normal_test_recall']:.2%} | {a['portfolio_test_coverage']:.2%} | {b['portfolio_test_coverage']:.2%} |")
    lines+=['','## 连续目标到实际文本','', 'stage_coverage.csv在相同Test划分、K和阈值下记录各阶段覆盖。最近自然Train锚点是独立诊断对照，不冒称实际初始化；若实际初始化可取得则另列。各阶段集合不保证包含关系，连续覆盖不是所有可能文本集合的理论上界。','',
        '| 数据集 | 阶段 | 覆盖 |','|---|---|---:|']
    for r in stages:lines.append(f"| {r['dataset']} | {r['stage']} | {r['test_covered']}/{r['test_total']} ({r['test_coverage']:.2%}) |")
    lines+=['','## 任务关系判断的复用与缺口','', '| 数据集 | 总覆盖 | 主结果匹配对 | 已有判断覆盖对数 | 缺失对数 | 完整审计 |','|---|---:|---:|---:|---:|---|']
    for r in audit['datasets']:lines.append(f"| {r['dataset']} | {r['main_total_covered']}/{r['test_total']} | {r['main_matched_pairs']} | {r['main_judged_pairs']} | {r['main_missing_pairs']} | {r['audit_complete']} |")
    lines+=['','复用按键文本与规范化表示查询的精确内容匹配，并保留来源记录。task_audit_pending_pairs.csv只包含未判定或冲突的原文/脱敏并集配对。缺失判断不当作同任务，未完整审计的数据不报告完整不同任务覆盖率。该程序没有调用API。','',
        '## 后续资源相关工作','', '本轮完成已有产物可支撑的CPU分析。若需新增同等生成资源的方法对照，必须另行冻结预算、候选生成和筛选协议后运行；本轮没有将已有不等成本消融包装为等预算比较。待判任务对可直接交由原双模型协议，需使用可用API配置；未取得的判断不填数。','',
        '## 复现','', 'code目录保存完整脚本；experiment_plan.json定义分析协议；input_manifest.json记录输入文件SHA256；每个数据集保存冻结文本、Train排序及Train/Test相似度矩阵。输出不覆盖原始实验文件。']
    (OUT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
