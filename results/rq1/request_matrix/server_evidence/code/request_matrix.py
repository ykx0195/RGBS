"""Reproducible request-only extension of the RQ1 collision matrix.

Source-aligned derived fields and pinned Presidio are separate PII pipelines.
No generation, judge API, RKGB selection, or test-dependent threshold tuning.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, collections, csv, datetime, hashlib, importlib.metadata, json, math, time
from pathlib import Path

DATASETS=('banking77','abcd','multidogo')
METHODS=('generic','typed','typed_numbered','delete','mask_length')
FAMILIES=('dataset_fields','presidio')
FIXED=(.75,.775,.8,.825,.85,.875,.9)
MODELS={
 'minilm':'/root/autodl-tmp/ykx_crossmodel_abcd_20261026/models/minilm',
 'gte':'/root/autodl-tmp/ykx_crossmodel_abcd_20261026/models/gte',
 'e5':'/root/autodl-tmp/ykx_crossmodel_abcd_20261026/models/e5',
 'bge':'/root/autodl-tmp/liman/models/bge-small-en-v1.5'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)

def csvout(p,rows):
    assert rows
    with Path(str(p)+'.tmp').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    Path(str(p)+'.tmp').replace(p)

def records(job):
    return [json.loads(l) for l in (job/'input/request_records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def jlines(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=Path(str(p)+'.tmp')
    with q.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    q.replace(p)

def render(text,spans):
    result={m:[] for m in METHODS};counts=collections.Counter();cur=0
    for s in sorted(spans,key=lambda s:(s['start'],s['end'])):
        a,b,label=int(s['start']),int(s['end']),s['label'].upper()
        assert cur<=a<b<=len(text)
        counts[label]+=1
        v={'generic':'[PII]','typed':f'[{label}]','typed_numbered':f'[{label}_{counts[label]}]',
           'delete':'','mask_length':'*'*(b-a)}
        for m in METHODS:result[m].extend((text[cur:a],v[m]))
        cur=b
    for x in result.values():x.append(text[cur:])
    out={k:''.join(v) for k,v in result.items()}
    assert len(out['mask_length'])==len(text)
    return out

def prepare_gold(job):
    rr=records(job)
    for ds in DATASETS:
        rows=[r for r in rr if r['dataset']==ds]
        train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
        assert not ({r['source_id'] for r in train}&{r['source_id'] for r in test})
        for split,x in [('train',train),('test',test)]:
            assert len({r['source_id'] for r in x})==len(x)
        for r in test:r.update({f'dataset_fields_{m}':t for m,t in render(r['source_text'],r['spans']).items()})
        jlines(job/f'output/prepared/{ds}_dataset_fields.jsonl',test)
        print('GOLD_PREPARED',ds,len(test),flush=True)
    atomic_json(job/'output/gold_ready.json',{'input_sha256':sha(job/'input/request_records.jsonl'),
        'span_provenance':'Strict exact raw/template alignment from derived benchmark materialization; not independent human PII labels.',
        'methods':METHODS})

def prepare_presidio(job):
    import logging
    logging.basicConfig(level=logging.ERROR)
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig,ConflictResolutionStrategy
    modelpath=job/'nlp/en_core_web_lg-3.8.0'
    nlp=NlpEngineProvider(nlp_configuration={'nlp_engine_name':'spacy','models':[{'lang_code':'en','model_name':str(modelpath)}]}).create_engine()
    analyzer=AnalyzerEngine(nlp_engine=nlp,supported_languages=['en'],default_score_threshold=.5)
    engine=AnonymizerEngine();audit=[]
    for ds in DATASETS:
        dest=job/f'output/prepared/{ds}_presidio.jsonl'
        if dest.exists():
            print('PRESIDIO_RESUME',ds,flush=True);continue
        rows=[r for r in records(job) if r['dataset']==ds and r['split']=='test']
        for i,r in enumerate(rows):
            t=r['source_text'];found=analyzer.analyze(text=t,language='en',score_threshold=.5)
            kept=engine.anonymize(text=t,analyzer_results=found,operators={'DEFAULT':OperatorConfig('keep')},
                conflict_resolution=ConflictResolutionStrategy.REMOVE_INTERSECTIONS,merge_entities_with_spaces=False)
            assert kept.text==t
            spans=sorted([{'start':x.start,'end':x.end,'label':x.entity_type} for x in kept.items],key=lambda x:(x['start'],x['end']))
            r['predicted_spans']=spans
            r['raw_analyzer_results']=[{'start':x.start,'end':x.end,'label':x.entity_type,'score':x.score} for x in found]
            r.update({f'presidio_{m}':v for m,v in render(t,spans).items()})
            audit.append({'dataset':ds,'record_id':r['record_id'],'source_fields':len(r['spans']),'detected_fields':len(spans)})
            if (i+1)%100==0:print('PRESIDIO',ds,i+1,'/',len(rows),flush=True)
        jlines(dest,rows)
        print('PRESIDIO_PREPARED',ds,len(rows),flush=True)
    versions={k:importlib.metadata.version(k) for k in ('presidio-analyzer','presidio-anonymizer','spacy')}
    atomic_json(job/'output/presidio_ready.json',{'versions':versions,'nlp_model_version':'3.8.0',
        'nlp_meta_sha256':sha(modelpath/'meta.json'),'score_threshold':.5,
        'overlap_resolution':'REMOVE_INTERSECTIONS; merge_entities_with_spaces=False',
        'entities':'Default supported English recognizers; no custom dataset recognizers'})
    if audit:csvout(job/'output/detection_audit.csv',audit)

def encode(job,key,batch):
    import numpy as np,torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SentenceTransformer(MODELS[key],device=device);model.eval();model.max_seq_length=256
    dest=job/f'output/embeddings/{key}';dest.mkdir(parents=True,exist_ok=True)
    weights={str(p.relative_to(MODELS[key])):sha(p) for p in Path(MODELS[key]).rglob('*') if p.is_file() and p.suffix in ('.bin','.safetensors','.json','.txt')}
    identity={'model':key,'model_files':weights,'prefix':'query: ' if key=='e5' else '',
        'max_seq_length':256,'dtype':'float32','device':device,'batch_size':batch,
        'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
        'sentence_transformers':importlib.metadata.version('sentence-transformers')}
    atomic_json(dest/'model_identity.json',identity)
    rr=records(job);total_encoded=0;start=time.time()
    def field(ds,name,texts):
        nonlocal total_encoded
        p=dest/f'{ds}_{name}.npy';lp=dest/f'{ds}_{name}_lengths.npy';meta=dest/f'{ds}_{name}.json'
        prepared=[('query: '+s) if key=='e5' else s for s in texts]
        textsha=hashlib.sha256(json.dumps(prepared,ensure_ascii=False).encode()).hexdigest()
        ident=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if p.exists() and lp.exists() and meta.exists():
            old=json.loads(meta.read_text())
            if old['text_sha256']==textsha and old['model_identity_sha256']==ident and old['embedding_sha256']==sha(p):
                print('CACHED',key,ds,name,flush=True);return
        lengths=np.asarray([len(x) for x in model.tokenizer(prepared,truncation=False,padding=False,add_special_tokens=True)['input_ids']],dtype=np.int32)
        unique=list(dict.fromkeys(prepared));index={s:i for i,s in enumerate(unique)}
        print('ENCODING',key,ds,name,len(texts),'unique',len(unique),device,flush=True)
        tick=time.time()
        with torch.inference_mode():
            emb=model.encode(unique,batch_size=batch,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
        arr=np.asarray(emb[[index[s] for s in prepared]],dtype=np.float32)
        assert arr.shape==(len(texts),384) and np.isfinite(arr).all()
        assert np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-5)
        with Path(str(p)+'.tmp').open('wb') as f:np.save(f,arr)
        Path(str(p)+'.tmp').replace(p);np.save(lp,lengths)
        atomic_json(meta,{'text_sha256':textsha,'model_identity_sha256':ident,'embedding_sha256':sha(p),
            'records':len(texts),'unique_encoded':len(unique),'seconds':time.time()-tick,'max_tokens':int(lengths.max())})
        total_encoded+=len(unique)
        atomic_json(job/f'output/progress_{key}.json',{'model':key,'dataset':ds,'field':name,'encoded':total_encoded,'seconds':time.time()-start})
    for ds in DATASETS:
        train=[r for r in rr if r['dataset']==ds and r['split']=='train']
        test=[r for r in rr if r['dataset']==ds and r['split']=='test']
        field(ds,'calibration_original',[r['source_text'] for r in train])
        field(ds,'original',[r['source_text'] for r in test])
        field(ds,'legacy_delex',[r['legacy_delex'] for r in test])
        for family in FAMILIES:
            path=job/f'output/prepared/{ds}_{family}.jsonl'
            while not path.exists():
                print('WAITING_PREPARATION',key,ds,family,flush=True);time.sleep(5)
            rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
            assert [r['record_id'] for r in rows]==[r['record_id'] for r in test]
            for method in METHODS:
                name=family+'_'+method;field(ds,name,[r[name] for r in rows])
    atomic_json(job/f'output/complete_{key}.json',{'model':key,'encoded':total_encoded,'seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

def analyze(job):
    import numpy as np
    rr=records(job);audits={};rows=[];legacy=[];threshold_rows=[]
    for key in MODELS:assert (job/f'output/complete_{key}.json').exists(),key
    for ds in DATASETS:
        train=[r for r in rr if r['dataset']==ds and r['split']=='train'];test=[r for r in rr if r['dataset']==ds and r['split']=='test']
        keep_cal=np.ones(len(train),bool);keep_test=np.ones(len(test),bool)
        names=['original','legacy_delex']+[f'{f}_{m}' for f in FAMILIES for m in METHODS]
        maxlens={}
        for key in MODELS:
            dest=job/f'output/embeddings/{key}'
            lens=np.load(dest/f'{ds}_calibration_original_lengths.npy');keep_cal&=lens<=256
            mx=np.maximum.reduce([np.load(dest/f'{ds}_{name}_lengths.npy') for name in names]);keep_test&=mx<=256
            maxlens[key]={'calibration_max_tokens':int(lens.max()),'test_all_versions_max_tokens':int(mx.max())}
        ci=np.flatnonzero(keep_cal);ti=np.flatnonzero(keep_test);assert len(ci)>100 and len(ti)>100
        audits[ds]={'calibration_source':'Original benchmark Train only; source-disjoint from Test. This differs from document-corpus 60/40 splits.',
            'calibration_original_records':len(train),'test_original_records':len(test),'calibration_retained':len(ci),'test_retained':len(ti),
            'test_record_ids':[test[i]['record_id'] for i in ti], 'calibration_record_ids':[train[i]['record_id'] for i in ci],
            'test_excluded_ids':[test[i]['record_id'] for i in np.flatnonzero(~keep_test)],'tokenizers':maxlens}
        pi,pj=np.triu_indices(len(ti),1);n=len(ti);den=len(pi)
        def pairs(x,ind):
            a=np.asarray(x[ind],dtype=np.float32);z=a@a.T;np.clip(z,-1,1,out=z)
            return z[np.triu_indices(len(ind),1)]
        def measure(a,b,tau):
            before=a>=tau;after=b>=tau;d=after.astype(np.int8)-before.astype(np.int8)
            up=int((~before&after).sum());down=int((before&~after).sum())
            inc=np.bincount(pi,weights=d,minlength=n)+np.bincount(pj,weights=d,minlength=n)
            leave=(int(d.sum())-inc)/((n-1)*(n-2)/2)
            se=math.sqrt((n-1)/n*float(((leave-leave.mean())**2).sum()));point=float(d.mean())
            assert up-down==int(after.sum())-int(before.sum())
            return {'n_test_records':n,'n_test_pairs':den,'original_above_count':int(before.sum()),'modified_above_count':int(after.sum()),
                'original_rate':float(before.mean()),'modified_rate':float(after.mean()),'upward_count':up,'downward_count':down,
                'net_count':up-down,'net_change':point,'delta_pp':100*point,
                'record_jackknife_ci95_low':point-1.96*se,'record_jackknife_ci95_high':point+1.96*se,
                'mean_cosine_change':float((b-a).mean())}
        for key in MODELS:
            dest=job/f'output/embeddings/{key}'
            original=pairs(np.load(dest/f'{ds}_original.npy'),ti)
            cal=pairs(np.load(dest/f'{ds}_calibration_original.npy'),ci)
            op=[('fixed',t,t) for t in FIXED]+[('original_tail',q,float(np.quantile(cal,1-q))) for q in (.001,.01)]
            for kind,target,tau in op:
                threshold_rows.append({'dataset':ds,'model':key,'threshold_kind':kind,'target':target,'threshold':tau,'n_calibration_records':len(ci),'n_calibration_pairs':len(cal)})
            del cal
            for family in FAMILIES:
                for method in METHODS:
                    changed=pairs(np.load(dest/f'{ds}_{family}_{method}.npy'),ti)
                    for kind,target,tau in op:
                        rows.append({'dataset':ds,'domain':ds,'model':key,'family':family,'method':method,
                            'cohort':'request_common_256','threshold_kind':kind,'target':target,'threshold':tau,
                            'n_calibration_records':len(ci),**measure(original,changed,tau)})
            changed=pairs(np.load(dest/f'{ds}_legacy_delex.npy'),ti)
            for tau in FIXED:legacy.append({'dataset':ds,'model':key,'threshold':tau,**measure(original,changed,tau)})
    assert len(rows)==3*4*2*5*9
    keys={(r['dataset'],r['model'],r['family'],r['method'],r['threshold_kind'],r['target']) for r in rows}
    assert len(keys)==len(rows)
    out=job/'output';csvout(out/'collision_results_all_thresholds.csv',rows)
    primary=[r for r in rows if r['threshold_kind']=='fixed' and r['target']==.85]
    assert len(primary)==120
    csvout(out/'collision_results_tau085.csv',primary);csvout(out/'calibrated_thresholds.csv',threshold_rows);csvout(out/'legacy_delex_comparison.csv',legacy)
    atomic_json(out/'cohort_audit.json',audits)
    atomic_json(out/'manifest.json',{'status':'complete','expected_matrix_rows':1080,'actual_matrix_rows':len(rows),
        'primary_rows':len(primary),'datasets':DATASETS,'models':list(MODELS),'families':FAMILIES,'methods':METHODS,
        'fixed_thresholds':FIXED,'original_tail_targets':[.001,.01],
        'input_sha256':sha(job/'input/request_records.jsonl'),'code_sha256':sha(Path(__file__)),
        'filter':'Common <=256 tokens including special tokens/prefix, across all four tokenizers and 12 Test representations. Original calibration records jointly filtered across four tokenizers.',
        'split':'Original benchmark Train for calibration; original Test for evaluation, no source overlap.',
        'derived_field_provenance':'Exact original/placeholder materialization alignment, not human PII annotations.',
        'uncertainty':'Pointwise source-record-delete jackknife normal 95% intervals conditional on calibration thresholds; no search/API calls; not simultaneous intervals.',
        'numerics':'Unit-normalized float32 embeddings and float32 cosine dot products; TF32 disabled.',
        'new_generation_calls':0,'new_judge_calls':0,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    md=['# 三请求集 RQ1 完整矩阵（新实验）','','固定阈值 0.85；数值为碰撞率变化，单位百分点。','',
        '| 数据集 | 编码器 | PII 来源 | Generic | Type | Type+index | Delete | Length mask |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for key in MODELS:
            for family in FAMILIES:
                vals=[next(r['delta_pp'] for r in primary if (r['dataset'],r['model'],r['family'],r['method'])==(ds,key,family,m)) for m in METHODS]
                md.append('| '+' | '.join([ds,key,family]+[f'{v:+.3f}' for v in vals])+' |')
    md+=['','完整矩阵 1080 行，0.85 主表 120 行。原始/脱敏碰撞率、匹配数量和记录级区间见 CSV。',
        '', 'dataset_fields 来自派生请求的原文/占位符严格对齐，不是新增人工标注。Presidio 使用独立的自动识别结果。',
        '', '校准沿用 benchmark Train，评价沿用 Test；文档数据使用其原有 60/40 划分。两组分组展示，不合并为同一总体。',
        '', '新规则采用方括号占位符，与旧角括号脱敏表示不同。旧表示另列 legacy_delex_comparison.csv，不覆盖旧结果。']
    (out/'table_tau085.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('ANALYSIS_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare-gold','prepare-presidio','encode','analyze']);p.add_argument('--job',type=Path,required=True);p.add_argument('--model',choices=list(MODELS));p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    (a.job/'output').mkdir(exist_ok=True,parents=True)
    if a.phase=='prepare-gold':prepare_gold(a.job)
    elif a.phase=='prepare-presidio':prepare_presidio(a.job)
    elif a.phase=='encode':encode(a.job,a.model,a.batch_size)
    else:analyze(a.job)
