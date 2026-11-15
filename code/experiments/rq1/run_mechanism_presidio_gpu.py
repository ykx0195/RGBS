"""Dose/type ablation and real Presidio pipeline replication on all records."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DOMAINS=('healthcare','finance','banking','insurance','legal')
MODELS=('minilm','gte','e5','bge')
METHODS=('generic','typed','typed_numbered','delete','star')
GROUPS={
    'person': {'first_name','last_name','middle_name','full_name','PERSON'},
    'temporal': {'date','date_of_birth','date_time','time','DATETIME'},
    'identifier': {'account_number','medical_record_number','health_plan_beneficiary_number','customer_id',
        'ssn','credit_debit_card','bank_routing_number','swift_bic','tax_id','cvv','CODE','passport_number',
        'drivers_license_number','national_id'},
}
FAMILIES=('gold_all','dose_one','dose_half','type_person','type_temporal','type_identifier','presidio')
FIELDS=('source_text',)+tuple(f'{family}_{method}' for family in FAMILIES for method in METHODS)
TARGETS=(0.001,0.01)
FIXED=(0.75,0.775,0.8,0.825,0.85,0.875,0.9)
SEED='pii_span_nested_ablation_20261018'

def variants(text,spans):
    result={m:[] for m in METHODS};count=Counter();cursor=0
    for s in sorted(spans,key=lambda x:(x['start'],x['end'])):
        a,b,label=int(s['start']),int(s['end']),s['label'].upper()
        assert cursor<=a<b<=len(text)
        count[label]+=1
        values={'generic':'[PII]','typed':f'[{label}]','typed_numbered':f'[{label}_{count[label]}]','delete':'','star':'*'}
        for m in METHODS:result[m].extend((text[cursor:a],values[m]))
        cursor=b
    for pieces in result.values():pieces.append(text[cursor:])
    return {m:''.join(p) for m,p in result.items()}

def expand(r):
    spans=sorted(r['spans'],key=lambda s:hashlib.sha256(f"{SEED}:{r['record_id']}:{s['start']}:{s['end']}".encode()).digest())
    chosen={'gold_all':spans,'dose_one':spans[:1],'dose_half':spans[:math.ceil(len(spans)/2)]}
    for group,labels in GROUPS.items():chosen['type_'+group]=[s for s in spans if s['label'] in labels][:1]
    result=dict(r);result['n_gold_spans']=len(spans)
    result['n_modified']={k:len(s) for k,s in chosen.items()};result['n_modified']['presidio']=len(r['predicted_spans'])
    result['selected_gold_spans']=chosen
    for family,selected in chosen.items():
        result.update({f'{family}_{m}':t for m,t in variants(r['source_text'],selected).items()})
    return result

def split_indices(records,domain):
    seed='TAB_threshold_replication_20261018' if domain=='legal' else '20261018_threshold_v1'
    order=sorted(range(len(records)),key=lambda i:hashlib.sha256(f"{seed}:{records[i]['record_id']}".encode()).digest())
    cut=round(.6*len(records))
    return np.sort(np.asarray(order[:cut])),np.sort(np.asarray(order[cut:]))

def write_rows(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pairs(embedding,indices):
    selected=np.asarray(embedding[indices]);matrix=selected@selected.T
    np.clip(matrix,-1,1,out=matrix)
    return matrix[np.triu_indices(len(indices),1)]

def interval(values,pi,pj,n):
    count=n*(n-1)//2
    incident=np.bincount(pi,weights=values,minlength=n)+np.bincount(pj,weights=values,minlength=n)
    leave=(float(values.sum())-incident)/((n-1)*(n-2)/2)
    se=math.sqrt((n-1)/n*float(np.sum((leave-leave.mean())**2)))
    point=float(values.mean());return point-1.96*se,point+1.96*se

def metrics(original,changed,tau,pi,pj,n):
    b=original>=tau;a=changed>=tau
    up=int((~b&a).sum());down=int((b&~a).sum());net=up-down
    assert net==int(a.sum())-int(b.sum())
    low,high=interval(a.astype(np.int8)-b.astype(np.int8),pi,pj,n)
    return {'n_test_records':n,'n_test_pairs':len(b),'original_above_count':int(b.sum()),'modified_above_count':int(a.sum()),
        'upward_count':up,'downward_count':down,'net_count':net,'upward_rate':up/len(b),'downward_rate':down/len(b),
        'net_change':net/len(b),'record_jackknife_ci95_low':low,'record_jackknife_ci95_high':high,
        'mean_cosine_change':float(np.mean(changed-original))}

def thresholds(original_cal):
    return [('original_tail',t,float(np.quantile(original_cal,1-t))) for t in TARGETS]+[('fixed',t,t) for t in FIXED]

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--models-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);assert torch.cuda.is_available()
    models={key:SentenceTransformer(str(args.models_dir/key),device='cuda') for key in MODELS}
    rows=[];contrasts=[];dose_contrasts=[];audits=[];started=time.time()
    for domain in DOMAINS:
        source_path=args.input_dir.parent/'input'/f'{domain}.jsonl'
        expected=sum(1 for _ in source_path.open(encoding='utf-8'))
        deadline=time.time()+1800
        while True:
            path=args.input_dir/f'{domain}.jsonl'
            try:
                loaded=[json.loads(s) for s in path.open(encoding='utf-8')]
                if len(loaded)==expected:break
            except (FileNotFoundError,json.JSONDecodeError):pass
            if time.time()>deadline:raise TimeoutError(f'Incomplete Presidio input for {domain}')
            time.sleep(5)
        candidates=[expand(r) for r in loaded]
        assert len(candidates)==len({r['record_id'] for r in candidates})
        parent_cal,parent_test=split_indices(candidates,domain)
        fit=np.ones(len(candidates),dtype=bool);token_audit={}
        for key,model in models.items():
            lengths=np.zeros(len(candidates),dtype=np.int32)
            for field in FIELDS:
                texts=[('query: ' if key=='e5' else '')+r[field] for r in candidates]
                for start in range(0,len(texts),256):
                    lens=np.asarray([len(x) for x in model.tokenizer(texts[start:start+256],padding=False,truncation=False,add_special_tokens=True)['input_ids']])
                    lengths[start:start+len(lens)]=np.maximum(lengths[start:start+len(lens)],lens)
            fit &= lengths<=256
            token_audit[key]={'max_tokens':int(lengths.max()),'excluded':int((lengths>256).sum())}
            print('token',domain,key,token_audit[key],flush=True)
        records=[r for r,keep in zip(candidates,fit) if keep];assert len(records)>.8*len(candidates)
        # Retain original parent split membership after common token filtering.
        old_to_new=np.full(len(candidates),-1,dtype=int);old_to_new[fit]=np.arange(len(records))
        cal=old_to_new[parent_cal[fit[parent_cal]]];test=old_to_new[parent_test[fit[parent_test]]]
        out=args.output_dir/domain;out.mkdir(exist_ok=True)
        with (out/'selected_records.jsonl').open('w',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        dose=np.asarray([r['n_gold_spans']>=3 for r in records])
        cohorts={'full':(cal,test),'dose_ge_3':(cal[dose[cal]],test[dose[test]])}
        for group in GROUPS:
            eligible=np.asarray([r['n_modified']['type_'+group]>0 for r in records]);cohorts['type_'+group]=(cal[eligible[cal]],test[eligible[test]])
        audit={'domain':domain,'candidate_records':len(candidates),'retained_records':len(records),'token_audit':token_audit,
            'cohorts':{name:{'cal_records':len(c),'test_records':len(t)} for name,(c,t) in cohorts.items()}}
        audits.append(audit); print('cohorts',domain,json.dumps(audit),flush=True)
        for key,model in models.items():
            embeddings={};cache={}
            for field in FIELDS:
                texts=[r[field] for r in records]
                missing=list(dict.fromkeys(t for t in texts if t not in cache))
                if missing:
                    prepared=[('query: ' if key=='e5' else '')+t for t in missing]
                    mat=np.asarray(model.encode(prepared,batch_size=args.batch_size,show_progress_bar=False,
                        convert_to_numpy=True,normalize_embeddings=True,precision='float32'),dtype=np.float32)
                    for text,vector in zip(missing,mat):cache[text]=vector
                matrix=np.stack([cache[t] for t in texts]);assert matrix.shape==(len(records),384)
                np.save(out/f'{key}_{field}.npy',matrix);embeddings[field]=matrix
                print('encoded',domain,key,field,'unique_new',len(missing),flush=True)
            # Original thresholds for full and dose cohorts; type cohorts share the
            # full-domain original threshold, making their operating point explicit.
            reference={name:thresholds(pairs(embeddings['source_text'],c)) for name,(c,t) in cohorts.items() if name in ('full','dose_ge_3')}
            for family in FAMILIES:
                cohort='dose_ge_3' if family.startswith('dose_') else family if family.startswith('type_') else 'full'
                c,t=cohorts[cohort]
                if len(t)<30:
                    print('SKIP small cohort',domain,key,family,len(t),flush=True);continue
                original=pairs(embeddings['source_text'],t);pi,pj=np.triu_indices(len(t),1)
                threshold_ref='dose_ge_3' if cohort=='dose_ge_3' else 'full'
                for method in METHODS:
                    changed=pairs(embeddings[f'{family}_{method}'],t)
                    for threshold_kind,target,tau in reference[threshold_ref]:
                        rows.append({'dataset':records[0]['dataset'],'domain':domain,'model':key,'family':family,'method':method,
                            'cohort':cohort,'threshold_reference':threshold_ref,'threshold_kind':threshold_kind,'target':target,'threshold':tau,
                            'n_calibration_records':len(cohorts[threshold_ref][0]),'mean_modified_spans':float(np.mean([records[i]['n_modified'][family] for i in t])),
                            **metrics(original,changed,tau,pi,pj,len(t))})
                    if family=='presidio':
                        gold=pairs(embeddings[f'gold_all_{method}'],t)
                        for threshold_kind,target,tau in reference['full']:
                            delta=(changed>=tau).astype(np.int8)-(gold>=tau).astype(np.int8)
                            low,high=interval(delta,pi,pj,len(t))
                            contrasts.append({'dataset':records[0]['dataset'],'domain':domain,'model':key,'method':method,
                                'threshold_kind':threshold_kind,'target':target,'threshold':tau,'n_test_records':len(t),'n_test_pairs':len(delta),
                                'presidio_minus_gold_net_change':float(delta.mean()),'record_jackknife_ci95_low':low,'record_jackknife_ci95_high':high})
            # Full redaction evaluated on the exact same >=3-span dose cohort.
            c,t=cohorts['dose_ge_3'];original=pairs(embeddings['source_text'],t);pi,pj=np.triu_indices(len(t),1)
            for method in METHODS:
                changed=pairs(embeddings[f'gold_all_{method}'],t)
                for threshold_kind,target,tau in reference['dose_ge_3']:
                    rows.append({'dataset':records[0]['dataset'],'domain':domain,'model':key,'family':'dose_all','method':method,
                        'cohort':'dose_ge_3','threshold_reference':'dose_ge_3','threshold_kind':threshold_kind,'target':target,'threshold':tau,
                        'n_calibration_records':len(c),'mean_modified_spans':float(np.mean([records[i]['n_gold_spans'] for i in t])),
                        **metrics(original,changed,tau,pi,pj,len(t))})
                one=pairs(embeddings[f'dose_one_{method}'],t)
                half=pairs(embeddings[f'dose_half_{method}'],t)
                for comparison,first,second in [('half_minus_one',one,half),('all_minus_half',half,changed)]:
                    for threshold_kind,target,tau in reference['dose_ge_3']:
                        delta=(second>=tau).astype(np.int8)-(first>=tau).astype(np.int8)
                        low,high=interval(delta,pi,pj,len(t))
                        dose_contrasts.append({'dataset':records[0]['dataset'],'domain':domain,'model':key,'method':method,
                            'comparison':comparison,'threshold_kind':threshold_kind,'target':target,'threshold':tau,
                            'n_test_records':len(t),'n_test_pairs':len(delta),'incremental_net_change':float(delta.mean()),
                            'record_jackknife_ci95_low':low,'record_jackknife_ci95_high':high})
            del embeddings,cache
            write_rows(args.output_dir/'collision_results.csv',rows);write_rows(args.output_dir/'presidio_vs_gold_contrasts.csv',contrasts)
            write_rows(args.output_dir/'dose_increment_contrasts.csv',dose_contrasts)
            print('completed',domain,key,'elapsed_min',round((time.time()-started)/60,1),flush=True)
    manifest={'design':'All previous main records; common <=256 token filter over all 36 versions and four tokenizers. Parent 60/40 membership retained after filtering.',
        'dose':'Same >=3-gold-span records for one, ceil(n/2), all; nested hash-ranked spans; deterministic seed '+SEED,
        'type':'One hash-ranked eligible span per type; only records containing that group evaluated; full-domain original thresholds shared; cohorts differ, so cross-type magnitude is descriptive.',
        'groups':{k:sorted(v) for k,v in GROUPS.items()},'thresholds':{'tail_targets':TARGETS,'fixed':FIXED},
        'models':MODELS,'methods':METHODS,'fields':FIELDS,'gpu':torch.cuda.get_device_name(0),'audits':audits,
        'software':{p:importlib.metadata.version(p) for p in ('torch','numpy','sentence-transformers','transformers')},
        'uncertainty':'Record-delete jackknife normal pointwise 95% interval; original calibration threshold held fixed; no multiple-comparison correction.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('DONE',len(rows),'collision rows',len(contrasts),'contrast rows',flush=True)

if __name__=='__main__':main()
