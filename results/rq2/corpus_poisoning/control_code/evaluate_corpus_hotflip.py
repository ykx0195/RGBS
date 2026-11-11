"""Evaluate predeclared Train-frozen HotFlip portfolios; never select on Test."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rr):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def main():
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    base=Path.home()/'Desktop/results';root=base/'CorpusPoisoning_HotFlip对比_20261031';plan=read(root/'experiment_plan.json')
    out=root/'evaluation';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'code'));from hybrid_search import BGE
    selections={};inputs=[]
    for dataset,k in plan['datasets'].items():
        portfolios={}
        for budget in plan['checkpoint_budgets']:
            rr=[]
            for cluster in range(k):
                run=root/'runs'/dataset/f'cluster_{cluster:02d}';final=read(run/'final_cluster.json');assert final['api_calls']==0 and final['heldout_loaded'] is False
                assert final['completed_iterations']==5000
                checkpoint=read(run/f'budget_{budget:05d}.json');rr.append({'cluster':cluster,**checkpoint['selected']})
                inputs.extend([run/'final_cluster.json',run/f'budget_{budget:05d}.json'])
            portfolios[str(budget)]=rr
        selections[dataset]={'k':k,'portfolios':portfolios,'selection_uses_test':False,'checkpoint_rule':'Full-cluster Train mean score only'}
    # Freeze all budget portfolios before opening any held-out arrays in this stage.
    save(out/'frozen_portfolios.json',selections)
    encoder=BGE(Path.home()/'Desktop/ykx/bge-small-en-v1.5','cpu','bge')
    metrics=[];comparison=[];keyrecords=[]
    for dataset,selection in selections.items():
        dd=out/dataset;dd.mkdir(exist_ok=True);texts=list(dict.fromkeys(x['text'] for rr in selection['portfolios'].values() for x in rr))
        vectors=encoder.encode(texts,32);lookup={t:v for t,v in zip(texts,vectors)}
        save(dd/'key_texts.json',texts);np.savez_compressed(dd/'key_vectors.npz',vectors=vectors)
        prior=base/'固定文本_原文脱敏配对_20261031';prior=prior/'ABCD补充/abcd' if dataset=='abcd' else prior/dataset
        embid=read(prior/'embedding_identity.json');identity=read(root/'inputs'/dataset/'train_vectors_identity.json')
        assert embid['model_sha256']==identity['model_sha256']==sha(Path.home()/'Desktop/ykx/bge-small-en-v1.5/model.safetensors')
        test=np.load(prior/'embeddings.npz')['delex'];train=np.load(root/'inputs'/dataset/'train_vectors.npy');k=selection['k']
        matrices={}
        for budget,rr in selection['portfolios'].items():
            keys=np.stack([lookup[x['text']] for x in rr]);matrix=test@keys.T;tr=train@keys.T;matrices['budget_'+budget]=matrix
            for threshold in [.8,.825,.85,.875,.9]:
                mask=(matrix>=threshold).any(axis=1)
                metrics.append({'dataset':dataset,'method':'CorpusPoisoning_adapted','seed':42,'iteration_budget_per_cluster':int(budget),
                                'k_budget':k,'unique_output_texts':len({x['text'] for x in rr}),'threshold':threshold,
                                'train_covered':int((tr>=threshold).any(axis=1).sum()),'train_total':len(train),
                                'test_covered':int(mask.sum()),'test_total':len(test),'test_coverage':float(mask.mean())})
        np.savez_compressed(dd/'test_similarities.npz',**matrices)
        raw=next(x for x in metrics if x['dataset']==dataset and x['iteration_budget_per_cluster']==5000 and x['threshold']==.85)
        comparison.append(raw)
        reference=read(base/'三数据集_脱敏_32x50_20261030/runs'/dataset/'final_report.json')
        comparison.append({'dataset':dataset,'method':'RKGB','seed':42,'iteration_budget_per_cluster':None,'k_budget':k,
                           'unique_output_texts':len({x['text'] for x in reference['selected']}),'threshold':.85,
                           'train_covered':reference['metrics']['train']['covered'],'train_total':reference['metrics']['train']['total'],
                           'test_covered':reference['metrics']['test']['covered'],'test_total':reference['metrics']['test']['total'],
                           'test_coverage':reference['metrics']['test']['coverage']})
        for method,rr in [('CorpusPoisoning_adapted',selection['portfolios']['5000']),('RKGB',reference['selected'])]:
            for j,x in enumerate(rr):keyrecords.append({'dataset':dataset,'method':method,'slot':j,'text':x['text']})
    save(root/'quality_inputs.json',{'keys':keyrecords,'evaluation_protocol':'Both methods: local Qwen2.5-1.5B PPL<=150 and local customer-request readability. No API or textual repair. Not equivalent to the full RKGB in-search acceptance pipeline.'})
    csvwrite(out/'all_budget_coverage.csv',metrics);csvwrite(out/'primary_raw_comparison.csv',comparison)
    save(out/'raw_summary.json',{'raw_evaluation_complete':True,'quality_evaluation_complete':False,'metrics':comparison,'all_metrics':metrics,
                                'api_calls':0,'source_inputs_sha256':{str(p):sha(p) for p in set(inputs)}})
    print('HOTFLIP_FROZEN_RAW_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
