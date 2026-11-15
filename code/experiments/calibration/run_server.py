import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
MODELS={'bge-small-en-v1.5':'bge','all-MiniLM-L6-v2':'minilm','gte-small':'gte','e5-small-v2':'e5'}

def read(path):
    return json.loads(path.read_text('utf-8'))

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def worker(name):
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    directory=Path('/root/autodl-tmp/liman/models')/name if MODELS[name]=='bge' else ROOT/'models'/name
    expected=next(r for r in read(ROOT/'inputs/expected_models.json') if r['model']==name)
    weight=directory/'model.safetensors'
    if not weight.exists(): weight=directory/'pytorch_model.bin'
    for path,field in [(weight,'model_sha256'),(directory/'tokenizer.json','tokenizer_sha256'),(directory/'1_Pooling/config.json','pooling_sha256'),(directory/'sentence_bert_config.json','sequence_sha256')]:
        assert sha(path)==expected[field],(name,field)
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model=AutoModel.from_pretrained(directory,local_files_only=True,dtype=torch.float32).to('cuda').eval()
    assert next(model.parameters()).dtype==torch.float32
    pooling=read(directory/'1_Pooling/config.json')
    max_length=read(directory/'sentence_bert_config.json')['max_seq_length']
    prefix='query: ' if MODELS[name]=='e5' else ''
    destination=ROOT/'outputs'/name
    destination.mkdir(parents=True,exist_ok=True)
    save(destination/'identity.json',{**expected,'device':'cuda','dtype':'float32','max_length':max_length,'batch_size':64,'api_calls':0})
    for dataset in ['banking77','abcd','dogo']:
        payload=read(ROOT/'inputs'/(dataset+'.json'))
        texts=list(dict.fromkeys([r['text'] for r in payload['validation']]+payload['keys']+[r['text'] for r in payload['test']]))
        chunks=[]
        for start in range(0,len(texts),64):
            batch=tokenizer([prefix+t for t in texts[start:start+64]],padding=True,truncation=True,max_length=max_length,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                hidden=model(**batch).last_hidden_state
                if pooling.get('pooling_mode_cls_token'):
                    vectors=hidden[:,0]
                else:
                    mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    vectors=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
                vectors=torch.nn.functional.normalize(vectors,p=2,dim=-1)
            chunks.append(vectors.cpu().numpy())
        vectors=np.concatenate(chunks).astype(np.float32)
        assert np.isfinite(vectors).all()
        norms=np.linalg.norm(vectors,axis=1)
        assert np.max(abs(norms-1))<1e-5,(name,dataset,float(norms.min()),float(norms.max()),[(i,texts[i]) for i in np.where(abs(norms-1)>=1e-5)[0][:3]])
        lookup={t:v for t,v in zip(texts,vectors)}
        validation=np.stack([lookup[r['text']] for r in payload['validation']])
        test=np.stack([lookup[r['text']] for r in payload['test']])
        keys=np.stack([lookup[t] for t in payload['keys']])
        pair_matrix=validation@validation.T
        pairs=pair_matrix[np.triu_indices(len(validation),1)]
        similarities=test@keys.T
        np.savez_compressed(destination/(dataset+'.npz'),validation_vectors=validation,key_vectors=keys,test_vectors=test,background_scores=pairs,test_similarities=similarities)
        print(name,dataset,'complete',len(pairs),'background pairs','fixed085',int((similarities>=.85).any(1).sum()),flush=True)

def summarize():
    records=[]
    for dataset in ['banking77','abcd','dogo']:
        bge=np.load(ROOT/'outputs/bge-small-en-v1.5'/(dataset+'.npz'))
        target=int((bge['background_scores']>=.85).sum())
        total=len(bge['background_scores'])
        payload=read(ROOT/'inputs'/(dataset+'.json'))
        for name in MODELS:
            array=np.load(ROOT/'outputs'/name/(dataset+'.npz'))
            scores=array['background_scores']
            if MODELS[name]=='bge':
                threshold=.85
            else:
                values,counts=np.unique(scores,return_counts=True)
                possible=np.r_[0,np.cumsum(counts[::-1])]
                # Counts increase monotonically, so argmin ties choose fewer matches.
                index=int(np.argmin(np.abs(possible-target)))
                threshold=float(np.nextafter(float(values[-1]),np.inf)) if index==0 else float(values[::-1][index-1])
            count=int((scores>=threshold).sum())
            similarities=array['test_similarities']
            fixed=int((similarities>=.85).any(1).sum())
            calibrated=int((similarities>=threshold).any(1).sum())
            records.append({'dataset':dataset,'encoder':name,'threshold':threshold,'bge_reference_threshold':.85,'validation_pairs':total,'reference_background_matches':target,'calibrated_background_matches':count,'background_count_error':count-target,'reference_background_rate_pct':100*target/total,'calibrated_background_rate_pct':100*count/total,'k':len(payload['keys']),'test_total':len(payload['test']),'covered_queries':calibrated,'coverage_pct':100*calibrated/len(payload['test']),'fixed085_hits':fixed})
        assert records[-4]['fixed085_hits']==payload['expected_fixed085_hits'],(dataset,records[-4])
    output=ROOT/'outputs'
    with (output/'calibrated_coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save(output/'summary.json',{'complete':True,'api_calls':0,'records':records,'calibration_uses_test':False,'reoptimized':False})
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model');args=parser.parse_args()
    if args.model:
        worker(args.model)
    else:
        (ROOT/'outputs').mkdir(exist_ok=True)
        processes=[]
        for name in MODELS:
            log=(ROOT/'outputs'/(name+'.log')).open('w')
            processes.append((name,subprocess.Popen([sys.executable,str(Path(__file__)),'--model',name],stdout=log,stderr=subprocess.STDOUT)))
        failures=[name for name,process in processes if process.wait()!=0]
        assert not failures,failures
        summarize()
