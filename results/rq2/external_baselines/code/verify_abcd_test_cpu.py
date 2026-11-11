"""Independent Test-only re-encoding of already frozen ABCD portfolios."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='8'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
sys.path.insert(0,str(ROOT/'code_existing_config'))
from hybrid_search import BGE,texts_from_jsonl,sha256,save_json

def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    data=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data/abcd/desensitized')
    model=Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5')
    manifest=json.loads((data/'manifest.json').read_text())
    assert sha256(data/'test.jsonl')==manifest['files']['test.jsonl']['sha256']
    records=[json.loads(l) for l in (data/'test.jsonl').read_text().splitlines() if l.strip()]
    assert len(records)==manifest['counts']['test']==476
    assert all(r['split']=='test' and r['representation']=='delexed' for r in records)
    sets=[];sources={}
    p=ROOT/'abcd_main_frozen_texts.json';a=json.loads(p.read_text())
    assert len(a['texts'])==5
    sets.append(('RKGB_main',a['texts'],a['source_test']['covered']))
    sources[str(p)]=sha256(p)
    for obj in ['mean_cosine','soft_threshold']:
        base=ROOT/'runs/gaslite'/obj/'abcd'
        p=base/'train_report.json';frozen=json.loads(p.read_text())
        assert frozen['heldout_loaded'] is False
        final=json.loads((base/'final_report.json').read_text())
        assert frozen['raw_selected']==final['raw_selected'] and frozen['screened_selected']==final['screened_selected']
        assert len(frozen['raw_selected'])==5
        for variant,key in [('raw','raw_selected'),('screened','screened_selected')]:
            sets.append(('GASLITE_'+obj+'_'+variant,[r['text'] for r in frozen[key]],final['test'][variant]['covered']))
        sources[str(p)]=sha256(p)
        sources[str(base/'final_report.json')]=sha256(base/'final_report.json')
    out=ROOT/'independent_test_cpu';out.mkdir(exist_ok=True)
    # Save the imported frozen texts before any new Test embedding is computed.
    selection={'purpose':'Independent verification, no optimization, no Train reads or Test-based portfolio choice.',
        'sources':sources,'selected':{name:texts for name,texts,_ in sets}}
    save_json(out/'frozen_inputs.json',selection)
    begin=time.time();enc=BGE(model,'cpu','bge')
    test=enc.encode([r['text'] for r in records],32)
    metrics=[];query_rows=[]
    for name,texts,expected in sets:
        if texts:
            vectors=enc.encode(texts,32);scores=test @ vectors.T;best=scores.max(1);hit=best>=.85
            margin=float(np.min(abs(best-.85)))
        else:best=np.full(len(test),float('-inf'));hit=np.zeros(len(test),dtype=bool);margin=None
        count=int(hit.sum())
        metrics.append({'method':name,'split':'test','key_count':len(texts),'covered':count,'total':len(test),
            'coverage_pct':100*count/len(test),'source_report_covered':expected,'exact_count_reproduced':count==expected,
            'delta_vs_main_pp':100*(count-308)/len(test),'minimum_distance_from_threshold':margin})
        for i,(record,score,matched) in enumerate(zip(records,best,hit)):
            query_rows.append({'method':name,'test_index':i,'source_id':record['source_id'],
                               'max_cosine':float(score) if np.isfinite(score) else '', 'covered':int(matched)})
        print(name,count,'/',len(test),'keys',len(texts),'exact',count==expected,flush=True)
    for filename,rows in [('test_comparison.csv',metrics),('query_level_test.csv',query_rows)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(out/'verification.json',{'split':'test','dataset':'abcd','threshold':.85,'encoder':'bge-small-en-v1.5',
        'gpu_used':False,'no_search':True,'no_quality_rescreening':True,'quality_screen_scope':'Preserves each original saved screened portfolio, including zero keys. Does not adjudicate fairness or validate quality scorer.',
        'test_sha256':sha256(data/'test.jsonl'),'encoder_sha256':sha256(model/'model.safetensors'),
        'code_sha256':sha256(Path(__file__)),'seconds':time.time()-begin,'metrics':metrics,
        'all_counts_reproduced':all(r['exact_count_reproduced'] for r in metrics)})
    if not all(r['exact_count_reproduced'] for r in metrics):raise SystemExit('Count mismatch; inspect before claiming reproduction.')

if __name__=='__main__':main()
