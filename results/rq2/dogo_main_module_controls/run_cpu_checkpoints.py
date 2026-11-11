"""Main-83.09% fixed-archive controls. CPU only; no generation or API calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='8'
from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();out=a.root
    sys.path.insert(0,str(out/'archived_code'))
    from src.portfolio import greedy_initialize,optimize_one_swap,portfolio_metrics
    from src.common import normalized_text
    sys.path.insert(0,str(ROOT/'code_existing_config'))
    from hybrid_search import BGE,sha256,save_json
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    inp=out/'input';m=json.loads((inp/'manifest.json').read_text())
    assert all(sha256(inp/n)==h for n,h in m['input_hashes'].items())
    begin=time.time();train=np.load(inp/'train_vectors.npy');v=np.load(inp/'candidate_vectors.npy')
    rows=[json.loads(l) for l in (inp/'candidates.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    assert train.shape==(3993,384) and v.shape==(len(rows),384)
    pool=json.loads((inp/'portfolios.json').read_text())[0]
    prefix=json.loads((inp/'prefix_portfolio.json').read_text())
    assert pool['selected_on']=='train' and not pool['validation_used_for_selection']
    fulltexts=[r['text'] for r in pool['selected']];prefixtexts=[r['text'] for r in prefix['selected']]
    assert len(fulltexts)==len(prefixtexts)==10
    indices=[i for i,r in enumerate(rows) if r['origin']=='seed' and r['iteration']==0 and r['language_feasible']]
    seeds=[rows[i] for i in indices];seedv=v[indices]
    print('Train-only natural-seed pool:',len(seeds),'rows',flush=True)
    init=greedy_initialize(seeds,seedv,train,.85,10)
    selected,history=optimize_one_swap(seeds,seedv,train,.85,init,m['original_portfolio_config']['swap_passes'])
    seedtexts=[seeds[i]['text'] for i in selected]
    assert len({normalized_text(t) for t in seedtexts})==10
    variants={'full_main':fulltexts,'without_final_global_reselection':prefixtexts,
              'without_generated_candidates_seed_pool':seedtexts}
    # Freeze every text set before loading Test or computing Test embeddings.
    freeze={'dataset':'dogo','threshold':.85,'k':10,'selected_on':'train','heldout_loaded':False,
       'post_hoc':True,'gpu_used':False,'new_generation_calls':0,'new_judge_calls':0,
       'source_manifest_sha256':sha256(inp/'manifest.json'),
       'source_main_portfolio_sha256':sha256(inp/'portfolios.json'),
       'protocol':{'without_final_global_reselection':m['no_reselection_control'],
                   'without_generated_candidates_seed_pool':m['seed_control']},
       'seed_pool_rows':len(seeds),'seed_pool_distinct_texts':len({normalized_text(s['text']) for s in seeds}),
       'seed_control_initial_indices':init,'seed_control_selected_indices':selected,'seed_control_swap_history':history,
       'original_swap_passes':m['original_portfolio_config']['swap_passes'],
       'variants':variants}
    save_json(out/'train_frozen_selections.json',freeze)
    data=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data/multidogo/desensitized')
    testpath=data/'test.jsonl'
    assert sha256(testpath)==m['test_sha256']
    testrows=[json.loads(l) for l in testpath.read_text(encoding='utf8').splitlines() if l.strip()]
    assert len(testrows)==1094 and all(r['split']=='test' and r['representation']=='delexed' for r in testrows)
    enc=BGE(Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5'),'cpu','bge')
    test=enc.encode([r['text'] for r in testrows],32)
    trainrows=[json.loads(l) for l in (inp/'train.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    probe=np.linspace(0,len(train)-1,5,dtype=int)
    assert np.allclose(enc.encode([trainrows[i]['text'] for i in probe],5),train[probe],atol=2e-4,rtol=2e-4)
    metrics=[];query=[];matches=[]
    for name,texts in variants.items():
        vectors=enc.encode(texts,32);trainmetrics=portfolio_metrics(train,vectors,.85)
        score=test @ vectors.T;hit=score>=.85;covered=hit.any(1)
        row={'dataset':'dogo','variant':name,'k':len(texts),'threshold':.85,
             'train_covered':trainmetrics['covered'],'train_total':len(train),'test_covered':int(covered.sum()),
             'test_total':len(test),'test_coverage_pct':100*float(covered.mean()),
             'delta_vs_main_pp':100*(int(covered.sum())-909)/len(test)}
        metrics.append(row)
        for i,record in enumerate(testrows):
            query.append({'variant':name,'test_index':i,'source_id':record['source_id'],
                          'covered':int(covered[i]),'max_cosine':float(score[i].max())})
            for key in np.flatnonzero(hit[i]):
                matches.append({'variant':name,'test_index':i,'source_id':record['source_id'],'key_index':int(key),
                    'cosine':float(score[i,key]),'key_text':texts[key],'query_text':record['text']})
        print(name,row['test_covered'],'/',len(test),f"{row['test_coverage_pct']:.2f}%",flush=True)
    assert metrics[0]['test_covered']==909,'Main selected strings did not reproduce 83.09%'
    for name,rr in [('coverage_comparison.csv',metrics),('query_level.csv',query),('matched_pairs.csv',matches)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    save_json(out/'final_report.json',{'freeze_sha256':sha256(out/'train_frozen_selections.json'),
       'heldout_loaded':True,'source_main_reproduced':True,'gpu_used':False,'new_api_calls':0,
       'test_sha256':sha256(testpath),'encoder_sha256':sha256(encdir/'model.safetensors') if False else sha256(Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5/model.safetensors')),
       'code_sha256':sha256(Path(__file__)),'seconds':time.time()-begin,'metrics':metrics,
       'interpretation':'Post-hoc frozen-archive controls. Natural-seed candidate-pool restriction keeps Train selection and final reselection, not a recorded K-text initialization or a full fresh zero-rewrite rerun. Before-reselection texts are the archived sequential prefix.'})
    save_json(out/'status.json',{'phase':'complete','main_reproduced':True,'gpu_used':False,'seconds':time.time()-begin})
if __name__=='__main__':main()
