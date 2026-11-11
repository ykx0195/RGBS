"""Rerun zero text-search with original guides, cached seed eligibility and selector."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(text):return ' '.join(text.split()).casefold()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def jsonl(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    inp=root/'input';out=root/'output';out.mkdir(exist_ok=True)
    started=time.time();m=json.loads((inp/'manifest.json').read_text(encoding='utf-8'))
    assert all(sha(inp/n)==h for n,h in m['input_sha256'].items())
    assert all(sha(root/'archived_code/src'/n)==h for n,h in m['code_sha256'].items())
    sys.path.insert(0,str(root/'archived_code'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from src.gradient_search import search_target
    from src.objectives import ResidualCoverageObjective
    from src.portfolio import greedy_initialize,optimize_one_swap,portfolio_metrics
    cfg=json.loads((inp/'source_config.json').read_text())
    settings=dict(cfg['search']);settings['iterations']=0
    assert cfg['search']['iterations']>0 and cfg['portfolio']['swap_passes']==12
    eligibility=json.loads((inp/'frozen_seed_eligibility.json').read_text(encoding='utf-8'))['decisions']
    class Gate:
        ppl_enabled=False
        settings=cfg['language']
        def evaluate(self,texts,parents,intents,skip_readability):
            fields={'language_feasible','rule_feasible','ppl_feasible','readability_feasible','ppl','language_failures'}
            return [{k:v for k,v in eligibility[norm(t)].items() if k in fields} for t in texts]
    class Engine:
        def token_count(self,text):return eligibility[norm(text)]['token_count']
    train=np.load(inp/'train.npy');guides=np.load(inp/'target_vectors.npy');train_rows=jsonl(inp/'train.jsonl')
    targets=sorted(jsonl(inp/'targets.jsonl'),key=lambda r:(r.get('guide_round',r['vector_index']),r['target_id']))
    assert train.shape==(3993,384) and len(targets)==10
    covered=np.zeros(len(train),dtype=bool);selected_texts=[];pool=[];blocks=[];prefix=[];stats=[]
    for i,target in enumerate(targets,1):
        objective=ResidualCoverageObjective(train,guides[target['vector_index']],.85,settings,covered)
        seeds=[dict(s,vector=train[s['train_index']]) for s in target['seeds']]
        assert all(s['text']==train_rows[s['train_index']]['text'] for s in seeds)
        rows,selected,st,trace=search_target('gradient_residual_retention',objective,seeds,Engine(),settings,Gate(),train_rows,
            rewriter=None,excluded_texts=selected_texts,quality_judge=None)
        assert st['completed_iterations']==st['gradient_calls']==st['rewrite_calls']==0
        assert all(r['origin']=='seed' and r['iteration']==0 for r in rows)
        vectors=np.stack([r['vector'] for r in rows]);offset=len(pool)
        index=next(k for k,r in enumerate(rows) if r is selected)
        serial=[{k:v for k,v in r.items() if k not in {'vector','_coverage_mask'}} for r in rows]
        pool.extend(serial);blocks.append(vectors);prefix.append(offset+index)
        selected_texts.append(selected['text']);covered|=selected['_coverage_mask']
        st['seed_eligibility_replayed']=True;stats.append(st)
        print('zero-search target',i,'natural candidates',len(rows),'Train union',int(covered.sum()),flush=True)
    vectors=np.concatenate(blocks)
    initial=greedy_initialize(pool,vectors,train,.85,10)
    selected,history=optimize_one_swap(pool,vectors,train,.85,initial,12)
    assert len({norm(pool[i]['text']) for i in selected})==10
    frozen={'dataset':'dogo','threshold':.85,'k':10,'selected_on':'train','heldout_loaded':False,
        'source_main_covered':909,'original_search_iterations':cfg['search']['iterations'],'search_iterations':0,
        'gradient_calls':0,'rewrite_calls':0,'new_judge_calls':0,'new_ppl_scoring':False,'frozen_seed_eligibility_replayed':True,
        'candidate_rows':len(pool),'distinct_candidate_texts':len({norm(r['text']) for r in pool}),
        'selected':[pool[i] for i in selected],'sequential_prefix':[pool[i] for i in prefix],
        'greedy_initial_indices':initial,'selected_indices':selected,'swap_history':history,'swap_passes':12,
        'train_metrics':portfolio_metrics(train,vectors[selected],.85),'source_manifest_sha256':sha(inp/'manifest.json')}
    save(out/'train_frozen_selections.json',frozen)
    np.save(out/'frozen_selected_vectors.npy',vectors[selected])
    save(out/'zero_search_stats.json',stats)
    print('Train frozen; loading held-out vectors now',flush=True)
    # The cached Test embeddings are admitted only after text selection is frozen.
    cache=Path('/root/autodl-tmp/rq2_riemann_nearest_20261022/output/multidogo')
    assert sha(cache/'test_vectors.npy')==m['expected_test_vector_sha256']
    report=json.loads((cache/'report.json').read_text())
    assert report['input_sha256']['test.jsonl']==m['heldout_hash']
    test=np.load(cache/'test_vectors.npy');assert test.shape==(1094,384)
    original_pool=jsonl(inp/'source_candidates.jsonl');original_vectors=np.load(inp/'source_candidate_vectors.npy')
    def map_vectors(texts):
        result=[]
        for text in texts:
            ix=next(i for i,r in enumerate(original_pool) if r['text']==text)
            result.append(original_vectors[ix])
        return np.stack(result)
    full=json.loads((inp/'source_portfolios.json').read_text())[0]['selected']
    source_prefix=json.loads((inp/'source_prefix.json').read_text())['selected']
    variants={'full_main':map_vectors([r['text'] for r in full]),
        'without_joint_text_search':vectors[selected],
        'zero_search_sequential_prefix':vectors[prefix],
        'without_final_global_selection':map_vectors([r['text'] for r in source_prefix])}
    metrics=[];queries=[]
    for name,vv in variants.items():
        scores=test@vv.T;hits=scores>=.85;covered_test=hits.any(axis=1)
        record={'dataset':'dogo','variant':name,'k':len(vv),'threshold':.85,'covered':int(covered_test.sum()),
            'total':len(test),'coverage_pct':100*float(covered_test.mean()),'delta_vs_main_pp':100*(int(covered_test.sum())-909)/len(test)}
        metrics.append(record)
        for i in range(len(test)):
            queries.append({'variant':name,'test_index':i,'covered':int(covered_test[i]),'max_cosine':float(scores[i].max())})
        print(name,record['covered'],record['total'],record['coverage_pct'],flush=True)
    assert metrics[0]['covered']==909
    assert metrics[-1]['covered']==846
    # Independent implementation from threshold bitsets verifies query membership.
    selected_vectors=vectors[selected]
    loop_hits=np.zeros(len(test),dtype=bool)
    for vector in selected_vectors:loop_hits|=(test@vector>=.85)
    matrix_hits=(test@selected_vectors.T>=.85).any(axis=1)
    assert np.array_equal(loop_hits,matrix_hits)
    for filename,rows in [('coverage_comparison.csv',metrics),('query_level.csv',queries)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    final={'scope':m['scope'],'protocol':m['protocol'],'metrics':metrics,'freeze_sha256':sha(out/'train_frozen_selections.json'),
        'source_main_reproduced':True,'source_no_selection_reproduced':True,'test_vector_hash_verified':True,
        'source_code_hashes_verified':True,'independent_query_membership_verified':True,
        'heldout_loaded_after_freezing':True,'gpu_used':False,'new_api_calls':0,'new_ppl_scoring':False,
        'elapsed_seconds':time.time()-started,'code_sha256':sha(Path(__file__))}
    save(out/'final_report.json',final)
    save(out/'artifact_sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_sha256.json'})
    save(out/'status.json',{'phase':'complete','elapsed_seconds':time.time()-started})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
