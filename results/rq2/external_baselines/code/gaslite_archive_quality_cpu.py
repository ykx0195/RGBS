"""Post-hoc, predeclared archive quality selection; original results unchanged.

Per slot: seed, saved Train-selected raw keys, top 64 objective candidates,
uniform sample 256 candidates (seed 20261102). No Test-based selection/tuning.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='8'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import hashlib,json,random,sys,time
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
sys.path.insert(0,str(ROOT/'code_existing_config'))
from hybrid_search import Fluency,BGE,valid_text,texts_from_jsonl,sha256,save_json
from run_configurable import word_change_count,typed_placeholder_context_ok,surface_fluency_ok,AREAS,APIRewriter,load_provider_specs
from portfolio import greedy_portfolio,swap_portfolio,union_mask

def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    outroot=ROOT/'gaslite_archive_quality_cpu';outroot.mkdir(exist_ok=True)
    protocol={'post_hoc':True,'motivation':'Original top-coverage shortlist excluded quality-passing seeds and other archived candidates.',
      'candidate_budget_per_slot':{'top_train_objective':64,'uniform_archive':256,'include_seed':True,'include_raw_train_selected':True},
      'sampling_seed':20261102,'quality_thresholds_unchanged':True,'test_based_tuning':False,
      'gpu_used':False,'new_generation_calls':0,'key_budget':5,'threshold':.85,
      'scope':'Additional sampled-archive adaptation, not archive-wide screening or original GASLITE result. Original frozen reports retained.',
      'precision':'CPU float32 models; local shared PPL/readability followed by both existing API domain judges.'}
    save_json(outroot/'protocol.json',protocol)
    data=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data/abcd/desensitized')
    manifest=json.loads((data/'manifest.json').read_text())
    assert sha256(data/'train.jsonl')==manifest['files']['train.jsonl']['sha256']
    flu=Fluency(Path('/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct'),'cpu')
    enc=BGE(Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5'),'cpu','bge')
    train=enc.encode(texts_from_jsonl(data/'train.jsonl'),32)
    api=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json')
    for objective in ['mean_cosine','soft_threshold']:
        out=outroot/objective;out.mkdir(exist_ok=True)
        if (out/'final_report.json').exists():continue
        base=ROOT/'runs/gaslite'/objective/'abcd'
        frozen=json.loads((base/'train_report.json').read_text());assert frozen['heldout_loaded'] is False
        rawtexts={r['text'] for r in frozen['raw_selected']};plans=[]
        rng=random.Random(20261102)
        for i in range(1,6):
            p=base/f'slot_{i:02d}.json';a=json.loads(p.read_text())
            chosen=[a['seed']]+[r['text'] for r in a['archive'] if r['text'] in rawtexts]
            chosen += [r['text'] for r in sorted(a['archive'],key=lambda r:r['objective'],reverse=True)[:64]]
            chosen += [r['text'] for r in rng.sample(a['archive'],min(256,len(a['archive'])))]
            plans.append({'slot':i,'source_sha256':sha256(p),'seed':a['seed'],'candidates':list(dict.fromkeys(chosen))})
        save_json(out/'frozen_candidate_plan.json',{'protocol':protocol,'objective':objective,'slots':plans})
        providers=[APIRewriter(s,api.parent,out/'api_cache') for s in load_provider_specs(api)]
        pool={};total=0;begin=time.time()
        for plan in plans:
            checkpoint=out/f"slot_{plan['slot']:02d}_screen.json"
            if checkpoint.exists():screen=json.loads(checkpoint.read_text())
            else:
                seed=plan['seed'];seedppl=flu.ppl(seed);gate=min(150,1.5*seedppl);eligible=[];diagnostics=[]
                for text in plan['candidates']:
                    structural=valid_text(text,seed) and typed_placeholder_context_ok(text) and surface_fluency_ok(text) and word_change_count(seed,text)<=10
                    ppl=flu.ppl(text) if structural else None
                    readable=flu.judge_readability([text],AREAS['abcd'])[0] if structural and ppl<=gate else False
                    diagnostics.append({'text':text,'structural':structural,'ppl':ppl,'readable':readable})
                    if readable:eligible.append({'text':text,'ppl':ppl})
                    total+=1
                    if total%20==0:
                        save_json(outroot/'status.json',{'phase':'screening','objective':objective,'slot':plan['slot'],
                            'scored_rows_current_objective':total,'elapsed_seconds':time.time()-begin})
                verdicts=[p.judge([r['text'] for r in eligible],'abcd') for p in providers]
                accepted=[r for j,r in enumerate(eligible) if all(v[j] for v in verdicts)]
                screen={'slot':plan['slot'],'seed_ppl':seedppl,'ppl_gate':gate,'candidate_count':len(plan['candidates']),
                    'local_eligible':len(eligible),'accepted':accepted,'diagnostics':diagnostics}
                save_json(checkpoint,screen)
            for row in screen['accepted']:pool.setdefault(row['text'],row)
        rows=list(pool.values())
        if rows:
            vec=enc.encode([r['text'] for r in rows],32)
            for row,v in zip(rows,vec):row.update(vector=v,mask=train @ v>=.85,target_cosine=0.0)
        selected=swap_portfolio(greedy_portfolio(rows,5),rows,4) if len(rows)>=5 else rows
        report={'protocol':protocol,'objective':objective,'heldout_loaded':False,'candidate_plan_sha256':sha256(out/'frozen_candidate_plan.json'),
            'qualified_pool_size':len(rows),'selected':[{'text':r['text'],'ppl':r['ppl']} for r in selected],
            'train_covered':int(union_mask([r['mask'] for r in selected]).sum()) if selected else 0,
            'train_total':len(train),'effective_key_count':len(selected)}
        save_json(out/'train_report.json',report)
        # First Test access for this separately labelled frozen portfolio.
        assert sha256(data/'test.jsonl')==manifest['files']['test.jsonl']['sha256']
        test=enc.encode(texts_from_jsonl(data/'test.jsonl'),32)
        count=int(((test @ enc.encode([r['text'] for r in selected]).T).max(1)>=.85).sum()) if selected else 0
        save_json(out/'final_report.json',{**report,'heldout_loaded':True,'test':{'covered':count,'total':len(test),'coverage':count/len(test)},
                                           'test_sha256':sha256(data/'test.jsonl')})
        print(objective,'qualified',len(rows),'Test',count,'/',len(test),flush=True)
    save_json(outroot/'status.json',{'phase':'complete','gpu_used':False,'original_portfolios_changed':False})

if __name__=='__main__':main()
