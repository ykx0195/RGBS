"""Train-artifact-only diagnosis; never accesses Test or changes a portfolio.

Predeclared sample: saved raw Train-selected outputs, each starting/final text,
10 highest objective candidates per slot, and 10 uniform archive samples per
slot. Including seeds checks whether a zero screened portfolio describes only
the original coverage-ranked shortlist rather than the whole candidate archive.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='8'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import csv,hashlib,json,random,sys,time
import torch
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
sys.path.insert(0,str(ROOT/'code_existing_config'))
from hybrid_search import Fluency,valid_text,sha256,save_json
from run_configurable import word_change_count,typed_placeholder_context_ok,surface_fluency_ok,AREAS

def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    out=ROOT/'gaslite_quality_cpu';out.mkdir(exist_ok=True)
    records=[];source_hashes={};rng=random.Random(20261102)
    for objective in ['mean_cosine','soft_threshold']:
        base=ROOT/'runs/gaslite'/objective/'abcd'
        p=base/'train_report.json';frozen=json.loads(p.read_text())
        assert frozen['heldout_loaded'] is False
        source_hashes[str(p)]=sha256(p)
        selected={r['text'] for r in frozen['raw_selected']}
        for index in range(1,6):
            p=base/f'slot_{index:02d}.json';a=json.loads(p.read_text());source_hashes[str(p)]=sha256(p)
            def add(text,origin):
                records.append({'objective':objective,'slot':index,'origin':origin,'text':text,'seed':a['seed']})
            add(a['seed'],'natural_seed');add(a['final_text'],'final_optimized_text')
            for r in a['archive']:
                if r['text'] in selected:add(r['text'],'raw_train_selected')
            for r in sorted(a['archive'],key=lambda r:r['objective'],reverse=True)[:10]:add(r['text'],'top_train_objective')
            for r in rng.sample(a['archive'],min(10,len(a['archive']))):add(r['text'],'random_archive_sample')
    # Freeze diagnosis selection before loading the quality model.
    protocol={'dataset':'abcd','sampling_seed':20261102,'source_hashes':source_hashes,
        'sample_rows':len(records),'uses_train_and_archived_outputs_only':True,'test_read':False,'gpu_used':False,
        'role':'Diagnostic sample, not an archive-wide pass rate or new baseline selection.',
        'sampling':'Per objective and slot: natural seed, final text, raw Train-selected keys belonging to slot, top 10 Train-objective archive candidates, uniform random 10 archive candidates.',
        'quality_rules':'Existing shared structure, total 10 word edits, PPL<=150 and <=1.5*seed PPL; local readability only for candidates passing preceding gates. No threshold retuning.',
        'precision':'CPU float32 language model; the original search scored GPU float16. Scores may differ slightly.'}
    save_json(out/'protocol.json',protocol)
    flu=Fluency(Path('/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct'),'cpu')
    begin=time.time();rows=[]
    for i,r in enumerate(records):
        text,seed=r['text'],r['seed'];seedppl=flu.ppl(seed);ppl=flu.ppl(text);cutoff=min(150,1.5*seedppl)
        structural=valid_text(text,seed) and typed_placeholder_context_ok(text) and surface_fluency_ok(text) and word_change_count(seed,text)<=10
        ppl_ok=ppl<=cutoff
        readable=flu.judge_readability([text],AREAS['abcd'])[0] if structural and ppl_ok else None
        rows.append({**r,'seed_ppl':seedppl,'candidate_ppl':ppl,'ppl_cutoff':cutoff,'structural_pass':structural,
                     'ppl_pass':ppl_ok,'readability_pass':readable,'passes_local_quality':bool(structural and ppl_ok and readable)})
        if (i+1)%10==0:
            save_json(out/'status.json',{'phase':'running','completed_rows':i+1,'total_rows':len(records),'seconds':time.time()-begin})
            print('quality_rows',i+1,'/',len(records),flush=True)
    with (out/'candidate_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for obj in ['mean_cosine','soft_threshold']:
        for origin in sorted({r['origin'] for r in rows}):
            rr=[r for r in rows if r['objective']==obj and r['origin']==origin]
            summary.append({'objective':obj,'origin':origin,'sample_rows':len(rr),'distinct_texts':len({r['text'] for r in rr}),
                'structural_pass':sum(r['structural_pass'] for r in rr),'ppl_pass':sum(r['ppl_pass'] for r in rr),
                'local_quality_pass':sum(r['passes_local_quality'] for r in rr)})
    with (out/'diagnostic_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save_json(out/'status.json',{'phase':'complete','completed_rows':len(rows),'total_rows':len(rows),'seconds':time.time()-begin,
        'gpu_used':False,'test_read':False,'portfolios_changed':False,'summary':summary})
    print('CPU quality diagnosis complete',flush=True)

if __name__=='__main__':main()
