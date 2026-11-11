from pathlib import Path
import csv,hashlib,json,shutil,tarfile
import numpy as np
BASE=Path(r'C:\Users\ykx01\Desktop')
SOURCE=BASE/'results/ykx现行实验/v10_current'
RUN=SOURCE/'multidogo/results/runs/multidogo_riemann_k10_t0p85_s42_dataset_scope_v1'
OUT=BASE/'results/MultiDoGO_8309_模块与重选_CPU_20261103'
INPUT=OUT/'input';INPUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
metadata={'source_run':str(RUN),'expected_main_covered':909,'expected_test_total':1094,
 'threshold':.85,'k':10,'post_hoc':True,'selection_split':'train','gpu_used':False,
 'no_generation':True,'no_new_model_judgments':True,
 'seed_control':'Restrict the frozen full candidate archive to origin=seed, iteration=0, language_feasible. Apply the archived Train greedy initializer and one-swap selector with the original 12-pass limit. This is a fixed-archive natural-seed-pool control, not a recorded K-text initialization or a from-scratch counterfactual search.',
 'no_reselection_control':'The saved sequential prefix_portfolio of ten texts, before global greedy initialization and swaps.',
 'sources':{}}
route=RUN/'02_residual_search/original__gradient_residual_retention'
allrows=[];blocks=[];offset=0
for folder in sorted(route.glob('round_*')):
    rows=read(folder/'candidates.jsonl');v=np.load(folder/'vectors.npy')
    assert len(rows)==len(v)
    for i,row in enumerate(rows):allrows.append({**row,'round_folder':folder.name,'local_candidate_index':i,'pool_index':offset+i})
    blocks.append(v);offset+=len(rows)
    for n in ['candidates.jsonl','vectors.npy','selected.json','stats.json']:metadata['sources'][str(folder/n)]=sha(folder/n)
np.save(INPUT/'candidate_vectors.npy',np.concatenate(blocks))
(INPUT/'candidates.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in allrows)+'\n',encoding='utf8')
for source,name in [(RUN/'01_targets/train.npy','train_vectors.npy'),(RUN/'01_targets/train.jsonl','train.jsonl'),
                    (route/'prefix_portfolio.json','prefix_portfolio.json'),
                    (RUN/'03_portfolio_evaluation/portfolios.json','portfolios.json'),
                    (RUN/'03_portfolio_evaluation/report.json','train_validation_report.json'),
                    (RUN/'04_final_test/report.json','source_main_test_report.json')]:
    shutil.copy2(source,INPUT/name);metadata['sources'][str(source)]=sha(source)
cfg=json.loads((RUN/'config_snapshot.json').read_text())
metadata['original_portfolio_config']=cfg['portfolio'];metadata['source_config_sha256']=sha(RUN/'config_snapshot.json')
report=json.loads((INPUT/'source_main_test_report.json').read_text());row=report['reports'][0]
assert row['text_test']['covered']==909 and report['test_rows']==1094
portfolio=json.loads((INPUT/'portfolios.json').read_text())[0]
assert [r['text'] for r in row['selected']]==[r['text'] for r in portfolio['selected']]
assert portfolio['selected_on']=='train' and not portfolio['validation_used_for_selection']
metadata['test_sha256']=report['test_sha256'];metadata['candidate_count']=len(allrows)
metadata['natural_seed_rows']=sum(r['origin']=='seed' and r['iteration']==0 and r['language_feasible'] for r in allrows)
metadata['input_hashes']={p.name:sha(p) for p in INPUT.iterdir() if p.is_file()}
(INPUT/'manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
code=OUT/'code/src';code.mkdir(parents=True,exist_ok=True)
for n in ['__init__.py','portfolio.py','objectives.py','common.py']:
    shutil.copy2(SOURCE/'src'/n,code/n)
with tarfile.open(OUT/'dogo_checkpoint_input.tar.gz','w:gz') as t:
    t.add(INPUT,arcname='input');t.add(OUT/'code',arcname='archived_code')
shutil.copy2(Path(__file__),OUT/'prepare_input.py')
print(json.dumps({'main':909,'candidates':len(allrows),'seed_rows':metadata['natural_seed_rows'],'archive':str(OUT/'dogo_checkpoint_input.tar.gz')},ensure_ascii=True))
