from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np

ROOT=Path(r'C:\Users\ykx01\Desktop\results')
BASE=ROOT/'ykx现行实验/v10_current'
SOURCE=BASE/'multidogo/results/runs/multidogo_riemann_k10_t0p85_s42_dataset_scope_v1'
OUT=ROOT/'MultiDoGO_8309_关闭联合搜索_20261104'
INP=OUT/'input'
INP.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['targets.jsonl','target_vectors.npy','train.npy','train.jsonl']:
    shutil.copy2(SOURCE/'01_targets'/name,INP/name)
shutil.copy2(SOURCE/'config_snapshot.json',INP/'source_config.json')
shutil.copy2(SOURCE/'02_residual_search/language_gate.json',INP/'source_language_gate.json')
shutil.copy2(SOURCE/'04_final_test/report.json',INP/'source_main_report.json')
shutil.copy2(SOURCE/'03_portfolio_evaluation/portfolios.json',INP/'source_portfolios.json')
route=SOURCE/'02_residual_search/original__gradient_residual_retention'
shutil.copy2(route/'prefix_portfolio.json',INP/'source_prefix.json')
targets=[json.loads(l) for l in (INP/'targets.jsonl').read_text(encoding='utf-8').splitlines()]
cache={}
round_stats=[]
allrows=[]
vectors=[]
for folder in sorted(route.glob('round_*')):
    rr=[json.loads(l) for l in (folder/'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    vv=np.load(folder/'vectors.npy')
    assert len(rr)==len(vv)
    allrows.extend(rr);vectors.append(vv)
    seed_rows=[r for r in rr if r['origin']=='seed' and r['iteration']==0]
    stats=json.loads((folder/'stats.json').read_text(encoding='utf-8'))
    assert len(seed_rows)==stats['initial_seed_count']
    round_stats.append({'folder':folder.name,'accepted_seed_rows':len(seed_rows),'source_stats_sha256':sha(folder/'stats.json')})
    for row in seed_rows:
        key=' '.join(row['text'].split()).casefold()
        if key in cache:assert cache[key]['ppl']==row['ppl']
        cache[key]=row
for target in targets:
    for seed in target['seeds']:
        key=' '.join(seed['text'].split()).casefold()
        if key not in cache:
            cache[key]={'text':seed['text'],'train_index':seed['train_index'],'language_feasible':False,
                'cached_decision_source':'Not admitted as an initial natural seed in the recorded source run; raw PPL is unavailable.'}
assert len(cache)==195
assert sum(bool(r['language_feasible']) for r in cache.values())==184
(INP/'frozen_seed_eligibility.json').write_text(json.dumps({'decisions':cache,'source_round_stats':round_stats,
    'scope':'Replay historical seed-eligibility decisions. No new language or PPL scoring; no generated candidates admitted.'},ensure_ascii=False,indent=2),encoding='utf-8')
(INP/'source_candidates.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in allrows)+'\n',encoding='utf-8')
np.save(INP/'source_candidate_vectors.npy',np.concatenate(vectors))
for name in ['__init__.py','common.py','encoder.py','objectives.py','portfolio.py','gradient_search.py']:
    target=OUT/'archived_code/src'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(BASE/'src'/name,target)
manifest={'dataset':'dogo','source_run':str(SOURCE),'source_main_covered':909,'test_total':1094,'k':10,'threshold':.85,
    'protocol':'Disable every text-search iteration on the original ten geometric guides and their seed lists; replay frozen seed eligibility; retain original Train global greedy selection and 12 one-swap passes. Freeze before held-out access.',
    'scope':'Independent rerun of a frozen-input zero-search counterfactual, not a fresh generation or a fresh PPL audit.',
    'heldout_hash':'1f9bd20a2dff467e2f6ea9d6a7f86d604009a9db705643f1256b46dcbc3c18fc',
    'expected_test_vector_sha256':sha(ROOT/'RQ2_Riemann-nearest_20261022/multidogo/test_vectors.npy'),
    'input_sha256':{p.name:sha(p) for p in INP.iterdir() if p.is_file()},
    'code_sha256':{p.name:sha(p) for p in (OUT/'archived_code/src').iterdir() if p.is_file()}}
(INP/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(OUT/'zero_search_input.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for folder in [INP,OUT/'archived_code']:
        for path in folder.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(OUT).as_posix())
print(json.dumps({'output':str(OUT),'unique_initial_seeds':len(cache),'accepted_initial_seeds':184,
    'original_search_iterations':json.loads((INP/'source_config.json').read_text())['search']['iterations'],
    'archive_bytes':(OUT/'zero_search_input.zip').stat().st_size},ensure_ascii=False))
