from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import ast,json,re,sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
root=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031';out=root/'evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
summary=read(root/'summary.json');assert summary['complete'] and summary['api_calls']==0
plan=read(root/'experiment_plan.json');raw=read(out/'raw_summary.json');quality=read(root/'quality_results.json');frozen=read(out/'frozen_portfolios.json')
assert read(root/'adapter_local_validation.json')['passed']
pattern=re.compile(r'<[A-Z][A-Z0-9_]*>');cluster_count=0
official=(root/'official_source/src/attack_poison.py').read_text(encoding='utf-8');core=(root/'code/official_hotflip_core.py').read_text(encoding='utf-8')
def func(text):return ast.dump(next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='hotflip_attack'),include_attributes=False)
assert func(official)==func(core)
for dataset,k in plan['datasets'].items():
    seeds=read(root/'inputs'/dataset/'seed_selection.json')['selected'];tr=np.load(root/'inputs'/dataset/'train_vectors.npy');labels=np.load(root/'inputs'/dataset/'cluster_labels.npy')
    textorder=read(out/dataset/'key_texts.json');vectors=np.load(out/dataset/'key_vectors.npz')['vectors'];lookup={t:v for t,v in zip(textorder,vectors)}
    assert np.max(abs(np.linalg.norm(vectors,axis=1)-1))<1e-5
    matrices=np.load(out/dataset/'test_similarities.npz')
    for j in range(k):
        final=read(root/'runs'/dataset/f'cluster_{j:02d}/final_cluster.json');cluster_count+=1
        assert final['completed_iterations']==5000 and final['heldout_loaded'] is False and final['api_calls']==0
        assert final['initial']['text']==seeds[j]['text']
        assert Counter(pattern.findall(final['selected']['text']))==Counter(pattern.findall(seeds[j]['text']))
        mean=tr[labels==j].mean(axis=0)
        exact=float(lookup[final['selected']['text']]@mean)
        assert abs(exact-final['selected']['full_cluster_mean_cosine'])<2e-4
    for budget in plan['checkpoint_budgets']:
        rr=frozen[dataset]['portfolios'][str(budget)]
        for j,x in enumerate(rr):
            actual=read(root/'runs'/dataset/f'cluster_{j:02d}'/f'budget_{budget:05d}.json')
            assert x['text']==actual['selected']['text'] and actual['selected']['selected_at_iteration']<=budget
        matrix=matrices['budget_'+str(budget)]
        for metric in [x for x in raw['all_metrics'] if x['dataset']==dataset and x['iteration_budget_per_cluster']==budget]:
            hit=(matrix>=metric['threshold']).any(axis=1)
            assert int(hit.sum())==metric['test_covered'] and abs(float(hit.mean())-metric['test_coverage'])<1e-12
    for metric in [x for x in summary['metrics'] if x['dataset']==dataset]:
        method=metric['method'];rr=sorted([x for x in quality['results'] if x['dataset']==dataset and x['method']==method],key=lambda x:x['slot'])
        for x in rr:assert x['quality_pass']==(x['ppl']<=150 and x['local_readability_pass'])
        kept=np.array([x['quality_pass'] for x in rr],bool)
        if method=='CorpusPoisoning_adapted':matrix=matrices['budget_5000']
        else:
            p=Path.home()/'Desktop/results/固定文本_原文脱敏配对_20261031';p=p/'ABCD补充/abcd' if dataset=='abcd' else p/dataset
            matrix=np.load(p/'similarities.npz')['delex']
        mask=(matrix[:,kept]>=.85).any(axis=1)
        assert len(rr)==k and int(kept.sum())==metric['local_quality_passed_texts']
        assert int(mask.sum())==metric['quality_filtered_test_covered']
assert cluster_count==30
(root/'artifact_validation.json').write_text(json.dumps({'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'clusters':30,
    'api_calls':0,'checks':['exact official gradient core AST','initial Train selections','placeholder preservation',
    'full-cluster checkpoint score reproduction','preset-budget snapshot identities','all threshold coverage counts',
    'common local quality gate and filtered coverage'],'scope':'Adapted algorithm, single seed42, no cross-task or answer-reuse evaluation.'},indent=2)+'\n',encoding='utf-8')
print('CORPUS_HOTFLIP_ARTIFACT_VALIDATION_PASSED',flush=True)
