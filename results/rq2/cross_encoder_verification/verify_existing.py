"""Read-only verification of existing transfer evidence for the ORIGINAL main portfolios."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
BASE=Path(r'C:\Users\ykx01\Desktop\results')
TRANSFER=BASE/'跨编码器迁移_本地CPU_仅总覆盖_20261031'
DOGO=BASE/'MultiDoGO_8309_跨编码器_CPU_20261103'
MAIN=WORK/'gptcache_experiment'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
artifact=read(TRANSFER/'artifact_sha256.json')['files']
sourcehashes={};result=[]
def track(p):sourcehashes[str(p)]=sha(p);return p
def checked(p):
    rel=p.relative_to(TRANSFER).as_posix()
    assert sha(p)==artifact[rel]['sha256'],rel
    return track(p)
expected={'banking77':{'bge-small-en-v1.5':1109,'all-MiniLM-L6-v2':71,'gte-small':1467,'e5-small-v2':1481},
 'abcd':{'bge-small-en-v1.5':308,'all-MiniLM-L6-v2':11,'gte-small':463,'e5-small-v2':470},
 'dogo':{'bge-small-en-v1.5':909,'all-MiniLM-L6-v2':228,'gte-small':1084,'e5-small-v2':1083}}
models=read(checked(TRANSFER/'model_manifest.json'));modelhash={r['model']:r['model_sha256'] for r in models}
for ds,n,k in [('banking77',1485,15),('abcd',476,5),('dogo',1094,10)]:
    mainkeys=read(track(MAIN/'inputs'/ds/'constructed.json'));texts=[r['text'] for r in mainkeys]
    assert len(texts)==k
    if ds!='dogo':
        frozen=read(checked(TRANSFER/'frozen_inputs'/ds/'frozen_keys.json'))
        assert frozen['key_texts']==texts
        assert sha(MAIN/'inputs'/ds/'test.jsonl')==frozen['test_sha256']
        assert sha(checked(TRANSFER/'frozen_inputs'/ds/'test_queries.jsonl'))==frozen['test_sha256']
    else:
        frozen=read(track(DOGO/'frozen_inputs.json'));assert frozen['frozen_keys']==texts
        assert sha(MAIN/'inputs'/ds/'test.jsonl')==frozen['test_sha256']
        assert read(track(DOGO/'final_report.json'))['protocol']['frozen_keys']==texts
    for model in ['bge-small-en-v1.5','all-MiniLM-L6-v2','gte-small','e5-small-v2']:
        if model=='bge-small-en-v1.5':
            p=track(MAIN/'outputs'/ds/'constructed_only/query_results.json');rr=read(p)
            count=sum(r['cache_hit'] for r in rr);assert len(rr)==n
            assert [r['selected_text'] for r in rr if r['cache_hit'] and r['selected_text'] not in texts]==[]
        else:
            if ds!='dogo':
                p=checked(TRANSFER/model/ds/'test_similarities.npz')
                identity=read(checked(TRANSFER/model/ds/'embedding_identity.json'))
            else:
                p=track(DOGO/model/'similarities.npz');identity=read(track(DOGO/model/'identity.json'))
            assert identity['model_sha256']==modelhash[model]
            mat=np.load(p)['similarities'];assert mat.shape==(n,k)
            count=int((mat>=.85).any(axis=1).sum())
        assert count==expected[ds][model],(ds,model,count)
        result.append({'dataset':ds,'target_encoder':model,'source_encoder':'bge-small-en-v1.5','threshold':.85,
          'k':k,'covered_queries':count,'test_total':n,'coverage_percent':100*count/n,
          'same_main_strings':True,'target_specific_reselection':False,'target_specific_search':False,
          'source':str(p),'source_sha256':sha(p)})
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
save(ROOT/'verified_results.json',result)
with (ROOT/'verified_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
save(ROOT/'verification.json',{'passed':True,'configurations':len(result),'source_files_sha256':sourcehashes,
 'datasets_use_current_original_main_portfolios':True,'strict_pairwise_admission_portfolios_not_used':True,
 'dogo_older_transfer_portfolio_excluded':True,'fixed_numerical_threshold':.85,
 'new_encoding':False,'new_api_calls':0,'new_search':False})
print(json.dumps([{'dataset':ds,'coverage':{r['target_encoder']:round(r['coverage_percent'],2) for r in result if r['dataset']==ds}} for ds in expected]))
