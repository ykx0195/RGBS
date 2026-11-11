"""Re-use the preceding real-service implementation with new frozen text sets."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib,importlib.metadata,json,logging,shutil,sys,time
from pathlib import Path
import numpy as np
OLD=Path('/root/autodl-tmp/icde_admission_20261105')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(OLD))
import run_admission as base

def main():
    out=ROOT/'outputs';out.mkdir(exist_ok=False)
    base.torch.set_num_threads(1);base.torch.set_num_interop_threads(1)
    logging.getLogger('gptcache').setLevel(logging.ERROR)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    # Load the preceding service's checked query vectors before changing its output root.
    encoder=base.Encoder()
    loaded={};all_texts=set()
    for ds,info in manifest['datasets'].items():
        p=ROOT/'inputs'/ds
        for name,digest in info['files'].items():assert base.sha(p/name)==digest
        f=json.loads((p/'train_frozen_selection.json').read_text())
        ref=json.loads((p/'test_coverage.json').read_text())
        v=np.load(p/'selected_vectors.npy');assert v.shape==(info['k'],384)
        pair=v@v.T;np.fill_diagonal(pair,-np.inf);assert np.all(pair<np.float32(.85))
        assert f['texts'] and len(f['texts'])==len(v)
        for text,vector in zip(f['texts'],v):encoder.lookup[text]=vector.copy()
        loaded[ds]=(f,ref);all_texts.update(f['texts'])
    base.ROOT=ROOT;base.OUT=out
    (ROOT/'provider_records').mkdir(exist_ok=True)
    copied=[]
    for p in (OLD/'provider_records').glob('*.json'):
        r=json.loads(p.read_text())
        request=r['request']
        if request['model']=='qwen3.8-flash' and request['messages'][-1]['content'] in all_texts and request['messages'][0]['content']==base.SYSTEM:
            q=ROOT/'provider_records'/p.name;shutil.copyfile(p,q);copied.append({'file':p.name,'source_sha256':base.sha(p)})
    provider=base.Provider();deid=base.KnownFieldDeidentifier()
    base.save(out/'protocol.json',{'threshold':base.TAU,'pairwise_threshold':'strictly <0.85','sequence':'saved offline selected-list order',
       'protocol_sha256':base.sha(ROOT/'protocol.md'),'script_sha256':base.sha(__file__),
       'previous_service_script_sha256':base.sha(OLD/'run_admission.py'),
       'input_manifest_sha256':base.sha(ROOT/'inputs/manifest.json'),
       'model_sha256':base.sha(base.MODEL/'model.safetensors'),'synthetic_field_registry':deid.registry,
       'deidentifier':'same placeholder-preserving known-field reference rule as previous experiment; not automatic PII detection',
       'initial_copied_provider_records':copied,'admission':'same GPTCache adapt miss-only callback; generated nonempty answer',
       'direct_preload':False,'test_creates_entries':False,'answer_validity_evaluated':False,
       'encoding':'verified frozen Test vectors plus exact selected vectors from the offline strict-pairwise analysis',
       'versions':{k:importlib.metadata.version(k) for k in ['gptcache','torch','transformers','numpy','faiss-cpu','sqlalchemy','openai']}})
    summary=[]
    for ds,info in manifest['datasets'].items():
        f,ref=loaded[ds];texts=f['texts'];testpath=OLD/'inputs'/ds/'test.jsonl'
        assert base.sha(testpath)==ref['test_sha256'];test=base.rows(testpath)
        for condition,raw_mode in [('placeholder_saved_order',False),('materialized_saved_order',True)]:
            service=base.ServingCache(out/ds/condition,deid,encoder,provider)
            for i,text in enumerate(texts):
                submitted=deid.materialize(text) if raw_mode else text
                service.submit(submitted,text,f'{ds}:strict_selected:{i}')
            result=service.evaluate(test,texts,ref['test_covered'])
            result.update({'dataset':ds,'condition':condition,'submitted_candidates':len(texts),
               'early_hits':sum(e['hit'] for e in service.events),'representations_preserved':sum(e['representation_equal'] for e in service.events),
               'actual_coverage_percent':100*result['actual_hits']/len(test),'offline_expected_hits':ref['test_covered'],
               'offline_expected_coverage_percent':ref['test_coverage_percent'],'pairwise_max':f['pairwise_max_float64'],
               'selection_freeze_sha256':base.sha(ROOT/'inputs'/ds/'train_frozen_selection.json')})
            base.save(service.path/'summary.json',result);summary.append(result);base.save(out/'summary.json',summary)
            service.close();print('RESULT',json.dumps(result),flush=True)
    base.save(out/'completion.json',{'complete':True,'conditions_completed':len(summary),'new_api_calls':provider.calls,
       'api_response_replays':provider.replays,'copied_previous_response_count':len(copied),
       'distinct_saved_response_count':len(list((ROOT/'provider_records').glob('*.json'))),
       'all_candidates_admitted':all(s['admitted_entries']==s['submitted_candidates'] for s in summary),
       'all_coverage_matches_offline':all(s['actual_hits']==s['offline_expected_hits'] for s in summary),
       'unexpected_encoding_calls':encoder.missing,'new_text_generation_search_calls':0,'completed_epoch':time.time()})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
