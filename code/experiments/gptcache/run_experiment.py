"""GPTCache preload/lookup experiment; marker payloads, no LLM or answer-validity claim."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
import csv, hashlib, importlib.metadata, inspect, json, logging, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from gptcache import Cache, Config
from gptcache.adapter.api import put, get
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.processor.pre import get_prompt
from gptcache.processor.post import first
from gptcache.similarity_evaluation import SimilarityEvaluation

ROOT = Path(__file__).resolve().parent
MODEL = Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5')
OUT = ROOT / 'outputs'
THRESHOLD = .85

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', 'utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    return [json.loads(x) for x in path.read_text('utf-8').splitlines() if x.strip()]

class CosineEvaluation(SimilarityEvaluation):
    """Expose raw cosine in [0,1] operating range, so Config(.85) means cos>=.85."""
    def evaluation(self, src_dict, cache_dict, **kwargs):
        return float(np.dot(np.asarray(src_dict['embedding'], dtype=np.float32),
                            np.asarray(cache_dict['embedding'], dtype=np.float32)))
    def range(self):
        return (0., 1.)

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=True)
    logging.getLogger('gptcache').setLevel(logging.ERROR)
    model_hash = sha(MODEL/'model.safetensors')
    assert model_hash == '3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad'
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(MODEL), local_files_only=True).float().eval()
    pooling = json.loads((MODEL/'1_Pooling/config.json').read_text())
    assert pooling['pooling_mode_cls_token'] and not pooling['pooling_mode_mean_tokens']
    max_length = json.loads((MODEL/'sentence_bert_config.json').read_text())['max_seq_length']
    manifest = json.loads((ROOT/'inputs/manifest.json').read_text('utf-8'))
    provenance = {'model_sha256': model_hash, 'pooling_sha256': sha(MODEL/'1_Pooling/config.json'),
                  'tokenizer_sha256': sha(MODEL/'tokenizer.json'), 'max_length': max_length,
                  'pooling': 'CLS', 'normalize': True, 'dtype': 'float32', 'device': 'cpu',
                  'versions': {k:importlib.metadata.version(k) for k in ['gptcache','torch','transformers','faiss-cpu','numpy','sqlalchemy']},
                  'threshold': THRESHOLD, 'top_k': 1, 'postprocess': 'first',
                  'vector_index': 'FAISS IndexFlatL2 via GPTCache', 'test_cache_writes': False,
                  'answer_payload': 'unique provenance marker; no generated answers; no validity assessment',
                  'admission': 'direct put (force preload), not user-query admission',
                  'latency_scope': 'lookup with precomputed BGE embeddings, NOT end-to-end serving latency',
                  'input_manifest_sha256': sha(ROOT/'inputs/manifest.json'), 'script_sha256': sha(Path(__file__)),
                  'frozen_embedding_manifest_sha256': sha(ROOT/'frozen_embeddings/manifest.json'),
                  'encoding_source': 'verified frozen paper embeddings; only missing main-set strings re-encoded'}
    save(OUT/'protocol.json', provenance)
    all_summary=[]
    for dataset, info in manifest['datasets'].items():
        print('DATASET', dataset, flush=True)
        folder=ROOT/'inputs'/dataset
        for filename, digest in info['files'].items():
            assert sha(folder/filename)==digest, filename
        train=rows(folder/'train.jsonl'); test=rows(folder/'test.jsonl')
        constructed=json.loads((folder/'constructed.json').read_text('utf-8'))
        texts=list(dict.fromkeys([r['text'] for r in train+test+constructed]))
        embfile=OUT/(dataset+'_embeddings.npz')
        start=time.perf_counter()
        if embfile.exists():
            stored=np.load(embfile)
            assert stored['texts'].tolist()==texts
            vectors=stored['vectors']
        else:
            frozen_info=json.loads((ROOT/'frozen_embeddings/manifest.json').read_text('utf-8'))[dataset]
            frozen_path=ROOT/'frozen_embeddings'/(dataset+'.npz')
            assert sha(frozen_path)==frozen_info['output_sha256']
            frozen=np.load(frozen_path)
            known=dict(zip(frozen['texts'].tolist(),frozen['vectors']))
            missing=[t for t in texts if t not in known]
            for i in range(0,len(missing),64):
                features=tokenizer(missing[i:i+64],padding=True,truncation=True,max_length=max_length,return_tensors='pt')
                with torch.inference_mode():
                    h=model(**features).last_hidden_state[:,0]
                    batch=torch.nn.functional.normalize(h,p=2,dim=-1).cpu().numpy().astype(np.float32)
                known.update(zip(missing[i:i+64],batch))
            vectors=np.array([known[t] for t in texts],dtype=np.float32)
            save(OUT/(dataset+'_encoding_provenance.json'),{'frozen_reused_texts':len(texts)-len(missing),'newly_encoded_texts':missing,'frozen_source_manifest':frozen_info})
            np.savez_compressed(embfile,texts=np.array(texts),vectors=vectors)
        lookup=dict(zip(texts,vectors))
        print('ENCODE_DONE',dataset,round(time.perf_counter()-start,2),flush=True)
        dataset_out=OUT/dataset; dataset_out.mkdir(exist_ok=True)
        for condition, raw_entries in [('constructed_only',constructed),('normal_train_only',train)]:
            directory=dataset_out/condition
            if directory.exists(): directory.rename(dataset_out/(condition+'_previous_'+str(time.time_ns())))
            directory.mkdir()
            # Collapse identical keys; provenance remains explicit. Full source IDs retained below.
            entries=[]; seen=set()
            for i,row in enumerate(raw_entries):
                if row['text'] in seen: continue
                seen.add(row['text'])
                entries.append({'text':row['text'],'source_id':row.get('source_id'),
                                'category':row.get('category'),'marker':f'CACHE_ENTRY::{dataset}::{condition}::{i:06d}'})
            payload_map={e['marker']:e for e in entries}
            save(directory/'entries.json',entries)
            dm=get_data_manager(CacheBase('sqlite',sql_url='sqlite:///'+str(directory/'cache.sqlite')),
                                VectorBase('faiss',dimension=384,index_path=str(directory/'faiss.index'),top_k=1),
                                max_size=max(20000,len(entries)+100),clean_size=1)
            cache=Cache()
            embed_calls=[0]
            def embed(text,**kwargs):
                embed_calls[0]+=1
                return lookup[text].copy()
            cache.init(pre_embedding_func=get_prompt,embedding_func=embed,data_manager=dm,
                       similarity_evaluation=CosineEvaluation(),post_process_messages_func=first,
                       config=Config(similarity_threshold=THRESHOLD,auto_flush=1000000))
            for e in entries: put(e['text'],e['marker'],cache_obj=cache)
            cache.flush()
            # Confirm every preloaded entry and index row is physically present.
            import sqlite3
            con=sqlite3.connect(directory/'cache.sqlite')
            tables=con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            save(directory/'sqlite_counts_before.json',{t:con.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for (t,) in tables})
            before=con.execute('SELECT COUNT(*) FROM gptcache_question').fetchone()[0]
            assert before==len(entries),(before,len(entries))
            kv=np.array([lookup[e['text']] for e in entries],dtype=np.float32)
            qv=np.array([lookup[r['text']] for r in test],dtype=np.float32)
            similarities=qv@kv.T
            oracle_indices=similarities.argmax(axis=1)
            oracle_max=similarities.max(axis=1)
            output=[]
            for i,q in enumerate(test):
                tick=time.perf_counter()
                returned=get(q['text'],cache_obj=cache,top_k=1,temperature=0)
                elapsed=time.perf_counter()-tick
                selected=payload_map.get(returned)
                assert returned is None or selected is not None
                actual_cos=float(np.dot(lookup[q['text']],lookup[selected['text']] )) if selected else None
                oracle_hit=bool(oracle_max[i]>=THRESHOLD)
                assert (selected is not None)==oracle_hit, ('hit discrepancy',dataset,condition,i,oracle_max[i],returned)
                if selected:
                    assert actual_cos>=THRESHOLD
                    assert abs(actual_cos-float(oracle_max[i]))<2e-5
                output.append({'test_index':i,'test_source_id':q.get('source_id'),'test_text':q['text'],
                               'test_category':q.get('category'),'cache_hit':selected is not None,
                               'returned_marker':returned,'selected_text':selected['text'] if selected else None,
                               'selected_source_id':selected['source_id'] if selected else None,
                               'selected_category':selected['category'] if selected else None,
                               'selected_cosine':actual_cos,'oracle_max_cosine':float(oracle_max[i]),
                               'oracle_hit':oracle_hit,'lookup_ms_precomputed_embedding':1000*elapsed})
            cache.flush()
            after=con.execute('SELECT COUNT(*) FROM gptcache_question').fetchone()[0]
            assert before==after
            con.close()
            hits=sum(x['cache_hit'] for x in output)
            if condition=='constructed_only': assert hits==info['expected_constructed_hits'],(dataset,hits,info['expected_constructed_hits'])
            save(directory/'query_results.json',output)
            with (directory/'query_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
            summary={'dataset':dataset,'condition':condition,'source_entries':len(raw_entries),'unique_cache_entries':len(entries),
                     'test_total':len(test),'cache_hits':hits,'hit_rate_percent':100*hits/len(test),
                     'oracle_hits':int((oracle_max>=THRESHOLD).sum()),'oracle_disagreements':0,
                     'cache_rows_before':before,'cache_rows_after':after,'embedding_callback_calls':embed_calls[0],
                     'distinct_returned_entries':len({r['returned_marker'] for r in output if r['cache_hit']}),
                     'answer_validity_evaluated':False,'admission_evaluated':False}
            save(directory/'summary.json',summary)
            all_summary.append(summary); save(OUT/'summary.json',all_summary)
            dm.close()
            print('RESULT',json.dumps(summary),flush=True)
    save(OUT/'verification.json',{'complete':True,'conditions_completed':len(all_summary),'all_query_oracle_agreement':True,
                                  'all_cache_sizes_unchanged':True,'all_paper_constructed_hits_reproduced':True})
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
