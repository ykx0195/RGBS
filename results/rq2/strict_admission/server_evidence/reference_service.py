"""Pre-registered ordinary-request admission; no direct preload or chosen answers."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import collections, csv, hashlib, importlib.metadata, json, logging, re, sqlite3, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from gptcache import Cache, Config
from gptcache.adapter.adapter import adapt
from gptcache.adapter.api import get
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.processor.post import first
from gptcache.similarity_evaluation import SimilarityEvaluation

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
MODEL=Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5')
CONFIG=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json')
TAU=.85
PLACEHOLDER=re.compile(r'<([A-Z][A-Z0-9_]*?)_(\d+)>')
SYSTEM=('You are a customer-support assistant. Answer the customer request in concise natural English. '
        'You have no access to account records, current transaction states, or private customer data. '
        'Do not claim to have accessed records or performed actions. Give relevant general next steps. '
        'Typed placeholders are opaque; do not invent personal facts.')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8');tmp.replace(p)
def rows(p):return [json.loads(l) for l in Path(p).read_text('utf-8').splitlines() if l.strip()]
def jsonl(p,rr):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),'utf-8')

class KnownFieldDeidentifier:
    """Controlled finite known-field registry; no automatic PII-detection claim."""
    def __init__(self):
        self.registry={
          'PERSON_1':'Alice Morgan','ACCOUNT_NUMBER_1':'8100000141','CREDIT_CARD_1':'4111111111111111',
          'CARD_NUMBER_1':'5555555555554444','ORDER_ID_1':'ORD-62004017','ACCOUNT_ID_1':'ACC-62004018',
          'BOOKING_ID_1':'BK-62004019','EMAIL_ADDRESS_1':'alice.morgan@example.com',
          'PHONE_NUMBER_1':'+1-202-555-0147','STREET_ADDRESS_1':'417 Cedar Avenue',
          'USERNAME_1':'alice_morgan_test','ZIP_CODE_1':'90210','SSN_LAST4_1':'7318',
          'POLICY_ID_1':'POL-62004020','ACCOUNT_ID_LAST4_1':'8429'}
        self.reverse={v:k for k,v in self.registry.items()}
        assert len(self.registry)==len(self.reverse)
        self.pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(v) for v in sorted(self.reverse,key=len,reverse=True))+r')(?!\w)')
    def materialize(self,s):
        return PLACEHOLDER.sub(lambda m:self.registry[m.group(1)+'_'+m.group(2)],s)
    def __call__(self,s):
        # Preserve existing placeholders; never modify an entire placeholder's spelling.
        parts=re.split(r'(<[A-Z][A-Z0-9_]*_\d+>)',s)
        return ''.join(p if PLACEHOLDER.fullmatch(p) else self.pattern.sub(lambda m:'<'+self.reverse[m.group(0)]+'>',p) for p in parts)

class CosineEvaluation(SimilarityEvaluation):
    def evaluation(self,src_dict,cache_dict,**kw):
        return float(np.dot(np.asarray(src_dict['embedding'],dtype=np.float32),np.asarray(cache_dict['embedding'],dtype=np.float32)))
    def range(self):return (0.,1.)

class Encoder:
    def __init__(self):
        self.lookup={}; self.calls=0; self.missing=[]
        assert sha(MODEL/'model.safetensors')=='3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad'
        manifest=json.loads((ROOT/'frozen_embeddings/manifest.json').read_text('utf-8'))
        for ds,info in manifest.items():
            p=ROOT/'frozen_embeddings'/(ds+'.npz');assert sha(p)==info['output_sha256']
            z=np.load(p)
            for t,v in zip(z['texts'].tolist(),z['vectors']):
                if t in self.lookup:assert np.allclose(self.lookup[t],v,atol=1e-5)
                self.lookup[t]=v.astype(np.float32)
        self.tokenizer=None;self.model=None
    def __call__(self,s,**kw):
        self.calls+=1
        if s not in self.lookup:
            if self.model is None:
                self.tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
                self.model=AutoModel.from_pretrained(str(MODEL),local_files_only=True).float().eval()
            b=self.tokenizer(s,return_tensors='pt',max_length=512,truncation=True)
            with torch.inference_mode():v=torch.nn.functional.normalize(self.model(**b).last_hidden_state[:,0],p=2,dim=-1)[0].numpy().astype(np.float32)
            self.lookup[s]=v;self.missing.append(s)
        return self.lookup[s].copy()

class Provider:
    def __init__(self):
        self.spec=next(x for x in json.loads(CONFIG.read_text())['api_generators'] if x['name']=='qwen_api')
        assert self.spec['model']=='qwen3.8-flash'
        p=Path(self.spec['key_file']);p=p if p.is_absolute() else CONFIG.parent/p
        self.client=OpenAI(api_key=p.read_text().strip(),base_url=self.spec['base_url'],timeout=60,max_retries=2)
        self.calls=0;self.replays=0
    def answer(self,s):
        request={'model':self.spec['model'],'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':s}],
                 'temperature':0,'max_tokens':384,'extra_body':self.spec.get('extra_body',{})}
        digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        p=ROOT/'provider_records'/(digest+'.json')
        replay=p.exists()
        if replay:
            r=json.loads(p.read_text());assert r['request']==request;self.replays+=1
        else:
            reply=self.client.chat.completions.create(**request)
            assert reply.choices[0].finish_reason=='stop', 'Truncated/nonfinal provider answer'
            r={'request':request,'response':reply.model_dump(),'answer':reply.choices[0].message.content,
               'created_utc_epoch':time.time()}
            assert r['answer'] and r['answer'].strip(),'Empty answer is not admitted'
            save(p,r);self.calls+=1
        return r['answer'],{'record_sha256':sha(p),'record_file':str(p.relative_to(ROOT)),
                            'api_response_id':r['response']['id'],'replayed':replay}

class ServingCache:
    def __init__(self,path,deid,encoder,provider):
        self.path=path;path.mkdir(parents=True,exist_ok=False)
        self.entries=[];self.events=[];self.deid=deid;self.encoder=encoder;self.provider=provider
        self.manager=get_data_manager(CacheBase('sqlite',sql_url='sqlite:///'+str(path/'cache.sqlite')),
            VectorBase('faiss',dimension=384,index_path=str(path/'faiss.index'),top_k=1),max_size=20000,clean_size=1)
        self.cache=Cache()
        self.cache.init(pre_embedding_func=lambda data,**kw:self.deid(data['prompt']),embedding_func=encoder,
            data_manager=self.manager,similarity_evaluation=CosineEvaluation(),post_process_messages_func=first,
            config=Config(similarity_threshold=TAU,auto_flush=1000000,enable_token_counter=False,disable_report=True))
    def submit(self,raw,expected,identifier):
        processed=self.deid(raw)
        e={'candidate_id':identifier,'submitted_text':raw,'encoder_input':processed,'expected_text':expected,
           'representation_equal':processed==expected,'hit':False,'written':False,'provider_record':None,'selected_text':None}
        assert processed==expected,'Unexpected representation change: preserve failure evidence'
        def hit_callback(matches):
            e['hit']=True;e['selected_text']=matches[0][0];e['selected_cosine']=float(matches[0][1])
        def handler(*args,**kw):
            ans,record=self.provider.answer(processed);e['provider_record']=record;return ans
        def update(ans,write,*args,**kw):
            assert ans and ans.strip()
            write(ans);e['written']=True
            self.entries.append({'candidate_id':identifier,'text':processed,'answer':ans,'provider_record':e['provider_record']})
            return ans
        answer=adapt(handler,lambda value:value,update,cache_obj=self.cache,prompt=raw,top_k=1,temperature=0,hit_callback=hit_callback)
        e['returned_answer']=answer
        assert e['hit'] != e['written']
        if e['hit']:assert e['provider_record'] is None
        self.events.append(e)
        self.cache.flush();save(self.path/'admission_events.json',self.events);save(self.path/'entries.json',self.entries)
        print('SUBMIT',self.path.name,identifier,'HIT' if e['hit'] else 'MISS_WRITE',flush=True)
        return e
    def db_state(self):
        self.cache.flush()
        with sqlite3.connect(self.path/'cache.sqlite') as c:
            n=c.execute('SELECT COUNT(*) FROM gptcache_question').fetchone()[0]
            columns=c.execute('PRAGMA table_info(gptcache_question)').fetchall()
            data=c.execute('SELECT * FROM gptcache_question').fetchall()
            data=[[{'blob_bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} if isinstance(v,bytes) else v for v in row] for row in data]
        assert n==len(self.entries),(n,len(self.entries))
        state={'question_rows':n,'columns':[x[1] for x in columns],'rows':data}
        save(self.path/'sqlite_state.json',state)
        return n
    def evaluate(self,test,full_texts,reference_hits):
        before=self.db_state()
        keys=np.array([self.encoder(e['text']) for e in self.entries]);queries=np.array([self.encoder(self.deid(r['text'])) for r in test])
        sims=queries@keys.T;full=queries@np.array([self.encoder(s) for s in full_texts]).T
        oracle=(sims.max(axis=1)>=TAU);assert int((full.max(axis=1)>=TAU).sum())==reference_hits
        output=[]
        for i,q in enumerate(test):
            processed=self.deid(q['text']);assert processed==q['text']
            ans=get(q['text'],cache_obj=self.cache,top_k=1,temperature=0)
            hit=ans is not None
            assert hit==bool(oracle[i]),('Oracle mismatch',i)
            if hit:
                # Answer provenance must be consistent with a highest-similarity admitted entry.
                eligible=[j for j,e in enumerate(self.entries) if e['answer']==ans and sims[i,j]>=TAU and abs(sims[i,j]-sims[i].max())<2e-5]
                assert eligible,('Unexpected returned answer',i)
            output.append({'test_index':i,'source_id':q.get('source_id'),'text':q['text'],'encoder_input':processed,
                           'cache_hit':hit,'oracle_hit':bool(oracle[i]),'max_cosine':float(sims[i].max()),'returned_answer':ans})
        after=self.db_state();assert before==after
        jsonl(self.path/'query_results.jsonl',output)
        np.savez_compressed(self.path/'similarities.npz',actual=sims,full=full)
        return {'test_total':len(test),'actual_hits':sum(x['cache_hit'] for x in output),'full_portfolio_hits':reference_hits,
                'admitted_entries':before,'rows_after_test':after,'oracle_disagreements':0,'test_model_calls':0,'test_writes':0}
    def close(self):self.cache.flush();self.manager.close()

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    logging.getLogger('gptcache').setLevel(logging.ERROR)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    deid=KnownFieldDeidentifier();provider=Provider();encoder=Encoder()
    save(OUT/'protocol.json',{'preregistered_protocol_sha256':sha(ROOT/'protocol.md'),'script_sha256':sha(__file__),
        'input_manifest_sha256':sha(ROOT/'inputs/manifest.json'),'model_sha256':sha(MODEL/'model.safetensors'),
        'deidentifier':'preserve existing typed placeholders; finite known synthetic-field registry, not automatic PII detection',
        'synthetic_field_registry':deid.registry,'threshold':TAU,'admission':'normal GPTCache adapt miss/update, nonempty generated answers',
        'provider_model':provider.spec['model'],'provider_system':SYSTEM,'provider_temperature':0,'provider_max_tokens':384,
        'empty_cache_per_condition':True,'benign_competitors':False,'test_cache_writes':False,'answer_correctness_evaluated':False,
        'versions':{k:importlib.metadata.version(k) for k in ['gptcache','torch','transformers','numpy','faiss-cpu','sqlalchemy','openai']}})
    summaries=[];singletons=[]
    for ds,info in manifest['datasets'].items():
        p=ROOT/'inputs'/ds
        for name,digest in info['files'].items():assert sha(p/name)==digest
        selected=json.loads((p/'constructed.json').read_text());test=rows(p/'test.jsonl');texts=[r['text'] for r in selected]
        assert len(texts)==info['k']
        for condition,order,raw_mode in [('placeholder_saved_order',list(range(len(texts))),False),
                                         ('materialized_saved_order',list(range(len(texts))),True),
                                         ('placeholder_reverse_order',list(reversed(range(len(texts)))),False)]:
            service=ServingCache(OUT/ds/condition,deid,encoder,provider)
            for i in order:service.submit(deid.materialize(texts[i]) if raw_mode else texts[i],texts[i],selected[i]['key_id'])
            result=service.evaluate(test,texts,info['expected_constructed_hits'])
            result.update({'dataset':ds,'condition':condition,'submitted_candidates':len(texts),'submission_order':order,
                           'early_hits':sum(e['hit'] for e in service.events),'representations_preserved':sum(e['representation_equal'] for e in service.events)})
            result['actual_coverage_percent']=100*result['actual_hits']/len(test)
            save(service.path/'summary.json',result);service.close();summaries.append(result);save(OUT/'summary.json',summaries)
            print('RESULT',json.dumps(result),flush=True)
        for i,s in enumerate(texts):
            service=ServingCache(OUT/ds/'singletons'/str(i),deid,encoder,provider)
            event=service.submit(deid.materialize(s),s,selected[i]['key_id'])
            assert not event['hit'] and event['written'] and service.db_state()==1
            singletons.append({'dataset':ds,'candidate_id':selected[i]['key_id'],'written':True,'representation_equal':True})
            service.close();save(OUT/'singleton_summary.json',singletons)
    save(OUT/'encoding_provenance.json',{'source':'verified frozen paper embeddings; missing exact strings encoded by matching BGE',
         'newly_encoded_texts':encoder.missing,'encoding_callback_calls':encoder.calls,'serving_latency_evaluated':False})
    records=[json.loads(p.read_text()) for p in (ROOT/'provider_records').glob('*.json')]
    usage=collections.Counter()
    for r in records:
        for k in ['prompt_tokens','completion_tokens','total_tokens']:usage[k]+=r['response'].get('usage',{}).get(k,0)
    save(OUT/'completion.json',{'complete':True,'conditions':len(summaries),'singletons':len(singletons),
         'new_api_calls_this_completed_run':provider.calls,'distinct_saved_api_responses_all_attempts':len(records),
         'api_response_replays_this_completed_run':provider.replays,'saved_api_usage_all_attempts':dict(usage),
         'all_oracle_checks_passed':True,'no_direct_preload':True,'completed_epoch':time.time()})
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
