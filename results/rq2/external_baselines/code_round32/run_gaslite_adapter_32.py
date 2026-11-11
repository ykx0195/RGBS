"""Official GASLITE token search adapted to whole cache queries.

Train-only common RKGB initialization. Two explicit objective variants.
Raw and quality-screened portfolios are frozen separately before Test access.
No claim of reproducing the original retrieval/ranking protocol unchanged.
"""
import argparse, importlib, json, os, random, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
sys.path.insert(0,str(ROOT/'code'))
from hybrid_search import BGE, Fluency, PLACEHOLDER, texts_from_jsonl, sha256, save_json, valid_text
from portfolio import greedy_portfolio, swap_portfolio, union_mask
from target_generation import prepare
from run_configurable import APIRewriter, load_provider_specs, typed_placeholder_context_ok, surface_fluency_ok, word_change_count, AREAS
sys.path.insert(0,str(ROOT/'official_gaslite'))
gas=importlib.import_module('src.attacks.gaslite')

class TokenizerProxy:
    def __init__(self,tok): self.tok=tok; self.replacements={}
    def __getattr__(self,key):return getattr(self.tok,key)
    def __call__(self,*a,**kw):return self.tok(*a,**kw)
    def canonical(self,text):
        # BERT lowercases and spaces typed placeholders. Restoring their spelling
        # leaves token IDs unchanged; this is checked for every initial seed.
        return re.sub(r'<\s*[a-zA-Z0-9_ ]+\s*>',lambda m:self.replacements.get(re.sub(r'\s+','',m.group()).lower(),m.group()),text)
    def decode(self,*a,**kw):return self.canonical(self.tok.decode(*a,**kw))
    def batch_decode(self,*a,**kw):return [self.canonical(t) for t in self.tok.batch_decode(*a,**kw)]

class DisabledFluency:
    # Official default flu_alpha=0 still initializes a GPT model solely for its
    # logging metric. Skip that download; do not fabricate a fluency measurement.
    def calc_fluency_scores(self,inputs=None,text_inputs=None,**kw):
        n=len(text_inputs) if text_inputs is not None else len(inputs['input_ids'])
        return torch.full((n,),float('nan'),device='cuda')

class Adapter:
    def __init__(self,enc,queries,objective):
        self.enc=enc; self.tokenizer=TokenizerProxy(enc.tokenizer)
        self.queries=torch.as_tensor(queries,device='cuda');self.objective=objective
        self.seen={}; self.grad_calls=0;self.eval_rows=0
    def get_input_embeddings(self):return self.enc.embeddings
    def get_special_token_ids(self):
        return sorted(set(self.tokenizer.all_special_ids)|{i for i,t in enumerate(self.enc.tokens) if t.startswith('[unused') or not self.tokenizer.decode([i]).isascii()})
    def score(self,v):
        if self.objective=='mean_cosine':return v @ self.queries.mean(0)
        return torch.sigmoid((v @ self.queries.T-0.85)/0.02).mean(1)
    def embed(self,texts=None,inputs=None,to_cpu=False):
        if texts is not None:
            v=torch.as_tensor(self.enc.encode(texts,32),device='cuda')
        else:
            chunks=[]
            for start in range(0,len(inputs['input_ids']),32):
                part={k:v[start:start+32] for k,v in inputs.items()}
                with torch.no_grad():
                    h=self.enc.model(**part).last_hidden_state
                    chunks.append(F.normalize(self.enc._pool(h,part['attention_mask']),dim=-1))
            v=torch.cat(chunks)
        return v.cpu() if to_cpu else v
    def calc_loss_for_grad(self,inputs_embeds,inputs_attention_mask,**kw):
        self.grad_calls+=1
        h=self.enc.model(inputs_embeds=inputs_embeds,attention_mask=inputs_attention_mask).last_hidden_state
        return self.score(F.normalize(self.enc._pool(h,inputs_attention_mask),dim=-1)).mean()
    def calc_loss_for_eval(self,inputs,**kw):
        v=self.embed(inputs=inputs); scores=self.score(v)
        texts=self.tokenizer.batch_decode(inputs['input_ids'],skip_special_tokens=True)
        self.eval_rows+=len(texts)
        # Archive decoded candidates, retaining the entire search trajectory.
        # Structural rules guard typed placeholders and the shared edit limit.
        for i,text in enumerate(texts):
            ok=valid_text(text,self.seed) and typed_placeholder_context_ok(text) and surface_fluency_ok(text) and word_change_count(self.seed,text)<=10
            if ok:self.seen.setdefault(text,float(scores[i]))
            else:scores[i]=-1e9
        return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['abcd','dogo','banking77'])
    ap.add_argument('--objective',required=True,choices=['mean_cosine','soft_threshold'])
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    k={'abcd':5,'dogo':10,'banking77':15}[a.dataset]
    out=ROOT/('gaslite_32_smoke' if a.smoke else 'runs/gaslite_32')/a.objective/a.dataset
    out.mkdir(parents=True,exist_ok=True)
    if (out/'final_report.json').exists(): print('Already completed',out,flush=True);return
    data=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data')/a.dataset.replace('dogo','multidogo')/'desensitized'
    manifest=json.loads((data/'manifest.json').read_text())
    assert sha256(data/'train.jsonl')==manifest['files']['train.jsonl']['sha256']
    texts=texts_from_jsonl(data/'train.jsonl')
    enc=BGE(Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5'),'cuda','bge')
    train=enc.encode(texts,64)
    # Use exactly the common initialization prepared by the full paired lane.
    common=ROOT/'runs/full'/a.dataset
    if a.smoke:common=ROOT/'smoke/full'
    if not (common/'targets/seed_selection.json').exists():
        raise RuntimeError('Common initialization not prepared yet: '+str(common))
    identity=json.loads((common/'train_vectors.json').read_text())
    assert identity['train_sha256']==sha256(data/'train.jsonl')
    assert identity['train_rows']==len(texts)
    targets=np.load(common/'targets/target_vectors.npy')
    seeds=json.loads((common/'targets/seed_selection.json').read_text())['selected']
    groups=(train @ targets.T).argmax(1)
    config={'dataset':a.dataset,'k':k,'threshold':0.85,'seed':42,'objective':a.objective,
            'official_commit':'8dc7646f1ac8cd68f38463df01578c74fff22795',
            'n_iter':1 if a.smoke else 32,'n_grad':2 if a.smoke else 50,
            'k_candidates':16 if a.smoke else 256,'beam_candidates':4 if a.smoke else 128,
            'beam_width':1,'n_flip':2 if a.smoke else 20,'flu_alpha':0,
            'common_train_sha256':sha256(data/'train.jsonl'),
            'common_seed_sha256':sha256(common/'targets/seed_selection.json'),
            'quality_archive_cap_per_slot':256,'quality_accept_cap_per_slot':20,
            'quality':{'ppl':150,'relative_ppl':1.5,'total_word_edits':10,'local_readability':True,'two_api_scope_judges':True},
            'note':'Adapted whole-query interface and shared geometric initialization, not original paper ranking evaluation. Default no differentiable fluency; use Train-ranked archive screening.'}
    save_json(out/'config.json',config)
    gas.initialize_fluency_model=lambda *args,**kwargs:DisabledFluency()
    originals=[];pool=[]
    for index,seedrow in enumerate(seeds):
        slot=out/f'slot_{index+1:02d}.json'
        if slot.exists():
            payload=json.loads(slot.read_text())
            assert payload.get("n_iter") == config["n_iter"], "Checkpoint iteration budget differs"
        else:
            started=time.time();seed=seedrow['text'];guide=seedrow['guide_index']
            q=train[groups==guide]
            if not len(q):q=train[[seedrow['train_index']]]
            model=Adapter(enc,q,a.objective);model.seed=seed
            model.tokenizer.replacements={x.lower():x for x in PLACEHOLDER.findall(seed)}
            inputs=model.tokenizer(seed,return_tensors='pt').to('cuda')
            enc.tokenizer.model_max_length=inputs['input_ids'].shape[1]
            # Ensure restoring placeholder spelling preserves all input IDs.
            decoded=model.tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=True)
            assert torch.equal(model.tokenizer(decoded,return_tensors='pt')['input_ids'],inputs['input_ids'].cpu())
            model.seen[seed]=float(model.score(torch.as_tensor(enc.encode([seed]),device='cuda')))
            ids,metrics=gas.gaslite_attack(model,inputs,trigger_slice=slice(1,inputs['input_ids'].shape[1]-1),
                n_iter=config['n_iter'],n_grad=config['n_grad'],k_candidates=config['k_candidates'],
                n_flips_to_gen=512,backprop_batch_size=10,evaluate_flips_on_strs=True,
                beam_search_config={'perform':True,'n_cand':config['beam_candidates'],'B':1,'n_flip':config['n_flip'],
                                    'n_cand__sample_rate':0.5,'beam_indices':'random'},
                flu_alpha=0,log_to=None,emb_targets=torch.as_tensor(q.mean(0,keepdims=True),device='cuda'))
            payload={'n_iter':config['n_iter'],'slot':index+1,'seed':seed,'guide':int(guide),'final_text':model.tokenizer.decode(ids[0],skip_special_tokens=True),
                     'archive':[{'text':t,'objective':s} for t,s in model.seen.items()],
                     'gradient_calls':model.grad_calls,'candidate_evaluations':model.eval_rows,'seconds':time.time()-started,
                     'note':'Logging-only fluency disabled; PPL measured separately. Official token search unchanged.'}
            save_json(slot,payload)
        pool.extend(row['text'] for row in payload['archive']);originals.append(payload['seed'])
        print(f"slot={index+1}/{k} archived={len(payload['archive'])} seconds={payload['seconds']:.1f}",flush=True)
        if a.smoke:print('Smoke complete; Test not opened',flush=True);return
    # Train ranking freezes raw selection before any Test reads.
    unique=list(dict.fromkeys(pool));vectors=enc.encode(unique,64)
    rows=[{'text':t,'vector':v,'mask':train @ v>=0.85,'target_cosine':0.0,'ppl':None} for t,v in zip(unique,vectors)]
    raw=swap_portfolio(greedy_portfolio(rows,k),rows,4)
    api=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json')
    providers=[APIRewriter(spec,api.parent,out/'api_cache') for spec in load_provider_specs(api)]
    flu=Fluency(Path('/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct'),'cuda')
    qualified={};quality_log=[]
    for index,seed in enumerate(originals):
        archive=json.loads((out/f'slot_{index+1:02d}.json').read_text())['archive']
        rowmap={r['text']:r for r in rows}
        ranked=sorted([rowmap[r['text']] for r in archive],key=lambda r:int(r['mask'].sum()),reverse=True)[:256]
        seedppl=flu.ppl(seed);eligible=[]
        for row in ranked:
            ppl=flu.ppl(row['text'])
            if ppl>min(150,1.5*seedppl):continue
            if not flu.judge_readability([row['text']],AREAS[a.dataset])[0]:continue
            eligible.append(dict(row,ppl=ppl))
            if len(eligible)>=20:break
        verdicts=[p.judge([r['text'] for r in eligible],a.dataset) for p in providers]
        for j,row in enumerate(eligible):
            if all(v[j] for v in verdicts):qualified[row['text']]=row
        quality_log.append({'slot':index+1,'archive':len(archive),'considered':len(ranked),'ppl_and_local_eligible':len(eligible),'two_judge_accepted':sum(all(v[j] for v in verdicts) for j in range(len(eligible)))})
        save_json(out/'quality_screen_progress.json',{'completed_slots':index+1,'slots':quality_log})
    eligible=list(qualified.values())
    screened=swap_portfolio(greedy_portfolio(eligible,k),eligible,4) if len(eligible)>=k else eligible
    serialize=lambda rr:[{'text':r['text'],'ppl':r['ppl']} for r in rr]
    frozen={'config':config,'raw_selected':serialize(raw),'screened_selected':serialize(screened),
            'quality_screen':quality_log,'screened_key_count':len(screened),'heldout_loaded':False,
            'train':{name:int(union_mask([r['mask'] for r in rr]).sum()) if rr else 0 for name,rr in [('raw',raw),('screened',screened)]}}
    save_json(out/'train_report.json',frozen)
    # First Test access, after BOTH portfolios are frozen.
    assert sha256(data/'test.jsonl')==manifest['files']['test.jsonl']['sha256']
    test=enc.encode(texts_from_jsonl(data/'test.jsonl'),64)
    metrics={}
    for name,rr in [('raw',raw),('screened',screened)]:
        count=int(((test @ enc.encode([r['text'] for r in rr]).T).max(1)>=0.85).sum()) if rr else 0
        metrics[name]={'covered':count,'total':len(test),'coverage':count/len(test),'key_count':len(rr)}
    save_json(out/'final_report.json',{**frozen,'heldout_loaded':True,'test_sha256':sha256(data/'test.jsonl'),'test':metrics})
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
