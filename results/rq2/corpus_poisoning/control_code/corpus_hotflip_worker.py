"""Corpus-poisoning HotFlip adapted to query-only BGE semantic-cache evaluation.

Official gradient-vocabulary core is imported unchanged; remaining adapters and
checkpoint rules are documented in experiment_plan.json. Never opens Test.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,random,re,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from official_hotflip_core import hotflip_attack

PLACEHOLDER=re.compile(r'<[A-Z][A-Z0-9_]*>')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()

class Adapter:
    def __init__(self,path,device):
        self.device=torch.device(device)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,use_fast=True)
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.embedding=self.model.get_input_embeddings();self.max_length=512
        self.forward_texts=0;self.backward_calls=0
    def encode(self,texts,batch=32):
        vectors=[]
        for start in range(0,len(texts),batch):
            f=self.tokenizer(texts[start:start+batch],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to(self.device)
            with torch.no_grad():z=F.normalize(self.model(**f).last_hidden_state[:,0],p=2,dim=-1)
            vectors.append(z.detach());self.forward_texts+=len(texts[start:start+batch])
        return torch.cat(vectors)
    def gradient(self,text,target):
        f=self.tokenizer(text,truncation=True,max_length=self.max_length,return_offsets_mapping=True,return_tensors='pt')
        offsets=f.pop('offset_mapping')[0].tolist();f=f.to(self.device)
        embedded=self.embedding(f['input_ids']).detach().requires_grad_(True)
        kwargs={k:v for k,v in f.items() if k!='input_ids'}
        hidden=self.model(inputs_embeds=embedded,**kwargs).last_hidden_state
        vector=F.normalize(hidden[:,0],p=2,dim=-1)[0];objective=vector@target
        grad=torch.autograd.grad(objective,embedded)[0][0].detach()
        self.forward_texts+=1;self.backward_calls+=1
        assert torch.isfinite(grad).all()
        ranges=[];protected=set()
        for match in PLACEHOLDER.finditer(text):
            loc=[i for i,(a,b) in enumerate(offsets) if b>a and a<match.end() and b>match.start()]
            assert loc and loc==list(range(min(loc),max(loc)+1))
            ranges.append((min(loc),max(loc)+1,match.group()));protected.update(loc)
        ids=f['input_ids'][0].tolist();specials=set(self.tokenizer.all_special_ids)
        editable=[i for i,(a,b) in enumerate(offsets) if b>a and i not in protected and ids[i] not in specials]
        return ids,grad,ranges,editable,vector.detach()
    def render(self,ids,ranges):
        chunks=[];cursor=0
        for left,right,text in ranges:
            chunk=self.tokenizer.decode(ids[cursor:left],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
            if chunk:chunks.append(chunk)
            chunks.append(text);cursor=right
        chunk=self.tokenizer.decode(ids[cursor:],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
        if chunk:chunks.append(chunk)
        return ' '.join(chunks).strip()

def run(args):
    root=Path(__file__).resolve().parent.parent;inp=root/'inputs'/args.dataset
    out=root/('smoke' if args.smoke else 'runs')/args.dataset/f'cluster_{args.cluster:02d}';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    rng=np.random.default_rng(args.seed+10007*args.cluster);prng=random.Random(args.seed+10007*args.cluster)
    cfg=read(root/'host_config.json');plan=read(root/'experiment_plan.json');seeds=read(inp/'seed_selection.json')
    identity=read(inp/'input_identity.json')
    assert sha(inp/'train_vectors.npy')==identity['vectors_sha256'] and sha(inp/'train.jsonl')==identity['train_sha256']
    rows=[json.loads(x) for x in (inp/'train.jsonl').read_text().splitlines() if x.strip()]
    train=np.load(inp/'train_vectors.npy');labels=np.load(inp/'cluster_labels.npy');indices=np.flatnonzero(labels==args.cluster)
    start=seeds['selected'][args.cluster];assert start['cluster_id']==args.cluster
    assert start['text']==rows[start['train_index']]['text'] and len(indices)==start['cluster_size']
    modelpath=Path(cfg['model']);assert sha(modelpath/'model.safetensors')==seeds['model_sha256']
    config={'dataset':args.dataset,'cluster':args.cluster,'seed':args.seed,'max_iterations':args.iterations,
            'num_candidates':args.candidates,'query_batch':64,'candidate_batch':args.candidate_batch,'threshold':.85,
            'train_sha256':identity['train_sha256'],'vectors_sha256':identity['vectors_sha256'],'model_sha256':seeds['model_sha256'],
            'official_commit':plan['official_commit'],'search_code_sha256':sha(__file__),
            'official_core_sha256':sha(Path(__file__).parent/'official_hotflip_core.py'),'device':args.device,
            'checkpoint_selection':'highest full cluster Train mean cosine; no heldout used','smoke':args.smoke}
    configpath=out/'config.json'
    if configpath.exists():assert read(configpath)==config,'Resume configuration differs'
    else:save(configpath,config)
    if (out/'final_cluster.json').exists():print('CLUSTER_ALREADY_COMPLETE',args.dataset,args.cluster,flush=True);return
    model=Adapter(modelpath,args.device);queries=torch.as_tensor(train[indices],device=model.device);fullmean=queries.mean(dim=0)
    initial=start['text'];text=initial;initial_v=model.encode([text])[0];full_score=float(initial_v@fullmean)
    best={'text':text,'full_cluster_mean_cosine':full_score,'selected_at_iteration':0}
    initial_count=Counter(PLACEHOLDER.findall(initial))
    accepted=0;attempted=0;valid_total=0;iteration_start=0;stale=0;accounted_seconds=0
    statepath=out/'state.json';previous_forward=previous_backward=0
    if statepath.exists():
        state=read(statepath);assert state['config']==config
        text=state['current_text'];best=state['best'];iteration_start=state['completed_iterations'];accepted=state['accepted_updates']
        attempted=state['proposed_candidates'];valid_total=state['valid_candidates'];stale=state.get('stale_updates',0)
        rng.bit_generator.state=state['rng_state']
        def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,list) else x
        prng.setstate(tuples(state['python_rng_state']));accounted_seconds=state['active_seconds']
        previous_forward=state['forward_texts'];previous_backward=state['backward_calls']
    # Verify forward and gradient paths agree and initial anchors reproduce cached vectors.
    if iteration_start==0:
        difference=float(torch.max(torch.abs(initial_v-torch.as_tensor(train[start['train_index']],device=model.device))))
        assert difference<2e-4,('BGE initial cache mismatch',difference)
        save(out/'preflight.json',{'max_abs_initial_embedding_delta':difference,'initial_train_index':start['train_index'],
             'placeholder_count':dict(initial_count),'cluster_queries':len(indices),'heldout_loaded':False})
    started=time.monotonic();budgets=set(plan['checkpoint_budgets'])|{args.iterations}
    def checkpoint(i):
        save(statepath,{'config':config,'completed_iterations':i,'current_text':text,'best':best,'accepted_updates':accepted,
             'proposed_candidates':attempted,'valid_candidates':valid_total,'stale_updates':stale,
             'rng_state':rng.bit_generator.state,'python_rng_state':prng.getstate(),
             'active_seconds':accounted_seconds+time.monotonic()-started,
             'forward_texts':previous_forward+model.forward_texts,'backward_calls':previous_backward+model.backward_calls,
             'heldout_loaded':False,'checkpointed_at_utc':now()})
        if i in budgets:save(out/f'budget_{i:05d}.json',{'iteration_budget':i,'selected':best,'accepted_updates':accepted,
                    'active_seconds':accounted_seconds+time.monotonic()-started,'heldout_loaded':False})
    if iteration_start==0:checkpoint(0)
    print('CLUSTER_SEARCH_START',args.dataset,args.cluster,'device',args.device,'resume',iteration_start,'queries',len(indices),flush=True)
    trace=(out/'trace.jsonl').open('a',buffering=1)
    no_editable=False
    for iteration in range(iteration_start+1,args.iterations+1):
        gid=rng.choice(len(queries),size=min(64,len(queries)),replace=False);eid=rng.choice(len(queries),size=min(64,len(queries)),replace=False)
        gradient_target=queries[torch.as_tensor(gid,device=model.device)].mean(dim=0)
        score_target=queries[torch.as_tensor(eid,device=model.device)].mean(dim=0)
        ids,grad,ranges,editable,current_vector=model.gradient(text,gradient_target)
        if not editable:no_editable=True;checkpoint(iteration-1);break
        position=prng.choice(editable)
        gradient=grad[position]
        candidates=hotflip_attack(gradient,model.embedding.weight,increase_loss=True,num_candidates=args.candidates)
        # Independently check the imported core's ascent direction on the first step.
        if iteration==1:
            expected=(model.embedding.weight.detach()@gradient).topk(args.candidates).indices
            assert torch.equal(expected,candidates)
            forward=model.encode([text])[0]
            assert float(torch.max(abs(current_vector-forward)))<2e-5
        proposed=[];seen={text};specials=set(model.tokenizer.all_special_ids)
        for candidate in candidates.tolist():
            attempted+=1
            if candidate in specials or candidate==ids[position]:continue
            changed=list(ids);changed[position]=candidate;candidate_text=model.render(changed,ranges)
            if candidate_text in seen or not candidate_text:continue
            seen.add(candidate_text)
            if Counter(PLACEHOLDER.findall(candidate_text))!=initial_count:continue
            residue=PLACEHOLDER.sub('',candidate_text)
            if '<' in residue or '>' in residue or any(ord(c)<32 and not c.isspace() for c in candidate_text):continue
            nids=model.tokenizer(candidate_text,add_special_tokens=True)['input_ids']
            if len(nids)>512:continue
            proposed.append(candidate_text)
        valid_total+=len(proposed);old_score=float(current_vector@score_target);new_score=old_score
        if proposed:
            vv=model.encode(proposed,args.candidate_batch);scores=vv@score_target;winner=int(torch.argmax(scores));new_score=float(scores[winner])
            if new_score>old_score+1e-7:
                text=proposed[winner];accepted+=1;stale=0
                exact_score=float(vv[winner]@fullmean)
                if exact_score>best['full_cluster_mean_cosine']+1e-7:
                    best={'text':text,'full_cluster_mean_cosine':exact_score,'selected_at_iteration':iteration}
            else:stale+=1
        else:stale+=1
        if iteration%25==0 or iteration==1 or iteration in budgets:
            trace.write(json.dumps({'iteration':iteration,'accepted_updates':accepted,'best_full_train_mean':best['full_cluster_mean_cosine'],
                                   'sample_old_score':old_score,'sample_best_score':new_score,'valid_candidates':len(proposed),
                                   'active_seconds':accounted_seconds+time.monotonic()-started})+'\n')
        if iteration%100==0 or iteration in budgets:
            checkpoint(iteration)
            print('CLUSTER_PROGRESS',args.dataset,args.cluster,iteration,args.iterations,'accepted',accepted,'best',round(best['full_cluster_mean_cosine'],6),flush=True)
    trace.close()
    completed=iteration-1 if no_editable else args.iterations;checkpoint(completed)
    assert Counter(PLACEHOLDER.findall(best['text']))==initial_count
    save(out/'final_cluster.json',{'config':config,'initial':start,'selected':best,'completed_iterations':completed,
          'termination':'no_editable_tokens' if no_editable else 'iteration_budget_exhausted',
          'accepted_updates':accepted,'proposed_candidates':attempted,'valid_candidates':valid_total,
          'active_seconds':accounted_seconds+time.monotonic()-started,'forward_texts':previous_forward+model.forward_texts,
          'backward_calls':previous_backward+model.backward_calls,'heldout_loaded':False,'api_calls':0,'finished_at_utc':now()})
    print('CLUSTER_COMPLETE',args.dataset,args.cluster,completed,best['full_cluster_mean_cosine'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['banking77','abcd','dogo']);p.add_argument('--cluster',type=int,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cuda','cpu'],required=True);p.add_argument('--iterations',type=int,default=5000)
    p.add_argument('--candidates',type=int,default=100);p.add_argument('--candidate-batch',type=int,default=32);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
