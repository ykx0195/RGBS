"""Synthetic auxiliary-model interface check, never an experimental result."""
from pathlib import Path
import json,shutil,sys
import torch
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
sys.path.insert(0,str(ROOT))
from run_gaslite_flu import Adapter,BGE,gas,LocalAuthorFluency
from src.nanoGPT.model import GPT,GPTConfig

torch.set_num_threads(2)
stub=ROOT/'flu_interface_stub';stub.mkdir(exist_ok=True)
if not (stub/'e5_tokenizer').exists():shutil.copytree(ROOT/'flu_assets/e5_tokenizer',stub/'e5_tokenizer')
model_args={'block_size':512,'vocab_size':30522,'n_layer':1,'n_head':4,'n_embd':32,'dropout':0.0,'bias':True}
model=GPT(GPTConfig(**model_args))
torch.save({'model_args':model_args,'model':model.state_dict()},stub/'ckpt.pt');del model
(stub/'verified_assets.json').write_text(json.dumps({'type':'synthetic random toy weights; interface check only'}))
enc=BGE(Path('/root/autodl-tmp/liman/models/bge-small-en-v1.5'),'cuda','bge')
seeds=json.loads((ROOT/'runs/full/abcd/targets/seed_selection.json').read_text())['selected']
seed=seeds[0]['text'];v=enc.encode([seed,seeds[1]['text']]);adapter=Adapter(enc,v,'mean_cosine');adapter.seed=seed
adapter.tokenizer.replacements={x.lower():x for x in __import__('re').findall(r'<[A-Z][A-Z0-9_]*>',seed)}
inputs=adapter.tokenizer(seed,return_tensors='pt').to('cuda');enc.tokenizer.model_max_length=inputs['input_ids'].shape[1]
flu=LocalAuthorFluency(stub,adapter.tokenizer)
gas.initialize_fluency_model=lambda *a,**kw:flu
ids,metrics=gas.gaslite_attack(adapter,inputs,slice(1,inputs['input_ids'].shape[1]-1),
    n_iter=1,n_grad=2,k_candidates=16,n_flips_to_gen=16,backprop_batch_size=2,
    beam_search_config={'perform':True,'n_cand':4,'B':1,'n_flip':2,'n_cand__sample_rate':0.5,'beam_indices':'random'},
    flu_alpha=.025,flu_alpha_inc={'factor':1,'freq':1,'filter_to_n_most_readable':300},
    fluency_model_name='gpt2',emb_targets=torch.as_tensor(v.mean(0,keepdims=True),device='cuda'))
check={'interface_smoke_passed':True,'real_author_checkpoint':False,'used_for_results':False,
       'test_opened':False,'gradient_calls':adapter.grad_calls,'output_shape':list(ids.shape),
       'peak_cuda_memory_mib':torch.cuda.max_memory_allocated()/1024**2}
(ROOT/'flu_interface_stub_check.json').write_text(json.dumps(check,indent=2));print(json.dumps(check),flush=True)
