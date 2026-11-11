from pathlib import Path
from collections import Counter
import ast,importlib.util,json,sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8',line_buffering=True);torch.set_num_threads(4)
root=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031';sys.path.insert(0,str(root/'code'))
spec=importlib.util.spec_from_file_location('adapted_worker',root/'code/corpus_hotflip_worker.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=(root/'official_source/src/attack_poison.py').read_text(encoding='utf-8');extracted=(root/'code/official_hotflip_core.py').read_text(encoding='utf-8')
def function_ast(text):return ast.dump(next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='hotflip_attack'),include_attributes=False)
assert function_ast(original)==function_ast(extracted)
model=module.Adapter(Path.home()/'Desktop/ykx/bge-small-en-v1.5','cpu');results=[]
for dataset in ['banking77','abcd','dogo']:
    inp=root/'inputs'/dataset;seeds=json.loads((inp/'seed_selection.json').read_text(encoding='utf-8'))['selected'];train=np.load(inp/'train_vectors.npy');labels=np.load(inp/'cluster_labels.npy')
    seed=next((x for x in seeds if module.PLACEHOLDER.search(x['text'])),seeds[0]);cluster=seed['cluster_id'];target=torch.tensor(train[labels==cluster].mean(axis=0))
    ids,grad,ranges,editable,vector=model.gradient(seed['text'],target);encoded=model.encode([seed['text']])[0]
    assert torch.max(abs(vector-encoded))<2e-6
    delta=float(torch.max(abs(encoded-torch.tensor(train[seed['train_index']]))));assert delta<2e-4
    assert editable and all(i not in {j for a,b,_ in ranges for j in range(a,b)} for i in editable)
    position=editable[len(editable)//2];cand=module.hotflip_attack(grad[position],model.embedding.weight,increase_loss=True,num_candidates=100)
    assert torch.equal(cand,(model.embedding.weight.detach()@grad[position]).topk(100).indices)
    structural=0
    for token in cand.tolist():
        if token in model.tokenizer.all_special_ids:continue
        changed=list(ids);changed[position]=token;text=model.render(changed,ranges)
        assert Counter(module.PLACEHOLDER.findall(text))==Counter(module.PLACEHOLDER.findall(seed['text']))
        structural+=1
    # Compare autograd with a finite directional derivative through the encoder.
    f=model.tokenizer(seed['text'],return_tensors='pt');embedded=model.embedding(f['input_ids']).detach();direction=torch.zeros_like(embedded)
    direction[0,position]=grad[position]/torch.linalg.norm(grad[position]);eps=.002
    kwargs={k:v for k,v in f.items() if k!='input_ids'}
    with torch.no_grad():
        plus=torch.nn.functional.normalize(model.model(inputs_embeds=embedded+eps*direction,**kwargs).last_hidden_state[:,0],p=2,dim=-1)[0]@target
        minus=torch.nn.functional.normalize(model.model(inputs_embeds=embedded-eps*direction,**kwargs).last_hidden_state[:,0],p=2,dim=-1)[0]@target
    finite=float((plus-minus)/(2*eps));analytic=float(torch.linalg.norm(grad[position]));rel=abs(finite-analytic)/max(abs(analytic),1e-7)
    assert rel<.05,(dataset,finite,analytic,rel)
    results.append({'dataset':dataset,'initial_cache_max_abs_delta':delta,'forward_gradient_paths_agree':True,
                    'placeholder_preserving_candidate_checks':structural,'finite_difference_relative_error':rel})
(root/'adapter_local_validation.json').write_text(json.dumps({'passed':True,'official_core_ast_identical':True,'test_data_loaded':False,'api_calls':0,'checks':results},indent=2)+'\n',encoding='utf-8')
print('HOTFLIP_ADAPTER_LOCAL_VALIDATION_PASSED',json.dumps(results),flush=True)
