from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import torch

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
from hybrid_search import Fluency
AREAS={'banking77':'banking support','abcd':'account access, manage account, order issue, product defect, purchase dispute, shipping issue, subscription inquiry, or site troubleshooting',
       'dogo':'airline, fast food, finance, insurance, media, or software support'}
torch.set_num_threads(4);torch.set_num_interop_threads(1)
cfg=json.loads((ROOT/'host_config.json').read_text());inputs=json.loads((ROOT/'quality_inputs.json').read_text())
device='cuda' if torch.cuda.is_available() else 'cpu';fluency=Fluency(Path(cfg['lm']),device);results=[];cache={};started=time.monotonic()
for i,x in enumerate(inputs['keys'],1):
    key=(x['dataset'],x['text'])
    if key not in cache:
        ppl=fluency.ppl(x['text']);readable=fluency.judge_readability([x['text']],AREAS[x['dataset']])[0]
        cache[key]={'ppl':ppl,'ppl_gate':ppl<=150,'local_readability_pass':readable,'quality_pass':ppl<=150 and readable}
    results.append({**x,**cache[key]});print('POSTHOC_QUALITY_PROGRESS',i,len(inputs['keys']),flush=True)
result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':inputs['evaluation_protocol'],'results':results,
        'device':device,'dtype':'float16' if device=='cuda' else 'float32','api_calls':0,'text_repaired':False,'elapsed_seconds':time.monotonic()-started,
        'limitation':'Automatic local judge only; no dual external API judgment or human review. Failure removes a key, never backfills it.'}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
result['quality_model_weight_sha256']={p.name:digest(p) for p in sorted(Path(cfg['lm']).glob('*.safetensors'))}
p=ROOT/'quality_results.json';temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
print('HOTFLIP_POSTHOC_QUALITY_COMPLETE',flush=True)
