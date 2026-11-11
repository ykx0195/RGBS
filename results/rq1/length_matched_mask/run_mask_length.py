from pathlib import Path
import os,json,time,hashlib,zipfile
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parent
MODELS=Path('/root/autodl-tmp/semantic_cache_redaction/models')
KEYS=['minilm','gte','e5','bge']
torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=ROOT/'outputs';out.mkdir(exist_ok=True)
inputs={p.stem:[json.loads(l) for l in p.open(encoding='utf-8')] for p in sorted((ROOT/'input').glob('*.jsonl'))}
fits={k:np.ones(len(v),dtype=bool) for k,v in inputs.items()}
audit={k:{'n_original_eval':len(v),'models':{}} for k,v in inputs.items()}
def fields(k):return ['source_text','pii_mask_length','control_mask_length'] if k.startswith('matched') else ['source_text','mask_length']
for model in KEYS:
 tok=AutoTokenizer.from_pretrained(MODELS/model,local_files_only=True)
 for name,rows in inputs.items():
  maximum=np.zeros(len(rows),dtype=int);stats={}
  for field in fields(name):
   texts=[('query: ' if model=='e5' else '')+r[field] for r in rows]
   lengths=[]
   for a in range(0,len(texts),128):lengths.extend(map(len,tok(texts[a:a+128],padding=False,truncation=False,add_special_tokens=True)['input_ids']))
   maximum=np.maximum(maximum,lengths);stats[field]={'max_tokens':int(max(lengths)),'over256':sum(n>256 for n in lengths)}
  fits[name]&=maximum<=256;audit[name]['models'][model]=stats
 print('TOKEN_AUDIT',model,flush=True)
for name,fit in fits.items():
 np.save(out/f'{name}_retained_positions.npy',np.flatnonzero(fit));audit[name]['n_retained_eval']=int(fit.sum());audit[name]['excluded_ids']=[r['record_id'] for r,b in zip(inputs[name],fit) if not b]
 print('COHORT',name,len(fit),int(fit.sum()),flush=True)
(out/'token_audit.json').write_text(json.dumps(audit,indent=2))
meta={'cuda_device':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,'token_limit':256,'precision':'float32','normalize':True,'TF32':False,'model_files':{},'started_utc':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())}
start=time.time()
for model in KEYS:
 modelpath=MODELS/model
 meta['model_files'][model]={str(p.relative_to(modelpath)):hashlib.sha256(p.read_bytes()).hexdigest() for p in modelpath.glob('*') if p.name in ['model.safetensors','pytorch_model.bin','config.json','tokenizer.json','modules.json']}
 encoder=SentenceTransformer(str(modelpath),device='cuda');encoder.max_seq_length=256
 for name,rows in inputs.items():
  kept=[r for r,fit in zip(rows,fits[name]) if fit]
  for field in fields(name):
   texts=[('query: ' if model=='e5' else '')+r[field] for r in kept]
   emb=encoder.encode(texts,batch_size=96,show_progress_bar=False,normalize_embeddings=True,convert_to_numpy=True,precision='float32')
   assert emb.shape==(len(kept),384) and np.isfinite(emb).all()
   np.save(out/f'{name}_{model}_{field}.npy',emb)
   print('ENCODED',name,model,field,len(kept),'elapsed',round(time.time()-start,1),flush=True)
 del encoder;torch.cuda.empty_cache()
meta['seconds']=time.time()-start
(out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
(out/'sha256.json').write_text(json.dumps(hashes,indent=2))
with zipfile.ZipFile(ROOT/'outputs.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in out.iterdir():z.write(p,p.relative_to(ROOT))
print('FINISHED',round(time.time()-start,1),'seconds',flush=True)
