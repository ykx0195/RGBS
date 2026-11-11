import os,time,json,gc
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(1)
x=json.load(open('/root/autodl-tmp/rq1_requests_complete_20261104/input/benchmark_texts.json'))
paths={'minilm':'/root/autodl-tmp/ykx_crossmodel_abcd_20261026/models/minilm','gte':'/root/autodl-tmp/ykx_crossmodel_abcd_20261026/models/gte','e5':'/root/autodl-tmp/ykx_crossmodel_abcd_20261026/models/e5','bge':'/root/autodl-tmp/liman/models/bge-small-en-v1.5'}
r=[]
for key,path in paths.items():
 m=SentenceTransformer(path,device='cpu');m.max_seq_length=256;v=['query: '+s if key=='e5' else s for s in x]
 m.encode(v[:4],batch_size=4,show_progress_bar=False,normalize_embeddings=True)
 t=time.perf_counter();m.encode(v,batch_size=16,show_progress_bar=False,normalize_embeddings=True);elapsed=time.perf_counter()-t
 r.append({'model':key,'n':len(v),'seconds':elapsed,'texts_per_second':len(v)/elapsed});print(json.dumps(r[-1]),flush=True)
 json.dump(r,open('/root/autodl-tmp/rq1_requests_complete_20261104/benchmark.json','w'),indent=2)
 del m;gc.collect()
print('DONE',flush=True)
