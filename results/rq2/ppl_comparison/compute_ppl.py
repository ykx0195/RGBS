from pathlib import Path
import os,json,csv,hashlib,time,math,re,importlib.util
import numpy as np
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(r'C:\Users\ykx01\Desktop\results');OUT=Path(__file__).resolve().parent
prior=ROOT/'原始问句与生成句_PPL对比_20261031';provenance=json.loads((prior/'sample_n200_delex_provenance.json').read_text(encoding='utf-8'))
# Reuse the exact normalization/loss implementation that scored the fixed natural samples.
spec=importlib.util.spec_from_file_location('prior_scorer',prior/'score_ppl.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
inputs=[];sources={};scores=[]
for ds in ['banking77','abcd','dogo']:
 for filename,method in [('CorpusPoisoning_HotFlip对比_20261031/困惑度单指标重算/逐句PPL判定.csv','Corpus poisoning'),('AdversarialDecoding对比_20261031/evaluation/advdec_sentences_and_ppl.csv','Adversarial Decoding')]:
  p=ROOT/filename;sources[str(p)]=sc.sha(p)
  rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
  rows=[r for r in rows if r['dataset']==ds and (r.get('method')=='CorpusPoisoning_adapted' if method=='Corpus poisoning' else True)]
  assert len(rows)=={'banking77':15,'abcd':5,'dogo':10}[ds]
  for i,r in enumerate(rows):inputs.append({'dataset':ds,'method':method,'split':'generated','representation':'deidentified','source_id':str(i),'variant_id':'','text':r['text']})
 p=ROOT/'VLDB补充方法对比_20261102'/'runs'/'gaslite_flu_32'/'mean_cosine'/ds/'final_report.json';sources[str(p)]=sc.sha(p)
 for i,r in enumerate(json.loads(p.read_text(encoding='utf-8'))['raw_selected']):inputs.append({'dataset':ds,'method':'GASLITE-Flu','split':'generated','representation':'deidentified','source_id':str(i),'variant_id':'','text':r['text']})
 if ds!='dogo':
  p=ROOT/'三数据集_脱敏_32x50_20261030'/'runs'/ds/'final_report.json';sources[str(p)]=sc.sha(p)
  raw=json.loads(p.read_text(encoding='utf-8'))['selected'];texts=[r['text'] for r in raw]
 else:
  p=ROOT/'MultiDoGO_8309_模块与重选_CPU_20261103'/'train_frozen_selections.json';sources[str(p)]=sc.sha(p)
  texts=json.loads(p.read_text(encoding='utf-8'))['variants']['full_main']
 for i,text in enumerate(texts):inputs.append({'dataset':ds,'method':'RKGB','split':'generated','representation':'deidentified','source_id':str(i),'variant_id':'','text':text})
assert len(inputs)==120
natural_path=prior/'sample_n200_delex_per_text.csv';sources[str(natural_path)]=sc.sha(natural_path)
natural=[r for r in csv.DictReader(natural_path.open(encoding='utf-8-sig')) if r['split']=='train' and r['representation']=='delex']
assert len(natural)==600
# Verify frozen source data have not changed since the original natural scoring.
for ds in ['banking77','abcd','dogo']:
 old=provenance['inputs'][ds]['train'];assert sc.sha(Path(old['delex_path']))==old['delex_sha256']
(OUT/'input_texts.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
torch.set_num_threads(6)
model_path=Path(provenance['model']['path']);assert sc.sha(model_path/'model.safetensors')==provenance['model']['weights_sha256']
print('Loading local scorer; 120 generated texts; 600 existing natural Train scores.',flush=True)
tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32).eval()
start=time.time()
for a in range(0,len(inputs),8):
 batch=inputs[a:a+8];scored=sc.score_batch(model,tok,batch)
 for r,original in zip(scored,batch):r['method']=original['method'];r['text']=original['text'];scores.append(r)
 print('Scored',len(scores),'/120; elapsed',round(time.time()-start,1),'s',flush=True)
 sc.save_csv(OUT/'generated_per_text.csv',scores)
summary=[]
for ds in ['banking77','abcd','dogo']:
 nat=np.array([float(r['ppl']) for r in natural if r['dataset']==ds]);assert len(nat)==200
 for method in ['Corpus poisoning','Adversarial Decoding','GASLITE-Flu','RKGB']:
  val=np.array([float(r['ppl']) for r in scores if r['dataset']==ds and r['method']==method])
  summary.append({'dataset':ds,'method':method,'natural_n':len(nat),'natural_median':float(np.median(nat)),'natural_q1':float(np.quantile(nat,.25)),'natural_q3':float(np.quantile(nat,.75)),'natural_q95':float(np.quantile(nat,.95)),'natural_le150_count':int((nat<=150).sum()),'generated_n':len(val),'generated_median':float(np.median(val)),'generated_q1':float(np.quantile(val,.25)),'generated_q3':float(np.quantile(val,.75)),'generated_mean':float(val.mean()),'generated_le150_count':int((val<=150).sum())})
sc.save_csv(OUT/'ppl_summary.csv',summary)
(OUT/'provenance.json').write_text(json.dumps({'natural_sample':'200 deidentified Train requests per dataset; fixed existing seed 20261031','generated':'all raw retained texts in Table III; current 83.09% MultiDoGO RKGB portfolio','model':provenance['model'],'definition':provenance['ppl_definition'],'precision':'CPU float32; batch8; same scorer implementation as natural sample','sources':sources,'seconds':time.time()-start,'post_hoc':True,'threshold_not_retuned':150},ensure_ascii=False,indent=2),encoding='utf-8')
print('FINISHED',flush=True)
for r in summary:print(r['dataset'],r['method'],'Natural',round(r['natural_median'],2),'Generated',round(r['generated_median'],2),'pass',r['generated_le150_count'],r['generated_n'],flush=True)
