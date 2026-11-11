from pathlib import Path
import sys,json,numpy as np,torch
base=Path("/root/autodl-tmp/ykx_32x50_20261030")
sys.path.insert(0,str(base/"code"))
from hybrid_search import BGE
model=Path("/root/autodl-tmp/ykx_v10_abcd_pair_t085_s42_20261019/models/bge-small-en-v1.5")
encoder=BGE(model,"cuda","bge")
output={}
for dataset in ("banking77","dogo","abcd"):
 folder="multidogo" if dataset=="dogo" else dataset
 run=base/"runs"/dataset
 slot_files=sorted(run.glob("slot_*.json"))
 k={"abcd":5,"banking77":15,"dogo":10}[dataset]
 if len(slot_files)<k:
  output[dataset]={"status":"incomplete","completed_slots":len(slot_files),"required_slots":k}
  continue
 test_path=base/"data"/folder/"desensitized"/"test.jsonl"
 tests=[json.loads(line)["text"] for line in test_path.open(encoding="utf-8") if line.strip()]
 tv=encoder.encode(tests,64)
 slots=[json.loads(p.read_text(encoding="utf-8")) for p in slot_files]
 cases={}
 for cutoff in (26,28,32):
  texts=[s["trace"][min(cutoff,len(s["trace"]))-1]["beam"][0] for s in slots]
  vectors=encoder.encode(texts,64)
  count=int(((tv@vectors.T).max(axis=1)>=0.85).sum())
  cases[str(cutoff)]={"covered":count,"total":len(tests),"coverage":count/len(tests),"selected_texts":texts,"early_stopped_slots":[s["slot"] for s in slots if len(s["trace"])<cutoff]}
 final_path=run/"final_report.json"
 final=json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
 output[dataset]={"status":"complete","interpretation":"truncate each completed 32-step slot trace at cutoff; later slots remain conditioned on 32-step predecessors; coordinate-beam selection only","cases":cases,"published_32x50_final":final["metrics"]["test"] if final else None,"published_32x50_selection":final["portfolio_source"] if final else None}
path=base/"analysis_step_cutoffs.json"
path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for d,v in output.items():
 print(d,v["status"],[(c,x["covered"],x["total"]) for c,x in v.get("cases",{}).items()],"actual",v.get("published_32x50_final"))
