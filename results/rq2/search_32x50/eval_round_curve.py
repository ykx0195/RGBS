from pathlib import Path
import sys,json,numpy as np
base=Path("/root/autodl-tmp/ykx_32x50_20261030")
sys.path.insert(0,str(base/"code"))
from hybrid_search import BGE
model=Path("/root/autodl-tmp/ykx_v10_abcd_pair_t085_s42_20261019/models/bge-small-en-v1.5")
encoder=BGE(model,"cuda","bge")
output={}
for dataset in ("abcd","banking77","dogo"):
 run=base/"runs"/dataset
 k={"abcd":5,"banking77":15,"dogo":10}[dataset]
 files=sorted(run.glob("slot_*.json"))
 if len(files)<k:
  output[dataset]={"status":"incomplete","completed_slots":len(files),"required_slots":k}
  continue
 slots=[json.loads(p.read_text(encoding="utf-8")) for p in files]
 folder="multidogo" if dataset=="dogo" else dataset
 path=base/"data"/folder/"desensitized"/"test.jsonl"
 test=[json.loads(line)["text"] for line in path.open(encoding="utf-8") if line.strip()]
 tv=encoder.encode(test,64)
 def chosen(s,rounds):
  return s["starting_text"] if rounds==0 else s["trace"][min(rounds,len(s["trace"]))-1]["beam"][0]
 texts=list(dict.fromkeys(chosen(s,r) for r in range(33) for s in slots))
 sv=encoder.encode(texts,64)
 mask={text:(tv@vec>=0.85) for text,vec in zip(texts,sv)}
 curve=[]
 for r in range(33):
  selected=[chosen(s,r) for s in slots]
  covered=int(np.logical_or.reduce([mask[t] for t in selected]).sum())
  curve.append({"round":r,"covered":covered,"total":len(test),"coverage":covered/len(test)})
 final_path=run/"final_report.json"
 final=json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
 output[dataset]={"status":"complete","type":"fixed_32_step_trajectory_coordinate_beam","points":curve,"actual_32_step_final":final["metrics"]["test"] if final else None,"actual_32_step_selection":final["portfolio_source"] if final else None,"early_stopped_slots":{str(s["slot"]):len(s["trace"]) for s in slots if len(s["trace"])<32}}
result=base/"test_coverage_by_round.json"
result.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for dataset,x in output.items():
 print(dataset,x["status"],[(r,x["points"][r]["covered"]) for r in (0,1,10,20,26,28,32)] if x["status"]=="complete" else x)
