import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
PY=ROOT/'venv/bin/python'
DATA=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data')
MODELS=Path('/root/autodl-tmp/liman/models')
mode=sys.argv[1]
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
state={'lane':mode,'pid':os.getpid(),'started':time.time(),'jobs':[]}
def save():
    p=ROOT/f'{mode}_status.json'
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(p)
save()
for dataset in ['abcd','dogo','banking77']:
    out=ROOT/'runs'/mode/dataset
    cmd=[str(PY),'-u',str(ROOT/'code/run_configurable.py'),'--comparison-mode',mode,
         '--generation-cap-per-slot','256','--dataset',dataset,'--representation','desensitized',
         '--encoder','bge','--data-dir',str(DATA/dataset.replace('dogo','multidogo')/'desensitized'),
         '--model-dir',str(MODELS/'bge-small-en-v1.5'),'--lm-dir',str(MODELS/'Qwen2.5-1.5B-Instruct'),
         '--api-config','/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json',
         '--out-dir',str(out),'--threshold','0.85','--seed','42','--device','cuda',
         '--steps-per-slot','32','--beam-width','50']
    job={'dataset':dataset,'out':str(out),'started':time.time(),'status':'running','attempts':[]}
    state['jobs'].append(job);save()
    for attempt in range(2):
        with (ROOT/'logs'/f'{mode}_{dataset}_attempt{attempt}.log').open('w') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            job['child_pid']=p.pid;save()
            rc=p.wait()
        job['attempts'].append({'attempt':attempt,'exit_code':rc,'ended':time.time()});save()
        if rc==0:break
        time.sleep(15)
    job.update(status='completed' if rc==0 and (out/'final_report.json').exists() else 'failed',ended=time.time())
    save()
state['ended']=time.time();state['status']='completed' if all(j['status']=='completed' for j in state['jobs']) else 'has_failures';save()
