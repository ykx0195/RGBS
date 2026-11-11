import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
state={'lane':'gaslite','pid':os.getpid(),'started':time.time(),'jobs':[]}
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
def save():
    p=ROOT/'gaslite_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
save()
for ds in ['abcd','dogo','banking77']:
    for objective in ['mean_cosine','soft_threshold']:
        job={'dataset':ds,'objective':objective,'status':'queued','queued':time.time()}
        state['jobs'].append(job);save()
        # Each dataset's common seed initialization is shared with the paired lane.
        while not (ROOT/'runs/full'/ds/'targets/seed_selection.json').exists():
            status_path=ROOT/'full_status.json'
            if status_path.exists():
                full=json.loads(status_path.read_text())
                matching=[j for j in full.get('jobs',[]) if j['dataset']==ds]
                if matching and matching[0]['status']=='failed':
                    raise RuntimeError('Missing shared initialization for failed full job '+ds)
            time.sleep(30)
        job.update(status='running',started=time.time());save()
        cmd=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'run_gaslite_adapter.py'),'--dataset',ds,'--objective',objective]
        with (ROOT/'logs'/f'gaslite_{ds}_{objective}.log').open('w') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            job['child_pid']=p.pid;save();rc=p.wait()
        job.update(status='completed' if rc==0 else 'failed',exit_code=rc,ended=time.time());save()
state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else 'has_failures',ended=time.time());save()
