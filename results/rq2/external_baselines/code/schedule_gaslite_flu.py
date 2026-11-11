"""Author-Flu queue: verify real assets, smoke, then memory-aware dataset jobs."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
PY=ROOT/'venv/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
state={'pid':os.getpid(),'phase':'waiting_assets','started':time.time(),'max_parallel':2,
       'fixed_settings':{'objective':'mean_cosine','flu_alpha':.025,'readable_token_filter':300,'iterations':100,
                         'n_grad':50,'candidate_tokens':256,'beam_candidates':128,'retained_K':{'abcd':5,'dogo':10,'banking77':15}},
       'jobs':[{'dataset':d,'status':'queued','attempts':[]} for d in ['abcd','dogo','banking77']]}
def save():
    p=ROOT/'gaslite_flu_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
def free_mib():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(s.strip().splitlines()[0])
def base_jobs():
    p=ROOT/'gaslite_status.json'
    if not p.exists():return 0
    return sum(j['status']=='running' for j in json.loads(p.read_text())['jobs'])
def report_exists(ds):return (ROOT/'runs/gaslite_flu/mean_cosine'/ds/'final_report.json').exists()
save()
while not (ROOT/'flu_assets/verified_assets.json').exists():
    p=ROOT/'flu_assets/status.json'
    if p.exists():state['asset_status']=json.loads(p.read_text());save()
    time.sleep(15)
state.update(phase='real_checkpoint_smoke');save()
cmd=[str(PY),'-u',str(ROOT/'run_gaslite_flu.py'),'--dataset','abcd','--objective','mean_cosine','--smoke']
with (ROOT/'logs/real_gaslite_flu_smoke.log').open('w') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    state['smoke_pid']=p.pid;save();rc=p.wait()
if rc or not (ROOT/'gaslite_flu_smoke/mean_cosine/abcd/slot_01.json').exists():
    state.update(phase='real_smoke_failed',smoke_exit_code=rc);save();raise SystemExit(2)
state.update(phase='scheduled',real_checkpoint_smoke_passed=True);save()
active={}
while True:
    for ds,(proc,log,job) in list(active.items()):
        rc=proc.poll()
        if rc is None:continue
        log.close();del active[ds]
        job['attempts'].append({'exit_code':rc,'ended':time.time()})
        if rc==0 and report_exists(ds):job.update(status='completed',ended=time.time())
        elif len(job['attempts'])<2:job.update(status='queued',last_exit_code=rc)
        else:job.update(status='failed',ended=time.time(),last_exit_code=rc)
        save()
    if all(j['status'] in ['completed','failed'] for j in state['jobs']):break
    free=free_mib();remaining_base=base_jobs()
    # Existing unconstrained workers later load a ~3GB quality model each.
    # Reserve that future growth plus 2.5GB for the new worker and 1GB margin.
    minimum=3500+3000*remaining_base
    state.update(free_gpu_mib=free,required_free_mib=minimum,existing_base_gpu_jobs=remaining_base)
    if len(active)<2 and free>=minimum:
        job=next((j for j in state['jobs'] if j['status']=='queued'),None)
        if job:
            ds=job['dataset']
            if report_exists(ds):job.update(status='completed',reused_completed=True);save();continue
            cmd=[str(PY),'-u',str(ROOT/'run_gaslite_flu.py'),'--dataset',ds,'--objective','mean_cosine']
            attempt=len(job['attempts']);logpath=ROOT/'logs'/f'gaslite_flu_{ds}_attempt{attempt}.log'
            log=logpath.open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            job.update(status='running',started=time.time(),child_pid=p.pid,log=str(logpath))
            active[ds]=(p,log,job);save()
            # Allow allocation to become visible before admitting a second job.
            time.sleep(20);continue
    state['phase']='running' if active else 'waiting_gpu_memory';save();time.sleep(15)
state.update(phase='complete' if all(j['status']=='completed' for j in state['jobs']) else 'has_failures',ended=time.time());save()
