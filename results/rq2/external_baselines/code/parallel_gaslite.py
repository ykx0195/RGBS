"""Two independent GASLITE dataset lanes; adopt running MultiDoGO process."""
import json,os,subprocess,threading,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
old=json.loads((ROOT/'gaslite_status.json').read_text())
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
state={'lane':'gaslite','mode':'parallel_by_dataset','pid':os.getpid(),'started':time.time(),
       'jobs':[j for j in old['jobs'] if j['dataset']=='abcd']}
lock=threading.RLock()
def save():
    with lock:
        p=ROOT/'gaslite_status.json';t=p.with_suffix('.tmp')
        t.write_text(json.dumps(state,indent=2));t.replace(p)
def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    try:return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def completed(ds,obj):
    p=ROOT/'runs/gaslite'/obj/ds/'final_report.json'
    if not p.exists():return False
    report=json.loads(p.read_text())
    return report.get('heldout_loaded') is True and 'test' in report
def lane(ds):
    for objective in ['mean_cosine','soft_threshold']:
        job={'dataset':ds,'objective':objective,'status':'queued','queued':time.time(),'attempts':[]}
        with lock:state['jobs'].append(job);save()
        try:
            if completed(ds,objective):job.update(status='completed',reused_completed=True);save();continue
            if not (ROOT/'runs/full'/ds/'targets/seed_selection.json').exists():
                raise RuntimeError('Shared initialization missing: '+ds)
            existing=next((j for j in old['jobs'] if j['dataset']==ds and j['objective']==objective and j['status']=='running'),None)
            job.update(status='running',started=time.time());save()
            if existing and alive(existing['child_pid']):
                pid=existing['child_pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode()
                if 'run_gaslite_adapter.py' not in cmd or ds not in cmd or objective not in cmd:
                    raise RuntimeError('Adopted worker identity differs')
                job.update(child_pid=pid,adopted=True);save()
                while alive(pid):time.sleep(15)
                job['attempts'].append({'adopted':True,'report_complete':completed(ds,objective)});save()
            for attempt in range(2):
                if completed(ds,objective):break
                cmd=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'run_gaslite_adapter.py'),'--dataset',ds,'--objective',objective]
                logpath=ROOT/'logs'/f'parallel_gaslite_{ds}_{objective}_attempt{attempt}.log'
                with logpath.open('w') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
                    job.update(child_pid=p.pid,adopted=False,log=str(logpath));save();rc=p.wait()
                job['attempts'].append({'attempt':attempt,'exit_code':rc,'ended':time.time()});save()
                if completed(ds,objective):break
                time.sleep(15)
            job.update(status='completed' if completed(ds,objective) else 'failed',ended=time.time());save()
        except Exception as exc:
            job.update(status='failed',error=str(exc),ended=time.time());save()
save()
threads=[threading.Thread(target=lane,args=(ds,)) for ds in ['dogo','banking77']]
for t in threads:t.start()
for t in threads:t.join()
state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else 'has_failures',ended=time.time());save()
