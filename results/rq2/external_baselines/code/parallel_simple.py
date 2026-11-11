"""Run independent datasets concurrently; adopt the active ABCD worker."""
import json, os, shutil, subprocess, threading, time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
PY=ROOT/'venv/bin/python'
DATA=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data')
MODELS=Path('/root/autodl-tmp/liman/models')
refs=json.loads((ROOT/'existing_main_configs.json').read_text())
previous=json.loads((ROOT/'simple_existing_config_status.json').read_text())
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
state={'pid':os.getpid(),'started':time.time(),'mode':'parallel_datasets','jobs':[
    {'dataset':ds,'status':'queued','attempts':[]} for ds in ['abcd','dogo','banking77']]}
lock=threading.RLock()
def save():
    with lock:
        p=ROOT/'parallel_simple_status.json';t=p.with_suffix('.tmp')
        t.write_text(json.dumps(state,indent=2));t.replace(p)
def alive(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    try:return (p/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def run_job(job):
    ds=job['dataset'];ref=refs[ds]
    out=ROOT/'runs/simple_existing_config'/ds
    try:
        out.mkdir(parents=True,exist_ok=True)
        shared=ROOT/'runs/full'/ds
        for filename in ['train_vectors.npy','train_vectors.json']:
            if not (out/filename).exists():shutil.copy2(shared/filename,out/filename)
        if not (out/'targets').exists():shutil.copytree(shared/'targets',out/'targets')
        prior=ROOT/'runs/simple'/ds/'api_cache'
        if prior.exists() and not (out/'api_cache').exists():shutil.copytree(prior,out/'api_cache')
        cmd=[str(PY),'-u',str(ROOT/'code_existing_config/run_configurable.py'),
             '--comparison-mode','simple','--generation-cap-per-slot','0','--dataset',ds,
             '--representation','desensitized','--encoder','bge',
             '--data-dir',str(DATA/ds.replace('dogo','multidogo')/'desensitized'),
             '--model-dir',str(MODELS/'bge-small-en-v1.5'),
             '--lm-dir',str(MODELS/'Qwen2.5-1.5B-Instruct'),
             '--api-config','/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json',
             '--out-dir',str(out),'--device','cuda']
        for key in ['k','threshold','seed','batch_size','steps_per_slot','beam_width',
                    'positions_per_parent','tokens_per_position','rewrites_per_api','max_words_per_step',
                    'max_total_word_changes','prompt_examples','feedback_examples','focus_size',
                    'temperature','align_weight','max_ppl','relative_ppl','ppl_prefilter','swap_passes']:
            cmd+=['--'+key.replace('_','-'),str(ref[key])]
        adopted=next((j['child_pid'] for j in previous.get('jobs',[]) if j['dataset']==ds and j['status']=='running'),None)
        job.update(status='running',started=time.time(),out=str(out));save()
        if adopted and alive(adopted):
            command=(Path('/proc')/str(adopted)/'cmdline').read_bytes().decode()
            if 'code_existing_config/run_configurable.py' not in command or str(out) not in command:
                raise RuntimeError('Adopted worker identity differs')
            job.update(child_pid=adopted,adopted=True);save()
            while alive(adopted):time.sleep(15)
            job['attempts'].append({'adopted':True,'completed_report':(out/'final_report.json').exists()});save()
        for attempt in range(2):
            if (out/'final_report.json').exists():break
            with (ROOT/'logs'/f'parallel_simple_{ds}_attempt{attempt}.log').open('w') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
                job.update(child_pid=p.pid,adopted=False);save();rc=p.wait()
            job['attempts'].append({'attempt':attempt,'exit_code':rc,'ended':time.time()});save()
            if rc==0:break
            time.sleep(15)
        job.update(status='completed' if (out/'final_report.json').exists() else 'failed',ended=time.time());save()
    except Exception as exc:
        job.update(status='failed',error=str(exc),ended=time.time());save()
save()
threads=[threading.Thread(target=run_job,args=(j,)) for j in state['jobs']]
for t in threads:t.start()
for t in threads:t.join()
state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else 'has_failures',ended=time.time());save()
