"""Parallel cluster tasks; no Test access and no external API."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    p=Path(p);tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def alive(pid):
    try:
        st=(Path('/proc')/str(pid)/'status').read_text()
        return next(x for x in st.splitlines() if x.startswith('State:')).split()[1]!='Z'
    except (OSError,StopIteration):return False
def resources():
    mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    cg=Path('/sys/fs/cgroup')
    if (cg/'memory.max').exists() and (cg/'memory.max').read_text().strip()!='max':
        mem=min(mem,int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text()))
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]);usage={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
        a=line.split(',')
        if len(a)==2 and a[0].strip().isdigit() and a[1].strip().isdigit():usage[int(a[0])]=int(a[1])
    return mem//1048576,free,usage

def main():
    lock=(ROOT/'cluster_scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'experiment_plan.json').read_text());python=str(ROOT/'venv/bin/python');worker=str(ROOT/'code/corpus_hotflip_worker.py')
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    statuspath=ROOT/'queue_status.json';states=json.loads(statuspath.read_text()) if statuspath.exists() else {'created_at_utc':now(),'jobs':{}}
    states.update(status='smoke_test',api_calls=0,parallel_policy={'max_gpu_workers':4,'max_cpu_workers':2,'gpu_headroom_per_worker_mib':4000,'threads_per_worker':4})
    save(statuspath,states)
    # Pilot exercises the real gradient, placeholder-safe serialization and resume files.
    smoke=ROOT/'smoke/banking77/cluster_00/final_cluster.json'
    if not smoke.exists():
        cmd=[python,'-u',worker,'--dataset','banking77','--cluster','0','--device','cuda','--iterations','2','--smoke']
        with (ROOT/'smoke.log').open('a') as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        if rc:states.update(status='failed_smoke',exit_code=rc);save(statuspath,states);raise RuntimeError('HotFlip smoke failed; inspect smoke.log')
    smoke_data=json.loads(smoke.read_text());assert smoke_data['completed_iterations']==2 and smoke_data['api_calls']==0
    save(ROOT/'smoke_verified.json',{'passed':True,'completed_iterations':2,'gradient_and_forward_path_checked':True,
         'placeholder_preservation_checked':True,'heldout_loaded':False,'verified_at_utc':now()})
    print('HOTFLIP_SMOKE_PASSED_STARTING_PARALLEL_JOBS',flush=True)
    names=[(dataset,cluster) for cluster in range(max(plan['datasets'].values())) for dataset,k in plan['datasets'].items() if cluster<k]
    children={}
    for dataset,cluster in names:
        name=f'{dataset}_cluster{cluster:02d}';out=ROOT/'runs'/dataset/f'cluster_{cluster:02d}'
        states['jobs'].setdefault(name,{'status':'pending','dataset':dataset,'cluster':cluster,'out_dir':str(out)})
    # Adopt existing workers exactly, refusing duplicate task processes.
    adopted={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().decode().split('\0')
            if worker not in argv or '--smoke' in argv:continue
            dataset=argv[argv.index('--dataset')+1];cluster=int(argv[argv.index('--cluster')+1]);name=f'{dataset}_cluster{cluster:02d}'
            assert name not in adopted,'Duplicate cluster worker';adopted[name]=True
            states['jobs'][name].update(status='running',pid=int(p.name),device=argv[argv.index('--device')+1],argv=[x for x in argv if x])
        except (OSError,UnicodeError):pass
    states['status']='running';save(statuspath,states)
    while True:
        for name,job in states['jobs'].items():
            out=Path(job['out_dir']);final=out/'final_cluster.json'
            if final.exists():
                result=json.loads(final.read_text());job.update(status='complete',completed_iterations=result['completed_iterations'],
                    selected_at_iteration=result['selected']['selected_at_iteration'],active_seconds=result['active_seconds'])
                if name in children:children[name].poll()
            elif job['status']=='running' and not alive(job['pid']):
                rc=children[name].poll() if name in children else None
                job.update(status='failed',exit_code=rc,finished_at_utc=now());print('HOTFLIP_CLUSTER_FAILED',name,rc,flush=True)
        mem,free,usage=resources();running=[x for x in states['jobs'].values() if x['status']=='running']
        gpu=[x for x in running if x.get('device')=='cuda'];cpu=[x for x in running if x.get('device')=='cpu']
        reserved=sum(max(0,4000-usage.get(x['pid'],0)) for x in gpu)
        for name,job in states['jobs'].items():
            if job['status']!='pending':continue
            if mem<12000:break
            if len(gpu)<4 and free-reserved>=4000:device='cuda'
            elif len(cpu)<2:device='cpu'
            else:break
            out=Path(job['out_dir']);out.mkdir(parents=True,exist_ok=True)
            cmd=[python,'-u',worker,'--dataset',job['dataset'],'--cluster',str(job['cluster']),'--seed','42',
                 '--device',device,'--iterations',str(plan['iterations_per_cluster']),'--candidates',str(plan['gradient_candidates']),
                 '--candidate-batch',str(plan['candidate_encoding_batch_size'])]
            # Preserve a task's chosen device for exact checkpoint resumes.
            if (out/'config.json').exists():
                old=json.loads((out/'config.json').read_text());device=old['device'];cmd[cmd.index('--device')+1]=device
                if device=='cuda' and (len(gpu)>=4 or free-reserved<4000):continue
                if device=='cpu' and len(cpu)>=2:continue
            with (out/'run.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            children[name]=child;job.update(status='running',pid=child.pid,device=device,argv=cmd,started_at_utc=now())
            if device=='cuda':gpu.append(job);reserved+=4000
            else:cpu.append(job)
            mem-=3000;save(statuspath,states);print('HOTFLIP_CLUSTER_STARTED',name,child.pid,device,flush=True)
        states.update(updated_at_utc=now(),completed_cluster_count=sum(x['status']=='complete' for x in states['jobs'].values()),
                      expected_cluster_count=len(names),resource_snapshot={'ram_available_mib':mem,'gpu_free_mib':free})
        save(statuspath,states)
        if all(x['status'] in ['complete','failed'] for x in states['jobs'].values()):
            states['status']='complete' if all(x['status']=='complete' for x in states['jobs'].values()) else 'failed'
            states['finished_at_utc']=now();save(statuspath,states);print('HOTFLIP_SCHEDULER_FINISHED',states['status'],flush=True);break
        time.sleep(20)

if __name__=='__main__':main()
