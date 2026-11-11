from pathlib import Path
from datetime import datetime,timezone
import getpass,hashlib,json,paramiko,shlex,shutil,sys

sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
base=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031';root='/root/autodl-tmp/corpus_hotflip_baseline_20261031'
password=getpass.getpass('SSH password for 35564 (not saved): ')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('connect.nmb2.seetacloud.com',port=35564,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);password='';c.get_transport().set_keepalive(20)
def execute(cmd):
    _,o,e=c.exec_command(cmd,timeout=180);data=o.read().decode('utf-8','replace');rc=o.channel.recv_exit_status()
    if rc:raise RuntimeError(e.read().decode('utf-8','replace')[-2000:])
    return data.strip()
prepare=r'''from pathlib import Path
from datetime import datetime,timezone
import os,signal,json,time,shutil,hashlib
root=Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031');worker=str(root/'code/corpus_hotflip_worker.py');scheduler=str(root/'corpus_hotflip_scheduler.py')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=root/'device_migrations'/stamp;backup.mkdir(parents=True,exist_ok=True)
def argv_of(p):
 try:return (Path('/proc')/str(p)/'cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeError):return []
parents=[];cpu=[];preserved=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 argv=argv_of(p.name)
 if scheduler in argv:parents.append(int(p.name))
 if worker in argv and '--smoke' not in argv:
  if argv[argv.index('--device')+1]=='cpu':cpu.append((int(p.name),argv))
  else:preserved.append(int(p.name))
for pid in parents:os.kill(pid,signal.SIGTERM)
time.sleep(.6)
shutil.copyfile(root/'queue_status.json',backup/'queue_status_before.json')
if (root/'experiment_plan.json').exists():shutil.copyfile(root/'experiment_plan.json',backup/'experiment_plan_before.json')
if (root/'corpus_hotflip_scheduler.py').exists():shutil.copyfile(root/'corpus_hotflip_scheduler.py',backup/'scheduler_before.py')
for pid,argv in cpu:
 current=argv_of(pid)
 assert worker in current and current[current.index('--device')+1]=='cpu'
 os.kill(pid,signal.SIGTERM)
for _ in range(50):
 if all(worker not in argv_of(pid) for pid,_ in cpu):break
 time.sleep(.1)
assert all(worker not in argv_of(pid) for pid,_ in cpu),'CPU worker did not stop'
status=json.loads((root/'queue_status.json').read_text());migration=[]
for pid,argv in cpu:
 dataset=argv[argv.index('--dataset')+1];cluster=int(argv[argv.index('--cluster')+1]);name=f'{dataset}_cluster{cluster:02d}'
 out=root/'runs'/dataset/f'cluster_{cluster:02d}'
 assert out.resolve().is_relative_to((root/'runs').resolve())
 if (out/'final_cluster.json').exists():continue
 shutil.copytree(out,backup/name)
 config=json.loads((out/'config.json').read_text());state=json.loads((out/'state.json').read_text())
 assert config==state['config'] and config['device']=='cpu'
 assert config['search_code_sha256']==hashlib.sha256(Path(worker).read_bytes()).hexdigest()
 migration.append({'job':name,'dataset':dataset,'cluster':cluster,'source_pid':pid,'out_dir':str(out),
                   'backup_dir':str(backup/name),'completed_iterations':state['completed_iterations'],
                   'source_config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
                   'source_state_sha256':hashlib.sha256((out/'state.json').read_bytes()).hexdigest(),
                   'source_active_seconds':state['active_seconds'],'source_device':'cpu','target_device':'cuda',
                   'reason':'User requests GPU-only queuing; resume last complete atomic checkpoint; unsaved iterations are rerun.',
                   'migrated_at_utc':datetime.now(timezone.utc).isoformat()})
manifest={'migrations':migration,'preserved_gpu_worker_pids':preserved,'parent_pids_replaced':parents,'backup_dir':str(backup)}
(root/'pending_gpu_migrations.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
'''
result=json.loads(execute('/root/miniconda3/bin/python -c '+shlex.quote(prepare)))
print('CPU_WORKERS_STOPPED_GPU_WORKERS_PRESERVED',json.dumps(result),flush=True)
# Rebase saved selection scores on the destination device; all text and RNG state
# remain fixed. The unchanged worker's strict config check then validates resume.
rebase=r'''from pathlib import Path
import json,sys,numpy as np,torch
root=Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031');sys.path.insert(0,str(root/'code'))
from corpus_hotflip_worker import Adapter
torch.set_num_threads(4);manifest=json.loads((root/'pending_gpu_migrations.json').read_text())
model=None
for item in manifest['migrations']:
 if model is None:model=Adapter(Path(json.loads((root/'host_config.json').read_text())['model']),'cuda')
 out=Path(item['out_dir']);state=json.loads((out/'state.json').read_text());config=json.loads((out/'config.json').read_text())
 inp=root/'inputs'/item['dataset'];train=np.load(inp/'train_vectors.npy');labels=np.load(inp/'cluster_labels.npy')
 queries=torch.as_tensor(train[labels==item['cluster']],device=model.device);mean=queries.mean(dim=0)
 old_best_score=state['best']['full_cluster_mean_cosine'];rebased=float(model.encode([state['best']['text']])[0]@mean)
 state['best']['full_cluster_mean_cosine']=rebased;config['device']='cuda';state['config']=config
 item.update(saved_best_cpu_score=old_best_score,rebased_best_cuda_score=rebased,
             ranking_rebase_abs_difference=abs(old_best_score-rebased),text_and_rng_unchanged=True,
             source_device_history_remains_part_of_experiment=True)
 for filename,data in [('config.json',config),('state.json',state),('device_migration.json',item)]:
  p=out/filename;temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
 status=json.loads((root/'queue_status.json').read_text());job=status['jobs'][item['job']]
 job.update(status='pending',device='cuda',last_cpu_completed_iterations=item['completed_iterations'],device_migration_record=str(out/'device_migration.json'))
 job.pop('pid',None)
 if 'argv' in job:job['argv'][job['argv'].index('--device')+1]='cuda'
 status['status']='running';status.pop('finished_at_utc',None)
 p=root/'queue_status.json';temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(p)
manifest['gpu_only_policy']=True
(root/'gpu_migration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan=json.loads((root/'experiment_plan.json').read_text());plan['parallel_policy']='GPU only, up to four simultaneous cluster workers; pending tasks wait in the GPU queue.'
plan['cpu_to_gpu_migration']='Two earlier CPU workers resume from preserved checkpoints; per-task CPU history and GPU score rebasing are recorded separately. Not all-GPU-from-start for migrated clusters.'
(root/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest))
'''
manifest=json.loads(execute(shlex.quote(root+'/venv/bin/python')+' -c '+shlex.quote(rebase)))
print('CPU_CHECKPOINTS_REBASED_AND_QUEUED_FOR_GPU',json.dumps(manifest),flush=True)
s=c.open_sftp();s.put(str(Path(__file__).parent/'corpus_hotflip_scheduler.py'),root+'/corpus_hotflip_scheduler.py')
for name in ['gpu_migration_manifest.json','experiment_plan.json','queue_status.json']:
    s.get(root+'/'+name,str(base/name))
for item in manifest['migrations']:
    dest=base/'runs'/item['dataset']/f"cluster_{item['cluster']:02d}";dest.mkdir(parents=True,exist_ok=True)
    for name in ['device_migration.json','config.json','state.json']:s.get(item['out_dir']+'/'+name,str(dest/name))
s.close()
shutil.copyfile(Path(__file__).parent/'corpus_hotflip_scheduler.py',base/'corpus_hotflip_scheduler.py')
shutil.copyfile(Path(__file__).parent/'corpus_hotflip_scheduler.py',base/'control_code/corpus_hotflip_scheduler.py')
shutil.copyfile(Path(__file__),base/'control_code/switch_hotflip_gpu_only.py')
cmd='nohup '+shlex.quote(root+'/venv/bin/python')+' -u '+shlex.quote(root+'/corpus_hotflip_scheduler.py')+' >> '+shlex.quote(root+'/scheduler.log')+' 2>&1 < /dev/null & echo $!'
print('GPU_ONLY_SCHEDULER_STARTED',execute('bash -lc '+shlex.quote(cmd)),flush=True)
verify=r'''from pathlib import Path
import json,time
root=Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031');worker=str(root/'code/corpus_hotflip_worker.py')
time.sleep(2);running=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().decode().split('\0')
  if worker in argv and '--smoke' not in argv:
   device=argv[argv.index('--device')+1];assert device=='cuda'
   running.append({'pid':int(p.name),'dataset':argv[argv.index('--dataset')+1],'cluster':int(argv[argv.index('--cluster')+1]),'device':device})
 except (OSError,UnicodeError):pass
status=json.loads((root/'queue_status.json').read_text());assert status['parallel_policy']['max_cpu_workers']==0
print(json.dumps({'verified_gpu_only':True,'active_workers':running,'pending_gpu_jobs':sum(x['status']=='pending' for x in status['jobs'].values()),'policy':status['parallel_policy']}))
'''
verified=json.loads(execute('/root/miniconda3/bin/python -c '+shlex.quote(verify)))
(base/'gpu_only_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
print('GPU_ONLY_VERIFIED',json.dumps(verified),flush=True);c.close()
