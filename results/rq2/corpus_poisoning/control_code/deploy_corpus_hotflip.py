from pathlib import Path
import getpass,hashlib,json,paramiko,shlex,sys,time
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
base=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031'
remote='/root/autodl-tmp/corpus_hotflip_baseline_20261031'
password=getpass.getpass('SSH password for 35564 (not saved): ')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('connect.nmb2.seetacloud.com',port=35564,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);password='';c.get_transport().set_keepalive(20)
def execute(cmd):
    _,o,e=c.exec_command(cmd,timeout=180);v=o.read().decode('utf-8','replace');rc=o.channel.recv_exit_status()
    if rc:raise RuntimeError(e.read().decode('utf-8','replace')[-1800:])
    return v.strip()
active=r'''from pathlib import Path
import json
root=Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031')
targets={str(root/'corpus_hotflip_scheduler.py'),str(root/'code/corpus_hotflip_worker.py')}
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=set((p/'cmdline').read_bytes().decode().split('\0'))
  if targets&argv:found.append(p.name)
 except (OSError,UnicodeError):pass
print(json.dumps(found))
'''
found=json.loads(execute('/root/miniconda3/bin/python -c '+shlex.quote(active)))
if found:print('ACTIVE_HOTFLIP_SEARCHES_NO_OVERWRITE',found);c.close();raise SystemExit()
for name in ['corpus_hotflip_worker.py']:
    import shutil
    shutil.copyfile(Path(__file__).parent/name,base/'code'/name)
for name in ['corpus_hotflip_scheduler.py']:
    import shutil
    shutil.copyfile(Path(__file__).parent/name,base/name)
config={'port':35564,'model':'/root/autodl-tmp/liman/models/bge-small-en-v1.5','lm':'/root/autodl-tmp/liman/models/Qwen2.5-1.5B-Instruct','root':remote}
(base/'host_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
s=c.open_sftp();files=[]
for folder in ['inputs','official_source','code']:
    files.extend(p for p in (base/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(base/n for n in ['experiment_plan.json','host_config.json','corpus_hotflip_scheduler.py'])
for p in files:
    rel=p.relative_to(base).as_posix();dest=remote+'/'+rel
    execute('mkdir -p '+shlex.quote(dest.rsplit('/',1)[0]))
    s.put(str(p),dest)
manifest={p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(base/'deployment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');s.put(str(base/'deployment_manifest.json'),remote+'/deployment_manifest.json');s.close()
print('HOTFLIP_INPUTS_AND_CODE_UPLOADED',len(files),flush=True)
ready=False
for _ in range(90):
    probe=r'''import importlib.util
print(all(importlib.util.find_spec(x) is not None for x in ['torch','transformers','sklearn','numpy']))'''
    if execute(shlex.quote(remote+'/venv/bin/python')+' -c '+shlex.quote(probe))=='True':ready=True;break
    time.sleep(10)
if not ready:raise RuntimeError('Isolated environment not ready; input files retained')
verify=r'''from pathlib import Path
import hashlib,json,numpy,torch,transformers,sklearn
root=Path('/root/autodl-tmp/corpus_hotflip_baseline_20261031')
for rel,digest in json.loads((root/'deployment_manifest.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
cfg=json.loads((root/'host_config.json').read_text());expected=json.loads((root/'inputs/banking77/train_vectors_identity.json').read_text())
assert hashlib.sha256((Path(cfg['model'])/'model.safetensors').read_bytes()).hexdigest()==expected['model_sha256']
env={'python':__import__('sys').version,'torch':torch.__version__,'numpy':numpy.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0)}
(root/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print(json.dumps(env))
'''
print('REMOTE_HOTFLIP_PREFLIGHT',execute(shlex.quote(remote+'/venv/bin/python')+' -c '+shlex.quote(verify)),flush=True)
cmd='nohup '+shlex.quote(remote+'/venv/bin/python')+' -u '+shlex.quote(remote+'/corpus_hotflip_scheduler.py')+' > '+shlex.quote(remote+'/scheduler.log')+' 2>&1 < /dev/null & echo $!'
print('PARALLEL_HOTFLIP_SCHEDULER_LAUNCHED',execute('bash -lc '+shlex.quote(cmd)),flush=True);c.close()
