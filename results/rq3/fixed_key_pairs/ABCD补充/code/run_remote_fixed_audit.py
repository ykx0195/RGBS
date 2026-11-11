"""Use existing server-side API configuration without copying credentials."""
from pathlib import Path
import argparse, getpass, json, shlex, sys, time
import paramiko


def main():
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    p=argparse.ArgumentParser()
    p.add_argument('--local',type=Path,required=True)
    p.add_argument('--script',type=Path,required=True)
    p.add_argument('--host',default='connect.nmb2.seetacloud.com')
    p.add_argument('--port',type=int,default=49163)
    p.add_argument('--api-config')
    p.add_argument('--restart-own-audit',action='store_true')
    p.add_argument('--datasets',nargs='+',choices=['banking77','dogo','abcd'],default=['banking77','dogo'])
    p.add_argument('--remote-root',default='/root/autodl-tmp/fixed_query_pair_audit_20261031')
    args=p.parse_args()
    password=getpass.getpass('SSH password (not saved): ')
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect(args.host,port=args.port,username='root',password=password,
                   look_for_keys=False,allow_agent=False,timeout=15)
    password=''
    root=args.remote_root
    if args.restart_own_audit and root!='/root/autodl-tmp/fixed_query_pair_audit_20261031':
        raise ValueError('Use separate output directories for new datasets; restart flag only supports the original audit')
    config='/root/autodl-tmp/banking77_global_rebuild_k15_20261017/api_providers.json'
    def execute(command):
        _,stdout,stderr=client.exec_command(command,timeout=60)
        data=stdout.read().decode('utf-8','replace')
        code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError('Remote preflight failed: '+stderr.read().decode('utf-8','replace')[-1000:])
        return data
    print('SSH_CONNECTED',flush=True)
    if args.restart_own_audit:
        stop_code='''import os,signal\nfrom pathlib import Path\ntarget="/root/autodl-tmp/fixed_query_pair_audit_20261031/remote_fixed_task_audit.py"\nfor p in Path("/proc").iterdir():\n if not p.name.isdigit():continue\n try:\n  argv=(p/"cmdline").read_bytes().decode().split("\\0")\n  if target in argv:\n   os.kill(int(p.name),signal.SIGTERM);print("STOPPED_OWN_AUDIT",p.name)\n except (OSError,UnicodeError):pass\n'''
        print(execute('/root/miniconda3/bin/python -c '+shlex.quote(stop_code)),flush=True)
    # Only a file-existence probe, never a credential readback.
    discovery=r'''
import os,json
from pathlib import Path
found=[]
root=Path('/root/autodl-tmp')
for base,dirs,files in os.walk(root):
    rel=Path(base).relative_to(root)
    dirs[:]=[d for d in dirs if d not in ['api_cache','judge_cache','models','node_modules','.git','__pycache__'] and len(rel.parts)<6]
    for name in files:
        if not name.endswith('.json') or not any(s in name.lower() for s in ['config','provider']):continue
        p=Path(base)/name
        try:
            if p.stat().st_size>500000:continue
            d=json.loads(p.read_text())
            if not isinstance(d,dict):continue
            specs=d.get('api_generators') or d.get('language',{}).get('api_generators')
            if not isinstance(specs,list) or len(specs)<2:continue
            ready=True
            for s in specs[:2]:
                key=next((os.environ.get(n,'').strip() for n in s.get('key_envs',[]) if os.environ.get(n,'').strip()),'')
                if not key and s.get('key_file'):
                    k=Path(s['key_file']);k=k if k.is_absolute() else p.parent/k
                    key=k.read_text().strip() if k.is_file() else ''
                if not key:key=s.get('api_key') or s.get('key') or ''
                ready=ready and bool(key)
            found.append({'path':str(p),'ready':ready,'providers':[{'name':s.get('name'),'model':s.get('model')} for s in specs[:2]]})
        except Exception:pass
print(json.dumps(found))
'''
    if args.api_config:
        config=args.api_config
    else:
        candidates=json.loads(execute('/root/miniconda3/bin/python -c '+shlex.quote(discovery)))
        ready=[x for x in candidates if x['ready']]
        ready.sort(key=lambda x:not any(p['model']=='qwen3.8-flash' for p in x['providers']))
        if not ready:raise RuntimeError('No server-side configuration with available credentials found')
        config=ready[0]['path']
    print('SERVER_API_CONFIG_READY',config,flush=True)
    start=time.monotonic()
    while not (args.local/'coverage_comparison.csv').exists():
        if time.monotonic()-start>1800:raise TimeoutError('Coverage outputs not ready')
        time.sleep(5)
    execute('mkdir -p '+' '.join(shlex.quote(root+'/'+d) for d in args.datasets))
    sftp=client.open_sftp()
    sftp.put(str(args.script),root+'/remote_fixed_task_audit.py')
    for dataset in args.datasets:
        sftp.put(str(args.local/dataset/'audit_pairs.json'),root+'/'+dataset+'/audit_pairs.json')
    cmd='/root/miniconda3/bin/python -u '+shlex.quote(root+'/remote_fixed_task_audit.py')+' --root '+shlex.quote(root)+' --api-config '+shlex.quote(config)+' --workers 8 --datasets '+' '.join(args.datasets)
    _,stdout,stderr=client.exec_command(cmd,timeout=3600)
    for line in stdout:print(line.rstrip(),flush=True)
    code=stdout.channel.recv_exit_status()
    if code:
        print(stderr.read().decode('utf-8','replace')[-2000:],flush=True)
        raise RuntimeError('Remote audit failed; server response caches retained')
    for name in ['task_judgments.jsonl','task_judge_manifest.json']:
        sftp.get(root+'/'+name,str(args.local/name))
    # Download non-secret request-result records for reproducibility.
    local_cache=args.local/'judge_cache'
    for provider in sftp.listdir(root+'/judge_cache'):
        (local_cache/provider).mkdir(parents=True,exist_ok=True)
        for name in sftp.listdir(root+'/judge_cache/'+provider):
            if name.endswith('.json'):
                sftp.get(root+'/judge_cache/'+provider+'/'+name,str(local_cache/provider/name))
    sftp.close();client.close()
    print('AUDIT_DOWNLOADED',str(args.local),flush=True)


if __name__=='__main__':main()
