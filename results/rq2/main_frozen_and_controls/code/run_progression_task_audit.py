"""Run only missing content-aligned task judgments on an authorized host."""
from pathlib import Path
import argparse,getpass,json,shlex,sys,time
import paramiko

ROOT=Path(r'C:\Users\ykx01\Desktop\results\递进主线补充实验_20261101')
SCRIPT=Path(__file__).with_name('remote_fixed_task_audit.py')
REMOTE='/root/autodl-tmp/vldb_progression_task_audit_20261101'

def main():
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True);args=p.parse_args()
    password=getpass.getpass('SSH password (not saved): ')
    client=paramiko.SSHClient();client.load_system_host_keys()
    try:
        client.connect(args.host,port=args.port,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20)
    finally:password=''
    print('SSH_CONNECTED',flush=True)
    def execute(command,timeout=90):
        _,stdout,stderr=client.exec_command(command,timeout=timeout)
        output=stdout.read().decode('utf-8','replace');status=stdout.channel.recv_exit_status()
        if status:raise RuntimeError('Remote command failed; status='+str(status))
        return output
    # Discover only existing research provider configurations. Credentials stay remote.
    discovery=r'''
from pathlib import Path
import os,json
found=[]
for base,dirs,files in os.walk('/root/autodl-tmp'):
    rel=Path(base).relative_to('/root/autodl-tmp')
    dirs[:]=[d for d in dirs if d not in ['api_cache','judge_cache','models','node_modules','.git','__pycache__'] and len(rel.parts)<6]
    for name in files:
        if not (name.endswith('.json') and ('provider' in name.lower() or 'config' in name.lower())):continue
        p=Path(base)/name
        try:
            if p.stat().st_size>500000:continue
            d=json.loads(p.read_text());ss=d.get('api_generators') or d.get('language',{}).get('api_generators')
            if not isinstance(ss,list) or len(ss)<2:continue
            ss=ss[:2]
            if {s.get('model') for s in ss}!={'qwen3.8-flash','deepseek-flash'}:continue
            ready=[]
            for s in ss:
                key=next((os.environ.get(n) for n in s.get('key_envs',[]) if os.environ.get(n)),None)
                if not key and s.get('key_file'):
                    k=Path(s['key_file']);k=k if k.is_absolute() else p.parent/k
                    key=k.read_text().strip() if k.is_file() else None
                if not key:key=s.get('api_key') or s.get('key')
                ready.append(bool(key))
            if all(ready):found.append({'path':str(p),'providers':[{'name':s['name'],'model':s['model']} for s in ss]})
        except Exception:pass
print(json.dumps(sorted(found,key=lambda x:x['path'])))
'''
    candidates=json.loads(execute('/root/miniconda3/bin/python -c '+shlex.quote(discovery)))
    if not candidates:raise RuntimeError('No ready configuration found for the exact prior two judge models')
    candidate=candidates[0];config=candidate['path'];print('PRIOR_JUDGE_MODELS_READY',candidate['providers'],flush=True)
    execute('mkdir -p '+shlex.quote(REMOTE+'/dogo'))
    sftp=client.open_sftp();sftp.put(str(SCRIPT),REMOTE+'/remote_fixed_task_audit.py')
    sftp.put(str(ROOT/'dogo/audit_pairs.json'),REMOTE+'/dogo/audit_pairs.json')
    cmd='/root/miniconda3/bin/python -u '+shlex.quote(REMOTE+'/remote_fixed_task_audit.py')+' --root '+shlex.quote(REMOTE)+' --api-config '+shlex.quote(config)+' --datasets dogo --workers 4 --batch-size 12'
    _,stdout,stderr=client.exec_command(cmd,timeout=1800)
    for line in stdout:print(line.rstrip(),flush=True)
    status=stdout.channel.recv_exit_status()
    if status:raise RuntimeError('Remote audit exited '+str(status)+'; cached results remain in its dedicated output directory')
    for name in ['task_judgments.jsonl','task_judge_manifest.json']:
        sftp.get(REMOTE+'/'+name,str(ROOT/name))
    dest=ROOT/'judge_cache'
    for provider in sftp.listdir(REMOTE+'/judge_cache'):
        (dest/provider).mkdir(parents=True,exist_ok=True)
        for name in sftp.listdir(REMOTE+'/judge_cache/'+provider):
            if name.endswith('.json'):sftp.get(REMOTE+'/judge_cache/'+provider+'/'+name,str(dest/provider/name))
    (ROOT/'remote_audit_execution.json').write_text(json.dumps({'host':args.host,'port':args.port,'remote_output':REMOTE,
        'provider_models':candidate['providers'],'datasets':['dogo'],'requested_pairs':331,'existing_pairs_reused':614,
        'gpu_used':False,'instruction_source':str(SCRIPT),'credential_saved':False},ensure_ascii=False,indent=2),encoding='utf-8')
    sftp.close();client.close();print('AUDIT_DOWNLOADED',flush=True)

if __name__=='__main__':main()
