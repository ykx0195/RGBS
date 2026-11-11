from pathlib import Path
from datetime import datetime,timezone
import getpass,hashlib,json,paramiko,shlex,subprocess,sys,tarfile,time

sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
base=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031'
root='/root/autodl-tmp/corpus_hotflip_baseline_20261031';scriptdir=Path(__file__).resolve().parent
password=getpass.getpass('SSH password for 35564 (memory only): ')
def connect():
    c=paramiko.SSHClient();c.load_system_host_keys();c.connect('connect.nmb2.seetacloud.com',port=35564,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);c.get_transport().set_keepalive(20);return c
c=connect();last=None;downloaded=set()
def execute(cmd):
    _,o,e=c.exec_command(cmd,timeout=180);v=o.read().decode('utf-8','replace');rc=o.channel.recv_exit_status()
    if rc:raise OSError(e.read().decode('utf-8','replace')[-1400:])
    return v.strip()
def safe_extract(archive,dest):
    with tarfile.open(archive,'r:gz') as t:
        for m in t.getmembers():
            p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and (m.isdir() or m.isfile())
        t.extractall(dest,filter='data')
while True:
    try:
        s=c.open_sftp()
        try:
            with s.open(root+'/queue_status.json') as f:status=json.loads(f.read().decode())
        except OSError:
            s.close();print('HOTFLIP_WAITING_FOR_ENVIRONMENT_AND_SCHEDULER',flush=True);time.sleep(30);continue
        tmp=base/'queue_status.json.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(base/'queue_status.json')
        for name in ['scheduler.log','smoke.log','environment.json','smoke_verified.json','gpu_migration_manifest.json','experiment_plan.json']:
            try:s.get(root+'/'+name,str(base/name))
            except OSError:pass
        events=[]
        for name,job in status.get('jobs',{}).items():
            dest=base/'runs'/job['dataset']/f"cluster_{job['cluster']:02d}";dest.mkdir(parents=True,exist_ok=True);out=job['out_dir'];iterations=0
            if job['status']=='complete' and name not in downloaded:
                for fn in ['final_cluster.json','config.json','preflight.json','state.json','trace.jsonl','device_migration.json']+[f'budget_{b:05d}.json' for b in [0,100,500,1000,2500,5000]]:
                    try:s.get(out+'/'+fn,str(dest/fn))
                    except OSError:pass
                downloaded.add(name)
            elif job['status']=='running':
                try:
                    with s.open(out+'/state.json') as f:state=json.loads(f.read().decode())
                    iterations=state['completed_iterations'];(dest/'state.json').write_text(json.dumps(state,ensure_ascii=False)+'\n',encoding='utf-8')
                except (OSError,ValueError):pass
            elif job['status']=='failed':
                try:s.get(out+'/run.log',str(dest/'run.log'))
                except OSError:pass
            events.append({'job':name,'status':job['status'],'device':job.get('device'),'iteration':job.get('completed_iterations',iterations)})
        s.close();running=[x for x in events if x['status']=='running'];complete=[x for x in events if x['status']=='complete']
        failed=[x for x in events if x['status']=='failed']
        if failed:print('HOTFLIP_FAILED_CLUSTERS_NEED_ATTENTION',json.dumps(failed),flush=True)
        lines=['# Corpus Poisoning／HotFlip适配基线：运行状态','',
               '本地更新时间（UTC）：'+datetime.now(timezone.utc).isoformat(),'',
               f"队列状态：{status.get('status')}；完成聚类：{len(complete)}/30；运行聚类：{len(running)}。",'',
               '固定seed42；Banking77/ABCD/MultiDoGO分别输出15/5/10条文本；每簇5000次词元更新、100个候选；搜索API调用为0。',
               '当前仅GPU调度，最多4个任务并行；此前两个CPU任务的检查点及迁移记录保留在gpu_migration_manifest.json。',
               '', '| 当前任务 | 设备 | 已保存迭代数 |','|---|---|---:|']
        for x in running:lines.append(f"| {x['job']} | {x['device']} | {x['iteration']} |")
        lines+=['','所有任务只使用Train；当前没有最终Test结果时不提前报告覆盖。完整源码适配与预算定义见experiment_plan.json。']
        (base/'运行状态.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        signature=json.dumps(events,sort_keys=True)
        if signature!=last:print('HOTFLIP_PARALLEL_PROGRESS',len(complete),'/30 complete','running',json.dumps(running),flush=True);last=signature
        if status.get('status') in ['failed','failed_smoke']:
            print('HOTFLIP_SEARCH_NEEDS_ATTENTION',status.get('status'),flush=True);break
        if status.get('status')=='complete':
            print('ALL_HOTFLIP_CLUSTERS_COMPLETE_DOWNLOADING',flush=True)
            archive_remote=root+'/completed_search_results.tar.gz'
            digest=execute('tar -czf '+shlex.quote(archive_remote)+' -C '+shlex.quote(root)+' runs smoke smoke_verified.json code official_source queue_status.json environment.json scheduler.log experiment_plan.json device_migrations gpu_migration_manifest.json pending_gpu_migrations.json && sha256sum '+shlex.quote(archive_remote)).split()[0]
            archive=base/'completed_search_results.tar.gz';s=c.open_sftp();s.get(archive_remote,str(archive));assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest;safe_extract(archive,base)
            subprocess.run([sys.executable,str(scriptdir/'evaluate_corpus_hotflip.py')],check=True)
            s.put(str(base/'quality_inputs.json'),root+'/quality_inputs.json');s.put(str(scriptdir/'corpus_hotflip_quality.py'),root+'/corpus_hotflip_quality.py')
            cmd=shlex.quote(root+'/venv/bin/python')+' -u '+shlex.quote(root+'/corpus_hotflip_quality.py')+' 2>&1'
            _,o,_=c.exec_command(cmd,timeout=10800)
            for line in o:print(line.rstrip(),flush=True)
            if o.channel.recv_exit_status():raise RuntimeError('Local-model quality evaluation failed; raw search results preserved')
            s.get(root+'/quality_results.json',str(base/'quality_results.json'));s.close()
            for name in ['summarize_corpus_hotflip.py','validate_corpus_hotflip.py','plot_corpus_hotflip.py','archive_corpus_hotflip.py']:
                subprocess.run([sys.executable,str(scriptdir/name)],check=True)
            print('CORPUS_HOTFLIP_EXPERIMENT_FULLY_COMPLETE',flush=True);break
    except (OSError,EOFError,paramiko.SSHException) as exc:
        print('HOTFLIP_MONITOR_RECONNECT',type(exc).__name__,flush=True)
        try:c.close()
        except Exception:pass
        c=connect()
    time.sleep(45)
c.close();password=''
