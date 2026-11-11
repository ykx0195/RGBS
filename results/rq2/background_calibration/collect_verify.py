import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import paramiko

ROOT=Path(__file__).resolve().parent
DEST=Path.home()/'Desktop/results/BGE085背景碰撞率校准_20261111'
REMOTE='/root/autodl-tmp/bge085_background_calibration_20261111'
DEST.mkdir(exist_ok=True)

def download(remote,local):
    import stat
    local.mkdir(parents=True,exist_ok=True)
    for item in sftp.listdir_attr(remote):
        source=remote+'/'+item.filename;target=local/item.filename
        if stat.S_ISDIR(item.st_mode):download(source,target)
        else:sftp.get(source,str(target))

if not globals().get('offline',False):
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.nmb2.seetacloud.com',port=35564,username='root',password=password,timeout=30)
    sftp=client.open_sftp()
    download(REMOTE+'/outputs',DEST/'outputs')
    download(REMOTE+'/inputs',DEST/'inputs')
    sftp.get(REMOTE+'/driver.log',str(DEST/'driver.log'))
    sftp.close();client.close()
for name in ['run_server.py','prepare.py','collect_verify.py']:
    shutil.copyfile(ROOT/name,DEST/name)
summary=json.loads((DEST/'outputs/summary.json').read_text('utf-8'))
records=summary['records']
assert len(records)==12 and summary['complete'] and summary['api_calls']==0
expected=json.loads((Path.home()/'Desktop/results/全文审查_第2项跨编码器核对_20261105/verified_results.json').read_text('utf-8'))
checks=[]
for row in records:
    data=np.load(DEST/'outputs'/row['encoder']/(row['dataset']+'.npz'))
    vectors=data['validation_vectors']
    recomputed=(vectors@vectors.T)[np.triu_indices(len(vectors),1)]
    scores=data['background_scores']
    assert np.allclose(recomputed,scores,atol=1e-6,rtol=0)
    assert len(scores)==row['validation_pairs']
    actual=int((scores>=row['threshold']).sum())
    assert actual==row['calibrated_background_matches']
    values,counts=np.unique(scores,return_counts=True)
    attainable=np.r_[0,np.cumsum(counts[::-1])]
    assert abs(actual-row['reference_background_matches'])==np.abs(attainable-row['reference_background_matches']).min()
    matrix=data['test_vectors']@data['key_vectors'].T
    assert np.allclose(matrix,data['test_similarities'],atol=1e-6,rtol=0)
    covered=int((matrix>=row['threshold']).any(1).sum())
    assert covered==row['covered_queries']
    prior=next(r for r in expected if r['dataset']==row['dataset'] and r['target_encoder']==row['encoder'])
    old_hits=prior['covered_queries']
    assert row['fixed085_hits']==old_hits,(row,old_hits)
    checks.append({'dataset':row['dataset'],'encoder':row['encoder'],'fixed085_baseline_reproduced':True,'calibration_count_optimal':True,'coverage_recomputed':True,'background_count_error':row['background_count_error'],'background_score_max_cpu_difference':float(np.max(abs(recomputed-scores))),'test_score_max_cpu_difference':float(np.max(abs(matrix-data['test_similarities'])))})
names=['bge-small-en-v1.5','all-MiniLM-L6-v2','gte-small','e5-small-v2']
labels=['BGE-small (source)','MiniLM-L6','GTE-small','E5-small-v2']
datasets=['banking77','abcd','dogo']
lines=['# BGE 0.85 背景碰撞率校准结果','','使用各数据集全部脱敏 Validation 自然请求的无序异源记录对，以 BGE-small 在 0.85 下的背景碰撞计数为基准，为其他编码器选择最接近该计数的阈值。相同误差时选择较少匹配。冻结当前论文主集合，Test 只用于最终评价。背景碰撞包含同任务请求间匹配。未重新优化文本，未调用生成模型 API。','','## Test coverage (%)','','| Encoder | Banking77 | ABCD | MultiDoGO |','|---|---:|---:|---:|']
for name,label in zip(names,labels):
    cells=[next(r for r in records if r['encoder']==name and r['dataset']==ds) for ds in datasets]
    lines.append('| '+label+' | '+' | '.join(f"{r['coverage_pct']:.2f}" for r in cells)+' |')
lines+=['','## Calibrated thresholds','','| Encoder | Banking77 | ABCD | MultiDoGO |','|---|---:|---:|---:|']
for name,label in zip(names,labels):
    cells=[next(r for r in records if r['encoder']==name and r['dataset']==ds) for ds in datasets]
    lines.append('| '+label+' | '+' | '.join(f"{r['threshold']:.6f}" for r in cells)+' |')
lines+=['','## BGE reference background rate','','| Dataset | Validation records | Pairs | Matches | Rate (%) |','|---|---:|---:|---:|---:|']
for ds in datasets:
    r=next(r for r in records if r['encoder']==names[0] and r['dataset']==ds)
    n=len(json.loads((DEST/'inputs'/(ds+'.json')).read_text('utf-8'))['validation'])
    lines.append(f"| {ds} | {n} | {r['validation_pairs']} | {r['reference_background_matches']} | {r['reference_background_rate_pct']:.6f} |")
(DEST/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(DEST/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2),encoding='utf-8')
files={p.relative_to(DEST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'}
(DEST/'sha256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'destination':str(DEST),'passed':True,'records':records},ensure_ascii=False,indent=2))
