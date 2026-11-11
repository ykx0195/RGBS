from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
root=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031'
assert json.loads((root/'artifact_validation.json').read_text(encoding='utf-8'))['passed']
code=root/'control_code';code.mkdir(exist_ok=True)
names=['prepare_corpus_hotflip_bundle','prepare_hotflip_environment','deploy_corpus_hotflip','verify_hotflip_adapter_local',
       'corpus_hotflip_worker','corpus_hotflip_scheduler','watch_corpus_hotflip','evaluate_corpus_hotflip','corpus_hotflip_quality',
       'summarize_corpus_hotflip','validate_corpus_hotflip','plot_corpus_hotflip','archive_corpus_hotflip','switch_hotflip_gpu_only']
for name in names:shutil.copyfile(Path(__file__).parent/(name+'.py'),code/(name+'.py'))
archive=root/'CorpusPoisoning_HotFlip对比_完整结果.zip'
files={p.relative_to(root).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
       for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['artifact_sha256.json',archive.name] and '__pycache__' not in p.parts}
(root/'artifact_sha256.json').write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in list(files)+['artifact_sha256.json']:z.write(root/name,arcname=name)
print('CORPUS_HOTFLIP_RESULT_ARCHIVE_SAVED',len(files),'files',flush=True)
