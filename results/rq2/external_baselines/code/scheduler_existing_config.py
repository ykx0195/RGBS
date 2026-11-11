import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
PY=ROOT/'venv/bin/python'
DATA=Path('/root/autodl-tmp/ykx_migrated_pending_20261031/data')
MODELS=Path('/root/autodl-tmp/liman/models')
mode='simple'
lane='simple_existing_config'
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
state={'lane':lane,'pid':os.getpid(),'started':time.time(),'jobs':[]}
def save():
    p=ROOT/f'{lane}_status.json'
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(p)
save()
references=json.loads((ROOT/'existing_main_configs.json').read_text())
# Prepare shared Train initialization only; no complete-method generation runs.
# GASLITE can continue to use these files after the redundant full lane is stopped.
for ds in ['abcd','dogo','banking77']:
    reference=references[ds]
    shared=ROOT/'runs/full'/ds
    cmd=[str(PY),'-u',str(ROOT/'code_existing_config/run_configurable.py'),
         '--dataset',ds,'--representation','desensitized','--encoder','bge',
         '--data-dir',str(DATA/ds.replace('dogo','multidogo')/'desensitized'),
         '--model-dir',str(MODELS/'bge-small-en-v1.5'),'--out-dir',str(shared),
         '--threshold',str(reference['threshold']),'--seed',str(reference['seed']),
         '--device','cuda','--prepare-only']
    with (ROOT/'logs'/f'prepare_shared_{ds}.log').open('w') as log:
        rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    if rc:raise RuntimeError('Shared initialization preparation failed: '+ds)
    import hashlib, shutil
    for filename,key in [('train.jsonl','train_sha256')]:
        path=DATA/ds.replace('dogo','multidogo')/'desensitized'/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference[key]
    for filename,key in [('seed_selection.json','seeds_sha256'),('target_vectors.npy','target_sha256')]:
        actual=hashlib.sha256((shared/'targets'/filename).read_bytes()).hexdigest()
        if key=='seeds_sha256':
            assert actual==reference[key],(ds,key,'Natural starting texts differ from existing main config')
        else:
            # Recomputed floating-point guides may differ bytewise across runs.
            # Keep both hashes; never claim byte-identical vectors when they differ.
            (shared/'existing_config_alignment.json').write_text(json.dumps({
                'reference_target_sha256':reference[key],'recomputed_target_sha256':actual,
                'target_bytes_match':actual==reference[key],
                'seed_file_sha256':reference['seeds_sha256'],'seed_bytes_match':True},indent=2))
    print('Prepared and checked existing initialization:',ds,flush=True)
for dataset in ['abcd','dogo','banking77']:
    out=ROOT/'runs'/lane/dataset
    reference=references[dataset]
    out.mkdir(parents=True,exist_ok=True)
    shared=ROOT/'runs/full'/dataset
    for filename in ['train_vectors.npy','train_vectors.json']:
        if not (out/filename).exists():shutil.copy2(shared/filename,out/filename)
    if not (out/'targets').exists():shutil.copytree(shared/'targets',out/'targets')
    prior=ROOT/'runs/simple'/dataset/'api_cache'
    if prior.exists() and not (out/'api_cache').exists():shutil.copytree(prior,out/'api_cache')
    cmd=[str(PY),'-u',str(ROOT/'code_existing_config/run_configurable.py'),'--comparison-mode',mode,
         '--generation-cap-per-slot','0','--dataset',dataset,'--representation','desensitized',
         '--encoder','bge','--data-dir',str(DATA/dataset.replace('dogo','multidogo')/'desensitized'),
         '--model-dir',str(MODELS/'bge-small-en-v1.5'),'--lm-dir',str(MODELS/'Qwen2.5-1.5B-Instruct'),
         '--api-config','/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json',
         '--out-dir',str(out),'--threshold','0.85','--seed','42','--device','cuda',
         '--steps-per-slot','32','--beam-width','50']
    # Read every configurable search parameter from the existing completed run.
    for key in ['k','threshold','seed','batch_size','steps_per_slot','beam_width',
                'positions_per_parent','tokens_per_position','rewrites_per_api','max_words_per_step',
                'max_total_word_changes','prompt_examples','feedback_examples','focus_size',
                'temperature','align_weight','max_ppl','relative_ppl','ppl_prefilter','swap_passes']:
        cmd += ['--'+key.replace('_','-'),str(reference[key])]
    job={'dataset':dataset,'out':str(out),'started':time.time(),'status':'running','attempts':[]}
    state['jobs'].append(job);save()
    for attempt in range(2):
        with (ROOT/'logs'/f'{lane}_{dataset}_attempt{attempt}.log').open('w') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            job['child_pid']=p.pid;save()
            rc=p.wait()
        job['attempts'].append({'attempt':attempt,'exit_code':rc,'ended':time.time()});save()
        if rc==0:break
        time.sleep(15)
    job.update(status='completed' if rc==0 and (out/'final_report.json').exists() else 'failed',ended=time.time())
    save()
state['ended']=time.time();state['status']='completed' if all(j['status']=='completed' for j in state['jobs']) else 'has_failures';save()
