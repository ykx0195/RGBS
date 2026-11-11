"""Stage and launch three materialized 0.85 searches on server 34868."""

from __future__ import annotations

import getpass
import hashlib
import json
from pathlib import Path
import shlex

import paramiko


ROOT = '/root/autodl-tmp/ykx_materialized_t085_32x50_20261031'
SOURCE_CODE = '/root/autodl-tmp/ykx_migrated_pending_20261031/code'
LOCAL_DATA = Path(r'C:\Users\ykx01\Desktop\ykx\data')
LOCAL_SCHEDULER = Path(__file__).with_name('materialized_085_scheduler.py')
DATASETS = ['abcd', 'banking77', 'multidogo']
EXPECTED_TRAIN = {
    'abcd': '71349b3361a6', 'banking77': 'b4779b894f10',
    'multidogo': '61ed77d22b6c',
}
EXPECTED_TEST = {
    'abcd': '59eed2e4a7e6', 'banking77': '6e1f68813fa2',
    'multidogo': '3b69bf45e4ef',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(client: paramiko.SSHClient, code: str, timeout: int = 120) -> str:
    _, out, err = client.exec_command(code, timeout=timeout)
    stdout = out.read().decode(errors='replace')
    stderr = err.read().decode(errors='replace')
    rc = out.channel.recv_exit_status()
    if rc:
        raise RuntimeError(f'Remote exit={rc}: {stderr[:500]}')
    return stdout


def main() -> None:
    for dataset in DATASETS:
        directory = LOCAL_DATA / dataset / 'non_desensitized'
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['representation'] == 'materialized'
        assert dataset in manifest['dataset'].lower()
        for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl'):
            path = directory / name
            assert sha(path) == manifest['files'][name]['sha256'], path
        assert sha(directory / 'train.jsonl').startswith(EXPECTED_TRAIN[dataset])
        assert sha(directory / 'test.jsonl').startswith(EXPECTED_TEST[dataset])
        print('LOCAL_VALID', dataset, manifest['counts'], flush=True)

    password = getpass.getpass('SSH password for port 34868: ')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.nmb2.seetacloud.com', port=34868,
                   username='root', password=password, timeout=20)
    try:
        if command(client, 'test -e ' + shlex.quote(ROOT + '/queue_status.json')
                   + ' && echo exists || true').strip() == 'exists':
            raise RuntimeError('This experiment queue already exists; refusing duplicate launch')
        command(client, 'mkdir -p ' + shlex.quote(ROOT)
                + ' && cp -a ' + shlex.quote(SOURCE_CODE) + ' ' + shlex.quote(ROOT + '/code'))
        sftp = client.open_sftp()
        for dataset in DATASETS:
            source = LOCAL_DATA / dataset / 'non_desensitized'
            target = ROOT + '/data/' + dataset + '/non_desensitized'
            command(client, 'mkdir -p ' + shlex.quote(target))
            for name in ('manifest.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'):
                sftp.put(str(source / name), target + '/' + name)
            print('UPLOADED', dataset, flush=True)
        with sftp.open(ROOT + '/scheduler.py.part', 'wb') as out:
            out.write(LOCAL_SCHEDULER.read_bytes())
        sftp.rename(ROOT + '/scheduler.py.part', ROOT + '/scheduler.py')
        sftp.close()

        check = r'''
import hashlib,json,pathlib,torch,transformers,openai
r=pathlib.Path('/root/autodl-tmp/ykx_materialized_t085_32x50_20261031')
assert hashlib.sha256((r/'code/run_global_round_budget.py').read_bytes()).hexdigest()=='87928a2ef7380dcf72d6cd164d9e27c55bc9add14b3c92c6a3b42ef64f8d0f8d'
for d in ['abcd','banking77','multidogo']:
    p=r/'data'/d/'non_desensitized'
    m=json.load(open(p/'manifest.json'))
    assert m['representation']=='materialized'
    for n in ['train.jsonl','validation.jsonl','test.jsonl']:
        assert hashlib.sha256((p/n).read_bytes()).hexdigest()==m['files'][n]['sha256']
print('REMOTE_VALID',torch.__version__,transformers.__version__,openai.__version__)
'''
        print(command(client, '/root/miniconda3/bin/python -c ' + shlex.quote(check)).strip(), flush=True)

        probe = r'''
import json,pathlib
from openai import OpenAI
p=pathlib.Path('/root/autodl-tmp/ykx_migrated_pending_20261031/api_providers.json')
x=json.load(open(p));specs=x.get('language',{}).get('api_generators') if 'language' in x else x.get('api_generators')
for spec in specs:
    key=(p.parent/spec['key_file']).read_text().strip()
    c=OpenAI(api_key=key,base_url=spec['base_url'],timeout=30,max_retries=0)
    z=c.chat.completions.create(model=spec['model'],messages=[{'role':'user','content':'Reply OK.'}],max_tokens=8,temperature=0,extra_body=spec.get('extra_body'))
    print(spec['name'],'OK',z.choices[0].finish_reason)
'''
        print(command(client, '/root/miniconda3/bin/python -c ' + shlex.quote(probe), 90).strip(), flush=True)

        launcher = r'''
import os,subprocess
root='/root/autodl-tmp/ykx_materialized_t085_32x50_20261031'
log=open(root+'/scheduler.log','ab',buffering=0)
p=subprocess.Popen(['/root/miniconda3/bin/python','-u',root+'/scheduler.py'],
                   cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                   start_new_session=True,close_fds=True)
print('SCHEDULER_PID',p.pid)
'''
        print(command(client, '/root/miniconda3/bin/python -c ' + shlex.quote(launcher)).strip(), flush=True)
        import time
        time.sleep(25)
        status = json.loads(command(client, 'cat ' + shlex.quote(ROOT + '/queue_status.json')))
        print('JOBS', {k: (v['status'], v.get('pid')) for k,v in status['jobs'].items()}, flush=True)
        print(command(client,'tail -n 12 '+shlex.quote(ROOT+'/scheduler.log')).strip(),flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
