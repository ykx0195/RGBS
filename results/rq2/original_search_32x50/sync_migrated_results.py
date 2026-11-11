"""Sync status and completed reports from the 34868 round-budget queue."""

from __future__ import annotations

import getpass
import argparse
import json
from pathlib import Path
import stat

import paramiko


REMOTE = '/root/autodl-tmp/ykx_migrated_pending_20261031'
LOCAL = Path(r'C:\Users\ykx01\Desktop\results\三数据集_迁移待跑_34868_20261031')


def exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def copy_file(sftp: paramiko.SFTPClient, source: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.part')
    sftp.get(source, str(temp))
    temp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', default=REMOTE)
    parser.add_argument('--local', type=Path, default=LOCAL)
    args = parser.parse_args()
    remote = args.remote.rstrip('/')
    local = args.local
    password = getpass.getpass('SSH password for port 34868: ')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.nmb2.seetacloud.com', port=34868,
                   username='root', password=password, timeout=20)
    try:
        sftp = client.open_sftp()
        local.mkdir(parents=True, exist_ok=True)
        for name in ('queue_status.json', 'scheduler.log'):
            if exists(sftp, remote + '/' + name):
                copy_file(sftp, remote + '/' + name, local / name)
        root = remote + '/runs'
        for item in sftp.listdir_attr(root):
            if not stat.S_ISDIR(item.st_mode):
                continue
            name = item.filename
            remote_run = root + '/' + name
            if not exists(sftp, remote_run + '/final_report.json'):
                continue
            output = local / 'runs' / name
            for filename in ('final_report.json', 'train_report.json', 'config.json',
                             'state.json', 'run.log'):
                if exists(sftp, remote_run + '/' + filename):
                    copy_file(sftp, remote_run + '/' + filename, output / filename)
            for item2 in sftp.listdir_attr(remote_run):
                if item2.filename.startswith('slot_') and item2.filename.endswith('.json'):
                    copy_file(sftp, remote_run + '/' + item2.filename,
                              output / item2.filename)
            report = json.loads((output / 'final_report.json').read_text(encoding='utf-8'))
            test = report['metrics']['test']
            print(name, f"Test {test['covered']}/{test['total']} ({test['coverage']:.2%})")
        sftp.close()
    finally:
        client.close()


if __name__ == '__main__':
    main()
