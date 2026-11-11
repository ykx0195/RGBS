"""Preserve old runs and create isolated 32-round adaptations."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import difflib
import hashlib
import json
import os
import py_compile
import shutil
import subprocess

ROOT = Path('/root/autodl-tmp/vldb_supplement_20261102')
code = ROOT / 'code_round32'
code.mkdir(exist_ok=True)
shutil.copy2(ROOT / 'official_fluency_local.py', code / 'official_fluency_local.py')
record = json.loads((ROOT / 'switch_to_32_record.json').read_text())
assert not (ROOT / 'remaining32_status.json').exists(), 'Inspect existing 32-round scheduler before relaunch'
changes = []
for original, destination, old_subdir, new_subdir in [
    ('run_gaslite_adapter.py', 'run_gaslite_adapter_32.py', 'gaslite', 'gaslite_32'),
    ('run_gaslite_flu.py', 'run_gaslite_flu_32.py', 'gaslite_flu', 'gaslite_flu_32'),
]:
    old = (ROOT / original).read_text()
    assert old.count("'n_iter':1 if a.smoke else 100") == 1
    new = old.replace("'n_iter':1 if a.smoke else 100", "'n_iter':1 if a.smoke else 32")
    new = new.replace(f"'{old_subdir}_smoke'", f"'{new_subdir}_smoke'")
    new = new.replace(f"'runs/{old_subdir}'", f"'runs/{new_subdir}'")
    old_cache = '        if slot.exists():payload=json.loads(slot.read_text())'
    assert new.count(old_cache) == 1
    new = new.replace(old_cache, '        if slot.exists():\n            payload=json.loads(slot.read_text())\n            assert payload.get("n_iter") == config["n_iter"], "Checkpoint iteration budget differs"')
    needle = "payload={'slot':index+1,"
    assert new.count(needle) == 1
    new = new.replace(needle, "payload={'n_iter':config['n_iter'],'slot':index+1,")
    marker = "'config':config,"
    assert "'n_iter':1 if a.smoke else 100" not in new
    assert f"'runs/{new_subdir}'" in new
    target = code / destination
    target.write_text(new)
    py_compile.compile(str(target), doraise=True)
    diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=original, tofile=destination))
    (code / (destination + '.patch')).write_text(diff)
    changes.append({'original': original, 'new': destination, 'original_sha256': hashlib.sha256(old.encode()).hexdigest(),
                    'new_sha256': hashlib.sha256(new.encode()).hexdigest(), 'changes': ['100 to 32 outer iterations', 'separate output directories', 'explicit iteration-budget checkpoint metadata and check']})
scheduler = ROOT / 'schedule_remaining32.py'
py_compile.compile(str(scheduler), doraise=True)
with (ROOT / 'logs/remaining32_controller.log').open('a') as log:
    proc = subprocess.Popen([str(ROOT / 'venv/bin/python'), '-u', str(scheduler)], cwd=ROOT,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
manifest = {'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'scheduler_pid': proc.pid, 'iterations': 32, 'changes': changes, 'old_switch_record': record,
            'selection_policy': 'Train-only search and selection; Test after freezing. Old 100-round checkpoints are not imported.',
            'budget_scope': 'Equal outer iterations, not a claim of equal wall time, API cost or computation.',
            'jobs': ['gaslite_32/soft_threshold/banking77', 'gaslite_flu_32/mean_cosine/banking77',
                     'gaslite_flu_32/mean_cosine/dogo', 'gaslite_flu_32/mean_cosine/abcd']}
(ROOT / 'remaining32_launch_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'scheduler_pid': proc.pid, 'iterations': 32, 'code': str(code), 'jobs': manifest['jobs']}))
