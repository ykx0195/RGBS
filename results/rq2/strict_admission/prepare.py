import hashlib,json,shutil,tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'candidate_pool_strict085'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'scope':'sequential ordinary admission of existing strict-pairwise reselections','datasets':{}}
for ds in ['banking77','abcd','dogo']:
    out=ROOT/'inputs'/ds;out.mkdir(parents=True,exist_ok=True)
    src=SOURCE/ds
    for name in ['train_frozen_selection.json','selected_vectors.npy','test_coverage.json']:
        shutil.copyfile(src/name,out/name)
    f=json.loads((src/'train_frozen_selection.json').read_text(encoding='utf-8'))
    v=np.load(src/'selected_vectors.npy');s=v@v.T;np.fill_diagonal(s,-np.inf)
    assert np.all(s<np.float32(.85))
    manifest['datasets'][ds]={'k':len(f['texts']),'files':{p.name:sha(p) for p in out.iterdir()},
       'selection_source':str(src/'train_frozen_selection.json'),'selection_source_sha256':sha(src/'train_frozen_selection.json')}
(ROOT/'inputs/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with tarfile.open(ROOT/'inputs.tar.gz','w:gz') as t:t.add(ROOT/'inputs',arcname='inputs')
print({ds:r['k'] for ds,r in manifest['datasets'].items()})
