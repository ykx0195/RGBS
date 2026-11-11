import hashlib,json,tarfile
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
results=Path(r'C:\Users\ykx01\Desktop\results')
target=root/'frozen_embeddings'; target.mkdir(exist_ok=True)
manifest={}
for name in ['banking77','abcd','dogo']:
    folder=root/'inputs'/name
    train=[json.loads(x) for x in (folder/'train.jsonl').read_text('utf-8').splitlines() if x]
    test=[json.loads(x) for x in (folder/'test.jsonl').read_text('utf-8').splitlines() if x]
    train_src=results/'优先补充实验_20261031/baselines'/name/'train_vectors.npy'
    identity_src=train_src.with_name('train_vectors_identity.json')
    identity=json.loads(identity_src.read_text('utf-8'))
    assert identity['train_sha256']==hashlib.sha256((folder/'train.jsonl').read_bytes()).hexdigest()
    assert identity['model_sha256']=='3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad'
    pairfolder=results/'固定文本_原文脱敏配对_20261031'/('ABCD补充/abcd' if name=='abcd' else name)
    paired=np.load(pairfolder/'embeddings.npz')
    paired_identity=json.loads((pairfolder/'embedding_identity.json').read_text('utf-8'))
    assert paired_identity['delex_test_sha256']==hashlib.sha256((folder/'test.jsonl').read_bytes()).hexdigest()
    assert paired_identity['model_sha256']==identity['model_sha256']
    texts=[x['text'] for x in train+test]
    arrays=[np.load(train_src),paired['delex']]
    keys=json.loads((folder/'constructed.json').read_text('utf-8'))
    if name!='dogo':
        assert json.loads((pairfolder/'frozen_keys.json').read_text('utf-8'))['texts']==[x['text'] for x in keys]
        texts += [x['text'] for x in keys]
        arrays.append(paired['keys'])
    vectors=np.concatenate(arrays).astype(np.float32)
    assert len(texts)==len(vectors)
    assert np.max(abs(np.linalg.norm(vectors,axis=1)-1))<1e-5
    np.savez_compressed(target/(name+'.npz'),texts=np.array(texts),vectors=vectors)
    sources=[train_src,identity_src,pairfolder/'embeddings.npz',pairfolder/'embedding_identity.json']
    manifest[name]={'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'train_identity':identity,'paired_identity':paired_identity,'vectors':len(vectors),'output_sha256':hashlib.sha256((target/(name+'.npz')).read_bytes()).hexdigest()}
(target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
with tarfile.open(root/'frozen_embeddings.tar.gz','w:gz') as t: t.add(target,arcname='frozen_embeddings')
print('prepared',[(k,v['vectors']) for k,v in manifest.items()])
