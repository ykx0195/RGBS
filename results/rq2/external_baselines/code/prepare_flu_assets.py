"""Fetch author's published fluency checkpoint; verify against primary hashes."""
import hashlib,json,time
from pathlib import Path
import requests
ROOT=Path('/root/autodl-tmp/vldb_supplement_20261102')
OUT=ROOT/'flu_assets';OUT.mkdir(exist_ok=True)
MODEL_REV='1f5966c89d321fc512fb70d6f601eb582a28769a'
CKPT_SHA='0ca14aed7ee89bf9fe86d23fc0ff73722f6494b425ba7dcfb7228d2ed60887ad'
CKPT_SIZE=1310260186
TOK_REV='f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
TOK_BLOBS={'config.json':'24b4776266ab6abf7a9d6672f42b77f06ef77b87',
 'special_tokens_map.json':'a8b3208c2884c4efb86e49300fdd3dc877220cdf',
 'tokenizer.json':'688882a79f44442ddc1f60d70334a7ff5df0fb47',
 'tokenizer_config.json':'8046d4d4aff4529a10e0ad47043bdabd6304fb58',
 'vocab.txt':'fb140275c155a9c7c5a3b3e0e77a9e839594a938'}
def status(**kw):
    p=OUT/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(kw,indent=2));t.replace(p)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def download(repo,rev,name,dest,size=None,sha=None,blob=None):
    def check(p):
        if not p.exists():return False
        if size is not None and p.stat().st_size!=size:return False
        if sha is not None:return digest(p)==sha
        if blob is not None:
            b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==blob
        return True
    if check(dest):return
    part=dest.with_suffix(dest.suffix+'.part')
    for attempt in range(4):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if size is not None and offset==size:
                if check(part):part.replace(dest);return
                raise RuntimeError('Downloaded checkpoint hash mismatch')
            url=f'https://hf-mirror.com/{repo}/resolve/{rev}/{name}?download=true&vldb_fetch={int(time.time())}'
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(url,headers=headers,stream=True,timeout=(12,60)) as r:
                r.raise_for_status()
                if offset and r.status_code!=206:offset=0
                mode='ab' if offset else 'wb';last=offset
                with part.open(mode) as f:
                    for chunk in r.iter_content(4*1024*1024):
                        f.write(chunk);offset+=len(chunk)
                        if offset-last>=64*1024*1024:
                            status(phase='downloading',file=name,bytes=offset,total=size,attempt=attempt)
                            print('downloaded',name,offset,flush=True);last=offset
            if not check(part):raise RuntimeError('Asset size/hash mismatch')
            part.replace(dest);return
        except Exception as exc:
            status(phase='retrying',file=name,attempt=attempt,error=type(exc).__name__)
            if attempt==3:raise
            time.sleep(5)
def main():
    started=time.time();status(phase='preparing',started=started)
    tokenizer=OUT/'e5_tokenizer';tokenizer.mkdir(exist_ok=True)
    for name,blob in TOK_BLOBS.items():download('intfloat/e5-base-v2',TOK_REV,name,tokenizer/name,blob=blob)
    download('MatanBT/nanoGPT-BERT-Tokenizer',MODEL_REV,'ckpt.pt',OUT/'ckpt.pt',size=CKPT_SIZE,sha=CKPT_SHA)
    meta={'checkpoint_repo':'MatanBT/nanoGPT-BERT-Tokenizer','revision':MODEL_REV,'checkpoint_sha256':CKPT_SHA,
        'checkpoint_size':CKPT_SIZE,'tokenizer_repo':'intfloat/e5-base-v2','tokenizer_revision':TOK_REV,
        'tokenizer_git_blob_sha1':TOK_BLOBS,'transport':'hf-mirror.com, verified against primary huggingface.co API revision and object hashes',
        'seconds':time.time()-started}
    (OUT/'verified_assets.json').write_text(json.dumps(meta,indent=2))
    status(phase='complete',seconds=time.time()-started,checkpoint_sha256=CKPT_SHA)
    print('Author fluency assets verified',flush=True)
if __name__=='__main__':main()
