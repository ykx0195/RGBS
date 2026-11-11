from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,urllib.request
import numpy as np
from sklearn.cluster import KMeans

sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
base=Path.home()/'Desktop/results';root=base/'CorpusPoisoning_HotFlip对比_20261031';root.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
original=root/'official_source';original.mkdir(exist_ok=True)
if (original/'provenance.json').exists():commit=read(original/'provenance.json')['commit']
else:
    req=urllib.request.Request('https://api.github.com/repos/princeton-nlp/corpus-poisoning/commits/main',headers={'User-Agent':'Research-baseline-reproduction'})
    try:
        with urllib.request.urlopen(req,timeout=40) as f:commit=json.load(f)['sha']
    except urllib.error.HTTPError as exc:
        if exc.code not in [403,429]:raise
        # Save exact content hashes when GitHub's unauthenticated commit API is limited.
        # This is a source snapshot, never represented as a resolved commit SHA.
        commit='main-snapshot-with-file-sha256'
hashes={}
for name in ['src/attack_poison.py','src/utils.py','README.md','LICENSE','requirements.txt']:
    dest=original/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        revision='main' if commit=='main-snapshot-with-file-sha256' else commit
        with urllib.request.urlopen('https://raw.githubusercontent.com/princeton-nlp/corpus-poisoning/'+revision+'/'+name,timeout=40) as f:dest.write_bytes(f.read())
    hashes[name]=sha(dest)
save(original/'provenance.json',{'repository':'https://github.com/princeton-nlp/corpus-poisoning','commit':commit,'files':hashes})
code=root/'code';code.mkdir(exist_ok=True)
source=(original/'src/attack_poison.py').read_text(encoding='utf-8');tree=ast.parse(source)
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='hotflip_attack')
extracted=ast.get_source_segment(source,node)
license_text=(original/'LICENSE').read_text(encoding='utf-8')
(code/'official_hotflip_core.py').write_text('"""'+license_text+'\nExact hotflip_attack extracted from pinned official source '+commit+'.\n"""\nimport torch\n\n'+extracted+'\n',encoding='utf-8')
src=base/'newcode/src'
for name in ['hybrid_search.py','portfolio.py','riemann_common.py']:
    shutil.copyfile(src/name,code/name)
plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'official_commit':commit,'seed':42,'datasets':{'banking77':15,'abcd':5,'dogo':10},
      'representation':'desensitized','encoder':'bge-small-en-v1.5','threshold':.85,
      'iterations_per_cluster':5000,'gradient_candidates':100,'query_batch_size':64,'candidate_encoding_batch_size':32,
      'checkpoint_budgets':[0,100,500,1000,2500,5000],'threads_per_worker':4,
      'objective':'Mean cosine similarity to a KMeans cluster. Random Train minibatch for gradient and separate minibatch for candidate acceptance; full cluster Train mean selects best saved text.',
      'initialization':'One natural Train question nearest to each normalized KMeans center, with deterministic index ties; no Riemann target or RKGB-generated text.',
      'search':'Exact official HotFlip top-k gradient vocabulary scoring; random editable token; actual decoded/re-encoded candidate text scoring; strict sampled mean-score improvement.',
      'adaptations':['JSONL queries replace BEIR query-passage pairs','BGE CLS and L2 normalization replace original retriever interface',
                     'same original cached Train vectors and model identity','special tokens and original placeholder spans protected',
                     'no artificial positive documents or fake qrels','original gold-ranking checkpoint criterion replaced by full-Train cluster mean',
                     'variable seed-derived text lengths, max encoder length 512','decoded candidate must preserve placeholder spelling and occurrence counts',
                     'final evaluation is threshold union coverage, not corpus-relative retrieval rank'],
      'candidate_vocabulary':'All tokenizer vocabulary except special tokens; no RKGB alphabet-only or word-level edits restriction.',
      'no_api_calls':True,'no_LLM_repair':True,'no_test_for_search_or_checkpoint_selection':True,
      'quality_evaluation':'Raw coverage first. Local Qwen PPL/readability may be evaluated after freezing, equally on RKGB and this baseline, with exact evaluator disclosure. No claim of dual-API-validated quality without actual judgments.',
      'cost_scope':'Use official default 5000 updates and 100 candidates, do not equate 5000 token updates to RKGB 32 rounds. Record timing, encoder work and accepted updates.',
      'parallel_policy':'Independent clusters across all three datasets run concurrently under memory admission controls. CPU or CUDA allowed; per-job device recorded.',
      'method_name':'Corpus Poisoning adapted to semantic-cache coverage, not exact reproduction of original BEIR experiments.'}
if not (root/'experiment_plan.json').exists():save(root/'experiment_plan.json',plan)
else:
    old=read(root/'experiment_plan.json');assert all(old[k]==v for k,v in plan.items() if k!='created_at_utc')
for dataset,k in plan['datasets'].items():
    dest=root/'inputs'/dataset;dest.mkdir(parents=True,exist_ok=True)
    baseline=base/'优先补充实验_20261031/baselines'/dataset
    folder='multidogo' if dataset=='dogo' else dataset
    trpath=Path.home()/'Desktop/ykx/data'/folder/'desensitized/train.jsonl'
    rows=[json.loads(x) for x in trpath.read_text(encoding='utf-8').splitlines() if x.strip()]
    vectors=np.load(baseline/'train_vectors.npy');identity=read(baseline/'train_vectors_identity.json')
    assert sha(trpath)==identity['train_sha256'] and vectors.shape==(len(rows),384)
    fit=KMeans(n_clusters=k,random_state=42,n_init=10).fit(vectors)
    centers=fit.cluster_centers_.astype('float32');centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    seeds=[]
    for cluster in range(k):
        indices=np.flatnonzero(fit.labels_==cluster);scores=vectors[indices]@centers[cluster]
        best=int(indices[int(np.argmax(scores))])
        seeds.append({'cluster_id':cluster,'train_index':best,'source_id':rows[best]['source_id'],
                      'text':rows[best]['text'],'cluster_size':len(indices),'initial_center_cosine':float(vectors[best]@centers[cluster])})
    cluster_identity={'dataset':dataset,'k':k,'seed':42,'train_sha256':sha(trpath),'train_vectors_sha256':sha(baseline/'train_vectors.npy'),
                      'model_sha256':identity['model_sha256'],'kmeans_n_init':10,'selection_uses_Train_only':True,
                      'selected':seeds,'cluster_sizes':[int((fit.labels_==i).sum()) for i in range(k)]}
    if (dest/'seed_selection.json').exists():assert read(dest/'seed_selection.json')==cluster_identity
    else:save(dest/'seed_selection.json',cluster_identity)
    np.save(dest/'cluster_labels.npy',fit.labels_.astype('int32'));np.save(dest/'kmeans_centers.npy',centers)
    shutil.copyfile(trpath,dest/'train.jsonl');shutil.copyfile(baseline/'train_vectors.npy',dest/'train_vectors.npy')
    shutil.copyfile(baseline/'train_vectors_identity.json',dest/'train_vectors_identity.json')
    save(dest/'input_identity.json',{'train_sha256':sha(dest/'train.jsonl'),'vectors_sha256':sha(dest/'train_vectors.npy'),
                                   'labels_sha256':sha(dest/'cluster_labels.npy'),'seed_selection_sha256':sha(dest/'seed_selection.json'),
                                   'heldout_accessed_for_preparation':False})
    print('TRAIN_ONLY_HOTFLIP_INPUTS_READY',dataset,k,cluster_identity['cluster_sizes'],flush=True)
save(root/'code_manifest.json',{p.name:sha(p) for p in code.glob('*.py')})
print('HOTFLIP_BUNDLE_READY',str(root),'official commit',commit,flush=True)
