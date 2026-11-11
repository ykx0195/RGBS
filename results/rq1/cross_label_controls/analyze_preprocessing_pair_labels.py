"""Preprocessing-only query-pair control, without selected/generated cache keys.

Reuse existing source-aligned query embeddings. Pair counts exclude records
with the same source_id. Source categories are task proxies, not adjudications.
"""
from pathlib import Path
import csv, hashlib, json, math, sys
import numpy as np

sys.stdout.reconfigure(encoding='utf8')
BASE=Path(r'C:\Users\ykx01\Desktop')
FIXED=BASE/'results/固定文本_原文脱敏配对_20261031'
DATA=BASE/'ykx/data'
OUT=BASE/'results/纯脱敏_跨标签碰撞_20261102';OUT.mkdir(exist_ok=True)
TAUS=(.75,.775,.8,.825,.85,.875,.9)
CONFIG={'banking77':('banking77',FIXED/'banking77'),
        'abcd':('abcd',FIXED/'ABCD补充/abcd'),
        'dogo':('multidogo',FIXED/'dogo')}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return [json.loads(line) for line in p.read_text(encoding='utf8').splitlines() if line.strip()]
def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[];samples=[]
manifest={'protocol':'All unordered natural Test-query pairs from different source IDs; same record pairs, encoder and numerical threshold across original/deidentified forms. No generated keys, K, attack optimization, selection or new encoder/judge calls.',
          'label_scope':'Source category is a task proxy. No human or model task judgment is inferred from it.',
          'encoder':'bge-small-en-v1.5; CLS; unit normalized; cached float32 CPU embeddings',
          'thresholds':TAUS,'primary_threshold':.85,
          'uncertainty':'Delete-one-source record jackknife for paired collision-rate differences; conditional on labels and fixed data, not independent-pair binomial intervals.',
          'post_hoc':True,'new_generation_calls':0,'new_judge_calls':0,'datasets':{}}
for ds,(folder,cache) in CONFIG.items():
    rp=DATA/folder/'non_desensitized/test.jsonl';dp=DATA/folder/'desensitized/test.jsonl'
    raw,de=read(rp),read(dp)
    ident=json.loads((cache/'embedding_identity.json').read_text(encoding='utf8'))
    assert sha(rp)==ident['raw_test_sha256'] and sha(dp)==ident['delex_test_sha256']
    assert len(raw)==len(de)
    assert [(r['source_id'],r['variant_id']) for r in raw]==[(r['source_id'],r['variant_id']) for r in de]
    assert all(r.get('category')==d.get('category') and r.get('category') not in [None,''] for r,d in zip(raw,de))
    assert all(r['split']=='test' for r in raw+de)
    with np.load(cache/'embeddings.npz') as z:x=z['raw'].copy();y=z['delex'].copy()
    n=len(raw);assert x.shape==y.shape==(n,384)
    assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5) and np.allclose(np.linalg.norm(y,axis=1),1,atol=1e-5)
    # Only the query arrays above are read; cached generated-key vectors are unused.
    sx=x @ x.T;sy=y @ y.T
    sid=np.array([r['source_id'] for r in raw]);label=np.array([r['category'] for r in raw])
    distinct=sid[:,None]!=sid[None,:]
    source_names,source_index=np.unique(sid,return_inverse=True);g=len(source_names)
    fields=['category']+(['flow'] if ds=='abcd' else ['domain'] if ds=='dogo' else [])
    for field in fields:
        assert all(r.get(field)==d.get(field) and r.get(field) not in [None,''] for r,d in zip(raw,de))
        labels=np.array([r[field] for r in raw]);same=labels[:,None]==labels[None,:]
        for tau in TAUS:
            a=sx>=tau;b=sy>=tau
            for stratum,eligible in [('all',distinct),('same_label',distinct&same),('different_label',distinct&~same)]:
                upper=np.triu(eligible,1);den=int(upper.sum())
                assert den>0
                rcount=int((a&upper).sum());dcount=int((b&upper).sum())
                new=(~a)&b&upper;lost=a&(~b)&upper
                delta=(b.astype(np.int8)-a.astype(np.int8))*eligible
                total_delta=dcount-rcount
                assert delta.sum()==2*total_delta
                row_incident=eligible.sum(1);row_delta=delta.sum(1)
                incidents=np.bincount(source_index,weights=row_incident,minlength=g)
                changes=np.bincount(source_index,weights=row_delta,minlength=g)
                jack=(total_delta-changes)/(den-incidents)
                se=math.sqrt((g-1)/g*float(((jack-jack.mean())**2).sum()))
                dpp=100*total_delta/den
                summary.append(dict(dataset=ds,split='test',query_count=n,unique_sources=g,label_field=field,label_count=len(set(labels)),
                    threshold=tau,stratum=stratum,eligible_pairs=den,raw_matched_pairs=rcount,delex_matched_pairs=dcount,
                    newly_matched_pairs=int(new.sum()),lost_matches=int(lost.sum()),
                    raw_collision_pct=100*rcount/den,delex_collision_pct=100*dcount/den,delta_pp=dpp,
                    record_jackknife_low_pp=dpp-1.96*100*se,record_jackknife_high_pp=dpp+1.96*100*se))
                if field=='category' and tau==.85 and stratum=='different_label':
                    ii,jj=np.where(new)
                    rng=np.random.default_rng(20261102)
                    pick=rng.choice(len(ii),size=min(20,len(ii)),replace=False)
                    for pos in sorted(pick):
                        i,j=int(ii[pos]),int(jj[pos])
                        samples.append(dict(dataset=ds,i=i,j=j,source_i=sid[i],source_j=sid[j],label_i=label[i],label_j=label[j],
                            raw_cosine=float(sx[i,j]),delex_cosine=float(sy[i,j]),raw_i=raw[i]['text'],raw_j=raw[j]['text'],
                            delex_i=de[i]['text'],delex_j=de[j]['text'],audit_status='UNJUDGED_SOURCE_LABEL_PROXY_ONLY'))
    manifest['datasets'][ds]={'query_count':n,'unique_sources':g,'source_category_count':len(set(label)),
        'raw_data_sha256':sha(rp),'delex_data_sha256':sha(dp),'query_embedding_cache_sha256':sha(cache/'embeddings.npz'),
        'embedding_identity':ident,'excluded_same_source_pairs':int(np.triu(~distinct,1).sum())}
csvout('pair_collision_by_label.csv',summary);csvout('new_cross_label_pair_samples_085.csv',samples)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
primary=[r for r in summary if r['threshold']==.85 and r['label_field']=='category']
csvout('primary_085.csv',primary)
for ds in CONFIG:
    r=next(r for r in primary if r['dataset']==ds and r['stratum']=='different_label')
    allrow=next(r for r in primary if r['dataset']==ds and r['stratum']=='all')
    print(ds,'cross-label',r['raw_matched_pairs'],'->',r['delex_matched_pairs'],'/',r['eligible_pairs'],
          'rates',round(r['raw_collision_pct'],4),round(r['delex_collision_pct'],4),
          'new cross/all',r['newly_matched_pairs'],allrow['newly_matched_pairs'])
report=['# 纯脱敏部分：自然问句配对跨标签碰撞检查','',
        '只比较自然 Test 问句之间的原文与脱敏表示，不使用生成键、K预算、RKGB或选句算法。对每个数据集枚举所有来源ID不同的无序问句对，固定记录对、BGE与阈值。category仅是任务代理；没有将样例标为双模型或人工确认的不同任务。',
        '', '| 数据集 | 异标签候选对数 | 原文匹配 | 脱敏匹配 | 原文率 | 脱敏率 | 净增pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
for ds in CONFIG:
    r=next(r for r in primary if r['dataset']==ds and r['stratum']=='different_label')
    report.append(f"| {ds} | {r['eligible_pairs']:,} | {r['raw_matched_pairs']:,} | {r['delex_matched_pairs']:,} | {r['raw_collision_pct']:.3f}% | {r['delex_collision_pct']:.3f}% | {r['delta_pp']:+.3f} |")
report+=['','该结果是第一部分碰撞实验的任务代理对照，与第二部分主集合的83.09%等覆盖结果分开。Banking77标签最接近原始意图；ABCD和MultiDoGO来源标签在派生请求中的保真度有额外限制。Nemotron/TAB文档类型不被当作任务真值。',
         '', '七个固定阈值与同标签/异标签/全部记录对的完整有符号结果见pair_collision_by_label.csv。区间使用按source_id删除的记录jackknife，未将相关的记录对视为独立样本。该补充分析是已有数据上的事后分析，不包含标签错误或数据生成不确定性。',
         '', 'new_cross_label_pair_samples_085.csv仅为每集随机20个新增异标签匹配的待审样例，没有开展新的模型评判。没有运行目标召回率标定，也没有调用生成/评判API。']
(OUT/'实验报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
print('Saved:',OUT)
