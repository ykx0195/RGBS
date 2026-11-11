"""Frozen-key, paired query-representation evaluation. Never search or change keys."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, shutil, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import transformers


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.code))
    from hybrid_search import BGE
    model_hash = sha(args.model / 'model.safetensors')
    args.out.mkdir(parents=True, exist_ok=True)
    print('MODEL_VERIFIED', model_hash, flush=True)
    encoder = BGE(args.model, 'cpu', 'bge')
    all_summaries = []
    for dataset in args.datasets:
        data_folder = 'multidogo' if dataset == 'dogo' else dataset
        out = args.out / dataset
        out.mkdir(exist_ok=True)
        source = args.runs / dataset / 'final_report.json'
        report = json.loads(source.read_text(encoding='utf-8'))
        assert report['config']['model_sha256'] == model_hash
        assert report['config']['threshold'] == 0.85
        texts = [row['text'] for row in report['selected']]
        assert len(set(texts)) == len(texts)
        raw_path = args.data / data_folder / 'non_desensitized' / 'test.jsonl'
        delex_path = args.data / data_folder / 'desensitized' / 'test.jsonl'
        raw = read_rows(raw_path)
        delex = read_rows(delex_path)
        assert len(raw) == len(delex) == report['metrics']['test']['total']
        assert sha(delex_path) == report['test_sha256']
        ids_raw = [(row['source_id'], row['variant_id']) for row in raw]
        ids_delex = [(row['source_id'], row['variant_id']) for row in delex]
        assert len(set(ids_raw)) == len(raw)
        assert ids_raw == ids_delex, 'Source/variant order differs'
        for r, d in zip(raw, delex):
            for field in ('category', 'flow', 'domain', 'official_intent', 'banking77_source_text', 'multidogo_source_text', 'abcd_source_text'):
                assert r.get(field) == d.get(field), (field, r['source_id'])
            assert r['split'] == d['split'] == 'test'
        key_bytes = json.dumps(texts, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        identity = {
            'source_report_sha256': sha(source), 'key_texts_sha256': hashlib.sha256(key_bytes).hexdigest(),
            'raw_test_sha256': sha(raw_path), 'delex_test_sha256': sha(delex_path),
            'model_sha256': model_hash, 'pooling_sha256': sha(args.model/'1_Pooling/config.json'),
            'tokenizer_sha256': sha(args.model/'tokenizer.json'),
            'max_seq_length': encoder.max_seq_length,
            'code_sha256': sha(args.code/'hybrid_search.py'),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'device': 'cpu', 'dtype': 'float32', 'batch_size': 64,
        }
        identity_path = out/'embedding_identity.json'
        cache = out/'embeddings.npz'
        if cache.exists():
            assert json.loads(identity_path.read_text(encoding='utf-8')) == identity, 'Cache identity differs'
            vectors = np.load(cache)
            kv, rv, dv = vectors['keys'], vectors['raw'], vectors['delex']
            print(dataset, 'CACHED', flush=True)
        else:
            print(dataset, 'ENCODING_KEYS', len(texts), flush=True)
            kv = encoder.encode(texts, 64)
            print(dataset, 'ENCODING_RAW', len(raw), flush=True)
            rv = encoder.encode([row['text'] for row in raw], 64)
            print(dataset, 'ENCODING_DELEX', len(delex), flush=True)
            dv = encoder.encode([row['text'] for row in delex], 64)
            np.savez_compressed(cache, keys=kv, raw=rv, delex=dv)
            save(identity_path, identity)
        assert np.allclose(np.linalg.norm(kv,axis=1), 1, atol=1e-5)
        sr, sd = rv @ kv.T, dv @ kv.T
        np.savez_compressed(out/'similarities.npz', raw=sr, delex=sd)
        mr, md = sr >= 0.85, sd >= 0.85
        cr, cd = mr.any(axis=1), md.any(axis=1)
        archived = report['metrics']['test']['covered']
        reproduced = int(cd.sum())
        if reproduced != archived:
            save(out/'baseline_reproduction_warning.json', {
                'archived': archived, 'reproduced': reproduced,
                'nearest_delex_threshold_distances': sorted(np.abs(sd.max(axis=1)-0.85).tolist())[:20],
            })
            raise RuntimeError(f'{dataset}: delex baseline {reproduced} != archived {archived}; inspect before interpreting')
        curve = []
        for tau in [0.75,0.775,0.8,0.825,0.85,0.875,0.9]:
            ar, ad = (sr>=tau).any(axis=1), (sd>=tau).any(axis=1)
            curve.append({'dataset':dataset,'k':len(texts),'threshold':tau,'total':len(raw),
                'raw_covered':int(ar.sum()),'delex_covered':int(ad.sum()),
                'both_covered':int((ar&ad).sum()),'delex_only':int((~ar&ad).sum()),
                'raw_only':int((ar&~ad).sum()),'neither':int((~ar&~ad).sum()),
                'raw_coverage':float(ar.mean()),'delex_coverage':float(ad.mean()),
                'delta_pp':float((ad.mean()-ar.mean())*100)})
        main = dict(next(x for x in curve if x['threshold']==0.85))
        assert main['delex_only']-main['raw_only']==main['delex_covered']-main['raw_covered']
        # Query-level paired bootstrap, frozen texts and threshold; not search-seed uncertainty.
        diff=cd.astype(np.int8)-cr.astype(np.int8)
        rng=np.random.default_rng(20261031)
        boot=[]
        for _ in range(5000):
            boot.append(float(diff[rng.integers(0,len(diff),size=len(diff))].mean()))
        main['delta_pp_paired_query_bootstrap_95']=(np.quantile(boot,[.025,.975])*100).tolist()
        main['bootstrap_note']='Fixed portfolio, paired source-aligned queries, 5000 resamples. Does not include search or dataset-source uncertainty.'
        paired=[]
        for i,(r,d) in enumerate(zip(raw,delex)):
            state='both' if cr[i] and cd[i] else 'delex_only' if cd[i] else 'raw_only' if cr[i] else 'neither'
            paired.append({'test_index':i,'source_id':r['source_id'],'variant_id':r['variant_id'],
                'category':r.get('category',''),'domain':r.get('domain',''),'official_intent':r.get('official_intent',''),
                'status':state,'raw_max_cosine':float(sr[i].max()),'delex_max_cosine':float(sd[i].max()),
                'raw_best_key':int(sr[i].argmax()),'delex_best_key':int(sd[i].argmax()),
                'raw_matching_keys':json.dumps(np.flatnonzero(mr[i]).tolist()),
                'delex_matching_keys':json.dumps(np.flatnonzero(md[i]).tolist()),
                'raw_query':r['text'],'delex_query':d['text']})
        candidates=[]
        for i,j in np.argwhere(mr|md):
            i,j=int(i),int(j)
            candidates.append({'pair_id':f'{dataset}:{i}:{j}','dataset':dataset,
                'test_index':i,'key_index':j,'source_id':delex[i]['source_id'],
                'key_text':texts[j],'query_text':delex[i]['text'],
                'raw_query_text':raw[i]['text'], 'query_status':paired[i]['status'],
                'raw_matches':bool(mr[i,j]),'delex_matches':bool(md[i,j]),
                'raw_cosine':float(sr[i,j]),'delex_cosine':float(sd[i,j])})
        save(out/'audit_pairs.json',{'protocol':'Union of raw/delex threshold pairs. Judge key and canonical delex query once; share task label across representations. Numeric scores, source categories and representation status withheld from judges.',
             'threshold':0.85,'pairs':candidates})
        save(out/'frozen_keys.json', {'texts':texts,'source':str(source),'source_sha256':sha(source)})
        shutil.copy2(source,out/'source_final_report.json')
        csv_write(out/'paired_queries.csv',paired)
        csv_write(out/'newly_covered_queries.csv',[row for row in paired if row['status']=='delex_only'])
        csv_write(out/'threshold_curve.csv',curve)
        save(out/'coverage_summary.json',{'protocol':'Frozen latest delexed keys; only Test query representation changes. No search, no answer generation, no cache simulation.',
            'identity':identity,'source_alignment':{'count':len(raw),'source_id_and_variant_id_order_equal':True,'metadata_equal':True},
            'baseline_exactly_reproduced':True,'main':main,'thresholds':curve,
            'task_audit_pairs':len(candidates),'newly_covered_queries':int((~cr&cd).sum()),
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}})
        all_summaries.append(main)
        print('RESULT',json.dumps(main,ensure_ascii=False), 'AUDIT_PAIRS',len(candidates),flush=True)
    csv_write(args.out/'coverage_comparison.csv',all_summaries)
    print('COVERAGE_COMPLETE', str(args.out), flush=True)


if __name__=='__main__':
    home=Path.home()
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,default=home/'Desktop/results/newcode/src')
    p.add_argument('--model',type=Path,default=home/'Desktop/ykx/bge-small-en-v1.5')
    p.add_argument('--data',type=Path,default=home/'Desktop/ykx/data')
    p.add_argument('--runs',type=Path,default=home/'Desktop/results/三数据集_脱敏_32x50_20261030/runs')
    p.add_argument('--out',type=Path,default=home/'Desktop/results/固定文本_原文脱敏配对_20261031')
    p.add_argument('--datasets',nargs='+',choices=['banking77','dogo','abcd'],default=['banking77','dogo'])
    evaluate(p.parse_args())
