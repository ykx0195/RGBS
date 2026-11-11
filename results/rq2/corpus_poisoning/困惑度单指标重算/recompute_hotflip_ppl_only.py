"""Recompute the frozen HotFlip/RKGB comparison using PPL<=150 alone."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import shutil

import numpy as np


BASE = Path.home() / 'Desktop' / 'results'
ROOT = BASE / 'CorpusPoisoning_HotFlip对比_20261031'
OUT = ROOT / '困惑度单指标重算'
OUT.mkdir(exist_ok=True)
THRESHOLDS = (0.8, 0.825, 0.85, 0.875, 0.9)
PPL_LIMIT = 150.0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, records):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


quality_path = ROOT / 'quality_results.json'
quality = read(quality_path)
assert quality['api_calls'] == 0 and quality['text_repaired'] is False
assert len(quality['results']) == 60
frozen = read(ROOT / 'evaluation' / 'frozen_portfolios.json')
raw = list(csv.DictReader((ROOT / 'evaluation' / 'primary_raw_comparison.csv').open(encoding='utf-8-sig', newline='')))
assert len(raw) == 6

primary = []
all_thresholds = []
details = []
source_hashes = {'quality_results.json': sha(quality_path)}
for dataset, selection in frozen.items():
    k = selection['k']
    for method in ('CorpusPoisoning_adapted', 'RKGB'):
        rows = sorted((x for x in quality['results'] if x['dataset'] == dataset and x['method'] == method),
                      key=lambda x: x['slot'])
        assert len(rows) == k and [x['slot'] for x in rows] == list(range(k))
        if method == 'CorpusPoisoning_adapted':
            expected = [x['text'] for x in selection['portfolios']['5000']]
            matrix_path = ROOT / 'evaluation' / dataset / 'test_similarities.npz'
            matrix = np.load(matrix_path)['budget_5000']
        else:
            prior = BASE / '固定文本_原文脱敏配对_20261031'
            prior = prior / 'ABCD补充' / 'abcd' if dataset == 'abcd' else prior / dataset
            expected = read(prior / 'frozen_keys.json')['texts']
            matrix_path = prior / 'similarities.npz'
            matrix = np.load(matrix_path)['delex']
        assert [x['text'] for x in rows] == expected
        assert matrix.shape[1] == k and matrix.shape[0] > 0
        source_hashes[str(matrix_path)] = sha(matrix_path)
        keep = np.array([np.isfinite(x['ppl']) and x['ppl'] <= PPL_LIMIT for x in rows], dtype=bool)
        assert all(bool(x['ppl_gate']) == bool(y) for x, y in zip(rows, keep))
        for x, retained in zip(rows, keep):
            details.append({'dataset': dataset, 'method': method, 'slot': x['slot'], 'text': x['text'],
                            'ppl': x['ppl'], 'ppl_at_most_150': bool(retained)})
        reference = next(x for x in raw if x['dataset'] == dataset and x['method'] == method)
        for threshold in THRESHOLDS:
            unfiltered = (matrix >= threshold).any(axis=1)
            filtered = (matrix[:, keep] >= threshold).any(axis=1)
            result = {'dataset': dataset, 'method': method, 'k_budget': k, 'threshold': threshold,
                      'test_total': int(matrix.shape[0]), 'raw_test_covered': int(unfiltered.sum()),
                      'raw_test_coverage': float(unfiltered.mean()),
                      'ppl_passed_texts': int(keep.sum()), 'ppl_filtered_test_covered': int(filtered.sum()),
                      'ppl_filtered_test_coverage': float(filtered.mean()), 'ppl_limit': PPL_LIMIT}
            assert result['ppl_filtered_test_covered'] <= result['raw_test_covered']
            if threshold == 0.85:
                assert result['raw_test_covered'] == int(reference['test_covered'])
                assert result['test_total'] == int(reference['test_total'])
                primary.append(result)
            all_thresholds.append(result)

assert len(primary) == 6 and len(details) == 60 and len(all_thresholds) == 30
assert Counter((x['method'], x['ppl_at_most_150']) for x in details) == {
    ('CorpusPoisoning_adapted', False): 28, ('CorpusPoisoning_adapted', True): 2,
    ('RKGB', True): 30,
}
write_csv(OUT / 'PPL单指标主表_085.csv', primary)
write_csv(OUT / 'PPL单指标五阈值.csv', all_thresholds)
write_csv(OUT / '逐句PPL判定.csv', details)
payload = {'complete': True, 'rule': 'Retain final text iff finite local Qwen2.5-1.5B-Instruct PPL <= 150; ignore readability verdict.',
           'selection': 'Final portfolios already frozen; no reoptimization or backfill.', 'primary_threshold': 0.85,
           'test_coverage_definition': 'Fraction of all held-out Test queries with max cosine to retained text >= threshold.',
           'quality_in_search_caveat': 'RKGB candidate search used this same local model for PPL gating; original seeds were admitted without that gate.',
           'human_review': False, 'new_model_calls': 0, 'api_calls': 0,
           'quality_model_device': quality['device'], 'quality_model_dtype': quality['dtype'],
           'source_sha256': source_hashes, 'metrics': primary}
(OUT / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
lines = ['# HotFlip 与 RKGB：仅按困惑度重算', '',
         '保留各方法已冻结的最终句子中 PPL≤150 的文本，忽略通顺性 YES／NO 判断；用保留句子重算完整 Test 在余弦阈值 0.85 下的覆盖率。',
         '', '| 数据集 | 方法 | 原始 Test 覆盖 | PPL 合格句子 | 仅按 PPL 筛选后覆盖 |',
         '|---|---|---:|---:|---:|']
for x in primary:
    name = 'HotFlip' if x['method'] == 'CorpusPoisoning_adapted' else 'RKGB'
    dataset_name = 'MultiDoGO' if x['dataset'] == 'dogo' else ('ABCD' if x['dataset'] == 'abcd' else 'Banking77')
    lines.append(f"| {dataset_name} | {name} | {x['raw_test_covered']}/{x['test_total']} ({x['raw_test_coverage']:.2%}) | "
                 f"{x['ppl_passed_texts']}/{x['k_budget']} | "
                 f"{x['ppl_filtered_test_covered']}/{x['test_total']} ({x['ppl_filtered_test_coverage']:.2%}) |")
lines += ['', 'PPL 来自原有本地 Qwen2.5-1.5B-Instruct 逐句结果；计算时将类型占位符展开为普通词组。此次只重算已有分数和相似度矩阵，没有调用模型或 API。',
          '', '这是自动 PPL 门槛结果。PPL 衡量语言模型预测文本的难易程度，不能单独证明句子通顺、意思合理或任务正确。RKGB 的改写候选在搜索时用同一模型筛过 PPL，因此这项指标对 RKGB 不是独立质量验证。',
          '', '五个阈值的完整结果见 PPL单指标五阈值.csv；逐句判定见 逐句PPL判定.csv。原先 PPL＋通顺性联合筛选表仍保留。']
(OUT / '报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
shutil.copyfile(Path(__file__), OUT / 'recompute_hotflip_ppl_only.py')
print(json.dumps(primary, ensure_ascii=False, indent=2))
