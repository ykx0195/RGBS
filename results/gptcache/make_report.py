"""Build the human-readable report from completed, verified server results."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'outputs'
summary=json.loads((out/'summary.json').read_text('utf-8'))
verification=json.loads((out/'verification.json').read_text('utf-8'))
assert verification['complete'] and len(summary)==6
protocol=json.loads((out/'protocol.json').read_text('utf-8'))
lines=['# GPTCache 两组缓存实验（2026-11-04）','',
'实际运行位置：`/root/autodl-tmp/icde_gptcache_20261104`。所有结果来自真实 GPTCache put/get 调用，使用 SQLite 保存条目、FAISS Flat 检索；并非只重新统计离线相似度。','',
'## 结果','',
'| 数据集 | 配置 | 缓存条目数（去重后） | 实际命中 / Test | 实际返回率 |',
'|---|---|---:|---:|---:|']
for r in summary:
    condition='仅构造文本' if r['condition']=='constructed_only' else '仅正常 Train 文本'
    lines.append(f"| {r['dataset']} | {condition} | {r['unique_cache_entries']:,} | {r['cache_hits']:,} / {r['test_total']:,} | {r['hit_rate_percent']:.2f}% |")
lines += ['', '正常组放入全部 Train 问题，构造组使用论文指定的 K 条冻结文本。两组缓存大小不同，因此此表不是同预算算法优劣比较。', '',
'## 协议与验证','',
'- 使用脱敏（delexed）Train/Test，直接预填缓存；未测自然请求的缓存准入。',
'- 构造文本分别为 Banking77 15 条、ABCD 5 条、MultiDoGO 10 条。MultiDoGO 使用论文 909/1094 的指定主集合。',
'- BGE-small-en-v1.5、CLS pooling、L2 归一化、float32，模型 SHA-256 与论文记录一致。',
'- 使用自定义原始余弦评估器，使 GPTCache 阈值 0.85 明确表示 cosine >= 0.85；top_k=1，确定性返回首个合格结果。该配置不是 GPTCache 默认配置。',
'- 编码回调复用经过模型、文本顺序和来源哈希核验的论文冻结向量；冻结文件中缺失的主集合文本在本服务器重新编码。不修改向量。日志耗时不代表包含在线编码或 LLM 推理的端到端延迟。',
'- 每条缓存值为唯一来源标记，通过实际 get 返回值确定条目来源。没有生成、评判真实业务答案，没有调用外部 LLM API。',
'- 每个 Test 的实际命中与独立矩阵计算逐条一致，命中条目的相似度和最相似值也经过核对。',
'- 每组测试前后的缓存问题数一致，Test 未写入缓存；原始输入文件哈希已验证。',
'- 三组构造文本实际命中数均须通过断言与论文主结果一致，只有全部通过才产生完成标记。', '',
'## 结论边界','',
'这些实验验证：在指定编码器、阈值、直接预填、无正常条目竞争的构造缓存中，冻结文本确实能通过 GPTCache 返回缓存值。正常组提供独立的常规缓存参照。', '',
'尚不能据此声称：正常条目竞争下的投毒成功率、通过普通请求成功植入全部构造句子的概率，或实际错误答案复用率。唯一来源标记是观测工具，不是无效答案的证据。', '',
'## 可复现材料','',
'- `run_experiment.py`：服务端实验脚本。',
'- `inputs/manifest.json`：输入和来源哈希。',
'- `outputs/protocol.json`：模型、依赖版本、检索及阈值设置。',
'- `outputs/summary.json`、`outputs/verification.json`：结果与通过状态。',
'- `outputs/<dataset>/<condition>/query_results.csv`：逐条 Test 的问题、返回标记、选中缓存问题及相似度。',
'- 远端保留实际 SQLite、FAISS 索引、批量编码文件与完整日志；本地结果包不重复下载大型索引和编码文件。', '',
'依赖版本：`'+json.dumps(protocol['versions'],ensure_ascii=False)+'`。','']
(root/'实验报告.md').write_text('\n'.join(lines),'utf-8')
print('\n'.join(lines[:18]))
