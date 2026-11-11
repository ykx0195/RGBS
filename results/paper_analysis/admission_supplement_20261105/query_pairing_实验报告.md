# 已入库新集合：固定键的原文／去标识化查询配对比较

日期：2026-11-05。仅使用已经正常顺序入库的新集合，K 为 15/5/10。没有重新选句、生成或改写文本，没有新增编码、模型调用或缓存服务运行。论文正文尚未修改。

## 核心结果

固定同一组缓存键、BGE 编码器、余弦阈值 **≥ 0.85**，只切换同源 Test 请求的原文／去标识化表示。

| 数据集 | 固定键数 | 原文查询覆盖 | 去标识化查询覆盖 | 增加 | 增加的 95% 配对区间 |
|---|---:|---:|---:|---:|---|
| Banking77 | 15 | 233/1,485（15.69%） | 1,083/1,485（72.93%） | **+57.24 pp** | [54.75, 59.80] pp |
| ABCD | 5 | 101/476（21.22%） | 305/476（64.08%） | **+42.86 pp** | [38.45, 47.48] pp |
| MultiDoGO | 10 | 235/1,094（21.48%） | 909/1,094（83.09%） | **+61.61 pp** | [58.77, 64.63] pp |

三组新增覆盖分别为 850、204、674 条。在这些固定集合和该阈值下，没有出现原文已覆盖、去标识化后失去覆盖的请求。这是查询是否被任意键覆盖的结果，不表示每个键—查询对的相似度都增加。

## 比较对象与控制变量

- 使用实际新缓存的全部条目及其顺序，逐项核对入库文本与冻结选句文件；两种查询条件共用完全相同的 15/5/10 个键向量。
- 新键集合的句间约束仍为严格 cosine < 0.85；该约束没有因本次查询比较改变。
- 原文与去标识化 Test 的 `source_id`、`variant_id`、split 和顺序逐项一致，共 3,055 个唯一来源记录。
- 本文“原文”指论文派生请求集合中带具体字段值的 materialized 请求，与对应占位符请求配对；不是另外换用一批官方 benchmark 原始句。
- 只复用已保存的原文和去标识化查询向量。历史查询向量档案内附带的键向量没有参与本次计算；固定键来自实际新缓存的冻结输入。
- 编码器、CLS pooling、L2 归一化、模型权重哈希、查询文件哈希与已有记录一致，覆盖阈值固定为 ≥ 0.85。

## 计算及验证

1. 对固定键分别计算原文查询和去标识化查询的相似度矩阵；每个查询只要有至少一个键达到阈值就计为覆盖一次。
2. 去标识化侧逐查询覆盖与真实 GPTCache 新入库实验记录完全一致，复现 1,083、305、909 次命中。
3. 原文侧是 **保持同一新缓存键的离线反事实对照**：以原文向量查询，不先执行去标识化。本次没有把它描述为另一轮实测服务请求。
4. 95% 区间使用 5,000 次来源对齐的记录级配对 bootstrap，seed=20261105；每次对两种表示采样相同索引。
5. 独立脚本以 float64 和不同点积计算路径重算全部查询，原文与去标识化的逐查询覆盖均与主计算一致；源文件哈希保持不变。

这些区间以固定键和当前 Test 集合为条件，不包含搜索随机性、其他工作负载或不同服务策略的变化。

## 可以支持的结论

在所测试固定阈值和指定表示下，已经能够正常顺序入库的少量新键，对查询侧去标识化仍表现出明显的覆盖扩张。该结果把“正常准入可行性”与“去标识化扩大固定键匹配范围”连接起来。

本轮不评判任务关系或答案适用性，不将新增覆盖直接解释为跨任务匹配或错误答案复用，也不推断任意生产去标识化管线都具有相同效果。

## 正文预览（尚未插入）

> With the admitted keys held fixed, we further compare source-aligned Test requests in their original and deidentified forms at the same cosine threshold of 0.85. Query-side deidentification increases coverage from 15.69% to 72.93% on Banking77, from 21.22% to 64.08% on ABCD, and from 21.48% to 83.09% on MultiDoGO. The corresponding increases are 57.24, 42.86, and 61.61 percentage points, with 95% paired bootstrap intervals of [54.75, 59.80], [38.45, 47.48], and [58.77, 64.63]. The deidentified-query outcomes match the recorded cache returns. The original-query condition is an offline fixed-key counterfactual. These results show that query-side deidentification expands threshold coverage even for the reselected sets that can be admitted sequentially.

## 材料

- `summary.json`、`summary.csv`：三数据集主结果和区间。
- `<dataset>/paired_queries.csv`：逐来源原文、占位符文本、覆盖状态、最高相似度及最佳键编号。
- `<dataset>/newly_covered_queries.csv`：新增覆盖请求。
- `<dataset>/lost_queries.csv`：失去覆盖请求，三组均为空。
- `<dataset>/similarities.npz`：两种查询表示对同一新键集合的相似度矩阵。
- `<dataset>/fixed_keys.json`：固定键文本与来源哈希。
- `protocol.json`、`verification.json`、`independent_verification.json`：协议、来源核验及独立复核。
- `analyze.py`、`verify.py`：分析与验证代码。
