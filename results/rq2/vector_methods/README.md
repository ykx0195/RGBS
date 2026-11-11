# Train 向量与 Validation 覆盖率实验

本目录包含 Banking77、ABCD、MultiDoGO 的脱敏和未脱敏两个版本，合计六套数据的结果。五种方法分别在各自的 Train 上生成目标向量，仅在对应的 Validation 上计算覆盖率；Test 没有参与。句向量由本地 `BAAI/bge-small-en-v1.5` 生成，并做 L2 归一化。

覆盖率定义：对每条 Validation 向量，若它与任一目标向量的余弦相似度大于或等于当前阈值，则计为覆盖；覆盖率为覆盖条数除以 Validation 总条数。

- `metrics.csv`：全部 1050 个组合的结果，包含请求的 `k`、实际向量数、Train 和 Validation 覆盖数。
- `plots/threshold_curves_k*.png`：固定 `k`，横轴为七个余弦阈值；共五张。
- `plots/k_curves_t*.png`：固定阈值，横轴为五个 `k` 值；共七张。
- `embeddings/`：六套数据的 Train、Validation 句向量。
- `cache/`：向量生成过程的缓存。KMeans 和球面 KMeans 按数据版本及 `k` 缓存；KMeans＋黎曼复用 KMeans 分簇，但每个阈值都重新优化。贪婪和纯黎曼按阈值分别运行到 `k=15`，较小 `k` 使用其前缀。
- `run_info.json`：模型哈希、参数与运行口径。
- `run.log`：完整远端运行日志。

阈值为 `0.750、0.775、0.800、0.825、0.850、0.875、0.900`；`k` 为 `1、3、5、10、15`。随机种子为 42。黎曼优化中的阈值与对应的覆盖率阈值同步。

纯黎曼在部分低阈值组合中提前覆盖完 Train 或不再产生新增覆盖，导致实际向量数小于请求的 `k`。`metrics.csv` 的 `actual_k` 记录了这一情况（共 37 个结果点）。其余结果点的实际向量数与请求的 `k` 一致。

计算脚本位于上一级目录的 `run_coverage_study.py`，方法代码位于上一级目录的 `vector_methods/`。本次远端计算从 2026-10-09 22:14:06 至 22:20:27，共约 6 分 21 秒（不含传输时间）。
