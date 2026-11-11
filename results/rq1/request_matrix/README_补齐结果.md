# 本次补齐说明

本次在用户开启的 RTX 3090 上并行运行四个编码器，新增三个请求集的完整碰撞矩阵。每个编码器完成约 4.94 万次文本编码；四路任务从启动到完成约 132–135 秒（含等待自动 PII 预处理），其后执行统计和校验。

## 直接查看

- RQ1_八数据组_0.85大表.md：文档与请求两组统一展示。
- RQ1_eight_groups_tau085.csv：320 个主配置，含原始/脱敏率、数量和记录级区间。
- RQ1_eight_groups_all_thresholds.csv：2880 行完整矩阵。
- collision_results_all_thresholds.csv：本次新增请求结果 1080 行。
- table_tau085.md：仅三个请求集的新主表。
- MERGE_VALIDATION.json：组合完整性与来源校验。
- server_evidence/：服务器输入、代码、原始向量、检测记录、日志与校验清单。

## 协议与边界

新请求组：Banking77、ABCD、MultiDoGO；MiniLM/GTE/E5/BGE；派生数据字段与 Presidio 两条来源；generic、typed、typed_numbered、delete、mask_length 五种替换；七个固定阈值 0.75 到 0.90、步长 0.025；两个原始校准上尾概率 0.1% 与 1%。

四种编码器共同检查 256-token 条件，含特殊符号及 E5 的 query: 前缀。所有测试记录和 Train 校准记录均保留。校准只用原始 Train，Test 不参与阈值选择。文档组保留其原有划分，未重新计算或修改。

派生字段通过原文与占位符文本的严格对齐恢复；3055 条实际变换的 Test 记录均通过唯一边界检查。一个训练记录的字段边界存在歧义，但训练侧仅使用原始文本做校准，其字段不参与任何替换或检测参照，故不影响结果；记录见服务器 validation.json。不得称其为新的人工作业标注。

Presidio Analyzer/Anonymizer 2.2.364，spaCy 3.8.16，en_core_web_lg 模型 3.8.0；默认英文识别器，阈值 0.5，REMOVE_INTERSECTIONS，merge_entities_with_spaces=False。PII 类型标签沿用各自来源，不强行视为相同标签体系。

新规则使用方括号占位符；原有请求数据的角括号脱敏版本单独评价。BGE 在主阈值 0.85 下的三组原始/脱敏计数均精确复现。其他阈值存在少量计数差异（单项最多 3 对），已核实全部变化对距阈值不超过 2.39e-7，来自 CPU/GPU float32 数值边界；保留新旧对照，不覆盖旧结果。数据组之间不要合并成一个总体碰撞率，也不要把固定阈值变化称为实际错误复用率。

完整性验证涵盖 1080 个请求配置、156 份向量文件、24 组独立计数复核和72项删除单记录检查。区间为固定校准阈值条件下的逐记录 jackknife 正态点区间，不包含阈值标定不确定性或多重比较校正。

服务器目录：/root/autodl-tmp/rq1_requests_complete_20261104。
新结果同时复制至 ICDE_v1/analysis/RQ1_八数据组_20261104；没有修改 main.tex、旧实验结果或 VLDB_v1。

## 归档校验

服务器归档 512 个文件均通过 SHA-256 校验。原始向量坐标最大差异 4.77e-7，配对相似度最大差异 7.75e-7；详见 TRANSFER_AND_NUMERICAL_VALIDATION.json、CPU_GPU_boundary_checks.csv。
