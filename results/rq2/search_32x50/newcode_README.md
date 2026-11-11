# ABCD 大预算实验代码（newcode）

`src/run_abcd_k5_trainonly.py` 是当前 24×20、28×24、32×28 三组实验实际运行的入口；`hybrid_search.py` 实现 BGE 编码、HotFlip 与困惑度计算，`portfolio.py` 实现组合选择。三组使用同一份代码，只调整 `--steps-per-slot` 和 `--beam-width`。

`src/run_abcd_k5.py` 是原有包含最终 Validation/Test 评估的入口，本轮搜索没有运行它。`inputs/abcd` 包含脱敏数据、冻结目标向量和五句起点。`provenance` 保存当时的服务器启动脚本与记录，其中早期脚本包含后来已停止的 Banking77 试跑；当前只运行 ABCD。

远端运行代码：`/root/autodl-tmp/ykx_large_budget_20261030/code/`。原始运行时的文件清单及 `hybrid_search.py` 保存在 `provenance/`，目标和起点的 SHA-256 已与启动时上传的文件核对。API 密钥和模型权重未包含。`config/api_providers.example.json` 用环境变量读取两套密钥，是无密钥配置示例，因此其 SHA-256 与远端真实配置不同；模型与 API 端点相同。

示例命令（Linux，需先提供 BGE 与 Qwen2.5 模型路径）：

```bash
export QWEN_API_KEY='…' DEEPSEEK_API_KEY='…'
python -u src/run_abcd_k5_trainonly.py \
  --data-dir inputs/abcd \
  --model-dir /path/to/bge-small-en-v1.5 \
  --target-file inputs/abcd/frozen_targets.npy \
  --expected-train-sha 701bc13ed4d2bab2bdc800a612994844bb57e6bac36d4d9dc02b99dc0190f3e6 \
  --expected-target-sha dfa29a0ebe7a0a16ad3f012a36253ae1a8d9c010d163e6270d70093cbf9d366d \
  --seeds-file inputs/abcd/seed_selection.json \
  --api-config config/api_providers.example.json \
  --lm-dir /path/to/Qwen2.5-1.5B-Instruct \
  --out-dir runs/abcd_24x20 --device cuda --threshold 0.85 \
  --steps-per-slot 24 --beam-width 20
```

其余两组分别改为 `--steps-per-slot 28 --beam-width 24` 和 `--steps-per-slot 32 --beam-width 28`，并使用不同输出目录。当前搜索只读取 Train；Val/Test 的独立评估应在最终句子通过语言审查后进行。

## 可配置新入口

`src/run_configurable.py` 在所选 Train 数据上重新编码，运行与 `v10_current/src/riemann_common.py` 相同的 KMeans→黎曼球面优化，再按边际 Train 覆盖排序目标。每个目标从其自然 Train 锚点选一条起始句。随后复用本包的 HotFlip 引导、双 API 改写、束搜索、困惑度过滤和组合选择。原来运行中的 `run_abcd_k5_trainonly.py` 保留不变。

参数：

- `--dataset abcd|banking77|dogo`，其中 `dogo` 对应 MultiDoGO；默认 K 分别为 5、15、10，也可用 `--k` 改。
- `--representation desensitized|non_desensitized`。默认数据目录为 `~/Desktop/ykx/data/<dataset>/<representation>`；远端可用 `--data-root` 或 `--data-dir` 指定。
- `--encoder bge|e5|gte|minilm`。默认模型目录为 `~/Desktop/ykx/<model-name>`；远端可用 `--models-root` 或 `--model-dir` 指定。BGE 用 CLS 池化，其他三个用均值池化，E5 加 `query: ` 前缀，向量均归一化。
- `--threshold` 只能选 `0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9`。改变阈值、K、种子、数据或模型时必须使用新的 `--out-dir`，避免误用目标或断点。

示例（仅准备目标，不调用 API 或困惑度模型）：

```bash
python src/run_configurable.py --dataset dogo --representation desensitized \
  --encoder gte --threshold 0.8 --prepare-only --out-dir runs/dogo_delex_gte_t080 \
  --data-root /path/to/data --models-root /path/to/models --device cuda
```

完整搜索在上面的命令中去掉 `--prepare-only`，增加 `--api-config config/api_providers.example.json --lm-dir /path/to/Qwen2.5-1.5B-Instruct --steps-per-slot 24 --beam-width 20`。ABCD 候选须属于八类客服流程之一，Banking77 须是银行客服问题，MultiDoGO 须属于六个 Train 领域之一；本地 Qwen 通顺度判断和双 API 判断都使用对应数据集的领域范围。脱敏候选保留相同的占位符集合及次数；未脱敏候选没有这一占位符约束。搜索与选句只读 Train；先写出冻结的 `train_report.json`，然后读取 Test。最终覆盖率以 `final_report.json` 的 `metrics.test.coverage` 为准，Train 覆盖率只作诊断；Validation 不参与。

历史运行时的 `hybrid_search.py` 原件与原 SHA-256 清单保存在 `provenance/`；当前目录的 SHA-256 清单对应修改后的代码。当前运行中的远端任务文件没有被改动。

通用入口对每次 API 改写和 API 质量判断设置 60 秒 HTTP 超时；在 Linux 主线程上还有 90 秒硬上限。单轮中某家 API 连续失败三次会明确报错并保留上一个完成句位的断点和已写入的 API 缓存，而不会无限等待。继续运行时使用同一 `--out-dir` 即可复用缓存。2026-10-30 的 32×28 运行自行恢复并完成，仍使用原脚本；这个保护补丁未参与该结果。
