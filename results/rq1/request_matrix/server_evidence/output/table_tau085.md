# 三请求集 RQ1 完整矩阵（新实验）

固定阈值 0.85；数值为碰撞率变化，单位百分点。

| 数据集 | 编码器 | PII 来源 | Generic | Type | Type+index | Delete | Length mask |
|---|---|---|---:|---:|---:|---:|---:|
| banking77 | minilm | dataset_fields | +0.358 | +0.182 | +0.206 | +0.040 | +0.092 |
| banking77 | minilm | presidio | +0.107 | +0.019 | +0.024 | -0.018 | +0.001 |
| banking77 | gte | dataset_fields | +44.958 | +5.383 | +12.859 | -11.699 | -13.320 |
| banking77 | gte | presidio | +15.230 | -5.571 | -0.757 | -9.005 | -10.776 |
| banking77 | e5 | dataset_fields | +45.049 | +23.084 | +31.619 | +5.342 | +20.657 |
| banking77 | e5 | presidio | +12.405 | +1.580 | +6.630 | -4.328 | +5.296 |
| banking77 | bge | dataset_fields | +4.910 | +1.746 | +2.237 | +0.578 | +1.004 |
| banking77 | bge | presidio | +2.378 | +0.379 | +0.615 | +0.183 | +0.404 |
| abcd | minilm | dataset_fields | +0.295 | +0.209 | +0.194 | +0.135 | +0.144 |
| abcd | minilm | presidio | +0.085 | -0.007 | -0.009 | +0.027 | +0.024 |
| abcd | gte | dataset_fields | +29.679 | +8.157 | +10.322 | -2.985 | -5.505 |
| abcd | gte | presidio | +17.956 | +1.798 | +4.564 | -1.357 | -3.974 |
| abcd | e5 | dataset_fields | +40.435 | +19.981 | +25.006 | +9.160 | +23.498 |
| abcd | e5 | presidio | +19.754 | +6.164 | +10.908 | +2.005 | +11.025 |
| abcd | bge | dataset_fields | +4.885 | +1.494 | +1.708 | +0.456 | +1.209 |
| abcd | bge | presidio | +2.617 | +0.300 | +0.438 | +0.242 | +0.590 |
| multidogo | minilm | dataset_fields | +1.400 | +1.203 | +1.125 | +0.334 | +0.528 |
| multidogo | minilm | presidio | +0.555 | -0.145 | -0.122 | -0.049 | -0.042 |
| multidogo | gte | dataset_fields | +43.969 | +5.083 | +14.142 | +0.275 | -2.479 |
| multidogo | gte | presidio | +36.239 | +6.222 | +16.841 | +1.049 | -0.729 |
| multidogo | e5 | dataset_fields | +56.236 | +21.093 | +31.699 | +18.693 | +38.880 |
| multidogo | e5 | presidio | +39.964 | +15.871 | +25.375 | +8.619 | +20.512 |
| multidogo | bge | dataset_fields | +4.139 | +2.661 | +3.104 | +1.291 | +1.694 |
| multidogo | bge | presidio | +3.223 | +1.105 | +1.845 | +0.728 | +1.342 |

完整矩阵 1080 行，0.85 主表 120 行。原始/脱敏碰撞率、匹配数量和记录级区间见 CSV。

dataset_fields 来自派生请求的原文/占位符严格对齐，不是新增人工标注。Presidio 使用独立的自动识别结果。

校准沿用 benchmark Train，评价沿用 Test；文档数据使用其原有 60/40 划分。两组分组展示，不合并为同一总体。

新规则采用方括号占位符，与旧角括号脱敏表示不同。旧表示另列 legacy_delex_comparison.csv，不覆盖旧结果。
