# RQ2 两张预览图

1. 01_geometry_gradient_ablation.png / .svg：主 RKGB 与去黎曼优化、去梯度提示的独立对照。Banking77 去黎曼采用完整独立复跑 73.40%，没有选用历史较高运行。MultiDoGO 的对照选择实现与主运行不同，以 * 标注；不把这些单次运行差值当作严格配对或显著性证据。去梯度提示保留双模型改写及束搜索，不是移除整个文本搜索模块。
2. 02_rounds_beam_sensitivity.png / .svg：轮数、束宽双面板；各点来自既有独立构造，不是对同一冻结集合改评分；主设置为 32 轮、宽度 50。保留实际非单调波动。

图仅供预览，未修改论文。PNG 用于直接查看，SVG 为可编辑矢量版。精确数值、来源路径与哈希见 plot_data_and_sources.json；复现脚本为 generate_figures.py。
