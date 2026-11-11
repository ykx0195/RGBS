from pathlib import Path
import csv,hashlib,json,shutil,sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
root=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031';out=root/'evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csvwrite(p,rr):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
raw=read(out/'raw_summary.json');quality=read(root/'quality_results.json');frozen=read(out/'frozen_portfolios.json')
final=[];cost=[]
for row in raw['metrics']:
    dataset=row['dataset'];method=row['method'];rr=sorted([x for x in quality['results'] if x['dataset']==dataset and x['method']==method],key=lambda x:x['slot'])
    k=row['k_budget'];assert len(rr)==k
    if method=='CorpusPoisoning_adapted':
        matrix=np.load(out/dataset/'test_similarities.npz')['budget_5000']
        expected=[x['text'] for x in frozen[dataset]['portfolios']['5000']]
    else:
        base=Path.home()/'Desktop/results/固定文本_原文脱敏配对_20261031';base=base/'ABCD补充/abcd' if dataset=='abcd' else base/dataset
        matrix=np.load(base/'similarities.npz')['delex'];expected=read(base/'frozen_keys.json')['texts']
    assert [x['text'] for x in rr]==expected
    kept=np.array([x['quality_pass'] for x in rr],bool);filtered=(matrix[:,kept]>=.85).any(axis=1)
    final.append({**row,'local_quality_passed_texts':int(kept.sum()),'quality_filtered_test_covered':int(filtered.sum()),
                  'quality_filtered_test_coverage':float(filtered.mean()),'quality_judge':'Local Qwen PPL<=150 and local readability; both methods rechecked identically'})
for dataset,k in read(root/'experiment_plan.json')['datasets'].items():
    rr=[read(root/'runs'/dataset/f'cluster_{j:02d}/final_cluster.json') for j in range(k)]
    migrations=[read(p) for p in (root/'runs'/dataset).glob('cluster_*/device_migration.json') if p.stat().st_size>0]
    cost.append({'dataset':dataset,'clusters':k,'iteration_budget_per_cluster':5000,'accepted_updates':sum(x['accepted_updates'] for x in rr),
                 'encoded_candidate_and_gradient_texts':sum(x['forward_texts'] for x in rr),'backward_calls':sum(x['backward_calls'] for x in rr),
                 'sum_cluster_active_seconds':sum(x['active_seconds'] for x in rr),'devices':','.join(sorted({x['config']['device'] for x in rr})),
                 'cpu_to_gpu_migrated_clusters':len(migrations),'saved_cpu_prefix_seconds':sum(x['source_active_seconds'] for x in migrations),
                 'api_calls':0,'note':'Summed concurrent worker active intervals, not end-to-end elapsed time or an equal-budget RKGB comparison.'})
csvwrite(out/'primary_comparison.csv',final);csvwrite(out/'compute_accounting.csv',cost);csvwrite(out/'local_quality_details.csv',quality['results'])
lines=['# Corpus Poisoning／HotFlip：语义缓存适配对比','',
       '基于官方公开HotFlip梯度词元评分函数，适配本研究的查询JSONL、BGE CLS归一化和Test余弦阈值覆盖。保存原始源码快照及SHA256；若GitHub提交API受限，provenance.json明确标为源码快照，不伪称已固定某个commit。',
       '', '固定seed42、K分别15/5/10、每簇5000次更新、每次100个词元候选。只用Train聚类和选择初始原句；每簇独立搜索，训练簇平均余弦相似度选择检查点，无黎曼优化、API改写或全局候选重选。查询梯度批次与候选评分批次分别随机抽样；质量评价不参与这项原始搜索。',
       '', '| 数据集 | 方法 | 原始Test总覆盖 | 通过共同本地质量筛选的句子数 | 筛选后Test总覆盖 |','|---|---|---:|---:|---:|']
for x in final:lines.append(f"| {x['dataset']} | {x['method']} | {x['test_coverage']:.2%} | {x['local_quality_passed_texts']}/{x['k_budget']} | {x['quality_filtered_test_coverage']:.2%} |")
lines+=['','所有Test覆盖以完整Test查询数为分母，阈值0.85。共同质量筛选为同一本地Qwen的PPL≤150和客服请求通顺性判断，两种方法全部重新评价。没有API修复、外部评判API、补齐被过滤句子或人工判断；不得称为双API验证或人工验证的自然性。',
        '', '运行期间按用户要求改为仅GPU调度，两个已有CPU任务从第300、500次迭代迁移至GPU继续，保留原检查点和迁移前备份。最终设备字段为cuda不表示全部历史迭代均在GPU执行；compute_accounting.csv单列迁移簇数和已保存CPU前缀时间，该时间已经计入累计活跃时间，不重复相加。',
        '', '与原官方实验的必要区别：不再使用BEIR相关文档排名或gold文档初始化；使用各Train簇中心最近的自然问题初始化、原始占位符保护、实际解码再编码后评分、全Train簇平均相似度保存检查点。不能把本表称为原BEIR攻击成功率复现。',
        '', '保留的原始结构是KMeans簇、随机词元位置、HotFlip梯度词表候选、实际相似度接受替换。只保护特殊词元和占位符，没有加入RKGB的字母词元限制、每轮最多3词规则或API润色。',
        '', '5000次词元更新不等于RKGB的32轮API改写，不能宣称等搜索预算。实际编码文本数、反向次数和各工作进程活跃区间之和见compute_accounting.csv；并发活跃时间之和不是整批任务的墙钟耗时。不同运行设备需结合记录理解。',
        '', '0、100、500、1000、2500、5000次的固定预算检查点全部仅按Train选择，所有检查点在搜索结束并冻结后才评价Test。曲线不允许按Test挑选最高预算点作为主结果。主结果固定使用5000次预算。',
        '', '本实验没有新增跨任务判断或答案复用评价。它用于比较少量高覆盖文本的构造能力和共同本地语言质量约束下的覆盖，不把总覆盖称为错误复用率。',
        '', '本次为一个种子的外部基线对照，不能代替多种子稳定性证据。所有原始字符串、初始文本、训练轨迹和检查点已保存；即使基线得分高于RKGB也如实报告。',
        '', '原论文及代码：https://github.com/princeton-nlp/corpus-poisoning 。详细预定配置见experiment_plan.json。']
(root/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'summary.json').write_text(json.dumps({'complete':True,'metrics':final,'cost':cost,'api_calls':0,'quality':quality['protocol'],'human_verified':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('CORPUS_HOTFLIP_COMPARISON_SUMMARIZED',flush=True)
