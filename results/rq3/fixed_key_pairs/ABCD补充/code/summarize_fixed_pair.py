from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, shutil, sys
import numpy as np


def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def paired_ci(values, seed=20261031):
    rng=np.random.default_rng(seed)
    estimates=[values[rng.integers(0,len(values),size=len(values))].mean() for _ in range(5000)]
    return (np.quantile(estimates,[.025,.975])*100).tolist()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.home()/'Desktop/results/固定文本_原文脱敏配对_20261031')
    parser.add_argument('--datasets',nargs='+',choices=['banking77','dogo','abcd'],default=['banking77','dogo'])
    args=parser.parse_args()
    root=args.root
    names={'banking77':'Banking77','dogo':'MultiDoGO','abcd':'ABCD'}
    dataset_names='、'.join(names[x] for x in args.datasets)
    judgments_path=root/'task_judgments.jsonl'
    judgments=[json.loads(line) for line in judgments_path.read_text(encoding='utf-8').splitlines() if line.strip()] if judgments_path.exists() else []
    judge_manifest=json.loads((root/'task_judge_manifest.json').read_text(encoding='utf-8')) if judgments else None
    if judge_manifest:
        allowed={p['model'] for p in judge_manifest['providers']}
        excluded=[]
        for path in (root/'judge_cache').glob('*/*.json'):
            item=json.loads(path.read_text(encoding='utf-8'))
            if item.get('model') not in allowed:
                target=root/'provenance/excluded_initial_endpoint_probe'/path.parent.name/path.name
                target.parent.mkdir(parents=True,exist_ok=True)
                path.rename(target)
                excluded.append({'file':str(target.relative_to(root)),'model':item.get('model'),
                    'reason':'Initial discovery found an older model configuration; excluded from the final 2804-pair audit and all statistics.'})
        if excluded:save(root/'provenance/excluded_initial_probe.json',excluded)
    byid=defaultdict(dict)
    for row in judgments:
        assert row['provider'] not in byid[row['pair_id']]
        byid[row['pair_id']][row['provider']]=row
    summaries=[];main_audits=[];example_rows=[]
    report=['# 固定文本的原文／脱敏查询配对实验','',
        '日期：2026-10-31。本报告评价 '+dataset_names+'。',
        '','## 实验协议','',
        '- 固定最新 32 轮、束宽 50、seed 42、BGE、阈值 0.85 的最终文本；K 采用各正式报告中的最终句数。未重新搜索、改写或选择任何句子。',
        '- 同一组脱敏键同时匹配来源一一对应的原文 Test 与脱敏 Test。只有查询侧表示改变；键始终包含原有占位符。',
        '- 核查 source_id、variant_id、顺序及任务元数据一致，核查数据与模型哈希，并精确复现原正式报告的脱敏覆盖数。',
        '- 本地 CPU float32 使用原 newcode 的 BGE 编码类、相同 CLS 池化与 512-token 配置。没有使用远端 GPU，也没有变更既有搜索运行。',
        '- 覆盖是至少匹配一条固定键的查询比例。原文／脱敏的逐查询配对 bootstrap 区间基于 5,000 次重采样，条件于这组固定键与阈值；不包括搜索随机性或新来源的不确定性。',
        '', '## 覆盖结果', '',
        '| 数据集 | K | 原文查询 | 脱敏查询 | 新增覆盖 | 失去覆盖 | 净增，pp | 配对区间，pp |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for dataset in args.datasets:
        display=names[dataset]
        d=root/dataset
        s=json.loads((d/'coverage_summary.json').read_text(encoding='utf-8'))
        main=s['main'];summaries.append(main)
        ci=main['delta_pp_paired_query_bootstrap_95']
        report.append(f"| {display} | {main['k']} | {main['raw_covered']}/{main['total']}（{main['raw_coverage']:.2%}） | {main['delex_covered']}/{main['total']}（{main['delex_coverage']:.2%}） | {main['delex_only']} | {main['raw_only']} | +{main['delta_pp']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
        if not judgments:continue
        pairs=json.loads((d/'audit_pairs.json').read_text(encoding='utf-8'))['pairs']
        assert all(len(byid[p['pair_id']])==2 for p in pairs)
        query_rows=list(csv.DictReader((d/'paired_queries.csv').open(encoding='utf-8-sig')))
        n=len(query_rows)
        masks=np.load(d/'similarities.npz')
        raw_mask,delex_mask=masks['raw']>=.85,masks['delex']>=.85
        diff_matrix=np.zeros_like(raw_mask,dtype=bool)
        same_matrix=np.zeros_like(raw_mask,dtype=bool)
        unresolved_matrix=np.zeros_like(raw_mask,dtype=bool)
        pair_results=[];joint=Counter()
        providers=[p['name'] for p in judge_manifest['providers']]
        for p in pairs:
            jj=byid[p['pair_id']]
            a,b=[jj[name]['label'] for name in providers]
            joint[(a,b)]+=1
            verdict='BOTH_DIFFERENT_TASK' if a==b=='DIFFERENT_TASK' else 'BOTH_SAME_TASK' if a==b=='SAME_TASK' else 'UNRESOLVED'
            i,j=p['test_index'],p['key_index']
            diff_matrix[i,j]=verdict=='BOTH_DIFFERENT_TASK'
            same_matrix[i,j]=verdict=='BOTH_SAME_TASK'
            unresolved_matrix[i,j]=verdict=='UNRESOLVED'
            row={**p,'consensus':verdict}
            for name in providers:
                row[name+'_label']=jj[name]['label'];row[name+'_reason']=jj[name]['reason']
            pair_results.append(row)
        write_csv(d/'task_pair_audit.csv',pair_results)
        grouped={}
        for rep,mask in [('raw',raw_mask),('delex',delex_mask)]:
            cross=(mask&diff_matrix).any(axis=1)
            same=(mask&same_matrix).any(axis=1)
            unresolved=(mask&unresolved_matrix).any(axis=1)
            has=mask.any(axis=1)
            all_diff=has&np.all(~mask|diff_matrix,axis=1)
            grouped[rep]={'covered_queries':int(has.sum()),'matched_pairs':int(mask.sum()),
                'both_different_pairs':int((mask&diff_matrix).sum()),
                'both_same_pairs':int((mask&same_matrix).sum()),
                'unresolved_pairs':int((mask&unresolved_matrix).sum()),
                'different_pair_fraction':float((mask&diff_matrix).sum()/mask.sum()),
                'queries_with_any_both_different':int(cross.sum()),
                'cross_task_query_coverage':float(cross.mean()),
                'queries_with_any_both_same':int(same.sum()),
                'queries_with_unresolved_match':int(unresolved.sum()),
                'queries_all_matches_both_different':int(all_diff.sum())}
        raw_cross=(raw_mask&diff_matrix).any(axis=1)
        delex_cross=(delex_mask&diff_matrix).any(axis=1)
        newly=(~raw_mask.any(axis=1))&delex_mask.any(axis=1)
        new_stats={'queries':int(newly.sum()),
            'with_any_both_different_match':int((newly&delex_cross).sum()),
            'with_any_both_same_match':int((newly&(delex_mask&same_matrix).any(axis=1)).sum()),
            'with_unresolved_match':int((newly&(delex_mask&unresolved_matrix).any(axis=1)).sum()),
            'all_matches_both_different':int((newly&np.all(~delex_mask|diff_matrix,axis=1)).sum())}
        audit={'dataset':dataset,'threshold':.85,'total_queries':n,'judge_models':judge_manifest['providers'],
            'protocol':judge_manifest['protocol'],'human_audit_performed':False,'representations':grouped,
            'newly_covered_query_audit':new_stats,
            'cross_task_query_coverage_delta_pp':float((delex_cross.mean()-raw_cross.mean())*100),
            'cross_task_query_coverage_delta_pp_conditional_bootstrap_95':paired_ci(delex_cross.astype(int)-raw_cross.astype(int)),
            'union_pair_count':len(pairs),
            'joint_labels':[{'first':a,'second':b,'count':v} for (a,b),v in sorted(joint.items())],
            'interpretation':'Model-confirmed task-discordant threshold opportunities; not human ground truth, answer errors or actual cache selection. Unresolved includes disagreement and uncertainty and is not counted as same-task.'}
        save(d/'task_audit_summary.json',audit);main_audits.append(audit)
        qstats=[]
        for i,row in enumerate(query_rows):
            qstats.append({**row,'raw_any_both_different':bool(raw_cross[i]),'delex_any_both_different':bool(delex_cross[i]),
                'delex_both_different_keys':json.dumps(np.flatnonzero(delex_mask[i]&diff_matrix[i]).tolist()),
                'delex_both_same_keys':json.dumps(np.flatnonzero(delex_mask[i]&same_matrix[i]).tolist()),
                'delex_unresolved_keys':json.dumps(np.flatnonzero(delex_mask[i]&unresolved_matrix[i]).tolist())})
        write_csv(d/'query_task_audit.csv',qstats)
        # Human-review candidates selected deterministically, without cherry-picking verdicts.
        rng=np.random.default_rng(20261031)
        new_pairs=[p for p in pair_results if p['query_status']=='delex_only' and p['delex_matches']]
        indices=rng.choice(len(new_pairs),min(100,len(new_pairs)),replace=False)
        review=[]
        for index in sorted(indices):
            p=new_pairs[int(index)]
            review.append({'pair_id':p['pair_id'],'question_a':p['key_text'],'question_b':p['query_text'],
                'human_1_label':'','human_1_reason':'','human_2_label':'','human_2_reason':''})
        write_csv(d/'human_review_blinded_100.csv',review)
        # Select representative cases by similarity; this list is illustrative, not an estimator.
        for verdict in ['BOTH_DIFFERENT_TASK','BOTH_SAME_TASK','UNRESOLVED']:
            subset=[p for p in new_pairs if p['consensus']==verdict]
            subset=sorted(subset,key=lambda p:p['delex_cosine'],reverse=True)
            seen_queries=set()
            for row in subset:
                if row['test_index'] in seen_queries:continue
                seen_queries.add(row['test_index']);example_rows.append(row)
                if len(seen_queries)==3:break
    if main_audits:
        report += ['', '## 任务关系审计（双模型自动判断）','',
            '任务判断使用固定键文本与统一的脱敏查询文本；同一键—查询配对只判一次，标签用于原文／脱敏两种匹配矩阵。评判模型看不到相似度、原文／脱敏匹配状态、数据集类别或 seed 来源。只比较请求的操作、对象类型及业务状态，忽略实体值和占位符拼写。',
            '', '模型：'+', '.join(p['model'] for p in judge_manifest['providers'])+'。每对两次判定，temperature=0。不同任务仅在两模型共同确认时计入；分歧和不确定另列，不能当作同任务。尚未进行人工复核。',
            '', '| 数据集 | 表示 | 匹配对 | 双判不同任务 | 配对比例 | 有跨任务匹配的查询／全部 Test | 查询覆盖率 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for a in main_audits:
            name=names[a['dataset']]
            for rep,label in [('raw','原文查询'),('delex','脱敏查询')]:
                x=a['representations'][rep]
                report.append(f"| {name} | {label} | {x['matched_pairs']} | {x['both_different_pairs']} | {x['different_pair_fraction']:.2%} | {x['queries_with_any_both_different']}/{a['total_queries']} | {x['cross_task_query_coverage']:.2%} |")
        report += ['', '### 新增覆盖查询中的任务差异','',
            '| 数据集 | 新增覆盖查询 | 至少一项双判不同任务匹配 | 全部匹配均双判不同任务 | 至少一项双判同任务匹配 | 含未决匹配 |',
            '|---|---:|---:|---:|---:|---:|']
        for a in main_audits:
            x=a['newly_covered_query_audit'];name=names[a['dataset']]
            report.append(f"| {name} | {x['queries']} | {x['with_any_both_different_match']} | {x['all_matches_both_different']} | {x['with_any_both_same_match']} | {x['with_unresolved_match']} |")
        report+=['','这些列不是互斥分类：同一查询可能同时匹配不同任务键、同任务键和未决键，不能将各列相加。查询层面逐条去重，配对比例与查询覆盖率也不能相乘估计。',
            '', '“至少一个跨任务匹配”只说明存在此类阈值匹配机会，不说明缓存 Top-1 会选中该键。全部匹配均被判不同任务也不等于已验证回答错误。',
            '', '每个数据集另保存从新增覆盖配对中简单随机抽取的 100 对人工盲审空表。人审表没有展示模型标签或相似度，尚未填写。']
    else:
        report += ['', '## 任务审计状态', '', '覆盖计算已完成，双模型任务审计结果尚未收齐。不能将新增覆盖视作不同任务或错误匹配。']
    report += ['', '## 可支持的结论与限制','',
        '本实验评价针对脱敏空间构造并冻结的文本键，量化查询侧脱敏带来的阈值匹配变化。该对照保持键不变，但原文侧键仍带占位符，因此属于查询侧表示敏感性实验，不是原文空间重新搜索的算法对比，也不是两侧同时改变表示的完整管线因果比较。',
        '', ('双模型审计：'+'；'.join(f"{names[a['dataset']]} 新增覆盖 {a['newly_covered_query_audit']['queries']} 条，其中 {a['newly_covered_query_audit']['with_any_both_different_match']} 条至少出现一项两模型共同判定为不同任务的匹配" for a in main_audits)+'。判断来自模型，未经过人工复核。没有测量真实缓存竞争、缓存准入或返回答案。' if main_audits else '任务审计尚未完成，不能将新增覆盖全部视为跨任务匹配。'),
        '', '语言质量口径：按作者要求，以句子整体通顺性为标准，不单独把占位符名称与前后称谓的角色对应列为质量问题。固定文本保持原样；任务审计已明确忽略实体值及占位符拼写，只比较请求的操作、对象类型及业务状态。',
        '', ('服务器配置发现阶段的旧型号探测判断单独保存在 provenance/excluded_initial_endpoint_probe，没有参与正式结果。' if (root/'provenance/excluded_initial_probe.json').exists() else '正式任务审计使用 Qwen3.8 Flash 和 DeepSeek Flash，保留所有有效配对的两份模型判断。'),
        '', '## 文件索引','',
        '- coverage_comparison.csv：所评价数据集的 0.85 主结果。',
        '- 各数据集子目录的 coverage_summary.json：完整协议、哈希、环境、阈值扫描与复现核查。',
        '- paired_queries.csv、newly_covered_queries.csv：来源配对查询和新增查询清单。',
        '- similarities.npz、embeddings.npz：复算所需的矩阵和向量。',
        '- frozen_keys.json、source_final_report.json：未经修改的冻结句子与原报告。',
        '- audit_pairs.json、task_judgments.jsonl、task_pair_audit.csv、query_task_audit.csv：任务审计输入、原始判断与逐查询汇总。',
        '- judge_cache/：模型名称、逐批判断和 token 使用记录；不含 API 密钥。',
        '- code/：计算、审计及汇总脚本。',
        '- figures/：固定文本在七个阈值下的覆盖曲线。',
        '', '## 正文表述建议','',
        'We freeze the latest Train-selected deidentified text portfolios and vary only the representation of source-aligned Test queries. At a cosine threshold of 0.85, '+ '; '.join(f"coverage on {names[s['dataset']]} changes from {s['raw_coverage']:.2%} to {s['delex_coverage']:.2%}, with {s['delex_only']} newly covered queries" for s in summaries)+'. These measurements characterize the representation sensitivity of fixed deidentified keys; they do not compare independently optimized original-text portfolios or establish invalid answer reuse.',
        ('In a blinded dual-model task-relation audit, '+'; '.join(f"{a['newly_covered_query_audit']['with_any_both_different_match']} of the {a['newly_covered_query_audit']['queries']} newly covered {names[a['dataset']]} queries ({a['newly_covered_query_audit']['with_any_both_different_match']/a['newly_covered_query_audit']['queries']:.2%}) have at least one threshold match jointly labeled as expressing a different concrete task" for a in main_audits)+'. These are model-judged task-discordant matching opportunities, not verified cache-selection or answer-error rates.' if main_audits else ''),
        '']
    (root/'实验报告.md').write_text('\n'.join(report),encoding='utf-8')
    if main_audits:
        save(root/'task_audit_summary_all.json',main_audits)
        write_csv(root/'representative_new_match_examples.csv',example_rows)
    # Reproducible artifact copies, without any authentication material.
    code=root/'code';code.mkdir(exist_ok=True)
    here=Path(__file__).parent
    for name in ['fixed_portfolio_pair.py','remote_fixed_task_audit.py','run_remote_fixed_audit.py','summarize_fixed_pair.py']:
        shutil.copy2(here/name,code/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(args.datasets),figsize=(5*len(args.datasets),3.8),sharey=True,squeeze=False)
    axes=axes[0]
    for ax,dataset in zip(axes,args.datasets):
        rows=json.loads((root/dataset/'coverage_summary.json').read_text(encoding='utf-8'))['thresholds']
        name=names[dataset]+f" (K={rows[0]['k']})"
        ax.plot([x['threshold'] for x in rows],[x['raw_coverage']*100 for x in rows],'o-',label='Original queries',color='#3d6fa7')
        ax.plot([x['threshold'] for x in rows],[x['delex_coverage']*100 for x in rows],'s-',label='Deidentified queries',color='#c45a3c')
        ax.axvline(.85,color='#999999',ls=':',lw=1)
        ax.set(title=name,xlabel='Cosine threshold',ylim=(0,103));ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Test query coverage (%)');axes[-1].legend(frameon=False,fontsize=9)
    fig.suptitle('Frozen deidentified keys; only query representation changes',fontsize=11)
    fig.tight_layout()
    figures=root/'figures';figures.mkdir(exist_ok=True)
    fig.savefig(figures/'fixed_key_coverage.png',dpi=220)
    fig.savefig(figures/'fixed_key_coverage.svg')
    plt.close(fig)
    manifest={}
    for p in root.rglob('*'):
        if p.is_file() and p.name!='SHA256_manifest.json':
            manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(root/'SHA256_manifest.json',manifest)
    print('REPORT',root/'实验报告.md','audited_datasets',len(main_audits))


if __name__=='__main__':main()
