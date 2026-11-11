"""Combine completed frozen-key experiments without changing their source records."""
from pathlib import Path
import csv,hashlib,json,shutil,sys
from collections import Counter,defaultdict
import numpy as np


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    root=Path.home()/'Desktop/results/固定文本_原文脱敏配对_20261031'
    sources=[('banking77','Banking77',root),('abcd','ABCD',root/'ABCD补充'),('dogo','MultiDoGO',root)]
    out=root/'三数据集汇总';out.mkdir(exist_ok=True)
    table=[];audit_rows=[];inputs=[]
    for dataset,name,base in sources:
        p=base/dataset
        c=json.loads((p/'coverage_summary.json').read_text(encoding='utf-8'))
        a=json.loads((p/'task_audit_summary.json').read_text(encoding='utf-8'))
        f=json.loads((p/'frozen_keys.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(Path(f['source']).read_bytes()).hexdigest()==f['source_sha256']
        scores=np.load(p/'similarities.npz');sr,sd=scores['raw'],scores['delex']
        assert sr.shape==sd.shape==(c['main']['total'],len(f['texts']))
        assert int((sd>=.85).any(axis=1).sum())==c['main']['delex_covered']
        with (p/'query_task_audit.csv').open(encoding='utf-8-sig',newline='') as h:qr=list(csv.DictReader(h))
        assert len(qr)==c['main']['total']
        for rep in ['raw','delex']:
            stat=a['representations'][rep]
            assert sum(r[rep+'_any_both_different']=='True' for r in qr)==stat['queries_with_any_both_different']
            assert stat['matched_pairs']==stat['both_different_pairs']+stat['both_same_pairs']+stat['unresolved_pairs']
        m=c['main'];new=a['newly_covered_query_audit']
        row={'dataset':name,'k':m['k'],'threshold':.85,'test_total':m['total'],
             'original_covered':m['raw_covered'],'deidentified_covered':m['delex_covered'],
             'original_coverage':m['raw_coverage'],'deidentified_coverage':m['delex_coverage'],
             'coverage_delta_pp':m['delta_pp'],'newly_covered':m['delex_only'],'lost_covered':m['raw_only'],
             'new_queries_with_cross_task_match':new['with_any_both_different_match'],
             'new_queries_cross_task_fraction':new['with_any_both_different_match']/new['queries'],
             'new_queries_all_matches_cross_task':new['all_matches_both_different'],
             'original_cross_task_queries':a['representations']['raw']['queries_with_any_both_different'],
             'deidentified_cross_task_queries':a['representations']['delex']['queries_with_any_both_different'],
             'original_cross_task_query_coverage':a['representations']['raw']['cross_task_query_coverage'],
             'deidentified_cross_task_query_coverage':a['representations']['delex']['cross_task_query_coverage'],
             'cross_task_query_delta_pp':a['cross_task_query_coverage_delta_pp']}
        table.append(row);audit_rows.append(a)
        inputs.append({'dataset':dataset,'report':str(p/'coverage_summary.json'),
            'source_report_sha256':c['identity']['source_report_sha256'],
            'task_summary_sha256':hashlib.sha256((p/'task_audit_summary.json').read_bytes()).hexdigest(),
            'key_texts_sha256':c['identity']['key_texts_sha256']})
        print('VALIDATED',name,json.dumps(row,ensure_ascii=False))
    # Validate each model's complete output, with no duplicated pair/provider observations.
    for base,count in [(root,2804),(root/'ABCD补充',344)]:
        jj=[json.loads(x) for x in (base/'task_judgments.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        ids=Counter((x['pair_id'],x['provider']) for x in jj)
        assert all(v==1 for v in ids.values()) and len(jj)==2*count
        assert Counter(x['model'] for x in jj)=={'qwen3.8-flash':count,'deepseek-flash':count}
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    report=['# 三数据集：固定文本的原文／脱敏查询配对实验','',
        '日期：2026-10-31。使用最新 32 轮、束宽 50、seed 42 的 Train 选定最终组合。BGE-small-en-v1.5，CLS 池化，单位归一化，阈值 0.85。冻结文本不变，原文与脱敏 Test 按 source_id 和 variant_id 一一配对；只改变查询侧表示。',
        '', '## 覆盖与新增查询任务审计','',
        '| 数据集 | K | 原文查询覆盖 | 脱敏查询覆盖 | 净增，pp | 新增查询 | 其中有跨任务匹配 | 全部匹配均跨任务 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in table:
        report.append(f"| {x['dataset']} | {x['k']} | {x['original_covered']}/{x['test_total']}（{x['original_coverage']:.2%}） | {x['deidentified_covered']}/{x['test_total']}（{x['deidentified_coverage']:.2%}） | +{x['coverage_delta_pp']:.2f} | {x['newly_covered']} | {x['new_queries_with_cross_task_match']}（{x['new_queries_cross_task_fraction']:.2%}） | {x['new_queries_all_matches_cross_task']} |")
    report+=['','三个数据集均没有“原文已覆盖、脱敏后失去覆盖”的查询。不同键的个别匹配可以消失而查询仍由其他键覆盖，因此查询覆盖变化与配对变化分开记录。',
        '', '## 全部 Test 中的跨任务查询覆盖','',
        '| 数据集 | 原文查询 | 脱敏查询 | 差值，pp |','|---|---:|---:|---:|']
    for x in table:
        report.append(f"| {x['dataset']} | {x['original_cross_task_queries']}/{x['test_total']}（{x['original_cross_task_query_coverage']:.2%}） | {x['deidentified_cross_task_queries']}/{x['test_total']}（{x['deidentified_cross_task_query_coverage']:.2%}） | +{x['cross_task_query_delta_pp']:.2f} |")
    report+=['','跨任务查询覆盖定义为：全部 Test 查询中，至少与一个不同具体任务的冻结键达到阈值的比例。每条查询最多计一次，不是匹配对比例，也不是 Top-1 或错误答案率。',
        '', '## 任务审计协议与限制','',
        '- 对原文／脱敏任一侧匹配的全部唯一键—查询配对进行审计，共 3,148 对、6,296 条模型判断；ABCD 344 对／688 条，另两数据集合计 2,804 对／5,608 条。',
        '- 模型为 Qwen3.8 Flash 与 DeepSeek Flash，temperature=0；每对都给两模型，只展示固定键和统一的脱敏查询文本，隐藏相似度、来源类别及配对状态。',
        '- 同一配对在两种表示中共用任务标签。判断请求的实际操作、对象类型和业务状态，忽略实体值及占位符拼写；不生成或判断回答。',
        '- 只有两模型共同判为 DIFFERENT_TASK 才计跨任务；模型分歧或不确定单独保留，不当作同任务。结果尚未独立人工复核。各数据集均有随机 100 对人工盲审空表。',
        '- 固定键来自脱敏空间，因此该实验说明固定脱敏键对查询侧表示的敏感性；不能当作未脱敏侧独立搜索后的方法比较或完整管线因果效应。',
        '- 按作者语言标准，以整体通顺性为准，不把占位符角色对应单独作为质量问题。实验没有修改最终句子。',
        '', '## 论文可用表述','',
        'With the latest Train-selected text portfolios frozen, query-side deidentification increases Test coverage from 17.98% to 74.68% on Banking77, from 20.17% to 64.71% on ABCD, and from 25.87% to 81.17% on MultiDoGO at a cosine threshold of 0.85. Of the 842, 212, and 605 newly covered queries, respectively, 439 (52.14%), 125 (58.96%), and 252 (41.65%) have at least one match jointly labeled by two model judges as expressing a different concrete task. These results indicate that the coverage expansion includes task-discordant matching opportunities. The experiment varies query representation while retaining deidentified keys and does not establish actual cache selection or invalid answer reuse.',
        '', '## 原始报告','',
        '[Banking77 与 MultiDoGO 报告]('+ (root/'实验报告.md').as_posix()+')',
        '', '[ABCD 补充报告]('+(root/'ABCD补充/实验报告.md').as_posix()+')','']
    (out/'三数据集实验汇总.md').write_text('\n'.join(report),encoding='utf-8')
    (out/'input_manifest.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,(dataset,name,base) in zip(axes,sources):
        s=json.loads((base/dataset/'coverage_summary.json').read_text(encoding='utf-8'))
        curve=s['thresholds']
        for key,label,color,marker in [('raw_coverage','Original queries','#3d6fa7','o'),('delex_coverage','Deidentified queries','#c45a3c','s')]:
            ax.plot([x['threshold'] for x in curve],[x[key]*100 for x in curve],marker+'-',color=color,label=label)
        ax.axvline(.85,color='#999',ls=':',lw=1)
        ax.set(title=f"{name} (K={s['main']['k']})",xlabel='Cosine threshold',ylim=(0,103));ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Test query coverage (%)');axes[-1].legend(frameon=False,fontsize=8)
    fig.suptitle('Frozen deidentified keys; only query representation changes',fontsize=11)
    fig.tight_layout();fig.savefig(out/'three_dataset_coverage.png',dpi=220);fig.savefig(out/'three_dataset_coverage.svg');plt.close(fig)
    shutil.copy2(Path(__file__),out/'combine_fixed_pair.py')
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='SHA256_manifest.json'}
    (out/'SHA256_manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Keep parent manifest consistent with the newly added supplement and summary.
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p!=root/'SHA256_manifest.json'}
    (root/'SHA256_manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('COMBINED_REPORT',out/'三数据集实验汇总.md')


if __name__=='__main__':main()
