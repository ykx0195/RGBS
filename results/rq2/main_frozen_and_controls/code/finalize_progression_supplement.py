"""Merge missing judgments, independently check saved outputs, report findings."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,shutil,sys,importlib.util
import numpy as np

ROOT=Path(r'C:\Users\ykx01\Desktop\results\递进主线补充实验_20261101')
BASE=ROOT.parent
DATA=Path(r'C:\Users\ykx01\Desktop\ykx\data')
DS={'banking77':('banking77',15,1109),'abcd':('abcd',5,308),'dogo':('multidogo',10,909)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvread(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(name,rr):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bootstrap_delta(delta,source_ids):
    idx={s:i for i,s in enumerate(dict.fromkeys(source_ids))};gi=np.array([idx[s] for s in source_ids]);ng=len(idx)
    a=np.bincount(gi,weights=delta,minlength=ng);n=np.bincount(gi,minlength=ng);rng=np.random.default_rng(20261101)
    vals=[]
    for _ in range(2000):
        rr=rng.integers(0,ng,ng);vals.append(a[rr].sum()/n[rr].sum()*100)
    return np.quantile(vals,[.025,.975]).tolist()

def main():
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    pairrows=csvread('task_audit_pair_alignment.csv')
    pending=csvread('pre_audit/task_audit_pending_pairs.csv' if (ROOT/'pre_audit/task_audit_pending_pairs.csv').exists() else 'task_audit_pending_pairs.csv')
    judgments=[json.loads(s) for s in (ROOT/'task_judgments.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    manifest=read(ROOT/'task_judge_manifest.json')
    byid=defaultdict(dict)
    assert {r['model'] for r in judgments}=={'qwen3.8-flash','deepseek-flash'}
    for r in judgments:
        assert r['provider'] not in byid[r['pair_id']]
        assert r['label'] in ['SAME_TASK','DIFFERENT_TASK','UNCERTAIN']
        byid[r['pair_id']][r['provider']]=r
    assert set(byid)=={r['pair_id'] for r in pending}
    assert len(judgments)==2*len(pending)==662
    for r in pairrows:
        if r['pair_id'] in byid:
            mm=byid[r['pair_id']];assert set(mm)=={'qwen_api','deepseek_api'}
            a,b=mm['qwen_api']['label'],mm['deepseek_api']['label']
            r['consensus']='BOTH_DIFFERENT_TASK' if a==b=='DIFFERENT_TASK' else 'BOTH_SAME_TASK' if a==b=='SAME_TASK' else 'UNRESOLVED'
            r['source_judgments']=json.dumps(list(mm.values()),ensure_ascii=False)
            r['reused']='False'
    csvwrite('task_audit_completed_pairs.csv',pairrows)
    completed=[];queryrows=[];taskcurves=[];taskthresholds=[]
    for ds,(folder,k,expected) in DS.items():
        source=read(ROOT/ds/'main_frozen_texts.json');texts=source['texts']
        q=[json.loads(s) for s in (DATA/folder/'desensitized/test.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        p=np.load(ROOT/ds/'main_fixed_key_paired_similarities.npz');sr,sd=p['raw'],p['delex']
        annot=np.zeros(sd.shape,bool);diff=np.zeros_like(annot);same=np.zeros_like(annot)
        for r in pairrows:
            if r['dataset']!=ds:continue
            i,j=int(r['test_index']),int(r['key_index'])
            assert r['key_text']==texts[j] and r['query_text']==q[i]['text'] and r['source_id']==q[i]['source_id']
            assert not annot[i,j]
            assert r['consensus'] in ['BOTH_DIFFERENT_TASK','BOTH_SAME_TASK','UNRESOLVED']
            annot[i,j]=True;diff[i,j]=r['consensus']=='BOTH_DIFFERENT_TASK';same[i,j]=r['consensus']=='BOTH_SAME_TASK'
        assert np.all(~((sr>=.85)|(sd>=.85))|annot)
        np.savez_compressed(ROOT/ds/'completed_audit_membership.npz',judged=annot,different=diff,same=same)
        for rep,sim in [('raw',sr),('delex',sd)]:
            mask=sim>=.85;covered=mask.any(axis=1);anydiff=(mask&diff).any(axis=1);alldiff=covered&(~mask|diff).all(axis=1)
            unresolved=(mask&~diff&~same).any(axis=1)
            completed.append({'dataset':ds,'representation':rep,'threshold':.85,'key_representation':'fixed_deidentified',
                'covered_queries':int(covered.sum()),'test_total':len(q),'matched_pairs':int(mask.sum()),
                'judged_pairs':int((mask&annot).sum()),'different_task_pairs':int((mask&diff).sum()),
                'same_task_pairs':int((mask&same).sum()),'unresolved_pairs':int((mask&~diff&~same).sum()),
                'any_different_task_queries':int(anydiff.sum()),'all_matches_different_task_queries':int(alldiff.sum()),
                'queries_with_unresolved_matches':int(unresolved.sum()),'audit_complete':True})
            for i,r in enumerate(q):queryrows.append({'dataset':ds,'representation':rep,'test_index':i,'source_id':r['source_id'],
                'covered':int(covered[i]),'any_different_task':int(anydiff[i]),'all_matches_different_task':int(alldiff[i]),'has_unresolved':int(unresolved[i])})
            for tau in [.75,.775,.8,.825,.85,.875,.9,.925,.95,.975]:
                mask=sim>=tau;complete=bool(np.all(~mask|annot));cross=int((mask&diff).any(axis=1).sum())
                taskthresholds.append({'dataset':ds,'representation':rep,'threshold':tau,'covered_queries':int(mask.any(axis=1).sum()),
                    'test_total':len(q),'matched_pairs':int(mask.sum()),'missing_judgments':int((mask&~annot).sum()),'audit_complete':complete,
                    'any_different_task_queries':cross if complete else None,'confirmed_different_task_lower_bound':cross})
        order=read(ROOT/ds/'delex_RKGB_frozen_order.json')['order']
        for m in range(k+1):
            ix=order[:m];mask=sd[:,ix]>=.85;dd=diff[:,ix];ss=same[:,ix]
            cov=mask.any(axis=1);cr=(mask&dd).any(axis=1);al=cov&(~mask|dd).all(axis=1)
            taskcurves.append({'dataset':ds,'texts':m,'pool_size':k,'test_total':len(q),'covered_queries':int(cov.sum()),
                'any_different_task_queries':int(cr.sum()),'all_matches_different_task_queries':int(al.sum()),
                'unresolved_pairs':int((mask&~dd&~ss).sum()),'threshold':.85,'order_selected_on':'train'})
    csvwrite('task_audit_summary_completed.csv',completed);csvwrite('query_level_task_audit_completed.csv',queryrows)
    csvwrite('budget_task_coverage.csv',taskcurves);csvwrite('threshold_task_coverage.csv',taskthresholds)
    archive=ROOT/'pre_audit';archive.mkdir(exist_ok=True)
    for name in ['task_audit_pair_alignment.csv','task_audit_summary.csv','task_audit_pending_pairs.csv']:
        if not (archive/name).exists():shutil.copy2(ROOT/name,archive/name)
    csvwrite('task_audit_pair_alignment.csv',pairrows)
    csvwrite('task_audit_summary.csv',completed)
    with (ROOT/'task_audit_pending_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        csv.DictWriter(f,fieldnames=list(pending[0])).writeheader()
    save(ROOT/'audit_status.json',{'complete_at_main_threshold':True,'new_pair_count':len(pending),'new_judgments':len(judgments),
        'new_batches':manifest['completed_batches'],'usage':manifest['usage'],'datasets':completed,
        'human_labels':False,'judges':[x['model'] for x in manifest['providers']],'lower_thresholds_may_have_unjudged_pairs':True})
    # Separate method gain from the deidentification effect; uncertainty is paired by source.
    gains=[]
    for ds,(folder,k,expected) in DS.items():
        q=[json.loads(s) for s in (DATA/folder/'desensitized/test.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        ids=[r['source_id'] for r in q];dd={}
        for rep in ['raw','delex']:
            x=np.load(ROOT/ds/f'{rep}_RKGB_similarities.npz')['test'];b=np.load(ROOT/ds/f'{rep}_greedy_train_swap4_similarities.npz')['test']
            c=(x>=.85).any(axis=1);bc=(b>=.85).any(axis=1);delta=c.astype(int)-bc.astype(int);dd[rep]=delta
            lo,hi=bootstrap_delta(delta,ids)
            gains.append({'dataset':ds,'comparison':rep+'_RKGB_minus_greedy_swap','delta_pp':float(delta.mean()*100),
                'source_bootstrap_low_pp':lo,'source_bootstrap_high_pp':hi,'test_total':len(q)})
        interaction=dd['delex']-dd['raw'];lo,hi=bootstrap_delta(interaction,ids)
        gains.append({'dataset':ds,'comparison':'difference_in_method_gains_delex_minus_raw','delta_pp':float(interaction.mean()*100),
                'source_bootstrap_low_pp':lo,'source_bootstrap_high_pp':hi,'test_total':len(q)})
    csvwrite('method_gain_by_representation.csv',gains)
    # Independent consistency checks against saved reports, not only internal invariants.
    checks=[]
    for ds,(folder,k,expected) in DS.items():
        x=np.load(ROOT/ds/'delex_RKGB_similarities.npz');actual=int((x['test']>=.85).any(axis=1).sum())
        assert actual==expected;checks.append({'dataset':ds,'check':'main_count_exact','actual':actual,'expected':expected,'passed':True})
        rawref=read(BASE/f'三数据集_未脱敏_0.85_32x50_20261031/runs/{ds}/final_report.json')['metrics']['test']['covered']
        rawcount=int((np.load(ROOT/ds/'raw_RKGB_similarities.npz')['test']>=.85).any(axis=1).sum());assert rawcount==rawref
        checks.append({'dataset':ds,'check':'raw_frozen_report_exact','actual':rawcount,'expected':rawref,'passed':True})
        for rep in ['raw','delex']:
            for method in ['RKGB','greedy_train','greedy_train_swap4']:
                sim=np.load(ROOT/ds/f'{rep}_{method}_similarities.npz');order=read(ROOT/ds/f'{rep}_{method}_frozen_order.json')['order']
                # Recompute all budgets using explicit Python sets and compare CSV rows.
                published=[r for r in csvread('budget_coverage.csv') if r['dataset']==ds and r['representation']==rep and r['method']==method]
                for r in published:
                    selected=order[:int(r['texts'])];hit=set()
                    for j in selected:hit.update(np.flatnonzero(sim['test'][:,j]>=.85).tolist())
                    assert len(hit)==int(r['test_covered'])
        checks.append({'dataset':ds,'check':'budget_set_union_independent','actual':'all prefixes','expected':'CSV','passed':True})
        a=next(r for r in completed if r['dataset']==ds and r['representation']=='delex')
        assert a['covered_queries']==expected and a['matched_pairs']==a['judged_pairs']
        assert a['different_task_pairs']+a['same_task_pairs']+a['unresolved_pairs']==a['matched_pairs']
        checks.append({'dataset':ds,'check':'all_main_pairs_judged_partition','actual':a['judged_pairs'],'expected':a['matched_pairs'],'passed':True})
    original_manifest=read(ROOT/'input_manifest.json')
    external={path:expected for path,expected in original_manifest.items() if not Path(path).is_relative_to(ROOT)}
    derived={path:expected for path,expected in original_manifest.items() if Path(path).is_relative_to(ROOT)}
    if derived:save(ROOT/'derived_inputs_initial_sha256.json',derived)
    for path,expected in external.items():assert filehash(path)==expected,path
    save(ROOT/'input_manifest.json',external)
    save(ROOT/'validation_report.json',{'all_passed':True,'checks':checks,'source_inputs_unchanged':True,
        'coverage_order_uses_train_only':True,'new_judgments_complete':True,'new_judgment_pairs':331})
    interpretation(completed,gains,taskcurves)
    completion=read(ROOT/'completion.json');completion.update({'task_audit_complete_at_085':True,'new_api_judgments':662,
        'new_api_batches':manifest['completed_batches'],'gpu_used':False,'validation_all_passed':True,
        'api_calls':None,'api_call_count_note':'56 successful API batches; HTTP retries not separately measured.'})
    save(ROOT/'completion.json',completion)
    final_summary=read(ROOT/'analysis_summary.json');final_summary.update({'task_audit_complete_at_085':True,'completed_task_audit':completed,'method_gains':gains})
    save(ROOT/'analysis_summary.json',final_summary)
    # Add an explicit follow-up to the initial CPU-only plan, preserving its scope.
    plan=read(ROOT/'experiment_plan.json');plan['api_calls_cpu_phase']=plan.pop('api_calls',0)
    plan['followup_audit']={'completed':True,'new_pairs':331,'new_judgments':662,'successful_batches':manifest['completed_batches'],'gpu_used':False}
    save(ROOT/'experiment_plan.json',plan)
    plot_task_curves(taskcurves)
    shutil.copy2(__file__,ROOT/'code'/Path(__file__).name)
    for name in ['run_progression_task_audit.py','remote_fixed_task_audit.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'code'/name)
    output_manifest={str(p.relative_to(ROOT)):{'sha256':filehash(p),'bytes':p.stat().st_size} for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
    save(ROOT/'artifact_sha256.json',output_manifest)
    print(json.dumps({'audit_complete':True,'new_pairs':331,'new_judgments':662,'results':completed,'checks_passed':True},ensure_ascii=False,indent=2))

def interpretation(audit,gains,curves):
    p=ROOT/'实验报告.md';text=p.read_text(encoding='utf-8')
    start=text.index('## 任务关系判断的复用与缺口') if '## 任务关系判断的复用与缺口' in text else text.index('## 已完成的任务关系判断')
    end=text.index('## 复现',start)
    replacement=['## 已完成的任务关系判断','',
        'Banking77和ABCD全部复用既有判断。MultiDoGO按83.09%对应文本精确匹配，复用614个匹配对的判断，新增331对、662条模型判断。继续使用Qwen3.8 Flash与DeepSeek Flash原协议：隐藏算法、相似度及分组状态，仅比较请求操作、对象类型与业务状态；分歧和不确定单独保留。','',
        '| 数据集 | 查询表示 | 总覆盖 | 至少一项不同任务匹配 | 全部匹配均不同任务 | 未决配对 |','|---|---|---:|---:|---:|---:|']
    for r in audit:replacement.append(f"| {r['dataset']} | {r['representation']} | {r['covered_queries']}/{r['test_total']} | {r['any_different_task_queries']}/{r['test_total']} | {r['all_matches_different_task_queries']}/{r['test_total']} | {r['unresolved_pairs']} |")
    replacement+=['','此处键始终为指定的脱敏主集合，只有查询表示变化。任务判断是模型证据，不是人工真值。0.85及更高阈值的主集合匹配审计完整；更低阈值可能出现额外未判配对，threshold_task_coverage.csv明确标记缺失及已确认下界，不填补为同任务。','',
        '## 方法收益与脱敏放大作用分开评价','',
        '| 数据集 | 原文 RKGB相对普通贪心加交换 | 脱敏 RKGB相对普通贪心加交换 | 两种增益之差 |','|---|---:|---:|---:|']
    for ds in DS:
        dd={r['comparison']:r for r in gains if r['dataset']==ds}
        replacement.append(f"| {ds} | {dd['raw_RKGB_minus_greedy_swap']['delta_pp']:.2f} pp | {dd['delex_RKGB_minus_greedy_swap']['delta_pp']:.2f} pp | {dd['difference_in_method_gains_delex_minus_raw']['delta_pp']:.2f} pp |")
    replacement+=['','method_gain_by_representation.csv同时公开来源配对bootstrap区间。普通选句也在脱敏后提高覆盖，因此放大现象不能全部归因于RKGB。RKGB相对普通选句的增益是另一项方法证据；两个问题不相互替代。增益之差属于固定产物的探索性分析，不包含完整搜索重复的不确定性。','',
        '## 本轮结果怎样完善递进主线','',
        '1. 固定0.85下，来源对齐的原文与脱敏冻结集合比较在三集均增加覆盖；普通自然问句基线也同向。这支持脱敏放大集中匹配的条件性观察。',
        '2. 预算曲线直接展示少量文本怎样累积覆盖，而不仅给出最终K的端点。曲线限于各自完整K集合的Train前缀，不是所有预算上的独立最优构造。',
        '3. 在现有合成同任务Validation正例校准出的90%与95%召回操作点，三集脱敏集合覆盖仍高于原文集合，但绝对覆盖明显降低。本轮结果支持固定操作点的放大作用，不能声称在所有操作点都覆盖多数。Test实测同任务召回及校准样本的模型筛选性质一并呈现。',
        '4. 连续目标覆盖高于对应最近自然锚点；实际文本搜索在所报告配置中提高自然文本覆盖。该结果解释几何方向到语言接口的阶段差异，不证明黎曼模块在所有文本运行中均优于KMeans。',
        '5. 匹配中的任务关系由完整主阈值审计描述，与总覆盖分别报告。','',
        '## 尚未完成的资源相关方法对照','',
        '本轮完成CPU分析及主阈值缺失任务判断，未启动新的文本构造搜索。使用相同改写接口和筛选规则、对齐生成资源的简单搜索基线仍需单独冻结协议后运行；已有不等成本组件消融不能替代这项对照。旧MultiDoGO逐槽与组件产物若对应不同冻结文本，不能通过修改总覆盖率将其当作83.09%主集合的检查点消融。','',
        '独立检查已完成：主结果精确复现、原文报告复现、逐前缀集合并集复算、任务标签分区完整性及输入哈希不变。见validation_report.json。','',]
    text=text[:start]+'\n'.join(replacement)+text[end:]
    text=text.replace('全部新增计算使用本地CPU；未启动文本生成搜索，未使用GPU。','本地数值分析使用CPU；缺失任务判断通过原双模型API补齐。未启动文本生成搜索，未占用本地或该服务器GPU。')
    p.write_text(text,encoding='utf-8')

def plot_task_curves(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,ds,title in zip(axs,DS,['Banking77','ABCD','MultiDoGO']):
        rr=[r for r in curves if r['dataset']==ds]
        for field,color,label in [('covered_queries','#27659a','Total threshold coverage'),('any_different_task_queries','#b54d37','At least one different-task match')]:
            ax.plot([r['texts'] for r in rr],[100*r[field]/r['test_total'] for r in rr],'o-',color=color,label=label)
        ax.set(title=title,xlabel='Train-ordered prefix size',ylim=(0,100));ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.2)
    axs[0].set_ylabel('Fraction of all Test queries (%)');axs[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'budget_task_coverage.png',dpi=190);fig.savefig(ROOT/'budget_task_coverage.svg');plt.close(fig)

if __name__=='__main__':main()
