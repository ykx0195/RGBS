from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path.home()/'Desktop/results/CorpusPoisoning_HotFlip对比_20261031';data=json.loads((root/'summary.json').read_text(encoding='utf-8'))['metrics']
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,(dataset,title) in zip(axes,[('banking77','Banking77'),('abcd','ABCD'),('dogo','MultiDoGO')]):
    rr=[next(x for x in data if x['dataset']==dataset and x['method']==method) for method in ['CorpusPoisoning_adapted','RKGB']]
    for i,x in enumerate(rr):
        ax.bar(i-.17,100*x['test_coverage'],width=.32,color='#a3b5cb',label='Raw coverage' if i==0 else None)
        ax.bar(i+.17,100*x['quality_filtered_test_coverage'],width=.32,color='#3d6fa7',label='After common local quality screen' if i==0 else None)
    ax.set(title=title,xticks=[0,1],xticklabels=['HotFlip adapted','RKGB'],ylim=(0,105),ylabel='All Test queries covered (%)' if dataset=='banking77' else '')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    for i,x in enumerate(rr):
        ax.text(i-.17,100*x['test_coverage']+1,f"{100*x['test_coverage']:.1f}",ha='center',fontsize=9)
        ax.text(i+.17,100*x['quality_filtered_test_coverage']+1,f"{100*x['quality_filtered_test_coverage']:.1f}",ha='center',fontsize=9)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
fig.suptitle('Single-seed adapted baseline; fixed K and cosine threshold 0.85',fontsize=12)
fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(root/'corpus_hotflip_comparison.png',dpi=220);fig.savefig(root/'corpus_hotflip_comparison.svg');plt.close(fig)
print('CORPUS_HOTFLIP_FIGURE_SAVED',flush=True)
