from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\ykx01\Desktop\results')
SENS=Path(r'C:\Users\ykx01\Desktop\ICDE_v1\figures\search_sensitivity_data.json')
MAIN=Path(r'C:\Users\ykx01\Desktop\ICDE_v1\main.tex')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'svg.fonttype':'none','savefig.facecolor':'white'})
colors=['#286FA2','#D58B32','#398A74']
datasets=['Banking77','ABCD','MultiDoGO']
ids=['banking77','abcd','dogo']
main_counts=[(1109,1485),(308,476),(909,1094)]
main=[100*n/d for n,d in main_counts]
manifest={'main_reference':{'path':str(MAIN),'sha256':hashlib.sha256(MAIN.read_bytes()).hexdigest(),'counts':main_counts},'sources':[], 'notes':['No new experiment was run.','Controls are independent single runs, not a paired estimate of component effects.','MultiDoGO control selection implementation differs from the designated main run.','Banking77 no-Riemann uses the completed independent rerun (73.40%), not the historical 73.94% run.']}
values=[main]
for folder in ['去黎曼_seed42_独立复跑_20261101/runs','VLDB补充方法对比_20261102/runs/simple_existing_config']:
 row=[]
 for ds in ids:
  p=ROOT/folder/ds/'final_report.json'
  d=json.loads(p.read_text(encoding='utf-8'));m=d['metrics']['test'];c=d['config']
  assert c['threshold']==.85 and c['steps_per_slot']==32 and c['beam_width']==50 and c['seed']==42
  assert m['total']==main_counts[ids.index(ds)][1]
  row.append(100*m['covered']/m['total'])
  manifest['sources'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'dataset':ds,'covered':m['covered'],'total':m['total']})
 values.append(row)
values=np.array(values)
labels=['RKGB (main reference)','w/o Riemann refinement','w/o gradient hints']
fig,axes=plt.subplots(1,3,figsize=(10.5,4.4),sharey=True)
for j,ax in enumerate(axes):
 for k in range(3):
  ax.bar(k,values[k,j],width=.63,color=colors[k],edgecolor='white',linewidth=.8,zorder=3)
  ax.text(k,values[k,j]+1.5,f'{values[k,j]:.2f}',ha='center',va='bottom',fontsize=11)
 ax.set_title(datasets[j]+(' *' if j==2 else ''),pad=10,fontweight='semibold')
 ax.set_ylim(0,100);ax.set_xlim(-.65,2.65);ax.set_yticks(np.arange(0,101,20))
 ax.set_xticks([0,1,2],['RKGB','No Riemann','No hints']);ax.tick_params(axis='x',labelsize=9)
 ax.grid(axis='y',color='#E2E6EA',linewidth=.65,zorder=0)
axes[0].set_ylabel('Test coverage (%)')
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,columnspacing=1.5)
fig.subplots_adjust(top=.81,bottom=.24,left=.075,right=.985,wspace=.14)
fig.text(.075,.105,'BGE; threshold = 0.85; 32 rounds; beam width = 50; K = 15 / 5 / 10.',fontsize=9,color='#3D4650')
fig.text(.075,.055,'Independent control runs; * MultiDoGO uses a separate selection implementation. No error bars available.',fontsize=9,color='#3D4650')
for ext in ['png','svg']:fig.savefig(OUT/f'01_geometry_gradient_ablation.{ext}',dpi=240)
plt.close(fig)

sens=json.loads(SENS.read_text(encoding='utf-8'))
manifest['sources'].append({'path':str(SENS),'sha256':hashlib.sha256(SENS.read_bytes()).hexdigest()})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),sharey=True)
markers=['o','s','^']
for ax,key,title,xlab,main_x in zip(axes,['rounds','beam_width'],['(a) Search rounds; beam width = 50','(b) Beam width; search rounds = 32'],['Search rounds','Beam width'],[32,50]):
 arr=np.array(sens[key],dtype=float)
 for j,(name,color,marker) in enumerate(zip(datasets,colors,markers)):
  assert abs(arr[-1,j+1]-main[j])<.006
  ax.plot(arr[:,0],arr[:,j+1],color=color,marker=marker,markersize=4.8,lw=1.7,label=name,zorder=3)
  ax.scatter([main_x],[arr[-1,j+1]],s=85,facecolors='white',edgecolors=color,linewidths=1.6,zorder=4)
  ax.text(main_x+.65,arr[-1,j+1],f'{arr[-1,j+1]:.2f}',color=color,va='center',fontsize=9)
 ax.axvline(main_x,color='#88939B',ls='--',lw=.85,zorder=1)
 ax.set_title(title,pad=10)
 ax.set_xlabel(xlab);ax.set_ylim(44,89);ax.set_yticks([45,55,65,75,85])
 ax.set_xlim(arr[0,0]-1,main_x+(6 if key=='rounds' else 9));ax.set_xticks(arr[:,0])
 ax.tick_params(axis='x',labelsize=9)
 ax.grid(axis='y',color='#E2E6EA',linewidth=.65)
axes[0].set_ylabel('Test coverage (%)')
fig.legend(handles=[Line2D([0],[0],color=c,marker=m,lw=1.7,label=n) for c,m,n in zip(colors,markers,datasets)],loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,columnspacing=2.4)
fig.subplots_adjust(top=.81,bottom=.24,left=.075,right=.985,wspace=.13)
fig.text(.075,.105,'Separate Train-only construction runs; BGE; threshold = 0.85; K = 15 / 5 / 10.',fontsize=9,color='#3D4650')
fig.text(.075,.055,'Dashed lines and open circles mark the designated main settings; lines connect observed results.',fontsize=9,color='#3D4650')
for ext in ['png','svg']:fig.savefig(OUT/f'02_rounds_beam_sensitivity.{ext}',dpi=240)
plt.close(fig)
manifest['ablation_values_pct']={l:{ds:round(float(values[k,j]),6) for j,ds in enumerate(datasets)} for k,l in enumerate(labels)}
manifest['sensitivity']=sens
(OUT/'plot_data_and_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# RQ2 两张预览图\n\n1. 01_geometry_gradient_ablation.png / .svg：主 RKGB 与去黎曼优化、去梯度提示的独立对照。Banking77 去黎曼采用完整独立复跑 73.40%，没有选用历史较高运行。MultiDoGO 的对照选择实现与主运行不同，以 * 标注；不把这些单次运行差值当作严格配对或显著性证据。去梯度提示保留双模型改写及束搜索，不是移除整个文本搜索模块。\n2. 02_rounds_beam_sensitivity.png / .svg：轮数、束宽双面板；各点来自既有独立构造，不是对同一冻结集合改评分；主设置为 32 轮、宽度 50。保留实际非单调波动。\n\n图仅供预览，未修改论文。PNG 用于直接查看，SVG 为可编辑矢量版。精确数值、来源路径与哈希见 plot_data_and_sources.json；复现脚本为 generate_figures.py。\n''',encoding='utf-8')
print('Saved two PNGs, two SVGs and source audit to:',OUT)
