from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('outputs/reliability_controller_20261023');a=pd.read_csv(P/'analytical_comparison.csv');w=pd.read_csv(P/'workflow_envelope_comparison.csv')
policies=['worst_tick_uniform','workflow_uniform','workflow_allocated','constant_schedule'];labels=['Tick bound','Enqueue bound','Allocated risk','Constant cover'];fig,axes=plt.subplots(1,2,figsize=(9.5,3.4),layout='constrained')
base=a[(a.budget_factor==1)&~a.capped];wide=base.pivot(index='scenario',columns='policy',values='expected_byte_upper')
ratios=[(wide[p]/wide.worst_tick_uniform).to_numpy() for p in policies];axes[0].boxplot(ratios,tick_labels=labels,showfliers=False);axes[0].set(yscale='log',ylabel='Expected byte upper bound / tick-bound policy');axes[0].tick_params(axis='x',rotation=25)
for cap,offset,color,label in [(False,-.18,'#386cb0','Uncapped'),(True,.18,'#f39c34','Public cap')]:
 values=[]
 for p in policies:
  if p=='constant_schedule':values.append(96)
  else:values.append(int(a[(a.budget_factor==1)&(a.capped==cap)&(a.policy==p)].admitted.sum()))
 axes[1].bar(np.arange(4)+offset,values,width=.35,color=color,label=label)
axes[1].set_xticks(range(4),labels,rotation=25);axes[1].set(ylabel='Admitted synthetic scenarios (out of 96)',ylim=(0,102));axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(P/'reliability_admission.pdf');fig.savefig(P/'reliability_admission.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9.5,3.4),layout='constrained')
for tau,color in [(.1,'#386cb0'),(.5,'#e57a29')]:
 sub=w[w.interval_s==tau].groupby('compute_bound_s').byte_ratio.agg(['mean','min','max']);axes[0].errorbar(sub.index,sub['mean'],yerr=[sub['mean']-sub['min'],sub['max']-sub['mean']],marker='o',capsize=4,label=f'tick {tau:g} s',color=color)
axes[0].set(xlabel='Public compute bound per round (s)',ylabel='Envelope / periodic application bytes',ylim=(0,.5));axes[0].legend(fontsize=8)
for n,color in [(3,'#386cb0'),(20,'#34895d')]:
 sub=w[w.N==n];axes[1].scatter(1-sub.byte_ratio,sub.envelope_tool_p95_s/sub.periodic_tool_p95_s,label=f'{n} logical sites',color=color,alpha=.7)
axes[1].set(xlabel='Application-byte reduction versus periodic cover',ylabel='Projected tool p95 latency ratio');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(P/'workflow_envelope_tradeoff.pdf');fig.savefig(P/'workflow_envelope_tradeoff.png',dpi=150);plt.close(fig)
