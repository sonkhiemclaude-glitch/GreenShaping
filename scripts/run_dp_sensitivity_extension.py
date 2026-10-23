"""Fixed-arrival single-link simulation, NOT network or physical-energy data."""
import csv,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from greenshaping.shaping.dp_release import release_link,tick_times,laplace_scale
from greenshaping.analysis.bounds import expected_dp_padding_bytes,arrival_variant_padding_bound_per_tick,dp_accuracy_bound
from greenshaping.energy.cost_model import union_length
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('outputs/evaluation_extension_20261022');SEED=20261024;REPS=128

def save(name,rows):
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 plan={'seed':SEED,'repetitions_per_cell':REPS,'unit':'one link-direction; fixed arrivals, not feedback or a physical deployment','source':'existing release_link implementation; synthetic stress inputs','epsilon':[.1,.2,1,4],'interval_s':[.1,.5,1],'horizon_s':[2,10],'volume_bytes':[1024,8192],'delta_over_volume':[1,2,4],'arrival_bias_multipliers':[0,1,2,4,8],'patterns':['early','spread','late'],'adjacency':'all timing shifts of at most V bytes require Delta >= 2V; Delta=V excluded from valid-controller pool','controller_scope':'reduced analytical-risk budget rule, not validation of full empirical Algorithm 1','energy':'C = released_bytes + lambda * tick_awake_seconds; lambda in byte-equivalents/second; no calibrated joules','zero_failure_gate':'zero failures observed in 128 draws per pattern; NOT a universal completion guarantee'}
 (OUT/'sensitivity_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];profiles=[];rng=np.random.default_rng(SEED)
 grid=list(itertools.product([.1,.2,1,4],[.1,.5,1],[2,10],[1024,8192],[1,2,4]))
 for ei,(eps,tau,H,V,ratio) in enumerate(grid):
  delta=ratio*V;ticks=tick_times(tau,H)
  for mode,bias_mult in [('queue',0)]+[('arrivals',b) for b in [0,1,2,4,8]]:
   scale=laplace_scale(delta,eps,len(ticks),mode);bias=bias_mult*scale;config={'mode':mode,'eps_dir':eps,'interval_s':tau,'horizon_s':H,'volume_bytes':V,'delta_bytes':delta,'bias_multiplier':bias_mult};cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16];group=[]
   for pattern in ['early','spread','late']:
    arrivals=[(.05*H,V,0)] if pattern=='early' else ([(H-.1*tau,V,0)] if pattern=='late' else [(k*H/5,V//4,k-1) for k in range(1,5)])
    padding=[];released=[];remaining=[];completion=[]
    for rep in range(REPS):
     obs,done,left=release_link(arrivals,ticks,scale,mode,bias,rng);sent=sum(b for t,b,r in obs);padding.append(sent-(V-left));released.append(sent);remaining.append(left)
     if left==0:completion.append(max(done.values()))
    row=dict(config_id=cid,**config,pattern=pattern,repetitions=REPS,ticks=len(ticks),adjacency_timing_valid=delta>=2*V,overflow_fraction=float(np.mean(np.array(remaining)>0)),mean_overflow_bytes=float(np.mean(remaining)),mean_released_bytes=float(np.mean(released)),mean_padding_bytes=float(np.mean(padding)),padding_mc_se=float(np.std(padding,ddof=1)/np.sqrt(REPS)),mean_completion_s_success_only=float(np.mean(completion)) if completion else None,queue_padding_theory=expected_dp_padding_bytes(H,tau,delta,eps) if mode=='queue' else None,arrival_padding_upper=len(ticks)*arrival_variant_padding_bound_per_tick(bias,delta,eps) if mode=='arrivals' else None)
    rows.append(row);group.append(row)
   profiles.append(dict(config_id=cid,**config,adjacency_timing_valid=delta>=2*V,worst_overflow_fraction=max(r['overflow_fraction'] for r in group),mean_released_bytes=float(np.mean([r['mean_released_bytes'] for r in group])),formal_binary_accuracy_bound_one_direction=dp_accuracy_bound(eps),awake_seconds=union_length(ticks,.05,H+.05)))
  if ei%24==0:print('sensitivity_grid',ei,'/',len(grid),flush=True)
 save('dp_sensitivity',rows);save('simulation_profiles',profiles)
 selected=[]
 for lam,mult in itertools.product([0,1e3,1e5,1e7],[2,10,100,1000,10000]):
  V=8192;budget=mult*V;base_cost=V+lam*.1;candidates=[p for p in profiles if p['volume_bytes']==V and p['adjacency_timing_valid'] and p['worst_overflow_fraction']==0]
  feasible=[p for p in candidates if p['mean_released_bytes']+lam*p['awake_seconds']-base_cost<=budget]
  best=min(feasible,key=lambda p:(p['formal_binary_accuracy_bound_one_direction'],p['mean_released_bytes']+lam*p['awake_seconds']-base_cost)) if feasible else None
  selected.append(dict(lambda_byte_equiv_per_s=lam,extra_budget_byte_equiv=budget,budget_volume_multiple=mult,feasible_candidates=len(feasible),selected_config=best['config_id'] if best else 'BASELINE_ONLY',selected_eps_dir=best['eps_dir'] if best else None,selected_formal_bound=best['formal_binary_accuracy_bound_one_direction'] if best else 1.0,selected_mode=best['mode'] if best else 'none',selected_bias_multiplier=best['bias_multiplier'] if best else None,selected_modelled_extra_cost=best['mean_released_bytes']+lam*best['awake_seconds']-base_cost if best else 0))
 save('reduced_controller_sensitivity',selected)
 fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
 for bias in [0,1,2,4,8]:
  rr=sorted([r for r in rows if r['mode']=='arrivals' and r['eps_dir']==.2 and r['interval_s']==.5 and r['horizon_s']==10 and r['volume_bytes']==8192 and r['delta_bytes']==16384],key=lambda r:r['bias_multiplier']);r=next(r for r in rr if r['bias_multiplier']==bias and r['pattern']=='late');axes[0].scatter(bias,r['overflow_fraction'],color='#386cb0')
 x=np.linspace(0,8,100);axes[0].plot(x,.5*np.exp(-x),'--',label='last-tick theory: 0.5 exp(-bias/scale)');axes[0].set(xlabel='Arrival bias / Laplace scale',ylabel='Late-arrival overflow fraction',ylim=(-.02,.55));axes[0].legend(fontsize=6)
 for lam in [0,1e3,1e5,1e7]:
  rr=[r for r in selected if r['lambda_byte_equiv_per_s']==lam];axes[1].plot([r['budget_volume_multiple'] for r in rr],[r['selected_formal_bound'] for r in rr],marker='o',label=f'lambda={lam:g}')
 axes[1].set(xscale='log',xlabel='Extra budget / input bytes',ylabel='Selected analytical accuracy bound');axes[1].legend(fontsize=6)
 fig.savefig(OUT/'dp_sensitivity_controller.pdf');fig.savefig(OUT/'dp_sensitivity_controller.png',dpi=140);plt.close(fig)
 (OUT/'sensitivity_complete.json').write_text(json.dumps({'complete':True,'cells':len(rows),'single_link_replays':len(rows)*REPS,'controller_scenarios':len(selected),'physical_energy_measured':False,'final_test_opened':False,'release_source_sha256':hashlib.sha256(Path('src/greenshaping/shaping/dp_release.py').read_bytes()).hexdigest()},indent=2)+'\n')
 print('complete',len(rows),len(rows)*REPS,flush=True)
if __name__=='__main__':main()
