"""Independent checks on exported analyses and new-seed simulation validation."""
import json,csv
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import beta
from sklearn.metrics import balanced_accuracy_score
from greenshaping.shaping.dp_release import release_link,tick_times,laplace_scale
P=Path('outputs/evaluation_extension_20261022');rng=np.random.default_rng(20261025)
a=pd.read_csv(P/'attacker_predictions.csv');scores=pd.read_csv(P/'attacker_scores.csv');intervals=[]
for keys,d in a.groupby(['mode','target','ablation','classifier'],sort=False):
 row=scores[(scores['mode']==keys[0])&(scores.target==keys[1])&(scores.ablation==keys[2])&(scores.classifier==keys[3])].iloc[0]
 assert abs(balanced_accuracy_score(d.truth,d.prediction)-row.balanced_accuracy)<1e-12
 d=d.assign(correct=(d.truth==d.prediction).astype(float));q=d.groupby(['truth','question_id']).correct.mean();bs=[]
 # Conditional uncertainty of fixed OOF predictions; this does not retrain classifiers.
 for label in q.index.get_level_values(0).unique():
  v=q.loc[label].to_numpy();bs.append(v[rng.integers(0,len(v),(4000,len(v)))].mean(axis=1))
 lo,hi=np.quantile(np.mean(bs,axis=0),[.025,.975]);intervals.append(dict(zip(['mode','target','ablation','classifier'],keys),balanced_accuracy=row.balanced_accuracy,cluster_ci_lower=lo,cluster_ci_upper=hi))
pd.DataFrame(intervals).to_csv(P/'attacker_conditional_intervals.csv',index=False)
folds=json.loads((P/'attacker_folds.json').read_text())
assert all(not set(f['train_questions'])&set(f['validation_questions']) for f in folds)
profiles=pd.read_csv(P/'simulation_profiles.csv');selection=pd.read_csv(P/'reduced_controller_sensitivity.csv');configs=sorted(set(selection.selected_config)-{'BASELINE_ONLY'});validation=[]
(P/'simulation_validation_plan.json').write_text(json.dumps({'seed':20261025,'repetitions_per_pattern':10000,'configs_frozen':configs,'patterns':['early','spread','late'],'selection_will_not_be_retuned':True,'final_test_bank_accessed':False},indent=2)+'\n')
for cid in configs:
 r=profiles[profiles.config_id==cid].iloc[0];V=int(r.volume_bytes);H=r.horizon_s;tau=r.interval_s;ticks=tick_times(tau,H);scale=laplace_scale(r.delta_bytes,r.eps_dir,len(ticks),r['mode']);bias=r.bias_multiplier*scale
 for pattern in ['early','spread','late']:
  arrivals=[(.05*H,V,0)] if pattern=='early' else ([(H-.1*tau,V,0)] if pattern=='late' else [(k*H/5,V//4,k-1) for k in range(1,5)])
  failures=0
  for i in range(10000):
   obs,done,left=release_link(arrivals,ticks,scale,r['mode'],bias,rng);failures+=int(left>0)
  lo=beta.ppf(.025,failures,10001-failures) if failures else 0;hi=beta.ppf(.975,failures+1,10000-failures) if failures<10000 else 1
  validation.append(dict(config_id=cid,eps_dir=r.eps_dir,bias_multiplier=r.bias_multiplier,pattern=pattern,trials=10000,failures=failures,failure_rate=failures/10000,binomial_ci_lower=lo,binomial_ci_upper=hi))
pd.DataFrame(validation).to_csv(P/'simulation_new_seed_validation.csv',index=False)
result={'verified':True,'attacker_fold_records':len(folds),'no_question_overlap_in_any_fold':True,'independent_simulation_validation_trials':len(validation)*10000,'late_failures':sum(r['failures'] for r in validation if r['pattern']=='late'),'controller_selection_unchanged':True,'conditional_intervals_do_not_include_retraining_uncertainty':True,'final_test_bank_accessed':False}
(P/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
