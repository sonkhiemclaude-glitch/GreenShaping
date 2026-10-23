"""Retain all sampling flags; use a fresh diagnostic replication, not retuning."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binom
from benchmark_reliability_controller import simulate
from greenshaping.controller.reliability import DirectionContract,completion_bound,expected_release_upper
P=Path('outputs/reliability_controller_20261023');a=pd.read_csv(P/'analytical_comparison.csv');v=pd.read_csv(P/'new_seed_validation.csv');cert={r['scenario']:r for r in json.loads((P/'certificates.json').read_text())}
reference=a[(a.budget_factor==1)&~a.capped];wide=reference.pivot(index='scenario',columns='policy',values='expected_byte_upper');assert (wide.workflow_uniform<wide.worst_tick_uniform).all()
certified=v[v.policy.isin(['worst_tick_uniform','workflow_uniform','workflow_allocated'])].copy();flags=certified[certified.ci_lower>certified.completion_bound+1e-12].copy();flags.to_csv(P/'retained_pointwise_sampling_flags.csv',index=False)
replication=[]
(P/'diagnostic_replication_plan.json').write_text(json.dumps({'reason':'Pointwise 95% lower limits above mathematical bound in initial small Monte Carlo samples; all original observations retained.','scope':'post-hoc implementation diagnostic, not confirmatory performance evaluation','seed':2026102901,'repetitions':200000,'flagged_rows':flags[['scenario','policy','capped','pattern']].to_dict('records'),'no_parameter_retuning':True},indent=2)+'\n')
for index,row in enumerate(flags.itertuples()):
 c=cert[row.scenario];cs=[DirectionContract(**d) for d in c['contracts']];bb=c['biases'][row.policy]
 r=simulate(cs,bb,row.pattern,200000,2026102901+index,capped=row.capped);r.pop('mean_direction_bytes');exact=1-np.prod([1-completion_bound(d,b) for d,b in zip(cs,bb)]) if cs[0].occupied_tick_cap==1 and row.pattern=='late' else None
 replication.append(dict(scenario=row.scenario,policy=row.policy,capped=row.capped,pattern=row.pattern,**r,completion_bound=row.completion_bound,exact_last_tick_probability=exact))
pd.DataFrame(replication).to_csv(P/'diagnostic_fresh_seed_replication.csv',index=False)
# The simultaneous flag check is reported as an additional post-hoc diagnostic.
pvalues=[float(binom.sf(r.failures-1,r.repetitions,r.completion_bound)) for r in certified.itertuples()]
result={'input_validation_rows':len(v),'certified_validation_rows':len(certified),'retained_pointwise_95pct_sampling_flags':len(flags),'replication_trials':len(replication)*200000,'simultaneous_diagnostic_threshold':.05/len(certified),'rows_below_threshold':sum(p<.05/len(certified) for p in pvalues),'workflow_to_worst_expected_bound_ratio_mean':float((wide.workflow_uniform/wide.worst_tick_uniform).mean()),'admitted_counts_base_budget':reference.groupby('policy').admitted.sum().astype(int).to_dict(),'all_solver_runs_successful':bool(a[a.policy=='workflow_allocated'].optimizer_success.all()),'raw_sampling_flags_not_deleted':True,'no_retuning_after_validation':True}
for name,c in cert.items():
 for policy in ['uniform','allocated']:
  d=c[policy];assert d['failure_upper']<=float(name.split('_rho')[-1])*(1+1e-10)
  assert all(x>0 for x in d['risks'])
 # Algorithm returns cannot exceed baseline objective; this is not global-optimum proof.
 assert c['allocated']['worst_budget_ratio']<=c['uniform']['worst_budget_ratio']+1e-8
(P/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
