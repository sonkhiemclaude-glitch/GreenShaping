"""Independent exported-data checks using analytical Laplace variance."""
import json
from pathlib import Path
import pandas as pd,numpy as np
P=Path('outputs/evaluation_extension_20261022');a=pd.read_csv(P/'dp_sensitivity.csv');q=a[a['mode']=='queue'];scale=q.delta_bytes*q.ticks/q.eps_dir;se=scale*np.sqrt(.75*q.ticks/128);z=((q.mean_padding_bytes-q.queue_padding_theory)/se).abs();late=a[(a['mode']=='arrivals')&(a.pattern=='late')];p=.5*np.exp(-late.bias_multiplier)
assert ((late.overflow_fraction-p).abs()<=6*np.sqrt(p*(1-p)/128)+2/128).all();assert z.max()<6
v=pd.read_csv(P/'simulation_new_seed_validation.csv');assert (v.groupby('config_id').failures.sum()>0).all()
f=json.loads((P/'attacker_folds.json').read_text());assert all(not set(r['train_questions'])&set(r['validation_questions']) for r in f)
r={'verified':True,'queue_padding_max_abs_known_variance_z':float(z.max()),'queue_cells':len(q),'variance_formula':'3 K s^2/(4 n), ignoring <= K/2 byte rounding shift','self_normalized_diagnostic_max':float(((q.mean_padding_bytes-q.queue_padding_theory)/q.padding_mc_se).abs().max()),'note':'Small heavy-tailed samples give unstable empirical standard errors; the crosscheck uses known Laplace variance. No outcomes removed.','late_arrival_cells':len(late),'late_overflow_formula_check_passed':True,'all_three_pilot_selected_configs_have_validation_failures':True,'validation_total_failures':int(v.failures.sum()),'question_disjoint_fold_records':len(f)}
(P/'analytical_crosschecks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
