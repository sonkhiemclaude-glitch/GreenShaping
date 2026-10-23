"""Predeclared synthetic deployment benchmark; no cloud, no held-out bank."""
import csv,hashlib,itertools,json,math,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from scipy.stats import beta
from greenshaping.controller.reliability import DirectionContract,allocate,bias_for_risk,completion_bound,expected_release_upper
OUT=Path('outputs/reliability_controller_20261023');PILOT_REPS=128;VALID_REPS=5000

def save(name,rows):
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def simulate(contracts,biases,pattern,repetitions,seed,*,capped=False,constant=False):
 D=len(contracts);K=contracts[0].ticks;V=np.array([c.volume_cap for c in contracts]);M=contracts[0].occupied_tick_cap;s=np.array([c.scale for c in contracts]);mu=np.array(biases);rng=np.random.default_rng(seed);totals=np.zeros(D);sq=np.zeros(D);failed=0;max_total=0
 slots=[0] if pattern=='early' else ([K-1] if pattern=='late' else list(np.linspace(0,K-1,M,dtype=int)))
 assert len(set(slots))==len(slots)
 for start in range(0,repetitions,512):
  n=min(512,repetitions-start);Q=np.zeros((n,D),dtype=np.int64);total=np.zeros_like(Q)
  for k in range(K):
   if k in slots:
    position=slots.index(k);a=V//len(slots)+(V%len(slots) if position==len(slots)-1 else 0)
   else:a=np.zeros(D,dtype=np.int64)
   Q+=a
   if constant:b=np.broadcast_to(V,(n,D))
   else:
    b=np.maximum(0,np.rint(a+mu+rng.laplace(0.,s,size=(n,D)))).astype(np.int64)
    if capped:b=np.minimum(b,V)
   Q-=np.minimum(Q,b);total+=b
  failed+=int(np.any(Q>0,axis=1).sum());totals+=total.sum(axis=0);sq+=(total.astype(float)**2).sum(axis=0);max_total=max(max_total,int(total.sum(axis=1).max()))
 return {'failures':failed,'repetitions':repetitions,'failure_rate':failed/repetitions,'mean_direction_bytes':totals/repetitions,'max_observed_total_bytes':max_total,'ci_lower':float(beta.ppf(.025,failed,repetitions-failed+1)) if failed else 0.,'ci_upper':float(beta.ppf(.975,failed+1,repetitions-failed)) if failed<repetitions else 1.}

def main():
 OUT.mkdir(exist_ok=True)
 plan={'frozen_before_scores':True,'date':'2026-10-23','scope':'synthetic fixed-arrival deployment, all sites active; not a physical network or measured energy','N':[3,10,20],'K':[10,20],'M':[1,3],'heterogeneity':['uniform','heterogeneous'],'epsilon_session':[1.,4.],'completion_rho':[.01,.001],'time_interval_s':.5,'input_contract':'each direction total <= V, at most M occupied ticks; all inputs arrive before final release; Delta=2V','privacy_accounting':'epsilon per direction = epsilon_session/(2N); fixed across policies within a scenario','public_policy':'all contracts, caps, allocations and budgets fixed before private arrivals; violation/abort privacy outside certificate','policies':['fixed_bias_8','pilot_zero_failures','worst_tick_uniform','workflow_uniform','workflow_allocated','constant_schedule'],'cap_ablation':'Every stochastic policy is evaluated both uncapped and capped at public V; same bias and risk allocation reused across cap ablation. Constant V per tick is a mandatory stronger-privacy baseline.','pilot_repetitions_per_pattern':128,'validation_repetitions_per_pattern':5000,'validation_subset':'N in {3,20}, K=20, heterogeneous, epsilon_session=4; all M,rho choices; early,spread,late','budget_sweep_factors':[.8,1.,1.2],'heterogeneous_budget_shares':'reference uniform-workflow upper cost times cycle [.90,1.15,1.30], orchestrator 1.10; synthetic stress design, not empirical energy','cost':'public weighted bytes + fixed tick-awake proxy; not calibrated joules','seeds':{'pilot':2026102701,'validation':2026102702},'resource_guard':'batch <=512 repetitions; no per-packet output; max wall 900s'}
 (OUT/'benchmark_plan.json').write_text(json.dumps(plan,indent=2)+'\n');deadline=time.monotonic()+900;analytic=[];empirical=[];certificates=[]
 scenarios=list(itertools.product([3,10,20],[10,20],[1,3],['uniform','heterogeneous'],[1.,4.],[.01,.001]))
 for number,(N,K,M,hetero,eps,rho) in enumerate(scenarios):
  if time.monotonic()>deadline:raise TimeoutError('finite benchmark wall-time exceeded')
  D=2*N;cid=f'N{N}_K{K}_M{M}_{hetero}_eps{eps:g}_rho{rho:g}'
  v=[8192]*N if hetero=='uniform' else [int([1024,8192,32768][i%3]) for i in range(N)]
  cs=[DirectionContract(f'site{i//2}_{i%2}',v[i//2],M,K,eps/D,2*v[i//2]) for i in range(D)];worst=[replace(c,occupied_tick_cap=K) for c in cs]
  A=np.zeros((N+1,D))
  for i in range(N):A[i,2*i:2*i+2]=1. if hetero=='uniform' else [1.,3.,10.][i%3]
  A[-1,:]=1.;awake=np.r_[np.repeat(K*.05,N),K*.05]*1000
  reference=allocate(cs,rho,A,awake,np.repeat(1e30,N+1),optimize=False)
  shares=np.ones(N+1) if hetero=='uniform' else np.r_[[[.90,1.15,1.30][i%3] for i in range(N)],1.10]
  budgets=np.array(reference.node_cost_upper)*shares
  uniform=allocate(cs,rho,A,awake,budgets,optimize=False);optimized=allocate(cs,rho,A,awake,budgets);uniformK=allocate(worst,rho,A,awake,budgets,optimize=False)
  assert optimized.worst_budget_ratio<=uniform.worst_budget_ratio+1e-8
  configs={'fixed_bias_8':([8*c.scale for c in cs],True),'worst_tick_uniform':(uniformK.biases,True),'workflow_uniform':(uniform.biases,True),'workflow_allocated':(optimized.biases,True),'constant_schedule':([0.]*D,True)}
  selected=None
  for candidate in [0,1,2,4,8]:
   bb=[candidate*c.scale for c in cs];failures=sum(simulate(cs,bb,p,PILOT_REPS,2026102701+1000*number+candidate*10+i)['failures'] for i,p in enumerate(['early','spread','late']))
   if failures==0:selected=bb;break
  configs['pilot_zero_failures']=(selected or [8*c.scale for c in cs],selected is not None)
  certificates.append({'scenario':cid,'contracts':[asdict(c) for c in cs],'matrix':A.tolist(),'awake_proxy':awake.tolist(),'budgets':budgets.tolist(),'uniform':asdict(uniform),'allocated':asdict(optimized),'pilot_selected':selected is not None,'biases':{k:b for k,(b,ok) in configs.items()}})
  validate=N in (3,20) and K==20 and hetero=='heterogeneous' and eps==4.
  for name,(bb,available) in configs.items():
   for capped in ([False] if name=='constant_schedule' else [False,True]):
    constant=name=='constant_schedule';bound=0. if constant else sum(completion_bound(c,b) for c,b in zip(cs,bb));byteupper=np.array([K*c.volume_cap if constant else expected_release_upper(replace(c,release_cap=c.volume_cap) if capped else c,b) for c,b in zip(cs,bb)]);cost=A@byteupper+awake
    for factor in [.8,1.,1.2]:
     ratio=float(np.max(cost/(budgets*factor)));certified=available and bound<=rho*(1+1e-10)
     analytic.append(dict(scenario=cid,N=N,K=K,M=M,heterogeneity=hetero,epsilon_session=eps,rho_completion=rho,policy=name,capped=capped,budget_factor=factor,available=available,completion_bound=min(1.,bound),completion_certified=certified,expected_byte_upper=float(byteupper.sum()),worst_model_budget_ratio=ratio,admitted=certified and ratio<=1+1e-9,optimizer_success=optimized.optimizer_success if name=='workflow_allocated' else True))
    if validate and available:
     for i,pattern in enumerate(['early','spread','late']):
      result=simulate(cs,bb,pattern,VALID_REPS,2026102702+1000*number+100*i+(1 if capped else 0),capped=capped,constant=constant);mean=result.pop('mean_direction_bytes');empirical.append(dict(scenario=cid,N=N,K=K,M=M,epsilon_session=eps,rho_completion=rho,policy=name,capped=capped,pattern=pattern,**result,mean_total_bytes=float(mean.sum()),worst_mean_model_budget_ratio=float(np.max((A@mean+awake)/budgets)),completion_bound=min(1.,bound)))
  if number%12==0:print('scenario',number,'/',len(scenarios),flush=True)
 save('analytical_comparison',analytic);save('new_seed_validation',empirical);(OUT/'certificates.json').write_text(json.dumps(certificates,indent=2)+'\n');(OUT/'complete.json').write_text(json.dumps({'complete':True,'scenarios':len(scenarios),'analytical_rows':len(analytic),'new_seed_validation_cases':len(empirical),'new_seed_validation_deployments':sum(r['repetitions'] for r in empirical),'pilot_trials_excluded_from_validation':True,'final_test_bank_opened':False,'new_cloud_sessions':0,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/greenshaping/controller/reliability.py'),Path('src/greenshaping/shaping/dp_release.py'),Path(__file__)]}},indent=2)+'\n');print('completed',len(scenarios),sum(r['repetitions'] for r in empirical),flush=True)
if __name__=='__main__':main()
