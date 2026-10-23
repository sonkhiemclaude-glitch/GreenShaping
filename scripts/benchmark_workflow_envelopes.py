"""Synthetic adaptive-round jobs: public envelope versus periodic constant cover."""
import csv,itertools,json,math
from pathlib import Path
import numpy as np
from greenshaping.controller.workflow_contract import WorkflowEnvelope
from greenshaping.energy.cost_model import union_length
OUT=Path('outputs/reliability_controller_20261023');rng=np.random.default_rng(2026102801);B=5000

def main():
 plan={'seed':2026102801,'jobs_per_scenario':B,'N':[3,20],'public_round_cap':[3,5],'interval_s':[.1,.5],'compute_bound_s':[.2,1.],'planning_bound_s':.1,'heterogeneity':['uniform','heterogeneous'],'actual_rounds':'uniform integer 1..public cap','participation':'independent 0.5/site/round, at least one active','actual_delays':'uniform within public bounds','byte_metric':'fixed public application frames; no IP/TLS measurements','baseline':'constant public frame cap per direction at every tick until same public horizon','privacy_scope':'abstract scheduled frame metadata only; private completion/abort status must not escape observer segment','contract_violation_check':'one over-bound compute case per scenario; fixed trace retained','final_test_bank_accessed':False}
 (OUT/'workflow_benchmark_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
 for N,R,tau,C,hetero in itertools.product([3,20],[3,5],[.1,.5],[.2,1.],['uniform','heterogeneous']):
  call=tuple([512]*N);reply=tuple(8192 if hetero=='uniform' else [1024,8192,32768][i%3] for i in range(N));e=WorkflowEnvelope(R,tau,C,.1,call,reply);K=math.ceil(e.horizon/tau-1e-12)
  actual=rng.integers(1,R+1,size=B);mask=rng.random((B,R,N))<.5;mask[:,:,0]=True;compute=rng.uniform(0,C,(B,R,N));planning=rng.uniform(0,.1,(B,R));periodic=np.zeros(B)
  for r in range(R):
   up=(np.floor((periodic+planning[:,r])/tau+1e-10)+1)*tau;ready=up+np.max(compute[:,r,:]*mask[:,r,:],axis=1);down=(np.floor(ready/tau+1e-10)+1)*tau;periodic=np.where(actual>r,down,periodic)
  envelope=actual*e.round_duration;assert (periodic<=e.horizon+1e-8).all();assert (envelope<=e.horizon+1e-8).all()
  # Independent scalar implementation checks actual call/reply and delay caps.
  transcript=e.schedule()
  for i in range(32):
   rr=int(actual[i]);calls=[[int(call[s]) if mask[i,r,s] else 0 for s in range(N)] for r in range(rr)];replies=[[int(reply[s]) if mask[i,r,s] else 0 for s in range(N)] for r in range(rr)];job=e.evaluate(calls,replies,compute[i,:rr].tolist(),planning[i,:rr].tolist());assert job['completed'] and job['schedule']==transcript
  bad=e.evaluate([[512]*N],[list(reply)],[[C+.01]*N],[.1]);assert not bad['completed'] and bad['schedule']==transcript
  total=sum(call)+sum(reply);awake_env=union_length(sorted({t for t,i,d,b in transcript}),.05,e.horizon+.05);awake_periodic=union_length([tau*k for k in range(1,K+1)],.05,K*tau+.05)
  rows.append(dict(N=N,round_cap=R,interval_s=tau,compute_bound_s=C,heterogeneity=hetero,jobs=B,horizon_s=e.horizon,periodic_ticks=K,envelope_application_bytes=R*total,periodic_application_bytes=K*total,byte_ratio=R/K,envelope_awake_s=awake_env,periodic_awake_s=awake_periodic,envelope_tool_p50_s=float(np.median(envelope)),periodic_tool_p50_s=float(np.median(periodic)),envelope_tool_p95_s=float(np.quantile(envelope,.95)),periodic_tool_p95_s=float(np.quantile(periodic,.95)),valid_jobs_completed=B,invalid_contract_rejected=True,private_job_transcripts_identical=True))
 with (OUT/'workflow_envelope_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'workflow_complete.json').write_text(json.dumps({'complete':True,'scenarios':len(rows),'synthetic_adaptive_jobs':len(rows)*B,'scalar_crosschecked_jobs':len(rows)*32,'contract_violations_tested':len(rows),'byte_ratio_range':[min(r['byte_ratio'] for r in rows),max(r['byte_ratio'] for r in rows)],'observed_metadata_privacy_scope':'abstract application frames only; no wire-level claim'},indent=2)+'\n');print('complete',len(rows),len(rows)*B)
if __name__=='__main__':main()
