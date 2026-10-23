import importlib.util,math
from pathlib import Path
import numpy as np
import pytest
from greenshaping.controller.reliability import DirectionContract,bias_for_risk,completion_bound,expected_release_upper,allocate,public_workflow_horizon
from greenshaping.shaping.dp_release import release_link

@pytest.mark.parametrize('ticks,occupied,eps,risk',[(20,3,.2,.001),(1,1,1.,.1),(100,5,4.,1e-12)])
def test_bias_meets_union_bound(ticks,occupied,eps,risk):
 c=DirectionContract('x',8192,occupied,ticks,eps,16384)
 b=bias_for_risk(c,risk)
 assert completion_bound(c,b)<=risk*(1+1e-12)
 assert completion_bound(c,max(0,b-c.scale))>risk

def test_sharper_public_contract_reduces_bound_and_cost():
 a=DirectionContract('a',8192,20,20,.2,16384);b=DirectionContract('b',8192,3,20,.2,16384)
 assert bias_for_risk(b,.001)<bias_for_risk(a,.001)
 assert expected_release_upper(b,bias_for_risk(b,.001))<expected_release_upper(a,bias_for_risk(a,.001))

def test_controller_rechecks_feasibility_and_can_reject():
 cc=[DirectionContract(str(i),1024,2,10,1,2048) for i in range(4)]
 A=np.array([[1.,1,0,0],[0,0,1.,1],[1,1,1,1]]);awake=[0,0,0]
 ref=allocate(cc,.01,A,awake,[1e9]*3,optimize=False)
 budget=np.array(ref.node_cost_upper)*[.96,1.08,1.02]
 opt=allocate(cc,.01,A,awake,budget);plain=allocate(cc,.01,A,awake,budget,optimize=False)
 assert opt.worst_budget_ratio<=plain.worst_budget_ratio+1e-9
 assert sum(opt.risks)<=.01*(1+1e-12)
 assert opt.failure_upper<=.01*(1+1e-12)
 assert not allocate(cc,.01,A,awake,[1.,1,1]).accepted
 assert len(set(round(v,9) for v in opt.risks))>1

def test_cap_matches_postprocessing_for_single_tick_and_keeps_real_delivery():
 for seed in range(80):
  args=([(0.5,8,0)],[1.],16.,'arrivals',32.)
  unbounded,done,left=release_link(*args,np.random.default_rng(seed))
  capped,done2,left2=release_link(*args,np.random.default_rng(seed),release_cap_bytes=8)
  assert [(t,min(n,8),real) for t,n,real in unbounded]==capped
  assert left==left2 and done==done2

def test_uncapped_replay_is_backward_identical():
 import hashlib,json
 fixtures=json.loads((Path(__file__).parent/'fixtures/release_v050_hashes.json').read_text())
 for row in fixtures:
  result=release_link([(0.1,8,0),(.8,8,1)],[.25,.5,.75,1.],16.,row['mode'],4.,np.random.default_rng(row['seed']))
  assert hashlib.sha256(repr(result).encode()).hexdigest()==row['sha256']

def test_invalid_contract_and_cost_inputs_rejected():
 with pytest.raises(ValueError):DirectionContract('bad',10,3,2,1,20)
 with pytest.raises(ValueError):DirectionContract('bad',10,1,2,1,19)
 with pytest.raises(ValueError):DirectionContract('bad',10,1,2,1,20,9)
 c=DirectionContract('x',10,1,2,1,20)
 with pytest.raises(ValueError):bias_for_risk(c,0)
 with pytest.raises(ValueError):allocate([c],.1,[[1]],[-1],[1])
 assert public_workflow_horizon(3,.5,.2,.1)==pytest.approx(3.9)

def test_public_workflow_has_identical_transcripts_for_different_private_jobs():
 from greenshaping.controller.workflow_contract import WorkflowEnvelope
 e=WorkflowEnvelope(3,.1,.2,.05,(256,512),(1024,2048))
 a=e.evaluate([[100,0]],[[900,0]],[[.1,0]],[.01])
 b=e.evaluate([[0,200],[100,300]],[[0,1000],[100,1000]],[[0,.2],[.1,.05]],[.02,.03])
 assert a['completed'] and b['completed']
 assert a['schedule']==b['schedule'] and a['direction_bytes']==b['direction_bytes']
 assert a['tool_completion']<b['tool_completion']<=e.horizon
 # Worst-case legal plan/compute delays still fit between fixed release slots.
 for r in range(3):
  call=r*e.round_duration+e.planning_bound+e.interval
  reply=(r+1)*e.round_duration
  assert reply>=call+e.compute_bound

def test_contract_failure_never_expands_the_public_schedule():
 from greenshaping.controller.workflow_contract import WorkflowEnvelope
 e=WorkflowEnvelope(1,.1,.2,.05,(256,),(1024,))
 bad=e.evaluate([[257]],[[1024]],[[.3]],[.1])
 assert not bad['completed']
 assert set(bad['contract_violations'])=={'call_volume','compute_delay','planning_delay'}
 assert bad['schedule']==e.schedule()

def test_envelope_admission_uses_only_public_contract_and_can_reject():
 from greenshaping.controller.workflow_contract import WorkflowEnvelope,admit_envelope
 e=WorkflowEnvelope(3,.1,.2,.05,(256,),(1024,));A=[[1.,1.]]
 assert admit_envelope(e,A,[0],[4000],2.)['accepted']
 assert not admit_envelope(e,A,[0],[3000],2.)['accepted']
 assert not admit_envelope(e,A,[0],[4000],1.)['accepted']

def test_public_clipping_keeps_tail_atoms_without_resampling():
 from scipy.stats import laplace
 eps=1.;delta=16.;scale=delta/eps;cap=16;bias=8.
 def distribution(arrival):
  centre=arrival+bias
  return np.array([laplace.cdf(.5-centre,scale=scale)]+[laplace.cdf(i+.5-centre,scale=scale)-laplace.cdf(i-.5-centre,scale=scale) for i in range(1,cap)]+[laplace.sf(cap-.5-centre,scale=scale)])
 p,q=distribution(0),distribution(16)
 assert p.sum()==pytest.approx(1.) and q.sum()==pytest.approx(1.)
 assert np.max(np.abs(np.log(p/q)))<=eps+1e-12

def test_policy_selector_respects_cost_kind_privacy_and_common_contract():
 from greenshaping.controller.reliability import PolicyCertificate,choose_certified_policy
 cheap=PolicyCertificate('expected','same',1.,.001,(1.,),2.)
 fixed=PolicyCertificate('fixed','same',0.,0.,(2.,),2.,'deterministic_application_cost')
 assert choose_certified_policy([cheap,fixed],[3],1,.01,3).name=='expected'
 assert choose_certified_policy([cheap,fixed],[3],1,.01,3,require_deterministic_cost=True).name=='fixed'
 assert choose_certified_policy([cheap],[3],0,.01,3) is None
 with pytest.raises(ValueError):choose_certified_policy([cheap,PolicyCertificate('x','different',0,0,(1.,),1.)],[3],1,.01,3)

def test_per_tick_cap_can_be_smaller_than_session_volume_with_public_contract():
 c=DirectionContract('stream',32,4,8,1.,64.,release_cap=8,max_tick_arrival=8)
 assert expected_release_upper(c,bias_for_risk(c,.001))<=64
 with pytest.raises(ValueError):DirectionContract('bad',32,4,8,1.,64.,release_cap=7,max_tick_arrival=8)

def test_public_delivery_bound_leaves_time_for_both_network_directions():
 from greenshaping.controller.workflow_contract import WorkflowEnvelope
 e=WorkflowEnvelope(3,.1,.2,.05,(256,),(1024,),delivery_bound=.04)
 for r in range(3):
  up=r*e.round_duration+e.delivery_bound+e.planning_bound+e.interval
  down=(r+1)*e.round_duration
  assert up+e.delivery_bound+e.compute_bound<=down
  assert down+e.delivery_bound<=e.horizon
  if r<2:
   next_up=(r+1)*e.round_duration+e.delivery_bound+e.planning_bound+e.interval
   assert down+e.delivery_bound+e.planning_bound<=next_up
