"""Public-contract completion bounds and per-node expected-cost admission.

rho is a utility/completion risk, NOT approximate-DP delta. All inputs must be
public and fixed before protected arrivals. Modelled costs are NOT measured J.
No hard realised-cost guarantee is inferred from an expected cost bound.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize

@dataclass(frozen=True)
class DirectionContract:
    name: str
    volume_cap: int
    occupied_tick_cap: int
    ticks: int
    epsilon: float
    delta: float
    release_cap: int | None = None
    max_tick_arrival: int | None = None

    def __post_init__(self):
        for name in ('volume_cap','occupied_tick_cap','ticks'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(name+' must be a positive integer')
        if self.occupied_tick_cap>self.ticks:raise ValueError('occupied ticks exceed schedule')
        if not math.isfinite(self.epsilon) or self.epsilon<=0:raise ValueError('epsilon must be finite and positive')
        if not math.isfinite(self.delta) or self.delta<2*self.volume_cap:raise ValueError('timing-shift adjacency needs Delta >= 2 volume cap')
        if self.max_tick_arrival is not None and (isinstance(self.max_tick_arrival,bool) or not isinstance(self.max_tick_arrival,int) or not 1<=self.max_tick_arrival<=self.volume_cap):raise ValueError('invalid per-tick arrival cap')
        if self.release_cap is not None and (isinstance(self.release_cap,bool) or not isinstance(self.release_cap,int) or self.release_cap<self.tick_arrival_cap):raise ValueError('release cap must be integer >= per-tick arrival cap')

    @property
    def tick_arrival_cap(self):return self.volume_cap if self.max_tick_arrival is None else self.max_tick_arrival

    @property
    def scale(self):return self.delta/self.epsilon

def bias_for_risk(c:DirectionContract,risk:float)->float:
    if not math.isfinite(risk) or not 0<risk<1:raise ValueError('completion risk must be in (0,1)')
    # log form avoids M/(2 risk) overflow for very small risk.
    return max(0.,c.scale*(math.log(c.occupied_tick_cap)-math.log(2.)-math.log(risk))-.5)

def completion_bound(c:DirectionContract,bias:float)->float:
    if not math.isfinite(bias) or bias<0:raise ValueError('bias must be finite and nonnegative')
    return min(1.,c.occupied_tick_cap*.5*math.exp(-(bias+.5)/c.scale))

def expected_release_upper(c:DirectionContract,bias:float)->float:
    if not math.isfinite(bias) or bias<0:raise ValueError('bias must be finite and nonnegative')
    result=c.volume_cap+c.ticks*(bias+.5*c.scale*math.exp(-bias/c.scale)+.5)
    return min(result,c.ticks*c.release_cap) if c.release_cap is not None else result

def public_workflow_horizon(rounds:int,interval:float,compute_bound:float,planning_bound:float=0.)->float:
    if isinstance(rounds,bool) or not isinstance(rounds,int) or rounds<1:raise ValueError('rounds must be positive integer')
    if not math.isfinite(interval) or interval<=0:raise ValueError('interval must be positive')
    if any(not math.isfinite(x) or x<0 for x in (compute_bound,planning_bound)):raise ValueError('workflow delays must be finite nonnegative')
    return rounds*(2*interval+compute_bound+planning_bound)

@dataclass
class Admission:
    accepted: bool
    policy: str
    risks: list[float]
    biases: list[float]
    failure_upper: float
    direction_byte_upper: list[float]
    node_cost_upper: list[float]
    worst_budget_ratio: float
    optimizer_success: bool
    message: str

def allocate(contracts:list[DirectionContract],rho:float,cost_matrix,awake_cost,budgets,*,optimize:bool=True)->Admission:
    if not contracts:raise ValueError('empty deployment')
    if not math.isfinite(rho) or not 0<rho<1:raise ValueError('rho must be in (0,1)')
    A=np.asarray(cost_matrix,dtype=float);awake=np.asarray(awake_cost,dtype=float);budget=np.asarray(budgets,dtype=float);D=len(contracts)
    if A.ndim!=2 or A.shape[1]!=D or awake.shape!=(A.shape[0],) or budget.shape!=awake.shape:raise ValueError('node/direction dimensions mismatch')
    if not np.isfinite(A).all() or not np.isfinite(awake).all() or not np.isfinite(budget).all() or (A<0).any() or (awake<0).any() or (budget<=0).any():raise ValueError('invalid cost coefficients or budgets')
    uniform=np.full(D,1/D)
    def profile(fractions):
        risks=rho*np.asarray(fractions);bias=np.array([bias_for_risk(c,float(r)) for c,r in zip(contracts,risks)])
        release=np.array([expected_release_upper(c,float(b)) for c,b in zip(contracts,bias)])
        return risks,bias,release,A@release+awake
    def derivative(fractions):
        _,bias,release,_=profile(fractions);g=[]
        for c,b,f,y in zip(contracts,bias,fractions,release):
            if b<=0 or (c.release_cap is not None and y>=c.ticks*c.release_cap):g.append(0.)
            else:g.append(-c.ticks*c.scale*(1-.5*math.exp(-b/c.scale))/f)
        return np.array(g)
    def cons(x):return x[-1]-(profile(x[:-1])[3]/budget)
    def jac(x):return np.column_stack((-A*derivative(x[:-1])[None,:]/budget[:,None],np.ones(len(budget))))
    candidate=uniform;success=True;message='equal allocation'
    if optimize:
        x0=np.r_[uniform,float(np.max(profile(uniform)[3]/budget))]
        result=minimize(lambda x:x[-1],x0,jac=lambda x:np.r_[np.zeros(D),1.],method='SLSQP',bounds=[(1e-10,1.)]*D+[(0.,None)],constraints=[{'type':'eq','fun':lambda x:np.sum(x[:-1])-1,'jac':lambda x:np.r_[np.ones(D),0.]},{'type':'ineq','fun':cons,'jac':jac}],options={'maxiter':500,'ftol':1e-10})
        success=bool(result.success);message=str(result.message)
        proposed=np.asarray(result.x[:-1]);valid=np.isfinite(proposed).all() and (proposed>0).all() and abs(proposed.sum()-1)<1e-6
        if valid:
            proposed=proposed/proposed.sum()
            if np.max(profile(proposed)[3]/budget)<=np.max(profile(uniform)[3]/budget)+1e-10:candidate=proposed
    risks,bias,release,cost=profile(candidate);upper=sum(completion_bound(c,float(b)) for c,b in zip(contracts,bias));ratio=float(np.max(cost/budget))
    # Recompute every certificate after numerical optimization; never trust solver status alone.
    accepted=upper<=rho*(1+1e-10) and ratio<=1+1e-9
    return Admission(accepted,'optimized_public_contract' if optimize else 'uniform_public_contract',risks.tolist(),bias.tolist(),upper,release.tolist(),cost.tolist(),ratio,success,message)

@dataclass(frozen=True)
class PolicyCertificate:
    name: str
    contract_id: str
    epsilon_upper: float
    completion_risk_upper: float
    node_cost_upper: tuple[float,...]
    horizon: float
    cost_kind: str = 'expected'

    def __post_init__(self):
        if not self.name or not self.contract_id:raise ValueError('public identity required')
        if self.cost_kind not in ('expected','deterministic_application_cost'):raise ValueError('unknown cost certificate kind')
        if not self.node_cost_upper or any(not math.isfinite(x) or x<0 for x in self.node_cost_upper):raise ValueError('invalid node costs')
        if not math.isfinite(self.epsilon_upper) or self.epsilon_upper<0 or not 0<=self.completion_risk_upper<=1 or not math.isfinite(self.horizon) or self.horizon<=0:raise ValueError('invalid policy certificate')

def choose_certified_policy(candidates:list[PolicyCertificate],budgets,epsilon:float,rho:float,deadline:float,*,require_deterministic_cost=False):
    """Select only comparable public certificates; no private job inputs.

    Privacy/cost certificates must refer to the declared abstract observation
    and common contract. This function cannot verify their mathematical premises.
    """
    budget=np.asarray(budgets,dtype=float)
    if budget.ndim!=1 or not len(budget) or not np.isfinite(budget).all() or (budget<=0).any():raise ValueError('invalid budgets')
    if not math.isfinite(epsilon) or epsilon<0 or not math.isfinite(rho) or not 0<=rho<1 or not math.isfinite(deadline) or deadline<=0:raise ValueError('invalid public targets')
    if len({c.contract_id for c in candidates})>1:raise ValueError('policies have different workload contracts')
    feasible=[]
    for c in candidates:
        if len(c.node_cost_upper)!=len(budget):raise ValueError('node dimensions mismatch')
        if require_deterministic_cost and c.cost_kind!='deterministic_application_cost':continue
        ratio=float(np.max(np.asarray(c.node_cost_upper)/budget))
        if c.epsilon_upper<=epsilon+1e-12 and c.completion_risk_upper<=rho+1e-12 and c.horizon<=deadline+1e-12 and ratio<=1+1e-12:feasible.append((ratio,c.horizon,c.name,c))
    return min(feasible,key=lambda item:item[:3])[-1] if feasible else None
