"""Fixed public workflow cover schedule. Abstract application frames only.

All protected tasks share the same public envelope. A violation never extends
or resizes this schedule. This does not certify packet-level TLS behaviour.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class WorkflowEnvelope:
    rounds: int
    interval: float
    compute_bound: float
    planning_bound: float
    call_caps: tuple[int,...]
    reply_caps: tuple[int,...]
    delivery_bound: float = 0.0

    def __post_init__(self):
        if isinstance(self.rounds,bool) or not isinstance(self.rounds,int) or self.rounds<1:raise ValueError('positive integer rounds required')
        if not math.isfinite(self.interval) or self.interval<=0:raise ValueError('positive interval required')
        if any(not math.isfinite(x) or x<0 for x in (self.compute_bound,self.planning_bound,self.delivery_bound)):raise ValueError('invalid delay bounds')
        if not self.call_caps or len(self.call_caps)!=len(self.reply_caps):raise ValueError('site dimensions mismatch')
        if any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in self.call_caps+self.reply_caps):raise ValueError('positive integer frame caps required')

    @property
    def round_duration(self):return self.planning_bound+self.compute_bound+2*self.interval+2*self.delivery_bound
    @property
    def horizon(self):return self.rounds*self.round_duration+self.delivery_bound
    def schedule(self):
        rows=[]
        for r in range(self.rounds):
            up=r*self.round_duration+self.delivery_bound+self.planning_bound+self.interval
            down=(r+1)*self.round_duration
            for i,(call,reply) in enumerate(zip(self.call_caps,self.reply_caps)):
                rows.extend([(up,i,'up',call),(down,i,'down',reply)])
        return sorted(rows)
    def direction_byte_bounds(self):return [self.rounds*x for pair in zip(self.call_caps,self.reply_caps) for x in pair]

    def evaluate(self,calls,replies,compute_delays,planning_delays):
        """One fixed-round-index job; arrays [actual round][site]. Private inputs.

        Frames for unused public rounds/sites still appear in schedule(). Returned
        admission/violation metadata is internal and is not part of the transcript.
        """
        R=len(calls);N=len(self.call_caps)
        if R<1 or any(len(x)!=R for x in (replies,compute_delays)) or len(planning_delays)!=R:raise ValueError('job rounds mismatch')
        if any(len(row)!=N for table in (calls,replies,compute_delays) for row in table):raise ValueError('job site mismatch')
        if any(not math.isfinite(float(v)) or v<0 for table in (calls,replies,compute_delays) for row in table for v in row) or any(not math.isfinite(float(v)) or v<0 for v in planning_delays):raise ValueError('invalid job values')
        reason=[]
        if R>self.rounds:reason.append('round_count')
        for r in range(R):
            if planning_delays[r]>self.planning_bound:reason.append('planning_delay')
            for i in range(N):
                if calls[r][i]>self.call_caps[i]:reason.append('call_volume')
                if replies[r][i]>self.reply_caps[i]:reason.append('reply_volume')
                if calls[r][i] and compute_delays[r][i]>self.compute_bound:reason.append('compute_delay')
                if bool(calls[r][i])!=bool(replies[r][i]):reason.append('call_reply_participation')
        ok=not reason
        return {'completed':ok,'contract_violations':sorted(set(reason)),'tool_completion':R*self.round_duration+self.delivery_bound if ok else None,'public_horizon':self.horizon,'direction_bytes':self.direction_byte_bounds(),'schedule':self.schedule()}

def admit_envelope(envelope:WorkflowEnvelope,cost_matrix,awake_cost,budgets,deadline:float):
    """Public pre-arrival admission, based on deterministic frame byte bounds.

    Cost units follow supplied nonnegative coefficients; no implicit J claim.
    Rejection denotes failure to certify this envelope, not global infeasibility.
    """
    import numpy as np
    A=np.asarray(cost_matrix,dtype=float);awake=np.asarray(awake_cost,dtype=float);budget=np.asarray(budgets,dtype=float)
    counts=np.asarray(envelope.direction_byte_bounds(),dtype=float)
    if not math.isfinite(deadline) or deadline<=0:raise ValueError('invalid deadline')
    if A.ndim!=2 or A.shape[1]!=len(counts) or awake.shape!=(A.shape[0],) or budget.shape!=awake.shape:raise ValueError('cost dimensions mismatch')
    if not np.isfinite(A).all() or not np.isfinite(awake).all() or not np.isfinite(budget).all() or (A<0).any() or (awake<0).any() or (budget<=0).any():raise ValueError('invalid public cost model')
    cost=A@counts+awake;ratio=float(np.max(cost/budget))
    return {'accepted':envelope.horizon<=deadline+1e-12 and ratio<=1+1e-12,'node_cost_bound':cost.tolist(),'worst_budget_ratio':ratio,'completion_risk_under_contract':0.,'public_horizon':envelope.horizon,'privacy_scope':'fixed application-frame transcript; not wire-level certification'}
