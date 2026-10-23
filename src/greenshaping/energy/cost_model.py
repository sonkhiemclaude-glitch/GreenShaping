"""Per-node energy cost model, Eq. (3).

E_raw_n(theta) = e_tx * B_tx_n + e_rx * B_rx_n + p_aw * T_aw_n
E_hat_n(theta) = E_raw_n(theta) - E_raw_n(unshaped)

T_aw_n is the length of the union of [t, t + t_tail] over all times t at which
node n sends, receives or wakes for a DP tick. The model RANKS configurations in
Algorithm 1; every energy figure reported as a result must be measured (Sec. 4.4).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..data.schemas import DOWN, UP, Session

ORCHESTRATOR = "orchestrator"


def site_node(i: int) -> str:
    return f"site{i}"


@dataclass(frozen=True)
class CostModel:
    """Coefficients of Eq. (3) for one node class. Must be fitted from measurements (energy.fit)."""

    node_class: str
    e_tx: float      # J per byte sent
    e_rx: float      # J per byte received
    p_aw: float      # W above idle while active
    t_tail: float    # s of activity after each event
    fitted: bool = False   # False for placeholder coefficients (synthetic demo only)

    def __post_init__(self) -> None:
        for k in ("e_tx", "e_rx", "p_aw", "t_tail"):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError(f"{k} must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


def union_length(times: list[float], tail: float, horizon: float | None = None) -> float:
    """Length of the union of [t, t + tail] (clipped at `horizon` if given)."""
    if not times or tail <= 0:
        return 0.0
    cap = float("inf") if horizon is None else horizon
    ts = sorted(times)
    total = 0.0
    start, end = ts[0], min(ts[0] + tail, cap)
    for t in ts[1:]:
        if t <= end:
            end = max(end, min(t + tail, cap))
        else:
            total += max(0.0, end - start)
            start, end = t, min(t + tail, cap)
    return total + max(0.0, end - start)


def node_energy_raw(sess: Session, model: CostModel | dict[str, CostModel]) -> dict[str, float]:
    """E_raw_n for every site and the orchestrator (Eq. 3). `model` may map node name -> CostModel."""
    def m_for(node: str) -> CostModel:
        if isinstance(model, CostModel):
            return model
        if node not in model:
            raise KeyError(f"no cost model for node {node}")
        return model[node]

    horizon = max(sess.duration(), max((m.t for m in sess.messages), default=0.0))
    res: dict[str, float] = {}
    all_times: list[float] = []
    o_tx = o_rx = 0
    for s in range(sess.n_sites):
        link = sess.link(s)
        tx = sum(m.size for m in link if m.direction == DOWN)
        rx = sum(m.size for m in link if m.direction == UP)
        times = [m.t for m in link] + list(sess.wake.get(s, []))
        all_times += times
        o_tx += rx
        o_rx += tx
        cm = m_for(site_node(s))
        res[site_node(s)] = cm.e_tx * tx + cm.e_rx * rx + cm.p_aw * union_length(times, cm.t_tail, horizon + cm.t_tail)
    cm = m_for(ORCHESTRATOR)
    res[ORCHESTRATOR] = cm.e_tx * o_tx + cm.e_rx * o_rx + cm.p_aw * union_length(all_times, cm.t_tail, horizon + cm.t_tail)
    return res


def extra_energy(shaped: Session, unshaped: Session, model: CostModel | dict[str, CostModel]) -> dict[str, float]:
    """E_hat_n(theta) = E_raw_n(theta) - E_raw_n(unshaped), Eq. (3)."""
    a, b = node_energy_raw(shaped, model), node_energy_raw(unshaped, model)
    return {n: a[n] - b[n] for n in a}


def bytes_on_wire(sess: Session) -> int:
    return sum(m.size for m in sess.messages)
