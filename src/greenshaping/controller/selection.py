"""Algorithm 1: Green Shaping configuration selection (Sec. 4.5).

Lines refer to Algorithm 1 in the paper.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..attack.evaluation import assert_disjoint, evaluate_attacker, max_advantage
from ..data.schemas import Session
from ..energy.cost_model import CostModel, bytes_on_wire, extra_energy
from ..metrics.metrics import bandwidth_overhead, latency_summary
from ..shaping.pipeline import ShapingConfig, apply_shaping, resolve_delta
from ..shaping.stats import last_real_reply
from ..utils.random import child_seed

log = logging.getLogger(__name__)
INFEASIBLE = "INFEASIBLE"


@dataclass
class Profile:
    """Everything Algorithm 1 needs about one candidate theta (plus reporting fields)."""

    config: ShapingConfig
    overflow_any: bool
    failure_rate: float
    extra_energy: dict[str, float]           # mean E_hat_n over selection sessions (Eq. 3)
    advantage: dict[str, dict[str, float]] = field(default_factory=dict)
    adv_max: float = float("nan")
    bandwidth_overhead: float = float("nan")
    latency: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        r: dict[str, Any] = {"config": self.config.label(), "n_primitives": self.config.n_primitives(),
                             "config_id": self.config.config_id(),
                             "overflow_any": self.overflow_any, "failure_rate": self.failure_rate,
                             "adv_max": self.adv_max, "bandwidth_overhead": self.bandwidth_overhead}
        for k, v in self.advantage.items():
            r[f"adv_{k}"] = v.get("adv_raw")          # raw normalised advantage per secret
            r[f"acc_{k}"] = v.get("acc")
            r[f"attacker_{k}"] = v.get("attacker")
        sites = [v for k, v in self.extra_energy.items() if k.startswith("site")]
        r["energy_modelled_site_max_J"] = max(sites) if sites else float("nan")
        r["energy_modelled_max_node_J"] = max(self.extra_energy.values()) if self.extra_energy else float("nan")
        r["energy_modelled_orchestrator_J"] = self.extra_energy.get("orchestrator", float("nan"))
        r["n_secrets_scored"] = sum(1 for v in self.advantage.values() if not math.isnan(v.get("adv", math.nan)))
        r.update(self.latency)
        return r


def shape_sessions(sessions: list[Session], cfg: ShapingConfig, seed: int) -> tuple[list[Session], list[Any]]:
    rng = np.random.default_rng(seed)
    shaped, stats = [], []
    for s in sessions:
        sh, st = apply_shaping(s, cfg, rng)
        shaped.append(sh)
        stats.append(st)
    return shaped, stats


def mean_node_energy(energies: list[dict[str, float]]) -> dict[str, float]:
    """Mean extra energy per node over the sessions that CONTAIN the node (review B2)."""
    nodes = sorted({n for e in energies for n in e})
    return {n: float(np.mean([e[n] for e in energies if n in e])) for n in nodes}


def profile_configuration(cfg: ShapingConfig, train: list[Session], select: list[Session],
                          cost_model: CostModel | dict[str, CostModel] | Callable[[Session], Any], seed: int = 0,
                          n_estimators: int = 200, skip_attack_if_infeasible: dict[str, float] | None = None,
                          attack_kw: dict[str, Any] | None = None) -> Profile:
    """Algorithm 1, lines 3-9 for one theta. A deployment has one N (Sec. 3.1): pass sessions
    of a single N (the runners split by N); mixed N is refused.

    `cost_model` may be a callable returning the per-node cost models of a session
    (used for per-site node classes from the manifests, review M4).
    """
    assert_disjoint(train, select)
    ns = {s.n_sites for s in train + select}
    if len(ns) > 1:
        raise ValueError(f"Algorithm 1 runs per deployment (one N); got N = {sorted(ns)}. Split sessions by N first.")
    cfg = resolve_delta(cfg, train + select, seed)                      # line 3
    sh_tr, st_tr = shape_sessions(train, cfg, child_seed(seed, cfg.key(), "train"))   # line 4
    sh_se, st_se = shape_sessions(select, cfg, child_seed(seed, cfg.key(), "select"))
    flags = [st.overflow_bytes > 0 for st in st_tr + st_se]
    fails = [st.privacy_failure for st in st_tr + st_se]
    energies = [extra_energy(a, b, cost_model(b) if callable(cost_model) else cost_model)
                for a, b in zip(sh_se, select, strict=False)]
    mean_e = mean_node_energy(energies)                                  # line 6
    lat = [st.real_done - last_real_reply(s.messages) for s, st in zip(select, st_se, strict=False)]
    prof = Profile(cfg, any(flags), float(np.mean(fails)) if fails else 0.0, mean_e,
                   bandwidth_overhead=bandwidth_overhead(sum(map(bytes_on_wire, sh_se)), sum(map(bytes_on_wire, select))),
                   latency=latency_summary(lat))
    if skip_attack_if_infeasible is not None and not is_feasible(prof, skip_attack_if_infeasible):
        return prof                                                      # lines 5 and 7: no attacker needed
    kw = {"n_estimators": n_estimators, **(attack_kw or {})}
    prof.advantage = evaluate_attacker(sh_tr, sh_se, seed=seed, **kw)   # line 8
    prof.adv_max = max_advantage(prof.advantage)                         # line 9
    return prof


def is_feasible(p: Profile, budgets: dict[str, float]) -> bool:
    """Membership in F: no overflow (line 5) and every budgeted node within J_n (line 7)."""
    if p.overflow_any or not math.isfinite(p.failure_rate) or p.failure_rate > 0:
        return False
    missing = [n for n in budgets if n not in p.extra_energy]
    if missing:
        raise KeyError(f"budgets name nodes {missing} with no energy value; budgets must match the deployment's N")
    return all(p.extra_energy[n] <= J for n, J in budgets.items())


def select_configuration(profiles: list[Profile], budgets: dict[str, float]) -> Profile | str:
    """Line 11: lexicographic argmin of (A_hat, max_n E_hat_n) over F; ties keep grid order."""
    feasible = [p for p in profiles if is_feasible(p, budgets) and not math.isnan(p.adv_max)]
    if not feasible:
        return INFEASIBLE
    return min(feasible, key=lambda p: (p.adv_max, max(p.extra_energy.values()) if p.extra_energy else 0.0))


def check_remark1(profiles: list[Profile], budgets: dict[str, float]) -> bool:
    """Remark 1: if F(Theta_1) is non-empty, A_hat(theta*) <= min over F(Theta_1)."""
    sel = select_configuration(profiles, budgets)
    singles = [p for p in profiles if p.config.n_primitives() <= 1 and is_feasible(p, budgets) and not math.isnan(p.adv_max)]
    if not singles:
        return True
    return isinstance(sel, Profile) and sel.adv_max <= min(p.adv_max for p in singles) + 1e-12


def uniform_budgets(n_sites: int, j_site: float, j_orchestrator: float | None = None) -> dict[str, float]:
    b = {f"site{i}": j_site for i in range(n_sites)}
    if j_orchestrator is not None:
        b["orchestrator"] = j_orchestrator
    return b
