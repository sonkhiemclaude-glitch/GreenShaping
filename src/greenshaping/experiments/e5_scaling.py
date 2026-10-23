"""E5: scaling with the number of sites and the query mix (Sec. 7.5).

For each N present in the data and each configured query mix, reports the unshaped
and shaped advantage, the modelled extra energy, and the Prop. 2(b) floor computed
from explicitly supplied, justified per-site lower bounds e_min,i. The modelled
mean extra cover energy is a diagnostic and is never substituted for a lower bound.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..analysis.bounds import any_traffic_detector_accuracy, network_cover_floor
from ..attack.evaluation import evaluate_attacker, max_advantage
from ..controller.grid import build_grid
from ..data.schemas import QUERY_FAMILIES
from ..energy.cost_model import extra_energy
from ..shaping.pipeline import RoundsParams, ShapingConfig
from .common import RunContext, node_models, shape_all


def _subset(sessions: list, n: int, fams: set[int]) -> list:
    return [s for s in sessions if s.n_sites == n and s.query_family in fams]


def _cover_e_min(sessions: list, rounds: RoundsParams, model: Any, seed: int) -> float:
    cfg = ShapingConfig(rounds=replace(rounds, cover_prob=1.0))
    shaped, _ = shape_all(sessions, cfg, seed, "e5-cover")
    vals = []
    for s, sh in zip(sessions, shaped, strict=False):
        e = extra_energy(sh, s, model(s))
        vals += [e[f"site{i}"] for i in range(s.n_sites) if not s.participation[i]]
    return float(np.mean(vals)) if vals else float("nan")


def run_e5(ctx: RunContext) -> list[dict[str, Any]]:
    a = ctx.cfg.get("attack", {})
    e5 = ctx.cfg.get("experiments", {}).get("e5", {})
    mixes = e5.get("query_mixes") or [list(QUERY_FAMILIES)]
    alpha = float(e5.get("alpha", 0.6))
    grid = [g for g in build_grid(ctx.cfg.get("grid", {})) if g.rounds is not None and g.n_primitives() == 1]
    model = node_models(ctx)
    n_boot = int(a.get("n_boot", 200))
    rows = []
    for n in ctx.n_values:
        for mix in mixes:
            fams = {QUERY_FAMILIES.index(f) for f in mix}
            fit, test = _subset(ctx.train + ctx.select, n, fams), _subset(ctx.test, n, fams)
            if len(fit) < 10 or len(test) < 5:
                continue
            base = evaluate_attacker(fit, test, seed=ctx.seed, n_boot=n_boot, **ctx.attack_kw())
            for g in grid:
                sh_fit, _ = shape_all(fit, g, ctx.seed, "e5-fit")
                sh_te, _ = shape_all(test, g, ctx.seed, "e5-test")
                res = evaluate_attacker(sh_fit, sh_te, seed=ctx.seed, n_boot=n_boot, **ctx.attack_kw())
                e = [extra_energy(x, y, model(y)) for x, y in zip(sh_te, test, strict=False)]
                assert g.rounds is not None
                e_min = _cover_e_min(test, g.rounds, model, ctx.seed)
                prior = float(np.mean([p for s in test for p in s.participation]))
                priors = np.mean([s.participation for s in test], axis=0)
                target_valid = bool(alpha >= max(max(float(p), 1 - float(p)) for p in priors))
                if e5.get("strict_target", ctx.data_kind == "real") and not target_valid:
                    raise ValueError("E5 alpha is below a per-site constant predictor; target is infeasible")
                lower_bounds = e5.get("e_min_per_site", {}).get(str(n))
                if lower_bounds is not None and len(lower_bounds) != n:
                    raise ValueError("E5 e_min_per_site must have exactly N justified lower bounds")
                certified_floor = (network_cover_floor(alpha, list(map(float, lower_bounds)))
                                   if target_valid and lower_bounds and len(lower_bounds) == n and e5.get("e_min_source") else float("nan"))
                q = g.rounds.cover_prob
                detector_by_site = [max(float(p), 1 - float(p), any_traffic_detector_accuracy(float(p), q)) for p in priors]
                det_acc = max(detector_by_site)
                rows.append({
                    "participation_prior": prior, "cover_prob": q,
                    "participation_priors_by_site": str(priors.tolist()), "target_alpha_valid": target_valid,
                    "any_traffic_detector_accuracy_prop2": det_acc,
                    "detector_accuracy_by_site": str(detector_by_site),
                    "target_alpha_met": bool(target_valid and det_acc <= alpha) if not np.isnan(det_acc) else False,
                    "experiment": "E5", "N": n, "query_mix": "+".join(mix), "config": g.label(),
                    "adv_max_unshaped": max_advantage(base), "adv_max_shaped": max_advantage(res),
                    "adv_participation_shaped": res["participation"]["adv_raw"],
                    "adv_participation_shaped_ci_low": res["participation"]["adv_ci_low"],
                    "adv_participation_shaped_ci_high": res["participation"]["adv_ci_high"],
                    "energy_modelled_total_sites_J": float(np.mean([sum(v for k, v in x.items() if k.startswith("site")) for x in e])),
                    "cover_energy_mean_modelled_J": e_min, "e_min_cover_modelled_J": float("nan"), "alpha": alpha,
                    "prop2_network_floor_modelled_J": certified_floor,
                    "e_min_source": e5.get("e_min_source", ""),
                })
    if not rows:
        raise ValueError("E5 produced no rows: need >= 10 fit and >= 5 test sessions per (N, mix)")
    return rows
