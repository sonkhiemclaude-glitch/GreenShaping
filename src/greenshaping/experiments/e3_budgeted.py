"""E3: budgeted shaping (Algorithm 1) against fixed-size padding and NetShaper (Sec. 7.3).

1. Profile every theta in the grid on attacker-train / selection questions (Alg. 1).
2. For each budget, select theta* (Alg. 1, line 11) and record the best single primitive.
3. Retrain the attacker on theta*'s shaped train+selection traffic and score it on the
   final-test questions; same for fixed-size padding and (if captures exist) NetShaper.
Runs for ONE deployment size N (the runner calls it per N; review B2).
Budgets: `budgets.site_J` (list). On synthetic or rehearsal data, quantiles of the
modelled per-node cost are used when it is absent (pipeline testing only); real data
must set budgets explicitly (review M3). E3 in the paper compares at MEASURED energy:
selected, best-single and baseline rows all carry the measured columns when
`energy.measured_energy_csv` is given.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..attack.evaluation import evaluate_attacker
from ..baselines.defences import FixedPaddingDefence
from ..baselines.netshaper import NetShaperCaptureAdapter
from ..controller.grid import build_grid
from ..controller.selection import (
    INFEASIBLE,
    Profile,
    check_remark1,
    profile_configuration,
    select_configuration,
    uniform_budgets,
)
from ..shaping.pipeline import ShapingConfig
from .common import RunContext, add_co2e, cost_summary, merge_measured_energy, node_models, row_identity
from .e2_primitives import attack_columns, profile_on_test

log = logging.getLogger(__name__)


def _defence_on_test(ctx: RunContext, defence: Any, node_class: str | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    defence.prepare(ctx.train + ctx.select)
    fit = [defence.apply(s, rng)[0] for s in ctx.train + ctx.select]
    te = [defence.apply(s, rng) for s in ctx.test]
    a = ctx.cfg.get("attack", {})
    res = evaluate_attacker(fit, [x[0] for x in te], seed=ctx.seed, n_boot=int(a.get("n_boot", 200)), **ctx.attack_kw())
    row: dict[str, Any] = {"config": defence.name, "defence_kind": defence.kind, "node_class": node_class or "per-manifest"}
    if isinstance(defence, FixedPaddingDefence):
        row.update(row_identity(ctx, ShapingConfig(fixed_unit=defence.unit), node_class))
    elif isinstance(defence, NetShaperCaptureAdapter):
        if ctx.data_kind == "real" and not defence.config.get("source_commit"):
            raise ValueError("NetShaper comparison requires a pinned source_commit in netshaper_config.json")
        row.update(row_identity(ctx, ShapingConfig(), node_class))
        row["config_id"] = defence.config_id()
    row.update(attack_columns(res))
    row.update(cost_summary(ctx, ctx.test, [x[0] for x in te], [x[1] for x in te], node_models(ctx, node_class)))
    return row


FINAL_KEYS = ("adv", "acc", "auc", "chance", "balanced", "train_majority", "attacker", "n_secrets", "energy", "co2e",
              "bandwidth", "latency", "overflow", "privacy_failure", "unpaired", "config_id")


def run_e3(ctx: RunContext, final_eval: bool = True,
           node_class: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (selection rows, profile rows of the whole grid) for one N.

    final_eval=False runs only Algorithm 1 (stage "calibrate"); True also scores the selected
    configurations and the baselines on the final-test questions (stage "evaluate").
    """
    if len(ctx.n_values) != 1:
        raise ValueError(f"E3 runs per deployment size; got N = {ctx.n_values} (use the runner, which splits by N)")
    n_sites = ctx.n_values[0]
    a = ctx.cfg.get("attack", {})
    grid = build_grid(ctx.cfg.get("grid", {}))
    models = node_models(ctx, node_class)
    profiles: list[Profile] = [profile_configuration(g, ctx.train, ctx.select, models, ctx.seed,
                                                      int(a.get("n_estimators", 200)), attack_kw=ctx.attack_kw())
                               for g in grid]
    prof_rows = [p.row() | {"experiment": "E3-profile", "node_class": node_class or "per-manifest"} for p in profiles]
    energy_cfg = ctx.cfg.get("energy", {})
    b = ctx.cfg.get("budgets", {})
    budgets = b.get("site_J")
    if not budgets:
        if ctx.data_kind == "real":
            raise ValueError("real data: set budgets.site_J (joules per site and session) from MEASURED energy (review M3)")
        costs = [max(v for k, v in p.extra_energy.items() if k.startswith("site")) for p in profiles if p.extra_energy]
        budgets = sorted({float(np.quantile(costs, q)) for q in (0.25, 0.5, 0.75, 1.0)})
        log.warning("budgets.site_J not set: using quantiles of MODELLED cost %s (pipeline test only)", budgets)
    rows: list[dict[str, Any]] = []
    for J in budgets:
        bud = uniform_budgets(n_sites, float(J), b.get("orchestrator_J"))
        sel = select_configuration(profiles, bud)
        singles = [p for p in profiles if p.config.n_primitives() <= 1]
        best_single = select_configuration(singles, bud)
        row: dict[str, Any] = {"experiment": "E3", "budget_site_J": float(J), "remark1_holds": check_remark1(profiles, bud),
                               "node_class": node_class or "per-manifest"}
        if sel == INFEASIBLE or not isinstance(sel, Profile):
            row.update({"selected": INFEASIBLE})
            rows.append(row)
            continue
        row.update({"selected": sel.config.label(), "selected_adv_selection": sel.adv_max})
        if isinstance(best_single, Profile):
            row.update({"best_single": best_single.config.label(), "best_single_adv_selection": best_single.adv_max})
        if not final_eval:
            rows.append(row)
            continue
        tests = [profile_on_test(ctx, sel.config, node_class)]
        if isinstance(best_single, Profile):
            tests.append(profile_on_test(ctx, best_single.config, node_class))
        merge_measured_energy(tests, energy_cfg.get("measured_energy_csv"), strict=ctx.data_kind == "real")
        add_co2e(tests, ctx.cfg)
        row.update({f"final_{k}": v for k, v in tests[0].items() if k.startswith(FINAL_KEYS)})
        if len(tests) > 1:
            row.update({f"best_single_final_{k}": v for k, v in tests[1].items()
                        if k.startswith(FINAL_KEYS)})
        rows.append(row)
    if not final_eval:
        return rows, prof_rows
    baselines = [_defence_on_test(ctx, FixedPaddingDefence(percentile=float(ctx.cfg.get("baselines", {}).get(
        "fixed_padding_percentile", 99.0))), node_class)]
    ns_root = ctx.cfg.get("baselines", {}).get("netshaper_root")
    if ns_root:
        baselines.append(_defence_on_test(ctx, NetShaperCaptureAdapter(ns_root), node_class))
    else:
        log.warning("NetShaper baseline skipped: baselines.netshaper_root not set (TODO N-1)")
    for r in baselines:
        r["experiment"] = "E3-baseline"
    merge_measured_energy(baselines, energy_cfg.get("measured_energy_csv"), strict=ctx.data_kind == "real")
    add_co2e(baselines, ctx.cfg)
    return rows + baselines, prof_rows


def final_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten only final-test results; selection scores cannot enter these plots."""
    out = []
    for r in rows:
        common = {k: r[k] for k in ('N', 'node_class', 'budget_site_J') if k in r}
        if r.get('experiment') == 'E3-baseline':
            out.append(r | {'role': 'baseline', 'experiment': 'E3-final'})
            continue
        for role, prefix, config_key in [('selected', 'final_', 'selected'),
                                          ('best_single', 'best_single_final_', 'best_single')]:
            if prefix + 'adv_max' not in r:
                continue
            metrics = {k.removeprefix(prefix): v for k, v in r.items() if k.startswith(prefix)}
            out.append(common | metrics | {'config': r[config_key], 'role': role, 'experiment': 'E3-final'})
    return out
