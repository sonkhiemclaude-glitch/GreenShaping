"""E2: cost profile and privacy gain per joule of each primitive alone (Sec. 7.2).

Energy columns: `energy_modelled_*` come from Eq. (3) with the configured cost model
(placeholder coefficients on synthetic data); `energy_measured_*` come only from a
testbed measurement CSV (`energy.measured_energy_csv`). Privacy gain per joule uses
final-test advantages and MEASURED energy when available, otherwise it is reported as
`privacy_gain_per_joule_modelled` (audit A-2).
"""
from __future__ import annotations

import math
from typing import Any

from ..analysis.bounds import arrival_variant_padding_bound_per_tick, expected_dp_padding_bytes
from ..attack.evaluation import max_advantage, n_secrets_scored
from ..controller.grid import build_grid
from ..metrics.metrics import privacy_gain_per_joule
from ..shaping.pipeline import ShapingConfig, resolve_delta
from .common import (
    RunContext,
    add_co2e,
    cost_summary,
    dp_padding_check,
    final_test_evaluation,
    merge_measured_energy,
    node_models,
    row_identity,
)


def attack_columns(res: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """adv_<secret> = raw normalised advantage with its CI; adv_max = max over secrets of the
    advantage clipped at 0 (Algorithm 1 line 9). One convention for every table (minor 9)."""
    row: dict[str, Any] = {}
    for secret, m in res.items():
        row[f"adv_{secret}"] = m["adv_raw"]
        row[f"adv_{secret}_ci_low"], row[f"adv_{secret}_ci_high"] = m["adv_ci_low"], m["adv_ci_high"]
        row[f"attacker_{secret}"] = m.get("attacker", "rf")
        for k in ("acc", "auc", "balanced_accuracy", "chance", "chance_global", "chance_mode", "train_majority_accuracy"):
            row[f"{k}_{secret}"] = m.get(k)
        for model, details in m.get("per_model", {}).items():
            for k in ("acc", "auc", "adv_raw", "balanced_accuracy"):
                row[f"{k}_{secret}_{model}"] = details.get(k)
    row["adv_max"] = max_advantage(res)
    row["n_secrets_scored"] = n_secrets_scored(res)
    return row


def profile_on_test(ctx: RunContext, cfg: ShapingConfig, node_class: str | None = None) -> dict[str, Any]:
    ev = final_test_evaluation(ctx, cfg)
    cfg_r: ShapingConfig = ev["config"]
    row: dict[str, Any] = {"config": cfg_r.label(), "n_primitives": cfg_r.n_primitives(),
                           "node_class": node_class or "per-manifest"}
    row.update(row_identity(ctx, cfg_r, node_class))
    row.update(attack_columns(ev["attack"]))
    row.update(cost_summary(ctx, ctx.test, ev["shaped_test"], ev["stats_test"], node_models(ctx, node_class)))
    if cfg_r.dp is not None:
        row.update(dp_padding_check(ctx.test, cfg_r, ev["shaped_test"], ctx.seed))
        d = cfg_r.dp
        assert d.delta is not None
        row["dp_delta_bytes"] = d.delta
        row["dp_volume_cap_bytes"] = d.volume_cap_bytes
        row["dp_sensitivity_mode"] = d.sensitivity_mode
        row["dp_session_epsilon_bound"] = 2 * ctx.n_values[0] * d.eps_dir
        if d.mode == "queue":
            row["dp_padding_eq4_bytes"] = expected_dp_padding_bytes(d.horizon, d.interval, d.delta, d.eps_dir)
        else:
            K = math.ceil(d.horizon / d.interval - 1e-12)
            row["dp_padding_remark2_upper_bytes"] = K * arrival_variant_padding_bound_per_tick(d.bias, d.delta, d.eps_dir)
    return row


def run_e2(ctx: RunContext, node_class: str | None = None) -> list[dict[str, Any]]:
    if len(ctx.n_values) > 1:
        return [r for n in ctx.n_values for r in run_e2(ctx.for_n(n), node_class)]
    grid = [g for g in build_grid(ctx.cfg.get("grid", {})) if g.n_primitives() <= 1]
    rows = [profile_on_test(ctx, g, node_class) for g in grid]
    base = next(r for r in rows if r["n_primitives"] == 0)
    merge_measured_energy(rows, ctx.cfg.get("energy", {}).get("measured_energy_csv"), strict=ctx.data_kind == "real")
    add_co2e(rows, ctx.cfg)
    for r in rows:
        r["experiment"] = "E2"
        r["privacy_gain_per_joule_measured"] = privacy_gain_per_joule(
            base["adv_max"], r["adv_max"], r.get("energy_measured_site_max_median_J", float("nan")))
        r["privacy_gain_per_joule_modelled"] = privacy_gain_per_joule(
            base["adv_max"], r["adv_max"], r["energy_modelled_site_max_J"])
    return rows


__all__ = ["run_e2", "profile_on_test", "attack_columns", "resolve_delta"]
