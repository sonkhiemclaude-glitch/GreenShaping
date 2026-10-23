"""Stage dispatcher used by the CLI and scripts/run_experiment.py."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .common import RunContext, context_from_config, write_outputs
from .e1_leakage import run_e1
from .e2_primitives import run_e2
from .e3_budgeted import final_comparison_rows, run_e3
from .e4_platform import run_e4
from .e5_scaling import run_e5

log = logging.getLogger(__name__)

STAGES = ("e1", "e2", "calibrate", "evaluate", "e3", "e4", "e5", "all")
ALIASES = {"leakage": "e1", "primitives": "e2", "ablation": "e2", "platform": "e4", "scaling": "e5"}


def _per_n(ctx: RunContext, fn: Any) -> Any:
    """Run fn(sub_ctx) for every deployment size N and tag rows with N (review B2)."""
    out: list[Any] = []
    for n in ctx.n_values:
        sub = ctx.for_n(n)
        if not (sub.train and sub.select and sub.test):
            log.warning("N=%d has no complete question split; skipped", n)
            continue
        res = fn(sub)
        out.append((n, res))
    if not out:
        raise ValueError("no deployment size N has sessions in all three splits")
    return out


def _tag(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return [{"N": n, **r} for r in rows]


def run_stage(ctx: RunContext, stage: str) -> list[Path]:
    stage = ALIASES.get(stage, stage)
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {STAGES} or aliases {sorted(ALIASES)}")
    written: list[Path] = []
    todo = ["e1", "e2", "e3", "e4", "e5"] if stage == "all" else [stage]
    for st in todo:
        if st == "e1":
            written.append(write_outputs(ctx, "e1_leakage", run_e1(ctx)))
        elif st == "e2":
            written.append(write_outputs(ctx, "e2_primitives", [r for n, rows in _per_n(ctx, run_e2) for r in _tag(rows, n)]))
        elif st in ("calibrate", "evaluate", "e3"):
            if st == "evaluate":
                written.append(write_outputs(ctx, "e1_leakage", run_e1(ctx)))
            final = st != "calibrate"
            res = _per_n(ctx, lambda c, final=final: run_e3(c, final_eval=final))
            sel = [r for n, (s_, _) in res for r in _tag(s_, n)]
            prof = [r for n, (_, p_) in res for r in _tag(p_, n)]
            written.append(write_outputs(ctx, "e3_profiles", prof))
            if st == "calibrate":
                written.append(write_outputs(ctx, "e3_selection", sel))
            else:
                written += [write_outputs(ctx, "e3_budgeted", [r for r in sel if r["experiment"] == "E3"]),
                            write_outputs(ctx, "e3_baselines", [r for r in sel if r["experiment"] == "E3-baseline"])]
                comparison = final_comparison_rows(sel)
                if comparison:
                    written.append(write_outputs(ctx, "e3_final_comparison", comparison))
        elif st == "e4":
            written.append(write_outputs(ctx, "e4_platform", [r for n, rows in _per_n(ctx, run_e4) for r in _tag(rows, n)]))
        elif st == "e5":
            written.append(write_outputs(ctx, "e5_scaling", run_e5(ctx)))
    return written


def run_from_config(cfg: dict[str, Any], stage: str, out_dir: str | None = None) -> list[Path]:
    ctx = context_from_config(cfg, out_dir)
    log.info("data: %s (%d sessions; %d/%d/%d train/select/test) -> %s", ctx.data_kind, len(ctx.sessions),
             len(ctx.train), len(ctx.select), len(ctx.test), ctx.out_dir)
    return run_stage(ctx, stage)
