"""Candidate grid Theta (Sec. 6.3). Built as the product of per-primitive options, so the
grid always contains every single-primitive configuration (Theta_1, Remark 1)."""
from __future__ import annotations

import itertools
from typing import Any

from ..shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig

PAPER_GRID: dict[str, Any] = {
    # Values of Sec. 6.3. Rounds reply frame R is given there as percentiles of reply sizes;
    # resolve them with `reply_frame_percentiles` before building the grid.
    "padding": {"ratio": [1.25, 1.5, 2.0, 4.0], "base": [512]},
    "rounds": {"window": [0.25, 0.5, 1.0], "cover_prob": [0.0, 0.5, 1.0], "reply_frame": [16384]},
    "dp": {"mode": ["queue", "arrivals"], "interval": [0.5, 1.0, 2.0], "eps_dir": [1.0, 2.0, 4.0, 8.0],
           "delta_percentile": [95.0, 100.0], "horizon": [20.0]},
}


def _expand(spec: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not spec:
        return []
    keys = list(spec)
    for k in keys:
        if not isinstance(spec[k], list) or not spec[k]:
            raise ValueError(f"grid entry {k!r} must be a non-empty list")
    return [dict(zip(keys, vals, strict=False)) for vals in itertools.product(*(spec[k] for k in keys))]


def build_grid(spec: dict[str, Any], include_none: bool = True) -> list[ShapingConfig]:
    """Product of {None} + padding options, {None} + rounds options, {None} + DP options."""
    pads: list[PaddingParams | None] = [None] + [PaddingParams(**d) for d in _expand(spec.get("padding"))]
    rounds: list[RoundsParams | None] = [None] + [RoundsParams(**d) for d in _expand(spec.get("rounds"))]
    dps: list[DPParams | None] = [None] + [DPParams(**d) for d in _expand(spec.get("dp"))]
    grid = [ShapingConfig(padding=p, rounds=r, dp=d) for p, r, d in itertools.product(pads, rounds, dps)]
    if not include_none:
        grid = [g for g in grid if g.n_primitives() > 0]
    return grid


def single_primitive_subset(grid: list[ShapingConfig]) -> list[ShapingConfig]:
    """Theta_1: configurations using at most one primitive."""
    return [g for g in grid if g.n_primitives() <= 1]


def reply_frame_percentiles(reply_sizes: list[int], percentiles: tuple[float, ...] = (50.0, 95.0)) -> list[int]:
    """Sec. 6.3: R at the median and 95th percentile of reply sizes."""
    import numpy as np

    if not reply_sizes:
        raise ValueError("no reply sizes")
    return sorted({max(1, int(np.ceil(np.percentile(reply_sizes, p)))) for p in percentiles})
