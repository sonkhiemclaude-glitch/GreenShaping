"""Bootstrap confidence intervals that resample questions (Sec. 6.2)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def bootstrap_by_group(groups: ArrayLike, statistic: Callable[[np.ndarray], float],
                       n_boot: int = 1000, level: float = 0.95, seed: int = 0,
                       report: dict[str, int] | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI of `statistic(index_array)` resampling GROUPS (questions) with replacement.

    `statistic` receives the indices of the resampled rows. Returns (nan, nan) with fewer than 2 groups.
    Replicates where the statistic is NaN are dropped; their number is stored in
    `report["dropped"]` when a dict is passed (review minor 10).
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    if uniq.size < 2:
        return float("nan"), float("nan")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by_group[u] for u in pick])
        v = statistic(idx)
        if not np.isnan(v):
            vals.append(v)
    if report is not None:
        report["dropped"] = n_boot - len(vals)
    if not vals:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
