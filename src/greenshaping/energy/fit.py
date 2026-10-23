"""Fit the coefficients of Eq. (3) per node class from microbenchmarks (Sec. 4.4, 6.4).

Each benchmark row gives, for one measured run on one node: bytes sent, bytes
received, the event times (sends, receives, wake-ups) and the MEASURED extra
energy above idle in joules (RAPL or meter, idle-subtracted). For each candidate
t_tail on a grid we solve non-negative least squares for (e_tx, e_rx, p_aw) and
keep the t_tail with the smallest residual. The paper requires residuals to be
reported; `FitResult` carries them. The fitting procedure itself is our choice
(audit M14).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .cost_model import CostModel, union_length


@dataclass(frozen=True)
class Benchmark:
    bytes_tx: int
    bytes_rx: int
    event_times: tuple[float, ...]
    measured_joules: float          # idle-subtracted
    run_id: str = ""

    def __post_init__(self) -> None:
        if self.bytes_tx < 0 or self.bytes_rx < 0 or not np.isfinite(self.measured_joules):
            raise ValueError("benchmark bytes must be nonnegative and energy finite")
        if any(not np.isfinite(t) or t < 0 for t in self.event_times):
            raise ValueError("event times must be finite and nonnegative")


@dataclass(frozen=True)
class FitResult:
    model: CostModel
    residuals: tuple[float, ...]
    rmse: float
    relative_rmse: float


def fit_cost_model(node_class: str, rows: list[Benchmark],
                   tail_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)) -> FitResult:
    if len(rows) < 4:
        raise ValueError("need at least 4 benchmark rows to fit 3 coefficients and t_tail")
    if not tail_grid or any(not np.isfinite(t) or t < 0 for t in tail_grid):
        raise ValueError("tail grid must be finite and nonnegative")
    y = np.array([r.measured_joules for r in rows], float)
    best: tuple[float, np.ndarray, float] | None = None
    for tail in tail_grid:
        X = np.array([[r.bytes_tx, r.bytes_rx, union_length(list(r.event_times), tail)] for r in rows], float)
        scale = X.max(axis=0)
        scale[scale == 0] = 1.0
        if np.linalg.matrix_rank(X / scale) < 3:
            continue
        coef, res = nnls(X / scale, y)
        coef = coef / scale
        if best is None or res < best[2]:
            best = (tail, coef, float(res))
    if best is None:
        raise ValueError("benchmark design cannot identify three coefficients; vary TX, RX and wakeups independently")
    tail, coef, _ = best
    X = np.array([[r.bytes_tx, r.bytes_rx, union_length(list(r.event_times), tail)] for r in rows], float)
    resid = y - X @ coef
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    denom = float(np.sqrt(np.mean(y ** 2))) or 1.0
    model = CostModel(node_class, float(coef[0]), float(coef[1]), float(coef[2]), float(tail), fitted=True)
    return FitResult(model, tuple(float(x) for x in resid), rmse, rmse / denom)


def validate_cost_model(rows: list[Benchmark], node_class: str, seed: int = 0) -> dict:
    """Hold out whole acquisition runs, then predict their measured energy."""
    groups = sorted({r.run_id for r in rows})
    if "" in groups or len(groups) < 10:
        raise ValueError("held-out validation needs at least 10 named independent acquisition runs")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    held = set(groups[:max(2, len(groups) // 5)])
    train, test = [r for r in rows if r.run_id not in held], [r for r in rows if r.run_id in held]
    fit = fit_cost_model(node_class, train)
    m = fit.model
    X = np.array([[r.bytes_tx, r.bytes_rx, union_length(list(r.event_times), m.t_tail)] for r in train], float)
    X /= np.maximum(np.max(X, axis=0), 1.)
    errors = [r.measured_joules - (m.e_tx * r.bytes_tx + m.e_rx * r.bytes_rx
                                  + m.p_aw * union_length(list(r.event_times), m.t_tail)) for r in test]
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denom = float(np.sqrt(np.mean([r.measured_joules ** 2 for r in test])))
    return {"training_runs": sorted(set(groups) - held), "validation_runs": sorted(held),
            "design_rank": int(np.linalg.matrix_rank(X)), "design_condition_number": float(np.linalg.cond(X)),
            "heldout_rmse_J": rmse, "heldout_relative_rmse": rmse / denom if denom else float("inf"),
            "heldout_residuals_J": errors, "independent_runs": len(groups)}
