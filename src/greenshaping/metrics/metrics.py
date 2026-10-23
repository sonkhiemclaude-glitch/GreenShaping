"""Metric formulas used in Sections 6 and 7. All handle empty inputs explicitly."""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        return float("nan")
    return float((yt == yp).mean())


def majority_class(y_train: ArrayLike) -> int:
    vals, cnt = np.unique(np.asarray(y_train), return_counts=True)
    if vals.size == 0:
        raise ValueError("empty training labels")
    return int(vals[np.argmax(cnt)])   # ties -> smallest label (np.unique sorts)


def chance_level(y_test: ArrayLike) -> float:
    """Majority-class chance: the largest class share in the evaluated test labels, i.e. the
    accuracy of the best constant predictor (audit A-1, review M2). A constant predictor
    therefore never scores a positive advantage. No test information enters the model."""
    yt = np.asarray(y_test)
    if yt.size == 0:
        return float("nan")
    _, cnt = np.unique(yt, return_counts=True)
    return float(cnt.max() / yt.size)


def normalised_advantage(acc: float, chance: float) -> float:
    """Sec. 6.2: (acc - chance) / (1 - chance). Returns 0 when chance == 1 (nothing to learn)."""
    if math.isnan(acc) or math.isnan(chance):
        return float("nan")
    if chance >= 1.0:
        return 0.0
    return (acc - chance) / (1.0 - chance)


def advantage_clipped(acc: float, chance: float) -> float:
    """Advantage clipped at 0 (used for the controller's max over secrets); raw value is also reported."""
    a = normalised_advantage(acc, chance)
    return a if math.isnan(a) else max(0.0, a)


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Rank-based ROC AUC (Mann-Whitney), ties averaged. NaN if only one class is present."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        log.warning("ROC AUC undefined with a single class; returning NaN")
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bandwidth_overhead(bytes_shaped: float, bytes_unshaped: float) -> float:
    """Relative extra bytes on the wire: shaped / unshaped - 1."""
    if bytes_unshaped <= 0:
        return float("nan")
    return bytes_shaped / bytes_unshaped - 1.0


def latency_summary(added_latency_s: Sequence[float]) -> dict[str, float]:
    """Added median and 95th-percentile latency (Sec. 7.2)."""
    x = np.asarray(added_latency_s, float)
    if x.size == 0:
        return {"latency_median_s": float("nan"), "latency_p95_s": float("nan")}
    return {"latency_median_s": float(np.median(x)), "latency_p95_s": float(np.percentile(x, 95))}


def privacy_gain_per_joule(adv_unshaped: float, adv_shaped: float, extra_joules: float) -> float:
    """Sec. 7.2: (A(empty) - A(theta)) / E_n(theta). NaN when E_n <= 0 (gain per joule undefined).

    Audit A-2: the caller chooses which advantage (final-test in the runners) is passed.
    """
    if extra_joules <= 0 or math.isnan(extra_joules):
        return float("nan")
    return (adv_unshaped - adv_shaped) / extra_joules


def median_iqr(values: Sequence[float], min_runs: int = 10) -> dict[str, float]:
    """Median and interquartile range over repeated runs (Sec. 6.4 asks for >= 10 runs)."""
    x = np.asarray(values, float)
    if x.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}
    if x.size < min_runs:
        log.warning("only %d runs; the protocol asks for at least %d", x.size, min_runs)
    return {"median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)), "n": int(x.size)}


def failure_rate(flags: Sequence[bool]) -> float:
    """Share of sessions outside the DP guarantee (overflow or above Delta)."""
    x = np.asarray(flags, bool)
    return float(x.mean()) if x.size else float("nan")


def pareto_front(rows: list[dict], keys: Sequence[str]) -> list[dict]:
    """Rows not dominated on all `keys` (all minimised)."""
    def dominated(r: dict) -> bool:
        return any(all(o[k] <= r[k] for k in keys) and any(o[k] < r[k] for k in keys) for o in rows)
    return [r for r in rows if not dominated(r)]
