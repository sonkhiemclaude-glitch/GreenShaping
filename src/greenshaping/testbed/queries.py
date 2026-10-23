"""The four query families of Sec. 6.1 executed on one site's table partition.

Each function returns (result, matching_rows). `matching_rows` defines the
contribution-size bucket (Sec. 6.1) and is written to the session manifest by the
driver; it is ground truth, never visible to the observer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _filtered(df: pd.DataFrame, where: dict[str, Any] | None) -> pd.DataFrame:
    if not where:
        return df
    mask = np.ones(len(df), bool)
    for col, val in where.items():
        if col not in df.columns:
            raise KeyError(f"unknown column {col!r}")
        if isinstance(val, dict):
            if set(val) != {'min', 'max'} or not np.isfinite([val['min'], val['max']]).all() or val['min'] > val['max']:
                raise ValueError('range filters require finite min <= max')
            mask &= df[col].between(val['min'], val['max'], inclusive='both').to_numpy()
        else:
            mask &= (df[col] == val).to_numpy()
    return df[mask]


def aggregate(df: pd.DataFrame, columns: list[str], where: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    d = _filtered(df, where)
    res = {c: {"count": int(d[c].count()), "sum": float(d[c].sum()), "sumsq": float((d[c] ** 2).sum())} for c in columns}
    return res, len(d)


def group_by(df: pd.DataFrame, by: str, column: str, where: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    d = _filtered(df, where)
    g = d.groupby(by)[column].agg(["count", "sum"])
    return {str(k): {"count": int(v["count"]), "sum": float(v["sum"])} for k, v in g.iterrows()}, len(d)


def train_round(df: pd.DataFrame, features: list[str], label: str, weights: list[float], lr: float = 0.1,
                where: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    """One local gradient step of logistic regression (a 'small training job' round)."""
    d = _filtered(df, where)
    X = np.c_[np.ones(len(d)), d[features].to_numpy(float)]
    y = d[label].to_numpy(float)
    w = np.asarray(weights, float)
    if w.shape[0] != X.shape[1]:
        raise ValueError(f"expected {X.shape[1]} weights (bias + features), got {w.shape[0]}")
    if len(d) == 0:
        return {"weights": w.tolist(), "n": 0}, 0
    p = 1 / (1 + np.exp(-X @ w))
    grad = X.T @ (p - y) / len(d)
    return {"weights": (w - lr * grad).tolist(), "n": int(len(d))}, len(d)


def record_count(df: pd.DataFrame, where: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    d = _filtered(df, where)
    return {"count": int(len(d))}, len(d)


def run_query(df: pd.DataFrame, family: str, params: dict[str, Any]) -> tuple[dict[str, Any], int]:
    # Public synthetic-cohort window: vary contribution size within every site.
    # Apply the filter first, then the deterministic window, exactly once.
    if "limit_rows" in params or "row_offset" in params:
        limit, offset = int(params.get("limit_rows", len(df))), int(params.get("row_offset", 0))
        if limit < 1 or offset < 0:
            raise ValueError("limit_rows must be positive and row_offset nonnegative")
        df = _filtered(df, params.get("where")).iloc[offset:offset + limit]
        params = {k: v for k, v in params.items() if k != "where"}
    if family == "aggregate":
        return aggregate(df, params["columns"], params.get("where"))
    if family == "group_by":
        return group_by(df, params["by"], params["column"], params.get("where"))
    if family == "train_round":
        return train_round(df, params["features"], params["label"], params["weights"], params.get("lr", 0.1),
                           params.get("where"))
    if family == "record_count":
        return record_count(df, params.get("where"))
    raise ValueError(f"unknown query family {family!r}")
