import math

import numpy as np
import pytest

from greenshaping.metrics.confidence_intervals import bootstrap_by_group
from greenshaping.metrics.metrics import (
    accuracy,
    advantage_clipped,
    bandwidth_overhead,
    chance_level,
    latency_summary,
    median_iqr,
    normalised_advantage,
    pareto_front,
    privacy_gain_per_joule,
    roc_auc,
)


def test_accuracy_and_chance():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    assert math.isnan(accuracy([], []))
    assert chance_level([0, 1, 1, 1]) == 0.75               # best constant predictor on the test labels
    # review M2: a constant predictor must not score a positive advantage
    y_test = [0, 0, 0, 1, 1]
    assert normalised_advantage(accuracy(y_test, [0] * 5), chance_level(y_test)) <= 0.0


def test_normalised_advantage():
    assert normalised_advantage(0.75, 0.5) == 0.5
    assert normalised_advantage(0.4, 0.5) == pytest.approx(-0.2)
    assert advantage_clipped(0.4, 0.5) == 0.0
    assert normalised_advantage(1.0, 1.0) == 0.0


def test_roc_auc_known_values_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert math.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_privacy_gain_per_joule_edge_cases():
    assert privacy_gain_per_joule(0.9, 0.3, 2.0) == pytest.approx(0.3)
    assert math.isnan(privacy_gain_per_joule(0.9, 0.3, 0.0))


def test_overheads_and_summaries():
    assert bandwidth_overhead(150, 100) == pytest.approx(0.5)
    assert math.isnan(bandwidth_overhead(1, 0))
    s = latency_summary([1, 2, 3, 4])
    assert s["latency_median_s"] == 2.5
    m = median_iqr(list(range(1, 12)))
    assert m["median"] == 6 and m["n"] == 11


def test_bootstrap_by_group_resamples_groups():
    groups = np.repeat(np.arange(10), 5)
    vals = np.repeat(np.linspace(0, 1, 10), 5)
    lo, hi = bootstrap_by_group(groups, lambda idx: float(vals[idx].mean()), n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert all(math.isnan(x) for x in bootstrap_by_group([1, 1, 1], lambda idx: 0.0))


def test_pareto_front():
    rows = [{"a": 1, "b": 3}, {"a": 2, "b": 2}, {"a": 3, "b": 3}, {"a": 1, "b": 4}]
    assert pareto_front(rows, ("a", "b")) == [{"a": 1, "b": 3}, {"a": 2, "b": 2}]
