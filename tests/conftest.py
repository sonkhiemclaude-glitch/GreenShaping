from __future__ import annotations

import numpy as np
import pytest

from greenshaping.data.schemas import DOWN, UP, Message, Session


def make_session(sizes_up, sizes_dn, n_sites=2, t=0.3, rounds=1, sid=0, qid=0):
    """Hand-built session: site i called iff sizes_up[i] > 0; one call/reply per round."""
    msgs = []
    for r in range(rounds):
        for s, (u, d) in enumerate(zip(sizes_up, sizes_dn, strict=False)):
            if u:
                msgs.append(Message(t + 2.0 * r, s, UP, u))
                msgs.append(Message(t + 2.0 * r + 0.05 + 0.01 * s, s, DOWN, d))
    part = [int(bool(u)) for u in sizes_up]
    buckets = [0 if p else -1 for p in part]
    return Session(sid, qid, n_sites, 0, part, buckets, msgs, 2.0 * rounds + 1)


@pytest.fixture
def session_factory():
    return make_session


@pytest.fixture
def rng():
    return np.random.default_rng(0)


TINY_CONFIG = {
    "run": {"seed": 0, "data_kind": "synthetic"},
    "data": {"source": "synthetic", "synthetic": {"n_sessions": 60, "n_sites": [3, 4], "n_questions": 24, "seed": 1}},
    "split": {"fractions": [0.4, 0.3, 0.3], "seed": 0},
    "attack": {"n_estimators": 10, "n_boot": 20},
    "energy": {"cost_models": {}},
    "grid": {"padding": {"ratio": [2.0], "base": [512]},
             "rounds": {"window": [0.5], "cover_prob": [1.0], "reply_frame": [16384]},
             "dp": {"mode": ["queue"], "interval": [1.0], "eps_dir": [4.0], "delta_percentile": [100.0], "horizon": [30.0]}},
    "budgets": {"site_J": [], "orchestrator_J": None},
    "baselines": {"fixed_padding_percentile": 99.0, "netshaper_root": None},
    "experiments": {"e5": {"alpha": 0.6, "query_mixes": None}},
}


@pytest.fixture
def tiny_config():
    import copy
    return copy.deepcopy(TINY_CONFIG)
