import numpy as np
import pytest

from greenshaping.attack.evaluation import assert_disjoint, evaluate_attacker, max_advantage, split_by_question
from greenshaping.attack.features import LINK_FEATURE_DIM, link_features, session_features, time_slice_matrix
from greenshaping.controller.grid import build_grid, reply_frame_percentiles, single_primitive_subset
from greenshaping.controller.selection import (
    INFEASIBLE,
    Profile,
    check_remark1,
    profile_configuration,
    select_configuration,
    uniform_budgets,
)
from greenshaping.data.synthetic import SyntheticConfig, generate_sessions
from greenshaping.energy.cost_model import CostModel
from greenshaping.shaping.pipeline import ShapingConfig


@pytest.fixture(scope="module")
def sessions():
    return generate_sessions(SyntheticConfig(n_sessions=120, n_sites=4, n_questions=24, seed=7))


def test_features_shapes(sessions):
    s = sessions[0]
    assert len(link_features(s, 0)) == LINK_FEATURE_DIM
    assert len(session_features(s)) == 4 + 2 * LINK_FEATURE_DIM
    m = time_slice_matrix(s, 0, 0.25, 40)
    assert m.shape == (4, 40) and (m >= 0).all()


def test_three_way_split_disjoint(sessions):
    a, b, c = split_by_question(sessions, (0.4, 0.3, 0.3), seed=1)
    assert_disjoint(a, b, c)
    assert len(a) + len(b) + len(c) == len(sessions)
    with pytest.raises(AssertionError):
        assert_disjoint(a, a)


def test_attacker_learns_unshaped_participation(sessions):
    a, b, c = split_by_question(sessions, seed=0)
    res = evaluate_attacker(a + b, c, seed=0, n_estimators=20, n_boot=30)
    assert res["participation"]["acc"] > 0.9            # who is called is visible without shaping
    assert 0.0 <= res["participation"]["auc"] <= 1.0
    assert not np.isnan(res["participation"]["adv_ci_low"])
    assert max_advantage(res) <= 1.0


def test_grid_contains_single_primitives():
    spec = {"padding": {"ratio": [2.0], "base": [512]}, "rounds": {"window": [0.5], "cover_prob": [1.0]},
            "dp": {"eps_dir": [1.0, 4.0]}}
    g = build_grid(spec)
    assert len(g) == 2 * 2 * 3
    singles = single_primitive_subset(g)
    assert len(singles) == 1 + 1 + 1 + 2
    assert reply_frame_percentiles([100, 200, 300, 10000]) == sorted(set(reply_frame_percentiles([100, 200, 300, 10000])))


def _p(name, adv, e, overflow=False, n=1):
    from greenshaping.shaping.pipeline import PaddingParams
    cfg = ShapingConfig(padding=PaddingParams(2.0 + n, 512), name=name) if n == 1 else ShapingConfig(name=name)
    p = Profile(cfg, overflow, 0.0, {"site0": e, "orchestrator": e})
    p.adv_max = adv
    return p


def test_algorithm1_selection_rules():
    profs = [_p("a", 0.5, 1.0), _p("b", 0.2, 5.0), _p("c", 0.2, 3.0), _p("d", 0.0, 2.0, overflow=True)]
    assert select_configuration(profs, {"site0": 4.0}).config.name == "c"        # lexicographic (adv, energy)
    assert select_configuration(profs, {"site0": 2.0}).config.name == "a"        # budget
    assert select_configuration(profs, {"site0": 0.5}) == INFEASIBLE
    assert check_remark1(profs, {"site0": 4.0})


def test_profile_configuration_end_to_end(sessions):
    a, b, _ = split_by_question(sessions, seed=0)
    model = CostModel("t", 1e-8, 1e-8, 1.0, 0.1)
    p = profile_configuration(ShapingConfig(), a, b, model, seed=0, n_estimators=10)
    assert p.extra_energy["orchestrator"] == pytest.approx(0.0) and p.adv_max > 0.5
    assert uniform_budgets(2, 1.0, 3.0) == {"site0": 1.0, "site1": 1.0, "orchestrator": 3.0}
