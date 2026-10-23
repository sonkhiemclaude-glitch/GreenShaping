from greenshaping.experiments.common import context_from_config
from greenshaping.experiments.e1_leakage import run_e1
from greenshaping.experiments.e2_primitives import run_e2
from greenshaping.utils.random import child_seed


def _strip(rows):
    """NaN-aware comparison key (json renders NaN identically)."""
    import json
    return json.dumps([{k: v for k, v in r.items() if k != "result_status"} for r in rows], sort_keys=True)


def test_same_seed_same_results(tmp_path, tiny_config):
    a = context_from_config(tiny_config, tmp_path / "a")
    b = context_from_config(tiny_config, tmp_path / "b")
    assert _strip(run_e1(a)) == _strip(run_e1(b))
    ra, rb = run_e2(a), run_e2(b)
    assert [r["adv_max"] for r in ra] == [r["adv_max"] for r in rb]
    assert [r["bandwidth_overhead"] for r in ra] == [r["bandwidth_overhead"] for r in rb]


def test_child_seed_is_stable():
    assert child_seed(0, "x", 1) == child_seed(0, "x", 1)
    assert child_seed(0, "x", 1) != child_seed(0, "y", 1)
    assert isinstance(child_seed(3, "abc"), int)
