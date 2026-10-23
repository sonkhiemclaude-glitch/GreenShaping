import numpy as np
import pytest

from greenshaping.baselines import (
    DummyDefence,
    FixedPaddingDefence,
    NetShaperCaptureAdapter,
    NoShaping,
    dp_only,
    padding_only,
    rounds_only,
)
from greenshaping.data.synthetic import SyntheticConfig, generate_sessions


@pytest.fixture(scope="module")
def sessions():
    return generate_sessions(SyntheticConfig(n_sessions=30, n_sites=3, n_questions=8, seed=1))


def test_kinds_are_separated():
    assert NoShaping().kind == "real" and FixedPaddingDefence().kind == "real"
    assert DummyDefence().kind == "dummy" and NetShaperCaptureAdapter(None).kind == "adapter"


def test_fixed_padding_uses_99th_percentile(sessions, rng):
    d = FixedPaddingDefence()
    d.prepare(sessions)
    sh, _ = d.apply(sessions[0], rng)
    assert all(m.size == d.unit for m in sh.messages)


def test_single_primitive_defences(sessions, rng):
    for d in (padding_only(2.0), rounds_only(0.5, 1.0, 16384), dp_only(4.0, 1.0, horizon=60.0)):
        d.prepare(sessions)
        sh, st = d.apply(sessions[0], rng)
        assert sh.n_sites == sessions[0].n_sites


def test_netshaper_adapter_refuses_without_captures(sessions):
    with pytest.raises(NotImplementedError):
        NetShaperCaptureAdapter(None).prepare(sessions)


def test_dummy_is_deterministic(sessions):
    a, _ = DummyDefence(10).apply(sessions[0], np.random.default_rng(0))
    b, _ = DummyDefence(10).apply(sessions[0], np.random.default_rng(1))
    assert a.messages == b.messages
