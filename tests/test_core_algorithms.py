"""Core algorithms on controlled examples: Eq. (1), rounds (Prop. 2), Eq. (2) (Prop. 3), Prop. 4, Eq. (4)."""
import math

import numpy as np
import pytest

from greenshaping.analysis.bounds import (
    arrival_variant_padding_bound_per_tick,
    expected_dp_padding_bytes,
    links_identical,
    padded_l1_bound,
    padding_class_bound,
)
from greenshaping.data.schemas import DOWN, UP
from greenshaping.data.synthetic import SyntheticConfig, generate_sessions
from greenshaping.shaping.dp_release import dp_release, laplace_scale, tick_times
from greenshaping.shaping.padding import pad_fixed, pad_session, pad_size
from greenshaping.shaping.pipeline import (
    DPParams,
    PaddingParams,
    RoundsParams,
    ShapingConfig,
    apply_shaping,
    calibrate_delta,
    resolve_delta,
)
from greenshaping.shaping.rounds import synchronised_rounds


# ---------------------------------------------------------------- Eq. (1) / Prop. 1
@pytest.mark.parametrize("r", [1.25, 1.5, 2.0, 4.0])
def test_pad_size_prop1(r):
    base = 512
    for m in list(range(0, 5000, 7)) + [10**5, 10**6 + 3]:
        p = pad_size(m, r, base)
        assert p >= m and p >= base
        if m > base:
            assert p < r * m + 1
    lo, hi = 600, 2_000_000
    classes = {pad_size(m, r, base) for m in range(lo, hi, 997)} | {pad_size(hi, r, base)}
    assert len(classes) <= padding_class_bound(lo, hi, r, base)


def test_pad_examples_from_paper():
    # 1 KB to 1 MB: at most 11 classes (r=2) and 6 classes (r=4) when beta < 1 KB
    assert padding_class_bound(1024, 1024**2, 2.0, 512) == 11
    assert padding_class_bound(1024, 1024**2, 4.0, 512) == 6


def test_pad_invalid():
    with pytest.raises(ValueError):
        pad_size(10, 1.0, 512)
    with pytest.raises(ValueError):
        pad_size(-1, 2.0, 512)


def test_padding_keeps_counts_and_times(session_factory):
    s = session_factory([700, 0], [3000, 0])
    sp, _ = pad_session(s, 2.0, 512)
    assert [(m.t, m.direction) for m in sp.messages] == [(m.t, m.direction) for m in s.messages]


def test_fixed_padding_splits(session_factory):
    s = session_factory([700, 0], [5000, 0])
    sp, _ = pad_fixed(s, 2048)
    assert all(m.size == 2048 for m in sp.messages)
    assert len(sp.link_direction(0, DOWN)) == 3


# ---------------------------------------------------------------- Prop. 4
def test_padded_l1_bound_and_counterexample():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(1, 12))
        sizes = rng.integers(1, 3000, size=n)
        for r in (1.5, 2.0, 4.0):
            assert sum(pad_size(int(m), r, 512) for m in sizes) <= padded_l1_bound(n, int(sizes.sum()), r, 512)
    assert sum(pad_size(100, 2.0, 512) for _ in range(10)) == 5120 <= padded_l1_bound(10, 1000, 2.0, 512)


def test_calibrated_delta_covers_bytes_entering_release():
    sess = generate_sessions(SyntheticConfig(n_sessions=40, n_sites=4, n_questions=12, seed=3))
    cfg = ShapingConfig(padding=PaddingParams(2.0, 512), dp=DPParams(delta=None))
    delta = calibrate_delta(sess, cfg, 100.0)
    for s in sess:
        sp, _ = pad_session(s, 2.0, 512)
        for i in range(s.n_sites):
            for d in (UP, DOWN):
                assert sum(m.size for m in sp.link_direction(i, d)) <= delta


def test_apply_requires_delta(session_factory, rng):
    with pytest.raises(ValueError):
        apply_shaping(session_factory([500, 0], [500, 0]), ShapingConfig(dp=DPParams(delta=None)), rng)


# ---------------------------------------------------------------- rounds / Prop. 2
def test_rounds_identical_links_at_q1():
    for s in generate_sessions(SyntheticConfig(n_sessions=60, n_sites=6, n_questions=16, seed=5)):
        sh, _ = synchronised_rounds(s, 0.5, 1.0, 16384, rng=np.random.default_rng(0))
        assert links_identical(sh)


@pytest.mark.parametrize("rounds", [1, 3])
def test_session_level_cover_floor(session_factory, rounds):
    """Any-traffic detector accuracy = 1 - q/2 for prior 1/2, whatever the number of rounds (Prop. 2b)."""
    rng = np.random.default_rng(3)
    for q in (0.0, 0.5, 1.0):
        correct, n = 0, 1200
        for _ in range(n):
            part = int(rng.random() < 0.5)
            s = session_factory([600, 600 if part else 0], [900, 900 if part else 0], rounds=rounds)
            sh, _ = synchronised_rounds(s, 0.5, q, 4096, rng=rng)
            correct += int(len(sh.link(1)) > 0) == part
        assert abs(correct / n - (1 - q / 2)) < 0.045


def test_rounds_reply_frames_cover_large_reply(session_factory, rng):
    s = session_factory([600, 600], [40000, 100])
    sh, st = synchronised_rounds(s, 0.5, 1.0, 16384, rng=rng)
    assert len(sh.link_direction(0, DOWN)) == 3 == len(sh.link_direction(1, DOWN))   # ceil(40000/16384)
    assert st.real_done >= max(m.t for m in sh.messages if m.direction == DOWN)


# ---------------------------------------------------------------- Eq. (2) / Prop. 3
def _laplace_cdf(x, s):
    return 0.5 * math.exp(x / s) if x < 0 else 1 - 0.5 * math.exp(-x / s)


def exact_release_pmf(Q, s, vmax):
    def mass(lo, hi):
        if lo >= 0:
            return 0.5 * (math.exp(-lo / s) - math.exp(-hi / s))
        if hi <= 0:
            return 0.5 * (math.exp(hi / s) - math.exp(lo / s))
        return _laplace_cdf(hi, s) - _laplace_cdf(lo, s)
    p = [_laplace_cdf(0.5 - Q, s)] + [mass(v - 0.5 - Q, v + 0.5 - Q) for v in range(1, vmax)]
    p.append(1 - sum(p))
    return np.array(p)


def _single_tick_samples(session_factory, Q, eps_dir, sens, n, seed):
    rng = np.random.default_rng(seed)
    s = session_factory([Q, 0], [1, 0])
    out = []
    for _ in range(n):
        sh, _ = dp_release(s, interval=1.0, horizon=1.0, eps_dir=eps_dir, sensitivity=sens, rng=rng)
        b = [m.size for m in sh.link_direction(0, UP)]
        out.append(b[0] if b else 0)
    return np.array(out)


def test_dp_release_matches_exact_pmf_and_catches_half_noise(session_factory):
    Q, sens, eps_dir, vmax = 500, 400, 1.0, 4000
    edges = np.array([0, 1, 200, 400, 500, 600, 800, 1200, 2000, vmax + 1])
    p = exact_release_pmf(Q, sens / eps_dir, vmax)
    exact = np.array([p[a:b].sum() for a, b in zip(edges[:-1], edges[1:], strict=False)])
    emp = np.histogram(np.minimum(_single_tick_samples(session_factory, Q, eps_dir, sens, 20000, 1), vmax), edges)[0] / 20000
    assert 0.5 * np.abs(emp - exact).sum() < 0.02
    mut = np.histogram(np.minimum(_single_tick_samples(session_factory, Q, 2 * eps_dir, sens, 20000, 2), vmax), edges)[0] / 20000
    assert 0.5 * np.abs(mut - exact).sum() > 0.05


def test_dp_release_exact_likelihood_ratio():
    sens, eps = 400, 1.0
    for Q in (0, 100, 500, 2000):
        p = exact_release_pmf(Q, sens / eps, 6000)[:-1]
        q = exact_release_pmf(Q + sens, sens / eps, 6000)[:-1]
        mask = (p > 1e-15) & (q > 1e-15)
        assert np.max(np.abs(np.log(p[mask] / q[mask]))) <= eps + 1e-9


def test_dp_schedule_data_independent_and_all_sites_awake(session_factory):
    a = session_factory([1000, 0], [3000, 0])
    b = session_factory([1000, 900], [3000, 50000])
    sa, _ = dp_release(a, 0.25, 5.0, 4.0, 4096, rng=np.random.default_rng(1))
    sb, _ = dp_release(b, 0.25, 5.0, 4.0, 4096, rng=np.random.default_rng(2))
    ticks = {round(t, 6) for t in tick_times(0.25, 5.0)}
    assert {round(m.t, 6) for m in sa.messages} <= ticks and {round(m.t, 6) for m in sb.messages} <= ticks
    assert sa.wake[1] == sb.wake[1] and len(sa.wake[1]) == 20


def test_dp_overflow_and_above_delta_flags(session_factory):
    s = session_factory([5000, 0], [5000, 0])
    _, st = dp_release(s, 1.0, 1.0, 1e9, 100, rng=np.random.default_rng(0))   # tiny Delta, negligible noise
    assert st.above_delta and st.privacy_failure


def test_eq4_expected_padding_monte_carlo(session_factory):
    """Mean padding of the queue release on an idle link equals K^2 Delta / (2 eps_dir) (Eq. 4)."""
    H, tau, delta, eps = 10.0, 1.0, 1000, 5.0
    s = session_factory([300, 0], [300, 0])
    rng = np.random.default_rng(11)
    pads = []
    for _ in range(1000):
        sh, _ = dp_release(s, tau, H, eps, delta, rng=rng)
        pads.append(sum(m.size for m in sh.link_direction(1, UP)))      # site 1 idle
    expected = expected_dp_padding_bytes(H, tau, delta, eps)
    assert abs(np.mean(pads) - expected) / expected < 0.05


def test_arrival_variant_padding_within_remark2_bound(session_factory):
    H, tau, delta, eps, mu = 10.0, 1.0, 1000, 2.0, 50.0
    s = session_factory([300, 0], [300, 0])
    rng = np.random.default_rng(5)
    pads = [sum(m.size for m in dp_release(s, tau, H, eps, delta, "arrivals", mu, rng=rng)[0].link_direction(1, UP))
            for _ in range(400)]
    assert np.mean(pads) <= 10 * arrival_variant_padding_bound_per_tick(mu, delta, eps) * 1.05


def test_laplace_scale_modes():
    assert laplace_scale(100, 2.0, 10, "queue") == 500.0
    assert laplace_scale(100, 2.0, 10, "arrivals") == 50.0
    with pytest.raises(ValueError):
        laplace_scale(100, 0.0, 10, "queue")


def test_pipeline_order_rounds_then_padding_then_dp(rng):
    sess = generate_sessions(SyntheticConfig(n_sessions=20, n_sites=3, n_questions=8, seed=2))
    cfg = resolve_delta(ShapingConfig(padding=PaddingParams(2.0, 512), rounds=RoundsParams(0.5, 1.0, 4096),
                                      dp=DPParams(interval=1.0, horizon=60.0)), sess)
    assert cfg.dp is not None and cfg.dp.delta is not None
    for s in sess:
        sh, st = apply_shaping(s, cfg, rng)
        assert not st.above_delta
        assert set(sh.wake) == set(range(s.n_sites))
