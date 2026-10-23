"""Closed-form quantities from Section 5 (Props. 1-4, Eq. 4, Remark 2).

These are analytical helpers used for reporting and for checking measured values
against the paper's statements. They are not results.
"""
from __future__ import annotations

import math

from ..data.schemas import Session


# ---------------------------------------------------------------- Proposition 1
def padding_class_bound(m_lo: float, m_hi: float, ratio: float, base: float) -> int:
    """Prop. 1(b): padded sizes of messages in [m_lo, m_hi] take <= ceil(log_r(m_hi/m_lo)) + 1 values."""
    if not base < m_lo <= m_hi:
        raise ValueError("Prop. 1(b) needs beta < m_lo <= m_hi")
    if ratio <= 1:
        raise ValueError("r must be > 1")
    return int(math.ceil(math.log(m_hi / m_lo, ratio) - 1e-12)) + 1


def padding_mi_bound_bits(m_lo: float, m_hi: float, ratio: float, base: float) -> float:
    """Prop. 1(c): I(X; pad(Z)) <= log2(class bound) bits per message."""
    return math.log2(padding_class_bound(m_lo, m_hi, ratio, base))


def padding_overhead_bound(m: int, ratio: float, base: int) -> float:
    """Prop. 1(a) with whole-byte rounding: pad(m) - m < (r-1)m + 1 for m > beta; beta - m otherwise."""
    return (ratio - 1) * m + 1 if m > base else float(base - m)


# ---------------------------------------------------------------- Proposition 2
def links_identical(sess: Session, active: set[int] | None = None) -> bool:
    """Prop. 2(a) check: every (active) link carries the same (t, d, s) sequence."""
    sites = sorted(active) if active is not None else list(range(sess.n_sites))
    seqs = {tuple((round(m.t, 9), m.direction, m.size) for m in sess.link(i)) for i in sites}
    return len(seqs) <= 1


def required_cover_probability(alpha: float, prior: float) -> float:
    """Prop. 2(b): q_i >= (1 - alpha) / (1 - pi_i) for detector accuracy <= alpha."""
    if not 0 < prior < 1:
        raise ValueError("prior pi_i must be in (0, 1)")
    if not max(prior, 1 - prior) <= alpha <= 1:
        raise ValueError("alpha must lie in [max(pi, 1-pi), 1]")
    return (1 - alpha) / (1 - prior)


def any_traffic_detector_accuracy(prior: float, cover_prob: float) -> float:
    """Prop. 2(b) proof: accuracy = pi + (1 - pi)(1 - q) = 1 - (1 - pi) q."""
    if not 0 <= prior <= 1 or not 0 <= cover_prob <= 1:
        raise ValueError("prior and cover_prob must be probabilities")
    return 1 - (1 - prior) * cover_prob


def cover_energy_floor(alpha: float, e_min: float) -> float:
    """Prop. 2(b): expected extra energy at a site >= (1 - alpha) e_min,i (J per session)."""
    if not math.isfinite(e_min) or e_min <= 0:
        raise ValueError("e_min must be > 0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return (1 - alpha) * e_min


def network_cover_floor(alpha: float, e_min_per_site: list[float]) -> float:
    """Network-wide floor (1 - alpha) * sum_i e_min,i, independent of the number of participants."""
    return sum(cover_energy_floor(alpha, e) for e in e_min_per_site)


# ---------------------------------------------------------------- Proposition 3 / Eq. (4)
def ticks_per_session(horizon: float, interval: float) -> int:
    if horizon <= 0 or interval <= 0:
        raise ValueError("H and tau must be > 0")
    return int(math.ceil(horizon / interval - 1e-12))


def dp_epsilon_total(eps_dir: float, d_link_directions: int) -> float:
    """Prop. 3: eps = d * K * eps_w = d * eps_dir (basic sequential composition)."""
    if eps_dir <= 0 or d_link_directions < 1:
        raise ValueError("eps_dir > 0 and d >= 1 are required")
    return d_link_directions * eps_dir


def dp_accuracy_bound(eps: float, prior: float = 0.5) -> float:
    """Prop. 3: any attacker's accuracy <= p e^eps / (p e^eps + 1 - p), p = max(prior, 1 - prior)."""
    if eps < 0:
        raise ValueError("eps must be >= 0")
    p = max(prior, 1 - prior)
    if not 0 < p < 1:
        return 1.0
    x = p * math.exp(min(eps, 700.0))
    return x / (x + 1 - p)


def expected_dp_padding_bytes(horizon: float, interval: float, delta: float, eps_dir: float) -> float:
    """Eq. (4): expected padding per link-direction and session = K^2 Delta / (2 eps_dir), K = ceil(H/tau).

    The paper's last form H^2 Delta / (2 tau^2 eps_dir) equals this when H/tau is an
    integer (audit P-2). Exact before rounding; rounding changes each tick by <= 0.5 B.
    """
    if delta <= 0 or eps_dir <= 0:
        raise ValueError("Delta and eps_dir must be > 0")
    K = ticks_per_session(horizon, interval)
    return K * K * delta / (2.0 * eps_dir)


def arrival_variant_padding_bound_per_tick(bias: float, delta: float, eps_dir: float) -> float:
    """Remark 2: expected padding per tick <= E[max(0, mu + L)] + 0.5, L ~ Laplace(Delta/eps_dir).

    E[max(0, mu + L)] for Laplace(0, s): mu + s/2 * exp(-mu/s) if mu >= 0.
    """
    if bias < 0:
        raise ValueError("Remark 2 uses mu >= 0")
    s = delta / eps_dir
    return bias + 0.5 * s * math.exp(-bias / s) + 0.5


# ---------------------------------------------------------------- Proposition 4
def padded_l1_bound(n_messages: int, delta0: float, ratio: float, base: int) -> float:
    """Prop. 4: n differing messages with total <= Delta_0 differ by <= r Delta_0 + n(beta + 1) after padding."""
    if n_messages < 0 or delta0 < 0:
        raise ValueError("n and Delta_0 must be >= 0")
    return ratio * delta0 + n_messages * (base + 1)
