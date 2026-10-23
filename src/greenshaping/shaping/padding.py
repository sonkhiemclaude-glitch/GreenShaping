"""Bucketed padding, Eq. (1), and the fixed-size padding baseline (Sec. 6.3)."""
from __future__ import annotations

import math

from ..data.schemas import Message, Session
from .stats import ShapeStats, last_real_reply


def pad_size(m: int, ratio: float, base: int) -> int:
    """Eq. (1): pad(m) = beta if m <= beta, else beta * r^ceil(log_r(m/beta)), in whole bytes.

    Rounding up to whole bytes adds less than one byte for non-integer r (Prop. 1 text).
    """
    if m < 0:
        raise ValueError(f"message size must be >= 0, got {m}")
    if ratio <= 1:
        raise ValueError(f"bucket ratio r must be > 1, got {ratio}")
    if base < 1:
        raise ValueError(f"padding base beta must be >= 1, got {base}")
    if m <= base:
        return base
    k = math.ceil(math.log(m / base, ratio) - 1e-12)
    out = int(math.ceil(base * ratio ** k))
    if out < m:  # guard against floating-point error in the logarithm
        out = int(math.ceil(base * ratio ** (k + 1)))
    return out


def pad_session(sess: Session, ratio: float = 2.0, base: int = 512) -> tuple[Session, ShapeStats]:
    """Apply Eq. (1) to every call and reply body; counts and timing are unchanged."""
    msgs = [Message(m.t, m.site, m.direction, pad_size(m.size, ratio, base), m.real) for m in sess.messages]
    return sess.with_messages(msgs), ShapeStats(real_done=last_real_reply(msgs))


def pad_fixed(sess: Session, unit: int) -> tuple[Session, ShapeStats]:
    """Baseline (Sec. 6.3): every message becomes ceil(size/unit) messages of exactly `unit` bytes.

    The chunks of one message are sent back to back (0.1 ms apart in this offline model).
    """
    if unit < 1:
        raise ValueError("unit must be >= 1")
    msgs: list[Message] = []
    for m in sess.messages:
        for j in range(max(1, math.ceil(m.size / unit))):
            msgs.append(Message(m.t + 1e-4 * j, m.site, m.direction, unit, m.real))
    return sess.with_messages(msgs), ShapeStats(real_done=last_real_reply(msgs))


def fixed_unit_for_percentile(sessions: list[Session], pct: float = 99.0) -> int:
    """Unit that covers the given percentile of message sizes (Sec. 6.3 uses the 99th)."""
    import numpy as np

    sizes = [m.size for s in sessions for m in s.messages]
    if not sizes:
        raise ValueError("no messages to size the fixed padding unit")
    return int(math.ceil(float(np.percentile(sizes, pct))))
