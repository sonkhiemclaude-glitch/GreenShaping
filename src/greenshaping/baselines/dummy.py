"""Dummy defence for unit tests ONLY. It is not a baseline from the paper and must never be reported."""
from __future__ import annotations

import numpy as np

from ..data.schemas import Message, Session
from ..shaping.stats import ShapeStats, last_real_reply
from .base import Defence


class DummyDefence(Defence):
    """Adds a constant number of bytes to every message (deterministic; tests only)."""

    name, kind = "dummy_test_only", "dummy"

    def __init__(self, extra_bytes: int = 100) -> None:
        self.extra_bytes = extra_bytes

    def apply(self, sess: Session, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
        msgs = [Message(m.t, m.site, m.direction, m.size + self.extra_bytes, m.real) for m in sess.messages]
        return sess.with_messages(msgs), ShapeStats(real_done=last_real_reply(msgs))
