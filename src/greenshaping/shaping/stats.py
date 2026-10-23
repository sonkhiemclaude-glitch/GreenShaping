"""Per-session bookkeeping returned by every primitive."""
from __future__ import annotations

from dataclasses import dataclass

from ..data.schemas import DOWN, Message


@dataclass
class ShapeStats:
    real_done: float = 0.0        # time the last real reply byte reaches the orchestrator
    overflow_bytes: int = 0       # bytes not released by the last DP tick (Sec. 4.3 iv)
    above_delta: bool = False     # some link-direction entered more than Delta bytes (Sec. 5)
    unpaired_bytes: int = 0       # reply bytes with no earlier call on their link (review M1)
    real_truncated: bool = False  # a real reply had not been delivered by the last tick

    @property
    def privacy_failure(self) -> bool:
        """Overflow and above-Delta sessions are outside the DP guarantee (Sec. 5, 6.3)."""
        return self.overflow_bytes > 0 or self.above_delta


def last_real_reply(msgs: list[Message]) -> float:
    return max((m.t for m in msgs if m.real and m.direction == DOWN), default=0.0)
