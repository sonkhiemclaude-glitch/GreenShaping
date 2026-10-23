"""Call/reply pairing used by the rounds and DP primitives (review M1).

Real captures do not give one DOWN message per UP message: a streamed reply with a
pause longer than the segment-merge gap becomes several DOWN messages, and calls
issued close together can merge into one UP message. Pairing is therefore by time,
not by index: every DOWN message belongs to the latest UP message on the same link
that precedes it (or starts at the same instant). A DOWN message with no earlier UP
message on its link is "unpaired"; its bytes are counted and never dropped.
"""
from __future__ import annotations

from ..data.schemas import DOWN, UP, Message


def pair_replies(msgs: list[Message]) -> tuple[dict[int, int | None], int]:
    """Map each DOWN message index to the index of its call (None if unpaired).

    Returns (parent, unpaired_bytes). Ties in time keep the original message order.
    """
    order = sorted(range(len(msgs)), key=lambda j: (msgs[j].t, msgs[j].direction != UP, j))
    last_up: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    unpaired = 0
    for j in order:
        m = msgs[j]
        if m.direction == UP:
            last_up[m.site] = j
        elif m.direction == DOWN:
            p = last_up.get(m.site)
            parent[j] = p
            if p is None:
                unpaired += m.size
    return parent, unpaired
