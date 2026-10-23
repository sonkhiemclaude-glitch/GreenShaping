"""Synchronised rounds with cover sites (Sec. 4.2, Prop. 2).

Construction (paper Sec. 4.2):
1. At session start each site the agent will not call becomes a cover site with
   probability q_i; the decision holds for the whole session. Called sites plus
   cover sites form the active set.
2. Calls are released only at multiples of the round length W. At each round
   boundary b every active site receives exactly one call frame whose size is the
   round's largest (merged) call, padded with Eq. (1). The orchestrator holds all
   calls it releases, so this is causal.
3. Replies travel in frames of fixed size R (R - 6 payload bytes after the frame
   header). Every active site sends one frame at b + W; if any reply is incomplete, the orchestrator polls every active site and
   each sends one more frame per further boundary. A called site fills frames with
   reply data that is ready; a cover site sends filler. Polling stops once every
   called site has signalled completion inside its encrypted frames, so all active
   sites send the same number J of frames.

Within a session every active link carries the same (time, direction, size)
sequence (Prop. 2a). Poll timing and size are not given in the paper (audit A-4):
polls are sent `poll_offset` seconds before each continuation frame.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..data.schemas import DOWN, UP, Message, Session
from ..testbed.framing import HEADER
from .exchanges import pair_replies
from .padding import pad_size
from .stats import ShapeStats


def group_rounds(calls: list[Message], gap: float = 0.1) -> list[list[Message]]:
    """Group the agent's calls into rounds: calls issued within `gap` s of the round's first call."""
    return [[m for _, m in r] for r in _group_indexed(list(enumerate(calls)), gap)]


def _group_indexed(calls: list[tuple[int, Message]], gap: float) -> list[list[tuple[int, Message]]]:
    rounds: list[list[tuple[int, Message]]] = []
    for j, c in sorted(calls, key=lambda x: (x[1].t, x[0])):
        if rounds and c.t - rounds[-1][0][1].t <= gap:
            rounds[-1].append((j, c))
        else:
            rounds.append([(j, c)])
    return rounds


def draw_active_set(called: set[int], n_sites: int, cover_prob: float | list[float],
                    rng: np.random.Generator) -> set[int]:
    """Session-level cover decision (Sec. 4.2): each uncalled site i is a cover site w.p. q_i."""
    q = [cover_prob] * n_sites if isinstance(cover_prob, (int, float)) else list(cover_prob)
    if len(q) != n_sites or any(not 0.0 <= x <= 1.0 for x in q):
        raise ValueError("cover_prob must be a probability or one probability per site")
    return set(called) | {s for s in range(n_sites) if s not in called and rng.random() < q[s]}


def synchronised_rounds(sess: Session, window: float = 0.5, cover_prob: float | list[float] = 1.0,
                        reply_frame: int = 16384, poll_size: int = 256, poll_offset: float | None = None,
                        call_ratio: float = 2.0, call_base: int = 512, group_gap: float = 0.1,
                        rng: np.random.Generator | None = None) -> tuple[Session, ShapeStats]:
    """Apply synchronised rounds with cover sites to one session (see module docstring).

    Replies are paired with calls by time (`exchanges.pair_replies`, review M1). Every
    frame carries the 6-byte frame header of `testbed.framing`, so a reply frame of R
    bytes carries R - 6 reply bytes and a call frame is sized for call + header.
    """
    if window <= 0 or reply_frame <= HEADER.size or poll_size < 0 or group_gap < 0:
        raise ValueError(f"window > 0, reply_frame > {HEADER.size}, poll_size >= 0 and group_gap >= 0 are required")
    rng = rng if rng is not None else np.random.default_rng(0)
    poll_offset = window / 2 if poll_offset is None else poll_offset
    if not 0 <= poll_offset < window:
        raise ValueError("poll_offset must be in [0, window)")
    payload = reply_frame - HEADER.size

    def next_boundary(t: float) -> float:
        return math.ceil(t / window - 1e-12) * window

    msgs = sess.messages
    calls = [(j, m) for j, m in enumerate(msgs) if m.direction == UP]
    parent, unpaired = pair_replies(msgs)
    replies_of: dict[int, list[Message]] = defaultdict(list)
    orphans: dict[int, list[Message]] = defaultdict(list)
    for j, p in parent.items():
        (replies_of[p] if p is not None else orphans[msgs[j].site]).append(msgs[j])
    if not calls and not orphans:
        return sess.with_messages(list(msgs)), ShapeStats()
    rounds = _group_indexed(calls, group_gap)
    if not rounds:                                   # replies without any call: one pseudo round
        rounds = [[]]
    called = {m.site for _, m in calls} | set(orphans)
    active = draw_active_set(called, sess.n_sites, cover_prob, rng)

    out: list[Message] = []
    prev_orig_end, prev_new_end, shift, real_done = 0.0, 0.0, 0.0, 0.0
    tj = 0.0
    for r_idx, rnd in enumerate(rounds):
        if rnd:
            t0, t_last = rnd[0][1].t, max(m.t for _, m in rnd)
        else:
            t0 = t_last = min(m.t for lst in orphans.values() for m in lst)
        planning_gap = t0 - prev_orig_end           # the agent's planning time between rounds
        # the frame leaves only once every call grouped into this round exists (review minor 1)
        b = next_boundary(max(prev_new_end + planning_gap, t_last + shift))
        merged: dict[int, int] = defaultdict(int)
        ready: dict[int, list[tuple[int, float]]] = defaultdict(list)   # site -> (bytes, ready time)
        for j, c in rnd:
            merged[c.site] += c.size
            for r in replies_of.get(j, []):
                ready[c.site].append((r.size, r.t + (b - c.t)))
        if r_idx == 0:
            for s_, lst in orphans.items():
                merged.setdefault(s_, 0)
                ready[s_] += [(r.size, r.t + (b - t0)) for r in lst]
        call_frame = pad_size(max(merged.values()) + HEADER.size, call_ratio, call_base)
        for s in sorted(active):
            out.append(Message(b, s, UP, call_frame, s in merged))
        remaining = {s: sum(x for x, _ in ready[s]) for s in merged}
        orig_end = max([t_last] + [r.t for j, _ in rnd for r in replies_of.get(j, [])]
                       + ([r.t for lst in orphans.values() for r in lst] if r_idx == 0 else []))
        j = 0
        while True:
            tj = b + window + j * window
            if j >= 1 and poll_size > 0:
                for s in sorted(active):
                    out.append(Message(tj - poll_offset, s, UP, poll_size, False))
            for s in sorted(active):
                avail = sum(x for x, t in ready.get(s, []) if t <= tj) - (sum(x for x, _ in ready.get(s, [])) - remaining.get(s, 0))
                carries = s in remaining and remaining[s] > 0 and avail > 0
                if carries:
                    remaining[s] = max(0, remaining[s] - min(payload, avail))
                out.append(Message(tj, s, DOWN, reply_frame, carries))
            j += 1
            if all(v == 0 for v in remaining.values()):
                break
        real_done = tj
        prev_orig_end, prev_new_end, shift = orig_end, tj, b - t0
    horizon = max(sess.duration() + shift, real_done) + 0.5
    return sess.with_messages(out, horizon=horizon), ShapeStats(real_done=real_done, unpaired_bytes=unpaired)
