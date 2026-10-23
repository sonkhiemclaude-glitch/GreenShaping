"""Differentially private release, Eq. (2) and Remark 2 (Sec. 4.2, 4.3, 5).

On every link and direction, for all N sites from the start of the session, the
shaper releases bytes only at fixed ticks k*tau, k = 1..K, K = ceil(H/tau)
(effective horizon K*tau >= H). Arrivals in ((k-1)tau, k*tau] form a_k.

* mode "queue" (Eq. 2, Prop. 3):  b_k = max(0, round(Q_k + L_k)),
  L_k ~ Laplace(Delta/eps_w), eps_w = eps_dir/K, Q_k = max(0, Q_{k-1} - b_{k-1}) + a_k.
* mode "arrivals" (Remark 2):     b_k = max(0, round(a_k + mu + L_k)),
  L_k ~ Laplace(Delta/eps_dir).

Exactly b_k bytes are sent: min(Q_k, b_k) queued bytes, the rest padding. Bytes
still queued after tick K are overflow (Sec. 4.3 iv): the session is a privacy
failure. A link-direction whose total arrivals exceed Delta is outside the
guarantee (Sec. 5, "Third") and is flagged `above_delta`.

Python's round() rounds half to even; for continuous Laplace noise ties have
probability zero, so the output distribution equals the paper's.
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np

from ..data.schemas import DIRECTIONS, DOWN, UP, Message, Session
from .exchanges import pair_replies
from .stats import ShapeStats

MODES = ("queue", "arrivals")


def tick_times(interval: float, horizon: float) -> list[float]:
    if interval <= 0 or horizon <= 0:
        raise ValueError("interval tau and horizon H must be > 0")
    K = int(math.ceil(horizon / interval - 1e-12))
    return [k * interval for k in range(1, K + 1)]


def laplace_scale(sensitivity: float, eps_dir: float, K: int, mode: str) -> float:
    """Noise scale: Delta/eps_w = Delta*K/eps_dir (queue) or Delta/eps_dir (arrivals)."""
    if sensitivity <= 0 or eps_dir <= 0 or K < 1:
        raise ValueError("Delta > 0, eps_dir > 0 and K >= 1 are required")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return sensitivity * K / eps_dir if mode == "queue" else sensitivity / eps_dir


class _LinkQueue:
    """State of the release on one link-direction (Eq. 2 / Remark 2)."""

    def __init__(self) -> None:
        self.Q = 0
        self.pending: deque[list[int]] = deque()   # [message index, bytes left]

    def push(self, idx: int, size: int) -> None:
        self.Q += size
        self.pending.append([idx, size])

    def release(self, a_k: int, tk: float, scale: float, mode: str, bias: float, rng: np.random.Generator,
                done: dict[int, float], release_cap_bytes: int | None = None) -> tuple[int, int]:
        """One tick: returns (b_k, real bytes sent); records completed message indices in `done`."""
        centre = self.Q if mode == "queue" else a_k + bias
        b = max(0, int(round(centre + rng.laplace(0.0, scale))))   # Eq. (2) / Remark 2
        if release_cap_bytes is not None:
            b = min(b, release_cap_bytes)
        send = min(self.Q, b)                                   # queued bytes; b - send is padding
        self.Q -= send
        left = send
        while left > 0 and self.pending:
            take = min(left, self.pending[0][1])
            self.pending[0][1] -= take
            left -= take
            if self.pending[0][1] == 0:
                done[self.pending[0][0]] = tk
                self.pending.popleft()
        return b, send


def release_link(arrivals: list[tuple[float, int, int]], ticks: list[float], scale: float, mode: str,
                 bias: float, rng: np.random.Generator, *, release_cap_bytes: int | None = None) -> tuple[list[tuple[float, int, bool]], dict[int, float], int]:
    """Run Eq. (2) / Remark 2 on one link-direction with FIXED arrival times (no dependencies).

    arrivals: (time, bytes, message index). Returns the released (tick, b_k, carries_real)
    for b_k > 0, the completion tick of each message index, and the overflow bytes. Used
    by tests and analysis; sessions are released jointly by `dp_release`.
    """
    if release_cap_bytes is not None and (isinstance(release_cap_bytes, bool) or not isinstance(release_cap_bytes, int) or release_cap_bytes < 1):
        raise ValueError('release cap must be a positive integer')
    arrivals = sorted(arrivals)
    q = _LinkQueue()
    a = 0
    released: list[tuple[float, int, bool]] = []
    done: dict[int, float] = {}
    for tk in ticks:
        a_k = 0
        while a < len(arrivals) and arrivals[a][0] <= tk:
            a_k += arrivals[a][1]
            q.push(arrivals[a][2], arrivals[a][1])
            a += 1
        b, send = q.release(a_k, tk, scale, mode, bias, rng, done, release_cap_bytes)
        if b > 0:
            released.append((tk, b, send > 0))
    return released, done, q.Q + sum(x[1] for x in arrivals[a:])


def dp_release(sess: Session, interval: float = 1.0, horizon: float = 20.0, eps_dir: float = 4.0,
               sensitivity: int = 65536, mode: str = "queue", bias: float = 0.0,
               session_start_bytes: int = 0,
               rng: np.random.Generator | None = None) -> tuple[Session, ShapeStats]:
    """Apply the DP release to all N sites and both directions of one session, jointly.

    Causality (Sec. 5, before Prop. 3; review B1): the release delays traffic, and later
    traffic depends on earlier releases. One shared tick clock drives all 2N queues:
    * a reply (DOWN) enters its queue at  done(call) + (reply.t - call.t), where its call
      is the latest earlier UP message on the same link (`exchanges.pair_replies`);
    * a call or poll (UP) u enters at  u.t + max(0, max_d done(d) - d.t)  over every DOWN
      message d (any link) with d.t < u.t: the agent's planning gap after each reply it
      could have seen is preserved, so no call of round r+1 leaves before round r's
      replies were delivered.
    A message whose dependency completes at tick k can enter no earlier than tick k+1.
    Arrivals in ((k-1)tau, k*tau] are released at tick k (paper convention; a message
    arriving exactly at a tick sees zero queueing delay there).
    Messages that never enter, or are still queued after tick K, count as overflow.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    ticks = tick_times(interval, horizon)
    scale = laplace_scale(sensitivity, eps_dir, len(ticks), mode)
    msgs = sess.messages
    parent, unpaired = pair_replies(msgs)
    stats = ShapeStats(unpaired_bytes=unpaired)
    for s in range(sess.n_sites):
        for d in DIRECTIONS:
            if sum(m.size for m in msgs if m.site == s and m.direction == d) > sensitivity:
                stats.above_delta = True

    downs = sorted((j for j, m in enumerate(msgs) if m.direction == DOWN), key=lambda j: (msgs[j].t, j))
    down_times = [msgs[j].t for j in downs]
    n_deps = {j: bisect.bisect_left(down_times, m.t) for j, m in enumerate(msgs) if m.direction == UP}
    prefix_delay = [0.0]           # prefix_delay[p] = max(0, max over first p DOWN of done - t)
    done: dict[int, float] = {}
    waiting = sorted(range(len(msgs)), key=lambda j: (msgs[j].t, j))
    queues = {(s, d): _LinkQueue() for s in range(sess.n_sites) for d in DIRECTIONS}

    def arrival(j: int) -> float | None:
        m = msgs[j]
        if m.direction == UP:
            p = n_deps[j]
            return m.t + prefix_delay[p] if p < len(prefix_delay) else None
        par = parent.get(j)
        if par is None:
            return m.t
        if par not in done:
            return None
        return done[par] + max(0.0, m.t - msgs[par].t)

    out: list[Message] = []
    if session_start_bytes > 0:                     # Sec. 4.3 (iii): start signal to all N sites
        out += [Message(0.0, s, UP, session_start_bytes, False) for s in range(sess.n_sites)]
    for tk in ticks:
        a_k: dict[tuple[int, int], int] = {k: 0 for k in queues}
        still: list[int] = []
        for j in waiting:
            t_in = arrival(j)
            if t_in is not None and t_in <= tk:
                m = msgs[j]
                queues[(m.site, m.direction)].push(j, m.size)
                a_k[(m.site, m.direction)] += m.size
            else:
                still.append(j)
        waiting = still
        for s in range(sess.n_sites):
            for d in DIRECTIONS:
                b, send = queues[(s, d)].release(a_k[(s, d)], tk, scale, mode, bias, rng, done)
                if b > 0:
                    out.append(Message(tk, s, d, b, send > 0))
        while len(prefix_delay) - 1 < len(downs) and downs[len(prefix_delay) - 1] in done:
            j = downs[len(prefix_delay) - 1]
            prefix_delay.append(max(prefix_delay[-1], done[j] - msgs[j].t))
    stats.overflow_bytes = sum(q.Q for q in queues.values()) + sum(msgs[j].size for j in waiting)
    real_downs = [j for j in downs if msgs[j].real]
    if real_downs:
        if all(j in done for j in real_downs):
            stats.real_done = max(done[j] for j in real_downs)
        else:                       # latency is then a lower bound; the session is an overflow failure
            stats.real_done, stats.real_truncated = ticks[-1], True
    wake = {s: list(ticks) for s in range(sess.n_sites)}   # every site awake at every tick (Eq. 3)
    return sess.with_messages(out, horizon=max(ticks[-1], sess.duration()), wake=wake), stats
