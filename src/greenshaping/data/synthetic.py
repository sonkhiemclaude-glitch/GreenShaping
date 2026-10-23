"""Synthetic session generator for smoke tests ONLY.

Every parameter below is an assumption chosen for plausibility, not a
measurement. Leakage, cost or energy figures computed on these sessions are
"synthetic smoke-test output, not a paper result". Real sessions come from
testbed captures through `data.adapters`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .schemas import DOWN, NOT_PARTICIPATING, QUERY_FAMILIES, UP, Message, Session, size_bucket


@dataclass(frozen=True)
class SyntheticConfig:
    n_sessions: int = 300
    n_sites: int = 10
    n_questions: int = 80
    p_participate: float = 0.5
    http_overhead: int = 220            # bytes of framing per message (assumed)
    plan_delay_median: float = 1.5      # s, LLM planning before each round (assumed)
    plan_delay_sigma: float = 0.5
    rtt_min: float = 0.005
    rtt_max: float = 0.040
    rows_log10_min: float = 3.0
    rows_log10_max: float = 6.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _question_params(rng: np.random.Generator, family: int) -> dict[str, int]:
    if family == 0:
        return {"rounds": 1, "call": int(rng.integers(450, 900)), "n_stats": int(rng.integers(1, 8))}
    if family == 1:
        return {"rounds": 1, "call": int(rng.integers(550, 1100)), "groups": int(rng.integers(5, 200))}
    if family == 2:
        return {"rounds": int(rng.integers(2, 6)), "call": int(rng.integers(500, 900)),
                "params": int(rng.integers(1_000, 10_000))}
    return {"rounds": 1, "call": int(rng.integers(300, 600))}


def generate_sessions(cfg: SyntheticConfig) -> list[Session]:
    """Generate `cfg.n_sessions` synthetic sessions (deterministic for a given seed)."""
    if cfg.n_sites < 1 or cfg.n_sessions < 1 or cfg.n_questions < len(QUERY_FAMILIES):
        raise ValueError("need n_sites >= 1, n_sessions >= 1 and n_questions >= 4")
    rng = np.random.default_rng(cfg.seed)
    rows = 10 ** rng.uniform(cfg.rows_log10_min, cfg.rows_log10_max, size=cfg.n_sites)
    rtt = rng.uniform(cfg.rtt_min, cfg.rtt_max, size=cfg.n_sites)
    questions = [(q % len(QUERY_FAMILIES), _question_params(rng, q % len(QUERY_FAMILIES)))
                 for q in range(cfg.n_questions)]
    sessions: list[Session] = []
    for sid in range(cfg.n_sessions):
        qid = int(rng.integers(0, cfg.n_questions))
        fam, qp = questions[qid]
        part = (rng.random(cfg.n_sites) < cfg.p_participate).astype(int)
        if part.sum() == 0:
            part[int(rng.integers(0, cfg.n_sites))] = 1
        msgs: list[Message] = []
        t = 0.0
        for _ in range(qp["rounds"]):
            t += float(rng.lognormal(math.log(cfg.plan_delay_median), cfg.plan_delay_sigma))
            finish = t
            for i in np.flatnonzero(part):
                ts = t + float(rng.uniform(0, 0.02))
                call = qp["call"] + (4 * qp["params"] if fam == 2 else 0)
                call += int(rng.integers(-40, 40)) + cfg.http_overhead
                msgs.append(Message(ts, int(i), UP, call))
                if fam == 0:
                    reply, work = 250 + 24 * qp["n_stats"], 0.002 + rows[i] * 2e-7
                elif fam == 1:
                    g = min(qp["groups"], max(1, int(rows[i] / 100)))
                    reply, work = 200 + 40 * g, 0.004 + rows[i] * 4e-7
                elif fam == 2:
                    reply, work = 4 * qp["params"] + 120, 0.05 + rows[i] * 3e-6
                else:
                    reply, work = 150 + len(str(int(rows[i]))), 0.001 + rows[i] * 5e-8
                reply += int(rng.integers(-20, 20)) + cfg.http_overhead
                tr = ts + rtt[i] + work * float(rng.lognormal(0, 0.15))
                msgs.append(Message(tr, int(i), DOWN, reply))
                finish = max(finish, tr + rtt[i] / 2)
            t = finish
        buckets = [size_bucket(rows[i]) if part[i] else NOT_PARTICIPATING for i in range(cfg.n_sites)]
        s = Session(sid, qid, cfg.n_sites, fam, part.tolist(), buckets, [], t + 0.5,
                    meta={"data_kind": "synthetic"})
        sessions.append(s.with_messages(msgs))
    return sessions
