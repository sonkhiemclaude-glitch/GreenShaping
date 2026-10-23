"""E4: server vs edge node classes (Sec. 7.4).

Repeats E2 and E3 for each configured node class (review M4). For a class c:
* sessions: those whose sites all ran on class c (manifest `node_classes`); synthetic
  sessions have no node classes and are used for every class;
* energy: Eq. (3) with class c's fitted coefficients for every site, the orchestrator
  with its own "orchestrator" model (or the default); MEASURED energy only from rows of
  the measurement CSV whose `node_class` is c.
Set `experiments.e4.repeat_e3: false` to skip the E3 repetition (it profiles the whole grid).
"""
from __future__ import annotations

import logging
from typing import Any

from .common import RunContext
from .e2_primitives import run_e2
from .e3_budgeted import run_e3

log = logging.getLogger(__name__)


def run_e4(ctx: RunContext) -> list[dict[str, Any]]:
    classes = [c for c in ctx.cost_models if c not in ("default", "orchestrator")] or ["default"]
    repeat_e3 = bool(ctx.cfg.get("experiments", {}).get("e4", {}).get("repeat_e3", True))
    rows: list[dict[str, Any]] = []
    for c in classes:
        sub = ctx.for_sessions(lambda s, c=c: not s.meta.get("node_classes") or set(s.meta["node_classes"]) == {c})
        if not (sub.train and sub.select and sub.test):
            log.warning("E4: no complete split of sessions recorded on node class %s; skipped", c)
            continue
        for r in run_e2(sub, node_class=c):
            r["experiment"] = "E4-E2"
            rows.append(r)
        if repeat_e3:
            sel, _ = run_e3(sub, final_eval=True, node_class=c)
            for r in sel:
                r["experiment"] = "E4-" + str(r.get("experiment", "E3"))
                r.setdefault("config", r.get("selected", ""))
                rows.append(r)
    if not rows:
        raise ValueError("E4 produced no rows")
    return rows
