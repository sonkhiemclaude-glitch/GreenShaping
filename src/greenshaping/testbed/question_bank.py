"""Analyst question bank (Sec. 6.1: >= 40 distinct questions per query family).

JSONL, one question per line:
{"question_id": 17, "family": "group_by", "text": "...", "plan": [[{"site": 0, "params": {...}}, ...], ...]}

`plan` is used only by the ScriptedOrchestrator (pipeline validation without an LLM).
For the LLM orchestrator, `text` is the analyst question and the plan is the agent's.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.schemas import QUERY_FAMILIES
from ..utils.io import read_jsonl

MIN_PER_FAMILY = 40


@dataclass(frozen=True)
class Question:
    question_id: int
    family: str
    text: str
    plan: list[list[dict[str, Any]]] = field(default_factory=list)   # rounds -> calls
    metadata: dict[str, Any] = field(default_factory=dict)


def load_question_bank(path: str | Path, require_min: bool = True) -> list[Question]:
    qs = []
    for d in read_jsonl(path):
        if d.get("family") not in QUERY_FAMILIES:
            raise ValueError(f"question {d.get('question_id')}: unknown family {d.get('family')!r}")
        qs.append(Question(int(d["question_id"]), d["family"], str(d["text"]), d.get("plan", []), d.get("metadata", {})))
    ids = [q.question_id for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question_id in question bank")
    if require_min:
        counts = Counter(q.family for q in qs)
        short = {f: counts.get(f, 0) for f in QUERY_FAMILIES if counts.get(f, 0) < MIN_PER_FAMILY}
        if short:
            raise ValueError(f"protocol needs >= {MIN_PER_FAMILY} questions per family; have {short}")
    return qs
