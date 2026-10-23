"""Testbed driver (Sec. 6.1): submit questions, write one manifest per session.

Captures are taken separately with tcpdump at the orchestrator interface
(testbed.capture_commands); the manifest's `start_time` uses the same clock.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
from pathlib import Path

from ..data.schemas import SessionManifest, SiteSpec
from .orchestrator import Orchestrator
from .question_bank import Question

log = logging.getLogger(__name__)


def resolve_sites(sites: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Replace host names (Tailscale, .local) by IPv4 addresses, as the capture shows IPs (minor 23)."""
    out = []
    for h, p in sites:
        try:
            ipaddress.ip_address(h)
            out.append((h, p))
        except ValueError:
            out.append((socket.gethostbyname(h), p))
    return out


def run_session(orch: Orchestrator, question: Question, session_id: int, sites: list[tuple[str, int]],
                node_classes: list[str], out_dir: str | Path, llm: str, overwrite: bool = False,
                data_kind: str | None = None) -> SessionManifest:
    """Run one question and write manifests/<session_id>.json (refuses to overwrite by default)."""
    path = Path(out_dir) / "manifests" / f"{session_id}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use another --start-sid or output directory")
    sites = resolve_sites(sites)
    start = time.time()
    rows = orch.run(question, sites)
    end = time.time()
    man = SessionManifest(
        session_id=session_id, question_id=question.question_id, query_family=question.family,
        start_time=start, llm=llm,
        sites=tuple(SiteSpec(ip=h, port=p, participates=int(i in rows), matching_rows=rows.get(i),
                             node_class=node_classes[i]) for i, (h, p) in enumerate(sites)),
        data_kind=data_kind, end_time=end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man.to_dict(), indent=1))
    log.info("session %d (question %d) -> %s", session_id, question.question_id, path)
    return man
