"""Drive testbed sessions and write manifests (Sec. 6.1). Run on the testbed, not in tests.

Sites must already be running (`python -m greenshaping.cli site-server ...` on each host),
and tcpdump must already capture the orchestrator interface (see `capture-commands`).

Usage:
  python scripts/run_testbed_sessions.py --questions questions.jsonl --sites sites.yaml --out runs/run01 \
      --orchestrator scripted
`--orchestrator scripted` replays question plans WITHOUT an LLM (pipeline validation only).
`--orchestrator langgraph` is not implemented yet (TODO L-1).
"""
from __future__ import annotations

import argparse

from greenshaping.testbed.driver import run_session
from greenshaping.testbed.orchestrator import LangGraphOrchestrator, ScriptedOrchestrator
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.utils.io import load_yaml
from greenshaping.utils.logging import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--sites", required=True, help="YAML: sites: [{host, port, node_class}]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--orchestrator", choices=["scripted", "langgraph"], default="scripted")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--allow-small-bank", action="store_true")
    ap.add_argument("--start-sid", type=int, default=0, help="first session id (use distinct ranges per N and repeat)")
    a = ap.parse_args()
    setup_logging()
    sites_cfg = load_yaml(a.sites)["sites"]
    sites = [(s["host"], int(s["port"])) for s in sites_cfg]
    classes = [s.get("node_class", "unknown") for s in sites_cfg]
    qs = load_question_bank(a.questions, require_min=not a.allow_small_bank)
    orch = ScriptedOrchestrator() if a.orchestrator == "scripted" else LangGraphOrchestrator()
    sid = a.start_sid
    for _ in range(a.repeat):
        for q in qs:
            run_session(orch, q, sid, sites, classes, a.out, getattr(orch, "llm", "unspecified"))
            sid += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
