"""Five local TLS sites, synthetic tables, optional Ollama planning, capture and offline analysis.

This is a rehearsal, not a paper experiment. Run unprivileged; start/stop a narrowly
filtered tcpdump separately when pilot_status.json requests it. No credentials are read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from greenshaping.data.adapters import SessionDirectoryAdapter, export_sessions, split_capture_by_sessions
from greenshaping.data.schemas import QUERY_FAMILIES
from greenshaping.experiments.runner import run_from_config
from greenshaping.testbed.capture_commands import tshark_export_command
from greenshaping.testbed.driver import run_session
from greenshaping.testbed.orchestrator import ScriptedOrchestrator
from greenshaping.testbed.question_bank import Question
from greenshaping.utils.io import dump_yaml, write_json, write_jsonl
from greenshaping.utils.provenance import file_sha256


def wait_for(path: Path, timeout: float = 180.0) -> None:
    until = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > until:
            raise TimeoutError(f"waiting for {path.name}")
        time.sleep(0.2)


def make_tables(root: Path, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    paths = []
    for i, rows in enumerate((8000, 12000, 40000, 80000, 200000)):
        x1, x2 = rng.normal(size=(2, rows))
        frame = pd.DataFrame({"cohort": np.arange(rows) % 4, "region": i % 3,
                              "x1": x1, "x2": x2, "value": 100 + 15 * x1 + rng.normal(size=rows),
                              "outcome": (x1 + x2 + rng.normal(size=rows) > 0).astype(int)})
        path = root / f"site_{i}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def make_questions(model: str | None, endpoint: str, out: Path, per_family: int,
                   seed: int) -> list[Question]:
    regions = [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]
    questions = []
    plans = []
    for family in QUERY_FAMILIES:
        for j in range(per_family):
            region = regions[j % len(regions)]
            where = {"cohort": j % 4} if j % 2 == 0 else None
            params: dict[str, Any]
            if family == "aggregate":
                params = {"columns": ["value", "x1"]}
            elif family == "group_by":
                params = {"by": "cohort", "column": "value"}
            elif family == "train_round":
                params = {"features": ["x1", "x2"], "label": "outcome", "weights": [0.0, 0.0, 0.0]}
            else:
                params = {}
            if where:
                params["where"] = where
            text = f"Run {family} for synthetic regions {region}, filters {where or {}}, parameters {params}."
            expected = [i for i in range(5) if i % 3 in region]
            chosen = expected
            evidence: dict[str, Any] = {"question_id": len(questions), "text": text,
                                        "expected_sites": expected, "planner": "scripted"}
            if model:
                catalog = [{"site_id": i, "region_id": i % 3} for i in range(5)]
                prompt = ("Filter a site catalog by exact region_id membership. Return site_id values, not region_id values. "
                          "For each catalog row independently, include it if and only if its region_id is in requested_regions. "
                          f"Input: {json.dumps({'catalog': catalog, 'requested_regions': region})}. "
                          f"Analyst question: {text} "
                          'Return only a JSON object {"sites": [integer site IDs]} sorted by site_id.')
                body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}],
                        "format": {"type": "object", "properties": {
                            "sites": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 4},
                                      "minItems": 1, "maxItems": 5, "uniqueItems": True}},
                            "required": ["sites"], "additionalProperties": False},
                        "options": {"temperature": 0, "seed": seed, "num_predict": 96, "num_ctx": 2048}}
                evidence.update(planner="ollama", model=model, prompt=prompt, attempts=[])
                plans.append(evidence)
                for attempt in range(3):
                    req = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=90) as response:
                        answer = json.load(response)
                    chosen = sorted(set(json.loads(answer["message"]["content"])["sites"]))
                    evidence["attempts"].append({"request": json.loads(json.dumps(body)), "response": answer,
                                                 "chosen_sites": chosen, "selection_correct": chosen == expected})
                    evidence.update(chosen_sites=chosen, selection_correct=chosen == expected)
                    write_jsonl(plans, out / "llm_plans.jsonl")
                    if chosen == expected:
                        break
                    body["messages"].extend([
                        {"role": "assistant", "content": answer["message"]["content"]},
                        {"role": "user", "content": "The selection failed the exact region membership check. "
                         "Recheck every row of the original catalog using numeric equality; include every matching row "
                         "and exclude every other row. Return the corrected JSON. Do not infer ranges of site IDs."}])
                    print(f"question {len(questions)}: rejected plan attempt {attempt + 1}", flush=True)
                else:
                    raise ValueError(f"LLM selected wrong sites for question {len(questions)} after 3 attempts; no fallback")
            else:
                plans.append(evidence)
            calls = [{"site": i, "params": params} for i in chosen]
            plan = [calls, calls] if family == "train_round" else [calls]
            questions.append(Question(len(questions), family, text, plan))
            print(f"planned question {len(questions)}/{len(QUERY_FAMILIES) * per_family}", flush=True)
    write_jsonl(plans, out / "llm_plans.jsonl")
    write_jsonl([q.__dict__ for q in questions], out / "questions.jsonl")
    # These are precomputed LLM site selections replayed by the scripted transport,
    # not an online LangGraph agent. Both the request and exact output are preserved.
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--questions-per-family", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--base-port", type=int, default=18400)
    args = parser.parse_args()
    if args.repeat < 1 or args.questions_per_family < 3:
        parser.error("repeat >= 1 and questions-per-family >= 3 are required")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for d in ("tables", "tls", "logs", "manifests", "captures"):
        (out / d).mkdir()
    status: dict[str, Any] = {"state": "running", "phase": "prepare", "data_kind": "rehearsal",
                              "N": 5, "physical_hosts": 1, "interface": "lo0",
                              "ports": list(range(args.base_port, args.base_port + 5)),
                              "energy_measured": False, "pid": os.getpid()}

    def update(phase: str, **extra: Any) -> None:
        status.update(phase=phase, **extra)
        write_json(status, out / "pilot_status.json")

    children = []
    handles = []
    try:
        update("generate_tables")
        paths = make_tables(out / "tables", args.seed)
        write_json({"generator": "independent normal features; synthetic binary outcome; four cohorts",
                    "seed": args.seed, "site_region": [i % 3 for i in range(5)],
                    "rows": [8000, 12000, 40000, 80000, 200000],
                    "sha256": {p.name: file_sha256(p) for p in paths}}, out / "dataset_metadata.json")
        cert, key = out / "tls/cert.pem", out / "tls/key.pem"
        with (out / "logs/tls.log").open("w") as log:
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
                            "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
                            "-keyout", str(key), "-out", str(cert)], check=True, stdout=log, stderr=log)
        key.chmod(0o600)
        update("llm_planning" if args.model else "scripted_planning")
        questions = make_questions(args.model, args.endpoint, out, args.questions_per_family, args.seed)
        update("start_sites")
        for i, path in enumerate(paths):
            log = (out / "logs" / f"site_{i}.log").open("w")
            handles.append(log)
            children.append(subprocess.Popen([sys.executable, "-m", "greenshaping.cli", "site-server", "--table", str(path),
                                              "--host", "127.0.0.1", "--port", str(args.base_port + i),
                                              "--certfile", str(cert), "--keyfile", str(key)], stdout=log, stderr=log))
        until = time.monotonic() + 60
        for port in status["ports"]:
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > until or any(p.poll() is not None for p in children):
                        raise RuntimeError("site failed to start; inspect site logs") from None
                    time.sleep(0.2)
        # Enforce TLS 1.3 for this pilot, leaving other experiments' SSL defaults alone.
        class PilotOrchestrator(ScriptedOrchestrator):
            def _conn(self, host: str, port: int) -> Any:
                conn = super()._conn(host, port)
                conn._context.minimum_version = ssl.TLSVersion.TLSv1_3
                conn._context.maximum_version = ssl.TLSVersion.TLSv1_3
                return conn

        orch = PilotOrchestrator(use_tls=True, cafile=str(cert))
        update("awaiting_capture", site_pids=[p.pid for p in children])
        wait_for(out / "capture.ready")
        sites = [("127.0.0.1", port) for port in status["ports"]]
        llm = f"Ollama {args.model}; precomputed site selections replayed" if args.model else "none (scripted rehearsal)"
        sid = 0
        order = [(rep, q) for rep in range(args.repeat) for q in questions]
        np.random.default_rng(args.seed).shuffle(order)
        for rep, q in order:
            update("collecting", session=sid, expected_sessions=len(order))
            manifest = run_session(orch, q, sid, sites, ["mac_studio_loopback"] * 5, out, llm, data_kind="rehearsal")
            write_json({**manifest.to_dict(), "pilot_repetition": rep}, out / "manifests" / f"{sid}.json")
            sid += 1
            time.sleep(0.05)
        update("awaiting_capture_stop", collected_sessions=sid)
        wait_for(out / "capture.stopped")
        update("export_capture")
        command = tshark_export_command(str(out / "traffic.pcap"))
        command[0] = shutil.which("tshark") or "/Applications/Wireshark.app/Contents/MacOS/tshark"
        with (out / "all.csv").open("w") as output, (out / "logs/tshark.log").open("w") as log:
            subprocess.run(command, check=True, stdout=output, stderr=log)
        counts = split_capture_by_sessions(out / "all.csv", out / "manifests", out / "captures")
        sessions = SessionDirectoryAdapter(out).load()
        if any(not s.messages for s in sessions) or len(sessions) != len(order):
            raise RuntimeError("capture integrity failure: missing sessions or messages")
        export_sessions(sessions, out / "sessions.jsonl")
        write_json({"sessions": len(sessions), "questions": len(questions), "rows_by_session": counts,
                    "data_kinds": sorted({s.meta["data_kind"] for s in sessions}),
                    "clipped_size_buckets": sum(s.meta["clipped_size_buckets"] for s in sessions),
                    "pcap_sha256": file_sha256(out / "traffic.pcap"), "model": args.model,
                    "planning_mode": "precomputed site choices replayed; family/parameters fixed by question",
                    "limitations": ["single Mac, loopback, synthetic tables", "no live shaping or measured energy",
                                    "connections recreated per session", "not the paper's online LangGraph agent"]},
                   out / "pilot_quality.json")
        update("offline_analysis")
        cfg = {"run": {"name": "mac_studio_tls_pilot", "seed": args.seed, "data_kind": "rehearsal",
                       "output_dir": str(out / "analysis")},
               "data": {"source": "directory", "root": str(out), "gap_s": 0.010},
               "split": {"fractions": [0.4, 0.3, 0.3], "seed": args.seed},
               "attack": {"models": ["rf"], "n_estimators": 50, "n_boot": 100},
               "energy": {"cost_models": {}, "measured_energy_csv": None},
               "grid": {"padding": {"ratio": [2.0], "base": [512]},
                        "rounds": {"window": [0.1], "cover_prob": [0.5, 1.0], "reply_frame": [2048]},
                        "dp": {"mode": ["queue", "arrivals"], "interval": [0.1], "eps_dir": [4.0],
                               "delta_percentile": [100.0], "horizon": [5.0]}},
               "budgets": {"site_J": []}, "baselines": {"netshaper_root": None},
               "experiments": {"e4": {"repeat_e3": False}}}
        dump_yaml(cfg, out / "pilot_config.yaml")
        # E4 compares hardware classes; a single-host pilot cannot supply that comparison.
        for stage in ("e1", "e2", "e3", "e5"):
            run_from_config(cfg, stage)
        from greenshaping.experiments.plot_results import make_all_plots
        make_all_plots(out / "analysis", out / "analysis/plots")
        update("complete", state="complete", skipped_stages={"e4": "single hardware class; no platform comparison"},
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return 0
    except Exception as exc:
        update("failed", state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
