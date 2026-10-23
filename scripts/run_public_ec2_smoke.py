"""Run the staged, bounded preparation matrix on T1-T3; never uses T4.

Prerequisite: source, venv, CSVs, certificates and deployment manifest staged in
the dedicated public_preparation_01 directory. No service or host configuration changes.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import ec2_runner as r
import numpy as np


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected, strict=True))
    if isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-10))
    return actual == expected


def read_remote(node, source):
    """Only idempotent inspections may retry after an uncertain SSH result."""
    for attempt in range(3):
        try:
            return r.python(node, source, timeout=30)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(1)
    raise AssertionError("unreachable")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r.require_available(["T1", "T2", "T3"])
    out = a.out
    deployment = json.loads((out / "deployment.json").read_text())
    root = deployment["remote_root"]
    if not root.startswith(r.INVENTORY["remote_root"] + "/runs/public_preparation_"):
        raise ValueError("unexpected run directory")
    if (out / "integration_summary.json").exists():
        raise FileExistsError("preserve existing results; stage a new run for a repetition")
    processes = []

    def launch(node, args, log, env=None):
        code = f'''import subprocess,os
from pathlib import Path
p=Path({root!r})
with (p/{log!r}).open('xb') as f:
 child=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True,
                         env=dict(os.environ,**{env or {}!r}))
print(child.pid)
'''
        pid = int(r.python(node, code))
        processes.append((node, pid))
        (out / "owned_processes.json").write_text(json.dumps(processes))

    try:
        for site, node, port in [(0, "T2", 8000), (1, "T3", 8000), (2, "T3", 5555)]:
            launch(node, ["timeout", "--kill-after=3s", "220s", root + "/venv/bin/python", "-u",
                          root + "/public_ec2_smoke.py", "server", "--root", root, "--host",
                          r.INVENTORY["nodes"][node]["private_ip"], "--port", str(port), "--site", str(site)],
                   f"site_{site}.log", {"PYTHONPATH": root + "/src", "OPENBLAS_NUM_THREADS": "1"})
        # Readiness from owned service logs, without consuming a TLS connection.
        for _ in range(10):
            ready = True
            for site, node in [(0, "T2"), (1, "T3"), (2, "T3")]:
                data = read_remote(node, f"from pathlib import Path\nprint((Path({root!r})/'site_{site}.log').read_text())")
                if "Traceback" in data:
                    raise RuntimeError(f"site {site} failed to start; retained service log contains traceback")
                ready &= '"ready": true' in data
            if ready:
                break
            time.sleep(1)
        if not ready:
            raise RuntimeError("site readiness failed; see retained logs")
        launch("T1", ["timeout", "--kill-after=3s", "180s", "sudo", "-n", "tcpdump", "-i", "ens5", "-nn", "-s", "0",
                      "-U", "-w", root + "/tools.pcap", "tcp and (host 172.31.34.237 or host 172.31.38.41) "
                      "and (port 8000 or port 5555)"], "tcpdump.log")
        args = ["timeout", "--kill-after=3s", "170s", "env", "PYTHONPATH=" + root + "/src", "OPENBLAS_NUM_THREADS=1",
                root + "/venv/bin/python", "-u", root + "/public_ec2_smoke.py", "client", "--root", root]
        result = r.run("T1", shlex.join(args), timeout=180)
        (out / "client.log").write_text(result.stdout + result.stderr)
        (out / "client_exit.json").write_text(json.dumps({"returncode": result.returncode}))
        r.copy("T1", out / "integration_results.json", root + "/results.json", get=True)
    except BaseException as exc:
        (out / "integration_failure.json").write_text(json.dumps({"error": repr(exc)}))
        raise
    finally:
        cleanup = []
        for node, pid in reversed(processes):
            try:
                r.python(node, f'''import os,signal,subprocess,time
from pathlib import Path
p=Path('/proc/{pid}/cmdline')
if p.exists() and {root.encode()!r} in p.read_bytes():
 subprocess.run(['sudo','-n','kill','-INT','--','-{pid}'],capture_output=True)
 time.sleep(.5)
 if p.exists() and {root.encode()!r} in p.read_bytes():
  subprocess.run(['sudo','-n','kill','-TERM','--','-{pid}'],capture_output=True)
''')
                cleanup.append({"node": node, "pid": pid, "signal_attempted": True})
            except Exception as exc:
                cleanup.append({"node": node, "pid": pid, "error": repr(exc), "bounded_by_timeout": True})
        (out / "cleanup.json").write_text(json.dumps(cleanup, indent=2))
        for site, node in [(0, "T2"), (1, "T3"), (2, "T3")]:
            r.copy(node, out / f"site_{site}.log", root + f"/site_{site}.log", get=True)
        for name in ("tcpdump.log", "tools.pcap"):
            exists = r.python("T1", f"from pathlib import Path\nprint((Path({root!r})/{name!r}).exists())").strip()
            if exists == "True":
                r.copy("T1", out / name, root + "/" + name, get=True)
    reference = json.loads((out / "smoke_reference.json").read_text())
    results = json.loads((out / "integration_results.json").read_text())
    checks = []
    for row in results:
        expected = reference[str(row["question_id"])]
        responses = [{k: v[k] for k in ("result", "matching_rows")} for v in row.get("responses", [])]
        checks.append({"question_id": row["question_id"], "mode": row["mode"], "status": row["status"],
                       "reference_match": equivalent(responses, expected),
                       "upstream_ok": bool(row.get("transport")) and row["transport"].get("measurement_valid", False) and all(
                           not v["cap_exceeded"] and not v["error"] and v["overflow_bytes"] == 0
                           for v in row["transport"]["upstream"])})
    summary = {"scope": "scripted public-data TLS correctness; not LLM, privacy, capacity or energy evidence",
               "logical_sites": 3, "physical_site_vms": 2, "client_vm": "T1", "T4_used": False,
               "sessions": len(checks), "expected_sessions": 40, "checks": checks,
               "all_passed": len(checks) == 40 and all(c["reference_match"] and c["upstream_ok"] and
                                                        c["status"] == "completed" for c in checks)}
    (out / "integration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
