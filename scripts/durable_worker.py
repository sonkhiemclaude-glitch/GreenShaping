"""Small POSIX persistent queue for approved, bounded remote preparation jobs.

No network listener, credentials or remote controller required. A per-attempt
wrapper records completion even when the queue supervisor restarts. Interrupted
safe jobs retry in NEW directories; measurement jobs require manual review.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    # Persist the directory entry as well as the file contents before acknowledging.
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def alive(pid, token):
    # Inspect only the recorded process, and require its attempt path to match.
    p = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True)
    return p.returncode == 0 and token in p.stdout


def execute(spec_path):
    spec_path = Path(spec_path).resolve()
    attempt = spec_path.parent
    with (attempt / "execute.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = read(spec_path)
        stopped = False

        def stop(sig, frame):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        started = time.time()
        with (attempt / "stdout.log").open("ab") as stdout, (attempt / "stderr.log").open("ab") as stderr:
            child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=dict(os.environ, **spec.get("env", {})),
                                     stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            timed_out = False
            while child.poll() is None:
                write(attempt / "heartbeat.json", {"epoch": time.time(), "wrapper_pid": os.getpid(), "child_pid": child.pid})
                timed_out = time.time() - started > spec["timeout_s"]
                if stopped or timed_out:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(.5)
            write(attempt / "result.json", {"returncode": child.returncode, "timed_out": timed_out,
                                            "interrupted": stopped, "start_epoch": started, "end_epoch": time.time()})


def tick(root, config, state):
    """One checkpoint transition, independently testable without launching a daemon."""
    jobs = state.setdefault("jobs", {})
    for job in config["jobs"]:
        jid = job["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", jid):
            raise ValueError("unsafe job id")
        item = jobs.setdefault(jid, {"status": "pending", "attempts": []})
        if item["status"] in ("completed", "completed_with_findings", "failed", "needs_review", "blocked"):
            continue
        if job.get("blocked_reason"):
            item.update(status="blocked", reason=job["blocked_reason"])
            continue
        current = item["attempts"][-1] if item["attempts"] else None
        if current:
            attempt = root / current["path"]
            heartbeat = read(attempt / "heartbeat.json", {})
            if not current["pid"] and heartbeat.get("wrapper_pid"):
                current["pid"] = heartbeat["wrapper_pid"]
            result = read(attempt / "result.json")
            if result:
                current["result"] = result
                if result["returncode"] in job.get("accepted_exit_codes", [0]) and not result["interrupted"] and not result["timed_out"]:
                    item["status"] = "completed" if result["returncode"] == 0 else "completed_with_findings"
                    continue
                if not job.get("safe_to_retry", False):
                    item.update(status="needs_review", reason="failed/interrupted acquisition; no automatic replay")
                    continue
            elif alive(current["pid"], str(attempt / "spec.json")):
                item["status"] = "running"
                return
            elif heartbeat.get("child_pid") and alive(heartbeat["child_pid"], str(attempt)):
                item.update(status="needs_review", reason="orphaned job still alive; no duplicate launch")
                continue
            elif time.time() - current["started_epoch"] < 5:
                return
            elif not job.get("safe_to_retry", False):
                item.update(status="needs_review", reason="completion uncertain after supervisor/host restart")
                continue
            if len(item["attempts"]) >= job.get("max_attempts", 1):
                item["status"] = "failed"
                continue
        if time.time() > config["deadline_epoch"]:
            item.update(status="blocked", reason="approved overnight start window expired")
            continue
        attempt = root / "jobs" / jid / f"attempt_{len(item['attempts']) + 1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        argv = [v.replace("$ATTEMPT", str(attempt)).replace("$ROOT", str(root)) for v in job["argv"]]
        spec = {"argv": argv, "cwd": job.get("cwd", str(root)), "env": job.get("env", {}), "timeout_s": job["timeout_s"]}
        write(attempt / "spec.json", spec)
        current = {"path": str(attempt.relative_to(root)), "pid": 0, "started_epoch": time.time()}
        item["attempts"].append(current)
        item["status"] = "running"
        # Commit ownership before launch; a restart can recover PID from heartbeat.
        write(root / "state.json", state)
        with (attempt / "wrapper.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--execute", str(attempt / "spec.json")],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        current["pid"] = child.pid
        write(root / "state.json", state)
        return


def supervise(root, once=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / "state.json", {"created_epoch": time.time(), "jobs": {}})
        config = read(root / "queue.json")
        while True:
            # Reap wrappers started in this process, avoiding zombies during long queues.
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            tick(root, config, state)
            state.update(heartbeat_epoch=time.time(), supervisor_pid=os.getpid())
            working = any(j["status"] in ("pending", "running") for j in state["jobs"].values())
            state["queue_status"] = "working" if working else "idle_complete"
            write(root / "state.json", state)
            if once or (not working and config.get("exit_when_idle", True)):
                return
            time.sleep(config.get("poll_s", 10))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    if a.execute:
        execute(a.execute)
    elif a.root:
        supervise(a.root, a.once)
    else:
        p.error("--root or --execute is required")


if __name__ == "__main__":
    main()
