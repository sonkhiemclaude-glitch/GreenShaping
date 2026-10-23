from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parents[1] / "scripts/durable_worker.py"


def setup(root, code, **kw):
    config = {"deadline_epoch": time.time() + 60, "jobs": [{"id": "probe", "argv": [sys.executable, "-c", code],
               "timeout_s": 10, "safe_to_retry": True, "max_attempts": 2, **kw}]}
    (root / "queue.json").write_text(json.dumps(config))


def tick(root):
    subprocess.run([sys.executable, str(WORKER), "--root", str(root), "--once"], check=True)
    return json.loads((root / "state.json").read_text())["jobs"]["probe"]


def finish(root):
    for _ in range(60):
        item = tick(root)
        if item["status"] not in ("pending", "running"):
            return item
        time.sleep(.1)
    raise AssertionError("job did not terminate")


def test_restart_reattaches_and_does_not_repeat_completed_job(tmp_path):
    marker = tmp_path / "calls"
    setup(tmp_path, f"import time;from pathlib import Path;time.sleep(1);p=Path({str(marker)!r});p.write_text('one')")
    first = tick(tmp_path)
    again = tick(tmp_path)  # Different supervisor process, same live job.
    assert first["attempts"][0]["pid"] == again["attempts"][0]["pid"]
    assert len(finish(tmp_path)["attempts"]) == 1
    assert tick(tmp_path)["status"] == "completed"
    assert marker.read_text() == "one"


def test_retry_preserves_failed_attempt_and_creates_fresh_directory(tmp_path):
    setup(tmp_path, "from pathlib import Path;import sys;p=Path('$ROOT/counter');n=int(p.read_text()) if p.exists() else 0;"
                   "p.write_text(str(n+1));Path('$ATTEMPT/artifact').write_text(str(n));sys.exit(1 if n==0 else 0)")
    item = finish(tmp_path)
    assert item["status"] == "completed" and len(item["attempts"]) == 2
    assert (tmp_path / "jobs/probe/attempt_001/artifact").read_text() == "0"
    assert (tmp_path / "jobs/probe/attempt_002/artifact").read_text() == "1"


def test_uncertain_measurement_is_not_automatically_replayed(tmp_path):
    setup(tmp_path, "raise RuntimeError('must not launch')", safe_to_retry=False)
    (tmp_path / "state.json").write_text(json.dumps({"jobs": {"probe": {"status": "running", "attempts": [
        {"path": "jobs/probe/attempt_001", "pid": 99999999, "started_epoch": time.time() - 60}]}}}))
    assert tick(tmp_path)["status"] == "needs_review"
    assert not (tmp_path / "jobs/probe/attempt_002").exists()


def test_timeout_is_bounded_and_quality_exit_is_not_retried(tmp_path):
    setup(tmp_path, "import time;time.sleep(20)", timeout_s=.2, max_attempts=1)
    item = finish(tmp_path)
    assert item["status"] == "failed" and item["attempts"][0]["result"]["timed_out"]
    other = tmp_path / "quality"
    other.mkdir()
    setup(other, "raise SystemExit(2)", accepted_exit_codes=[0, 2])
    item = finish(other)
    assert item["status"] == "completed_with_findings" and len(item["attempts"]) == 1


def test_default_worker_exits_when_queue_has_finished(tmp_path):
    setup(tmp_path, "pass")
    config = json.loads((tmp_path / "queue.json").read_text())
    config["poll_s"] = .05
    (tmp_path / "queue.json").write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(WORKER), "--root", str(tmp_path)], timeout=5)
    assert result.returncode == 0
    state = json.loads((tmp_path / "state.json").read_text())
    assert state['queue_status'] == 'idle_complete'
    assert state['jobs']['probe']['status'] == 'completed'
