"""Read cgroup memory/CPU counters for this run's Docker containers, bounded to 900s."""

import json
import subprocess
import sys
import time
from pathlib import Path

prefix = sys.argv[1]
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    output = subprocess.check_output(
        ["sudo", "-n", "docker", "ps", "--no-trunc", "--filter", "name=" + prefix, "--format", "{{.ID}} {{.Names}}"], text=True
    )
    for line in output.splitlines():
        cid, name = line.split(maxsplit=1)
        if not name.startswith(prefix):
            continue
        folder = Path("/sys/fs/cgroup/system.slice") / ("docker-" + cid + ".scope")
        row = dict(time_ns=time.time_ns(), container=name, container_id=cid)
        for field in ("memory.current", "memory.peak", "memory.events", "cpu.stat"):
            try:
                row[field] = (folder / field).read_text().strip()
            except OSError as exc:
                row[field] = repr(exc)
        print(json.dumps(row), flush=True)
    time.sleep(1)
