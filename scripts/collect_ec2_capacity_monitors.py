"""Stop only the recorded run monitors and retrieve their telemetry."""

import json
import shutil
from pathlib import Path

import ec2_runner as r

folder = r.EVIDENCE / "capacity_recovery_01"
remote = r.INVENTORY["remote_root"] + "/runs/capacity_recovery_01"
for pidfile, script, trace, nodes in (
    ("monitor_pids.json", "ec2_container_monitor.py", "memory_monitor.jsonl", ("T2", "T3")),
    ("nic_monitor_pids.json", "ec2_nic_monitor.py", "nic_monitor.jsonl", ("T1", "T2", "T3", "T4")),
):
    pids = json.loads((folder / pidfile).read_text())
    for node in nodes:
        pid = pids[node]["pid"]
        result = r.python(
            node,
            f"""import os,signal,json
from pathlib import Path
p=Path('/proc/{pid}/cmdline')
stopped=False
if p.exists():
 c=p.read_bytes()
 if {script.encode()!r} in c and b'greenshaping-netshaper-20261020' in c:
  try:os.killpg({pid},signal.SIGTERM);stopped=True
  except ProcessLookupError:pass
print(json.dumps({{'pid':{pid},'stopped':stopped,'script':{script!r}}}))
""",
        )
        (folder / f"{node}_{script}_cleanup.json").write_text(result)
        r.copy(node, folder / f"{node}_{trace}", remote + "/" + trace, get=True)
    shutil.copyfile(Path(__file__).parent / script, folder / script)
print("All recorded monitors stopped and telemetry retrieved.")
