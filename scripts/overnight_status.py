"""Fetch durable queue checkpoints after reconnecting; not required for remote execution."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import ec2_runner as ec2

MAC_ROOT = "/Users/sonha/Projects/GreenShaping/overnight_20261020"
EC2_ROOT = ec2.INVENTORY["remote_root"] + "/overnight_20261020"


def snapshot_source(root):
    return f'''import json,time,subprocess
from pathlib import Path
p=Path({root!r});s=json.loads((p/'state.json').read_text());cfg=json.loads((p/'queue.json').read_text())
for job in cfg['jobs']:
 s['jobs'].setdefault(job['id'],{{'status':'blocked' if job.get('blocked_reason') else 'pending','reason':job.get('blocked_reason')}})
s['root']=str(p);s['heartbeat_age_s']=time.time()-s['heartbeat_epoch']
s['probe_survived_ssh_disconnect']=(p/'jobs/disconnect_probe/attempt_001/survived.txt').exists()
s['higgs_download_bytes']=sum(f.stat().st_size for f in (p/'data/higgs').glob('HIGGS.csv.gz*'))
print(json.dumps(s))
'''


def fetch(node):
    try:
        if node == "MacStudio":
            helper = [sys.executable, str(ec2.ROOT / "scripts/remote_runner.py"), "--known-hosts",
                      str(ec2.ROOT / "outputs/remote_20261020/known_hosts"), "exec", "python3 -", "--timeout", "20"]
            p = subprocess.run(helper, input=snapshot_source(MAC_ROOT), capture_output=True, text=True, timeout=40)
            if p.returncode:
                raise RuntimeError(p.stderr[-1000:])
            data = json.loads(p.stdout)
        else:
            data = json.loads(ec2.python(node, snapshot_source(EC2_ROOT), timeout=35))
        return node, data
    except Exception as exc:
        return node, {"connection_error": str(exc), "remote_job_status": "unknown; connection failure does not stop remote service"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(fetch, ["MacStudio", "T1", "T2", "T3"]))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
