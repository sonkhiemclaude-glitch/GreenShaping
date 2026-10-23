"""Bounded, read-only EC2 samples. Reservation blocks timing even on an idle host.

An idle sample does not certify exclusive use or guarantee future host availability.
No command arguments, environment variables or private application files are read.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import ec2_runner as r

PROBE = '''import json,time,subprocess,os
from pathlib import Path
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=10)
 return {'returncode':p.returncode,'stdout':p.stdout[-16000:],'stderr':p.stderr[-1000:]}
def sample():
 m={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
 return {'cpu':[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]],
         'net':Path('/proc/net/dev').read_text(),'memory_available_bytes':m['MemAvailable'],
         'memory_total_bytes':m['MemTotal'],'load':os.getloadavg(),'epoch':time.time()}
samples=[]
for i in range(SAMPLE_COUNT):
 samples.append(sample())
 if i+1<SAMPLE_COUNT:time.sleep(INTERVAL)
print(json.dumps({'samples':samples,'cpu_count':os.cpu_count(),
 'processes':cmd(['ps','-eo','pid,ppid,user,comm,pcpu,pmem,rss,etime','--sort=-pcpu']),
 'containers':cmd(['sudo','-n','docker','ps','--format','{{.Names}} {{.Status}}']),
 'energy_paths':[str(p) for p in Path('/sys/class/powercap').rglob('energy_uj')]}))
'''


def assess(data: dict, reserved: bool = False) -> dict:
    samples = data.get("samples", [])
    blockers = []
    if reserved:
        blockers.append("host reserved for another workload; idle samples cannot override reservation")
    if data.get("error") or len(samples) < 2:
        return {"timing_candidate": False, "blockers": blockers + ["missing valid resource samples"]}
    busy, steal = [], []
    for a, b in zip(samples, samples[1:], strict=False):
        delta = [y - x for x, y in zip(a["cpu"], b["cpu"], strict=True)]
        total = sum(delta)
        if total <= 0 or min(delta) < 0:
            blockers.append("invalid CPU interval")
            continue
        busy.append(100 * (total - delta[3] - delta[4]) / total)
        steal.append(100 * delta[7] / total)
    if not busy:
        blockers.append("no valid CPU interval")
    elif max(busy) > 10:
        blockers.append("CPU busy exceeds 10% in at least one interval")
    if steal and max(steal) > 1:
        blockers.append("CPU steal exceeds 1%")
    if max(s["load"][0] for s in samples) / data["cpu_count"] > .25:
        blockers.append("one-minute load exceeds 0.25 per vCPU")
    if min(s["memory_available_bytes"] / s["memory_total_bytes"] for s in samples) < .5:
        blockers.append("less than 50% memory available")
    containers = data.get("containers", {})
    if containers.get("returncode") != 0:
        blockers.append("container inventory unavailable")
    elif containers.get("stdout", "").strip():
        blockers.append("running containers require explicit workload ownership review")
    return {"timing_candidate": not blockers, "blockers": blockers, "cpu_busy_pct": busy,
            "steal_pct": steal, "energy_sensor_paths": data.get("energy_paths", []),
            "exclusive_use_verified": False,
            "limitations": "CPU/load screen only; review network/disk deltas and competing jobs before and during acquisition"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reserved", nargs="*", choices=list(r.INVENTORY["nodes"]), default=["T4"])
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--interval", type=float, default=5.)
    a = p.parse_args()
    if not 2 <= a.samples <= 13 or not 1 <= a.interval <= 5:
        p.error("bounded sampling requires 2..13 samples and 1..5 seconds per interval")
    a.out.mkdir(parents=True, exist_ok=False)

    def probe(node):
        source = PROBE.replace("SAMPLE_COUNT", str(a.samples)).replace("INTERVAL", str(a.interval))
        try:
            data = json.loads(r.python(node, source, timeout=90))
        except Exception as exc:
            data = {"error": str(exc)}
        (a.out / f"{node}.json").write_text(json.dumps(data, indent=2))
        return node, assess(data, node in a.reserved)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        nodes = dict(pool.map(probe, r.INVENTORY["nodes"]))
    result = {"nodes": nodes, "reserved": a.reserved, "large_run_authorized_by_this_check": False,
              "energy_ready": False, "reason": "Sampling is not exclusive-use verification or calibrated physical-energy evidence"}
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if all(nodes[n]["timing_candidate"] for n in nodes if n not in a.reserved) else 2


if __name__ == "__main__":
    raise SystemExit(main())
