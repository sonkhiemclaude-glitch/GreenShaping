"""Measure RAPL energy of one window on an x86 Linux node (Sec. 6.4). Not usable on Apple silicon.

Usage: python scripts/rapl_session_energy.py --seconds 20 --idle-json idle.json --out session_energy.json
Reads counters before and after `--seconds`, corrects wrap-around and subtracts idle power x
duration per zone. Only `package-*` and `dram` zones are summed (core/uncore are inside the
package). The output records the window's start and end epoch times (same clock as the
manifests' `start_time` when the hosts are NTP-synchronised), so each window can be matched
to the session(s) it covers; start it just before the driver submits a session.
"""
from __future__ import annotations

import argparse
import json
import time

from greenshaping.energy.rapl import discover_zones, package_and_dram, session_energy_joules, snapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--idle-json", default=None, help='{"<zone>@<dir>": idle_watts}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zones = package_and_dram(discover_zones())
    if not zones:
        raise SystemExit("no RAPL zones (x86 Linux with intel-rapl powercap required)")
    idle = json.load(open(a.idle_json)) if a.idle_json else {}
    s0, t0 = snapshot(zones), time.time()
    time.sleep(a.seconds)
    s1, t1 = snapshot(zones), time.time()
    res = session_energy_joules(s0, s1, zones, idle, t1 - t0)
    json.dump({"start_epoch": t0, "end_epoch": t1, "duration_s": t1 - t0, "extra_joules_by_zone": res,
               "extra_joules_package_dram": sum(res.values()), "idle_subtracted": bool(idle)},
              open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
