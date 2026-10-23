"""Lightweight, 1 Hz interface counters; includes headers and other interface traffic."""

import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
with out.open("a") as handle:
    for _ in range(900):
        fields = next(
            line.split(":", 1)[1].split()
            for line in Path("/proc/net/dev").read_text().splitlines()
            if line.split(":", 1)[0].strip() == "ens5"
        )
        handle.write(json.dumps(dict(time_ns=time.time_ns(), rx_bytes=int(fields[0]), tx_bytes=int(fields[8]))) + "\n")
        handle.flush()
        time.sleep(1)
