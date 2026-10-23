"""Prepare a fresh Covertype reference bank, without inspecting final-test outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from greenshaping.testbed.public_workload import build_public_workload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sites", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261027)
    a = p.parse_args()
    print(json.dumps(build_public_workload(a.archive, a.out, a.sites, a.seed), indent=2))


if __name__ == "__main__":
    main()
