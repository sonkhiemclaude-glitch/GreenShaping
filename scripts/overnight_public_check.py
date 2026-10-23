"""Rebuild and audit the public engineering bank; never evaluates final-test outputs."""
import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--archive", required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
scripts = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(scripts / "prepare_public_workload.py"), "--archive", a.archive,
                "--out", str(a.out / "workload")], check=True)
subprocess.run([sys.executable, str(scripts / "audit_public_workload.py"), "--workload", str(a.out / "workload"),
                "--out", str(a.out / "audit")], check=True)
