"""Synthetic smoke-test demo: all stages E1-E5 on synthetic sessions, then plots.

Usage: python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml
Output: outputs/synthetic_demo/ ("synthetic smoke-test output, not a paper result").
"""
import sys

from greenshaping.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--config" not in argv:
        argv = ["--config", "configs/synthetic_demo.yaml", *argv]
    raise SystemExit(main(["demo", *argv]))
