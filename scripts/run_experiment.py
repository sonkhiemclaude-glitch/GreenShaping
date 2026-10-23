"""Run one stage from a config.

Usage:
  python scripts/run_experiment.py --config configs/experiment_template.yaml --stage calibrate
  python scripts/run_experiment.py --config configs/experiment_template.yaml --stage evaluate
Stages: e1 (leakage), e2 (primitives / ablation), calibrate (Algorithm 1 only),
evaluate (E1 + Algorithm 1 + final-test + baselines), e3, e4, e5, all.
"""
import sys

from greenshaping.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
