"""Plots from result CSVs: python scripts/make_plots.py --input outputs/<run> --output outputs/<run>/plots"""
import sys

from greenshaping.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["plot", *sys.argv[1:]]))
