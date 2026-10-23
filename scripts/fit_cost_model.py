"""Fit Eq. (3) coefficients for one node class from measured microbenchmarks (Sec. 6.4).

Usage: python scripts/fit_cost_model.py --benchmarks bench.csv --node-class x86_server --out models/x86_server.json
CSV columns: node_class,run,bytes_tx,bytes_rx,event_times,measured_joules
(event_times: semicolon-separated seconds; measured_joules: idle-subtracted).
"""
import sys

from greenshaping.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fit-energy", *sys.argv[1:]]))
