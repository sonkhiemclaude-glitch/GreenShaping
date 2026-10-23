"""Command-line interface: `python -m greenshaping.cli --help`.

Subcommands never download anything and never execute shell commands from configs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from . import SYNTHETIC_LABEL, __version__
from .utils.io import load_yaml, write_json
from .utils.logging import setup_logging

log = logging.getLogger("greenshaping")


def _cmd_run(a: argparse.Namespace) -> int:
    from .experiments.runner import run_from_config

    cfg = load_yaml(a.config)
    paths = run_from_config(cfg, a.stage, a.output)
    for p in paths:
        print(p)
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    from .experiments.plot_results import make_all_plots
    from .experiments.runner import run_from_config

    cfg = load_yaml(a.config)
    if cfg.get("data", {}).get("source", "synthetic") != "synthetic":
        print("demo only runs on synthetic data; use `run` for real data", file=sys.stderr)
        return 2
    out = a.output or cfg.get("run", {}).get("output_dir", "outputs/synthetic_demo")
    paths = run_from_config(cfg, "all", out)
    plots = make_all_plots(out, Path(out) / "plots")
    print(f"[{SYNTHETIC_LABEL}]")
    for p in paths + plots:
        print(p)
    return 0


def _cmd_plot(a: argparse.Namespace) -> int:
    from .experiments.plot_results import make_all_plots

    for p in make_all_plots(a.input, a.output):
        print(p)
    return 0


def _cmd_fit_energy(a: argparse.Namespace) -> int:
    from .energy.fit import Benchmark, fit_cost_model, validate_cost_model
    from .utils.provenance import file_sha256

    rows = []
    with open(a.benchmarks, newline="") as f:
        for r in csv.DictReader(f):
            if r["node_class"] != a.node_class:
                continue
            times = tuple(float(x) for x in r["event_times"].split(";") if x)
            rows.append(Benchmark(int(r["bytes_tx"]), int(r["bytes_rx"]), times, float(r["measured_joules"]), r.get("run", "")))
    res = fit_cost_model(a.node_class, rows)
    validation = validate_cost_model(rows, a.node_class)
    write_json(res.model.to_dict(), a.out)
    write_json({"node_class": a.node_class, "n_rows": len(rows), "rmse_J": res.rmse,
                "relative_rmse": res.relative_rmse, "residuals_J": list(res.residuals),
                "benchmark_sha256": file_sha256(a.benchmarks), "validation": validation},
               Path(a.out).with_suffix(".fit_report.json"))
    print(json.dumps(res.model.to_dict()))
    return 0


def _cmd_rapl(a: argparse.Namespace) -> int:
    from .energy.rapl import DEFAULT_POWERCAP, discover_zones, snapshot

    zones = discover_zones(Path(a.powercap) if a.powercap else DEFAULT_POWERCAP)
    if not zones:
        print("no intel-rapl zones found (RAPL needs x86 Linux; not available on Apple silicon)", file=sys.stderr)
        return 1
    print(json.dumps({"zones": [z.name for z in zones], "energy_uj": snapshot(zones)}))
    return 0


def _cmd_bounds(a: argparse.Namespace) -> int:
    from .analysis import bounds as b

    out = {
        "K": b.ticks_per_session(a.horizon, a.interval),
        "eq4_expected_padding_bytes_per_linkdir": b.expected_dp_padding_bytes(a.horizon, a.interval, a.delta, a.eps_dir),
        "prop3_eps_total": b.dp_epsilon_total(a.eps_dir, a.d),
        "prop3_accuracy_bound_uniform_prior": b.dp_accuracy_bound(b.dp_epsilon_total(a.eps_dir, a.d), 0.5),
        "note": "analytical values from Sec. 5, not measurements",
    }
    print(json.dumps(out, indent=1))
    return 0


def _cmd_validate_data(a: argparse.Namespace) -> int:
    from .data.adapters import SessionDirectoryAdapter

    problems = SessionDirectoryAdapter(a.root).check_layout()
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    sessions = SessionDirectoryAdapter(a.root, a.gap_s).load()
    empty = [s.sid for s in sessions if not s.messages]
    report = {"sessions": len(sessions), "questions": len({s.question_id for s in sessions}),
              "n_sites": sorted({s.n_sites for s in sessions}),
              "data_kinds": sorted({str(s.meta.get("data_kind")) for s in sessions}),
              "llms": sorted({str(s.meta.get("llm")) for s in sessions}),
              "sessions_without_messages": empty,
              "capture_rows_not_matching_manifest_sites": sum(int(s.meta.get("unmatched_rows", 0)) for s in sessions),
              "participating_sites_with_clipped_size_bucket": sum(int(s.meta.get("clipped_size_buckets", 0)) for s in sessions)}
    print(json.dumps(report, indent=1))
    if empty:
        print("PROBLEM: sessions with no messages (manifest IPs or ports do not match the capture?)", file=sys.stderr)
        return 1
    return 0


def _cmd_export_synthetic(a: argparse.Namespace) -> int:
    from .data.adapters import export_sessions
    from .data.synthetic import SyntheticConfig, generate_sessions

    cfg = load_yaml(a.config).get("data", {}).get("synthetic", {})
    export_sessions(generate_sessions(SyntheticConfig(**cfg)), a.out)
    print(a.out)
    return 0


def _cmd_split_capture(a: argparse.Namespace) -> int:
    from .data.adapters import split_capture_by_sessions

    counts = split_capture_by_sessions(a.capture, Path(a.root) / "manifests", Path(a.root) / "captures")
    print(json.dumps({"rows_per_session": counts}))
    return 0


def _cmd_capture_commands(a: argparse.Namespace) -> int:
    from .testbed.capture_commands import tcpdump_command, tshark_export_command

    print(json.dumps({"tcpdump": tcpdump_command(a.interface, a.pcap, a.ports),
                      "tshark_export": tshark_export_command(a.pcap)}, indent=1))
    return 0


def _cmd_site_server(a: argparse.Namespace) -> int:  # pragma: no cover - blocking server
    from .testbed.site_server import SiteServer

    srv = SiteServer(a.table, a.host, a.port, a.certfile, a.keyfile).start()
    print(f"site server on {srv.address}; Ctrl-C to stop")
    try:
        srv._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        srv.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="greenshaping", description=(
        "Green Shaping reference implementation. No paper-level experimental results are claimed by this "
        "repository unless real benchmark outputs are generated and their provenance is recorded."))
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=argparse.SUPPRESS, help="DEBUG, INFO, WARNING (also accepted before the subcommand)")

    s = sub.add_parser("run", parents=[common], help="run a stage (e1, e2, calibrate, evaluate, e3, e4, e5, all) from a config")
    s.add_argument("--config", required=True)
    s.add_argument("--stage", default="all")
    s.add_argument("--output", default=None)
    s.set_defaults(func=_cmd_run)

    s = sub.add_parser("demo", parents=[common], help="synthetic smoke-test demo (all stages + plots)")
    s.add_argument("--config", required=True)
    s.add_argument("--output", default=None)
    s.set_defaults(func=_cmd_demo)

    s = sub.add_parser("plot", parents=[common], help="make plots from result CSVs")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=_cmd_plot)

    s = sub.add_parser("fit-energy", parents=[common], help="fit Eq. (3) coefficients from microbenchmark CSV")
    s.add_argument("--benchmarks", required=True)
    s.add_argument("--node-class", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_fit_energy)

    s = sub.add_parser("rapl-snapshot", parents=[common], help="read RAPL counters (x86 Linux only)")
    s.add_argument("--powercap", default=None)
    s.set_defaults(func=_cmd_rapl)

    s = sub.add_parser("bounds", parents=[common], help="analytical quantities of Sec. 5 for given parameters")
    s.add_argument("--horizon", type=float, default=20.0)
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--delta", type=float, default=65536)
    s.add_argument("--eps-dir", type=float, default=1.0)
    s.add_argument("--d", type=int, default=2, help="number of differing link-directions")
    s.set_defaults(func=_cmd_bounds)

    s = sub.add_parser("validate-data", parents=[common], help="check a real run directory (manifests/, captures/)")
    s.add_argument("--root", required=True)
    s.add_argument("--gap-s", type=float, default=0.010)
    s.set_defaults(func=_cmd_validate_data)

    s = sub.add_parser("export-synthetic", parents=[common], help="write synthetic sessions to JSONL (smoke tests)")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_export_synthetic)

    s = sub.add_parser("capture-commands", parents=[common], help="print tcpdump/tshark commands (not executed)")
    s.add_argument("--interface", required=True)
    s.add_argument("--pcap", required=True)
    s.add_argument("--ports", type=int, nargs="+", required=True)
    s.set_defaults(func=_cmd_capture_commands)

    s = sub.add_parser("split-capture", parents=[common], help="split one tshark CSV into per-session captures")
    s.add_argument("--capture", required=True)
    s.add_argument("--root", required=True, help="run directory with manifests/")
    s.set_defaults(func=_cmd_split_capture)

    s = sub.add_parser("site-server", parents=[common], help="start one site tool server (testbed)")
    s.add_argument("--table", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--certfile", default=None)
    s.add_argument("--keyfile", default=None)
    s.set_defaults(func=_cmd_site_server)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
