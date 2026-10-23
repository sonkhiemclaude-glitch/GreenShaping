"""Read-only checks before training; --paper enforces the audited paper prerequisites.

Run from audit_and_code (paths follow the existing experiment runner's convention).
This checks local files and configuration; it never contacts hosts or starts training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from greenshaping.controller.grid import build_grid
from greenshaping.data.schemas import QUERY_FAMILIES
from greenshaping.experiments.common import context_from_config
from greenshaping.utils.io import load_yaml, write_json


def check(cfg: dict[str, Any], paper: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "scope": "paper prerequisites" if paper else "offline pipeline prerequisites",
        "python": sys.version.split()[0], "machine": platform.machine(),
        "platform": platform.system(), "errors": [], "warnings": [],
        "tools": {"tcpdump": shutil.which("tcpdump"), "tshark": shutil.which("tshark")},
        "paper_ready": False,
    }
    errors, warnings = report["errors"], report["warnings"]
    if not report["tools"]["tshark"]:
        mac_tshark = Path("/Applications/Wireshark.app/Contents/MacOS/tshark")
        if mac_tshark.is_file():
            report["tools"]["tshark"] = str(mac_tshark)
        else:
            warnings.append("tshark unavailable; needed for exporting live captures, not offline synthetic runs")
    try:
        grid = build_grid(cfg.get("grid", {}))
        report["grid_size_per_N"] = len(grid)
        report["single_primitive_and_none"] = sum(g.n_primitives() <= 1 for g in grid)
        models = cfg.get("attack", {}).get("models", ["rf"])
        if not models or set(models) - {"rf", "cnn", "rf_linkaware", "identity"}:
            errors.append("attack.models contains an unsupported attacker")
        if "cnn" in models and importlib.util.find_spec("torch") is None:
            errors.append("CNN selected but torch is not installed")
        with TemporaryDirectory(prefix="greenshaping-preflight-") as temp:
            ctx = context_from_config(cfg, out_dir=temp)
            report["data_kind"] = ctx.data_kind
            report["sessions"] = len(ctx.sessions)
            report["splits"] = {name: len(getattr(ctx, name)) for name in ("train", "select", "test")}
            report["N"] = ctx.n_values
            report["questions_by_family"] = {
                f: len({s.question_id for s in ctx.sessions if s.query_family == i})
                for i, f in enumerate(QUERY_FAMILIES)
            }
            report["splits_per_N"] = {
                str(n): {name: sum(s.n_sites == n for s in getattr(ctx, name))
                         for name in ("train", "select", "test")}
                for n in ctx.n_values
            }
            if any(not count for counts in report["splits_per_N"].values() for count in counts.values()):
                errors.append("at least one N has an empty question split")
            empty = [s.sid for s in ctx.sessions if not s.messages]
            if empty:
                errors.append(f"sessions without matched traffic: {empty[:20]}")
            qfamilies: dict[int, set[int]] = {}
            for s in ctx.sessions:
                qfamilies.setdefault(s.question_id, set()).add(s.query_family)
            if any(len(f) > 1 for f in qfamilies.values()):
                errors.append("question_id maps to different families; allocate global question IDs")
            if paper:
                if ctx.data_kind != "real":
                    errors.append("paper runs require real LLM sessions")
                if any(not m.fitted for m in ctx.cost_models.values()):
                    errors.append("paper runs require fitted energy models")
                if set(ctx.n_values) != {5, 10, 20}:
                    errors.append("full E1-E5 protocol requires N = 5, 10, 20")
                if any(v < 40 for v in report["questions_by_family"].values()):
                    errors.append("full protocol requires >= 40 distinct questions per family")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    existing = Path(cfg.get("run", {}).get("output_dir", "outputs/run"))
    if existing.exists() and any(existing.iterdir()):
        warnings.append(f"output directory already contains files: {existing}; use a fresh directory")
    if paper:
        errors.append("ShapedTransport live rounds/compositions and network timing remain unvalidated for the full protocol")
        for g in grid:
            if g.dp and g.dp.sensitivity_mode != "bounded_pair":
                errors.append("paper runs require bounded_pair sensitivity; legacy_volume is rehearsal-only")
                break
        energy = cfg.get("energy", {})
        if not cfg.get("budgets", {}).get("site_J"):
            errors.append("budgets.site_J must be set from measured costs")
        for label, value in (("energy.measured_energy_csv", energy.get("measured_energy_csv")),
                             ("baselines.netshaper_root", cfg.get("baselines", {}).get("netshaper_root"))):
            if not value or not Path(value).exists():
                errors.append(f"missing {label}")
        if energy.get("grid_g_per_kwh") is None or not energy.get("grid_factor_source"):
            errors.append("carbon reporting requires a grid factor and its source")
        if not {"rf", "cnn"} <= set(cfg.get("attack", {}).get("models", [])):
            errors.append("full protocol requires both RF and CNN attackers")
        if not cfg.get("experiments", {}).get("e5", {}).get("e_min_source"):
            errors.append("E5 floor needs justified per-site lower bounds; average cover energy is insufficient")
        llms = cfg.get("testbed", {}).get("llms", {})
        if not llms.get("local") or not llms.get("hosted"):
            errors.append("full protocol needs local and hosted model provenance")
        warnings.append("file checks do not certify TLS schedules, DP adjacency, meter quality or the scientific protocol")
    report["ok"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        report = check(load_yaml(args.config), args.paper)
    except Exception as exc:
        report = {"ok": False, "paper_ready": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    if args.out:
        write_json(report, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
