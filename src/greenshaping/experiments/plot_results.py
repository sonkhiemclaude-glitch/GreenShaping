"""Plots generated ONLY from result CSVs written by the runners (never from hard-coded numbers).

Every figure is titled with the result status read from run_metadata.json; for synthetic
data that is "synthetic smoke-test output, not a paper result". Categorical colours follow a
fixed order validated for colour-vision deficiency; series also differ by marker shape, and
the CSV next to each figure is its table view.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import SYNTHETIC_LABEL  # noqa: E402
from ..metrics.metrics import pareto_front  # noqa: E402
from ..utils.io import read_csv, read_json  # noqa: E402

log = logging.getLogger(__name__)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order (validated)
MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _f(x: str | None) -> float:
    try:
        return float(x) if x not in (None, "") else math.nan
    except ValueError:
        return math.nan


def _status(in_dir: Path) -> str:
    meta = in_dir / "run_metadata.json"
    if not meta.is_file():
        return "provenance unknown (run_metadata.json missing)"
    return str(read_json(meta).get("result_status", SYNTHETIC_LABEL))


def _axes(title: str, status: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"{title}\n[{status}]", fontsize=10, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def plot_e1(in_dir: Path, out_dir: Path) -> Path | None:
    p = in_dir / "e1_leakage.csv"
    if not p.is_file():
        return None
    rows = read_csv(p)
    status = _status(in_dir)
    secrets = ["participation", "size_bucket", "query_family"]
    groups = sorted({(int(r["N"]), r.get("llm", "n/a")) for r in rows})
    if len(groups) > len(SERIES):
        log.warning("E1 plot shows the first %d of %d (N, LLM) groups; see the CSV", len(SERIES), len(groups))
        groups = groups[:len(SERIES)]
    fig, ax = _axes("E1: attacker advantage on unshaped traffic (final-test questions)", status)
    width = 0.8 / max(1, len(groups))
    for j, (n, llm) in enumerate(groups):
        for i, sec in enumerate(secrets):
            r = next((x for x in rows if int(x["N"]) == n and x.get("llm", "n/a") == llm and x["secret"] == sec), None)
            if r is None:
                continue
            v, lo, hi = _f(r["advantage"]), _f(r["advantage_ci_low"]), _f(r["advantage_ci_high"])
            x = i + (j - (len(groups) - 1) / 2) * width
            lab = f"N = {n}" + (f", {llm}" if len({g[1] for g in groups}) > 1 else "")
            ax.bar(x, v, width * 0.9, color=SERIES[j % 4], label=lab if i == 0 else None)
            if not math.isnan(lo):
                ax.errorbar(x, v, yerr=[[max(0, v - lo)], [max(0, hi - v)]], color=INK, capsize=3, lw=1)
    ax.set_xticks(range(len(secrets)), secrets)
    ax.set_ylabel("normalised advantage (acc − chance)/(1 − chance)", color=MUTED, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    out = out_dir / "e1_leakage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


ENERGY_COLS = {
    "modelled": ("energy_modelled_site_max_J", "MODELLED extra energy, most loaded site (J/session, Eq. 3)"),
    "measured": ("energy_measured_site_max_median_J", "MEASURED extra energy, most loaded site (J/session, median of runs)"),
}


def plot_cost_vs_advantage(in_dir: Path, out_dir: Path, name: str, title: str,
                           energy: str = "modelled", keep: str | None = None) -> Path | None:
    """Advantage vs extra energy, one panel per N. Modelled and measured energy are never mixed
    in one figure: the measured figure (`<name>_measured.png`) shows only measured rows (minor 21)."""
    p = in_dir / f"{name}.csv"
    if not p.is_file():
        return None
    col, label = ENERGY_COLS[energy]
    rows = [r for r in read_csv(p) if keep is None or r.get("experiment") == keep]
    rows = [r for r in rows if not math.isnan(_f(r.get(col))) and not math.isnan(_f(r.get("adv_max")))]
    if not rows:
        return None
    status = _status(in_dir)
    ns = sorted({r.get("N", "") for r in rows}, key=lambda x: int(x) if x else 0)
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns) + 1.8, 4.2), facecolor=SURFACE, squeeze=False)
    fig.suptitle(f"{title}\n[{status}]", fontsize=10, color=INK, x=0.01, ha="left")
    kinds = [("padding", "pad("), ("rounds", "rounds("), ("DP release", "dp("), ("combinations / baselines", "")]
    for ax, n in zip(axes[0], ns, strict=False):
        sub = [r for r in rows if r.get("N", "") == n]
        ax.set_facecolor(SURFACE)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for k, (kname, prefix) in enumerate(kinds):
            pts = [r for r in sub if (prefix and r["config"].startswith(prefix) and "+" not in r["config"])
                   or (not prefix and ("+" in r["config"] or not r["config"].startswith(("pad(", "rounds(", "dp("))))]
            if pts:
                ax.scatter([_f(r.get(col)) for r in pts], [_f(r.get("adv_max")) for r in pts], s=36, color=SERIES[k],
                           marker=MARKERS[k], label=kname, edgecolors=SURFACE, linewidths=1.0)
        valid = [{"_x": _f(r.get(col)), "_y": _f(r.get("adv_max"))} for r in sub]
        front = sorted(pareto_front(valid, ("_x", "_y")), key=lambda r: r["_x"])
        if len(front) > 1:
            ax.plot([r["_x"] for r in front], [r["_y"] for r in front], color=MUTED, lw=1, ls="--", label="Pareto front")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel(label, color=MUTED, fontsize=7)
        ax.set_title(f"N = {n}" if n else "", fontsize=8, color=MUTED)
    axes[0][0].set_ylabel("max normalised advantage over 3 secrets", color=MUTED, fontsize=8)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    out = out_dir / (f"{name}.png" if energy == "modelled" else f"{name}_measured.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_e5(in_dir: Path, out_dir: Path) -> Path | None:
    p = in_dir / "e5_scaling.csv"
    if not p.is_file():
        return None
    rows = read_csv(p)
    status = _status(in_dir)
    mixes = list(dict.fromkeys(r["query_mix"] for r in rows))
    fig, axes = plt.subplots(1, len(mixes), figsize=(4.2 * len(mixes), 4.2), facecolor=SURFACE, squeeze=False)
    fig.suptitle(f"E5: modelled extra energy of all sites vs Prop. 2(b) network floor\n[{status}]",
                 fontsize=10, color=INK, x=0.01, ha="left")
    cfgs = sorted({r["config"] for r in rows})
    for ax, mix in zip(axes[0], mixes, strict=False):
        ax.set_facecolor(SURFACE)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for k, c in enumerate(cfgs[:4]):
            pts = sorted([r for r in rows if r["config"] == c and r["query_mix"] == mix], key=lambda r: int(r["N"]))
            ax.plot([int(r["N"]) for r in pts], [_f(r["energy_modelled_total_sites_J"]) for r in pts],
                    color=SERIES[k], marker=MARKERS[k], lw=2, label=c)
        fl = sorted({(int(r["N"]), _f(r["prop2_network_floor_modelled_J"])) for r in rows if r["query_mix"] == mix})
        fl = [(x, y) for x, y in fl if math.isfinite(y)]
        if fl:
            ax.plot([x for x, _ in fl], [y for _, y in fl], color=MUTED, lw=1, ls="--",
                    label=f"floor, alpha = {rows[0]['alpha']}")
        ns = sorted({int(r["N"]) for r in rows})
        ax.set_xticks(ns)
        ax.set_title(mix.replace("+", ", "), fontsize=8, color=MUTED)
        ax.set_xlabel("number of sites N", color=MUTED, fontsize=8)
    axes[0][0].set_ylabel("MODELLED extra energy, all sites (J/session)", color=MUTED, fontsize=8)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    out = out_dir / "e5_scaling.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def make_all_plots(in_dir: str | Path, out_dir: str | Path) -> list[Path]:
    i, o = Path(in_dir), Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    made = [plot_e1(i, o)]
    for energy in ("modelled", "measured"):
        made += [plot_cost_vs_advantage(i, o, "e2_primitives", "E2: single primitives, advantage vs extra energy", energy),
                 plot_cost_vs_advantage(i, o, "e3_profiles", "E3: all grid configurations (selection questions)", energy),
                 plot_cost_vs_advantage(i, o, "e3_final_comparison", "E3: selected, best single and baselines (final test)", energy),
                 plot_cost_vs_advantage(i, o, "e4_platform", "E4: single primitives per node class", energy, keep="E4-E2")]
    made.append(plot_e5(i, o))
    out = [m for m in made if m is not None]
    if not out:
        raise FileNotFoundError(f"no result CSVs found in {i}")
    log.info("wrote %d plots to %s", len(out), o)
    return out
