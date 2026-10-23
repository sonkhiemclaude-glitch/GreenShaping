"""Summarize retained iperf JSON and passive CPU/ENA counters; preserve failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ena(text):
    return {k: int(v) for k, v in re.findall(r"^\s*([\w]+):\s*(\d+)\s*$", text, re.M)}


def cpu(text):
    return {p[0]: list(map(int, p[1:9])) for line in text.splitlines() if (p := line.split()) and re.fullmatch(r"cpu\d*", p[0])}


def summarize(folder):
    rows = json.loads((folder / "summary.json").read_text())
    details = []
    for row in rows:
        obj = json.loads((folder / (row["id"] + ".json")).read_text())
        end = obj["iperf"].get("end", {})
        row["receiver_measurement_seconds"] = end.get("sum_received", {}).get("seconds")
        row["sender_measurement_seconds"] = end.get("sum_sent", {}).get("seconds")
        row["receiver_bytes"] = end.get("sum_received", {}).get("bytes")
        row["sender_bytes"] = end.get("sum_sent", {}).get("bytes")
        stats = {}
        for node in obj["before"]:
            a, b = obj["before"][node], obj["after"][node]
            ea, eb = ena(a["ena"]), ena(b["ena"])
            ca, cb = cpu(a["proc_stat"]), cpu(b["proc_stat"])
            busy = {}
            for core in ca:
                delta = [y - x for x, y in zip(ca[core], cb[core], strict=True)]
                total = sum(delta)
                busy[core] = 100 * (total - delta[3] - delta[4]) / total if total else 0
            stats[node] = dict(
                ena_allowance_deltas={k: eb[k] - v for k, v in ea.items() if "allowance" in k and k in eb},
                cpu_busy_percent=busy,
                window_seconds=(b["time_ns"] - a["time_ns"]) / 1e9,
            )
        details.append(dict(**row, metrics=stats))
    with (folder / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in details:
        groups[row["label"]].append(row)
    table = []
    for label, items in groups.items():
        values = [x["receiver_bps"] / 1e9 for x in items if x["receiver_bps"] is not None and not x["error"] and x["returncode"] == 0]
        table.append(
            dict(
                label=label,
                n=len(items),
                successful=len(values),
                median_Gbps=statistics.median(values) if values else None,
                min_Gbps=min(values) if values else None,
                max_Gbps=max(values) if values else None,
                passes_10Gbps=sum(x["reached_10Gbps"] for x in items),
            )
        )
    result = dict(
        target_Gbps=10, metric="iperf3 receiver average; can include backlog drain and warm-up carryover", groups=table, trials=details
    )
    (folder / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, reverse in zip(axes, [False, True], strict=True):
        for mode in ("no_shaping", "shaping"):
            points = defaultdict(list)
            for row in rows:
                if (
                    row["label"].startswith(mode + "_")
                    and "compatibility" not in row["label"]
                    and row["reverse"] == reverse
                    and row["receiver_bps"] is not None
                    and row["error"] is None
                    and row["returncode"] == 0
                ):
                    points[row["offered_aggregate_bps"] / 1e9].append(row["receiver_bps"] / 1e9)
            xs = sorted(points)
            ax.errorbar(
                xs,
                [statistics.median(points[x]) for x in xs],
                yerr=[
                    [statistics.median(points[x]) - min(points[x]) for x in xs],
                    [max(points[x]) - statistics.median(points[x]) for x in xs],
                ],
                marker="o",
                capsize=3,
                label=mode.replace("_", " "),
            )
        for index, mode in enumerate(("no_shaping", "shaping")):
            targets = [
                row
                for row in rows
                if row["label"].startswith(mode + "_") and row["reverse"] == reverse and row["offered_aggregate_bps"] == 10_000_000_000
            ]
            failed = sum(bool(row["error"] or row["returncode"]) for row in targets)
            if failed:
                ax.text(
                    0.98,
                    0.04 + 0.06 * index,
                    f"{mode.replace('_', ' ')}: {failed}/{len(targets)} failed at 10 Gbps",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=8,
                )
        ax.plot([0, 10], [0, 10], "--", color="gray", label="offered = received")
        ax.set(
            xlabel="Offered aggregate TCP rate (Gbps)",
            title="T4 → T1 (reverse)" if reverse else "T1 → T4 (forward)",
            xlim=(0, 10.3),
            ylim=(0, 10.5),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("iperf3 receiver average incl. drain (Gbps)")
    fig.suptitle("NetShaper on four m6i.2xlarge VMs — short bursts; receiver includes drain")
    fig.tight_layout()
    fig.savefig(folder / "throughput.png", dpi=180)
    fig.savefig(folder / "throughput.pdf")
    plt.close(fig)
    print(json.dumps(table, indent=2))
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in folder.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
