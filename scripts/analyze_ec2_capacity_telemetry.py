"""Analyze 1 Hz NIC/cgroup traces; NIC rates include headers and unrelated interface traffic."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(folder):
    traces = {}
    for node in ("T1", "T2", "T3", "T4"):
        path = folder / f"{node}_nic_monitor.jsonl"
        values = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        traces[node] = [
            dict(
                start=a["time_ns"] / 1e9,
                end=b["time_ns"] / 1e9,
                rx_Gbps=8 * (b["rx_bytes"] - a["rx_bytes"]) / (b["time_ns"] - a["time_ns"]),
                tx_Gbps=8 * (b["tx_bytes"] - a["tx_bytes"]) / (b["time_ns"] - a["time_ns"]),
            )
            for a, b in zip(values, values[1:], strict=False)
            if 0.5e9 < b["time_ns"] - a["time_ns"] < 2e9
        ]
    records = []
    for row in json.loads((folder / "summary.json").read_text()):
        obj = json.loads((folder / (row["id"] + ".json")).read_text())
        start = obj["iperf"].get("start", {}).get("timestamp", {}).get("timesecs")
        if start is None:
            continue
        bounds = (start + 3, start + 7)  # conservative interior of 2s warm-up + 5s measured sender window
        node = "T1" if row["reverse"] else "T4"
        points = [x["rx_Gbps"] for x in traces[node] if bounds[0] <= x["start"] and x["end"] <= bounds[1]]
        records.append(
            dict(
                id=row["id"],
                label=row["label"],
                node=node,
                sender_timestamp_seconds=start,
                sample_window_seconds=bounds,
                samples=len(points),
                median_rx_NIC_Gbps=statistics.median(points) if points else None,
                max_rx_NIC_Gbps=max(points) if points else None,
            )
        )
    memories = defaultdict(list)
    for node in ("T2", "T3"):
        path = folder / f"{node}_memory_monitor.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            memories[(node, row["container"])].append(row)
    memory_summary = []
    for (node, name), items in memories.items():
        peaks = [int(x["memory.peak"]) for x in items if x.get("memory.peak", "").isdigit()]
        runtime_path = folder / (name + "_runtime.json")
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        inspect = runtime.get("inspect") or []
        docker_oom = inspect[0]["State"]["OOMKilled"] if inspect else None
        memory_summary.append(
            dict(
                node=node,
                container=name,
                docker_OOMKilled=docker_oom,
                samples=len(items),
                peak_GiB=max(peaks) / 2**30 if peaks else None,
                oom_kill_seen=any(
                    int(match.group(1)) > 0 for x in items if (match := re.search(r"^oom_kill\s+(\d+)$", x.get("memory.events", ""), re.M))
                ),
            )
        )
    result = dict(
        nic_metric="1 Hz interface bytes, includes headers and background traffic; not application goodput",
        sample_alignment="iperf integer-second timestamp +3 through +7; interior of 2s warm-up and 5s test",
        traces=traces,
        trials=records,
        memory=memory_summary,
    )
    (folder / "telemetry_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    starts = [p["start"] for v in traces.values() for p in v]
    starts.extend(v["time_ns"] / 1e9 for items in memories.values() for v in items)
    if starts:
        t0 = min(starts)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for node in ("T1", "T4"):
            xs, ys = [], []
            previous = None
            for sample in traces[node]:
                if previous is not None and sample["start"] - previous > 2:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                xs.append(sample["end"] - t0)
                ys.append(sample["rx_Gbps"])
                previous = sample["end"]
            axes[0].plot(xs, ys, label=node + " receive", linewidth=1)
        axes[0].axhline(10, color="gray", linestyle="--", label="10 Gbps target")
        axes[0].set(ylabel="NIC receive rate (Gbps)", ylim=(0, 12))
        for node in ("T2", "T3"):
            points = sorted(
                (v["time_ns"] / 1e9, int(v["memory.current"]) / 2**30, name)
                for (n, name), items in memories.items()
                if n == node
                for v in items
                if v.get("memory.current", "").isdigit()
            )
            xs, ys = [], []
            previous = None
            for stamp, value, container in points:
                if previous and (container != previous[1] or stamp - previous[0] > 2):
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                xs.append(stamp - t0)
                ys.append(value)
                previous = (stamp, container)
            axes[1].plot(xs, ys, label=node + " container memory", linewidth=1)
        axes[1].axhline(16, color="gray", linestyle="--", label="16 GiB container cap")
        axes[1].set(xlabel="Seconds since first telemetry sample", ylabel="Container memory (GiB)", ylim=(0, 17))
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle("10 Gbps target campaign — interface counters and memory (1 Hz)")
        fig.tight_layout()
        fig.savefig(folder / "telemetry.png", dpi=180)
        fig.savefig(folder / "telemetry.pdf")
        plt.close(fig)
    print(json.dumps(dict(trials=records, memory=memory_summary), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    analyze(p.parse_args().folder)
