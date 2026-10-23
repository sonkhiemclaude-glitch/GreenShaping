"""External power meter traces for Raspberry Pi 5 / Jetson sites (Sec. 6.4).

Expected CSV: header `timestamp_s,power_w` (meter clock, seconds). The meter model,
sampling rate and resolution must be recorded in the run metadata. No meter-vendor
API is assumed; export the trace with the vendor's own tool.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    t, p = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"timestamp_s", "power_w"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns timestamp_s,power_w")
        for row in reader:
            t.append(float(row["timestamp_s"]))
            p.append(float(row["power_w"]))
    ta, pa = np.asarray(t), np.asarray(p)
    if len(ta) < 2 or np.any(np.diff(ta) <= 0):
        raise ValueError(f"{path}: need >= 2 samples with strictly increasing timestamps")
    return ta, pa


def integrate_power(t: np.ndarray, p: np.ndarray, start: float, end: float, clock_offset_s: float = 0.0) -> float:
    """Energy (J) between start and end (session clock) by the trapezoid rule; meter time = session time + offset."""
    a, b = start + clock_offset_s, end + clock_offset_s
    if not (t[0] <= a < b <= t[-1]):
        raise ValueError("integration window lies outside the meter trace")
    grid = np.concatenate([[a], t[(t > a) & (t < b)], [b]])
    vals = np.interp(grid, t, p)
    return float(np.sum((vals[1:] + vals[:-1]) / 2 * np.diff(grid)))


def extra_energy_joules(t: np.ndarray, p: np.ndarray, start: float, end: float, idle_power_w: float,
                        clock_offset_s: float = 0.0) -> float:
    """Measured energy of a session window minus idle power x duration (Sec. 6.4)."""
    return integrate_power(t, p, start, end, clock_offset_s) - idle_power_w * (end - start)
