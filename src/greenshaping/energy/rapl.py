"""Intel RAPL energy through the Linux powercap interface (Sec. 6.4).

Reads `energy_uj` and `max_energy_range_uj` from /sys/class/powercap/intel-rapl:*.
Only available on x86 Linux hosts with RAPL (not on Apple-silicon Mac Studio, not on
Raspberry Pi / Jetson). Reading may require root. RAPL covers package and DRAM, not
the network interface. This module never writes to sysfs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_POWERCAP = Path("/sys/class/powercap")


@dataclass(frozen=True)
class RaplZone:
    path: Path
    name: str            # e.g. "package-0", "dram"
    max_range_uj: int


def discover_zones(root: Path = DEFAULT_POWERCAP) -> list[RaplZone]:
    """List intel-rapl zones and subzones under `root` (empty list if RAPL is absent)."""
    zones: list[RaplZone] = []
    if not root.is_dir():
        return zones
    for p in sorted(root.glob("intel-rapl:*")):
        name_f, max_f = p / "name", p / "max_energy_range_uj"
        if name_f.is_file() and max_f.is_file() and (p / "energy_uj").is_file():
            zones.append(RaplZone(p, name_f.read_text().strip(), int(max_f.read_text().strip())))
    return zones


def read_energy_uj(zone: RaplZone) -> int:
    return int((zone.path / "energy_uj").read_text().strip())


def energy_delta_uj(start_uj: int, end_uj: int, max_range_uj: int) -> int:
    """Counter difference with one wrap-around correction (counter restarts after max_range_uj).

    Assumes at most one wrap between the two reads; sample more often than one wrap period.
    """
    if max_range_uj <= 0:
        raise ValueError("max_range_uj must be > 0")
    if not (0 <= start_uj <= max_range_uj and 0 <= end_uj <= max_range_uj):
        raise ValueError("counter values outside [0, max_range_uj]")
    return end_uj - start_uj if end_uj >= start_uj else end_uj + max_range_uj - start_uj


def package_and_dram(zones: list[RaplZone]) -> list[RaplZone]:
    """Zones to SUM for node energy: `package-*` and `dram` (minor 28).

    `core` and `uncore` are subzones counted inside their package; summing them as well
    would double-count.
    """
    return [z for z in zones if z.name.startswith("package") or z.name == "dram"]


def snapshot(zones: list[RaplZone]) -> dict[str, int]:
    return {z.name + "@" + z.path.name: read_energy_uj(z) for z in zones}


def session_energy_joules(start: dict[str, int], end: dict[str, int], zones: list[RaplZone],
                          idle_power_w: dict[str, float], duration_s: float) -> dict[str, float]:
    """Per-zone energy of one session window minus idle power x duration (Sec. 6.4)."""
    out = {}
    for z in zones:
        k = z.name + "@" + z.path.name
        joules = energy_delta_uj(start[k], end[k], z.max_range_uj) / 1e6
        out[k] = joules - idle_power_w.get(k, 0.0) * duration_s
    return out
