"""CO2-equivalent conversion (Sec. 6.4). The grid factor must be supplied with its source."""
from __future__ import annotations

J_PER_KWH = 3.6e6


def co2e_grams(joules: float, grid_g_per_kwh: float) -> float:
    """grams CO2e = kWh * factor. No default factor is provided on purpose (cite one per run)."""
    if grid_g_per_kwh < 0:
        raise ValueError("grid factor must be >= 0")
    return joules / J_PER_KWH * grid_g_per_kwh
