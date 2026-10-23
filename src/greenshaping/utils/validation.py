"""Input validation helpers with clear error messages."""
from __future__ import annotations

from collections.abc import Iterable


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_positive(name: str, value: float) -> None:
    require(value > 0, f"{name} must be > 0, got {value}")


def require_non_negative(name: str, value: float) -> None:
    require(value >= 0, f"{name} must be >= 0, got {value}")


def require_probability(name: str, value: float) -> None:
    require(0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}")


def require_keys(mapping: dict, keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    require(not missing, f"{where}: missing required keys {missing}")
