"""Common interface for defences. `kind` separates real, adapter and dummy implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..data.schemas import Session
from ..shaping.stats import ShapeStats

KINDS = ("real", "adapter", "dummy")


class Defence(ABC):
    name: str = "defence"
    kind: str = "real"   # "real" (implemented here from the paper), "adapter" (external), "dummy" (tests only)

    def prepare(self, calibration: list[Session]) -> None:  # noqa: B027 - optional hook
        """Fit parameters that depend on calibration sessions (e.g. Delta, fixed unit)."""

    @abstractmethod
    def apply(self, sess: Session, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
        """Return the session as the observer would see it under this defence."""
