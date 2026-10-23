"""Seed control. Every stochastic component takes an explicit numpy Generator."""
from __future__ import annotations

import random
import zlib

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a numpy Generator and also seed Python's `random` for any third-party use."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    return np.random.default_rng(seed)


def child_seed(seed: int, *labels: object) -> int:
    """Deterministic derived seed for a named sub-stream (stable across runs and platforms)."""
    ss = np.random.SeedSequence([seed, *[zlib.crc32(str(x).encode()) for x in labels]])
    return int(ss.generate_state(1)[0])
