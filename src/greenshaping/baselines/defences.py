"""Baselines that the paper specifies fully (Sec. 6.3) and are implemented here.

* `NoShaping`            : unprotected pipeline.
* `FixedPaddingDefence`  : every message padded to a unit covering the 99th-percentile
                           message, larger messages split.
* padding-only, rounds-only, DP-only : single-primitive configurations of Green Shaping
                           (Theta_1), built with `padding_only`, `rounds_only`, `dp_only`.
"""
from __future__ import annotations

import numpy as np

from ..data.schemas import Session
from ..shaping.padding import fixed_unit_for_percentile, pad_fixed
from ..shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig, apply_shaping, resolve_delta
from ..shaping.stats import ShapeStats, last_real_reply
from .base import Defence


class NoShaping(Defence):
    name, kind = "no_shaping", "real"

    def apply(self, sess: Session, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
        return sess.with_messages(list(sess.messages)), ShapeStats(real_done=last_real_reply(sess.messages))


class FixedPaddingDefence(Defence):
    name, kind = "fixed_padding", "real"

    def __init__(self, unit: int | None = None, percentile: float = 99.0) -> None:
        self.unit, self.percentile = unit, percentile

    def prepare(self, calibration: list[Session]) -> None:
        if self.unit is None:
            self.unit = fixed_unit_for_percentile(calibration, self.percentile)

    def apply(self, sess: Session, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
        if self.unit is None:
            raise RuntimeError("call prepare() with calibration sessions or give `unit`")
        return pad_fixed(sess, self.unit)


class ConfigDefence(Defence):
    """Any Green Shaping configuration theta used as a defence."""

    kind = "real"

    def __init__(self, cfg: ShapingConfig, name: str | None = None) -> None:
        self.cfg = cfg
        self.name = name or cfg.label()

    def prepare(self, calibration: list[Session]) -> None:
        self.cfg = resolve_delta(self.cfg, calibration)

    def apply(self, sess: Session, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
        return apply_shaping(sess, self.cfg, rng)


def padding_only(ratio: float, base: int = 512) -> ConfigDefence:
    return ConfigDefence(ShapingConfig(padding=PaddingParams(ratio, base)), f"padding_only(r={ratio})")


def rounds_only(window: float, cover_prob: float, reply_frame: int) -> ConfigDefence:
    return ConfigDefence(ShapingConfig(rounds=RoundsParams(window, cover_prob, reply_frame)),
                         f"rounds_only(W={window},q={cover_prob},R={reply_frame})")


def dp_only(eps_dir: float, interval: float, mode: str = "queue", horizon: float = 20.0,
            delta_percentile: float = 100.0) -> ConfigDefence:
    dp = DPParams(eps_dir=eps_dir, interval=interval, horizon=horizon, mode=mode, delta_percentile=delta_percentile)
    return ConfigDefence(ShapingConfig(dp=dp), f"dp_only({mode},eps={eps_dir},tau={interval},p={delta_percentile})")
