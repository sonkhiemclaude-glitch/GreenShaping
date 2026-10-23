"""Composition of primitives in the order of Prop. 4: rounds -> padding -> DP release.

Delta for the DP release is measured on the bytes that ENTER the release, after
rounds and padding (Algorithm 1 line 3; Prop. 4), so the guarantee of Prop. 3
applies to the sessions within Delta. A percentile below 100 leaves some sessions
above Delta; they are flagged as privacy failures.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..data.schemas import Session
from .dp_release import MODES, dp_release
from .padding import pad_fixed, pad_session
from .rounds import synchronised_rounds
from .stats import ShapeStats, last_real_reply


@dataclass(frozen=True)
class PaddingParams:
    ratio: float = 2.0
    base: int = 512


@dataclass(frozen=True)
class RoundsParams:
    window: float = 0.5
    cover_prob: float = 1.0
    reply_frame: int = 16384
    poll_size: int = 256
    poll_offset: float | None = None   # audit A-4; None = W/2
    call_ratio: float = 2.0            # call-frame ladder (audit A-4)
    call_base: int = 512
    group_gap: float = 0.1             # calls within this gap form one round


@dataclass(frozen=True)
class DPParams:
    eps_dir: float = 4.0
    interval: float = 1.0
    horizon: float = 20.0
    mode: str = "queue"
    bias: float = 0.0
    delta: int | None = None          # bytes; None -> calibrate at `delta_percentile`
    delta_percentile: float = 100.0
    session_start_bytes: int = 0
    sensitivity_mode: str = "bounded_pair"
    volume_cap_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"DP mode must be one of {MODES}")
        if not 0 < self.delta_percentile <= 100:
            raise ValueError("delta_percentile must be in (0, 100]")
        if self.sensitivity_mode not in ("bounded_pair", "legacy_volume"):
            raise ValueError("unknown sensitivity_mode")
        if any(not math.isfinite(v) or v <= 0 for v in (self.eps_dir, self.interval, self.horizon)):
            raise ValueError("epsilon, interval and horizon must be finite and positive")
        if not math.isfinite(self.bias) or self.bias < 0:
            raise ValueError("bias must be finite and nonnegative")
        if self.volume_cap_bytes is not None and self.volume_cap_bytes < 1:
            raise ValueError("volume_cap_bytes must be positive")
        if self.delta is not None and self.delta < (2 if self.sensitivity_mode == "bounded_pair" else 1):
            raise ValueError("Delta is too small for the sensitivity policy")


@dataclass(frozen=True)
class ShapingConfig:
    """One configuration theta (Algorithm 1). All fields None = no shaping."""

    padding: PaddingParams | None = None
    fixed_unit: int | None = None            # fixed-size padding baseline (Sec. 6.3)
    rounds: RoundsParams | None = None
    dp: DPParams | None = None
    name: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.padding is not None and self.fixed_unit is not None:
            raise ValueError("choose bucketed padding or fixed-size padding, not both")
        if self.fixed_unit is not None and self.dp is not None:
            raise ValueError("fixed-size padding is a stand-alone baseline; it is not combined with the DP release")

    def n_primitives(self) -> int:
        return sum(x is not None for x in (self.padding or self.fixed_unit, self.rounds, self.dp))

    def label(self) -> str:
        if self.name:
            return self.name
        parts = []
        if self.rounds:
            r = self.rounds
            extra = "".join(f",{k}={v}" for k, v in (("P", r.poll_size), ("po", r.poll_offset), ("cr", r.call_ratio),
                                                      ("cb", r.call_base), ("gg", r.group_gap))
                            if v != getattr(RoundsParams, {"P": "poll_size", "po": "poll_offset", "cr": "call_ratio",
                                                            "cb": "call_base", "gg": "group_gap"}[k]))
            parts.append(f"rounds(W={r.window},q={r.cover_prob},R={r.reply_frame}{extra})")
        if self.padding:
            parts.append(f"pad(r={self.padding.ratio},beta={self.padding.base})")
        if self.fixed_unit:
            parts.append(f"fixed({self.fixed_unit})")
        if self.dp:
            d = self.dp
            extra = (f",bias={d.bias}" if d.bias else "") + (f",start={d.session_start_bytes}" if d.session_start_bytes else "")
            extra += f",D={d.delta},V={d.volume_cap_bytes},adj={d.sensitivity_mode}"
            parts.append(f"dp({d.mode},eps={d.eps_dir},tau={d.interval},H={d.horizon},p={d.delta_percentile}{extra})")
        return "+".join(parts) if parts else "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ShapingConfig:
        known = {"padding", "fixed_unit", "rounds", "dp", "name"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown shaping keys {sorted(unknown)}")
        return ShapingConfig(
            padding=PaddingParams(**d["padding"]) if d.get("padding") else None,
            fixed_unit=d.get("fixed_unit"),
            rounds=RoundsParams(**d["rounds"]) if d.get("rounds") else None,
            dp=DPParams(**d["dp"]) if d.get("dp") else None,
            name=d.get("name", ""),
        )

    def key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def config_id(self) -> str:
        value = self.to_dict()
        value.pop("name", None)
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def link_direction_bytes(sess: Session) -> list[int]:
    """Total bytes per (site, direction) in one session (what Delta must cover, Sec. 5)."""
    tot: dict[tuple[int, int], int] = defaultdict(int)
    for m in sess.messages:
        tot[(m.site, m.direction)] += m.size
    return list(tot.values())


def pre_dp(sess: Session, cfg: ShapingConfig, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
    """Rounds then padding (the part of the pipeline before the DP release)."""
    st = ShapeStats(real_done=last_real_reply(sess.messages))
    if cfg.rounds is not None:
        r = cfg.rounds
        sess, st = synchronised_rounds(sess, r.window, r.cover_prob, r.reply_frame, r.poll_size, r.poll_offset,
                                       r.call_ratio, r.call_base, r.group_gap, rng=rng)
    if cfg.fixed_unit is not None:
        sess, st2 = pad_fixed(sess, cfg.fixed_unit)
        st.real_done = max(st.real_done, st2.real_done)
    if cfg.padding is not None:
        sess, st2 = pad_session(sess, cfg.padding.ratio, cfg.padding.base)
        st.real_done = max(st.real_done, st2.real_done)
    return sess, st


def calibrate_delta(sessions: list[Session], cfg: ShapingConfig, percentile: float = 100.0, seed: int = 0) -> int:
    """Delta = percentile of per-link-direction bytes entering the DP release (Alg. 1 line 3)."""
    if not sessions:
        raise ValueError("need calibration sessions to set Delta")
    rng = np.random.default_rng(seed)
    vals: list[int] = []
    for s in sessions:
        sh, _ = pre_dp(s, cfg, rng)
        vals += link_direction_bytes(sh)
    if not vals:
        raise ValueError("calibration sessions carry no traffic")
    return max(1, int(math.ceil(float(np.percentile(vals, percentile)))))


def calibrate_horizon(sessions: list[Session], cfg: ShapingConfig, percentile: float = 99.0, seed: int = 0,
                      margin: float = 1.0) -> float:
    """Sec. 4.3 (iv): H from calibration sessions, so that truncation is rare.

    Returns `margin` times the given percentile of the duration of the traffic entering
    the DP release (after rounds and padding), rounded up to a whole number of ticks.
    The DP release itself delays traffic, so check the overflow rate of the chosen H
    (Algorithm 1 line 5) and raise `margin` if needed.
    """
    if cfg.dp is None:
        raise ValueError("calibrate_horizon needs a configuration with a DP release")
    rng = np.random.default_rng(seed)
    durs = [max((m.t for m in pre_dp(s, cfg, rng)[0].messages), default=0.0) for s in sessions]
    if not durs:
        raise ValueError("need calibration sessions to set H")
    h = margin * float(np.percentile(durs, percentile))
    tau = cfg.dp.interval
    return max(tau, math.ceil(h / tau - 1e-12) * tau)


def resolve_delta(cfg: ShapingConfig, calibration: list[Session], seed: int = 0) -> ShapingConfig:
    """Return a copy of cfg with Delta fixed from calibration sessions if it was None."""
    if cfg.dp is None:
        return cfg
    d = cfg.dp
    if d.sensitivity_mode == "legacy_volume":
        delta = d.delta or calibrate_delta(calibration, cfg, d.delta_percentile, seed)
        cap = d.volume_cap_bytes
    else:
        cap = d.volume_cap_bytes or (d.delta // 2 if d.delta else calibrate_delta(calibration, cfg, d.delta_percentile, seed))
        delta = d.delta or 2 * cap
        if delta < 2 * cap:
            raise ValueError("bounded-pair adjacency requires Delta >= 2 * volume_cap_bytes")
    dp = DPParams(**{**asdict(d), "delta": delta, "volume_cap_bytes": cap})
    return ShapingConfig(cfg.padding, cfg.fixed_unit, cfg.rounds, dp, cfg.name)


def apply_shaping(sess: Session, cfg: ShapingConfig, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
    """Apply theta to one session. A DP release requires Delta resolved (see `resolve_delta`)."""
    sess, st = pre_dp(sess, cfg, rng)
    if cfg.dp is not None:
        d = cfg.dp
        if d.delta is None:
            raise ValueError("DP release needs Delta; call resolve_delta with calibration sessions first")
        outside_cap = d.volume_cap_bytes is not None and any(v > d.volume_cap_bytes for v in link_direction_bytes(sess))
        sess, st = dp_release(sess, d.interval, d.horizon, d.eps_dir, d.delta, d.mode, d.bias,
                              d.session_start_bytes, rng)
        st.above_delta |= outside_cap
    return sess, st
