"""Packet-level NetShaper baseline (Sec. 6.3): ADAPTER, not an implementation.

NetShaper (Sabzi et al., USENIX Security 2024) is a middlebox tunnel. It cannot be
re-created here without inventing details, so the comparison is done on CAPTURES:
run the authors' released NetShaper (or a separately documented reimplementation)
on the testbed, capture the orchestrator interface exactly as for the other
defences, and point this adapter to that run directory. The adapter loads those
shaped sessions with the same manifests as the unshaped run.

Expected layout (docs/baseline_adapters.md):
    <netshaper_root>/manifests/<session_id>.json   (same manifests as the unshaped run)
    <netshaper_root>/captures/<session_id>.csv     (capture with NetShaper active)
    <netshaper_root>/netshaper_config.json         (version, T, W, epsilon, delta, Delta_W)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.adapters import SessionDirectoryAdapter
from ..data.schemas import Session
from ..shaping.stats import ShapeStats, last_real_reply
from .base import Defence

REQUIRED_CONFIG_KEYS = ("implementation", "version", "interval_T_s", "window_W_s", "epsilon", "delta", "Delta_W_bytes")


class NetShaperCaptureAdapter(Defence):
    name, kind = "netshaper_packet_level", "adapter"

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        self._by_sid: dict[int, Session] = {}
        self.config: dict = {}

    def prepare(self, calibration: list[Session]) -> None:
        if self.root is None:
            raise NotImplementedError(
                "NetShaper baseline needs captures of a NetShaper run (set baselines.netshaper_root). "
                "See TODO_IMPLEMENTATION.md item N-1.")
        cfg_path = self.root / "netshaper_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"{cfg_path} missing: record the NetShaper version and parameters")
        self.config = json.loads(cfg_path.read_text())
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in self.config]
        if missing:
            raise ValueError(f"netshaper_config.json missing keys {missing}")
        self._by_sid = {s.sid: s for s in SessionDirectoryAdapter(self.root).load()}

    def apply(self, sess: Session, rng: np.random.Generator) -> tuple[Session, ShapeStats]:
        if sess.sid not in self._by_sid:
            raise KeyError(f"no NetShaper capture for session {sess.sid}")
        sh = self._by_sid[sess.sid]
        if any(getattr(sh, key) != getattr(sess, key)
               for key in ("question_id", "n_sites", "query_family", "participation", "size_bucket")):
            raise ValueError(f"session {sess.sid}: NetShaper manifest labels differ from the unshaped run")
        return sh, ShapeStats(real_done=last_real_reply(sh.messages))

    def config_id(self) -> str:
        return hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
