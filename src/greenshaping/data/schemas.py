"""Data structures for the objects of the paper (Sec. 3).

* `Message`  : what a passive observer sees on one orchestrator-site link: time,
               direction (call / reply) and size in bytes (Sec. 3.2).
* `Session`  : all calls and replies caused by one analyst question, plus the
               labels the observer tries to infer (Sec. 3.3): participation and
               contribution-size bucket per site, and the query family.
* `SiteSpec` / `SessionManifest` : what the testbed driver writes for every real
               session (Sec. 6.1); labels always come from the manifest, never
               from the capture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

UP = 0    # orchestrator -> site (tool call)
DOWN = 1  # site -> orchestrator (reply)
DIRECTIONS = (UP, DOWN)

QUERY_FAMILIES: tuple[str, ...] = ("aggregate", "group_by", "train_round", "record_count")

# Contribution-size buckets on matching rows (Sec. 6.1): [1e3,1e4), [1e4,1e5), [1e5,1e6].
SIZE_BUCKET_EDGES: tuple[float, ...] = (1e3, 1e4, 1e5, 1e6)
NOT_PARTICIPATING = -1


def size_bucket(matching_rows: float) -> int:
    """Bucket index 0, 1 or 2 for matching rows in [1e3, 1e6) (Sec. 6.1).

    Rows outside the paper's range (including 0 for a called site with no matching
    rows) are clipped to the nearest bucket; callers should report how many were clipped.
    """
    if matching_rows < 0:
        raise ValueError(f"matching_rows must be >= 0, got {matching_rows}")
    if matching_rows < 1:
        return 0
    b = int(math.floor(math.log10(matching_rows))) - 3
    return min(2, max(0, b))


def size_bucket_clipped(matching_rows: float) -> bool:
    """True when `matching_rows` lies outside the paper's [1e3, 1e6) range and was clipped."""
    return matching_rows < 1e3 or matching_rows >= 1e6


@dataclass(frozen=True)
class Message:
    """One application message on one link, as reconstructed by the observer."""

    t: float           # seconds since session start
    site: int          # link index 0..N-1
    direction: int     # UP or DOWN
    size: int          # bytes on the wire for this message
    real: bool = True  # False for padding-only or cover traffic; never used by the attacker

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be UP(0) or DOWN(1), got {self.direction}")
        if self.size < 0:
            raise ValueError(f"message size must be >= 0, got {self.size}")
        if self.site < 0:
            raise ValueError(f"site index must be >= 0, got {self.site}")
        if not math.isfinite(self.t):
            raise ValueError("message time must be finite")


@dataclass
class Session:
    """One analysis session with observer-visible messages and ground-truth labels."""

    sid: int
    question_id: int
    n_sites: int
    query_family: int
    participation: list[int]
    size_bucket: list[int]
    messages: list[Message] = field(default_factory=list)
    horizon: float | None = None
    # Extra wake-up times per site (DP ticks); used by the energy model only, never observable.
    wake: dict[int, list[float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_sites < 1:
            raise ValueError("n_sites must be >= 1")
        if len(self.participation) != self.n_sites or len(self.size_bucket) != self.n_sites:
            raise ValueError("participation and size_bucket must have one entry per site")
        if not 0 <= self.query_family < len(QUERY_FAMILIES):
            raise ValueError(f"query_family must index QUERY_FAMILIES, got {self.query_family}")
        for i, (p, b) in enumerate(zip(self.participation, self.size_bucket, strict=False)):
            if p not in (0, 1):
                raise ValueError(f"participation[{i}] must be 0 or 1")
            if p == 0 and b != NOT_PARTICIPATING:
                raise ValueError(f"size_bucket[{i}] must be -1 for a non-participating site")
            if p == 1 and b not in (0, 1, 2):
                raise ValueError(f"size_bucket[{i}] must be 0, 1 or 2 for a participating site")
        for m in self.messages:
            if m.site >= self.n_sites:
                raise ValueError(f"message on link {m.site} but session has {self.n_sites} sites")

    def link(self, site: int) -> list[Message]:
        return [m for m in self.messages if m.site == site]

    def link_direction(self, site: int, direction: int) -> list[Message]:
        return [m for m in self.messages if m.site == site and m.direction == direction]

    def duration(self) -> float:
        if self.horizon is not None:
            return self.horizon
        return max((m.t for m in self.messages), default=0.0)

    def with_messages(self, messages: list[Message], horizon: float | None = None,
                      wake: dict[int, list[float]] | None = None) -> Session:
        """Copy with new observable messages (labels unchanged)."""
        return replace(
            self,
            messages=sorted(messages, key=lambda m: (m.t, m.site, m.direction)),
            horizon=horizon if horizon is not None else self.horizon,
            wake=dict(wake) if wake is not None else dict(self.wake),
            participation=list(self.participation),
            size_bucket=list(self.size_bucket),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid, "question_id": self.question_id, "n_sites": self.n_sites,
            "query_family": self.query_family, "participation": self.participation,
            "size_bucket": self.size_bucket, "horizon": self.horizon,
            "messages": [[m.t, m.site, m.direction, m.size, int(m.real)] for m in self.messages],
            "wake": {str(k): v for k, v in self.wake.items()}, "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Session:
        msgs = [Message(float(t), int(s), int(di), int(sz), bool(r)) for t, s, di, sz, r in d["messages"]]
        return Session(
            sid=int(d["sid"]), question_id=int(d["question_id"]), n_sites=int(d["n_sites"]),
            query_family=int(d["query_family"]), participation=[int(x) for x in d["participation"]],
            size_bucket=[int(x) for x in d["size_bucket"]], messages=msgs,
            horizon=d.get("horizon"), wake={int(k): v for k, v in d.get("wake", {}).items()},
            meta=d.get("meta", {}),
        )


@dataclass(frozen=True)
class SiteSpec:
    """One site in a session manifest (Sec. 6.1)."""

    ip: str
    port: int
    participates: int
    matching_rows: int | None = None
    node_class: str = "unknown"   # e.g. x86_server, raspberry_pi_5, jetson

    def bucket(self) -> int:
        if not self.participates:
            return NOT_PARTICIPATING
        if self.matching_rows is None:
            raise ValueError("participating site needs matching_rows to define its size bucket")
        return size_bucket(self.matching_rows)

    def bucket_clipped(self) -> bool:
        return bool(self.participates and self.matching_rows is not None and size_bucket_clipped(self.matching_rows))


@dataclass(frozen=True)
class SessionManifest:
    """Labels and addressing written by the testbed driver for one session."""

    session_id: int
    question_id: int
    query_family: str
    start_time: float          # epoch seconds, same clock as the capture
    sites: tuple[SiteSpec, ...]
    llm: str = "unspecified"   # developer/model/version string (Elsevier AI-in-methods rule)
    data_kind: str | None = None   # explicit acquisition provenance; old manifests use the legacy LLM heuristic
    end_time: float | None = None  # epoch seconds; excludes traffic after a completed session
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_kind not in (None, "synthetic", "rehearsal", "real"):
            raise ValueError("manifest data_kind must be synthetic, rehearsal or real")
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("manifest end_time must be >= start_time")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "session_id": self.session_id, "question_id": self.question_id,
            "query_family": self.query_family, "start_time": self.start_time, "llm": self.llm,
            "sites": [s.__dict__ for s in self.sites],
        }
        if self.data_kind is not None:
            out["data_kind"] = self.data_kind
        if self.end_time is not None:
            out["end_time"] = self.end_time
        if self.metadata:
            out["metadata"] = self.metadata
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SessionManifest:
        for k in ("session_id", "question_id", "query_family", "start_time", "sites"):
            if k not in d:
                raise ValueError(f"manifest missing key {k!r}")
        if d["query_family"] not in QUERY_FAMILIES:
            raise ValueError(f"unknown query_family {d['query_family']!r}; expected one of {QUERY_FAMILIES}")
        sites = tuple(SiteSpec(ip=s["ip"], port=int(s["port"]), participates=int(s["participates"]),
                               matching_rows=s.get("matching_rows"), node_class=s.get("node_class", "unknown"))
                      for s in d["sites"])
        return SessionManifest(int(d["session_id"]), int(d["question_id"]), d["query_family"],
                               float(d["start_time"]), sites, d.get("llm", "unspecified"), d.get("data_kind"),
                               float(d["end_time"]) if d.get("end_time") is not None else None, d.get("metadata", {}))
