"""Adapters from real testbed files to `Session` objects. Nothing is downloaded.

Expected layout under `data.root` (documented in docs/data_format.md):

    <root>/manifests/<session_id>.json   SessionManifest written by the testbed driver
    <root>/captures/<session_id>.csv     tshark export of the orchestrator capture
    <root>/sessions.jsonl                (optional) cache written by `export_sessions`

The capture CSV has one row per TCP segment with payload, columns in this order:
frame.time_epoch, ip.src, ip.dst, tcp.srcport, tcp.dstport, tcp.len
(see `testbed.capture_commands.tshark_export_command`).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from ..utils.io import read_jsonl, write_jsonl
from .schemas import DOWN, QUERY_FAMILIES, UP, Message, Session, SessionManifest

log = logging.getLogger(__name__)


def data_kind_for_llm(llm: str) -> str:
    """Sessions driven by the scripted orchestrator (manifest llm "none ...") are pipeline
    rehearsals, not paper data (review M7)."""
    return "rehearsal" if str(llm).strip().lower().startswith("none") else "real"


class CaptureCsvAdapter:
    """Reconstruct per-link application messages from segments (observer side, Sec. 3.2).

    Segments on one link and direction are merged into one message while the gap
    between consecutive segments is at most `gap_s` seconds. The paper does not fix
    this threshold (audit A-3); 10 ms is the default.
    """

    def __init__(self, gap_s: float = 0.010) -> None:
        if gap_s <= 0:
            raise ValueError("gap_s must be > 0")
        self.gap_s = gap_s

    def load(self, capture_csv: str | Path, manifest: SessionManifest) -> Session:
        sites = {(s.ip, int(s.port)): i for i, s in enumerate(manifest.sites)}
        segs: dict[tuple[int, int], list[tuple[float, int]]] = {}
        unmatched = 0
        with open(capture_csv, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("frame") or row[0].startswith("#"):
                    continue
                if len(row) < 6:
                    raise ValueError(f"{capture_csv}: expected 6 columns, got {len(row)}: {row}")
                t, src, dst, sp, dp, ln = row[:6]
                if float(t) < manifest.start_time or (manifest.end_time is not None and float(t) > manifest.end_time):
                    continue
                if (dst, int(dp)) in sites:
                    key = (sites[(dst, int(dp))], UP)
                elif (src, int(sp)) in sites:
                    key = (sites[(src, int(sp))], DOWN)
                else:
                    unmatched += 1
                    continue
                segs.setdefault(key, []).append((float(t) - manifest.start_time, int(ln)))
        msgs: list[Message] = []
        for (link, d), lst in segs.items():
            lst.sort()
            start, size, last = lst[0][0], lst[0][1], lst[0][0]
            for ts, nbytes in lst[1:]:
                if ts - last <= self.gap_s:
                    size += nbytes
                else:
                    msgs.append(Message(start, link, d, size))
                    start, size = ts, nbytes
                last = ts
            msgs.append(Message(start, link, d, size))
        s = Session(manifest.session_id, manifest.question_id, len(manifest.sites),
                    QUERY_FAMILIES.index(manifest.query_family),
                    [s.participates for s in manifest.sites], [s.bucket() for s in manifest.sites],
                    [], None, meta={**manifest.metadata,
                                    "data_kind": manifest.data_kind or data_kind_for_llm(manifest.llm), "llm": manifest.llm,
                                    "node_classes": [s.node_class for s in manifest.sites],
                                    "unmatched_rows": unmatched,
                                    "clipped_size_buckets": sum(sp.bucket_clipped() for sp in manifest.sites)})
        return s.with_messages(msgs)


class SessionDirectoryAdapter:
    """Load all sessions of one run directory (`data.root`) on the Mac Studio or elsewhere."""

    def __init__(self, root: str | Path, gap_s: float = 0.010) -> None:
        self.root = Path(root)
        self.capture = CaptureCsvAdapter(gap_s)

    def check_layout(self) -> list[str]:
        """Return a list of problems with the directory layout (empty when valid)."""
        problems = []
        for sub in ("manifests", "captures"):
            if not (self.root / sub).is_dir():
                problems.append(f"missing directory {self.root / sub}")
        if not problems:
            for m in sorted((self.root / "manifests").glob("*.json")):
                if not (self.root / "captures" / f"{m.stem}.csv").is_file():
                    problems.append(f"no capture for manifest {m.name}")
        return problems

    def input_files(self) -> list[Path]:
        """Every file whose content determines the loaded sessions (hashed into run_metadata)."""
        return sorted(p for sub in ("manifests", "captures") for p in (self.root / sub).glob("*") if p.is_file())

    def load(self) -> list[Session]:
        problems = self.check_layout()
        if problems:
            raise FileNotFoundError("; ".join(problems))
        sessions = []
        for m in sorted((self.root / "manifests").glob("*.json")):
            with open(m) as f:
                man = SessionManifest.from_dict(json.load(f))
            sessions.append(self.capture.load(self.root / "captures" / f"{m.stem}.csv", man))
        if not sessions:
            raise FileNotFoundError(f"no manifests in {self.root / 'manifests'}")
        log.info("loaded %d real sessions from %s", len(sessions), self.root)
        return sessions


def export_sessions(sessions: list[Session], path: str | Path) -> None:
    write_jsonl([s.to_dict() for s in sessions], path)


def import_sessions(path: str | Path) -> list[Session]:
    return [Session.from_dict(d) for d in read_jsonl(path)]


def split_capture_by_sessions(capture_csv: str | Path, manifests_dir: str | Path, out_dir: str | Path) -> dict[int, int]:
    """Split one tshark export into captures/<session_id>.csv using manifest start times.

    Sessions are serialised (Sec. 4.3 v): rows with start_k <= t < start_{k+1} belong to
    session k (the last session takes all later rows). Returns rows written per session.
    """
    mans = []
    for m in sorted(Path(manifests_dir).glob("*.json")):
        with open(m) as f:
            d = json.load(f)
        mans.append((float(d["start_time"]), int(d["session_id"]), d.get("end_time")))
    if not mans:
        raise FileNotFoundError(f"no manifests in {manifests_dir}")
    mans.sort()
    starts = [s for s, _, _ in mans]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, list[list[str]]] = {sid: [] for _, sid, _ in mans}
    with open(capture_csv, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith(("frame", "#")):
                continue
            t = float(row[0])
            k = -1
            for i, s in enumerate(starts):
                if t >= s:
                    k = i
            if k >= 0 and (mans[k][2] is None or t <= float(mans[k][2])):
                buckets[mans[k][1]].append(row)
    counts = {}
    for sid, rows in buckets.items():
        with open(out / f"{sid}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        counts[sid] = len(rows)
    return counts
