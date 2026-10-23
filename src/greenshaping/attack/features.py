"""Observer features (Sec. 6.2). Only (time, link, direction, size) are used; never `real`.

Per link: message counts and byte totals per direction, size statistics, a histogram
of log2 sizes, first and last message times, inter-arrival statistics and the first
request-response gap. Session features aggregate per-link features over links.
`time_slice_matrix` gives the per-link packet-count / byte-volume time slices used
by the CNN attacker (AgentPrint representation).
"""
from __future__ import annotations

import numpy as np

from ..data.schemas import DOWN, UP, Session

LOG2_BIN_EDGES = np.array([0, 9, 10, 11, 12, 13, 14, 15, 16, 18, 40], float)
N_DIR_FEATURES = 7 + len(LOG2_BIN_EDGES) - 1


def _direction_features(sizes: list[int], times: list[float]) -> list[float]:
    if not sizes:
        return [0.0] * N_DIR_FEATURES
    s = np.asarray(sizes, float)
    t = np.sort(np.asarray(times, float))
    iat = np.diff(t) if len(t) > 1 else np.array([0.0])
    hist, _ = np.histogram(np.log2(s + 1), bins=LOG2_BIN_EDGES)
    return [float(len(s)), float(s.sum()), float(s.max()), float(s.mean()), float(s.std()),
            float(iat.mean()), float(iat.min())] + hist.astype(float).tolist()


def link_features(sess: Session, site: int) -> list[float]:
    link = sess.link(site)
    up = [m for m in link if m.direction == UP]
    dn = [m for m in link if m.direction == DOWN]
    times = [m.t for m in link]
    first = min(times) if times else -1.0
    last = max(times) if times else -1.0
    rr = -1.0
    if up and dn:
        t0 = min(m.t for m in up)
        later = [m.t for m in dn if m.t >= t0]
        rr = (min(later) - t0) if later else -1.0
    return ([first, last, (last - first) if times else 0.0, rr]
            + _direction_features([m.size for m in up], [m.t for m in up])
            + _direction_features([m.size for m in dn], [m.t for m in dn]))


LINK_FEATURE_DIM = 4 + 2 * N_DIR_FEATURES


def session_features(sess: Session) -> list[float]:
    per = np.array([link_features(sess, s) for s in range(sess.n_sites)], float)
    n_up_col, n_dn_col = 4, 4 + N_DIR_FEATURES
    active = float(((per[:, n_up_col] + per[:, n_dn_col]) > 0).sum())
    times = np.sort(np.array([m.t for m in sess.messages], float)) if sess.messages else np.array([0.0])
    bursts = 1 + int((np.diff(times) > 0.4).sum()) if len(times) > 1 else int(bool(sess.messages))
    total = float(sum(m.size for m in sess.messages))
    return [active, float(bursts), total, float(times.max() - times.min())] + per.mean(0).tolist() + per.max(0).tolist()


def build_tables(sessions: list[Session]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-link table (participation, size bucket) and per-session table (query family), with question groups."""
    L: dict[str, list] = {"X": [], "part": [], "bucket": [], "group": []}
    S: dict[str, list] = {"X": [], "family": [], "group": []}
    for sess in sessions:
        for s in range(sess.n_sites):
            L["X"].append(link_features(sess, s))
            L["part"].append(sess.participation[s])
            L["bucket"].append(sess.size_bucket[s])
            L["group"].append(sess.question_id)
        S["X"].append(session_features(sess))
        S["family"].append(sess.query_family)
        S["group"].append(sess.question_id)
    return ({k: np.asarray(v) for k, v in L.items()}, {k: np.asarray(v) for k, v in S.items()})


def time_slice_matrix(sess: Session, site: int, slice_s: float = 0.25, n_slices: int = 120) -> np.ndarray:
    """Per-link time slices: rows = (count up, bytes up, count down, bytes down), columns = slices.

    AgentPrint-style representation (paper Sec. 6.2); bytes are log1p-scaled.
    """
    if slice_s <= 0 or n_slices < 1:
        raise ValueError("slice_s > 0 and n_slices >= 1 required")
    out = np.zeros((4, n_slices), float)
    for m in sess.link(site):
        k = min(n_slices - 1, int(m.t // slice_s))
        row = 0 if m.direction == UP else 2
        out[row, k] += 1
        out[row + 1, k] += m.size
    out[1] = np.log1p(out[1])
    out[3] = np.log1p(out[3])
    return out


def build_time_slice_tables(sessions: list[Session], slice_s: float = 0.25,
                            n_slices: int = 120) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Same rows and labels as `build_tables`, with (4 x n_slices) time-slice matrices as X.

    Per-session X sums the time slices of all N links (so sessions with any N share one shape).
    """
    L: dict[str, list] = {"X": [], "part": [], "bucket": [], "group": []}
    S: dict[str, list] = {"X": [], "family": [], "group": []}
    for sess in sessions:
        mats = [time_slice_matrix(sess, s, slice_s, n_slices) for s in range(sess.n_sites)]
        for s, m in enumerate(mats):
            L["X"].append(m)
            L["part"].append(sess.participation[s])
            L["bucket"].append(sess.size_bucket[s])
            L["group"].append(sess.question_id)
        S["X"].append(np.sum(mats, axis=0) if mats else np.zeros((4, n_slices)))
        S["family"].append(sess.query_family)
        S["group"].append(sess.question_id)
    return ({k: np.asarray(v) for k, v in L.items()}, {k: np.asarray(v) for k, v in S.items()})


def build_linkaware_tables(sessions: list[Session], identity_only: bool = False) -> tuple[dict, dict]:
    """Public site identity and whole-network observations; labels never enter X."""
    links, global_rows = build_tables(sessions)
    ns = {s.n_sites for s in sessions}
    if len(ns) != 1:
        raise ValueError("link-aware attackers require one deployment size")
    local, whole = [], []
    for j, sess in enumerate(sessions):
        flat = np.asarray([link_features(sess, i) for i in range(sess.n_sites)]).ravel()
        for i in range(sess.n_sites):
            identity = np.eye(sess.n_sites)[i]
            local.append(identity if identity_only else np.concatenate([links["X"][j * sess.n_sites + i], identity, flat]))
        whole.append(np.zeros(1) if identity_only else np.concatenate([global_rows["X"][j], flat]))
    links["X"], global_rows["X"] = np.asarray(local), np.asarray(whole)
    links["site"] = np.asarray([i for s in sessions for i in range(s.n_sites)])
    return links, global_rows
