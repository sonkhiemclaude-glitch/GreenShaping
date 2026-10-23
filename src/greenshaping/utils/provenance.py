"""Run metadata and provenance records written next to every result file."""
from __future__ import annotations

import hashlib
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .. import REAL_LABEL, REHEARSAL_LABEL, SYNTHETIC_LABEL, __version__
from .io import write_json

DATA_KINDS = ("synthetic", "rehearsal", "real")
STATUS = {"synthetic": SYNTHETIC_LABEL, "rehearsal": REHEARSAL_LABEL, "real": REAL_LABEL}


def result_status(data_kind: str) -> str:
    if data_kind not in STATUS:
        raise ValueError(f"data_kind must be one of {DATA_KINDS}")
    return STATUS[data_kind]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(config: dict[str, Any], data_kind: str, inputs: list[str] | None = None) -> dict[str, Any]:
    """Metadata recorded with every run. `inputs` are files; each is hashed (SHA-256)."""
    status = result_status(data_kind)
    import numpy
    import sklearn

    meta: dict[str, Any] = {
        "package": "greenshaping",
        "version": __version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "data_kind": data_kind,
        "result_status": status,
        "config": config,
        "inputs": [],
    }
    for p in inputs or []:
        if not Path(p).is_file():
            raise FileNotFoundError(f"provenance input {p} is not a file; list files, not directories")
        meta["inputs"].append({"path": str(p), "sha256": file_sha256(p)})
    return meta


def write_run_metadata(out_dir: str | Path, meta: dict[str, Any]) -> Path:
    path = Path(out_dir) / "run_metadata.json"
    write_json(meta, path)
    return path
