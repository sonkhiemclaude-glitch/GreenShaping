"""Resumable UCI HIGGS download and schema audit; no private-site/energy claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz"
CATALOG_URL = "https://archive.ics.uci.edu/dataset/280/higgs"
SOURCE_HASH = "ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698"


def download_archive(part: Path, target: Path, expected_sha: str = SOURCE_HASH) -> None:
    # A fresh curl invocation re-reads the CURRENT partial size after code 18.
    # Keep partial data on every failure; never publish an unchecked archive.
    for attempt in range(3):
        result = subprocess.run([
            "curl", "--fail", "--location", "--continue-at", "-", "--connect-timeout", "30",
            "--max-time", "10800", "--max-filesize", "3000000000", "--output", str(part), URL,
        ])
        if result.returncode == 0:
            digest = hashlib.sha256()
            with part.open("rb") as f:
                while data := f.read(4 * 1024 * 1024):
                    digest.update(data)
            if digest.hexdigest() != expected_sha:
                raise ValueError("source SHA256 mismatch; partial retained for review")
            part.rename(target)
            return
        if result.returncode == 33 and part.exists():
            # Some UCI responses ignore Range. Preserve the old partial once,
            # then fetch a fresh complete archive rather than append HTTP 200.
            backup = part.with_name(part.name + ".before_restart")
            if backup.exists():
                raise RuntimeError("server rejected resume again; both partial downloads retained")
            if shutil.disk_usage(part.parent).free < 4_000_000_000:
                raise RuntimeError("need 4GB free for non-resumable restart; partial retained")
            part.rename(backup)
            print(json.dumps({"range_unsupported": True, "partial_preserved": str(backup),
                              "next_attempt": "fresh complete GET; SHA256 required before publication"}), flush=True)
        print(json.dumps({"download_attempt": attempt + 1, "curl_exit": result.returncode,
                          "retained_bytes": part.stat().st_size if part.exists() else 0}), flush=True)
    raise RuntimeError("three bounded resumable download attempts failed; partial retained")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.data.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    target, part = a.data / "HIGGS.csv.gz", a.data / "HIGGS.csv.gz.part"
    if not target.exists():
        if shutil.disk_usage(a.data).free < 8_000_000_000:
            raise RuntimeError("need at least 8GB free before download")
        download_archive(part, target)
    digest = hashlib.sha256()
    with target.open("rb") as f:
        while data := f.read(4 * 1024 * 1024):
            digest.update(data)
    if digest.hexdigest() != SOURCE_HASH:
        raise ValueError("existing source archive SHA256 mismatch")
    rows = 0
    for chunk in pd.read_csv(target, header=None, chunksize=250000, compression="gzip"):
        if chunk.shape[1] != 29 or not np.isfinite(chunk.to_numpy()).all() or not chunk[0].isin([0, 1]).all():
            raise ValueError("invalid HIGGS schema, label or nonfinite value")
        rows += len(chunk)
        (a.out / "progress.json").write_text(json.dumps({"rows_validated": rows, "expected_rows": 11000000}))
    if rows != 11000000:
        raise ValueError(f"unexpected HIGGS row count: {rows}")
    (a.out / "source_manifest.json").write_text(json.dumps({
        "url": URL, "catalog_url": CATALOG_URL, "doi": "10.24432/C5V312",
        "credit": "Whiteson, D. (2014). HIGGS. UCI Machine Learning Repository.",
        "license": "CC BY 4.0", "rows": rows, "features": 28, "label_columns": 1,
        "sha256": digest.hexdigest(), "gzip_bytes": target.stat().st_size,
        "data_generation": "Monte Carlo physics simulation, not clinical observations",
        "status": "public_source_schema_validated; question bank and per-split class support still pending",
        "final_question_test_acquired": False, "measured_energy": False,
    }, indent=2))


if __name__ == "__main__":
    main()
