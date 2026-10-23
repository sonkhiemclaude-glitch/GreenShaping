"""Covertype engineering workload; independent of frozen synthetic pilots.

This prepares reference plans, not LLM-generated plans or final-test results.
Aspect predicates and wording templates have disjoint train/select/test groups.
Site allocation is a reproducible synthetic federation of unique public records.
"""
from __future__ import annotations

import gzip
import io
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.schemas import QUERY_FAMILIES
from ..utils.io import write_json, write_jsonl
from ..utils.provenance import file_sha256
from .question_bank import Question

TEMPLATES = (
    "At {sites}, {task} for records with aspect between {lo} and {hi} degrees, inclusive.",
    "For aspect in the closed interval [{lo}, {hi}] degrees, ask {sites} to {task}.",
    "Use {sites}; retain aspect angles from {lo} through {hi} degrees and {task}.",
    "Restrict {sites} to aspect >= {lo} and <= {hi} degrees, then {task}.",
    "Please {task} at {sites}, using only aspect angles in [{lo}, {hi}] degrees.",
    "Select aspect angles from {lo} to {hi} degrees inclusive at {sites} and {task}.",
    "On {sites}, consider the inclusive {lo}–{hi} degree aspect band; {task}.",
    "Within {sites}, {task} after filtering to aspect between {lo} and {hi} degrees inclusive.",
    "For the closed aspect band {lo} to {hi} degrees, {task}; contact only {sites}.",
    "Contact {sites} to {task}, excluding records outside [{lo}, {hi}] degrees of aspect.",
    "After retaining aspect >= {lo} and aspect <= {hi} degrees, {task} using {sites}.",
    "Query {sites} for aspect in [{lo}, {hi}] degrees inclusive: {task}.",
)


def read_covertype(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as z:
        with gzip.GzipFile(fileobj=io.BytesIO(z.read("covtype.data.gz"))) as data:
            raw = pd.read_csv(data, header=None)
    if raw.shape != (581012, 55) or raw.isna().any().any():
        raise ValueError("unexpected Covertype dimensions or missing data")
    if not raw.iloc[:, 54].isin(range(1, 8)).all():
        raise ValueError("invalid cover type")
    return pd.DataFrame({
        "source_row_id": np.arange(len(raw)), "elevation": raw[0], "aspect": raw[1], "slope": raw[2],
        "wilderness": raw.iloc[:, 10:14].to_numpy().argmax(axis=1) + 1,
        "cover_type": raw[54], "elevation_scaled": raw[0] / 4000., "slope_scaled": raw[2] / 90.,
        "lodgepole": (raw[54] == 2).astype(int),
    })


def make_questions(n_sites: int, seed: int) -> list[Question]:
    if not 2 <= n_sites <= 20:
        raise ValueError("expected 2..20 logical sites")
    rng = np.random.default_rng(seed)
    tasks = {
        "aggregate": "return count, sum and sum of squares of elevation",
        "group_by": "return count and sum of elevation grouped by wilderness",
        "record_count": "count the matching records",
        "train_round": "take one logistic regression gradient step predicting lodgepole from elevation_scaled and "
                       "slope_scaled, with intercept and both feature weights initially zero and learning rate 0.1",
    }
    questions: list[Question] = []
    for family in QUERY_FAMILIES:
        for template_id, template in enumerate(TEMPLATES):
            partition, low, high = ("train", 0, 179) if template_id < 6 else (
                ("select", 180, 239) if template_id < 8 else ("test", 240, 360))
            for j in range(10):
                # Wide/narrow real predicates; no row copies, synthetic reply bytes or record limits.
                width = max(1, int((high - low) * (.1, .5, 1.)[j % 3]))
                lo = int(rng.integers(low, high - width + 1)) if high > low + width else low
                hi = lo + width
                chosen = [i for i in range(n_sites) if (i + j) % 3 != template_id % 3]
                if not chosen:
                    chosen = [j % n_sites]
                params: dict[str, Any] = {"where": {"aspect": {"min": lo, "max": hi}}}
                if family == "aggregate":
                    params["columns"] = ["elevation"]
                elif family == "group_by":
                    params.update(by="wilderness", column="elevation")
                elif family == "train_round":
                    params.update(features=["elevation_scaled", "slope_scaled"], label="lodgepole",
                                  weights=[0., 0., 0.], lr=.1)
                text = template.format(sites=", ".join(f"site {i}" for i in chosen), task=tasks[family], lo=lo, hi=hi)
                questions.append(Question(200000 + len(questions), family, text,
                                          [[{"site": i, "params": params} for i in chosen]],
                                          {"partition": partition, "template_group": template_id,
                                           "predicate_group": [low, high], "workload": "covertype-preparation-v1",
                                           "plan_source": "reference oracle; not an LLM acquisition"}))
    return questions


def build_public_workload(archive: Path, output: Path, n_sites: int = 3, seed: int = 20261027) -> dict:
    if output.exists():
        raise FileExistsError(output)
    questions = make_questions(n_sites, seed)
    df = read_covertype(archive)
    allocation = np.random.default_rng(seed).permutation(len(df))
    output.mkdir(parents=True)
    tables = []
    for site, indices in enumerate(np.array_split(allocation, n_sites)):
        path = output / f"site_{site}.csv"
        df.iloc[np.sort(indices)].to_csv(path, index=False)
        tables.append({"site": site, "rows": len(indices), "sha256": file_sha256(path)})
    for partition in ("train", "select", "test"):
        write_jsonl([asdict(q) for q in questions if q.metadata["partition"] == partition], output / f"{partition}.jsonl")
    report = {
        "status": "prepared_reference_workload_not_acquired", "seed": seed, "n_sites": n_sites,
        "source": {"doi": "10.24432/C50K5N", "license": "CC BY 4.0", "credit": "Blackard, J. (1998). Covertype. UCI.",
                   "url": "https://archive.ics.uci.edu/dataset/31/covertype", "zip_sha256": file_sha256(archive)},
        "source_rows": len(df), "partition": "disjoint seeded row permutation; synthetic sites, not geographic institutions",
        "tables": tables, "questions": len(questions),
        "splits": dict(Counter(q.metadata["partition"] for q in questions)),
        "question_files": {p: file_sha256(output / f"{p}.jsonl") for p in ("train", "select", "test")},
        "transforms": {"elevation_scaled": "elevation / 4000", "slope_scaled": "slope / 90",
                       "lodgepole": "cover_type == 2", "wilderness": "one-hot index + 1"},
        "limitations": ["Public benchmark; site ownership simulated", "One-round tools; no adaptive-agent claim",
                        "Aspect cohorts differ across splits; report cohort distribution shift",
                        "Fixed manuscript size buckets may be absent; audit train/select before protocol freeze",
                        "Test reference plans generated but not evaluated; never tune on final-test results",
                        "Network preparation only; no physical energy observations"],
    }
    write_json(report, output / "workload.json")
    return report
