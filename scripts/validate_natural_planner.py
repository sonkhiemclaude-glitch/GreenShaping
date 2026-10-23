"""Record actual local-Ollama planning attempts on a TRAIN-only smoke bank."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from greenshaping.testbed.natural_planner import planner_prompt, planner_schema, validate_plan, validate_requested_sites
from greenshaping.testbed.question_bank import load_question_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sites", type=int, default=3)
    p.add_argument("--endpoint", default="http://127.0.0.1:11434")
    p.add_argument("--attempts", type=int, choices=[1, 2, 3], default=1)
    a = p.parse_args()
    questions = load_question_bank(a.bank, require_min=False)
    if any(q.metadata.get("partition") != "train" for q in questions):
        raise ValueError("engineering planner validation accepts train questions only")
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    for q in questions:
        request = {"model": a.model, "stream": False, "format": planner_schema(a.sites),
                   "messages": [{"role": "user", "content": planner_prompt(q.text, a.sites)}],
                   "options": {"temperature": 0, "seed": 20261027, "num_predict": 1024, "num_ctx": 8192}}
        record = {"question_id": q.question_id, "start_epoch": time.time(), "attempts": []}
        for _ in range(a.attempts):
            attempt = {"request": json.loads(json.dumps(request)), "start_epoch": time.time()}
            try:
                req = urllib.request.Request(a.endpoint.rstrip("/") + "/api/chat", data=json.dumps(request).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as response:
                    attempt["response"] = json.loads(response.read())
                plan = validate_plan(json.loads(attempt["response"]["message"]["content"]), a.sites)
                validate_requested_sites(plan, q.text)
                attempt["valid"] = True
                record.update(valid_schema=True, reference_match=plan == {"family": q.family, "calls": q.plan[0]}, plan=plan)
                record.pop("error", None)
            except Exception as exc:
                attempt.update(valid=False, error=repr(exc))
                record.update(valid_schema=False, reference_match=False, error=repr(exc))
                # Feedback contains only validation errors, never the oracle/reference plan.
                request["messages"].append({"role": "user", "content": "Your previous response failed validation: " +
                                             str(exc) + ". Recheck the original analyst request and return corrected JSON."})
            attempt["end_epoch"] = time.time()
            record["attempts"].append(attempt)
            if attempt["valid"]:
                break
        record["end_epoch"] = time.time()
        records.append(record)
        (a.out / "attempts.json").write_text(json.dumps(records, indent=2))
        print(json.dumps({k: record[k] for k in ("question_id", "valid_schema", "reference_match")}), flush=True)
    summary = {"model": a.model, "questions": len(records), "valid_schema": sum(r["valid_schema"] for r in records),
               "reference_match": sum(r["reference_match"] for r in records), "fallback_used": False,
               "maximum_attempts_per_question": a.attempts,
               "scope": "train-only one-round planner validation; no tool execution or answer-utility evaluation"}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary["reference_match"] != summary["questions"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
