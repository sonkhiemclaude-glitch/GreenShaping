"""Small scripted TLS integration; roles run only inside a dedicated EC2 run directory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from greenshaping.shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=["server", "client"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--site", type=int)
    a = p.parse_args()
    root = a.root
    if a.role == "server":
        df = pd.read_csv(root / f"site_{a.site}.csv")
        server = TLSToolServer(a.host, a.port, str(root / "server.cert"), str(root / "server.key"),
                               lambda family, params: run_query(df, family, params))
        print(json.dumps({"ready": True, "site": a.site, "address": server.address}), flush=True)
        time.sleep(240)  # Outer timeout and owned-PID cleanup also bound this service.
        return
    deployment = json.loads((root / "deployment.json").read_text())
    sites = [tuple(x) for x in deployment["sites"]]
    dp = DPParams(eps_dir=8 / (2 * len(sites)), interval=.05, horizon=2., mode="arrivals",
                  delta=16384, volume_cap_bytes=8192, bias=16000)
    rounds = RoundsParams(window=.1, cover_prob=1., reply_frame=1024)
    modes = {"none": ShapingConfig(), "pad": ShapingConfig(padding=PaddingParams()),
             "rounds": ShapingConfig(rounds=rounds), "dp": ShapingConfig(dp=dp),
             "combo": ShapingConfig(padding=PaddingParams(), rounds=rounds, dp=dp)}
    questions = load_question_bank(root / "smoke_train.jsonl", require_min=False)
    results = []
    for mode, cfg in modes.items():
        tr = ShapedTransport(sites, str(root / "ca.cert"), cfg, start_lead_s=1.)
        try:
            for q in questions:
                row = {"question_id": q.question_id, "family": q.family, "mode": mode}
                try:
                    calls = q.plan[0]
                    tr.begin(called_sites=[c["site"] for c in calls])
                    started = time.time()
                    try:
                        row["responses"] = tr.call_round(q.family, calls)
                        row["tool_completed_epoch"] = time.time()
                        row["tool_elapsed_s"] = row["tool_completed_epoch"] - started
                    finally:
                        row["transport"] = tr.finish()
                    row["status"] = "completed"
                except Exception as exc:
                    row.update(status="failed", error=repr(exc))
                results.append(row)
                (root / "results.json").write_text(json.dumps(results, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k not in ("responses", "transport")}), flush=True)
        finally:
            tr.close()


if __name__ == "__main__":
    main()
