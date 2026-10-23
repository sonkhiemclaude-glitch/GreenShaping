"""E1: leakage of unshaped coordination traffic (Sec. 6, Table 4).

Attacker trained on attacker-train + selection questions, scored on final-test
questions; separately for every (number of sites N, LLM) in the data (Table 4; review
M5). Training and test sessions of one row come from the same LLM.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from ..attack.evaluation import evaluate_attacker
from ..utils.io import write_json
from .common import RunContext

log = logging.getLogger(__name__)


def run_e1(ctx: RunContext) -> list[dict[str, Any]]:
    a = ctx.cfg.get("attack", {})
    rows = []

    def llm(s: Any) -> str:
        return str(s.meta.get("llm", "n/a"))

    for n in ctx.n_values:
        for model in sorted({llm(s) for s in ctx.sessions if s.n_sites == n}):
            fit = [s for s in ctx.train + ctx.select if s.n_sites == n and llm(s) == model]
            test = [s for s in ctx.test if s.n_sites == n and llm(s) == model]
            if not fit or not test:
                log.warning("E1: no train or test sessions for N=%d, llm=%s; skipped", n, model)
                continue
            res = evaluate_attacker(fit, test, seed=ctx.seed, n_boot=int(a.get("n_boot", 200)), **ctx.attack_kw())
            model_id = hashlib.sha256(model.encode()).hexdigest()[:12]
            write_json(res, ctx.out_dir / "attacker_details" / f"e1_N{n}_{model_id}.json")
            for secret, m in res.items():
                rows.append({"experiment": "E1", "N": n, "llm": model, "secret": secret, "attacker": m["attacker"],
                             "accuracy": m["acc"], "chance": m["chance"],
                             "advantage": m["adv_raw"], "advantage_ci_low": m["adv_ci_low"],
                             "advantage_ci_high": m["adv_ci_high"], "auc": m["auc"], "n_test": m["n_test"],
                             "bootstrap_dropped": m["boot_dropped"]})
                for k in ("balanced_accuracy", "chance_global", "chance_mode", "train_majority_accuracy"):
                    rows[-1][k] = m.get(k)
                for attacker, details in m.get("per_model", {}).items():
                    for k in ("acc", "auc", "adv_raw", "balanced_accuracy"):
                        rows[-1][f"{k}_{attacker}"] = details.get(k)
    if not rows:
        raise ValueError("E1 produced no rows: check that every N has sessions in all splits")
    return rows
