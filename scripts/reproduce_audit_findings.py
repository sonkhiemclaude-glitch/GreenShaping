"""Minimal counterexamples from the 2026-10-20 audit; no network or training."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from greenshaping.data.schemas import DOWN, Message, Session
from greenshaping.experiments.common import merge_measured_energy
from greenshaping.shaping.pipeline import DPParams, ShapingConfig, calibrate_delta, resolve_delta
from greenshaping.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    a, b = (ShapingConfig(dp=DPParams(delta=d)) for d in (100, 10_000))
    # Equal volumes at different ticks have L1 distance twice either volume.
    sessions = [Session(i, i, 1, 0, [1], [0], [Message(t, 0, DOWN, 100)])
                for i, t in enumerate((0.5, 1.5))]
    corrected = resolve_delta(ShapingConfig(dp=DPParams()), sessions)
    evidence = {
        "scope": "historical counterexamples plus current corrections; legacy-schema demos are unverified",
        "bounded_pair_resolved": corrected.to_dict(),
        "dp_label_collision": {"label_a": a.label(), "label_b": b.label(), "collision": a.label() == b.label()},
        "delta_volume_is_not_pairwise_sensitivity": {
            "calibrated_delta": calibrate_delta(sessions, ShapingConfig(), 100),
            "arrivals_a": [100, 0], "arrivals_b": [0, 100], "pairwise_L1": 200,
            "scope": "counterexample for contribution-size/timing hypotheses; not a test of all feedback histories",
        },
    }
    with TemporaryDirectory() as temp:
        path = Path(temp) / "energy.csv"
        path.write_text("config_label,node,run,extra_joules,node_class,N\n"
                        "none,site0,0,1,x86_server,5\nnone,site0,0,9,x86_server,20\n")
        rows = [{"config": "none", "node_class": "x86_server", "N": n} for n in (5, 20)]
        try:
            merge_measured_energy(rows, str(path), strict=True)
        except ValueError:
            evidence['strict_real_rejects_legacy_energy'] = True
        else:
            evidence['strict_real_rejects_legacy_energy'] = False
        merge_measured_energy(rows, str(path))
        evidence["legacy_energy_N_pooling_unverified"] = {
            "expected_by_N": {"5": 1, "20": 9},
            "actual_by_N": {str(r["N"]): r["energy_measured_site_max_median_J"] for r in rows},
        }
        rows = [{"config": "none", "node_class": "per-manifest"}]
        merge_measured_energy(rows, str(path))
        evidence["legacy_energy_per_manifest_missing_unverified"] = {
            "measurement_present": True,
            "result_is_nan": math.isnan(rows[0]["energy_measured_site_max_median_J"]),
        }
    write_json(evidence, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
