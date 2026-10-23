"""Shared plumbing for the experiment runners: config loading, data, cost models, outputs."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..attack.evaluation import assert_disjoint, attack_options, evaluate_attacker, split_by_question
from ..data.adapters import SessionDirectoryAdapter, import_sessions
from ..data.schemas import Session
from ..data.synthetic import SyntheticConfig, generate_sessions
from ..energy.carbon import co2e_grams
from ..energy.cost_model import ORCHESTRATOR, CostModel, bytes_on_wire, extra_energy, site_node
from ..shaping.pipeline import ShapingConfig, link_direction_bytes, pre_dp, resolve_delta
from ..shaping.stats import last_real_reply
from ..utils.io import read_csv, read_json, write_csv, write_json
from ..utils.provenance import result_status, run_metadata, write_run_metadata
from ..utils.random import child_seed
from ..utils.validation import require

log = logging.getLogger(__name__)

# Placeholder coefficients for the SYNTHETIC demo only (not measured; see energy/fit.py).
PLACEHOLDER_COST_MODEL = CostModel("placeholder", e_tx=5e-8, e_rx=5e-8, p_aw=2.0, t_tail=0.1, fitted=False)


@dataclass
class RunContext:
    cfg: dict[str, Any]
    data_kind: str
    out_dir: Path
    seed: int
    sessions: list[Session]
    train: list[Session]
    select: list[Session]
    test: list[Session]
    cost_models: dict[str, CostModel]
    inputs: list[str]

    @property
    def is_synthetic(self) -> bool:
        return self.data_kind == "synthetic"

    def label(self) -> str:
        return result_status(self.data_kind)

    @property
    def n_values(self) -> list[int]:
        return sorted({s.n_sites for s in self.sessions})

    def for_n(self, n: int) -> RunContext:
        """Sub-context with the sessions of one deployment size N (review B2)."""
        def f(x: list[Session]) -> list[Session]:
            return [s for s in x if s.n_sites == n]
        return replace(self, sessions=f(self.sessions), train=f(self.train), select=f(self.select), test=f(self.test))

    def for_sessions(self, keep: Any) -> RunContext:
        """Sub-context restricted to sessions for which keep(session) is true."""
        def f(x: list[Session]) -> list[Session]:
            return [s for s in x if keep(s)]
        return replace(self, sessions=f(self.sessions), train=f(self.train), select=f(self.select), test=f(self.test))

    def attack_kw(self) -> dict[str, Any]:
        return attack_options(self.cfg)


def _kind_of(sessions: list[Session], source: str) -> str:
    kinds = {s.meta.get("data_kind", "unknown") for s in sessions}
    if len(kinds) != 1 or "unknown" in kinds:
        raise ValueError(f"{source}: sessions must all share one known data_kind, got {sorted(kinds)} (review M7)")
    return kinds.pop()


def load_sessions(cfg: dict[str, Any]) -> tuple[list[Session], str, list[str]]:
    """Sessions, their data kind (synthetic | rehearsal | real) and the input FILES to hash."""
    data = cfg.get("data", {})
    src = data.get("source", "synthetic")
    if src == "synthetic":
        syn = dict(data.get("synthetic", {}))
        ns = syn.pop("n_sites", 10)
        out: list[Session] = []
        for k, n in enumerate(ns if isinstance(ns, list) else [ns]):
            part = generate_sessions(SyntheticConfig(n_sites=int(n), **{**syn, "seed": int(syn.get("seed", 0)) + k}))
            for s in part:
                s.sid += len(out)
            out += part
        return out, "synthetic", []
    if src == "directory":
        root = data.get("root")
        require(bool(root), "data.root must point to the run directory (manifests/, captures/)")
        ad = SessionDirectoryAdapter(root, data.get("gap_s", 0.010))
        sessions = ad.load()
        return sessions, _kind_of(sessions, str(root)), [str(p) for p in ad.input_files()]
    if src == "jsonl":
        path = data.get("sessions_jsonl")
        require(bool(path), "data.sessions_jsonl must be set")
        sessions = import_sessions(path)
        return sessions, _kind_of(sessions, str(path)), [str(path)]
    raise ValueError(f"unknown data.source {src!r} (synthetic | directory | jsonl)")


def load_cost_models(cfg: dict[str, Any], data_kind: str) -> tuple[dict[str, CostModel], list[str]]:
    """Cost models per node class and the files they came from (model JSON + .fit_report.json).

    Real data requires fitted coefficients with a fit report next to the model file
    (review M7); placeholders are allowed only for synthetic data and rehearsals.
    """
    e = cfg.get("energy", {})
    models: dict[str, CostModel] = {}
    files: list[str] = []
    for node_class, path in (e.get("cost_models") or {}).items():
        models[node_class] = CostModel(**read_json(path))
        files.append(str(path))
        rep = Path(path).with_suffix(".fit_report.json")
        if rep.is_file():
            files.append(str(rep))
            if data_kind == "real":
                validation = read_json(rep).get("validation", {})
                error = float(validation.get("heldout_relative_rmse", float("inf")))
                require(validation.get("design_rank") == 3 and np.isfinite(error)
                        and error <= float(e.get("fit_max_relative_rmse", .2)),
                        f"cost model {path} needs full-rank held-out validation within energy.fit_max_relative_rmse")
        elif data_kind == "real":
            raise ValueError(f"cost model {path} has no fit report {rep.name}; fit it with scripts/fit_cost_model.py")
    if not models:
        require(data_kind != "real",
                "real data needs fitted cost models: set energy.cost_models (see scripts/fit_cost_model.py)")
        models["default"] = PLACEHOLDER_COST_MODEL
    if data_kind == "real":
        unfitted = [k for k, m in models.items() if not m.fitted]
        require(not unfitted, f"cost models {unfitted} are not fitted; refusing to use placeholders on real data")
    return models, files


def context_from_config(cfg: dict[str, Any], out_dir: str | Path | None = None) -> RunContext:
    seed = int(cfg.get("run", {}).get("seed", 0))
    sessions, kind, inputs = load_sessions(cfg)
    declared = cfg.get("run", {}).get("data_kind")
    if declared and declared != kind:
        raise ValueError(f"config declares data_kind={declared} but the data are {kind}")
    split = cfg.get("split", {})
    tr, se, te = split_by_question(sessions, tuple(split.get("fractions", (0.4, 0.3, 0.3))), int(split.get("seed", seed)))
    assert_disjoint(tr, se, te)
    out = Path(out_dir or cfg.get("run", {}).get("output_dir", "outputs/run"))
    out.mkdir(parents=True, exist_ok=True)
    models, model_files = load_cost_models(cfg, kind)
    for extra_key in ("measured_energy_csv",):
        pth = cfg.get("energy", {}).get(extra_key)
        if pth:
            model_files.append(str(pth))
    if kind == "real" and not all(cfg.get("run", {}).get(k) for k in ("id", "deployment_id")):
        raise ValueError("real experiments need explicit run.id and run.deployment_id")
    previous = out / "run_metadata.json"
    if previous.is_file():
        candidate = run_metadata(cfg, kind, inputs + model_files)
        if read_json(previous).get("run_fingerprint") != run_fingerprint(cfg, candidate["inputs"]):
            raise ValueError("output directory belongs to different code/config/inputs; choose a new output directory")
    return RunContext(cfg, kind, out, seed, sessions, tr, se, te, models, inputs + model_files)


def cost_model_for(ctx: RunContext, node_class: str = "default") -> CostModel:
    if node_class in ctx.cost_models:
        return ctx.cost_models[node_class]
    if "default" in ctx.cost_models:
        return ctx.cost_models["default"]
    raise KeyError(f"no cost model for node class {node_class!r}")


def node_models(ctx: RunContext, node_class: str | None = None) -> Any:
    """Callable session -> {node: CostModel} (review M4).

    Sites use the node class from the session manifest (`meta.node_classes`) unless
    `node_class` forces one class; the orchestrator uses the class "orchestrator" when
    a model for it is configured, otherwise the default model.
    """
    def f(sess: Session) -> dict[str, CostModel]:
        classes = sess.meta.get("node_classes") or ["default"] * sess.n_sites
        out = {site_node(i): cost_model_for(ctx, node_class or classes[i]) for i in range(sess.n_sites)}
        out[ORCHESTRATOR] = cost_model_for(ctx, "orchestrator")
        return out
    return f


def shape_all(sessions: list[Session], cfg: ShapingConfig, seed: int, tag: str) -> tuple[list[Session], list[Any]]:
    from ..shaping.pipeline import apply_shaping

    rng = np.random.default_rng(child_seed(seed, cfg.key(), tag))
    out = [apply_shaping(s, cfg, rng) for s in sessions]
    return [o[0] for o in out], [o[1] for o in out]


def final_test_evaluation(ctx: RunContext, cfg: ShapingConfig, n_boot: int | None = None) -> dict[str, Any]:
    """Retrain on shaped (train + select), score on shaped final-test (Sec. 6.2)."""
    cfg = resolve_delta(cfg, ctx.train + ctx.select, ctx.seed)
    a = ctx.cfg.get("attack", {})
    sh_fit, _ = shape_all(ctx.train + ctx.select, cfg, ctx.seed, "fit")
    sh_te, st_te = shape_all(ctx.test, cfg, ctx.seed, "test")
    res = evaluate_attacker(sh_fit, sh_te, seed=ctx.seed, n_boot=int(a.get("n_boot", 200) if n_boot is None else n_boot),
                            **ctx.attack_kw())
    write_json(res, ctx.out_dir / "attacker_details" / f"N{ctx.n_values[0]}_{cfg.config_id()}.json")
    return {"config": cfg, "attack": res, "shaped_test": sh_te, "stats_test": st_te}


def cost_summary(ctx: RunContext, unshaped: list[Session], shaped: list[Session], stats: list[Any],
                 model: Any) -> dict[str, float]:
    """Modelled extra energy (Eq. 3), bandwidth, latency, failures for one configuration.

    `model` is a CostModel, a {node: CostModel} dict or a callable session -> dict.
    Per-node means use only the sessions that contain the node (review B2).
    """
    from ..controller.selection import mean_node_energy

    e = [extra_energy(a, b, model(b) if callable(model) else model) for a, b in zip(shaped, unshaped, strict=False)]
    mean_e = mean_node_energy(e)
    sites = [v for n, v in mean_e.items() if n.startswith("site")]
    lat = [st.real_done - last_real_reply(s.messages) for s, st in zip(unshaped, stats, strict=False)]
    fitted = all(m.fitted for m in ctx.cost_models.values())
    out = {
        "energy_modelled_orchestrator_J": mean_e.get(ORCHESTRATOR, float("nan")),
        "energy_modelled_site_median_J": float(np.median(sites)) if sites else float("nan"),
        "energy_modelled_site_max_J": float(max(sites)) if sites else float("nan"),
        "energy_model_fitted": bool(fitted),
        "bandwidth_overhead": sum(map(bytes_on_wire, shaped)) / max(1, sum(map(bytes_on_wire, unshaped))) - 1.0,
        "latency_median_s": float(np.median(lat)) if lat else float("nan"),
        "latency_p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
        "latency_truncated_rate": float(np.mean([st.real_truncated for st in stats])) if stats else float("nan"),
        "overflow_rate": float(np.mean([st.overflow_bytes > 0 for st in stats])) if stats else float("nan"),
        "privacy_failure_rate": float(np.mean([st.privacy_failure for st in stats])) if stats else float("nan"),
        "privacy_domain_failure_rate": float(np.mean([st.above_delta for st in stats])) if stats else float("nan"),
        "unpaired_reply_bytes": int(sum(st.unpaired_bytes for st in stats)),
    }
    return out


def dp_padding_check(unshaped: list[Session], cfg: ShapingConfig, shaped: list[Session], seed: int) -> dict[str, float]:
    """Mean padding per link-direction and session (released - entering bytes), for comparison
    with Eq. (4) (queue variant) or Remark 2 (arrival variant). Excludes the session-start frames.

    Approximate (review minor 8): cover sites are redrawn when recomputing the entering
    bytes, and bytes still queued at the end (overflow) count as negative padding.
    Compare it with Eq. (4) only on configurations with overflow_rate == 0."""
    if cfg.dp is None:
        return {}
    rng = np.random.default_rng(seed)
    entering = sum(sum(link_direction_bytes(pre_dp(s, cfg, rng)[0])) for s in unshaped)
    start = sum(cfg.dp.session_start_bytes * s.n_sites for s in unshaped)
    released = sum(map(bytes_on_wire, shaped)) - start
    n_linkdir = sum(2 * s.n_sites for s in unshaped)
    return {"dp_padding_per_linkdir_bytes": (released - entering) / max(1, n_linkdir)}


def run_fingerprint(config: dict[str, Any], inputs: list[dict[str, Any]]) -> str:
    source_root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for p in sorted(source_root.rglob("*.py")):
        source.update(str(p.relative_to(source_root)).encode())
        source.update(p.read_bytes())
    identity = {"config": config, "inputs": inputs, "source_sha256": source.hexdigest()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def write_outputs(ctx: RunContext, name: str, rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> Path:
    """Write <out>/<name>.csv with a provenance column and run_metadata.json."""
    for r in rows:
        r["result_status"] = ctx.label()
    path = ctx.out_dir / f"{name}.csv"
    meta = run_metadata(ctx.cfg, ctx.data_kind, ctx.inputs)
    meta["run_fingerprint"] = run_fingerprint(ctx.cfg, meta["inputs"])
    meta["outputs"] = sorted({*(meta.get("outputs") or []), str(path.name)})
    meta["cost_models"] = {k: m.to_dict() for k, m in ctx.cost_models.items()}
    if extra:
        meta.update(extra)
    prev = ctx.out_dir / "run_metadata.json"
    if prev.is_file():
        old = read_json(prev)
        if old.get("run_fingerprint") != meta["run_fingerprint"]:
            raise ValueError("output directory belongs to different code/config/inputs; choose a new output directory")
        meta["outputs"] = sorted(set(old.get("outputs", [])) | set(meta["outputs"]))
    write_csv(rows, path)
    write_run_metadata(ctx.out_dir, meta)
    log.info("wrote %s (%d rows) [%s]", path, len(rows), ctx.label())
    return path


def merge_measured_energy(rows: list[dict[str, Any]], path: str | None, strict: bool = False) -> None:
    """Attach MEASURED extra energy from a testbed CSV (Sec. 6.4), if provided.

    CSV columns: config_label,node,run,extra_joules[,node_class] (idle-subtracted joules per
    session). With a node_class column, a row only receives measurements of its own
    `node_class` (E4, review M4). Adds, per configuration: median and IQR over runs at the
    orchestrator, and the median-over-runs of the most loaded site. Without a file the
    columns are NaN, so no modelled number can be mistaken for a measured one.
    """
    from ..energy.measurements import attach_measurements
    from ..metrics.metrics import median_iqr

    records = read_csv(path) if path else []
    if records and "config_id" in records[0]:
        attach_measurements(rows, records, strict=strict)
        return
    if records and strict:
        raise ValueError("real experiments require paired-v2 energy identity; legacy energy CSV is ambiguous")

    nan = float("nan")
    by: dict[tuple[str, str, str], list[float]] = {}
    if path:
        for d in read_csv(path):
            by.setdefault((d["config_label"], d["node"], d.get("node_class") or "*"), []).append(float(d["extra_joules"]))

    def match(c: str, cls: str, row: dict[str, Any]) -> bool:
        return c == str(row.get("config")) and (cls == "*" or cls == row.get("node_class"))

    for r in rows:
        r["energy_measurement_schema"] = "legacy-unverified" if records else "unavailable"
        orch_vals = [x for (c, n, cls), v in by.items() if n == "orchestrator" and match(c, cls, r) for x in v]
        orch = median_iqr(orch_vals)
        site_meds = [median_iqr(v)["median"] for (c, n, cls), v in by.items() if n.startswith("site") and match(c, cls, r)]
        r["energy_measured_orchestrator_median_J"] = orch["median"]
        r["energy_measured_orchestrator_q25_J"] = orch["q25"]
        r["energy_measured_orchestrator_q75_J"] = orch["q75"]
        r["energy_measured_runs"] = orch["n"]
        r["energy_measured_site_max_median_J"] = max(site_meds) if site_meds else nan


def row_identity(ctx: RunContext, cfg: ShapingConfig, node_class: str | None = None) -> dict[str, Any]:
    if len(ctx.n_values) != 1:
        raise ValueError("energy identity requires one N")
    run = ctx.cfg.get("run", {})
    classes: dict[str, str] = {}
    for s in ctx.sessions:
        for i, c in enumerate(s.meta.get("node_classes", ["default"] * s.n_sites)):
            node = f"site{i}"
            if node in classes and classes[node] != c:
                raise ValueError("node class changes within one deployment")
            classes[node] = node_class or c
    default_id = hashlib.sha256(json.dumps(ctx.cfg, sort_keys=True).encode()).hexdigest()[:16]
    return {"run_id": run.get("id", default_id), "deployment_id": run.get("deployment_id", "single-host-pilot"),
            "N": ctx.n_values[0], "llm": "+".join(sorted({str(s.meta.get("llm", "synthetic")) for s in ctx.sessions})),
            "query_mix": "+".join(str(f) for f in sorted({s.query_family for s in ctx.sessions})),
            "config_id": cfg.config_id(), "node_classes": json.dumps(classes, sort_keys=True)}


def add_co2e(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    """CO2e of the MEASURED extra energy (Sec. 6.4), when `energy.grid_g_per_kwh` is set.

    No default grid factor exists: without one, or without measurements, the columns are NaN.
    """
    e = cfg.get("energy", {})
    factor = e.get("grid_g_per_kwh")
    for r in rows:
        r["grid_factor_source"] = e.get("grid_factor_source", "") if factor is not None else ""
        for src, dst in (("energy_measured_site_max_median_J", "co2e_measured_site_max_g"),
                         ("energy_measured_orchestrator_median_J", "co2e_measured_orchestrator_g")):
            v = r.get(src, float("nan"))
            r[dst] = co2e_grams(float(v), float(factor)) if factor is not None and v == v else float("nan")


__all__ = ["RunContext", "context_from_config", "cost_model_for", "shape_all", "final_test_evaluation",
           "cost_summary", "dp_padding_check", "write_outputs", "merge_measured_energy", "add_co2e", "node_models",
           "write_json"]
