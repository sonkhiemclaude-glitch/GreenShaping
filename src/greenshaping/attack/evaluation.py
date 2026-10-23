"""Question-disjoint splits and attacker scoring (Sec. 6.2).

Three disjoint question sets: attacker training and selection (used by Algorithm 1)
and a final test set used only to report results. For every reported configuration
the attacker is retrained on shaped traffic of the training questions.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..data.schemas import Session
from ..metrics.confidence_intervals import bootstrap_by_group
from ..metrics.metrics import accuracy, advantage_clipped, chance_level, majority_class, normalised_advantage, roc_auc
from .classifiers import RandomForestAttacker, TimeSliceCNN
from .features import build_linkaware_tables, build_tables, build_time_slice_tables

SECRETS = ("participation", "size_bucket", "query_family")


def split_by_question(sessions: list[Session], fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
                      seed: int = 0) -> tuple[list[Session], list[Session], list[Session]]:
    """Attacker-train, selection and final-test sets with disjoint question ids.

    Stratified by query family (review minor 14): the questions of each family are split
    with the given fractions, so each set keeps the family mix of the question bank.
    """
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to 1")
    family_by_question: dict[int, int] = {}
    for session in sessions:
        if session.question_id in family_by_question and family_by_question[session.question_id] != session.query_family:
            raise ValueError("question_id belongs to different families")
        family_by_question[session.question_id] = session.query_family
    if any("partition" in s.meta for s in sessions):
        declared: dict[int, str] = {}
        for s in sessions:
            p = s.meta.get("partition")
            if p not in ("train", "select", "test"):
                raise ValueError("all sessions must have an explicit valid partition")
            if s.question_id in declared and declared[s.question_id] != p:
                raise ValueError("one question cannot belong to multiple partitions")
            declared[s.question_id] = p
        parts_explicit = tuple([s for s in sessions if declared[s.question_id] == p] for p in ("train", "select", "test"))
        if not all(parts_explicit):
            raise ValueError("explicit partition is empty")
        return parts_explicit  # type: ignore[return-value]
    fam_of: dict[int, int] = {}
    for s in sessions:
        if s.question_id in fam_of and fam_of[s.question_id] != s.query_family:
            raise ValueError("question_id belongs to different families")
        fam_of.setdefault(s.question_id, s.query_family)
    qids = sorted(fam_of)
    if len(qids) < 3:
        raise ValueError("need at least 3 distinct questions for three disjoint sets")
    rng = np.random.default_rng(seed)
    parts: list[set[int]] = [set(), set(), set()]
    leftovers: list[int] = []
    for fam in sorted(set(fam_of.values())):
        q = [x for x in qids if fam_of[x] == fam]
        rng.shuffle(q)
        if len(q) < 3:
            leftovers += q
            continue
        n1 = max(1, int(round(fractions[0] * len(q))))
        n2 = min(len(q) - 1, max(n1 + 1, n1 + int(round(fractions[1] * len(q)))))
        parts[0] |= set(q[:n1])
        parts[1] |= set(q[n1:n2])
        parts[2] |= set(q[n2:])
    for x in leftovers:                      # families with < 3 questions: fill the smallest set
        parts[min(range(3), key=lambda i: len(parts[i]) / fractions[i])].add(x)
    if not all(parts):
        raise ValueError("a split is empty; add questions")
    out = tuple([s for s in sessions if s.question_id in p] for p in parts)
    return out  # type: ignore[return-value]


def assert_disjoint(*sets: list[Session]) -> None:
    seen: set[int] = set()
    for s in sets:
        q = {x.question_id for x in s}
        if q & seen:
            raise AssertionError("question sets overlap: attacker or selection would see test questions")
        seen |= q


def _score(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, gte: np.ndarray,
           make_attacker: Callable[[], Any], with_auc: bool, n_boot: int, seed: int,
           gtr: np.ndarray | None = None, sites: np.ndarray | None = None,
           chance_mode: str = "global") -> dict[str, Any]:
    if len(ytr) == 0 or len(yte) == 0:
        return {"acc": float("nan"), "chance": float("nan"), "adv": float("nan"), "adv_raw": float("nan"),
                "auc": float("nan"), "adv_ci_low": float("nan"), "adv_ci_high": float("nan"), "n_test": 0,
                "boot_dropped": 0}
    if len(np.unique(ytr)) < 2:
        pred = np.full(len(yte), majority_class(ytr))
        proba1 = np.zeros(len(yte))
    else:
        clf = make_attacker().fit(Xtr, ytr, groups=gtr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        classes = list(getattr(clf, "classes_", np.unique(ytr)))
        proba1 = proba[:, classes.index(1)] if (with_auc and 1 in classes) else np.zeros(len(yte))
    acc = accuracy(yte, pred)
    def chance(idx: np.ndarray) -> float:
        if chance_mode == "site_conditioned" and sites is not None:
            return sum(np.sum(sites[idx] == s) * chance_level(yte[idx][sites[idx] == s])
                       for s in np.unique(sites[idx])) / len(idx)
        return chance_level(yte[idx])

    ch = chance(np.arange(len(yte)))

    def adv_on(idx: np.ndarray) -> float:
        return normalised_advantage(float((yte[idx] == pred[idx]).mean()), chance(idx))

    rep: dict[str, int] = {}
    lo, hi = bootstrap_by_group(gte, adv_on, n_boot=n_boot, seed=seed, report=rep) if n_boot > 0 else (float("nan"),) * 2
    return {"acc": acc, "chance": ch, "adv": advantage_clipped(acc, ch), "adv_raw": normalised_advantage(acc, ch),
            "auc": roc_auc(yte, proba1) if with_auc else float("nan"),
            "adv_ci_low": lo, "adv_ci_high": hi, "n_test": int(len(yte)), "boot_dropped": rep.get("dropped", 0),
            "chance_global": chance_level(yte), "chance_mode": chance_mode,
            "balanced_accuracy": float(np.mean([np.mean(pred[yte == c] == c) for c in np.unique(yte)])),
            "train_majority_accuracy": float(np.mean(yte == majority_class(ytr))),
            "n_test_classes": len(np.unique(yte)),
            "training_history": getattr(locals().get("clf"), "history_", []),
            "validation_groups": getattr(locals().get("clf"), "validation_groups_", []),
            "training_groups": getattr(locals().get("clf"), "training_groups_", [])}


ATTACKERS = ("rf", "cnn", "rf_linkaware", "identity")


def _evaluate_one(model: str, train: list[Session], test: list[Session], seed: int, n_estimators: int,
                  n_boot: int, cnn: dict[str, Any], chance_mode: str = "global") -> dict[str, dict[str, Any]]:
    make: Callable[[], Any]
    if model in ("rf", "rf_linkaware", "identity"):
        if model == "rf":
            Ltr, Str = build_tables(train)
            Lte, Ste = build_tables(test)
        else:
            Ltr, Str = build_linkaware_tables(train, model == "identity")
            Lte, Ste = build_linkaware_tables(test, model == "identity")

        def make() -> Any:
            return RandomForestAttacker(n_estimators=n_estimators, seed=seed)
    elif model == "cnn":
        sl, ns = float(cnn.get("slice_s", 0.25)), int(cnn.get("n_slices", 120))
        Ltr, Str = build_time_slice_tables(train, sl, ns)
        Lte, Ste = build_time_slice_tables(test, sl, ns)
        kw = {k: v for k, v in cnn.items() if k not in ("slice_s", "n_slices")}

        def make() -> Any:
            return TimeSliceCNN(seed=seed, **kw)
    else:
        raise ValueError(f"unknown attacker {model!r}; choose from {ATTACKERS}")
    sites = np.asarray([i for s in test for i in range(s.n_sites)])
    out = {"participation": _score(Ltr["X"], Ltr["part"], Lte["X"], Lte["part"], Lte["group"], make, True, n_boot, seed,
                                    Ltr["group"], sites, chance_mode)}
    # The size-bucket attacker is trained and scored on links that truly participate (it is
    # conditioned on participation; review minor 12, documented in audits/math_to_code_audit.md).
    mtr, mte = Ltr["part"] == 1, Lte["part"] == 1
    out["size_bucket"] = _score(Ltr["X"][mtr], Ltr["bucket"][mtr], Lte["X"][mte], Lte["bucket"][mte],
                                Lte["group"][mte], make, False, n_boot, seed, Ltr["group"][mtr], sites[mte], chance_mode)
    out["query_family"] = _score(Str["X"], Str["family"], Ste["X"], Ste["family"], Ste["group"], make, False, n_boot, seed,
                                 Str["group"])
    return out


def evaluate_attacker(train: list[Session], test: list[Session], seed: int = 0, n_estimators: int = 200,
                      n_boot: int = 0, models: tuple[str, ...] | list[str] = ("rf",),
                      cnn: dict[str, Any] | None = None, chance_mode: str = "global") -> dict[str, dict[str, Any]]:
    """Train each attacker in `models` on `train` and score all three secrets on `test`.

    For every secret the strongest attacker (highest raw advantage) is reported, with its
    name in "attacker" and every attacker's advantage in "adv_raw_<model>" (Sec. 6.2: the
    reported advantage is the maximum over attackers; review M6).
    """
    assert_disjoint(train, test)
    if not models:
        raise ValueError("at least one attacker model is required")
    if chance_mode not in ("global", "site_conditioned"):
        raise ValueError("chance_mode must be global or site_conditioned")
    per = {m: _evaluate_one(m, train, test, seed, n_estimators, n_boot, cnn or {}, chance_mode) for m in models}
    out: dict[str, dict[str, Any]] = {}
    for secret in SECRETS:
        best_m = max(models, key=lambda m: (-np.inf if np.isnan(per[m][secret]["adv_raw"]) else per[m][secret]["adv_raw"]))
        r: dict[str, Any] = dict(per[best_m][secret])
        r["attacker"] = best_m
        for m in models:
            r[f"adv_raw_{m}"] = per[m][secret]["adv_raw"]
            r[f"auc_{m}"] = per[m][secret]["auc"]
        r["per_model"] = {m: per[m][secret] for m in models}
        out[secret] = r
    return out


def max_advantage(result: dict[str, dict[str, Any]]) -> float:
    """Algorithm 1 line 9: maximum (clipped) advantage over the three secrets.

    Secrets with a NaN advantage (e.g. no participating link in the test set) are
    ignored; `n_secrets_scored` reports how many entered the maximum (review minor 10).
    """
    vals = [v["adv"] for v in result.values() if not np.isnan(v["adv"])]
    return max(vals) if vals else float("nan")


def n_secrets_scored(result: dict[str, dict[str, Any]]) -> int:
    return sum(1 for v in result.values() if not np.isnan(v["adv"]))


def attack_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """evaluate_attacker keyword arguments from the `attack` config section."""
    a = cfg.get("attack", {})
    return {"n_estimators": int(a.get("n_estimators", 200)), "models": tuple(a.get("models", ["rf"])),
            "cnn": dict(a.get("cnn", {})), "chance_mode": a.get("chance_mode", "global")}
