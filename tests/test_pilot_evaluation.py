import importlib.util
from pathlib import Path

import numpy as np

from greenshaping.data.schemas import DOWN, UP, Message, Session


def evaluator():
    spec = importlib.util.spec_from_file_location("pilot_evaluation", Path(__file__).resolve().parents[1] /
                                                "scripts/evaluate_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linkaware_features_use_observables_and_public_identity_only():
    ev = evaluator()
    messages = [Message(0.1, 0, UP, 300), Message(0.2, 0, DOWN, 900)]
    a = Session(0, 1, 2, 0, [1, 0], [0, -1], messages)
    b = Session(0, 99, 2, 3, [0, 1], [-1, 2], messages)
    la, sa = ev.linkaware_tables([a])
    lb, sb = ev.linkaware_tables([b])
    assert np.array_equal(la["X"], lb["X"])
    assert np.array_equal(sa["X"], sb["X"])
    assert not np.array_equal(la["X"][0], la["X"][1])  # known link identity retained


def test_score_exposes_prior_shift_instead_of_calling_negative_advantage_privacy():
    ev = evaluator()
    y, pred = np.array([1, 1, 1, 0]), np.zeros(4, int)
    result = ev.score(y, pred, np.zeros(4), np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), 0, 0)
    assert result["accuracy"] == 0.25 and result["chance_test"] == 0.75
    assert result["balanced_accuracy"] == 0.5 and result["auc"] == 0.5
    assert result["adv_raw"] == -2 and result["adv_clipped"] == 0


def test_candidate_list_contains_original_controller_choices_and_no_duplicates():
    ev = evaluator()
    sample = Session(0, 0, 1, 0, [1], [0], [Message(0.1, 0, UP, 500), Message(0.2, 0, DOWN, 900)])
    configs = ev.candidates([sample])
    assert len(configs) == 10 and len({c.key() for c in configs}) == 10
    assert configs[0].label() == "none"
    assert any(c.rounds and c.padding and c.dp for c in configs)
