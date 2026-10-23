import math

import pytest

from greenshaping.analysis import bounds as b


def test_prop2_required_cover_and_floor():
    assert b.required_cover_probability(0.75, 0.5) == pytest.approx(0.5)
    assert b.any_traffic_detector_accuracy(0.5, 0.5) == pytest.approx(0.75)
    assert b.cover_energy_floor(0.6, 2.0) == pytest.approx(0.8)
    assert b.network_cover_floor(0.6, [2.0] * 5) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        b.required_cover_probability(0.4, 0.5)   # alpha below max(pi, 1-pi)


def test_prop3_accounting_and_accuracy():
    assert b.dp_epsilon_total(1.0, 2) == 2.0
    assert b.dp_accuracy_bound(0.0, 0.5) == pytest.approx(0.5)
    assert b.dp_accuracy_bound(math.log(3), 0.5) == pytest.approx(0.75)
    assert b.dp_accuracy_bound(1.0, 0.8) == pytest.approx(0.8 * math.e / (0.8 * math.e + 0.2))
    assert b.dp_accuracy_bound(1.0, 0.2) == b.dp_accuracy_bound(1.0, 0.8)


def test_eq4_values_from_paper_text():
    # H = 20 s, Delta = 64 KiB, eps_dir = 1: about 13 MB at tau = 1 s and 1.3 GB at tau = 0.1 s
    assert b.expected_dp_padding_bytes(20, 1.0, 65536, 1.0) == pytest.approx(13.1e6, rel=0.01)
    assert b.expected_dp_padding_bytes(20, 0.1, 65536, 1.0) == pytest.approx(1.31e9, rel=0.01)


def test_prop1_mi_bits():
    assert b.padding_mi_bound_bits(1024, 1024**2, 2.0, 512) == pytest.approx(math.log2(11))
