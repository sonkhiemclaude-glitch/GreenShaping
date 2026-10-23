import numpy as np
import pytest

from greenshaping.energy.carbon import co2e_grams
from greenshaping.energy.cost_model import CostModel, extra_energy, node_energy_raw, union_length
from greenshaping.energy.fit import Benchmark, fit_cost_model
from greenshaping.energy.meter import extra_energy_joules, integrate_power, load_trace
from greenshaping.energy.rapl import RaplZone, discover_zones, energy_delta_uj, package_and_dram, session_energy_joules, snapshot
from greenshaping.shaping.dp_release import dp_release
from greenshaping.shaping.padding import pad_session


def test_union_length():
    assert union_length([0, 0.05, 1.0], 0.1) == pytest.approx(0.25)
    assert union_length([], 0.1) == 0.0


def test_eq3_padding_adds_only_byte_terms(session_factory):
    m = CostModel("t", 1e-6, 2e-6, 1.0, 0.1)
    s = session_factory([1000, 0], [3000, 0])
    sp, _ = pad_session(s, 4.0, 512)
    e = extra_energy(sp, s, m)
    tx_extra = sum(x.size for x in sp.link(0) if x.direction == 1) - 3000
    rx_extra = sum(x.size for x in sp.link(0) if x.direction == 0) - 1000
    assert e["site0"] == pytest.approx(1e-6 * tx_extra + 2e-6 * rx_extra)
    assert e["site1"] == 0.0


def test_eq3_dp_ticks_are_awake_time(session_factory):
    m = CostModel("t", 0.0, 0.0, 1.0, 0.1)
    s = session_factory([1000, 0], [3000, 0])
    sh, _ = dp_release(s, 1.0, 10.0, 1.0, 4096, rng=np.random.default_rng(0))
    assert node_energy_raw(sh, m)["site1"] == pytest.approx(10 * 0.1)


def test_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    true = CostModel("x", 3e-8, 2e-8, 4.0, 0.05)
    rows = []
    for _ in range(30):
        tx, rx = int(rng.integers(1e4, 1e7)), int(rng.integers(1e4, 1e7))
        times = tuple(sorted(rng.uniform(0, 10, size=int(rng.integers(1, 60)))))
        e = true.e_tx * tx + true.e_rx * rx + true.p_aw * union_length(list(times), true.t_tail)
        rows.append(Benchmark(tx, rx, times, e))
    res = fit_cost_model("x", rows)
    assert res.model.t_tail == 0.05 and res.model.fitted
    assert res.model.e_tx == pytest.approx(3e-8, rel=1e-3) and res.model.p_aw == pytest.approx(4.0, rel=1e-3)
    assert res.relative_rmse < 1e-6
    with pytest.raises(ValueError):
        fit_cost_model("x", rows[:3])


def test_rapl_wraparound_and_fake_sysfs(tmp_path):
    assert energy_delta_uj(100, 300, 1000) == 200
    assert energy_delta_uj(900, 100, 1000) == 200
    with pytest.raises(ValueError):
        energy_delta_uj(10, 2000, 1000)
    z = tmp_path / "intel-rapl:0"
    z.mkdir()
    (z / "name").write_text("package-0\n")
    (z / "max_energy_range_uj").write_text("1000000\n")
    (z / "energy_uj").write_text("999000\n")
    zones = discover_zones(tmp_path)
    s0 = snapshot(zones)
    (z / "energy_uj").write_text("4000\n")
    s1 = snapshot(zones)
    out = session_energy_joules(s0, s1, zones, {"package-0@intel-rapl:0": 0.001}, 2.0)
    assert out["package-0@intel-rapl:0"] == pytest.approx(0.005 - 0.002)
    assert discover_zones(tmp_path / "missing") == []


def test_meter_integration(tmp_path):
    p = tmp_path / "trace.csv"
    p.write_text("timestamp_s,power_w\n0,2\n1,2\n2,4\n3,4\n")
    t, w = load_trace(p)
    assert integrate_power(t, w, 0.0, 1.0) == pytest.approx(2.0)
    assert integrate_power(t, w, 1.0, 2.0) == pytest.approx(3.0)
    assert extra_energy_joules(t, w, 0.0, 3.0, idle_power_w=2.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        integrate_power(t, w, 0.0, 5.0)


def test_co2e():
    assert co2e_grams(3.6e6, 500.0) == pytest.approx(500.0)
    with pytest.raises(ValueError):
        co2e_grams(1.0, -1)


def test_rapl_sum_only_package_and_dram(tmp_path):
    zs = [RaplZone(tmp_path, n, 1) for n in ("package-0", "core", "uncore", "dram", "package-1")]
    assert [z.name for z in package_and_dram(zs)] == ["package-0", "dram", "package-1"]
