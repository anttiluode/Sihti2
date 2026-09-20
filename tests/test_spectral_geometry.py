import numpy as np

from spectral_geometry import (
    apply_lazy,
    dynamics,
    graph_from_rgb,
    heat_trace,
    iid_rgb,
    pi_rms,
    relaxation_time,
    small_spectrum,
)


def test_uniform_self_graph_equals_lattice():
    x = np.full((12, 14, 3), 0.5)
    a = graph_from_rgb(x, sigma=5.0, radius=1.5)
    b = graph_from_rgb(x, sigma=5.0, radius=1.5, lattice=True)
    assert np.allclose(a.weights, b.weights, atol=0.0, rtol=0.0)


def test_shuffling_preserves_exact_weight_multiset():
    x = iid_rgb(12, 4)
    a = graph_from_rgb(x, sigma=5.0, radius=1.5)
    b = graph_from_rgb(x, sigma=5.0, radius=1.5, shuffle_seed=99)
    assert np.array_equal(np.sort(a.weights), np.sort(b.weights))
    assert not np.array_equal(a.weights, b.weights)


def test_lazy_diffusion_contracts_stationary_measure_rms():
    x = iid_rgb(12, 3)
    g = graph_from_rgb(x, sigma=5.0, radius=1.5)
    vals = [pi_rms(x, g.degree)]
    y = x.copy()
    for _ in range(12):
        y = apply_lazy(y, g)
        vals.append(pi_rms(y, g.degree))
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_connected_graph_has_positive_spectral_gap():
    x = iid_rgb(12, 8)
    g = graph_from_rgb(x, sigma=5.0, radius=1.5)
    mu = small_spectrum(g, 6)
    assert abs(mu[0]) < 1e-7
    assert mu[1] > 0
    tau = relaxation_time(mu[1])
    assert np.isfinite(tau)
    assert tau > 0


def test_heat_trace_starts_below_dimension_and_decays_in_expectation():
    x = iid_rgb(10, 11)
    g = graph_from_rgb(x, sigma=8.0, radius=1.5)
    rows = heat_trace(g, max_depth=8, probes=64, seed=2)
    assert rows[0]["depth"] == 1
    assert rows[-1]["depth"] == 8
    n = x.shape[0] * x.shape[1]
    assert all(0.0 <= r["trace"] <= n + 1e-9 for r in rows)
    assert rows[-1]["trace"] <= rows[0]["trace"] + 1e-9


def test_dynamics_reports_geometry_native_energy():
    x = iid_rgb(10, 12)
    g = graph_from_rgb(x, sigma=5.0, radius=1.5)
    rows = dynamics(x, g, max_depth=8)
    assert [r["depth"] for r in rows] == [0, 1, 2, 4, 8]
    assert rows[-1]["pi_rms"] <= rows[0]["pi_rms"] + 1e-12
    assert rows[-1]["edge_disagreement"] <= rows[0]["edge_disagreement"] + 1e-12
