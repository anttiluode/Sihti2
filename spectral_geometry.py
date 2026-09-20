#!/usr/bin/env python3
"""Sihti2: spectral geometry of noise-written diffusion.

The experiment is intentionally exploratory. It compares four matched controls:

    self        signal writes its own conductance graph
    independent another IID draw writes the graph
    shuffled    self graph's edge weights are permuted over the same edge set
    lattice     colour term removed; only spatial coupling remains

It measures conductance heterogeneity, the small normalized-Laplacian spectrum,
relaxation time, stationary-measure variance, graph edge disagreement, and a
Hutchinson estimate of Tr(T^t) for the symmetric lazy operator.

Run:
    python spectral_geometry.py
    python spectral_geometry.py --seeds 8 --sigmas 3 5 8 12 --size 40
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
CONTROLS = ("self", "independent", "shuffled", "lattice")


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ RGB2XYZ.T) / WHITE_D65
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def neighbor_edges(h: int, w: int, radius: float = 1.5):
    """Undirected local edge list, matching Sihti's half-plane construction."""
    idx = np.arange(h * w).reshape(h, w)
    r = int(np.floor(radius))
    ii, jj, s2 = [], [], []
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx <= 0:
                continue
            dist2 = dy * dy + dx * dx
            if dist2 > radius * radius + 1e-12:
                continue
            y1 = h - dy
            x0, x1 = max(0, -dx), min(w, w - dx)
            if y1 <= 0 or x1 <= x0:
                continue
            i = idx[0:y1, x0:x1].ravel()
            j = idx[dy : dy + y1, x0 + dx : x1 + dx].ravel()
            ii.append(i)
            jj.append(j)
            s2.append(np.full(i.size, float(dist2)))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(s2)


@dataclass
class Graph:
    W: sp.csr_matrix
    degree: np.ndarray
    i: np.ndarray
    j: np.ndarray
    weights: np.ndarray
    floor: float


def graph_from_rgb(
    rgb: np.ndarray,
    sigma: float,
    radius: float = 1.5,
    floor: float = 1e-6,
    lattice: bool = False,
    shuffle_seed: int | None = None,
) -> Graph:
    h, w = rgb.shape[:2]
    i, j, s2 = neighbor_edges(h, w, radius)
    sigma_x = max(1.0, float(radius))
    spatial = np.exp(-s2 / (2.0 * sigma_x * sigma_x))
    if lattice:
        ew = spatial.copy()
    else:
        lab = srgb_to_lab(rgb).reshape(h * w, 3)
        d2 = np.sum((lab[i] - lab[j]) ** 2, axis=1)
        ew = np.exp(-d2 / (2.0 * float(sigma) ** 2)) * spatial
        ew = np.maximum(ew, floor)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        ew = ew[rng.permutation(ew.size)]

    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([ew, ew])
    W = sp.coo_matrix((data, (rows, cols)), shape=(h * w, h * w)).tocsr()
    degree = np.asarray(W.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise RuntimeError("graph contains a zero-degree vertex")
    return Graph(W=W, degree=degree, i=i, j=j, weights=ew, floor=floor)


def normalized_symmetric_operator(g: Graph, laziness: float = 0.5):
    inv_sqrt = 1.0 / np.sqrt(g.degree)
    S = sp.diags(inv_sqrt) @ g.W @ sp.diags(inv_sqrt)
    S = (S + S.T) * 0.5
    n = g.W.shape[0]
    lazy = (1.0 - laziness) * sp.identity(n, format="csr") + laziness * S
    return S.tocsr(), lazy.tocsr()


def small_spectrum(g: Graph, k: int = 8):
    S, _ = normalized_symmetric_operator(g)
    n = S.shape[0]
    L = sp.identity(n, format="csr") - S
    kk = max(2, min(int(k), n - 2))
    v0 = np.random.default_rng(0).standard_normal(n)
    try:
        mu = eigsh(L, k=kk, sigma=-1e-7, which="LM", v0=v0, tol=1e-8, maxiter=20000, return_eigenvectors=False)
    except Exception:
        mu = eigsh(L, k=kk, which="SM", v0=v0, tol=1e-8, maxiter=30000, return_eigenvectors=False)
    mu = np.sort(np.maximum(np.asarray(mu, dtype=np.float64), 0.0))
    return mu


def relaxation_time(mu: float, laziness: float = 0.5) -> float:
    gain = 1.0 - laziness * float(mu)
    if gain >= 1.0 - 1e-15:
        return math.inf
    if gain <= 0:
        return 0.0
    return float(-1.0 / math.log(gain))


def pi_rms(x: np.ndarray, degree: np.ndarray) -> float:
    X = np.asarray(x, dtype=np.float64).reshape(degree.size, -1)
    pi = degree / degree.sum()
    mean = np.sum(pi[:, None] * X, axis=0)
    var = np.sum(pi[:, None] * (X - mean) ** 2) / X.shape[1]
    return float(np.sqrt(max(var, 0.0)))


def plain_centered_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=(0, 1), keepdims=True)
    return float(np.sqrt(np.mean((x - mean) ** 2)))


def edge_disagreement(x: np.ndarray, g: Graph) -> float:
    X = np.asarray(x, dtype=np.float64).reshape(g.degree.size, -1)
    d2 = np.mean((X[g.i] - X[g.j]) ** 2, axis=1)
    return float(np.sum(g.weights * d2) / np.sum(g.weights))


def apply_lazy(x: np.ndarray, g: Graph, laziness: float = 0.5) -> np.ndarray:
    X = np.asarray(x, dtype=np.float64).reshape(g.degree.size, -1)
    Y = (1.0 - laziness) * X + laziness * ((g.W @ X) / g.degree[:, None])
    return Y.reshape(x.shape)


def dyadic_depths(max_depth: int):
    out = [0]
    d = 1
    while d <= max_depth:
        out.append(d)
        d *= 2
    if out[-1] != max_depth:
        out.append(max_depth)
    return sorted(set(out))


def dynamics(rgb: np.ndarray, g: Graph, max_depth: int, laziness: float = 0.5):
    depths = dyadic_depths(max_depth)
    want = set(depths)
    x = np.array(rgb, dtype=np.float64, copy=True)
    rows = []
    for t in range(max_depth + 1):
        if t in want:
            rows.append(
                {
                    "depth": int(t),
                    "plain_rms": plain_centered_rms(x),
                    "pi_rms": pi_rms(x, g.degree),
                    "edge_disagreement": edge_disagreement(x, g),
                }
            )
        if t < max_depth:
            x = apply_lazy(x, g, laziness)
    return rows


def heat_trace(
    g: Graph,
    max_depth: int,
    probes: int = 24,
    seed: int = 1234,
    laziness: float = 0.5,
):
    """Hutchinson trace estimate for the symmetric lazy operator."""
    _, T = normalized_symmetric_operator(g, laziness)
    n = T.shape[0]
    depths = [d for d in dyadic_depths(max_depth) if d > 0]
    want = set(depths)
    rng = np.random.default_rng(seed)
    Q = rng.choice(np.array([-1.0, 1.0]), size=(n, int(probes)))
    Y = Q.copy()
    traces = []
    for t in range(1, max_depth + 1):
        Y = T @ Y
        if t in want:
            vals = np.sum(Q * Y, axis=0)
            raw = float(np.mean(vals))
            traces.append({"depth": int(t), "trace": raw})
    for row in traces:
        row["excess"] = float(max(row["trace"] - 1.0, 1e-12))
    for k, row in enumerate(traces):
        row["spectral_dimension"] = None
        if k == 0:
            continue
        a, b = traces[k - 1], row
        if a["excess"] > 0 and b["excess"] > 0:
            slope = math.log(b["excess"] / a["excess"]) / math.log(b["depth"] / a["depth"])
            row["spectral_dimension"] = float(-2.0 * slope)
    return traces


def conductance_summary(g: Graph):
    w = g.weights
    q = np.quantile(w, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "edge_count": int(w.size),
        "floor_fraction": float(np.mean(w <= g.floor * (1.0 + 1e-12))),
        "weight_quantiles": {
            "q0": float(q[0]),
            "q01": float(q[1]),
            "q10": float(q[2]),
            "q50": float(q[3]),
            "q90": float(q[4]),
            "q99": float(q[5]),
            "q100": float(q[6]),
        },
        "degree_min": float(g.degree.min()),
        "degree_median": float(np.median(g.degree)),
        "degree_max": float(g.degree.max()),
    }


def iid_rgb(size: int, seed: int):
    return np.random.default_rng(seed).random((size, size, 3))


def make_control_graph(signal: np.ndarray, control: str, sigma: float, seed: int, radius: float, floor: float):
    if control == "self":
        return graph_from_rgb(signal, sigma=sigma, radius=radius, floor=floor)
    if control == "independent":
        graph_rgb = iid_rgb(signal.shape[0], seed + 1_000_003)
        return graph_from_rgb(graph_rgb, sigma=sigma, radius=radius, floor=floor)
    if control == "shuffled":
        return graph_from_rgb(signal, sigma=sigma, radius=radius, floor=floor, shuffle_seed=seed + 2_000_003)
    if control == "lattice":
        return graph_from_rgb(signal, sigma=sigma, radius=radius, floor=floor, lattice=True)
    raise ValueError(control)


def analyze_case(
    seed: int,
    sigma: float,
    control: str,
    size: int,
    max_depth: int,
    radius: float,
    floor: float,
    eig_k: int,
    probes: int,
):
    signal = iid_rgb(size, seed)
    g = make_control_graph(signal, control, sigma, seed, radius, floor)
    mu = small_spectrum(g, eig_k)
    mu2 = float(mu[1]) if len(mu) > 1 else math.nan
    dyn = dynamics(signal, g, max_depth)
    ht = heat_trace(g, max_depth, probes=probes, seed=seed + 31_337)
    return {
        "seed": int(seed),
        "sigma": float(sigma),
        "control": control,
        "size": int(size),
        "conductance": conductance_summary(g),
        "spectrum": {
            "mu": [float(v) for v in mu],
            "mu2": mu2,
            "tau2": relaxation_time(mu2),
        },
        "dynamics": dyn,
        "heat_trace": ht,
        "retention_pi_rms": float(dyn[-1]["pi_rms"] / max(dyn[0]["pi_rms"], 1e-15)),
        "retention_edge_disagreement": float(
            dyn[-1]["edge_disagreement"] / max(dyn[0]["edge_disagreement"], 1e-15)
        ),
    }


def run_sweep(args):
    rows = []
    for sigma in args.sigmas:
        for seed in range(args.seeds):
            for control in CONTROLS:
                print(f"sigma={sigma:g} seed={seed} control={control}", flush=True)
                rows.append(
                    analyze_case(
                        seed=seed,
                        sigma=sigma,
                        control=control,
                        size=args.size,
                        max_depth=args.max_depth,
                        radius=args.radius,
                        floor=args.floor,
                        eig_k=args.eig_k,
                        probes=args.probes,
                    )
                )
    return {
        "description": "Exploratory Sihti2 random-conductance spectral geometry sweep.",
        "claim_boundary": "Visual domains are metastable graph structure, not semantic objects or evidence of fractality by themselves.",
        "parameters": {
            "seeds": args.seeds,
            "sigmas": args.sigmas,
            "size": args.size,
            "max_depth": args.max_depth,
            "radius": args.radius,
            "floor": args.floor,
            "eig_k": args.eig_k,
            "probes": args.probes,
        },
        "rows": rows,
    }


def finite(v):
    return np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)


def plot_report(report, path: Path):
    import matplotlib.pyplot as plt

    rows = report["rows"]
    sigmas = report["parameters"]["sigmas"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for control in CONTROLS:
        med_tau, med_ret = [], []
        for sigma in sigmas:
            rr = [r for r in rows if r["control"] == control and r["sigma"] == sigma]
            tau = finite([r["spectrum"]["tau2"] for r in rr])
            ret = finite([r["retention_pi_rms"] for r in rr])
            med_tau.append(float(np.median(tau)) if tau.size else np.nan)
            med_ret.append(float(np.median(ret)) if ret.size else np.nan)
        axes[0, 0].plot(sigmas, med_tau, marker="o", label=control)
        axes[0, 1].plot(sigmas, med_ret, marker="o", label=control)

    self_floor = []
    for sigma in sigmas:
        rr = [r for r in rows if r["control"] == "self" and r["sigma"] == sigma]
        self_floor.append(float(np.median([r["conductance"]["floor_fraction"] for r in rr])))
    axes[1, 0].plot(sigmas, self_floor, marker="o")
    axes[1, 0].set_ylim(-0.02, 1.02)

    target_sigma = min(sigmas, key=lambda s: abs(float(s) - 5.0))
    for control in CONTROLS:
        r = next(
            x
            for x in rows
            if x["seed"] == 0 and x["control"] == control and x["sigma"] == target_sigma
        )
        x = [q["depth"] for q in r["heat_trace"]]
        y = [q["excess"] for q in r["heat_trace"]]
        axes[1, 1].loglog(x, y, marker=".", label=control)

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Median slow relaxation time")
    axes[0, 0].set_xlabel("self-EQ sigma")
    axes[0, 0].set_ylabel("tau2 (passes)")

    axes[0, 1].set_title("Median retained stationary-measure RMS")
    axes[0, 1].set_xlabel("self-EQ sigma")
    axes[0, 1].set_ylabel("RMS(max depth) / RMS(0)")

    axes[1, 0].set_title("Self graph: fraction of edges at floor")
    axes[1, 0].set_xlabel("self-EQ sigma")
    axes[1, 0].set_ylabel("floor fraction")

    axes[1, 1].set_title(f"Heat-trace excess, seed 0, sigma={target_sigma:g}")
    axes[1, 1].set_xlabel("Markov time")
    axes[1, 1].set_ylabel("Tr(T^t) - 1")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Sihti2 — noise-written random conductance geometry")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--sigmas", type=float, nargs="+", default=[3.0, 5.0, 8.0, 12.0])
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--max-depth", type=int, default=128)
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--floor", type=float, default=1e-6)
    p.add_argument("--eig-k", type=int, default=8)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--out", default="results/spectral_geometry.json")
    p.add_argument("--figure", default="figures/spectral_geometry.png")
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.seeds = max(1, int(args.seeds))
    args.size = max(8, min(int(args.size), 96))
    args.max_depth = max(1, min(int(args.max_depth), 4096))
    args.probes = max(2, int(args.probes))
    report = run_sweep(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {out}")

    if not args.no_figure:
        fig = Path(args.figure)
        plot_report(report, fig)
        print(f"wrote {fig}")


if __name__ == "__main__":
    main()
