"""percolab: correlation graphs, centrality, percolation and risk diffusion.

Percolation and diffusion are Monte Carlo procedures with explicit seeds; their
outputs are deterministic for a given seed on a given platform.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import (
    as_f64_1d,
    as_f64_2d,
    as_size,
    as_square,
    as_u32_1d,
    as_u64,
    double_ptr,
    empty_f64,
    int32_ptr,
    uint32_ptr,
)
from ._ffi import check, get_lib
from . import qflux


@dataclass(frozen=True)
class GraphComponents:
    labels: np.ndarray
    n_components: int
    largest_size: int
    sizes: np.ndarray

    @property
    def giant_fraction(self) -> float:
        if self.labels.size == 0:
            return float("nan")
        return self.largest_size / self.labels.size


@dataclass(frozen=True)
class MSTResult:
    u: np.ndarray
    v: np.ndarray
    weight: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return an ``(n-1, 3)`` array with columns u, v, weight."""
        return np.column_stack((self.u, self.v, self.weight))


@dataclass(frozen=True)
class PercolationScan:
    probabilities: np.ndarray
    giant_mean: np.ndarray
    giant_std: np.ndarray
    susceptibility: np.ndarray


@dataclass(frozen=True)
class DiffusionResult:
    mean_affected: np.ndarray
    infection_frequency: np.ndarray

    def final_affected(self) -> float:
        return float(self.mean_affected[-1])


@dataclass(frozen=True)
class CorrelationNetwork:
    correlation: np.ndarray
    adjacency: np.ndarray
    distance: np.ndarray
    components: GraphComponents
    strength: np.ndarray
    eigenvector: np.ndarray
    clustering: np.ndarray
    average_clustering: float
    mst: MSTResult
    density: float
    n_edges: int


def _components_from_summary(
    labels: np.ndarray, n_components: int, largest_size: int
) -> GraphComponents:
    sizes = np.bincount(labels, minlength=n_components).astype(np.int64)
    return GraphComponents(
        labels=np.asarray(labels, dtype=np.int32),
        n_components=int(n_components),
        largest_size=int(largest_size),
        sizes=sizes,
    )


def correlation_to_distance(correlation: Any) -> np.ndarray:
    """Mantegna distance ``sqrt(2 * (1 - rho))``."""
    matrix = as_square(correlation, "correlation")
    n = matrix.shape[0]
    out = empty_f64(n * n)
    check(
        get_lib().rflux_corr_to_distance(
            double_ptr(matrix), n, double_ptr(out)
        ),
        "rflux_corr_to_distance",
    )
    return out.reshape(n, n)


def threshold_components(adjacency: Any, threshold: float) -> GraphComponents:
    """Connected components of edges with weight >= threshold."""
    matrix = as_square(adjacency, "adjacency")
    n = matrix.shape[0]
    labels = np.empty(n, dtype=np.int32)
    n_components = ctypes.c_size_t(0)
    largest = ctypes.c_size_t(0)
    check(
        get_lib().rflux_graph_components(
            double_ptr(matrix),
            n,
            float(threshold),
            int32_ptr(labels),
            ctypes.byref(n_components),
            ctypes.byref(largest),
        ),
        "rflux_graph_components",
    )
    return _components_from_summary(labels, n_components.value, largest.value)


def network_strength(adjacency: Any) -> np.ndarray:
    """Weighted degree (sum of off-diagonal non-negative weights)."""
    matrix = as_square(adjacency, "adjacency")
    out = empty_f64(matrix.shape[0])
    check(
        get_lib().rflux_network_strength(
            double_ptr(matrix), matrix.shape[0], double_ptr(out)
        ),
        "rflux_network_strength",
    )
    return out


def centrality(adjacency: Any, kind: str = "eigenvector") -> np.ndarray:
    """``degree`` or L2-normalised ``eigenvector`` centrality."""
    matrix = as_square(adjacency, "adjacency")
    kinds = {"degree": 0, "eigenvector": 1}
    try:
        value = kinds[str(kind).lower()]
    except KeyError as exc:
        raise ValueError("kind must be 'degree' or 'eigenvector'") from exc
    out = empty_f64(matrix.shape[0])
    check(
        get_lib().rflux_network_centrality(
            double_ptr(matrix), matrix.shape[0], value, double_ptr(out)
        ),
        "rflux_network_centrality",
    )
    return out


def clustering(adjacency: Any) -> np.ndarray:
    """Local unweighted clustering coefficient; NaN when degree < 2."""
    matrix = as_square(adjacency, "adjacency")
    out = empty_f64(matrix.shape[0])
    check(
        get_lib().rflux_network_clustering(
            double_ptr(matrix), matrix.shape[0], double_ptr(out)
        ),
        "rflux_network_clustering",
    )
    return out


def mst_from_distance(distance: Any) -> MSTResult:
    """Minimum spanning tree of a complete symmetric distance matrix."""
    matrix = as_square(distance, "distance")
    n = matrix.shape[0]
    n_edges = ctypes.c_size_t(0)
    if n <= 1:
        check(
            get_lib().rflux_network_mst(
                double_ptr(matrix), n, None, None, None, ctypes.byref(n_edges)
            ),
            "rflux_network_mst",
        )
        empty_u32 = np.empty(0, dtype=np.uint32)
        empty_f64_array = np.empty(0, dtype=np.float64)
        return MSTResult(empty_u32, empty_u32.copy(), empty_f64_array)
    u = np.empty(n - 1, dtype=np.uint32)
    v = np.empty(n - 1, dtype=np.uint32)
    weights = empty_f64(n - 1)
    check(
        get_lib().rflux_network_mst(
            double_ptr(matrix),
            n,
            uint32_ptr(u),
            uint32_ptr(v),
            double_ptr(weights),
            ctypes.byref(n_edges),
        ),
        "rflux_network_mst",
    )
    count = int(n_edges.value)
    return MSTResult(u[:count].copy(), v[:count].copy(), weights[:count].copy())


def mst_from_correlation(correlation: Any) -> MSTResult:
    """Correlation-distance minimum spanning tree."""
    return mst_from_distance(correlation_to_distance(correlation))


def _percolation_scan(
    adjacency: Any,
    probabilities: Any,
    repetitions: int,
    seed: int,
    kind: str,
) -> PercolationScan:
    matrix = as_square(adjacency, "adjacency")
    probability_array = as_f64_1d(probabilities, "probabilities")
    n_probabilities = probability_array.size
    if n_probabilities == 0:
        raise ValueError("probabilities must not be empty")
    repetitions = as_size(repetitions, "repetitions")
    seed = as_u64(seed, "seed")
    mean = empty_f64(n_probabilities)
    std = empty_f64(n_probabilities)
    susceptibility = empty_f64(n_probabilities)
    function = (
        get_lib().rflux_bond_percolation
        if kind == "bond"
        else get_lib().rflux_site_percolation
    )
    check(
        function(
            double_ptr(matrix),
            matrix.shape[0],
            double_ptr(probability_array),
            n_probabilities,
            repetitions,
            seed,
            double_ptr(mean),
            double_ptr(std),
            double_ptr(susceptibility),
        ),
        f"rflux_{kind}_percolation",
    )
    return PercolationScan(
        probabilities=probability_array.copy(),
        giant_mean=mean,
        giant_std=std,
        susceptibility=susceptibility,
    )


def bond_percolation(
    adjacency: Any,
    probabilities: Any,
    repetitions: int = 100,
    seed: int = 42,
) -> PercolationScan:
    """Bond percolation over edges where ``adjacency > 0``."""
    return _percolation_scan(adjacency, probabilities, repetitions, seed, "bond")


def site_percolation(
    adjacency: Any,
    probabilities: Any,
    repetitions: int = 100,
    seed: int = 42,
) -> PercolationScan:
    """Site percolation; giant fraction is relative to the original n."""
    return _percolation_scan(adjacency, probabilities, repetitions, seed, "site")


def risk_diffusion(
    adjacency: Any,
    seeds: Any,
    *,
    beta: float = 0.3,
    gamma: float = 0.1,
    steps: int = 30,
    repetitions: int = 100,
    seed: int = 42,
) -> DiffusionResult:
    """Stochastic SIR risk diffusion on a weighted graph.

    ``mean_affected[t]`` is the mean fraction ever infected/recovered at step t;
    ``infection_frequency[i]`` is the probability that node i was ever infected.
    """
    matrix = as_square(adjacency, "adjacency")
    n = matrix.shape[0]
    steps = as_size(steps, "steps")
    repetitions = as_size(repetitions, "repetitions")
    seed = as_u64(seed, "seed")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if seeds is None:
        seed_array = np.empty(0, dtype=np.uint32)
    else:
        raw_seeds = np.asarray(seeds)
        seed_array = (
            np.empty(0, dtype=np.uint32)
            if raw_seeds.size == 0
            else as_u32_1d(raw_seeds, "seeds")
        )
    if seed_array.size:
        if int(seed_array.max()) >= n:
            raise ValueError("seed node index is out of range")
        seed_pointer = uint32_ptr(seed_array)
    else:
        seed_pointer = None
    mean_affected = empty_f64(steps + 1)
    infection_frequency = empty_f64(n)
    check(
        get_lib().rflux_risk_diffusion(
            double_ptr(matrix),
            n,
            seed_pointer,
            seed_array.size,
            float(beta),
            float(gamma),
            steps,
            repetitions,
            seed,
            double_ptr(mean_affected),
            double_ptr(infection_frequency),
        ),
        "rflux_risk_diffusion",
    )
    return DiffusionResult(mean_affected, infection_frequency)


def correlation_adjacency(
    correlation: Any, threshold: float = 0.5, *, weighted: bool = True
) -> np.ndarray:
    """Threshold a correlation matrix into a non-negative adjacency matrix."""
    matrix = as_square(correlation, "correlation")
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "correlation matrix contains NaN/inf; drop constant or "
            "insufficient series first"
        )
    mask = matrix >= float(threshold)
    np.fill_diagonal(mask, False)
    adjacency = np.zeros_like(matrix)
    if weighted:
        adjacency[mask] = np.maximum(matrix[mask], 0.0)
    else:
        adjacency[mask] = 1.0
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return np.ascontiguousarray(adjacency)


def network_from_correlation(
    correlation: Any, threshold: float = 0.5
) -> CorrelationNetwork:
    """Build a full correlation network summary from a correlation matrix."""
    matrix = np.ascontiguousarray(
        np.asarray(correlation, dtype=np.float64), dtype=np.float64
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be a square matrix")
    n = matrix.shape[0]
    adjacency = correlation_adjacency(matrix, threshold)
    distance = correlation_to_distance(matrix)
    components = threshold_components(adjacency, float(threshold))
    strength_values = network_strength(adjacency)
    eigenvector_values = centrality(adjacency, "eigenvector")
    clustering_values = clustering(adjacency)
    finite_clustering = clustering_values[np.isfinite(clustering_values)]
    average_clustering = (
        float(finite_clustering.mean()) if finite_clustering.size else float("nan")
    )
    tree = mst_from_distance(distance)
    n_edges = int(np.count_nonzero(np.triu(adjacency, 1) > 0.0))
    density = (
        2.0 * n_edges / (n * (n - 1)) if n > 1 else 0.0
    )
    return CorrelationNetwork(
        correlation=matrix.copy(),
        adjacency=adjacency,
        distance=distance,
        components=components,
        strength=strength_values,
        eigenvector=eigenvector_values,
        clustering=clustering_values,
        average_clustering=average_clustering,
        mst=tree,
        density=float(density),
        n_edges=n_edges,
    )


def correlation_network(
    returns_panel: Any, threshold: float = 0.5
) -> CorrelationNetwork:
    """Correlate a ``nobs x nassets`` return panel and build its graph."""
    panel = as_f64_2d(returns_panel, "returns_panel")
    if panel.shape[0] < 2 or panel.shape[1] < 1:
        raise ValueError("returns_panel must have at least 2 rows and 1 asset")
    return network_from_correlation(
        qflux.correlation_matrix(panel), threshold=threshold
    )


__all__ = [
    "CorrelationNetwork",
    "DiffusionResult",
    "GraphComponents",
    "MSTResult",
    "PercolationScan",
    "bond_percolation",
    "centrality",
    "clustering",
    "correlation_adjacency",
    "correlation_network",
    "correlation_to_distance",
    "mst_from_correlation",
    "mst_from_distance",
    "network_from_correlation",
    "network_strength",
    "risk_diffusion",
    "site_percolation",
    "threshold_components",
]
