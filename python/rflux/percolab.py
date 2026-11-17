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
    as_i32_1d,
    as_size,
    as_square,
    as_u32_1d,
    as_u64,
    call_i32,
    call_scalar,
    call_vector,
    double_ptr,
    empty_f64,
    int32_ptr,
    uint32_ptr,
)
from ._ffi import check, get_lib
from . import qflux


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class MSTResult:
    u: np.ndarray
    v: np.ndarray
    weight: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return an ``(n-1, 3)`` array with columns u, v, weight."""
        return np.column_stack((self.u, self.v, self.weight))


@dataclass(frozen=True, slots=True)
class PathStats:
    """Shortest-path summaries of an undirected unweighted graph."""

    efficiency: float
    average_path_length: float
    diameter: int
    n_reachable_pairs: int
    connected: bool


@dataclass(frozen=True, slots=True)
class PercolationScan:
    probabilities: np.ndarray
    giant_mean: np.ndarray
    giant_std: np.ndarray
    susceptibility: np.ndarray

    def critical_probability(self) -> float:
        """Probability with the largest finite-cluster susceptibility."""
        if self.susceptibility.size == 0:
            return float("nan")
        return float(self.probabilities[int(np.argmax(self.susceptibility))])

    def max_susceptibility(self) -> float:
        """Largest finite-cluster susceptibility in the scan."""
        if self.susceptibility.size == 0:
            return float("nan")
        return float(np.max(self.susceptibility))

    def giant_area(self) -> float:
        """Trapezoidal integral of ``giant_mean`` over probabilities.

        The scan probabilities should be increasing; the result is the area
        under the giant-component curve and is a compact summary of how early
        the network percolates.
        """
        if self.probabilities.size < 2:
            return 0.0
        widths = np.diff(self.probabilities)
        heights = 0.5 * (self.giant_mean[1:] + self.giant_mean[:-1])
        return float(np.sum(widths * heights))


@dataclass(frozen=True, slots=True)
class RemovalCurve:
    removed: np.ndarray
    giant_fraction: np.ndarray
    susceptibility: np.ndarray
    n_components: np.ndarray


@dataclass(frozen=True, slots=True)
class CoreScan:
    k: np.ndarray
    core_fraction: np.ndarray
    n_nodes: np.ndarray


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    mean_affected: np.ndarray
    infection_frequency: np.ndarray

    def final_affected(self) -> float:
        return float(self.mean_affected[-1])


@dataclass(frozen=True, slots=True)
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
    path_stats: PathStats
    core_numbers: np.ndarray
    transitivity: float
    assortativity: float

    @property
    def max_core(self) -> int:
        """Largest k for which the correlation graph has a non-empty k-core."""
        return int(self.core_numbers.max()) if self.core_numbers.size else 0


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


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = call_vector(
        "rflux_corr_to_distance", n * n, get_lib().rflux_corr_to_distance,
        double_ptr(matrix), n)
    return out.reshape(n, n)


def correlation_to_distance(correlation: Any) -> np.ndarray:
    """Mantegna distance ``sqrt(2 * (1 - rho))``."""
    return _distance_matrix(as_square(correlation, "correlation"))


def _components_matrix(matrix: np.ndarray, threshold: float) -> GraphComponents:
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


def threshold_components(adjacency: Any, threshold: float) -> GraphComponents:
    """Connected components of edges with weight >= threshold."""
    return _components_matrix(as_square(adjacency, "adjacency"), threshold)


def _strength_matrix(matrix: np.ndarray) -> np.ndarray:
    return call_vector(
        "rflux_network_strength", matrix.shape[0],
        get_lib().rflux_network_strength, double_ptr(matrix), matrix.shape[0])


def network_strength(adjacency: Any) -> np.ndarray:
    """Weighted degree (sum of off-diagonal non-negative weights)."""
    return _strength_matrix(as_square(adjacency, "adjacency"))


def _centrality_matrix(matrix: np.ndarray, kind_value: int) -> np.ndarray:
    return call_vector(
        "rflux_network_centrality", matrix.shape[0],
        get_lib().rflux_network_centrality,
        double_ptr(matrix), matrix.shape[0], kind_value)


def centrality(adjacency: Any, kind: str = "eigenvector") -> np.ndarray:
    """``degree`` or L2-normalised ``eigenvector`` centrality."""
    kinds = {"degree": 0, "eigenvector": 1}
    try:
        value = kinds[str(kind).lower()]
    except KeyError as exc:
        raise ValueError("kind must be 'degree' or 'eigenvector'") from exc
    return _centrality_matrix(as_square(adjacency, "adjacency"), value)


def _clustering_matrix(matrix: np.ndarray) -> np.ndarray:
    return call_vector(
        "rflux_network_clustering", matrix.shape[0],
        get_lib().rflux_network_clustering, double_ptr(matrix), matrix.shape[0])


def clustering(adjacency: Any) -> np.ndarray:
    """Local unweighted clustering coefficient; NaN when degree < 2."""
    return _clustering_matrix(as_square(adjacency, "adjacency"))


def _transitivity_matrix(matrix: np.ndarray) -> float:
    return call_scalar(
        "rflux_network_transitivity", get_lib().rflux_network_transitivity,
        double_ptr(matrix), matrix.shape[0])


def transitivity(adjacency: Any) -> float:
    """Global clustering coefficient ``3 * triangles / connected triples``."""
    return _transitivity_matrix(as_square(adjacency, "adjacency"))


def _path_statistics_matrix(matrix: np.ndarray) -> PathStats:
    efficiency = ctypes.c_double(0.0)
    average = ctypes.c_double(0.0)
    diameter = ctypes.c_size_t(0)
    reachable = ctypes.c_size_t(0)
    connected = ctypes.c_int32(0)
    check(
        get_lib().rflux_network_path_stats(
            double_ptr(matrix),
            matrix.shape[0],
            ctypes.byref(efficiency),
            ctypes.byref(average),
            ctypes.byref(diameter),
            ctypes.byref(reachable),
            ctypes.byref(connected),
        ),
        "rflux_network_path_stats",
    )
    return PathStats(
        efficiency=float(efficiency.value),
        average_path_length=float(average.value),
        diameter=int(diameter.value),
        n_reachable_pairs=int(reachable.value),
        connected=bool(connected.value),
    )


def path_statistics(adjacency: Any) -> PathStats:
    """Shortest-path efficiency, average path length and diameter."""
    return _path_statistics_matrix(as_square(adjacency, "adjacency"))


def _core_numbers_matrix(matrix: np.ndarray) -> np.ndarray:
    return call_i32(
        "rflux_network_core_numbers", matrix.shape[0],
        get_lib().rflux_network_core_numbers,
        double_ptr(matrix), matrix.shape[0],
    )


def core_numbers(adjacency: Any) -> np.ndarray:
    """k-core number of every node (isolated nodes receive zero)."""
    return _core_numbers_matrix(as_square(adjacency, "adjacency"))


def _assortativity_matrix(matrix: np.ndarray) -> float:
    return call_scalar(
        "rflux_network_assortativity", get_lib().rflux_network_assortativity,
        double_ptr(matrix), matrix.shape[0])


def assortativity(adjacency: Any) -> float:
    """Newman degree assortativity; NaN for empty or regular graphs."""
    return _assortativity_matrix(as_square(adjacency, "adjacency"))


def modularity(adjacency: Any, labels: Any) -> float:
    """Newman modularity of a partition of a weighted undirected graph.

    Labels may be arbitrary non-negative integers (for example sector codes);
    they are remapped to contiguous community indices before the native call.
    """
    matrix = as_square(adjacency, "adjacency")
    label_array = as_i32_1d(labels, "labels")
    n = matrix.shape[0]
    if label_array.size != n:
        raise ValueError("labels must contain one entry per node")
    if np.any(label_array < 0):
        raise ValueError("labels must be non-negative")
    communities, inverse = np.unique(label_array, return_inverse=True)
    contiguous = np.ascontiguousarray(inverse, dtype=np.int32)
    return call_scalar(
        "rflux_network_modularity", get_lib().rflux_network_modularity,
        double_ptr(matrix), n, int32_ptr(contiguous), communities.size)


def _betweenness_matrix(matrix: np.ndarray, normalized: bool) -> np.ndarray:
    return call_vector(
        "rflux_network_betweenness", matrix.shape[0],
        get_lib().rflux_network_betweenness,
        double_ptr(matrix), matrix.shape[0], 1 if normalized else 0)


def betweenness(adjacency: Any, normalized: bool = True) -> np.ndarray:
    """Brandes betweenness for an undirected unweighted graph."""
    return _betweenness_matrix(as_square(adjacency, "adjacency"), normalized)


def bridges(adjacency: Any) -> np.ndarray:
    """Bridges as a ``(k, 2)`` uint32 array, sorted by ``(u, v)``."""
    matrix = as_square(adjacency, "adjacency")
    n = matrix.shape[0]
    if n <= 1:
        return np.empty((0, 2), dtype=np.uint32)
    left = np.empty(n - 1, dtype=np.uint32)
    right = np.empty(n - 1, dtype=np.uint32)
    n_edges = ctypes.c_size_t(0)
    check(
        get_lib().rflux_network_bridges(
            double_ptr(matrix),
            n,
            uint32_ptr(left),
            uint32_ptr(right),
            ctypes.byref(n_edges),
        ),
        "rflux_network_bridges",
    )
    count = int(n_edges.value)
    return np.column_stack((left[:count], right[:count])).astype(
        np.uint32, copy=False
    )


def articulation_points(adjacency: Any) -> np.ndarray:
    """Boolean articulation-point mask with one entry per node."""
    matrix = as_square(adjacency, "adjacency")
    values = call_i32(
        "rflux_network_articulation_points", matrix.shape[0],
        get_lib().rflux_network_articulation_points,
        double_ptr(matrix), matrix.shape[0],
    )
    return values.astype(bool)


def truss_numbers(adjacency: Any) -> np.ndarray:
    """Symmetric ``n x n`` int32 edge-trussness matrix.

    Edges in no triangle have trussness 2; non-edges and the diagonal are 0.
    """
    matrix = as_square(adjacency, "adjacency")
    values = call_i32(
        "rflux_network_truss_numbers", matrix.shape[0] * matrix.shape[0],
        get_lib().rflux_network_truss_numbers,
        double_ptr(matrix), matrix.shape[0],
    )
    return values.reshape(matrix.shape[0], matrix.shape[0])


def pagerank(
    adjacency: Any,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Directed weighted PageRank; dangling rows use uniform teleportation."""
    matrix = as_square(adjacency, "adjacency")
    return call_vector(
        "rflux_pagerank", matrix.shape[0], get_lib().rflux_pagerank,
        double_ptr(matrix), matrix.shape[0], float(damping),
        as_size(max_iter, "max_iter"), float(tol))


def _mst_matrix(matrix: np.ndarray) -> MSTResult:
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


def mst_from_distance(distance: Any) -> MSTResult:
    """Minimum spanning tree of a complete symmetric distance matrix."""
    return _mst_matrix(as_square(distance, "distance"))


def mst_from_correlation(correlation: Any) -> MSTResult:
    """Correlation-distance minimum spanning tree."""
    matrix = as_square(correlation, "correlation")
    return _mst_matrix(_distance_matrix(matrix))


def _mst_clusters_matrix(
    matrix: np.ndarray, threshold: float
) -> GraphComponents:
    n = matrix.shape[0]
    labels = np.empty(n, dtype=np.int32)
    n_components = ctypes.c_size_t(0)
    largest = ctypes.c_size_t(0)
    check(
        get_lib().rflux_mst_clusters(
            double_ptr(matrix),
            n,
            float(threshold),
            int32_ptr(labels),
            ctypes.byref(n_components),
            ctypes.byref(largest),
        ),
        "rflux_mst_clusters",
    )
    return _components_from_summary(labels, n_components.value, largest.value)


def mst_clusters(distance: Any, threshold: float) -> GraphComponents:
    """Single-linkage clusters obtained by cutting the MST above ``threshold``.

    Nodes stay connected when every edge on their MST path has distance
    ``<= threshold``; this equals connected components of all edges with
    distance ``<= threshold``.
    """
    return _mst_clusters_matrix(as_square(distance, "distance"), threshold)


def _occupation_inputs(
    adjacency: Any,
    kind: str,
    edge_weights: Any,
    node_weights: Any,
) -> tuple[np.ndarray, str, np.ndarray | None, np.ndarray | None]:
    matrix = as_square(adjacency, "adjacency")
    normalized = str(kind).strip().lower()
    if normalized not in ("bond", "site"):
        raise ValueError("kind must be 'bond' or 'site'")
    if normalized == "bond" and node_weights is not None:
        raise ValueError("node_weights are only used by site percolation")
    if normalized == "site" and edge_weights is not None:
        raise ValueError("edge_weights are only used by bond percolation")
    edge_array = None
    if edge_weights is not None:
        edge_array = as_square(edge_weights, "edge_weights")
        if edge_array.shape[0] != matrix.shape[0]:
            raise ValueError(
                "edge_weights must have the same shape as adjacency"
            )
    node_array = None
    if node_weights is not None:
        node_array = as_f64_1d(node_weights, "node_weights")
        if node_array.size != matrix.shape[0]:
            raise ValueError("node_weights must have one entry per node")
    return matrix, normalized, edge_array, node_array


def percolation_scan(
    adjacency: Any,
    probabilities: Any,
    *,
    kind: str = "bond",
    edge_weights: Any = None,
    node_weights: Any = None,
    repetitions: int = 100,
    seed: int = 42,
) -> PercolationScan:
    """General bond/site percolation scan.

    Candidate edges are ``adjacency > 0``.  Bond occupancy uses
    ``min(1, p * edge_weight)``; site occupancy uses
    ``min(1, p * node_weight)`` and keeps an edge only when both endpoints
    survive.  Edge weights must be finite, symmetric and non-negative; node
    weights must be finite and non-negative.  Passing weights that do not
    belong to ``kind`` is an error.
    """
    matrix, normalized, edge_array, node_array = _occupation_inputs(
        adjacency, kind, edge_weights, node_weights
    )
    probability_array = as_f64_1d(probabilities, "probabilities")
    n_probabilities = probability_array.size
    if n_probabilities == 0:
        raise ValueError("probabilities must not be empty")
    repetitions = as_size(repetitions, "repetitions")
    seed = as_u64(seed, "seed")
    mean = empty_f64(n_probabilities)
    std = empty_f64(n_probabilities)
    susceptibility = empty_f64(n_probabilities)
    check(
        get_lib().rflux_percolation_scan(
            double_ptr(matrix),
            matrix.shape[0],
            double_ptr(edge_array) if edge_array is not None else None,
            double_ptr(node_array) if node_array is not None else None,
            1 if normalized == "site" else 0,
            double_ptr(probability_array),
            n_probabilities,
            repetitions,
            seed,
            double_ptr(mean),
            double_ptr(std),
            double_ptr(susceptibility),
        ),
        "rflux_percolation_scan",
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
    return percolation_scan(
        adjacency, probabilities, kind="bond", repetitions=repetitions,
        seed=seed,
    )


def site_percolation(
    adjacency: Any,
    probabilities: Any,
    repetitions: int = 100,
    seed: int = 42,
) -> PercolationScan:
    """Site percolation; giant fraction is relative to the original n."""
    return percolation_scan(
        adjacency, probabilities, kind="site", repetitions=repetitions,
        seed=seed,
    )


def bond_percolation_weighted(
    adjacency: Any,
    probabilities: Any,
    repetitions: int = 100,
    seed: int = 42,
) -> PercolationScan:
    """Bond percolation with edge retention probability ``p * weight``."""
    return percolation_scan(
        adjacency, probabilities, kind="bond", edge_weights=adjacency,
        repetitions=repetitions, seed=seed,
    )


def site_percolation_weighted(
    adjacency: Any,
    node_weights: Any,
    probabilities: Any,
    repetitions: int = 100,
    seed: int = 42,
) -> PercolationScan:
    """Site percolation with survival probability ``p * node_weight``."""
    return percolation_scan(
        adjacency, probabilities, kind="site", node_weights=node_weights,
        repetitions=repetitions, seed=seed,
    )


def cluster_size_distribution(
    adjacency: Any,
    probability: float,
    *,
    kind: str = "bond",
    edge_weights: Any = None,
    node_weights: Any = None,
    repetitions: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Mean number of clusters of each size at one occupation probability.

    The result has ``n + 1`` entries; index ``s`` counts components of size
    ``s`` (including the largest) and index 0 is always zero.
    """
    matrix, normalized, edge_array, node_array = _occupation_inputs(
        adjacency, kind, edge_weights, node_weights
    )
    value = float(probability)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    repetitions = as_size(repetitions, "repetitions")
    seed = as_u64(seed, "seed")
    out = empty_f64(matrix.shape[0] + 1)
    check(
        get_lib().rflux_cluster_size_distribution(
            double_ptr(matrix),
            matrix.shape[0],
            double_ptr(edge_array) if edge_array is not None else None,
            double_ptr(node_array) if node_array is not None else None,
            1 if normalized == "site" else 0,
            value,
            repetitions,
            seed,
            double_ptr(out),
        ),
        "rflux_cluster_size_distribution",
    )
    return out


def _removal_order_matrix(matrix: np.ndarray, strategy: str) -> np.ndarray:
    normalized = str(strategy).strip().lower()
    if normalized == "strength":
        scores = _strength_matrix(matrix)
    elif normalized == "degree":
        mask = matrix > 0.0
        np.fill_diagonal(mask, False)
        scores = mask.sum(axis=1).astype(np.float64)
    elif normalized == "eigenvector":
        scores = _centrality_matrix(matrix, 1)
    elif normalized == "betweenness":
        scores = _betweenness_matrix(matrix, False)
    elif normalized == "core":
        scores = -_core_numbers_matrix(matrix).astype(np.float64)
    else:
        raise ValueError(
            "strategy must be 'strength', 'degree', 'eigenvector', "
            "'betweenness' or 'core'"
        )
    order = np.argsort(-scores, kind="stable")
    return np.ascontiguousarray(order, dtype=np.uint32)


def removal_order(adjacency: Any, strategy: str = "strength") -> np.ndarray:
    """Return a deterministic node-removal permutation.

    ``strength`` removes the largest weighted degree first, ``degree`` the
    largest unweighted degree, ``eigenvector`` the largest eigenvector
    centrality, ``betweenness`` the largest Brandes betweenness and ``core``
    the lowest core number.  Ties keep ascending node order.
    """
    return _removal_order_matrix(as_square(adjacency, "adjacency"), strategy)


def targeted_removal(
    adjacency: Any,
    order: Any = None,
    *,
    checkpoints: int = 10,
    strategy: str = "strength",
) -> RemovalCurve:
    """Deterministic node-removal curve (attack tolerance).

    Nodes are removed in ``order`` (a permutation of ``0..n-1``) or in the
    order produced by :func:`removal_order`.  Points are reported at
    ``floor(k * n / checkpoints)`` removals for ``k = 0..checkpoints``; when
    ``checkpoints`` exceeds ``n`` some points repeat.
    """
    matrix = as_square(adjacency, "adjacency")
    n = matrix.shape[0]
    checkpoints = as_size(checkpoints, "checkpoints")
    if order is None:
        order_array = _removal_order_matrix(matrix, strategy)
    else:
        order_array = as_u32_1d(order, "order")
        if order_array.size != n:
            raise ValueError("order must contain one entry per node")
        if not np.array_equal(np.sort(order_array), np.arange(n)):
            raise ValueError("order must be a permutation of 0..n-1")

    removed = np.empty(checkpoints + 1, dtype=np.uintp)
    giant = empty_f64(checkpoints + 1)
    susceptibility = empty_f64(checkpoints + 1)
    n_components = np.empty(checkpoints + 1, dtype=np.uintp)
    check(
        get_lib().rflux_targeted_removal(
            double_ptr(matrix),
            n,
            uint32_ptr(order_array),
            checkpoints,
            removed.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            double_ptr(giant),
            double_ptr(susceptibility),
            n_components.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ),
        "rflux_targeted_removal",
    )
    return RemovalCurve(
        removed=removed.astype(np.int64),
        giant_fraction=giant,
        susceptibility=susceptibility,
        n_components=n_components.astype(np.int64),
    )


def core_percolation(adjacency: Any) -> CoreScan:
    """Deterministic k-core (core-percolation) curve from core numbers.

    For threshold ``k`` the surviving set is the k-core: the fixed point of
    repeatedly deleting nodes with fewer than ``k`` neighbours inside the
    remaining graph.  The curve requires no sampling and is not a random
    bootstrap percolation process.
    """
    matrix = as_square(adjacency, "adjacency")
    core = _core_numbers_matrix(matrix)
    n = core.size
    maximum = int(core.max()) if n else 0
    ks = np.arange(1, maximum + 1, dtype=np.int64)
    n_nodes = np.array(
        [int(np.count_nonzero(core >= k)) for k in ks], dtype=np.int64
    )
    fraction = n_nodes.astype(np.float64) / float(n)
    return CoreScan(k=ks, core_fraction=fraction, n_nodes=n_nodes)


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
    if steps == int(np.iinfo(np.uintp).max):
        raise ValueError("steps is too large")
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


def _adjacency_matrix(
    matrix: np.ndarray, threshold: float, *, weighted: bool = True
) -> np.ndarray:
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


def correlation_adjacency(
    correlation: Any, threshold: float = 0.5, *, weighted: bool = True
) -> np.ndarray:
    """Threshold a correlation matrix into a non-negative adjacency matrix."""
    return _adjacency_matrix(
        as_square(correlation, "correlation"), threshold, weighted=weighted
    )


def network_from_correlation(
    correlation: Any, threshold: float = 0.5
) -> CorrelationNetwork:
    """Build a full correlation network summary from a correlation matrix."""
    matrix = as_square(correlation, "correlation")
    n = matrix.shape[0]
    adjacency = _adjacency_matrix(matrix, threshold)
    distance = _distance_matrix(matrix)
    components = _components_matrix(adjacency, float(threshold))
    strength_values = _strength_matrix(adjacency)
    eigenvector_values = _centrality_matrix(adjacency, 1)
    clustering_values = _clustering_matrix(adjacency)
    finite_clustering = clustering_values[np.isfinite(clustering_values)]
    average_clustering = (
        float(finite_clustering.mean()) if finite_clustering.size else float("nan")
    )
    tree = _mst_matrix(distance)
    n_edges = int(np.count_nonzero(np.triu(adjacency, 1) > 0.0))
    density = (
        2.0 * n_edges / (n * (n - 1)) if n > 1 else 0.0
    )
    path = _path_statistics_matrix(adjacency)
    cores = _core_numbers_matrix(adjacency)
    global_clustering = _transitivity_matrix(adjacency)
    degree_assortativity = _assortativity_matrix(adjacency)
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
        path_stats=path,
        core_numbers=cores,
        transitivity=global_clustering,
        assortativity=degree_assortativity,
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
    "CoreScan",
    "CorrelationNetwork",
    "DiffusionResult",
    "GraphComponents",
    "MSTResult",
    "PathStats",
    "PercolationScan",
    "RemovalCurve",
    "assortativity",
    "articulation_points",
    "betweenness",
    "bond_percolation",
    "bond_percolation_weighted",
    "bridges",
    "centrality",
    "cluster_size_distribution",
    "clustering",
    "core_numbers",
    "core_percolation",
    "correlation_adjacency",
    "correlation_network",
    "correlation_to_distance",
    "modularity",
    "mst_clusters",
    "mst_from_correlation",
    "mst_from_distance",
    "network_from_correlation",
    "network_strength",
    "pagerank",
    "path_statistics",
    "percolation_scan",
    "removal_order",
    "risk_diffusion",
    "site_percolation",
    "site_percolation_weighted",
    "targeted_removal",
    "threshold_components",
    "transitivity",
    "truss_numbers",
]
