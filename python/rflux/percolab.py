"""percolab: correlation graphs, centrality, percolation and risk diffusion.

Every graph and percolation kernel lives in the C++ core
(``src/percolab/``); this module validates shapes and wraps the native results
in small dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import qflux
from ._common import (
    as_f64_1d,
    as_f64_2d,
    as_i32_1d,
    as_size,
    as_square,
    as_u32_1d,
    as_u64,
)
from ._native_backend import native_api as native_module, native_call


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

    def _summary(self) -> tuple[float, float, float, float]:
        return native_call(
            native_module, "percolation_summary",
            lambda: native_module.percolation_summary(
                self.probabilities, self.giant_mean, self.susceptibility
            ),
        )

    def critical_probability(self) -> float:
        """Scanned probability with the largest finite-cluster susceptibility."""
        return float(self._summary()[0])

    def p_half_giant(self) -> float:
        """First scanned probability whose giant fraction reaches half its
        final value."""
        return float(self._summary()[1])

    def max_susceptibility(self) -> float:
        """Largest finite-cluster susceptibility in the scan."""
        return float(self._summary()[2])

    def giant_area(self) -> float:
        """Trapezoidal integral of ``giant_mean`` over probabilities."""
        return float(self._summary()[3])


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
        return int(native_call(
            native_module, "max_i32",
            lambda: native_module.max_i32(self.core_numbers),
        ))


def _components(
    labels: np.ndarray, n_components: int, largest_size: int, sizes: np.ndarray
) -> GraphComponents:
    return GraphComponents(
        labels=np.asarray(labels, dtype=np.int32),
        n_components=int(n_components),
        largest_size=int(largest_size),
        sizes=np.asarray(sizes, dtype=np.int64),
    )


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    return native_call(
        native_module, "correlation_to_distance",
        lambda: native_module.correlation_to_distance(matrix),
    )


def correlation_to_distance(correlation: Any) -> np.ndarray:
    """Mantegna distance ``sqrt(2 * (1 - rho))``."""
    return _distance_matrix(as_square(correlation, "correlation"))


def _components_matrix(matrix: np.ndarray, threshold: float) -> GraphComponents:
    labels, n_components, largest, sizes = native_call(
        native_module, "connected_components",
        lambda: native_module.connected_components(matrix, float(threshold)),
    )
    return _components(labels, n_components, largest, sizes)


def threshold_components(adjacency: Any, threshold: float) -> GraphComponents:
    """Connected components of edges with weight >= threshold."""
    return _components_matrix(as_square(adjacency, "adjacency"), threshold)


def _strength_matrix(matrix: np.ndarray) -> np.ndarray:
    return native_call(
        native_module, "network_strength",
        lambda: native_module.network_strength(matrix),
    )


def network_strength(adjacency: Any) -> np.ndarray:
    """Weighted degree (sum of off-diagonal non-negative weights)."""
    return _strength_matrix(as_square(adjacency, "adjacency"))


def _centrality_matrix(matrix: np.ndarray, kind_value: int) -> np.ndarray:
    return native_call(
        native_module, "centrality",
        lambda: native_module.centrality(matrix, kind_value),
    )


def centrality(adjacency: Any, kind: str = "eigenvector") -> np.ndarray:
    """``degree`` or L2-normalised ``eigenvector`` centrality."""
    kinds = {"degree": 0, "eigenvector": 1}
    try:
        kind_value = kinds[str(kind).strip().lower()]
    except KeyError:
        raise ValueError("kind must be 'degree' or 'eigenvector'") from None
    return _centrality_matrix(as_square(adjacency, "adjacency"), kind_value)


def _clustering_matrix(matrix: np.ndarray) -> np.ndarray:
    return native_call(
        native_module, "clustering",
        lambda: native_module.clustering(matrix),
    )


def clustering(adjacency: Any) -> np.ndarray:
    """Local unweighted clustering coefficient; NaN when degree < 2."""
    return _clustering_matrix(as_square(adjacency, "adjacency"))


def _transitivity_matrix(matrix: np.ndarray) -> float:
    return float(native_call(
        native_module, "transitivity",
        lambda: native_module.transitivity(matrix),
    ))


def transitivity(adjacency: Any) -> float:
    """Global clustering coefficient of an undirected unweighted graph."""
    return _transitivity_matrix(as_square(adjacency, "adjacency"))


def _path_statistics_matrix(matrix: np.ndarray) -> PathStats:
    efficiency, average, diameter, reachable, connected = native_call(
        native_module, "path_statistics",
        lambda: native_module.path_statistics(matrix),
    )
    return PathStats(
        efficiency=float(efficiency),
        average_path_length=float(average),
        diameter=int(diameter),
        n_reachable_pairs=int(reachable),
        connected=bool(connected),
    )


def path_statistics(adjacency: Any) -> PathStats:
    """Shortest-path summaries of an undirected unweighted graph."""
    return _path_statistics_matrix(as_square(adjacency, "adjacency"))


def _core_numbers_matrix(matrix: np.ndarray) -> np.ndarray:
    return native_call(
        native_module, "core_numbers",
        lambda: native_module.core_numbers(matrix),
    )


def core_numbers(adjacency: Any) -> np.ndarray:
    """k-core number of every node; isolated nodes are zero."""
    return _core_numbers_matrix(as_square(adjacency, "adjacency"))


def _assortativity_matrix(matrix: np.ndarray) -> float:
    return float(native_call(
        native_module, "assortativity",
        lambda: native_module.assortativity(matrix),
    ))


def assortativity(adjacency: Any) -> float:
    """Newman degree assortativity for an undirected unweighted graph."""
    return _assortativity_matrix(as_square(adjacency, "adjacency"))


def modularity(adjacency: Any, labels: Any) -> float:
    """Newman modularity of a partition of a weighted undirected graph.

    Labels may be arbitrary non-negative integers; the C++ binding remaps them
    to contiguous community indices.
    """
    matrix = as_square(adjacency, "adjacency")
    label_array = as_i32_1d(labels, "labels")
    if label_array.size != matrix.shape[0]:
        raise ValueError("labels must contain one entry per node")
    if np.any(label_array < 0):
        raise ValueError("labels must be non-negative")
    return float(native_call(
        native_module, "modularity",
        lambda: native_module.modularity(matrix, label_array),
    ))


def _betweenness_matrix(matrix: np.ndarray, normalized: bool) -> np.ndarray:
    return native_call(
        native_module, "betweenness",
        lambda: native_module.betweenness(matrix, normalized),
    )


def betweenness(adjacency: Any, normalized: bool = True) -> np.ndarray:
    """Brandes betweenness for an undirected unweighted graph."""
    return _betweenness_matrix(as_square(adjacency, "adjacency"), normalized)


def bridges(adjacency: Any) -> np.ndarray:
    """Bridges as a ``(k, 2)`` uint32 array, sorted by ``(u, v)``."""
    return native_call(
        native_module, "bridges",
        lambda: native_module.bridges(as_square(adjacency, "adjacency")),
    )


def articulation_points(adjacency: Any) -> np.ndarray:
    """Boolean articulation-point mask with one entry per node."""
    values = native_call(
        native_module, "articulation_points",
        lambda: native_module.articulation_points(
            as_square(adjacency, "adjacency")
        ),
    )
    return np.asarray(values, dtype=bool)


def truss_numbers(adjacency: Any) -> np.ndarray:
    """Symmetric ``n x n`` int32 edge-trussness matrix."""
    return native_call(
        native_module, "truss_numbers",
        lambda: native_module.truss_numbers(as_square(adjacency, "adjacency")),
    )


def pagerank(
    adjacency: Any,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Directed weighted PageRank; dangling rows use uniform teleportation."""
    matrix = as_square(adjacency, "adjacency")
    return native_call(
        native_module, "pagerank",
        lambda: native_module.pagerank(
            matrix, float(damping), as_size(max_iter, "max_iter"), float(tol)
        ),
    )


def _mst_matrix(matrix: np.ndarray) -> MSTResult:
    u, v, weight = native_call(
        native_module, "minimum_spanning_tree",
        lambda: native_module.minimum_spanning_tree(matrix),
    )
    return MSTResult(
        np.asarray(u, dtype=np.uint32),
        np.asarray(v, dtype=np.uint32),
        np.asarray(weight, dtype=np.float64),
    )


def mst_from_distance(distance: Any) -> MSTResult:
    """Minimum spanning tree of a complete symmetric distance matrix."""
    return _mst_matrix(as_square(distance, "distance"))


def mst_from_correlation(correlation: Any) -> MSTResult:
    """Correlation-distance minimum spanning tree."""
    matrix = as_square(correlation, "correlation")
    return _mst_matrix(_distance_matrix(matrix))


def _mst_clusters_matrix(matrix: np.ndarray, threshold: float) -> GraphComponents:
    labels, n_components, largest, sizes = native_call(
        native_module, "mst_clusters",
        lambda: native_module.mst_clusters(matrix, float(threshold)),
    )
    return _components(labels, n_components, largest, sizes)


def mst_clusters(distance: Any, threshold: float) -> GraphComponents:
    """Single-linkage clusters obtained by cutting the MST above ``threshold``."""
    return _mst_clusters_matrix(as_square(distance, "distance"), threshold)


def _occupation_inputs(
    adjacency: Any,
    kind: str,
    edge_weights: Any,
    node_weights: Any,
) -> tuple[np.ndarray, bool, np.ndarray | None, np.ndarray | None]:
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
    return matrix, normalized == "site", edge_array, node_array


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
    """General bond/site percolation scan."""
    matrix, site, edge_array, node_array = _occupation_inputs(
        adjacency, kind, edge_weights, node_weights
    )
    probability_array = as_f64_1d(probabilities, "probabilities")
    if probability_array.size == 0:
        raise ValueError("probabilities must not be empty")
    mean, std, susceptibility = native_call(
        native_module, "percolation_scan",
        lambda: native_module.percolation_scan(
            matrix, probability_array, site, edge_array, node_array,
            as_size(repetitions, "repetitions"), as_u64(seed, "seed"),
        ),
    )
    return PercolationScan(
        probabilities=probability_array.copy(),
        giant_mean=np.asarray(mean, dtype=np.float64),
        giant_std=np.asarray(std, dtype=np.float64),
        susceptibility=np.asarray(susceptibility, dtype=np.float64),
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
    """Mean number of clusters of each size at one occupation probability."""
    matrix, site, edge_array, node_array = _occupation_inputs(
        adjacency, kind, edge_weights, node_weights
    )
    value = float(probability)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return native_call(
        native_module, "cluster_size_distribution",
        lambda: native_module.cluster_size_distribution(
            matrix, value, site, edge_array, node_array,
            as_size(repetitions, "repetitions"), as_u64(seed, "seed"),
        ),
    )


_STRATEGIES = {
    "strength": 0,
    "degree": 1,
    "eigenvector": 2,
    "betweenness": 3,
    "core": 4,
}


def _removal_order_matrix(matrix: np.ndarray, strategy: str) -> np.ndarray:
    try:
        value = _STRATEGIES[str(strategy).strip().lower()]
    except KeyError:
        raise ValueError(
            "strategy must be 'strength', 'degree', 'eigenvector', "
            "'betweenness' or 'core'"
        ) from None
    return native_call(
        native_module, "removal_order",
        lambda: native_module.removal_order(matrix, value),
    )


def removal_order(adjacency: Any, strategy: str = "strength") -> np.ndarray:
    """Return a deterministic node-removal permutation."""
    return _removal_order_matrix(as_square(adjacency, "adjacency"), strategy)


def targeted_removal(
    adjacency: Any,
    order: Any = None,
    *,
    checkpoints: int = 10,
    strategy: str = "strength",
) -> RemovalCurve:
    """Deterministic node-removal curve (attack tolerance)."""
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
    removed, giant, susceptibility, n_components = native_call(
        native_module, "targeted_removal",
        lambda: native_module.targeted_removal(
            matrix, order_array, checkpoints
        ),
    )
    return RemovalCurve(
        removed=np.asarray(removed, dtype=np.int64),
        giant_fraction=np.asarray(giant, dtype=np.float64),
        susceptibility=np.asarray(susceptibility, dtype=np.float64),
        n_components=np.asarray(n_components, dtype=np.int64),
    )


def core_percolation(adjacency: Any) -> CoreScan:
    """Deterministic k-core (core-percolation) curve from core numbers."""
    ks, fraction, n_nodes = native_call(
        native_module, "core_percolation",
        lambda: native_module.core_percolation(
            as_square(adjacency, "adjacency")
        ),
    )
    return CoreScan(
        k=np.asarray(ks, dtype=np.int64),
        core_fraction=np.asarray(fraction, dtype=np.float64),
        n_nodes=np.asarray(n_nodes, dtype=np.int64),
    )


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
    """Stochastic SIR risk diffusion on a weighted graph."""
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
        raw = np.asarray(seeds)
        seed_array = (
            np.empty(0, dtype=np.uint32)
            if raw.size == 0
            else as_u32_1d(raw, "seeds")
        )
    if seed_array.size and int(seed_array.max()) >= n:
        raise ValueError("seed node index is out of range")
    mean_affected, infection = native_call(
        native_module, "risk_diffusion",
        lambda: native_module.risk_diffusion(
            matrix, seed_array, float(beta), float(gamma), steps, repetitions,
            seed,
        ),
    )
    return DiffusionResult(
        mean_affected=np.asarray(mean_affected, dtype=np.float64),
        infection_frequency=np.asarray(infection, dtype=np.float64),
    )


def _adjacency_result(
    matrix: np.ndarray, threshold: float, *, weighted: bool = True
) -> tuple[np.ndarray, int, float]:
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "correlation matrix contains NaN/inf; drop constant or "
            "insufficient series first"
        )
    adjacency, n_edges, density = native_call(
        native_module, "correlation_adjacency",
        lambda: native_module.correlation_adjacency(
            matrix, float(threshold), weighted
        ),
    )
    return np.asarray(adjacency, dtype=np.float64), int(n_edges), float(density)


def correlation_adjacency(
    correlation: Any, threshold: float = 0.5, *, weighted: bool = True
) -> np.ndarray:
    """Threshold a correlation matrix into a non-negative adjacency matrix."""
    matrix = as_square(correlation, "correlation")
    adjacency, _, _ = _adjacency_result(
        matrix, threshold, weighted=weighted
    )
    return adjacency


def network_from_correlation(
    correlation: Any, threshold: float = 0.5
) -> CorrelationNetwork:
    """Build a full correlation network summary from a correlation matrix."""
    matrix = as_square(correlation, "correlation")
    n = matrix.shape[0]
    adjacency, n_edges, density = _adjacency_result(matrix, threshold)
    distance = _distance_matrix(matrix)
    components = _components_matrix(adjacency, float(threshold))
    strength_values = _strength_matrix(adjacency)
    eigenvector_values = _centrality_matrix(adjacency, 1)
    clustering_values = _clustering_matrix(adjacency)
    average_clustering = float(native_call(
        native_module, "finite_stats",
        lambda: native_module.finite_stats(clustering_values),
    )[2])
    tree = _mst_matrix(distance)
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
        density=density,
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
