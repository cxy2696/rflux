#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rflux/percolab/graph_types.hpp"

namespace rflux::percolab {

struct MstEdge {
    std::uint32_t u = 0;
    std::uint32_t v = 0;
    double weight = 0.0;
};

struct BridgeEdge {
    std::uint32_t u = 0;
    std::uint32_t v = 0; /* u < v */
};

struct DiffusionResult {
    std::vector<double> mean_affected;        /* length steps+1 */
    std::vector<double> infection_frequency;  /* length n */
};

struct PathStats {
    /* Global efficiency: mean over ordered pairs of 1/distance. */
    double efficiency = 0.0;
    /* Mean distance over reachable unordered pairs; NaN when none exist. */
    double average_path_length = 0.0;
    /* Largest finite distance; zero when no pair is reachable. */
    std::size_t diameter = 0;
    std::size_t n_reachable_pairs = 0;
    bool connected = false;
};

/* sqrt(2 * (1 - corr)); useful as input to minimum_spanning_tree(). */
std::vector<double> correlation_to_distance(const double* corr, std::size_t n);

/* Strength = sum of off-diagonal non-negative adjacency weights. */
std::vector<double> strength(const double* weights, std::size_t n);

/* Power iteration on non-negative weights; L2-normalised result. */
std::vector<double> eigenvector_centrality(const double* weights, std::size_t n);

/* Local unweighted clustering coefficient; NaN when degree < 2. */
std::vector<double> clustering_coefficients(const double* weights,
                                            std::size_t n);

/*
 * Global clustering coefficient (transitivity): 3 * triangles / connected
 * triples.  NaN when the graph has no connected triple.
 */
double transitivity(const double* weights, std::size_t n);

/*
 * Shortest-path summaries for an undirected unweighted graph (edges > 0).
 * Average path length averages over reachable unordered pairs only, while
 * efficiency averages 1/d over all ordered pairs and therefore treats
 * unreachable pairs as zero.  `connected` reports whether every unordered
 * pair is reachable.  For n = 1 the graph is connected, efficiency is zero
 * and average_path_length is NaN (there is no pair).
 */
PathStats path_statistics(const double* weights, std::size_t n);

/*
 * k-core decomposition.  out[i] is the largest k such that node i belongs to
 * the k-core; isolated nodes receive zero.  The implementation is the
 * Batagelj-Zaversnik degree-bucket peeling algorithm.
 */
std::vector<std::int32_t> core_numbers(const double* weights, std::size_t n);

/*
 * Newman degree assortativity for an undirected unweighted graph.  NaN when
 * the graph has no edges or every degree is identical (zero variance).
 */
double degree_assortativity(const double* weights, std::size_t n);

/*
 * Newman modularity of an undirected weighted partition.  labels has length n
 * and every entry must be in [0, n_labels).  The diagonal is ignored.  NaN
 * when the graph has no edge weight.  The result lies in [-1, 1] up to
 * floating-point round-off.
 */
double modularity(const double* weights, std::size_t n,
                  const std::int32_t* labels, std::size_t n_labels);

/*
 * Single-linkage clusters of a distance matrix: cut the minimum spanning tree
 * at edges with distance > threshold and label the remaining components.
 * This is equivalent to connected components of the graph of edges with
 * distance <= threshold and is deterministic.
 */
ComponentSummary mst_clusters(const double* distance, std::size_t n,
                              double threshold);

/*
 * PageRank on a directed non-negative weighted graph.  Dangling rows
 * distribute their mass uniformly.  damping must be in [0, 1), max_iter and
 * tol must be positive; non-convergence is a NumericalError.
 */
std::vector<double> pagerank(const double* adjacency, std::size_t n,
                             double damping, std::size_t max_iter, double tol);

/* Brandes betweenness for an undirected unweighted graph (edges > 0). */
std::vector<double> betweenness_centrality(const double* adjacency,
                                           std::size_t n, bool normalized);

/*
 * Bridges (cut edges) of an undirected unweighted graph (edges > 0).
 * The result is sorted by (u, v) and uses u < v.  Deterministic Tarjan
 * low-link DFS: a tree edge (p, v) is a bridge iff low[v] > disc[p].
 */
std::vector<BridgeEdge> bridges(const double* adjacency, std::size_t n);

/*
 * Articulation points of an undirected unweighted graph (edges > 0), as a 0/1
 * label per node.  The DFS root is an articulation point when it has more than
 * one child; any other node p is one when some child v has low[v] >= disc[p].
 */
std::vector<std::int32_t> articulation_points(const double* adjacency,
                                              std::size_t n);

/*
 * Edge trussness.  For an edge e, t(e) is the largest k such that e belongs to
 * the k-truss: the maximal subgraph in which every edge has at least k - 2
 * common neighbours (triangles) inside the subgraph.  An edge with no
 * triangle has trussness 2.  The output is a symmetric n x n int32 matrix with
 * t(e) at both endpoints of each edge, and zero on the diagonal and non-edges.
 * The decomposition stages k = 3..max_support+2 remove every edge whose
 * current support is below k - 2, assigning trussness k - 1; each removed
 * triangle decrements the support of its other two edges.  A lazy min-heap
 * keeps the peeling deterministic; with T triangles the total time is
 * O(n^3 + T log m) and memory is O(n^2) on a dense input matrix.
 */
std::vector<std::int32_t> truss_numbers(const double* adjacency, std::size_t n);

/* Minimum spanning tree of a complete finite symmetric distance matrix. */
std::vector<MstEdge> minimum_spanning_tree(const double* distance,
                                           std::size_t n);

/*
 * Stochastic SIR process on a symmetric weighted graph.  Infection probability
 * per edge per step is 1 - exp(-beta*w).  mean_affected includes step zero.
 */
DiffusionResult risk_diffusion(const double* adjacency, std::size_t n,
                               const std::uint32_t* seed_nodes,
                               std::size_t n_seed_nodes, double beta,
                               double gamma, std::size_t steps,
                               std::size_t repetitions, std::uint64_t seed);

}  // namespace rflux::percolab
