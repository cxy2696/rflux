#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rflux/percolab/graph_types.hpp"

namespace rflux::percolab {

struct PercolationPoint {
    double probability = 0.0;
    double giant_mean = 0.0;
    double giant_std = 0.0;
    double susceptibility = 0.0;
};

enum class PercolationKind {
    bond, /* edges are occupied */
    site  /* nodes are occupied */
};

/*
 * Optional occupation multipliers.  Edges with adjacency > 0 are candidates;
 * an edge (i,j) is retained with probability min(1, p * edge[i*n+j]) and a
 * node i survives with probability min(1, p * node[i]).  A null pointer means
 * a multiplier of one.  A non-null edge matrix must be finite, symmetric and
 * non-negative; node weights must be finite and non-negative.  Unused weights
 * for the selected kind are rejected so a misspelled argument cannot silently
 * change the model.
 */
struct OccupationWeights {
    const double* edge = nullptr;
    const double* node = nullptr;
};

/*
 * Undirected graph on a symmetric n x n matrix.  Edges exist where
 * weights[i*n+j] >= threshold (threshold components) or > 0 (percolation).
 */
ComponentSummary connected_components(const double* weights, std::size_t n,
                                      double threshold);

/*
 * General percolation scan.  Candidate edges are adjacency > 0.  Bond
 * percolation retains each edge independently with probability
 * min(1, p * w_ij); site percolation first keeps each node independently with
 * probability min(1, p * v_i) and then retains an edge only when both
 * endpoints survived.  `repetitions` independent draws are averaged for each
 * p.  Giant fractions always use n as the denominator.
 */
std::vector<PercolationPoint> percolation_scan(
    const double* adjacency, std::size_t n, PercolationKind kind,
    const OccupationWeights& weights, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed);

enum class RemovalStrategy {
    strength = 0,
    degree = 1,
    eigenvector = 2,
    betweenness = 3,
    core = 4
};

/* Deterministic node-removal permutation; ties keep ascending node order. */
[[nodiscard]] std::vector<std::uint32_t> removal_order(
    const double* adjacency, std::size_t n, RemovalStrategy strategy);

struct CoreCurve {
    std::vector<std::int64_t> k;
    std::vector<std::int64_t> n_nodes;
    std::vector<double> fraction;
};

/* k-core survival curve derived from the core-number decomposition. */
[[nodiscard]] CoreCurve core_percolation(const double* adjacency,
                                         std::size_t n);

struct PercolationSummary {
    double critical_probability = 0.0;
    double p_half_giant = 0.0;
    double max_susceptibility = 0.0;
    double giant_area = 0.0;
};

/* Descriptive summaries of a percolation scan. */
[[nodiscard]] PercolationSummary percolation_summary(
    const double* probabilities, const double* giant_mean,
    const double* susceptibility, std::size_t n);

/* Convenience wrappers around percolation_scan(). */
std::vector<PercolationPoint> bond_percolation(const double* adjacency,
                                               std::size_t n,
                                               const double* probabilities,
                                               std::size_t n_probabilities,
                                               std::size_t repetitions,
                                               std::uint64_t seed);

std::vector<PercolationPoint> site_percolation(const double* adjacency,
                                               std::size_t n,
                                               const double* probabilities,
                                               std::size_t n_probabilities,
                                               std::size_t repetitions,
                                               std::uint64_t seed);

/* Weighted variants use the adjacency itself as the occupation multiplier. */
std::vector<PercolationPoint> bond_percolation_weighted(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed);

std::vector<PercolationPoint> site_percolation_weighted(
    const double* adjacency, std::size_t n, const double* node_weights,
    const double* probabilities, std::size_t n_probabilities,
    std::size_t repetitions, std::uint64_t seed);

/*
 * Mean cluster-size distribution at one occupation probability.  Index s
 * (1..n) receives the mean number of components of size s across repetitions;
 * index 0 is always zero.  The largest cluster is included.  For site
 * percolation, nodes that do not survive are not counted.
 */
std::vector<double> cluster_size_distribution(
    const double* adjacency, std::size_t n, PercolationKind kind,
    const OccupationWeights& weights, double probability,
    std::size_t repetitions, std::uint64_t seed);

struct RemovalPoint {
    std::size_t removed = 0;
    double giant_fraction = 0.0;
    double finite_susceptibility = 0.0;
    std::size_t n_components = 0;
};

/*
 * Deterministic node-removal (attack) curve.  `order` is a permutation of
 * 0..n-1; nodes are removed in that order.  Points are reported at
 * removed = floor(k*n/checkpoints) for k = 0..checkpoints, so checkpoints must
 * be positive; when it exceeds n some points repeat.  Fractions use n as the
 * denominator, matching site percolation.  No random numbers are involved.
 */
std::vector<RemovalPoint> targeted_removal(const double* adjacency,
                                           std::size_t n,
                                           const std::uint32_t* order,
                                           std::size_t checkpoints);

}  // namespace rflux::percolab
