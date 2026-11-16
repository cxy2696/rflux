#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::percolab {

struct ComponentSummary {
    std::vector<std::int32_t> labels; /* 0..C-1 */
    std::size_t n_components = 0;
    std::size_t largest_size = 0;
};

struct PercolationPoint {
    double probability = 0.0;
    double giant_mean = 0.0;
    double giant_std = 0.0;
    double susceptibility = 0.0;
};

/*
 * Undirected graph on a symmetric n x n matrix.  Edges exist where
 * weights[i*n+j] >= threshold (threshold components) or > 0 (percolation).
 */
ComponentSummary connected_components(const double* weights, std::size_t n,
                                      double threshold);

/*
 * Classical bond percolation.  Edges with adjacency > 0 are retained with
 * probability p.  `repetitions` independent draws are averaged for each p.
 */
std::vector<PercolationPoint> bond_percolation(const double* adjacency,
                                               std::size_t n,
                                               const double* probabilities,
                                               std::size_t n_probabilities,
                                               std::size_t repetitions,
                                               std::uint64_t seed);

/* Site percolation with survival probability p, giant size relative to n. */
std::vector<PercolationPoint> site_percolation(const double* adjacency,
                                               std::size_t n,
                                               const double* probabilities,
                                               std::size_t n_probabilities,
                                               std::size_t repetitions,
                                               std::uint64_t seed);

/*
 * Weighted variants.  Edges are retained with probability min(1, p * w_ij);
 * nodes survive with probability min(1, p * w_i).  Weights must be finite and
 * non-negative.  The graph still has to be symmetric and non-negative.
 */
std::vector<PercolationPoint> bond_percolation_weighted(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed);

std::vector<PercolationPoint> site_percolation_weighted(
    const double* adjacency, std::size_t n, const double* node_weights,
    const double* probabilities, std::size_t n_probabilities,
    std::size_t repetitions, std::uint64_t seed);

}  // namespace rflux::percolab
