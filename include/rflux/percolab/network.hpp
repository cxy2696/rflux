#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::percolab {

struct MstEdge {
    std::uint32_t u = 0;
    std::uint32_t v = 0;
    double weight = 0.0;
};

struct DiffusionResult {
    std::vector<double> mean_affected;        /* length steps+1 */
    std::vector<double> infection_frequency;  /* length n */
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
