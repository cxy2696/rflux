#include "rflux/percolab/percolation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "internal/graph.hpp"

namespace rflux::percolab {
namespace {

using rflux::detail::component_labels;
using rflux::detail::mix_seed;
using rflux::detail::require;
using rflux::detail::require_nonnegative_weights;
using rflux::detail::require_pointer;
using rflux::detail::Rng;
using rflux::detail::UnionFind;
using rflux::detail::validate_symmetric_matrix;

struct Edge {
    std::size_t u;
    std::size_t v;
};

std::vector<Edge> positive_edges(const double* adjacency, std::size_t n) {
    std::vector<Edge> edges;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (adjacency[i * n + j] > 0.0) edges.push_back({i, j});
        }
    }
    return edges;
}

void validate_percolation_inputs(const double* adjacency, std::size_t n,
                                 const double* probabilities,
                                 std::size_t n_probabilities,
                                 std::size_t repetitions) {
    validate_symmetric_matrix(adjacency, n, "adjacency");
    require_nonnegative_weights(adjacency, n, "adjacency");
    require_pointer(probabilities, "probabilities must not be NULL");
    require(n_probabilities > 0, "n_probabilities must be positive");
    require(repetitions > 0, "repetitions must be positive");
    for (std::size_t p = 0; p < n_probabilities; ++p) {
        rflux::detail::require_probability(probabilities[p],
                                           "each probability");
    }
}

struct ClusterMoments {
    double giant_fraction = 0.0;
    double finite_susceptibility = 0.0;
};

ClusterMoments summarize_components(UnionFind& uf, const std::uint8_t* alive,
                                    std::size_t n) {
    std::vector<std::size_t> counts(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (alive != nullptr && alive[i] == 0) continue;
        ++counts[uf.find(i)];
    }
    std::size_t largest = 0;
    long double sum_squares = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (counts[i] == 0) continue;
        largest = std::max(largest, counts[i]);
        sum_squares += static_cast<long double>(counts[i]) *
                       static_cast<long double>(counts[i]);
    }
    ClusterMoments result;
    result.giant_fraction =
        static_cast<double>(largest) / static_cast<double>(n);
    long double finite_squares =
        sum_squares - static_cast<long double>(largest) *
                          static_cast<long double>(largest);
    if (finite_squares < 0.0L) finite_squares = 0.0L;
    result.finite_susceptibility =
        static_cast<double>(finite_squares / static_cast<long double>(n));
    return result;
}

std::vector<PercolationPoint> finish_scan(
    const double* probabilities, std::size_t n_probabilities,
    const std::vector<long double>& giant_sum,
    const std::vector<long double>& giant_square_sum,
    const std::vector<long double>& susceptibility_sum,
    std::size_t repetitions) {
    std::vector<PercolationPoint> out(n_probabilities);
    const long double reps = static_cast<long double>(repetitions);
    for (std::size_t p = 0; p < n_probabilities; ++p) {
        const long double mean = giant_sum[p] / reps;
        long double variance = giant_square_sum[p] / reps - mean * mean;
        // E[x^2] - E[x]^2 can be a tiny negative or positive round-off when
        // every replicate is identical; do not report spurious error bars.
        if (variance < 1e-24L) variance = 0.0L;
        out[p].probability = probabilities[p];
        out[p].giant_mean = static_cast<double>(mean);
        out[p].giant_std = std::sqrt(static_cast<double>(variance));
        out[p].susceptibility =
            static_cast<double>(susceptibility_sum[p] / reps);
    }
    return out;
}

}  // namespace

ComponentSummary connected_components(const double* weights, std::size_t n,
                                      double threshold) {
    validate_symmetric_matrix(weights, n, "weights");
    require(std::isfinite(threshold), "threshold must be finite");
    UnionFind uf(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (weights[i * n + j] >= threshold) uf.unite(i, j);
        }
    }
    ComponentSummary summary;
    summary.labels =
        component_labels(uf, n, &summary.n_components, &summary.largest_size);
    return summary;
}

std::vector<PercolationPoint> bond_percolation(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed) {
    validate_percolation_inputs(adjacency, n, probabilities, n_probabilities,
                                repetitions);
    const std::vector<Edge> edges = positive_edges(adjacency, n);

    std::vector<long double> giant_sum(n_probabilities, 0.0L);
    std::vector<long double> giant_square_sum(n_probabilities, 0.0L);
    std::vector<long double> susceptibility_sum(n_probabilities, 0.0L);

    for (std::size_t p = 0; p < n_probabilities; ++p) {
        for (std::size_t repetition = 0; repetition < repetitions;
             ++repetition) {
            Rng rng(mix_seed(seed, repetition));
            UnionFind uf(n);
            for (const Edge& edge : edges) {
                if (rng.uniform() < probabilities[p]) {
                    uf.unite(edge.u, edge.v);
                }
            }
            const ClusterMoments moments =
                summarize_components(uf, nullptr, n);
            giant_sum[p] += static_cast<long double>(moments.giant_fraction);
            giant_square_sum[p] +=
                static_cast<long double>(moments.giant_fraction) *
                static_cast<long double>(moments.giant_fraction);
            susceptibility_sum[p] +=
                static_cast<long double>(moments.finite_susceptibility);
        }
    }
    return finish_scan(probabilities, n_probabilities, giant_sum,
                       giant_square_sum, susceptibility_sum, repetitions);
}

std::vector<PercolationPoint> site_percolation(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed) {
    validate_percolation_inputs(adjacency, n, probabilities, n_probabilities,
                                repetitions);
    const std::vector<Edge> edges = positive_edges(adjacency, n);

    std::vector<long double> giant_sum(n_probabilities, 0.0L);
    std::vector<long double> giant_square_sum(n_probabilities, 0.0L);
    std::vector<long double> susceptibility_sum(n_probabilities, 0.0L);

    for (std::size_t p = 0; p < n_probabilities; ++p) {
        for (std::size_t repetition = 0; repetition < repetitions;
             ++repetition) {
            Rng rng(mix_seed(seed, repetition));
            std::vector<std::uint8_t> alive(n, 0);
            for (std::size_t i = 0; i < n; ++i) {
                alive[i] = rng.uniform() < probabilities[p] ? 1 : 0;
            }
            UnionFind uf(n);
            for (const Edge& edge : edges) {
                if (alive[edge.u] != 0 && alive[edge.v] != 0) {
                    uf.unite(edge.u, edge.v);
                }
            }
            const ClusterMoments moments =
                summarize_components(uf, alive.data(), n);
            giant_sum[p] += static_cast<long double>(moments.giant_fraction);
            giant_square_sum[p] +=
                static_cast<long double>(moments.giant_fraction) *
                static_cast<long double>(moments.giant_fraction);
            susceptibility_sum[p] +=
                static_cast<long double>(moments.finite_susceptibility);
        }
    }
    return finish_scan(probabilities, n_probabilities, giant_sum,
                       giant_square_sum, susceptibility_sum, repetitions);
}

}  // namespace rflux::percolab
