#include "rflux/percolab/percolation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "internal/graph.hpp"
#include "internal/parallel.hpp"

namespace rflux::percolab {
namespace {

using rflux::detail::component_labels;
using rflux::detail::mix_seed;
using rflux::detail::require;
using rflux::detail::require_adjacency;
using rflux::detail::RunningStats;
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

void validate_scan_inputs(const double* adjacency, std::size_t n,
                          const double* probabilities,
                          std::size_t n_probabilities,
                          std::size_t repetitions) {
    require_adjacency(adjacency, n, "adjacency");
    require_pointer(probabilities, "probabilities must not be NULL");
    require(n_probabilities > 0, "n_probabilities must be positive");
    require(repetitions > 0, "repetitions must be positive");
    for (std::size_t p = 0; p < n_probabilities; ++p) {
        rflux::detail::require_probability(probabilities[p],
                                           "each probability");
    }
}

void validate_node_weights(const double* weights, std::size_t n,
                           const char* what) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(weights[i])) {
            throw NonFinite(std::string(what) +
                            " must contain only finite values");
        }
        if (weights[i] < 0.0) {
            throw InvalidArgument(std::string(what) + " must be non-negative");
        }
    }
}

/* Validate the optional multipliers and reject those unused by the kind. */
void validate_occupation_weights(PercolationKind kind,
                                 const OccupationWeights& weights,
                                 std::size_t n) {
    if (kind == PercolationKind::bond) {
        if (weights.node != nullptr) {
            throw InvalidArgument(
                "node_weights are only used by site percolation");
        }
        if (weights.edge != nullptr) {
            require_adjacency(weights.edge, n, "edge_weights");
        }
    } else {
        if (weights.edge != nullptr) {
            throw InvalidArgument(
                "edge_weights are only used by bond percolation");
        }
        if (weights.node != nullptr) {
            validate_node_weights(weights.node, n, "node_weights");
        }
    }
}

struct ClusterMoments {
    double giant_fraction = 0.0;
    double finite_susceptibility = 0.0;
    std::size_t n_components = 0;
};

void sample_bond(const std::vector<Edge>& edges, std::size_t n, double p,
                 const double* edge_weights, Rng& rng, UnionFind& uf) {
    for (const Edge& edge : edges) {
        const double weight =
            edge_weights == nullptr ? 1.0 : edge_weights[edge.u * n + edge.v];
        if (rng.uniform() < std::min(1.0, p * weight)) {
            uf.unite(edge.u, edge.v);
        }
    }
}

void unite_alive_edges(const std::vector<Edge>& edges,
                       const std::uint8_t* alive, UnionFind& uf) {
    for (const Edge& edge : edges) {
        if (alive[edge.u] != 0 && alive[edge.v] != 0) {
            uf.unite(edge.u, edge.v);
        }
    }
}

void sample_site(const std::vector<Edge>& edges, std::size_t n, double p,
                 const double* node_weights, Rng& rng, UnionFind& uf,
                 std::vector<std::uint8_t>& alive) {
    for (std::size_t i = 0; i < n; ++i) {
        const double weight =
            node_weights == nullptr ? 1.0 : node_weights[i];
        alive[i] = rng.uniform() < std::min(1.0, p * weight) ? 1 : 0;
    }
    unite_alive_edges(edges, alive.data(), uf);
}

ClusterMoments summarize_components(UnionFind& uf, const std::uint8_t* alive,
                                    std::size_t n) {
    std::vector<std::size_t> counts(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (alive != nullptr && alive[i] == 0) continue;
        ++counts[uf.find(i)];
    }

    std::size_t largest = 0;
    for (std::size_t i = 0; i < n; ++i) {
        largest = std::max(largest, counts[i]);
    }

    std::size_t n_components = 0;
    long double finite_squares = 0.0L;
    bool excluded_largest = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (counts[i] == 0) continue;
        ++n_components;
        if (counts[i] == largest && !excluded_largest) {
            excluded_largest = true;
            continue;
        }
        finite_squares += static_cast<long double>(counts[i]) *
                          static_cast<long double>(counts[i]);
    }

    ClusterMoments result;
    result.giant_fraction =
        static_cast<double>(largest) / static_cast<double>(n);
    result.finite_susceptibility =
        static_cast<double>(finite_squares / static_cast<long double>(n));
    result.n_components = n_components;
    return result;
}

void accumulate_cluster_sizes(UnionFind& uf, const std::uint8_t* alive,
                              std::size_t n,
                              std::vector<long double>& counts) {
    std::vector<std::size_t> sizes(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (alive != nullptr && alive[i] == 0) continue;
        ++sizes[uf.find(i)];
    }
    for (std::size_t size : sizes) {
        if (size > 0) counts[size] += 1.0L;
    }
}

template <typename Replicate>
std::vector<PercolationPoint> run_scan(const double* probabilities,
                                       std::size_t n_probabilities,
                                       std::size_t repetitions,
                                       std::size_t node_count,
                                       std::uint64_t seed,
                                       Replicate replicate) {
    std::vector<RunningStats> giant(n_probabilities);
    std::vector<RunningStats> susceptibility(n_probabilities);
    /*
     * Probability points are independent, and every point keeps its
     * repetitions (and therefore its random stream and accumulation order) in
     * increasing order.  Parallelising over points is thus bit-for-bit
     * deterministic while using every core.  Tiny graphs skip the fork-join
     * because thread setup would cost more than the trials themselves.
     */
    const bool use_parallel =
        node_count >= 128 && n_probabilities >= 2 && repetitions >= 4;
    detail::parallel_for(
        n_probabilities,
        [&](std::size_t p) {
            for (std::size_t repetition = 0; repetition < repetitions;
                 ++repetition) {
                Rng rng(mix_seed(seed, repetition));
                const ClusterMoments moments =
                    replicate(probabilities[p], rng);
                giant[p].add(
                    static_cast<long double>(moments.giant_fraction));
                susceptibility[p].add(
                    static_cast<long double>(moments.finite_susceptibility));
            }
        },
        use_parallel ? 0 : 1);

    std::vector<PercolationPoint> out(n_probabilities);
    for (std::size_t p = 0; p < n_probabilities; ++p) {
        out[p].probability = probabilities[p];
        out[p].giant_mean = static_cast<double>(giant[p].mean);
        out[p].giant_std =
            std::sqrt(static_cast<double>(giant[p].variance()));
        out[p].susceptibility =
            static_cast<double>(susceptibility[p].mean);
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

std::vector<PercolationPoint> percolation_scan(
    const double* adjacency, std::size_t n, PercolationKind kind,
    const OccupationWeights& weights, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed) {
    validate_scan_inputs(adjacency, n, probabilities, n_probabilities,
                         repetitions);
    validate_occupation_weights(kind, weights, n);

    const std::vector<Edge> edges = positive_edges(adjacency, n);
    const bool site = kind == PercolationKind::site;
    return run_scan(probabilities, n_probabilities, repetitions, n, seed,
                    [&](double p, Rng& rng) {
                        UnionFind uf(n);
                        std::vector<std::uint8_t> alive;
                        if (site) {
                            alive.assign(n, 0);
                            sample_site(edges, n, p, weights.node, rng, uf,
                                        alive);
                        } else {
                            sample_bond(edges, n, p, weights.edge, rng, uf);
                        }
                        return summarize_components(
                            uf, site ? alive.data() : nullptr, n);
                    });
}


PercolationSummary percolation_summary(const double* probabilities,
                                       const double* giant_mean,
                                       const double* susceptibility,
                                       std::size_t n) {
    rflux::detail::require_pointer(probabilities,
                                   "probabilities must not be NULL");
    rflux::detail::require_pointer(giant_mean,
                                   "giant_mean must not be NULL");
    rflux::detail::require_pointer(susceptibility,
                                   "susceptibility must not be NULL");
    PercolationSummary result;
    result.critical_probability = rflux::detail::kNaN;
    result.p_half_giant = rflux::detail::kNaN;
    result.max_susceptibility = rflux::detail::kNaN;
    result.giant_area = 0.0;
    if (n == 0) return result;
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (susceptibility[i] > susceptibility[best]) best = i;
    }
    result.critical_probability = probabilities[best];
    double maximum = susceptibility[0];
    for (std::size_t i = 1; i < n; ++i) {
        maximum = std::max(maximum, susceptibility[i]);
    }
    result.max_susceptibility = maximum;
    const double terminal = giant_mean[n - 1];
    if (std::isfinite(terminal) && terminal > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            if (giant_mean[i] >= 0.5 * terminal) {
                result.p_half_giant = probabilities[i];
                break;
            }
        }
    }
    for (std::size_t i = 1; i < n; ++i) {
        const double width = probabilities[i] - probabilities[i - 1];
        const double height = 0.5 * (giant_mean[i] + giant_mean[i - 1]);
        result.giant_area += width * height;
    }
    return result;
}

std::vector<PercolationPoint> bond_percolation(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed) {
    return percolation_scan(adjacency, n, PercolationKind::bond,
                            OccupationWeights{}, probabilities,
                            n_probabilities, repetitions, seed);
}

std::vector<PercolationPoint> site_percolation(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed) {
    return percolation_scan(adjacency, n, PercolationKind::site,
                            OccupationWeights{}, probabilities,
                            n_probabilities, repetitions, seed);
}

std::vector<PercolationPoint> bond_percolation_weighted(
    const double* adjacency, std::size_t n, const double* probabilities,
    std::size_t n_probabilities, std::size_t repetitions, std::uint64_t seed) {
    OccupationWeights weights;
    weights.edge = adjacency;
    return percolation_scan(adjacency, n, PercolationKind::bond, weights,
                            probabilities, n_probabilities, repetitions, seed);
}

std::vector<PercolationPoint> site_percolation_weighted(
    const double* adjacency, std::size_t n, const double* node_weights,
    const double* probabilities, std::size_t n_probabilities,
    std::size_t repetitions, std::uint64_t seed) {
    OccupationWeights weights;
    weights.node = node_weights;
    return percolation_scan(adjacency, n, PercolationKind::site, weights,
                            probabilities, n_probabilities, repetitions, seed);
}

std::vector<double> cluster_size_distribution(
    const double* adjacency, std::size_t n, PercolationKind kind,
    const OccupationWeights& weights, double probability,
    std::size_t repetitions, std::uint64_t seed) {
    require_adjacency(adjacency, n, "adjacency");
    rflux::detail::require_probability(probability, "probability");
    require(repetitions > 0, "repetitions must be positive");
    validate_occupation_weights(kind, weights, n);

    const std::vector<Edge> edges = positive_edges(adjacency, n);
    const bool site = kind == PercolationKind::site;
    /*
     * Replicates are independent, so each worker accumulates a private count
     * vector and the reduction happens in worker order.  Counts are exact
     * integers, which keeps the result deterministic regardless of scheduling.
     */
    const std::size_t workers =
        (n >= 128 && repetitions >= 8) ? detail::worker_count(repetitions) : 1;
    std::vector<std::vector<long double>> partial(
        workers, std::vector<long double>(n + 1, 0.0L));
    const std::size_t chunk =
        repetitions / workers + (repetitions % workers != 0);
    detail::parallel_for(
        workers,
        [&](std::size_t worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(repetitions, begin + chunk);
            if (begin >= end) return;
            std::vector<long double>& counts = partial[worker];
            std::vector<std::uint8_t> alive;
            for (std::size_t repetition = begin; repetition < end;
                 ++repetition) {
                Rng rng(mix_seed(seed, repetition));
                UnionFind uf(n);
                if (site) {
                    alive.assign(n, 0);
                    sample_site(edges, n, probability, weights.node, rng, uf,
                                alive);
                    accumulate_cluster_sizes(uf, alive.data(), n, counts);
                } else {
                    sample_bond(edges, n, probability, weights.edge, rng, uf);
                    accumulate_cluster_sizes(uf, nullptr, n, counts);
                }
            }
        },
        workers);

    std::vector<long double> counts(n + 1, 0.0L);
    for (const std::vector<long double>& worker_counts : partial) {
        for (std::size_t size = 1; size <= n; ++size) {
            counts[size] += worker_counts[size];
        }
    }

    std::vector<double> out(n + 1, 0.0);
    const long double reps = static_cast<long double>(repetitions);
    for (std::size_t size = 1; size <= n; ++size) {
        out[size] = static_cast<double>(counts[size] / reps);
    }
    return out;
}

std::vector<RemovalPoint> targeted_removal(const double* adjacency,
                                           std::size_t n,
                                           const std::uint32_t* order,
                                           std::size_t checkpoints) {
    require_adjacency(adjacency, n, "adjacency");
    require_pointer(order, "order must not be NULL");
    require(checkpoints > 0, "checkpoints must be positive");
    require(n <= static_cast<std::size_t>(
                     std::numeric_limits<std::uint32_t>::max()),
            "n must fit in uint32_t");

    std::vector<std::uint8_t> seen(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t node = static_cast<std::size_t>(order[i]);
        require(node < n, "order entries must be in [0, n)");
        require(seen[node] == 0, "order must be a permutation of 0..n-1");
        seen[node] = 1;
    }

    const std::vector<Edge> edges = positive_edges(adjacency, n);
    std::vector<std::uint8_t> alive(n, 1);
    std::vector<RemovalPoint> out(checkpoints + 1);
    std::size_t removed = 0;
    for (std::size_t k = 0; k <= checkpoints; ++k) {
        const unsigned long long target =
            (static_cast<unsigned long long>(k) *
             static_cast<unsigned long long>(n)) /
            static_cast<unsigned long long>(checkpoints);
        while (removed < static_cast<std::size_t>(target)) {
            alive[static_cast<std::size_t>(order[removed])] = 0;
            ++removed;
        }

        UnionFind uf(n);
        unite_alive_edges(edges, alive.data(), uf);
        const ClusterMoments moments =
            summarize_components(uf, alive.data(), n);
        out[k].removed = removed;
        out[k].giant_fraction = moments.giant_fraction;
        out[k].finite_susceptibility = moments.finite_susceptibility;
        out[k].n_components = moments.n_components;
    }
    return out;
}

}  // namespace rflux::percolab
