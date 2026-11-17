#include "rflux/percolab/detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

#include "internal/common.hpp"
#include "internal/graph.hpp"

namespace rflux::percolab {
namespace {

using rflux::detail::kNaN;
using rflux::detail::mix_seed;
using rflux::detail::require;
using rflux::detail::UnionFind;

void validate_network(const FinancialNetwork& network) {
    require(network.num_nodes > 0, "network must contain at least one node");
    require(network.row_offsets.size() == network.num_nodes + 1,
            "network row_offsets must have num_nodes + 1 entries");
    require(network.row_offsets.front() == 0,
            "network row_offsets must start at zero");
    require(network.row_offsets.back() == network.column_indices.size(),
            "network row_offsets must terminate at edge_count");
    require(network.column_indices.size() == network.edge_weights.size(),
            "network columns and weights must have equal lengths");
    for (std::size_t i = 0; i < network.num_nodes; ++i) {
        require(network.row_offsets[i] <= network.row_offsets[i + 1],
                "network row_offsets must be monotonic");
        for (std::size_t e = network.row_offsets[i];
             e < network.row_offsets[i + 1]; ++e) {
            require(network.column_indices[e] < network.num_nodes,
                    "network column index out of range");
            require(std::isfinite(network.edge_weights[e]) &&
                        network.edge_weights[e] > 0.0,
                    "network edge weights must be finite and positive");
        }
    }
    for (const auto& entry : network.symbol_to_index) {
        require(entry.first.size() > 0, "network symbol must not be empty");
        require(entry.second < network.num_nodes,
                "network symbol index out of range");
    }
}

struct Trial {
    std::size_t largest = 0;
    double giant = 0.0;
    double susceptibility = 0.0;
};

Trial run_trial(const FinancialNetwork& network, double probability,
                std::uint64_t seed) {
    const std::size_t n = network.num_nodes;
    UnionFind components(n);
    std::vector<std::uint8_t> alive(n, 0);
    rflux::detail::Rng rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        alive[i] = rng.uniform() < probability ? 1U : 0U;
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!alive[i]) continue;
        for (std::size_t e = network.row_offsets[i];
             e < network.row_offsets[i + 1]; ++e) {
            const std::uint32_t j = network.column_indices[e];
            if (j > i && alive[j]) components.unite(i, j);
        }
    }

    std::vector<std::size_t> component_sizes(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (alive[i]) ++component_sizes[components.find(i)];
    }
    component_sizes.erase(
        std::remove(component_sizes.begin(), component_sizes.end(),
                    static_cast<std::size_t>(0)),
        component_sizes.end());
    Trial trial;
    if (component_sizes.empty()) return trial;
    std::sort(component_sizes.begin(), component_sizes.end());
    trial.largest = component_sizes.back();
    trial.giant = static_cast<double>(trial.largest) /
                  static_cast<double>(n);
    long double finite_squares = 0.0L;
    for (std::size_t i = 0; i + 1 < component_sizes.size(); ++i) {
        finite_squares += static_cast<long double>(component_sizes[i]) *
                          static_cast<long double>(component_sizes[i]);
    }
    trial.susceptibility = static_cast<double>(
        finite_squares / static_cast<long double>(n));
    return trial;
}

}  // namespace

FinancialNetwork make_network(const double* adjacency,
                              std::size_t num_nodes) {
    rflux::detail::require_pointer(adjacency, "adjacency must not be null");
    require(num_nodes > 0, "network must contain at least one node");
    FinancialNetwork network;
    network.num_nodes = num_nodes;
    network.row_offsets.assign(num_nodes + 1, 0);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        for (std::size_t j = 0; j < num_nodes; ++j) {
            const double weight = adjacency[i * num_nodes + j];
            require(std::isfinite(weight), "adjacency must be finite");
            if (i == j || !(weight > 0.0)) continue;
            network.column_indices.push_back(static_cast<std::uint32_t>(j));
            network.edge_weights.push_back(weight);
        }
        network.row_offsets[i + 1] = network.column_indices.size();
    }
    return network;
}

SitePercolationDetector::SitePercolationDetector()
    : SitePercolationDetector(Options{}) {}

SitePercolationDetector::SitePercolationDetector(Options options)
    : options_(options) {
    require(options_.repetitions > 0, "repetitions must be positive");
    require(options_.threshold_points > 1,
            "threshold_points must be at least two");
    require(std::isfinite(options_.crisis_fraction) &&
                options_.crisis_fraction > 0.0 &&
                options_.crisis_fraction <= 1.0,
            "crisis_fraction must be in (0, 1]");
}

PercolationResult SitePercolationDetector::simulate(
    const FinancialNetwork& network, double probability) const {
    validate_network(network);
    require(std::isfinite(probability) && probability >= 0.0 &&
                probability <= 1.0,
            "shock_probability must be in [0, 1]");
    long double giant_sum = 0.0L;
    long double giant_square_sum = 0.0L;
    long double susceptibility_sum = 0.0L;
    std::size_t largest_sum = 0;
    for (std::size_t trial = 0; trial < options_.repetitions; ++trial) {
        const std::uint64_t seed =
            mix_seed(options_.seed, static_cast<std::uint64_t>(trial));
        const Trial result = run_trial(network, probability, seed);
        giant_sum += result.giant;
        giant_square_sum += static_cast<long double>(result.giant) *
                            static_cast<long double>(result.giant);
        susceptibility_sum += result.susceptibility;
        largest_sum += result.largest;
    }
    const long double count = static_cast<long double>(options_.repetitions);
    const double mean_giant = static_cast<double>(giant_sum / count);
    const double variance = std::max(
        0.0, static_cast<double>(giant_square_sum / count) -
                 mean_giant * mean_giant);
    PercolationResult result;
    result.shock_probability = probability;
    result.percolation_threshold = threshold_.value_or(kNaN);
    result.giant_component_fraction = mean_giant;
    result.giant_component_std = std::sqrt(variance);
    result.susceptibility =
        static_cast<double>(susceptibility_sum / count);
    result.largest_component = static_cast<std::size_t>(std::llround(
        static_cast<double>(largest_sum) /
        static_cast<double>(options_.repetitions)));
    result.is_systemic_crisis =
        threshold_.has_value()
            ? probability >= *threshold_
            : mean_giant >= options_.crisis_fraction;
    result.repetitions = options_.repetitions;
    return result;
}

PercolationResult SitePercolationDetector::analyze(
    const FinancialNetwork& network, double shock_probability) {
    return simulate(network, shock_probability);
}

double SitePercolationDetector::estimate_threshold(
    const FinancialNetwork& network) {
    validate_network(network);
    double threshold = 1.0;
    for (std::size_t point = 0; point < options_.threshold_points; ++point) {
        const double p = static_cast<double>(point) /
                         static_cast<double>(options_.threshold_points - 1);
        const PercolationResult result = simulate(network, p);
        if (result.giant_component_fraction >= options_.crisis_fraction) {
            threshold = p;
            break;
        }
    }
    threshold_ = threshold;
    return threshold;
}

}  // namespace rflux::percolab
