#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rflux/core/algorithm.hpp"

namespace rflux::percolab {

/*
 * Sparse undirected financial network in CSR form.
 *
 * `row_offsets` has num_nodes+1 entries; row i owns columns
 * [row_offsets[i], row_offsets[i+1]).  Both directions of an undirected edge
 * may be stored; duplicate edges are harmless to the union-find consumer.
 */
struct FinancialNetwork {
    std::size_t num_nodes = 0;
    std::vector<std::size_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<double> edge_weights;
    std::unordered_map<std::string, std::uint32_t> symbol_to_index;

    [[nodiscard]] std::size_t edge_count() const noexcept {
        return column_indices.size();
    }
};

/* Build a CSR network from a dense row-major n x n adjacency matrix. */
[[nodiscard]] FinancialNetwork make_network(const double* adjacency,
                                             std::size_t num_nodes);

struct PercolationResult {
    double shock_probability = 0.0;
    /* NaN until estimate_threshold() has been called for this detector. */
    double percolation_threshold =
        std::numeric_limits<double>::quiet_NaN();
    double giant_component_fraction = 0.0;
    double giant_component_std = 0.0;
    double susceptibility = 0.0;
    /* Mean largest component size across trials, rounded to nearest. */
    std::size_t largest_component = 0;
    bool is_systemic_crisis = false;
    std::size_t repetitions = 0;
};

class BasePercolationDetector : public core::BaseAlgorithm {
public:
    /*
     * Analyze the network at one external shock probability p in [0, 1].
     * The result is a Monte Carlo estimate, not a deterministic graph query.
     */
    [[nodiscard]] virtual PercolationResult analyze(
        const FinancialNetwork& network, double shock_probability) = 0;

    /*
     * Estimate the critical occupation probability p_c by scanning a
     * configurable grid and returning the first p where the mean giant
     * component reaches the configured crisis fraction.
     */
    [[nodiscard]] virtual double estimate_threshold(
        const FinancialNetwork& network) = 0;
};

class SitePercolationDetector final : public BasePercolationDetector {
public:
    struct Options {
        std::size_t repetitions = 128;
        std::uint64_t seed = 42;
        double crisis_fraction = 0.5;
        std::size_t threshold_points = 21;
    };

    SitePercolationDetector();
    explicit SitePercolationDetector(Options options);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "site_percolation";
    }
    [[nodiscard]] PercolationResult analyze(
        const FinancialNetwork& network, double shock_probability) override;
    [[nodiscard]] double estimate_threshold(
        const FinancialNetwork& network) override;

    void clear_calibration() noexcept { threshold_.reset(); }
    [[nodiscard]] bool calibrated() const noexcept {
        return threshold_.has_value();
    }

private:
    [[nodiscard]] PercolationResult simulate(const FinancialNetwork& network,
                                             double probability) const;
    Options options_;
    std::optional<double> threshold_;
};

}  // namespace rflux::percolab
