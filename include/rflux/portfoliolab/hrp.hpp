#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "rflux/core/algorithm.hpp"

namespace rflux::portfoliolab {

/*
 * Hierarchical Risk Parity (López de Prado):
 *   correlation distance -> single-linkage tree -> quasi-diagonal order ->
 *   recursive bisection with inverse-variance cluster risk budgets.
 *
 * The result is long-only, non-negative and sums to one.  It uses only a
 * covariance matrix and never inverts it.
 */
class HRPAllocator final : public core::BaseAllocator {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "hrp";
    }
    [[nodiscard]] std::vector<double> weights(
        const double* covariance, std::size_t n_assets) const override;
};

/* Return the quasi-diagonal leaf order used by HRP (useful for diagnostics). */
[[nodiscard]] std::vector<std::size_t> quasi_diagonal_order(
    const double* covariance, std::size_t n_assets);

/* Diagonal inverse-variance weights; a simple risk-parity baseline. */
[[nodiscard]] std::vector<double> risk_parity_weights(
    const double* covariance, std::size_t n_assets);

/*
 * Nested clustered risk parity.  Correlations are grouped by single-linkage
 * distance, each group is HRP-allocated and groups receive inverse
 * cluster-variance budgets.
 */
[[nodiscard]] std::vector<double> nco_weights(
    const double* covariance, std::size_t n_assets,
    double distance_threshold = 0.5);

/* Per-row rebalance gate: use target when max |target-current| >= threshold. */
[[nodiscard]] std::vector<double> rebalance_targets(
    const double* current, const double* target, std::size_t n_rows,
    std::size_t n_assets, double threshold);

}  // namespace rflux::portfoliolab
