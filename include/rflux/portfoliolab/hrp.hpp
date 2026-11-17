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

}  // namespace rflux::portfoliolab
