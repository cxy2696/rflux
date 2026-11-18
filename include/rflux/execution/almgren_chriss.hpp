#pragma once

#include <cstddef>
#include <string_view>

#include "rflux/core/algorithm.hpp"

namespace rflux::execution {

/*
 * Almgren-Chriss optimal liquidation with linear temporary/permanent impact.
 *
 * units:
 *   quantity          shares
 *   horizon           trading horizon in the same time unit as volatility
 *   temporary_impact  price impact per (share / time) traded
 *   permanent_impact  price impact per share
 *   volatility        price volatility per time unit
 *   risk_aversion     non-negative risk-aversion coefficient
 *
 * The planner returns the deterministic optimal trajectory, the expected
 * implementation cost and its variance.  It does not model discrete lots,
 * tick sizes, dark pools or queue position; those belong to an execution
 * venue adapter.
 */
class AlmgrenChrissPlanner final : public core::BaseExecutionPlanner {
public:
    AlmgrenChrissPlanner(double temporary_impact, double permanent_impact,
                         double volatility, double risk_aversion);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "almgren_chriss";
    }
    [[nodiscard]] core::ExecutionPlan plan(double quantity,
                                           std::size_t slices,
                                           double horizon) const override;

private:
    double temporary_impact_;
    double permanent_impact_;
    double volatility_;
    double risk_aversion_;
};

struct OrderBookMetrics {
    std::vector<double> spread;
    std::vector<double> mid;
    std::vector<double> microprice;
    std::vector<double> imbalance;
};

/* Level-1 spread, mid, microprice and size imbalance. */
[[nodiscard]] OrderBookMetrics order_book_metrics(
    const double* bid_price, const double* bid_size, const double* ask_price,
    const double* ask_size, std::size_t n);

}  // namespace rflux::execution
