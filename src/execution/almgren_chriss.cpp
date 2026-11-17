#include "rflux/execution/almgren_chriss.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include "internal/common.hpp"

namespace rflux::execution {
namespace {

using rflux::detail::require;

void require_nonnegative(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw rflux::InvalidArgument(std::string(name) +
                                     " must be finite and non-negative");
    }
}

}  // namespace

AlmgrenChrissPlanner::AlmgrenChrissPlanner(double temporary_impact,
                                           double permanent_impact,
                                           double volatility,
                                           double risk_aversion)
    : temporary_impact_(temporary_impact),
      permanent_impact_(permanent_impact),
      volatility_(volatility),
      risk_aversion_(risk_aversion) {
    require(std::isfinite(temporary_impact_) && temporary_impact_ > 0.0,
            "temporary_impact must be finite and positive");
    require_nonnegative(permanent_impact_, "permanent_impact");
    require_nonnegative(volatility_, "volatility");
    require_nonnegative(risk_aversion_, "risk_aversion");
}

core::ExecutionPlan AlmgrenChrissPlanner::plan(double quantity,
                                               std::size_t slices,
                                               double horizon) const {
    require(std::isfinite(quantity) && quantity > 0.0,
            "quantity must be finite and positive");
    require(slices > 0, "slices must be positive");
    require(std::isfinite(horizon) && horizon > 0.0,
            "horizon must be finite and positive");

    const double tau = horizon / static_cast<double>(slices);
    const double kappa = std::sqrt(
        risk_aversion_ * volatility_ * volatility_ / temporary_impact_);
    const bool linear = !(kappa * horizon > 1e-8) || !std::isfinite(kappa);

    core::ExecutionPlan result;
    result.kappa = linear ? 0.0 : kappa;
    result.trades.assign(slices, 0.0);
    double remaining = quantity;
    long double temp_cost = 0.0L;
    long double variance = 0.0L;
    for (std::size_t j = 1; j <= slices; ++j) {
        double target = 0.0;
        if (!linear) {
            const double time_left =
                horizon - static_cast<double>(j) * tau;
            const double scaled_horizon = kappa * horizon;
            double ratio = 0.0;
            if (scaled_horizon > 20.0) {
                const double scaled_left = kappa * time_left;
                ratio = std::exp(scaled_left - scaled_horizon) *
                        (1.0 - std::exp(-2.0 * scaled_left)) /
                        (1.0 - std::exp(-2.0 * scaled_horizon));
            } else {
                ratio = std::sinh(kappa * time_left) /
                        std::sinh(scaled_horizon);
            }
            target = quantity * ratio;
        } else {
            target = quantity * (1.0 - static_cast<double>(j) /
                                            static_cast<double>(slices));
        }
        if (j == slices) target = 0.0;
        const double trade = remaining - target;
        result.trades[j - 1] = trade;
        const long double rate = static_cast<long double>(trade) /
                                 static_cast<long double>(tau);
        temp_cost += rate * rate;
        variance += static_cast<long double>(tau) *
                    static_cast<long double>(target) *
                    static_cast<long double>(target);
        remaining = target;
    }
    result.expected_cost =
        0.5 * permanent_impact_ * quantity * quantity +
        temporary_impact_ * static_cast<double>(temp_cost);
    result.expected_variance =
        volatility_ * volatility_ * static_cast<double>(variance);
    return result;
}

}  // namespace rflux::execution
