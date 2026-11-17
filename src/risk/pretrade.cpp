#include "rflux/risk/pretrade.hpp"

#include <cmath>
#include <string>
#include <utility>

#include "internal/common.hpp"

namespace rflux::risk {
namespace {

void validate_constraints(const core::RiskConstraints& constraints) {
    rflux::detail::require(
        std::isfinite(constraints.max_position_limit) &&
            constraints.max_position_limit > 0.0,
        "max_position_limit must be finite and positive");
    rflux::detail::require(
        std::isfinite(constraints.max_drawdown_limit) &&
            constraints.max_drawdown_limit >= 0.0 &&
            constraints.max_drawdown_limit <= 1.0,
        "max_drawdown_limit must be in [0, 1]");
    rflux::detail::require(
        std::isfinite(constraints.turnover_threshold) &&
            constraints.turnover_threshold >= 0.0,
        "turnover_threshold must be finite and non-negative");
}

RiskDecision reject(std::string reason) {
    RiskDecision decision;
    decision.allowed = false;
    decision.reason = std::move(reason);
    return decision;
}

}  // namespace

PreTradeRiskControl::PreTradeRiskControl(core::RiskConstraints constraints)
    : constraints_(constraints) {
    validate_constraints(constraints_);
}

RiskDecision PreTradeRiskControl::check(
    const core::Order& order, double projected_position_fraction,
    double current_drawdown_fraction, double projected_turnover,
    const percolab::PercolationResult* percolation) const {
    if (order.quantity <= 0) {
        return reject("order quantity must be positive");
    }
    if (!std::isfinite(order.price) || order.price <= 0.0) {
        return reject("order price must be finite and positive");
    }
    if (!std::isfinite(projected_position_fraction)) {
        return reject("projected position must be finite");
    }
    if (std::abs(projected_position_fraction) >
        constraints_.max_position_limit) {
        return reject("projected position exceeds max_position_limit");
    }
    if (!std::isfinite(current_drawdown_fraction) ||
        current_drawdown_fraction < 0.0) {
        return reject("current drawdown must be finite and non-negative");
    }
    if (current_drawdown_fraction > constraints_.max_drawdown_limit) {
        return reject("drawdown limit breached");
    }
    if (!std::isfinite(projected_turnover) || projected_turnover < 0.0) {
        return reject("projected turnover must be finite and non-negative");
    }
    if (projected_turnover > constraints_.turnover_threshold) {
        return reject("projected turnover exceeds turnover_threshold");
    }
    if (percolation != nullptr && percolation->is_systemic_crisis) {
        return reject("systemic percolation crisis");
    }
    return RiskDecision{true, {}};
}

}  // namespace rflux::risk
