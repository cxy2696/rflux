#pragma once

#include <string>

#include "rflux/core/market.hpp"
#include "rflux/percolab/detector.hpp"

namespace rflux::risk {

struct RiskDecision {
    bool allowed = true;
    std::string reason;
};

/*
 * Deterministic pre-trade gate.  It combines position, drawdown and turnover
 * limits with an optional systemic-risk signal from a calibrated percolation
 * detector.  The class holds no mutable state and is safe to share between
 * worker threads.
 */
class PreTradeRiskControl {
public:
    explicit PreTradeRiskControl(core::RiskConstraints constraints);

    [[nodiscard]] RiskDecision check(
        const core::Order& order, double projected_position_fraction,
        double current_drawdown_fraction, double projected_turnover,
        const percolab::PercolationResult* percolation = nullptr) const;

    [[nodiscard]] const core::RiskConstraints& constraints() const noexcept {
        return constraints_;
    }

private:
    core::RiskConstraints constraints_;
};

}  // namespace rflux::risk
