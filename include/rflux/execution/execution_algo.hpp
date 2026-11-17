#pragma once

#include <string_view>
#include <vector>

#include "rflux/core/algorithm.hpp"
#include "rflux/core/market.hpp"

namespace rflux::execution {

/*
 * High-frequency execution contract.
 *
 * The request carries symbols, target/current weights and portfolio value.
 * The order book belongs to one symbol and one timestamp; a venue router or a
 * multi-asset parent-order engine should split requests before calling this
 * interface.  Implementations must not assume unlimited liquidity or that every
 * child order fills.
 */
class BaseExecutionAlgo : public core::BaseAlgorithm {
public:
    [[nodiscard]] virtual std::vector<core::Order> execute(
        const core::ExecutionRequest& request,
        const core::OrderBookView& order_book) const = 0;
};

}  // namespace rflux::execution
