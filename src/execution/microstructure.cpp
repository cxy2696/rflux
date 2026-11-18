#include "rflux/execution/almgren_chriss.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "internal/common.hpp"

namespace rflux::execution {

OrderBookMetrics order_book_metrics(const double* bid_price,
                                    const double* bid_size,
                                    const double* ask_price,
                                    const double* ask_size, std::size_t n) {
    rflux::detail::require_pointer(bid_price, "bid_price must not be NULL");
    rflux::detail::require_pointer(bid_size, "bid_size must not be NULL");
    rflux::detail::require_pointer(ask_price, "ask_price must not be NULL");
    rflux::detail::require_pointer(ask_size, "ask_size must not be NULL");
    rflux::detail::require(n > 0, "n must be positive");
    OrderBookMetrics result;
    result.spread.resize(n);
    result.mid.resize(n);
    result.microprice.resize(n);
    result.imbalance.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double bid = bid_price[i];
        const double ask = ask_price[i];
        const double bid_qty = bid_size[i];
        const double ask_qty = ask_size[i];
        rflux::detail::require(std::isfinite(bid) && std::isfinite(ask),
                               "prices must be finite");
        rflux::detail::require(std::isfinite(bid_qty) &&
                                   std::isfinite(ask_qty),
                               "sizes must be finite");
        rflux::detail::require(bid > 0.0 && ask > 0.0,
                               "prices must be positive");
        rflux::detail::require(bid_qty >= 0.0 && ask_qty >= 0.0,
                               "sizes must be non-negative");
        rflux::detail::require(ask >= bid,
                               "ask price must not be below bid price");
        const double total = bid_qty + ask_qty;
        result.spread[i] = ask - bid;
        result.mid[i] = (ask + bid) * 0.5;
        if (total > 0.0) {
            result.microprice[i] =
                (ask * bid_qty + bid * ask_qty) / total;
            result.imbalance[i] = (bid_qty - ask_qty) / total;
        } else {
            result.microprice[i] = (ask + bid) * 0.5;
            result.imbalance[i] = 0.0;
        }
    }
    return result;
}

}  // namespace rflux::execution
