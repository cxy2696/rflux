#pragma once

/*
 * Cross-module data contracts.
 *
 * rflux does not expose Eigen types across its public boundary:
 * the core uses caller-owned contiguous buffers and standard containers, which
 * keeps the C++ interface usable from the C ABI, pybind11 and other language
 * runtimes without adopting Eigen's ABI/alignment requirements.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rflux/core/algorithm.hpp"

namespace rflux::core {

/* Owning market snapshot for research/simulation code. */
struct MarketData {
    std::int64_t timestamp_ns = 0;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    [[nodiscard]] std::size_t n_assets() const noexcept {
        return close.size();
    }
};

/* Non-owning view used by hot-path algorithms and language bindings. */
struct MarketSnapshotView {
    std::int64_t timestamp_ns = 0;
    std::size_t n_assets = 0;
    const double* open = nullptr;
    const double* high = nullptr;
    const double* low = nullptr;
    const double* close = nullptr;
    const double* volume = nullptr;
};

struct OrderBook {
    std::string symbol;
    std::int64_t timestamp_ns = 0;
    std::vector<double> bid_prices;
    std::vector<double> bid_volumes;
    std::vector<double> ask_prices;
    std::vector<double> ask_volumes;
};

struct OrderBookView {
    std::string symbol;
    std::int64_t timestamp_ns = 0;
    std::size_t n_levels = 0;
    const double* bid_prices = nullptr;
    const double* bid_volumes = nullptr;
    const double* ask_prices = nullptr;
    const double* ask_volumes = nullptr;
};

struct MatrixView {
    const double* data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
};

struct PredictionResult {
    std::vector<double> expected_returns;
    std::vector<double> confidence;
};

struct RiskConstraints {
    double max_position_limit = 1.0;   /* fraction of portfolio value */
    double max_drawdown_limit = 0.20;  /* positive loss fraction */
    double turnover_threshold = 0.10;  /* rebalance only above this drift */
};

enum class Side { buy, sell };

struct Order {
    std::string symbol;
    std::int64_t quantity = 0; /* signed not allowed; side carries direction */
    double price = 0.0;
    Side side = Side::buy;
};

struct ExecutionRequest {
    std::vector<std::string> symbols;
    std::vector<double> target_weights;
    std::vector<double> current_weights;
    double portfolio_value = 0.0;
    std::int64_t timestamp_ns = 0;
};

namespace detail {

inline void require_snapshot(const MarketSnapshotView& view) {
    rflux::core::detail::require(
        view.n_assets > 0, "snapshot must contain at least one asset");
    rflux::core::detail::require(
        view.open != nullptr && view.high != nullptr && view.low != nullptr &&
            view.close != nullptr && view.volume != nullptr,
        "snapshot price/volume buffers must not be null");
}

inline void require_order_book(const OrderBookView& view) {
    rflux::core::detail::require(
        view.n_levels > 0, "order book must contain at least one level");
    rflux::core::detail::require(
        view.bid_prices != nullptr && view.bid_volumes != nullptr &&
            view.ask_prices != nullptr && view.ask_volumes != nullptr,
        "order book buffers must not be null");
}

inline void require_matrix(const MatrixView& view) {
    rflux::core::detail::require(view.data != nullptr,
                                  "matrix data must not be null");
    rflux::core::detail::require(
        view.rows > 0 && view.columns > 0,
        "matrix must have positive dimensions");
}

}  // namespace detail

}  // namespace rflux::core
