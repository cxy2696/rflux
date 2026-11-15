#include "rflux/regimelab/regime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

#include "internal/common.hpp"
#include "rflux/qflux/window.hpp"

namespace rflux::regimelab {
namespace {

using rflux::detail::quantile_sorted;
using rflux::detail::require;
using rflux::detail::require_pointer;

constexpr std::size_t kMinReferenceObservations = 20;

}  // namespace

VolatilityRegimeResult volatility_regime(const double* returns, std::size_t n,
                                         std::size_t vol_window,
                                         std::size_t lookback, double low_q,
                                         double high_q) {
    require_pointer(returns, "returns must not be NULL");
    require(n > 0, "n must be positive");
    require(vol_window >= 2 && vol_window <= n,
            "vol_window must be in [2, n]");
    require(lookback >= kMinReferenceObservations,
            "lookback must be at least 20");
    require(std::isfinite(low_q) && std::isfinite(high_q) &&
                low_q >= 0.0 && low_q < high_q && high_q <= 1.0,
            "quantiles must satisfy 0 <= low_q < high_q <= 1");

    VolatilityRegimeResult result;
    result.volatility =
        rflux::qflux::rolling_std(returns, n, vol_window, 1);
    result.labels.assign(n, -1);

    std::deque<double> history;
    for (std::size_t t = 0; t < n; ++t) {
        if (!std::isfinite(result.volatility[t])) continue;
        if (history.size() >= kMinReferenceObservations) {
            std::vector<double> reference(history.begin(), history.end());
            std::sort(reference.begin(), reference.end());
            const double lower = quantile_sorted(reference, low_q);
            const double upper = quantile_sorted(reference, high_q);
            if (result.volatility[t] > upper) {
                result.labels[t] = 2;
            } else if (result.volatility[t] < lower) {
                result.labels[t] = 0;
            } else {
                result.labels[t] = 1;
            }
        }
        history.push_back(result.volatility[t]);
        if (history.size() > lookback) history.pop_front();
    }
    return result;
}

TrendRegimeResult trend_regime(const double* prices, std::size_t n,
                               std::size_t fast_window, std::size_t slow_window,
                               double threshold) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(fast_window > 0 && slow_window > 0,
            "moving-average windows must be positive");
    require(fast_window < slow_window,
            "fast_window must be smaller than slow_window");
    require(slow_window <= n, "slow_window must not exceed n");
    require(std::isfinite(threshold) && threshold >= 0.0,
            "threshold must be finite and non-negative");

    TrendRegimeResult result;
    result.fast_ma = rflux::qflux::rolling_mean(prices, n, fast_window);
    result.slow_ma = rflux::qflux::rolling_mean(prices, n, slow_window);
    result.labels.assign(n, -1);

    for (std::size_t t = 0; t < n; ++t) {
        if (!std::isfinite(result.fast_ma[t]) ||
            !std::isfinite(result.slow_ma[t])) {
            continue;
        }
        double spread = 0.0;
        if (result.slow_ma[t] != 0.0) {
            spread = (result.fast_ma[t] - result.slow_ma[t]) /
                     std::abs(result.slow_ma[t]);
        } else if (result.fast_ma[t] != 0.0) {
            spread = result.fast_ma[t] > 0.0
                         ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity();
        }
        if (spread > threshold) {
            result.labels[t] = 2;
        } else if (spread < -threshold) {
            result.labels[t] = 0;
        } else {
            result.labels[t] = 1;
        }
    }
    return result;
}

std::vector<std::int32_t> composite_regime(
    const std::int32_t* volatility_labels, const std::int32_t* trend_labels,
    std::size_t n) {
    require_pointer(volatility_labels, "volatility_labels must not be NULL");
    require_pointer(trend_labels, "trend_labels must not be NULL");
    require(n > 0, "n must be positive");
    std::vector<std::int32_t> out(n, -1);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t volatility = volatility_labels[i];
        const std::int32_t trend = trend_labels[i];
        require(volatility >= -1 && volatility <= 2,
                "volatility labels must be in [-1, 2]");
        require(trend >= -1 && trend <= 2,
                "trend labels must be in [-1, 2]");
        if (volatility < 0 || trend < 0) continue;
        out[i] = std::max(volatility, 2 - trend);
    }
    return out;
}

}  // namespace rflux::regimelab
