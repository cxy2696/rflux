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

    // Keep the reference history sorted instead of re-sorting it at every
    // observation.  The labels are identical to a full sort, but each step
    // costs one insertion/erase rather than a window sort.
    std::deque<double> history;
    std::vector<double> sorted_history;
    sorted_history.reserve(lookback + 1);
    for (std::size_t t = 0; t < n; ++t) {
        const double value = result.volatility[t];
        if (!std::isfinite(value)) continue;
        if (history.size() >= kMinReferenceObservations) {
            const double lower = quantile_sorted(sorted_history, low_q);
            const double upper = quantile_sorted(sorted_history, high_q);
            if (value > upper) {
                result.labels[t] = 2;
            } else if (value < lower) {
                result.labels[t] = 0;
            } else {
                result.labels[t] = 1;
            }
        }
        sorted_history.insert(
            std::lower_bound(sorted_history.begin(), sorted_history.end(),
                             value),
            value);
        history.push_back(value);
        if (history.size() > lookback) {
            const double expired = history.front();
            history.pop_front();
            const auto position = std::lower_bound(
                sorted_history.begin(), sorted_history.end(), expired);
            sorted_history.erase(position);
        }
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

StickyRegimeResult sticky_regime(const double* probabilities,
                                 std::size_t n_obs, std::size_t n_states,
                                 std::size_t min_duration,
                                 double entry_threshold,
                                 double exit_threshold) {
    require_pointer(probabilities, "probabilities must not be NULL");
    require(n_states > 0, "n_states must be positive");
    require(min_duration >= 1, "min_duration must be positive");
    require(std::isfinite(entry_threshold) &&
                std::isfinite(exit_threshold) && exit_threshold > 0.0 &&
                exit_threshold < entry_threshold && entry_threshold <= 1.0,
            "require 0 < exit_threshold < entry_threshold <= 1");
    StickyRegimeResult result;
    if (n_obs == 0) return result;
    require(probabilities != nullptr, "probabilities must not be NULL");
    std::vector<std::int32_t> labels(n_obs, 0);
    std::size_t current = 0;
    for (std::size_t state = 1; state < n_states; ++state) {
        if (probabilities[state] > probabilities[current]) current = state;
    }
    labels[0] = static_cast<std::int32_t>(current);
    std::size_t last_switch = 0;
    for (std::size_t row = 1; row < n_obs; ++row) {
        std::size_t challenger = 0;
        const double* probabilities_row = probabilities + row * n_states;
        for (std::size_t state = 1; state < n_states; ++state) {
            if (probabilities_row[state] > probabilities_row[challenger]) {
                challenger = state;
            }
        }
        if (challenger != current &&
            probabilities_row[challenger] >= entry_threshold &&
            probabilities_row[current] <= exit_threshold &&
            row - last_switch >= min_duration) {
            current = challenger;
            last_switch = row;
            result.switches.push_back(static_cast<std::int32_t>(row));
        }
        labels[row] = static_cast<std::int32_t>(current);
    }
    result.labels = std::move(labels);
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

double conditional_mean(const double* state_probs, const double* means,
                        std::size_t n_states, std::size_t n_features,
                        std::size_t feature) {
    require_pointer(state_probs, "state_probs must not be NULL");
    require_pointer(means, "means must not be NULL");
    require(n_states > 0, "n_states must be positive");
    require(n_features > 0, "n_features must be positive");
    require(feature < n_features, "feature is out of range");
    long double total = 0.0L;
    for (std::size_t state = 0; state < n_states; ++state) {
        require(std::isfinite(state_probs[state]) &&
                    state_probs[state] >= 0.0,
                "state_probs must be finite and non-negative");
        total += static_cast<long double>(state_probs[state]);
    }
    require(total > 0.0L, "state_probs must have positive mass");
    long double value = 0.0L;
    for (std::size_t state = 0; state < n_states; ++state) {
        value += static_cast<long double>(state_probs[state]) *
                 static_cast<long double>(
                     means[state * n_features + feature]);
    }
    return static_cast<double>(value / total);
}

}  // namespace rflux::regimelab
