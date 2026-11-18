#include "rflux/portfoliolab/advice.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/qflux/indicators.hpp"

namespace rflux::portfoliolab {
namespace {

using rflux::detail::kNaN;

std::size_t finite_count(const double* values, std::size_t n) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) ++count;
    }
    return count;
}

double last_finite_value(const double* values, std::size_t n) {
    for (std::size_t index = n; index > 0; --index) {
        if (std::isfinite(values[index - 1])) return values[index - 1];
    }
    return kNaN;
}

/* Return over exactly `window` calendar positions, matching the research
 * definition: the base is close[n - window - 1], not the previous finite
 * value. */
double trailing_return(const double* close, std::size_t n,
                       std::size_t window) {
    if (n <= window) return kNaN;
    const double current = close[n - 1];
    const double base = close[n - window - 1];
    if (!std::isfinite(current) || !std::isfinite(base) || base == 0.0) {
        return kNaN;
    }
    return current / base - 1.0;
}

std::vector<double> zscore(const std::vector<double>& values) {
    std::vector<double> result(values.size(), 0.0);
    std::size_t count = 0;
    long double sum = 0.0L;
    for (double value : values) {
        if (!std::isfinite(value)) continue;
        sum += static_cast<long double>(value);
        ++count;
    }
    if (count < 2) return result;
    const long double mean = sum / static_cast<long double>(count);
    long double variance = 0.0L;
    for (double value : values) {
        if (!std::isfinite(value)) continue;
        const long double diff = static_cast<long double>(value) - mean;
        variance += diff * diff;
    }
    variance /= static_cast<long double>(count);
    const double scale = std::sqrt(static_cast<double>(variance));
    if (!std::isfinite(scale) || scale <= 0.0) return result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) continue;
        const double value =
            (values[i] - static_cast<double>(mean)) / scale;
        result[i] = std::clamp(value, -3.0, 3.0);
    }
    return result;
}

bool share_count(double value, std::int64_t* result) {
    if (!std::isfinite(value)) return false;
    const long double floored =
        std::floor(static_cast<long double>(value));
    const long double maximum = static_cast<long double>(
        std::numeric_limits<std::int64_t>::max());
    if (floored < 0.0L || floored > maximum) return false;
    *result = static_cast<std::int64_t>(floored);
    return true;
}

std::vector<double> cap_weights(std::vector<double> weights,
                                double max_weight) {
    bool has_nonfinite = false;
    for (double& value : weights) {
        if (!std::isfinite(value)) {
            has_nonfinite = true;
            value = kNaN;
        } else {
            value = std::max(value, 0.0);
        }
    }
    long double total = 0.0L;
    if (!has_nonfinite) {
        for (double value : weights) {
            total += static_cast<long double>(value);
        }
    }
    if (has_nonfinite || !(total > 0.0L)) {
        std::fill(weights.begin(), weights.end(),
                  1.0 / static_cast<double>(weights.size()));
    } else {
        for (double& value : weights) {
            value = static_cast<double>(
                static_cast<long double>(value) / total);
        }
    }
    const std::size_t n = weights.size();
    for (std::size_t iteration = 0; iteration < n + 1; ++iteration) {
        std::vector<bool> over(n, false);
        bool any_over = false;
        for (std::size_t index = 0; index < n; ++index) {
            over[index] = weights[index] > max_weight + 1e-15;
            any_over = any_over || over[index];
        }
        if (!any_over) break;
        long double excess = 0.0L;
        for (std::size_t index = 0; index < n; ++index) {
            if (over[index]) {
                excess += static_cast<long double>(weights[index] - max_weight);
                weights[index] = max_weight;
            }
        }
        std::size_t free_count = 0;
        long double free_total = 0.0L;
        for (std::size_t index = 0; index < n; ++index) {
            if (over[index]) continue;
            ++free_count;
            free_total += static_cast<long double>(weights[index]);
        }
        if (free_count == 0) break;
        if (free_total <= 0.0L) {
            const double share =
                static_cast<double>(excess) /
                static_cast<double>(free_count);
            for (std::size_t index = 0; index < n; ++index) {
                if (!over[index]) weights[index] += share;
            }
        } else {
            for (std::size_t index = 0; index < n; ++index) {
                if (over[index]) continue;
                weights[index] += static_cast<double>(
                    excess * static_cast<long double>(weights[index]) /
                    free_total);
            }
        }
    }
    return weights;
}

}  // namespace

AdviceSignals advice_signals(const double* prices, const double* returns,
                             std::size_t n_obs, std::size_t n_assets) {
    rflux::detail::require_pointer(prices, "prices must not be NULL");
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require(n_obs >= 2, "n_obs must be at least two");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::checked_mul(n_obs, n_assets, "panel dimensions overflow");

    AdviceSignals result;
    result.rows.resize(n_assets);
    for (AdviceSignalRow& row : result.rows) {
        row.last_price = kNaN;
        row.momentum_20 = kNaN;
        row.momentum_60 = kNaN;
        row.trend = 0.0;
        row.rsi_14 = kNaN;
        row.daily_sigma = kNaN;
        row.reference_stop = kNaN;
    }
    std::vector<double> momentum_20(n_assets, kNaN);
    std::vector<double> momentum_60(n_assets, kNaN);
    std::vector<double> trend(n_assets, 0.0);
    std::vector<double> rsi_values(n_assets, kNaN);

    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        std::vector<double> close(n_obs, kNaN);
        for (std::size_t row = 0; row < n_obs; ++row) {
            close[row] = prices[row * n_assets + asset];
        }
        AdviceSignalRow& signal = result.rows[asset];
        signal.trend = 0.0;
        const std::size_t valid = finite_count(close.data(), close.size());
        signal.last_price = last_finite_value(close.data(), close.size());
        if (valid < 2) continue;

        const double last = signal.last_price;
        momentum_20[asset] = trailing_return(close.data(), close.size(), 20);
        momentum_60[asset] = trailing_return(close.data(), close.size(), 60);

        const std::size_t window = std::min<std::size_t>(20, n_obs);
        const std::vector<double> sma20 =
            rflux::qflux::sma(close.data(), n_obs, window);
        const double sma20_last = sma20.empty() ? kNaN : sma20.back();
        double trend_value = 0.0;
        if (std::isfinite(sma20_last) && sma20_last > 0.0) {
            trend_value += last > sma20_last ? 1.0 : -1.0;
        }
        const std::size_t long_window = std::min<std::size_t>(60, n_obs);
        const std::vector<double> sma60 =
            rflux::qflux::sma(close.data(), n_obs, long_window);
        const double sma60_last = sma60.empty() ? kNaN : sma60.back();
        if (std::isfinite(sma60_last) && sma60_last > 0.0) {
            trend_value += last > sma60_last ? 1.0 : -1.0;
        }
        trend[asset] = 0.5 * trend_value;

        try {
            const std::vector<double> rsi = rflux::qflux::rsi(
                close.data(), n_obs, 14);
            if (!rsi.empty() && std::isfinite(rsi.back())) {
                rsi_values[asset] = rsi.back();
            }
        } catch (const std::exception&) {
            rsi_values[asset] = kNaN;
        }

        const std::size_t start = n_obs > 20 ? n_obs - 20 : 0;
        long double sum = 0.0L;
        std::size_t count = 0;
        for (std::size_t row = start; row < n_obs; ++row) {
            const double value = returns[row * n_assets + asset];
            if (!std::isfinite(value)) continue;
            sum += static_cast<long double>(value);
            ++count;
        }
        if (count >= 2) {
            const long double mean = sum / static_cast<long double>(count);
            long double variance = 0.0L;
            for (std::size_t row = start; row < n_obs; ++row) {
                const double value = returns[row * n_assets + asset];
                if (!std::isfinite(value)) continue;
                const long double diff =
                    static_cast<long double>(value) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<long double>(count - 1);
            const double sigma = std::sqrt(static_cast<double>(variance));
            if (std::isfinite(sigma)) signal.daily_sigma = sigma;
        }

        signal.momentum_20 = momentum_20[asset];
        signal.momentum_60 = momentum_60[asset];
        signal.trend = trend[asset];
        signal.rsi_14 = rsi_values[asset];
        if (std::isfinite(signal.daily_sigma) && signal.daily_sigma > 0.0) {
            signal.reference_stop = last * (1.0 - 2.0 * signal.daily_sigma);
        }
    }

    const std::vector<double> z20 = zscore(momentum_20);
    const std::vector<double> z60 = zscore(momentum_60);
    result.scores.assign(n_assets, 0.0);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        double score = 0.5 * z20[asset] + 0.3 * z60[asset] +
                       0.2 * trend[asset];
        if (std::isfinite(rsi_values[asset])) {
            if (rsi_values[asset] >= 75.0) score -= 0.20;
            else if (rsi_values[asset] <= 25.0) score += 0.10;
        }
        result.scores[asset] = std::clamp(score, -1.0, 1.0);
    }
    return result;
}

AdviceAllocation advice_allocation(
    const double* covariance, const double* scores, std::size_t n_assets,
    const double* current_weights, bool has_current_weights,
    const double* market_returns, std::size_t n_obs, double periods_per_year,
    double max_weight, double target_volatility, double regime_scale) {
    rflux::detail::require_pointer(covariance, "covariance must not be NULL");
    rflux::detail::require_pointer(scores, "scores must not be NULL");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::checked_mul(n_assets, n_assets,
                               "covariance dimensions overflow");
    rflux::detail::require(std::isfinite(periods_per_year) &&
                               periods_per_year > 0.0,
                           "periods_per_year must be positive");
    rflux::detail::require(std::isfinite(max_weight) && max_weight > 0.0 &&
                               max_weight <= 1.0,
                           "max_weight must be in (0, 1]");
    rflux::detail::require(max_weight * static_cast<double>(n_assets) >=
                               1.0 - 1e-8,
                           "max_weight is too small for the number of assets");
    rflux::detail::require(std::isfinite(target_volatility) &&
                               target_volatility > 0.0,
                           "target_volatility must be positive");

    AdviceAllocation result;
    std::vector<double> base;
    if (n_assets == 1) {
        base.assign(1, 1.0);
    } else {
        try {
            base = HRPAllocator().weights(covariance, n_assets);
        } catch (const std::exception&) {
            base.clear();
        }
        if (base.size() != n_assets) {
            std::vector<double> inverse(n_assets, kNaN);
            bool has_inverse = false;
            for (std::size_t asset = 0; asset < n_assets; ++asset) {
                const double variance = covariance[asset * n_assets + asset];
                if (std::isfinite(variance) && variance > 0.0) {
                    inverse[asset] = 1.0 / variance;
                    has_inverse = true;
                }
            }
            if (!has_inverse) {
                base.assign(n_assets, 1.0 / static_cast<double>(n_assets));
            } else {
                base.assign(n_assets, 0.0);
                long double total = 0.0L;
                for (double value : inverse) {
                    if (std::isfinite(value)) {
                        total += static_cast<long double>(value);
                    }
                }
                for (std::size_t asset = 0; asset < n_assets; ++asset) {
                    if (std::isfinite(inverse[asset])) {
                        base[asset] = static_cast<double>(
                            static_cast<long double>(inverse[asset]) / total);
                    }
                }
            }
        }
    }
    base = cap_weights(std::move(base), max_weight);

    std::vector<double> tilted(n_assets, 0.0);
    bool has_positive = false;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        tilted[asset] = base[asset] * (1.0 + 0.5 * scores[asset]);
        if (tilted[asset] > 0.0) has_positive = true;
    }
    if (!has_positive) tilted = base;
    tilted = cap_weights(std::move(tilted), max_weight);

    const std::size_t market_count =
        market_returns != nullptr ? finite_count(market_returns, n_obs) : 0;
    double market_volatility = kNaN;
    if (market_count > 1) {
        long double sum = 0.0L;
        for (std::size_t row = 0; row < n_obs; ++row) {
            if (std::isfinite(market_returns[row])) {
                sum += static_cast<long double>(market_returns[row]);
            }
        }
        const long double mean =
            sum / static_cast<long double>(market_count);
        long double variance = 0.0L;
        for (std::size_t row = 0; row < n_obs; ++row) {
            if (!std::isfinite(market_returns[row])) continue;
            const long double diff =
                static_cast<long double>(market_returns[row]) - mean;
            variance += diff * diff;
        }
        variance /= static_cast<long double>(market_count - 1);
        const double stddev =
            std::sqrt(static_cast<double>(variance));
        if (std::isfinite(stddev)) {
            market_volatility = stddev * std::sqrt(periods_per_year);
        }
    }

    double portfolio_volatility = kNaN;
    bool covariance_finite = true;
    for (std::size_t i = 0; i < n_assets * n_assets; ++i) {
        if (!std::isfinite(covariance[i])) {
            covariance_finite = false;
            break;
        }
    }
    bool tilted_finite = true;
    for (double value : tilted) {
        if (!std::isfinite(value)) {
            tilted_finite = false;
            break;
        }
    }
    double tilted_sum = 0.0;
    for (double value : tilted) tilted_sum += value;
    if (covariance_finite && tilted_finite && tilted_sum > 0.0) {
        long double variance = 0.0L;
        for (std::size_t left = 0; left < n_assets; ++left) {
            for (std::size_t right = 0; right < n_assets; ++right) {
                variance +=
                    static_cast<long double>(tilted[left]) *
                    static_cast<long double>(
                        covariance[left * n_assets + right] *
                        periods_per_year) *
                    static_cast<long double>(tilted[right]);
            }
        }
        if (std::isfinite(static_cast<double>(variance)) &&
            variance > 0.0L) {
            portfolio_volatility =
                std::sqrt(static_cast<double>(variance));
        }
    }

    double volatility_scale = 1.0;
    if (std::isfinite(portfolio_volatility) &&
        portfolio_volatility > target_volatility) {
        volatility_scale = target_volatility / portfolio_volatility;
    } else if (!std::isfinite(portfolio_volatility)) {
        if (std::isfinite(market_volatility) &&
            market_volatility > target_volatility) {
            volatility_scale = target_volatility / market_volatility;
        }
    }

    const double exposure =
        std::clamp(volatility_scale * regime_scale, 0.0, 1.0);
    result.base_weights = base;
    result.target_weights.resize(n_assets, 0.0);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        result.target_weights[asset] = tilted[asset] * exposure;
    }
    result.current_weights.resize(n_assets, 0.0);
    if (has_current_weights && current_weights != nullptr) {
        std::copy(current_weights, current_weights + n_assets,
                  result.current_weights.begin());
    } else {
        std::fill(result.current_weights.begin(), result.current_weights.end(),
                  1.0 / static_cast<double>(n_assets));
    }
    result.deltas.resize(n_assets, 0.0);
    result.actions.assign(n_assets, 0);
    long double absolute_delta = 0.0L;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        const double delta = result.target_weights[asset] -
                             result.current_weights[asset];
        result.deltas[asset] = delta;
        absolute_delta += std::abs(static_cast<long double>(delta));
        if (delta > 0.02) result.actions[asset] = 1;
        else if (delta < -0.02) result.actions[asset] = -1;
    }
    long double target_sum = 0.0L;
    long double current_sum = 0.0L;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        target_sum += static_cast<long double>(result.target_weights[asset]);
        current_sum += static_cast<long double>(result.current_weights[asset]);
    }
    result.cash_weight = static_cast<double>(
        std::max(0.0L, 1.0L - target_sum));
    result.current_cash_weight = static_cast<double>(
        std::max(0.0L, 1.0L - current_sum));
    result.turnover = static_cast<double>(
        0.5L * (absolute_delta +
                std::abs(static_cast<long double>(result.cash_weight) -
                         static_cast<long double>(result.current_cash_weight))));
    result.market_annualized_volatility = market_volatility;
    result.portfolio_annualized_volatility = portfolio_volatility;
    result.volatility_scale = volatility_scale;
    result.exposure = exposure;
    return result;
}

AdviceRegime advice_regime(const double* filtered, const double* forecast,
                           const double* means, std::size_t n_states) {
    rflux::detail::require_pointer(filtered, "filtered must not be NULL");
    rflux::detail::require_pointer(forecast, "forecast must not be NULL");
    rflux::detail::require_pointer(means, "means must not be NULL");
    rflux::detail::require(n_states > 0, "n_states must be positive");
    AdviceRegime result;
    double latest = -std::numeric_limits<double>::infinity();
    for (std::size_t state = 0; state < n_states; ++state) {
        if (filtered[state] > latest) {
            latest = filtered[state];
            result.latest_state = static_cast<std::int32_t>(state);
        }
    }
    long double expected = 0.0L;
    long double stress = 0.0L;
    bool has_negative = false;
    for (std::size_t state = 0; state < n_states; ++state) {
        expected += static_cast<long double>(forecast[state]) *
                    static_cast<long double>(means[state]);
        if (means[state] < 0.0) {
            stress += static_cast<long double>(filtered[state]);
            has_negative = true;
        }
    }
    result.expected_next_return = static_cast<double>(expected);
    result.stress_probability =
        has_negative ? static_cast<double>(stress) : 0.0;
    if (result.expected_next_return < 0.0 && latest >= 0.5) {
        result.regime_scale = 0.5;
    } else if (result.stress_probability >= 0.65) {
        result.regime_scale = 0.7;
    } else {
        result.regime_scale = 1.0;
    }
    return result;
}

std::vector<AdviceSizingRow> advice_sizing(
    const double* target_weights, const double* last_prices,
    const double* reference_stops, std::size_t n_assets,
    double portfolio_value, bool has_portfolio_value, double risk_per_trade) {
    rflux::detail::require_pointer(target_weights,
                                   "target_weights must not be NULL");
    rflux::detail::require_pointer(last_prices,
                                   "last_prices must not be NULL");
    rflux::detail::require_pointer(reference_stops,
                                   "reference_stops must not be NULL");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    if (has_portfolio_value) {
        rflux::detail::require(std::isfinite(portfolio_value) &&
                                   portfolio_value > 0.0,
                               "portfolio_value must be positive");
        rflux::detail::require(std::isfinite(risk_per_trade) &&
                                   risk_per_trade > 0.0 &&
                                   risk_per_trade < 1.0,
                               "risk_per_trade must be in (0, 1)");
    }
    std::vector<AdviceSizingRow> result(n_assets);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        AdviceSizingRow& row = result[asset];
        const double last = last_prices[asset];
        const double stop = reference_stops[asset];
        if (std::isfinite(last) && last > 0.0 && std::isfinite(stop)) {
            row.stop_distance_pct = (last - stop) / last;
            row.has_stop_distance = true;
        }
        if (!has_portfolio_value) continue;
        const double notional =
            target_weights[asset] * portfolio_value;
        row.target_notional = notional;
        row.has_notional = true;
        if (std::isfinite(last) && last > 0.0) {
            row.has_target_shares =
                share_count(notional / last, &row.target_shares);
        }
        if (std::isfinite(last) && std::isfinite(stop) && last > stop &&
            stop > 0.0) {
            row.has_risk_limited_shares = share_count(
                portfolio_value * risk_per_trade / (last - stop),
                &row.risk_limited_shares);
        }
        if (row.has_target_shares) {
            row.recommended_shares = row.target_shares;
            if (row.has_risk_limited_shares) {
                row.recommended_shares =
                    std::min(row.target_shares, row.risk_limited_shares);
            }
            row.has_recommended_shares = true;
        }
    }
    return result;
}

}  // namespace rflux::portfoliolab
