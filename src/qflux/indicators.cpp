#include "rflux/qflux/indicators.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "internal/common.hpp"
#include "rflux/qflux/window.hpp"

namespace rflux::qflux {
namespace {

using rflux::detail::kNaN;
using rflux::detail::require;
using rflux::detail::require_pointer;

std::vector<double> ema_series(const double* values, std::size_t n,
                               std::size_t span) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(span > 0, "span must be positive");
    const double alpha = 2.0 / (static_cast<double>(span) + 1.0);
    std::vector<double> out(n, kNaN);
    bool initialized = false;
    double state = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            initialized = false;
            continue;
        }
        if (!initialized) {
            state = values[i];
            initialized = true;
        } else {
            state = alpha * values[i] + (1.0 - alpha) * state;
        }
        out[i] = state;
    }
    return out;
}

double rsi_from_averages(double average_gain, double average_loss) {
    if (average_loss == 0.0) {
        return average_gain == 0.0 ? 50.0 : 100.0;
    }
    if (average_gain == 0.0) return 0.0;
    const double relative_strength = average_gain / average_loss;
    return 100.0 - 100.0 / (1.0 + relative_strength);
}

}  // namespace

std::vector<double> returns(const double* prices, std::size_t n,
                            bool log_returns) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    std::vector<double> out(n, kNaN);
    for (std::size_t i = 1; i < n; ++i) {
        const double previous = prices[i - 1];
        const double current = prices[i];
        if (!std::isfinite(previous) || !std::isfinite(current)) continue;
        if (log_returns) {
            if (previous > 0.0 && current > 0.0) {
                out[i] = std::log(current / previous);
            }
        } else if (previous != 0.0) {
            out[i] = current / previous - 1.0;
        }
    }
    return out;
}

std::vector<double> sma(const double* values, std::size_t n,
                        std::size_t window) {
    return rolling_mean(values, n, window);
}

std::vector<double> ema(const double* values, std::size_t n,
                        std::size_t span) {
    return ema_series(values, n, span);
}

std::vector<double> rsi(const double* prices, std::size_t n,
                        std::size_t period) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(period > 0, "period must be positive");
    require(n > period, "n must be greater than period");

    std::vector<double> out(n, kNaN);
    std::size_t change_count = 0;
    long double gain_sum = 0.0L;
    long double loss_sum = 0.0L;
    double average_gain = 0.0;
    double average_loss = 0.0;
    bool seeded = false;
    const double period_d = static_cast<double>(period);
    const double smoothing = 1.0 / period_d;

    for (std::size_t i = 1; i < n; ++i) {
        if (!std::isfinite(prices[i]) || !std::isfinite(prices[i - 1])) {
            seeded = false;
            change_count = 0;
            gain_sum = 0.0L;
            loss_sum = 0.0L;
            continue;
        }

        const double delta = prices[i] - prices[i - 1];
        const double gain = std::max(delta, 0.0);
        const double loss = std::max(-delta, 0.0);

        if (!seeded) {
            gain_sum += static_cast<long double>(gain);
            loss_sum += static_cast<long double>(loss);
            ++change_count;
            if (change_count < period) continue;
            average_gain = static_cast<double>(gain_sum) / period_d;
            average_loss = static_cast<double>(loss_sum) / period_d;
            seeded = true;
        } else {
            average_gain = smoothing * gain + (1.0 - smoothing) * average_gain;
            average_loss = smoothing * loss + (1.0 - smoothing) * average_loss;
        }
        out[i] = rsi_from_averages(average_gain, average_loss);
    }
    return out;
}

MacdResult macd(const double* prices, std::size_t n, std::size_t fast,
                std::size_t slow, std::size_t signal) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(fast > 0 && slow > 0 && signal > 0,
            "MACD spans must be positive");
    require(fast < slow, "MACD fast span must be smaller than slow span");

    MacdResult result;
    result.macd.assign(n, kNaN);
    result.signal.assign(n, kNaN);
    result.histogram.assign(n, kNaN);

    const std::vector<double> fast_ema = ema_series(prices, n, fast);
    const std::vector<double> slow_ema = ema_series(prices, n, slow);
    std::vector<double> macd_line(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(fast_ema[i]) && std::isfinite(slow_ema[i])) {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }
    const std::vector<double> signal_line =
        ema_series(macd_line.data(), n, signal);

    result.macd = macd_line;
    result.signal = signal_line;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(macd_line[i]) && std::isfinite(signal_line[i])) {
            result.histogram[i] = macd_line[i] - signal_line[i];
        }
    }
    return result;
}

BollingerBands bollinger(const double* prices, std::size_t n,
                         std::size_t window, double n_std) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(window > 0 && window <= n, "window must be in [1, n]");
    require(std::isfinite(n_std) && n_std >= 0.0,
            "n_std must be finite and non-negative");

    BollingerBands result;
    result.middle = rolling_mean(prices, n, window);
    const std::vector<double> deviation = rolling_std(prices, n, window, 0);
    result.upper.assign(n, kNaN);
    result.lower.assign(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(result.middle[i]) &&
            std::isfinite(deviation[i])) {
            const double width = n_std * deviation[i];
            result.upper[i] = result.middle[i] + width;
            result.lower[i] = result.middle[i] - width;
        }
    }
    return result;
}

std::vector<double> atr(const double* high, const double* low,
                        const double* close, std::size_t n,
                        std::size_t period) {
    require_pointer(high, "high must not be NULL");
    require_pointer(low, "low must not be NULL");
    require_pointer(close, "close must not be NULL");
    require(n > 0, "n must be positive");
    require(period > 0, "period must be positive");
    require(n >= period, "n must be at least period");

    std::vector<double> out(n, kNaN);
    std::size_t count = 0;
    long double true_range_sum = 0.0L;
    double current_atr = 0.0;
    bool seeded = false;
    const double period_d = static_cast<double>(period);

    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(high[i]) || !std::isfinite(low[i])) {
            count = 0;
            true_range_sum = 0.0L;
            seeded = false;
            continue;
        }
        double true_range = high[i] - low[i];
        if (i > 0 && std::isfinite(close[i - 1])) {
            true_range = std::max(true_range, std::abs(high[i] - close[i - 1]));
            true_range = std::max(true_range, std::abs(low[i] - close[i - 1]));
        }
        if (!std::isfinite(true_range)) {
            count = 0;
            true_range_sum = 0.0L;
            seeded = false;
            continue;
        }

        if (!seeded) {
            true_range_sum += static_cast<long double>(true_range);
            ++count;
            if (count < period) continue;
            current_atr =
                static_cast<double>(true_range_sum) / period_d;
            seeded = true;
        } else {
            current_atr =
                (current_atr * (period_d - 1.0) + true_range) / period_d;
        }
        out[i] = current_atr;
    }
    return out;
}

std::vector<double> drawdown(const double* equity, std::size_t n) {
    require_pointer(equity, "equity must not be NULL");
    require(n > 0, "n must be positive");
    std::vector<double> out(n, kNaN);
    bool has_peak = false;
    double peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(equity[i])) continue;
        if (!has_peak) {
            peak = equity[i];
            has_peak = true;
        } else {
            peak = std::max(peak, equity[i]);
        }
        if (peak != 0.0) out[i] = equity[i] / peak - 1.0;
    }
    return out;
}

}  // namespace rflux::qflux
