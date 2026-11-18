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

double rsi_from_averages(double average_gain, double average_loss) {
    if (average_loss == 0.0) {
        return average_gain == 0.0 ? 50.0 : 100.0;
    }
    if (average_gain == 0.0) return 0.0;
    const double relative_strength = average_gain / average_loss;
    return 100.0 - 100.0 / (1.0 + relative_strength);
}

}  // namespace

void returns_into(const double* prices, std::size_t n, bool log_returns,
                  double* out) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require_pointer(out, "out must not be NULL");
    out[0] = kNaN;
    for (std::size_t i = 1; i < n; ++i) {
        const double previous = prices[i - 1];
        const double current = prices[i];
        double value = kNaN;
        if (std::isfinite(previous) && std::isfinite(current)) {
            if (log_returns) {
                if (previous > 0.0 && current > 0.0) {
                    value = std::log(current / previous);
                }
            } else if (previous != 0.0) {
                value = current / previous - 1.0;
            }
        }
        out[i] = value;
    }
}

std::vector<double> returns(const double* prices, std::size_t n,
                            bool log_returns) {
    std::vector<double> out(n, kNaN);
    returns_into(prices, n, log_returns, out.data());
    return out;
}

void sma_into(const double* values, std::size_t n, std::size_t window,
              double* out) {
    rolling_mean_into(values, n, window, out);
}

std::vector<double> sma(const double* values, std::size_t n,
                        std::size_t window) {
    std::vector<double> out(n, kNaN);
    sma_into(values, n, window, out.data());
    return out;
}

void ema_into(const double* values, std::size_t n, std::size_t span,
              double* out) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(span > 0, "span must be positive");
    require_pointer(out, "out must not be NULL");
    const double alpha = 2.0 / (static_cast<double>(span) + 1.0);
    bool initialized = false;
    double state = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            initialized = false;
            out[i] = kNaN;
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
}

std::vector<double> ema(const double* values, std::size_t n,
                        std::size_t span) {
    std::vector<double> out(n, kNaN);
    ema_into(values, n, span, out.data());
    return out;
}

void rsi_into(const double* prices, std::size_t n, std::size_t period,
              double* out) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(period > 0, "period must be positive");
    require(n > period, "n must be greater than period");
    require_pointer(out, "out must not be NULL");

    std::fill(out, out + n, kNaN);
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
}

std::vector<double> rsi(const double* prices, std::size_t n,
                        std::size_t period) {
    std::vector<double> out(n, kNaN);
    rsi_into(prices, n, period, out.data());
    return out;
}

void macd_into(const double* prices, std::size_t n, std::size_t fast,
               std::size_t slow, std::size_t signal, double* macd_out,
               double* signal_out, double* histogram_out) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(fast > 0 && slow > 0 && signal > 0, "MACD spans must be positive");
    require(fast < slow, "MACD fast span must be smaller than slow span");
    require_pointer(macd_out, "macd output must not be NULL");
    require_pointer(signal_out, "signal output must not be NULL");
    require_pointer(histogram_out, "histogram output must not be NULL");

    std::vector<double> fast_ema(n, kNaN);
    std::vector<double> slow_ema(n, kNaN);
    ema_into(prices, n, fast, fast_ema.data());
    ema_into(prices, n, slow, slow_ema.data());
    for (std::size_t i = 0; i < n; ++i) {
        macd_out[i] = (std::isfinite(fast_ema[i]) && std::isfinite(slow_ema[i]))
                          ? fast_ema[i] - slow_ema[i]
                          : kNaN;
    }
    ema_into(macd_out, n, signal, signal_out);
    for (std::size_t i = 0; i < n; ++i) {
        histogram_out[i] =
            (std::isfinite(macd_out[i]) && std::isfinite(signal_out[i]))
                ? macd_out[i] - signal_out[i]
                : kNaN;
    }
}

MacdResult macd(const double* prices, std::size_t n, std::size_t fast,
                std::size_t slow, std::size_t signal) {
    MacdResult result;
    result.macd.assign(n, kNaN);
    result.signal.assign(n, kNaN);
    result.histogram.assign(n, kNaN);
    macd_into(prices, n, fast, slow, signal, result.macd.data(),
              result.signal.data(), result.histogram.data());
    return result;
}

void bollinger_into(const double* prices, std::size_t n, std::size_t window,
                    double n_std, double* middle, double* upper, double* lower) {
    require_pointer(prices, "prices must not be NULL");
    require(n > 0, "n must be positive");
    require(window > 0 && window <= n, "window must be in [1, n]");
    require(std::isfinite(n_std) && n_std >= 0.0,
            "n_std must be finite and non-negative");
    require_pointer(middle, "middle output must not be NULL");
    require_pointer(upper, "upper output must not be NULL");
    require_pointer(lower, "lower output must not be NULL");

    rolling_mean_into(prices, n, window, middle);
    /* The upper band temporarily holds the population deviation so this call
     * needs no scratch vector; the band is then formed in place. */
    rolling_std_into(prices, n, window, 0, upper);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(middle[i]) || !std::isfinite(upper[i])) {
            upper[i] = kNaN;
            lower[i] = kNaN;
            continue;
        }
        const double width = n_std * upper[i];
        lower[i] = middle[i] - width;
        upper[i] = middle[i] + width;
    }
}

BollingerBands bollinger(const double* prices, std::size_t n,
                         std::size_t window, double n_std) {
    BollingerBands result;
    result.middle.assign(n, kNaN);
    result.upper.assign(n, kNaN);
    result.lower.assign(n, kNaN);
    bollinger_into(prices, n, window, n_std, result.middle.data(),
                   result.upper.data(), result.lower.data());
    return result;
}

void atr_into(const double* high, const double* low, const double* close,
              std::size_t n, std::size_t period, double* out) {
    require_pointer(high, "high must not be NULL");
    require_pointer(low, "low must not be NULL");
    require_pointer(close, "close must not be NULL");
    require(n > 0, "n must be positive");
    require(period > 0, "period must be positive");
    require(n >= period, "n must be at least period");
    require_pointer(out, "out must not be NULL");

    std::fill(out, out + n, kNaN);
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
            current_atr = static_cast<double>(true_range_sum) / period_d;
            seeded = true;
        } else {
            current_atr =
                (current_atr * (period_d - 1.0) + true_range) / period_d;
        }
        out[i] = current_atr;
    }
}

std::vector<double> atr(const double* high, const double* low,
                        const double* close, std::size_t n,
                        std::size_t period) {
    std::vector<double> out(n, kNaN);
    atr_into(high, low, close, n, period, out.data());
    return out;
}

void drawdown_into(const double* equity, std::size_t n, double* out) {
    require_pointer(equity, "equity must not be NULL");
    require(n > 0, "n must be positive");
    require_pointer(out, "out must not be NULL");
    std::fill(out, out + n, kNaN);
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
}

std::vector<double> drawdown(const double* equity, std::size_t n) {
    std::vector<double> out(n, kNaN);
    drawdown_into(equity, n, out.data());
    return out;
}

}  // namespace rflux::qflux
