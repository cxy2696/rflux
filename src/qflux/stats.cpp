#include "rflux/qflux/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "internal/common.hpp"

namespace rflux::qflux {
namespace {

using rflux::detail::finite_values;
using rflux::detail::kNaN;
using rflux::detail::quantile_sorted;
using rflux::detail::require;
using rflux::detail::require_pointer;

double finite_mean(const std::vector<double>& values) {
    long double sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value);
    return static_cast<double>(sum / static_cast<long double>(values.size()));
}

}  // namespace

Moments describe(const double* values, std::size_t n) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    const std::vector<double> data = finite_values(values, n);
    if (data.empty()) {
        throw InvalidArgument("describe requires at least one finite value");
    }

    Moments result;
    result.n_valid = data.size();
    const long double count = static_cast<long double>(data.size());
    const double mean = finite_mean(data);
    result.mean = mean;

    long double sum2 = 0.0L;
    long double sum3 = 0.0L;
    long double sum4 = 0.0L;
    for (double value : data) {
        const long double diff = static_cast<long double>(value - mean);
        const long double square = diff * diff;
        sum2 += square;
        sum3 += square * diff;
        sum4 += square * square;
    }

    result.variance = data.size() >= 2
                          ? static_cast<double>(sum2 / (count - 1.0L))
                          : kNaN;
    result.stddev = data.size() >= 2 ? std::sqrt(result.variance) : kNaN;
    result.skewness = kNaN;
    result.excess_kurtosis = kNaN;

    const long double m2 = sum2 / count;
    if (m2 > 0.0L && data.size() >= 3) {
        const long double m3 = sum3 / count;
        const long double g1 = m3 / std::pow(m2, 1.5L);
        const long double n = count;
        result.skewness = static_cast<double>(
            std::sqrt(n * (n - 1.0L)) / (n - 2.0L) * g1);
    }
    if (m2 > 0.0L && data.size() >= 4) {
        const long double m4 = sum4 / count;
        const long double g2 = m4 / (m2 * m2) - 3.0L;
        const long double n = count;
        result.excess_kurtosis = static_cast<double>(
            ((n - 1.0L) / ((n - 2.0L) * (n - 3.0L))) *
            ((n + 1.0L) * g2 + 6.0L));
    }
    return result;
}

double quantile(const double* values, std::size_t n, double q) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(q) && q >= 0.0 && q <= 1.0,
            "q must be in [0, 1]");
    std::vector<double> data = finite_values(values, n);
    if (data.empty()) {
        throw InvalidArgument("quantile requires at least one finite value");
    }
    std::sort(data.begin(), data.end());
    return quantile_sorted(data, q);
}

double pearson(const double* x, const double* y, std::size_t n) {
    require_pointer(x, "x must not be NULL");
    require_pointer(y, "y must not be NULL");
    require(n > 0, "n must be positive");

    std::size_t count = 0;
    long double sum_x = 0.0L;
    long double sum_y = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(x[i]) && std::isfinite(y[i])) {
            sum_x += static_cast<long double>(x[i]);
            sum_y += static_cast<long double>(y[i]);
            ++count;
        }
    }
    if (count < 2) return kNaN;

    const long double mean_x = sum_x / static_cast<long double>(count);
    const long double mean_y = sum_y / static_cast<long double>(count);
    long double covariance = 0.0L;
    long double variance_x = 0.0L;
    long double variance_y = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        const long double dx =
            static_cast<long double>(x[i]) - mean_x;
        const long double dy =
            static_cast<long double>(y[i]) - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if (variance_x <= 0.0L || variance_y <= 0.0L) return kNaN;
    double corr = static_cast<double>(covariance /
                                      std::sqrt(variance_x * variance_y));
    return std::max(-1.0, std::min(1.0, corr));
}

std::vector<double> correlation_matrix(const double* panel, std::size_t n_obs,
                                       std::size_t n_assets) {
    require_pointer(panel, "panel must not be NULL");
    require(n_obs > 0, "n_obs must be positive");
    require(n_assets > 0, "n_assets must be positive");
    detail::checked_mul(n_obs, n_assets, "n_obs*n_assets overflows size_t");
    const std::size_t entries = detail::checked_mul(
        n_assets, n_assets, "n_assets*n_assets overflows size_t");
    std::vector<double> matrix(entries, kNaN);

    for (std::size_t a = 0; a < n_assets; ++a) {
        for (std::size_t b = a; b < n_assets; ++b) {
            /* Pairwise-complete means, not full-sample means: missing values
             * make the two different, and correlation is not invariant to
             * that choice. */
            std::size_t count = 0;
            long double sum_a = 0.0L;
            long double sum_b = 0.0L;
            for (std::size_t r = 0; r < n_obs; ++r) {
                const double va = panel[r * n_assets + a];
                const double vb = panel[r * n_assets + b];
                if (!std::isfinite(va) || !std::isfinite(vb)) continue;
                sum_a += static_cast<long double>(va);
                sum_b += static_cast<long double>(vb);
                ++count;
            }
            if (count < 2) continue;
            const long double mean_a = sum_a / static_cast<long double>(count);
            const long double mean_b = sum_b / static_cast<long double>(count);
            long double covariance = 0.0L;
            long double variance_a = 0.0L;
            long double variance_b = 0.0L;
            for (std::size_t r = 0; r < n_obs; ++r) {
                const double va = panel[r * n_assets + a];
                const double vb = panel[r * n_assets + b];
                if (!std::isfinite(va) || !std::isfinite(vb)) continue;
                const long double da = static_cast<long double>(va) - mean_a;
                const long double db = static_cast<long double>(vb) - mean_b;
                covariance += da * db;
                variance_a += da * da;
                variance_b += db * db;
            }
            if (variance_a > 0.0L && variance_b > 0.0L) {
                double corr = static_cast<double>(
                    covariance / std::sqrt(variance_a * variance_b));
                corr = std::max(-1.0, std::min(1.0, corr));
                matrix[a * n_assets + b] = corr;
                matrix[b * n_assets + a] = corr;
            }
        }
    }
    return matrix;
}

double sharpe(const double* returns, std::size_t n, double periods_per_year,
              double risk_free) {
    require_pointer(returns, "returns must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(periods_per_year) && periods_per_year > 0.0,
            "periods_per_year must be finite and positive");
    require(std::isfinite(risk_free), "risk_free must be finite");

    std::vector<double> excess;
    excess.reserve(n);
    const double rf_period = risk_free / periods_per_year;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(returns[i])) excess.push_back(returns[i] - rf_period);
    }
    if (excess.size() < 2) return kNaN;
    const double mean = finite_mean(excess);
    long double sum = 0.0L;
    for (double value : excess) {
        const long double diff = static_cast<long double>(value - mean);
        sum += diff * diff;
    }
    const double variance =
        static_cast<double>(sum / static_cast<long double>(excess.size() - 1));
    if (!(variance > 0.0)) return kNaN;
    return mean / std::sqrt(variance) * std::sqrt(periods_per_year);
}

double sortino(const double* returns, std::size_t n, double periods_per_year,
               double risk_free) {
    require_pointer(returns, "returns must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(periods_per_year) && periods_per_year > 0.0,
            "periods_per_year must be finite and positive");
    require(std::isfinite(risk_free), "risk_free must be finite");

    std::vector<double> excess;
    excess.reserve(n);
    const double rf_period = risk_free / periods_per_year;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(returns[i])) excess.push_back(returns[i] - rf_period);
    }
    if (excess.size() < 2) return kNaN;
    const double mean = finite_mean(excess);
    long double downside_sum = 0.0L;
    for (double value : excess) {
        const double downside = std::min(value, 0.0);
        downside_sum += static_cast<long double>(downside * downside);
    }
    const double downside_deviation = std::sqrt(
        static_cast<double>(downside_sum /
                            static_cast<long double>(excess.size())));
    if (!(downside_deviation > 0.0)) return kNaN;
    return mean / downside_deviation * std::sqrt(periods_per_year);
}

TailRisk historical_var_cvar(const double* returns, std::size_t n,
                             double confidence) {
    require_pointer(returns, "returns must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(confidence) && confidence > 0.0 &&
                confidence < 1.0,
            "confidence must be in (0, 1)");
    std::vector<double> data = finite_values(returns, n);
    if (data.empty()) {
        throw InvalidArgument("var_cvar requires at least one finite value");
    }
    std::sort(data.begin(), data.end());
    const double lower_quantile = quantile_sorted(data, 1.0 - confidence);
    long double tail_sum = 0.0L;
    std::size_t tail_count = 0;
    for (double value : data) {
        if (value <= lower_quantile) {
            tail_sum += static_cast<long double>(value);
            ++tail_count;
        }
    }
    TailRisk result;
    result.var = -lower_quantile;
    result.cvar = tail_count > 0
                      ? -static_cast<double>(
                            tail_sum / static_cast<long double>(tail_count))
                      : kNaN;
    return result;
}

}  // namespace rflux::qflux
