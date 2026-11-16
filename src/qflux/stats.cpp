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
using rflux::detail::mean_of;
using rflux::detail::quantile_sorted;
using rflux::detail::require;
using rflux::detail::require_pointer;

std::vector<double> excess_returns(const double* returns, std::size_t n,
                                   double periods_per_year, double risk_free) {
    require_pointer(returns, "returns must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(periods_per_year) && periods_per_year > 0.0,
            "periods_per_year must be finite and positive");
    require(std::isfinite(risk_free), "risk_free must be finite");

    const double rf_period = risk_free / periods_per_year;
    require(std::isfinite(rf_period), "risk_free / periods_per_year overflows");
    std::vector<double> excess;
    excess.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(returns[i])) continue;
        const double value = returns[i] - rf_period;
        if (std::isfinite(value)) excess.push_back(value);
    }
    return excess;
}

/* Scale away the level so squaring cannot overflow on platforms where long
 * double is only 64-bit (notably arm64 macOS).  Mean/volatility ratios are
 * invariant under a common positive scale. */
bool scale_to_unit(std::vector<double>& values) {
    double scale = 0.0;
    for (double value : values) scale = std::max(scale, std::abs(value));
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    for (double& value : values) value /= scale;
    return true;
}

/*
 * Pearson correlation over the observations where both series are finite.
 * Means are recomputed on that pairwise-complete subset: using full-sample
 * means would change the result when the missing patterns differ.
 *
 * The stride arguments let the correlation matrix reuse this kernel without
 * copying each pair of columns out of the row-major panel.
 */
double pairwise_pearson(const double* x, const double* y, std::size_t n,
                        std::size_t stride_x = 1,
                        std::size_t stride_y = 1) {
    std::size_t count = 0;
    long double sum_x = 0.0L;
    long double sum_y = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const double xv = x[i * stride_x];
        const double yv = y[i * stride_y];
        if (!std::isfinite(xv) || !std::isfinite(yv)) continue;
        sum_x += static_cast<long double>(xv);
        sum_y += static_cast<long double>(yv);
        ++count;
    }
    if (count < 2) return kNaN;

    const long double mean_x = sum_x / static_cast<long double>(count);
    const long double mean_y = sum_y / static_cast<long double>(count);
    long double covariance = 0.0L;
    long double variance_x = 0.0L;
    long double variance_y = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const double xv = x[i * stride_x];
        const double yv = y[i * stride_y];
        if (!std::isfinite(xv) || !std::isfinite(yv)) continue;
        const long double dx =
            static_cast<long double>(xv) - mean_x;
        const long double dy =
            static_cast<long double>(yv) - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if (variance_x <= 0.0L || variance_y <= 0.0L) return kNaN;
    const double corr = static_cast<double>(
        covariance / std::sqrt(variance_x * variance_y));
    return std::max(-1.0, std::min(1.0, corr));
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
    const double mean = mean_of(data);
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

    return pairwise_pearson(x, y, n);
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
            const double corr = pairwise_pearson(
                panel + a, panel + b, n_obs, n_assets, n_assets);
            if (std::isfinite(corr)) {
                matrix[a * n_assets + b] = corr;
                matrix[b * n_assets + a] = corr;
            }
        }
    }
    return matrix;
}

double sharpe(const double* returns, std::size_t n, double periods_per_year,
              double risk_free) {
    std::vector<double> excess =
        excess_returns(returns, n, periods_per_year, risk_free);
    if (excess.size() < 2 || !scale_to_unit(excess)) return kNaN;
    const double mean = mean_of(excess);
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
    std::vector<double> excess =
        excess_returns(returns, n, periods_per_year, risk_free);
    if (excess.size() < 2 || !scale_to_unit(excess)) return kNaN;
    const double mean = mean_of(excess);
    long double downside_sum = 0.0L;
    for (double value : excess) {
        const long double downside =
            static_cast<long double>(std::min(value, 0.0));
        downside_sum += downside * downside;
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
