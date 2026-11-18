#include "rflux/factorlab/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "internal/common.hpp"

namespace rflux::factorlab {
namespace {

bool has_dispersion(const std::vector<double>& values) {
    if (values.size() < 2) return false;
    const double first = values.front();
    for (double value : values) {
        if (value != first) return true;
    }
    return false;
}

std::vector<double> average_ranks(const std::vector<double>& values) {
    const std::size_t n = values.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         return values[a] < values[b];
                     });
    std::vector<double> ranks(n, 0.0);
    std::size_t begin = 0;
    while (begin < n) {
        std::size_t end = begin + 1;
        while (end < n && values[order[end]] == values[order[begin]]) ++end;
        const double rank =
            0.5 * (static_cast<double>(begin + 1) +
                   static_cast<double>(end));
        for (std::size_t index = begin; index < end; ++index) {
            ranks[order[index]] = rank;
        }
        begin = end;
    }
    return ranks;
}

double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    long double sum_x = 0.0L;
    long double sum_y = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        sum_x += static_cast<long double>(x[i]);
        sum_y += static_cast<long double>(y[i]);
    }
    const long double mean_x = sum_x / static_cast<long double>(n);
    const long double mean_y = sum_y / static_cast<long double>(n);
    long double covariance = 0.0L;
    long double variance_x = 0.0L;
    long double variance_y = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const long double dx = static_cast<long double>(x[i]) - mean_x;
        const long double dy = static_cast<long double>(y[i]) - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if (!(variance_x > 0.0L) || !(variance_y > 0.0L)) {
        return rflux::detail::kNaN;
    }
    return static_cast<double>(
        covariance / std::sqrt(variance_x * variance_y));
}

}  // namespace

std::vector<double> information_coefficient(
    const double* factor, const double* forward_returns, std::size_t n_obs,
    std::size_t n_assets, bool spearman) {
    rflux::detail::require_pointer(factor, "factor must not be NULL");
    rflux::detail::require_pointer(forward_returns,
                                   "forward_returns must not be NULL");
    rflux::detail::require(n_obs > 0, "n_obs must be positive");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::checked_mul(n_obs, n_assets, "n_obs*n_assets overflows");
    std::vector<double> result(n_obs, rflux::detail::kNaN);
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(n_assets);
    y.reserve(n_assets);
    for (std::size_t row = 0; row < n_obs; ++row) {
        x.clear();
        y.clear();
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            const double fx = factor[row * n_assets + asset];
            const double fy = forward_returns[row * n_assets + asset];
            if (!std::isfinite(fx) || !std::isfinite(fy)) continue;
            x.push_back(fx);
            y.push_back(fy);
        }
        if (x.size() < 2 || !has_dispersion(x) || !has_dispersion(y)) continue;
        if (spearman) {
            x = average_ranks(x);
            y = average_ranks(y);
        }
        result[row] = pearson(x, y);
    }
    return result;
}

std::vector<double> factor_turnover(const double* weights, std::size_t n_obs,
                                    std::size_t n_assets) {
    rflux::detail::require_pointer(weights, "weights must not be NULL");
    rflux::detail::require(n_obs > 0, "n_obs must be positive");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::checked_mul(n_obs, n_assets, "n_obs*n_assets overflows");
    std::vector<double> result(n_obs, 0.0);
    for (std::size_t row = 0; row < n_obs; ++row) {
        double sum = 0.0;
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            const double value = weights[row * n_assets + asset];
            rflux::detail::require(std::isfinite(value),
                                   "weights must be finite");
            if (row > 0) {
                sum += std::abs(value -
                                weights[(row - 1) * n_assets + asset]);
            }
        }
        result[row] = 0.5 * sum;
    }
    return result;
}

}  // namespace rflux::factorlab
