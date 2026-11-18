#include "rflux/mllab/huber.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "internal/common.hpp"

namespace rflux::mllab {

std::vector<double> linear_predict(const double* features,
                                   std::size_t n_obs,
                                   std::size_t n_features,
                                   const double* coefficients,
                                   double intercept) {
    rflux::detail::require_pointer(features, "features must not be NULL");
    rflux::detail::require_pointer(coefficients,
                                   "coefficients must not be NULL");
    rflux::detail::require(n_obs > 0, "n_obs must be positive");
    rflux::detail::require(n_features > 0, "n_features must be positive");
    rflux::detail::checked_mul(n_obs, n_features,
                               "feature dimensions overflow");
    std::vector<double> result(n_obs, 0.0);
    for (std::size_t row = 0; row < n_obs; ++row) {
        long double value = static_cast<long double>(intercept);
        for (std::size_t feature = 0; feature < n_features; ++feature) {
            value +=
                static_cast<long double>(
                    features[row * n_features + feature]) *
                static_cast<long double>(coefficients[feature]);
        }
        result[row] = static_cast<double>(value);
    }
    return result;
}

double cvar_loss(const double* returns, std::size_t n, double alpha) {
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require(n > 0, "returns must not be empty");
    rflux::detail::require(std::isfinite(alpha) && alpha > 0.0 &&
                               alpha < 1.0,
                           "alpha must be in (0, 1)");
    std::vector<double> losses(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        rflux::detail::require(std::isfinite(returns[i]),
                               "returns must be finite");
        losses[i] = -returns[i];
    }
    const double cutoff = rflux::detail::quantile_select(
        losses.data(), losses.size(), alpha);
    long double sum = 0.0L;
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (losses[i] >= cutoff) {
            sum += static_cast<long double>(losses[i]);
            ++count;
        }
    }
    return static_cast<double>(sum / static_cast<long double>(count));
}

std::vector<double> graph_smooth(const double* signal, const double* adjacency,
                                 std::size_t n, double alpha,
                                 std::size_t steps) {
    rflux::detail::require_pointer(signal, "signal must not be NULL");
    rflux::detail::require_pointer(adjacency, "adjacency must not be NULL");
    rflux::detail::require(n > 0, "n must be positive");
    rflux::detail::require(std::isfinite(alpha) && alpha >= 0.0 &&
                               alpha <= 1.0,
                           "alpha must be in [0, 1]");
    rflux::detail::checked_mul(n, n, "n*n overflows");
    std::vector<double> result(signal, signal + n);
    std::vector<double> next(n, 0.0);
    std::vector<double> degree(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        rflux::detail::require(std::isfinite(signal[i]),
                               "signal must be finite");
        for (std::size_t j = 0; j < n; ++j) {
            const double value = adjacency[i * n + j];
            rflux::detail::require(std::isfinite(value),
                                   "adjacency must be finite");
            rflux::detail::require(value >= 0.0,
                                   "adjacency must be non-negative");
            degree[i] += value;
        }
    }
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t i = 0; i < n; ++i) {
            long double neighbor = 0.0L;
            if (degree[i] > 0.0) {
                for (std::size_t j = 0; j < n; ++j) {
                    neighbor += static_cast<long double>(adjacency[i * n + j]) *
                                static_cast<long double>(result[j]);
                }
                neighbor /= static_cast<long double>(degree[i]);
            }
            next[i] = (1.0 - alpha) * result[i] +
                      alpha * static_cast<double>(neighbor);
        }
        result.swap(next);
    }
    return result;
}

}  // namespace rflux::mllab
