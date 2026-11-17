#include "rflux/mllab/huber.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "internal/common.hpp"
#include "internal/linalg.hpp"

namespace rflux::mllab {
namespace {

using rflux::detail::solve_dense;

double median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 == 1) return upper;
    const double lower =
        *std::max_element(values.begin(), values.begin() + middle);
    return 0.5 * (lower + upper);
}

void validate_matrix(const double* features, const double* target,
                     std::size_t n_obs, std::size_t n_features) {
    rflux::detail::require_pointer(features, "features must not be null");
    rflux::detail::require_pointer(target, "target must not be null");
    if (n_obs < 2) {
        throw rflux::InvalidArgument("at least two observations are required");
    }
    if (n_features == 0) {
        throw rflux::InvalidArgument("at least one feature is required");
    }
    for (std::size_t i = 0; i < n_obs; ++i) {
        if (!std::isfinite(target[i])) {
            throw rflux::NonFinite("target contains a non-finite value");
        }
        for (std::size_t j = 0; j < n_features; ++j) {
            if (!std::isfinite(features[i * n_features + j])) {
                throw rflux::NonFinite(
                    "features contain a non-finite value");
            }
        }
    }
}

}  // namespace

HuberRegressor::HuberRegressor(double epsilon, std::size_t max_iter, double tol,
                               bool fit_intercept)
    : epsilon_(epsilon),
      max_iter_(max_iter),
      tol_(tol),
      fit_intercept_(fit_intercept) {
    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0) {
        throw rflux::InvalidArgument("epsilon must be finite and positive");
    }
    if (max_iter_ == 0) {
        throw rflux::InvalidArgument("max_iter must be positive");
    }
    if (!std::isfinite(tol_) || tol_ <= 0.0) {
        throw rflux::InvalidArgument("tol must be finite and positive");
    }
}

std::vector<long double> HuberRegressor::solve_weighted(
    const double* features, const double* target, std::size_t n_obs,
    std::size_t n_features, const std::vector<double>& weights) const {
    const std::size_t columns = n_features + (fit_intercept_ ? 1 : 0);
    std::vector<long double> matrix(columns * columns, 0.0L);
    std::vector<long double> rhs(columns, 0.0L);
    std::vector<long double> row(columns, 1.0L);
    for (std::size_t i = 0; i < n_obs; ++i) {
        row[0] = 1.0L;
        for (std::size_t j = 0; j < n_features; ++j) {
            row[j + (fit_intercept_ ? 1 : 0)] =
                static_cast<long double>(features[i * n_features + j]);
        }
        const long double weight =
            static_cast<long double>(weights[i]);
        for (std::size_t a = 0; a < columns; ++a) {
            for (std::size_t b = 0; b < columns; ++b) {
                matrix[a * columns + b] += weight * row[a] * row[b];
            }
            rhs[a] += weight * row[a] *
                      static_cast<long double>(target[i]);
        }
    }
    // A tiny ridge keeps the normal equations solvable for collinear factors.
    for (std::size_t i = 0; i < columns; ++i) {
        const long double diagonal = matrix[i * columns + i];
        matrix[i * columns + i] =
            diagonal + 1e-12L * (1.0L + std::abs(diagonal));
    }
    return solve_dense(std::move(matrix), std::move(rhs), columns,
                       "Huber normal equations");
}

long double HuberRegressor::linear_prediction(
    const double* features, std::size_t row,
    std::size_t n_features) const {
    long double prediction = intercept_;
    for (std::size_t j = 0; j < n_features; ++j) {
        prediction += static_cast<long double>(coefficients_[j]) *
                      static_cast<long double>(
                          features[row * n_features + j]);
    }
    return prediction;
}

std::vector<double> HuberRegressor::residuals(
    const double* features, const double* target, std::size_t n_obs,
    std::size_t n_features) const {
    std::vector<double> result(n_obs, 0.0);
    for (std::size_t i = 0; i < n_obs; ++i) {
        result[i] = static_cast<double>(
            static_cast<long double>(target[i]) -
            linear_prediction(features, i, n_features));
    }
    return result;
}

double HuberRegressor::robust_scale(const std::vector<double>& residuals) {
    if (residuals.empty()) return 1e-12;
    std::vector<double> centered = residuals;
    const double center = median(centered);
    for (double& value : centered) value = std::abs(value - center);
    const double mad = median(centered);
    return std::max(1.4826 * mad, 1e-12);
}

void HuberRegressor::fit(const double* features, const double* target,
                         std::size_t n_obs, std::size_t n_features) {
    validate_matrix(features, target, n_obs, n_features);
    const std::size_t columns = n_features + (fit_intercept_ ? 1 : 0);
    std::vector<double> weights(n_obs, 1.0);
    std::vector<long double> beta = solve_weighted(
        features, target, n_obs, n_features, weights);
    coefficients_.assign(n_features, 0.0);
    intercept_ = 0.0;
    iterations_ = 0;
    converged_ = false;

    const std::size_t offset = fit_intercept_ ? 1 : 0;
    for (std::size_t iteration = 1; iteration <= max_iter_; ++iteration) {
        for (std::size_t j = 0; j < n_features; ++j) {
            coefficients_[j] = static_cast<double>(beta[j + offset]);
        }
        intercept_ = fit_intercept_ ? static_cast<double>(beta[0]) : 0.0;
        const std::vector<double> residual = residuals(
            features, target, n_obs, n_features);
        scale_ = robust_scale(residual);
        const double cutoff = epsilon_ * scale_;
        for (std::size_t i = 0; i < n_obs; ++i) {
            const double absolute = std::abs(residual[i]);
            weights[i] = absolute <= cutoff
                             ? 1.0
                             : cutoff / std::max(absolute, 1e-300);
        }
        std::vector<long double> updated = solve_weighted(
            features, target, n_obs, n_features, weights);
        long double maximum = 0.0L;
        for (std::size_t i = 0; i < columns; ++i) {
            maximum = std::max(maximum, std::abs(updated[i] - beta[i]));
        }
        beta = std::move(updated);
        iterations_ = iteration;
        if (maximum <= static_cast<long double>(tol_)) {
            converged_ = true;
            break;
        }
    }

    for (std::size_t j = 0; j < n_features; ++j) {
        coefficients_[j] = static_cast<double>(beta[j + offset]);
    }
    intercept_ = fit_intercept_ ? static_cast<double>(beta[0]) : 0.0;
    scale_ = robust_scale(residuals(features, target, n_obs, n_features));
    fitted_ = true;
}

std::vector<double> HuberRegressor::predict(const double* features,
                                            std::size_t n_obs,
                                            std::size_t n_features) const {
    rflux::detail::require(fitted_, "HuberRegressor is not fitted");
    rflux::detail::require(n_features == coefficients_.size(),
            "features have the wrong number of columns");
    rflux::detail::require_pointer(features, "features must not be null");
    std::vector<double> result(n_obs, 0.0);
    for (std::size_t i = 0; i < n_obs; ++i) {
        result[i] = static_cast<double>(
            linear_prediction(features, i, n_features));
    }
    return result;
}

}  // namespace rflux::mllab
