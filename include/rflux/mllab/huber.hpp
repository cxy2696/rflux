#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "rflux/core/algorithm.hpp"

namespace rflux::mllab {

/* Huber linear regression fitted by iteratively reweighted least squares. */
class HuberRegressor final : public core::BasePredictor {
public:
    explicit HuberRegressor(double epsilon = 1.35, std::size_t max_iter = 100,
                            double tol = 1e-8, bool fit_intercept = true);

    [[nodiscard]] std::string_view name() const noexcept override {
        return "huber_regression";
    }
    void fit(const double* features, const double* target, std::size_t n_obs,
             std::size_t n_features) override;
    [[nodiscard]] std::vector<double> predict(
        const double* features, std::size_t n_obs,
        std::size_t n_features) const override;

    [[nodiscard]] const std::vector<double>& coefficients() const noexcept {
        return coefficients_;
    }
    [[nodiscard]] double intercept() const noexcept { return intercept_; }
    [[nodiscard]] double scale() const noexcept { return scale_; }
    [[nodiscard]] std::size_t iterations() const noexcept {
        return iterations_;
    }
    [[nodiscard]] bool converged() const noexcept { return converged_; }

private:
    [[nodiscard]] std::vector<long double> solve_weighted(
        const double* features, const double* target, std::size_t n_obs,
        std::size_t n_features, const std::vector<double>& weights) const;
    [[nodiscard]] long double linear_prediction(
        const double* features, std::size_t row,
        std::size_t n_features) const;
    [[nodiscard]] std::vector<double> residuals(
        const double* features, const double* target, std::size_t n_obs,
        std::size_t n_features) const;
    [[nodiscard]] static double robust_scale(
        const std::vector<double>& residuals);

    double epsilon_;
    std::size_t max_iter_;
    double tol_;
    bool fit_intercept_;
    std::vector<double> coefficients_;
    double intercept_ = 0.0;
    double scale_ = 1.0;
    std::size_t iterations_ = 0;
    bool converged_ = false;
    bool fitted_ = false;
};

}  // namespace rflux::mllab
