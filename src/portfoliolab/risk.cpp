#include "rflux/portfoliolab/risk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "internal/common.hpp"

namespace rflux::portfoliolab {

PortfolioRisk portfolio_risk(const double* returns, std::size_t n_obs,
                             std::size_t n_assets, const double* weights,
                             std::size_t ddof, double periods_per_year) {
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require_pointer(weights, "weights must not be NULL");
    rflux::detail::require(n_obs >= 2, "at least two observations are required");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::require(ddof >= 1, "ddof must be positive");
    rflux::detail::require(std::isfinite(periods_per_year) &&
                               periods_per_year > 0.0,
                           "periods_per_year must be positive");
    rflux::detail::checked_mul(n_obs, n_assets, "n_obs*n_assets overflows");
    rflux::detail::checked_mul(n_assets, n_assets,
                               "n_assets*n_assets overflows");

    double gross = 0.0;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        rflux::detail::require(std::isfinite(weights[asset]),
                               "weights must be finite");
        gross += std::abs(weights[asset]);
    }
    rflux::detail::require(gross > 0.0,
                           "weights must have positive gross exposure");

    std::vector<std::vector<double>> rows;
    rows.reserve(n_obs);
    for (std::size_t row = 0; row < n_obs; ++row) {
        bool complete = true;
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            if (!std::isfinite(returns[row * n_assets + asset])) {
                complete = false;
                break;
            }
        }
        if (complete) {
            rows.emplace_back(returns + row * n_assets,
                              returns + (row + 1) * n_assets);
        }
    }
    const std::size_t n = rows.size();
    rflux::detail::require(n > ddof,
                           "not enough complete rows for the covariance");

    std::vector<long double> mean(n_assets, 0.0L);
    for (const auto& row : rows) {
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            mean[asset] += static_cast<long double>(row[asset]);
        }
    }
    for (long double& value : mean) value /= static_cast<long double>(n);

    PortfolioRisk result;
    result.complete_observations = n;
    result.covariance.assign(n_assets * n_assets, 0.0);
    for (std::size_t i = 0; i < n_assets; ++i) {
        for (std::size_t j = i; j < n_assets; ++j) {
            long double value = 0.0L;
            for (const auto& row : rows) {
                value += (static_cast<long double>(row[i]) - mean[i]) *
                         (static_cast<long double>(row[j]) - mean[j]);
            }
            value /= static_cast<long double>(n - ddof);
            result.covariance[i * n_assets + j] = static_cast<double>(value);
            result.covariance[j * n_assets + i] = static_cast<double>(value);
        }
    }

    result.portfolio_returns.resize(n);
    for (std::size_t row = 0; row < n; ++row) {
        long double value = 0.0L;
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            value += static_cast<long double>(rows[row][asset]) *
                     static_cast<long double>(weights[asset]);
        }
        result.portfolio_returns[row] = static_cast<double>(value);
    }

    long double portfolio_variance = 0.0L;
    for (std::size_t i = 0; i < n_assets; ++i) {
        for (std::size_t j = 0; j < n_assets; ++j) {
            portfolio_variance +=
                static_cast<long double>(weights[i] * weights[j] *
                                         result.covariance[i * n_assets + j]);
        }
    }
    if (portfolio_variance < 0.0L && portfolio_variance > -1e-18L) {
        portfolio_variance = 0.0L;
    }
    rflux::detail::require(portfolio_variance >= 0.0L,
                           "portfolio covariance produced a negative variance");
    result.annualized_volatility = std::sqrt(
        static_cast<double>(portfolio_variance) * periods_per_year);

    result.asset_volatility.resize(n_assets);
    result.component_volatility.assign(
        n_assets, std::numeric_limits<double>::quiet_NaN());
    result.percent_contribution.assign(
        n_assets, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        const double variance = result.covariance[asset * n_assets + asset];
        result.asset_volatility[asset] =
            std::sqrt(std::max(variance, 0.0) * periods_per_year);
    }
    if (result.annualized_volatility > 0.0) {
        const double root_variance = std::sqrt(
            static_cast<double>(portfolio_variance));
        long double diversification = 0.0L;
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            long double marginal = 0.0L;
            for (std::size_t other = 0; other < n_assets; ++other) {
                marginal += static_cast<long double>(
                    result.covariance[asset * n_assets + other]) *
                    static_cast<long double>(weights[other]);
            }
            const double component =
                static_cast<double>(marginal / root_variance) *
                weights[asset] * std::sqrt(periods_per_year);
            result.component_volatility[asset] = component;
            result.percent_contribution[asset] =
                component / result.annualized_volatility;
            diversification +=
                static_cast<long double>(std::abs(weights[asset])) *
                static_cast<long double>(result.asset_volatility[asset]);
        }
        result.diversification_ratio =
            static_cast<double>(diversification) /
            result.annualized_volatility;
    }

    long double hhi = 0.0L;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        const long double normalized =
            static_cast<long double>(weights[asset]) /
            static_cast<long double>(gross);
        hhi += normalized * normalized;
    }
    result.gross_hhi = static_cast<double>(hhi);
    return result;
}

}  // namespace rflux::portfoliolab
