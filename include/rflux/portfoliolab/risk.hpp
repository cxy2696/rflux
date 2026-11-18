#pragma once

#include <cstddef>
#include <vector>

namespace rflux::portfoliolab {

struct PortfolioRisk {
    std::size_t complete_observations = 0;
    std::vector<double> covariance;
    std::vector<double> asset_volatility;
    std::vector<double> component_volatility;
    std::vector<double> percent_contribution;
    std::vector<double> portfolio_returns;
    double annualized_volatility = 0.0;
    double diversification_ratio = 0.0;
    double gross_hhi = 0.0;
};

/*
 * Sample covariance from complete rows, Euler volatility contributions and
 * the portfolio return path.  Weights are exposures and may be long-short.
 */
[[nodiscard]] PortfolioRisk portfolio_risk(
    const double* returns, std::size_t n_obs, std::size_t n_assets,
    const double* weights, std::size_t ddof, double periods_per_year);

}  // namespace rflux::portfoliolab
