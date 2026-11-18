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
 * Sample covariance over rows where every asset is finite.  Returns a
 * row-major n_assets x n_assets matrix; when there are not enough complete
 * rows it is all NaN and *complete_observations still reports the count.
 */
[[nodiscard]] std::vector<double> complete_case_covariance(
    const double* returns, std::size_t n_obs, std::size_t n_assets,
    std::size_t ddof, std::size_t* complete_observations);

/*
 * Row sum of returns * weights, skipping NaN cells (the Python nansum
 * convention).  A row without a finite term is 0.
 */
[[nodiscard]] std::vector<double> portfolio_series(
    const double* returns, std::size_t n_obs, std::size_t n_assets,
    const double* weights);

/*
 * Sample covariance from complete rows, Euler volatility contributions and
 * the portfolio return path.  Weights are exposures and may be long-short.
 */
[[nodiscard]] PortfolioRisk portfolio_risk(
    const double* returns, std::size_t n_obs, std::size_t n_assets,
    const double* weights, std::size_t ddof, double periods_per_year);

}  // namespace rflux::portfoliolab
