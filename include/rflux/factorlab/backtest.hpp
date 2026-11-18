#pragma once

#include <cstddef>
#include <vector>

namespace rflux::factorlab {

/*
 * Forward simple or log returns over `periods` bars.  `prices` is a row-major
 * n_obs x n_assets panel; the trailing `periods` rows are NaN because their
 * future price is unknown.  A pair is valid only when both prices are finite,
 * and the simple path also requires a non-zero base price.
 */
[[nodiscard]] std::vector<double> forward_returns(
    const double* prices, std::size_t n_obs, std::size_t n_assets,
    std::size_t periods, bool log);

/*
 * Equal-weight forward returns by factor quantile, shape
 * (n_quantiles, n_obs).  Eligible assets need a finite factor and forward
 * return; they are split with `np.array_split` semantics, so earlier buckets
 * carry one extra member when the cross-section does not divide evenly.
 * Empty buckets are NaN.
 */
[[nodiscard]] std::vector<double> quantile_portfolio_returns(
    const double* factor, const double* forward, std::size_t n_obs,
    std::size_t n_assets, std::size_t n_quantiles);

/* Result of the equal-weight top-minus-bottom factor portfolio. */
struct LongShortBacktest {
    std::vector<double> gross_returns;
    std::vector<double> costs;
    std::vector<double> turnover;
    std::vector<double> net_returns;
    std::vector<double> equity;           /* n_obs + 1 values, starting at 1 */
    std::vector<double> quantile_returns; /* n_quantiles x n_obs, row-major */
};

/*
 * Equal-weight top-minus-bottom portfolio with one-way turnover costs.
 * `cost_bps` is charged on `turnover[t] = 0.5 * sum_j |w[t,j] - w[t-1,j]|`.
 * Rows with fewer than two non-empty buckets are flat, but closing a previous
 * position still incurs cost so the capital path stays defined.
 */
[[nodiscard]] LongShortBacktest long_short_backtest(
    const double* factor, const double* forward, std::size_t n_obs,
    std::size_t n_assets, std::size_t n_quantiles, double cost_bps);

/* Per-period IC summary; `information_ratio` is mean/std, not annualised. */
struct IcSummary {
    double mean = 0.0;
    double std = 0.0;
    double information_ratio = 0.0;
    double positive_ratio = 0.0;
    std::size_t periods = 0;
};

[[nodiscard]] IcSummary summarize_ic(const double* ic, std::size_t n);

}  // namespace rflux::factorlab
