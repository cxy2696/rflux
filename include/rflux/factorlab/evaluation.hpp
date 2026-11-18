#pragma once

#include <cstddef>
#include <vector>

namespace rflux::factorlab {

/* Cross-sectional Pearson or Spearman IC for each row. */
[[nodiscard]] std::vector<double> information_coefficient(
    const double* factor, const double* forward_returns, std::size_t n_obs,
    std::size_t n_assets, bool spearman);

/* One-way turnover: 0.5 * sum_j |w[t,j] - w[t-1,j]|; first row is zero. */
[[nodiscard]] std::vector<double> factor_turnover(const double* weights,
                                                  std::size_t n_obs,
                                                  std::size_t n_assets);

}  // namespace rflux::factorlab
