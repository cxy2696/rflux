#pragma once

#include <cstddef>
#include <vector>

namespace rflux::risk {

/*
 * Empirical pairwise tail-dependence matrix.
 *
 * For each pair the two finite-sample conditionals are averaged, so the
 * result is symmetric.  The diagonal is one; a pair with an empty
 * conditioning tail is NaN.  `upper` selects the upper tail.
 */
[[nodiscard]] std::vector<double> tail_dependence(const double* returns,
                                                  std::size_t n_obs,
                                                  std::size_t n_assets,
                                                  double q, bool upper);

}  // namespace rflux::risk
