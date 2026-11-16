#pragma once

#include <cstddef>
#include <vector>

namespace rflux::qflux {

/*
 * All rolling functions return a vector with the same length as the input.
 * Warm-up positions are NaN.  A non-finite value invalidates every window that
 * contains it, so the output has NaN wherever the statistic is undefined.
 */

std::vector<double> rolling_sum(const double* values, std::size_t n,
                                std::size_t window);
std::vector<double> rolling_mean(const double* values, std::size_t n,
                                 std::size_t window);
std::vector<double> rolling_std(const double* values, std::size_t n,
                                std::size_t window, std::size_t ddof);
std::vector<double> rolling_min(const double* values, std::size_t n,
                                std::size_t window);
std::vector<double> rolling_max(const double* values, std::size_t n,
                                std::size_t window);
std::vector<double> rolling_zscore(const double* values, std::size_t n,
                                   std::size_t window, std::size_t ddof);
std::vector<double> rolling_corr(const double* x, const double* y, std::size_t n,
                                 std::size_t window);
std::vector<double> rolling_cov(const double* x, const double* y, std::size_t n,
                                std::size_t window, std::size_t ddof);
std::vector<double> rolling_beta(const double* y, const double* x, std::size_t n,
                                 std::size_t window);
std::vector<double> rolling_slope(const double* values, std::size_t n,
                                  std::size_t window);
std::vector<double> rolling_r2(const double* values, std::size_t n,
                               std::size_t window);

}  // namespace rflux::qflux
