#pragma once

#include <cstddef>
#include <vector>

namespace rflux::qflux {

/*
 * All rolling functions return a vector with the same length as the input.
 * Warm-up positions are NaN.  A non-finite value invalidates every window that
 * contains it, so the output has NaN wherever the statistic is undefined.
 *
 * Buffer-first contract
 * ---------------------
 * Every kernel has an `_into` overload that writes exactly `n` values to a
 * caller-owned buffer instead of allocating a result vector.  The C ABI and
 * the Python bridge use those overloads, so a call has no intermediate result
 * to copy.  `out` must not alias `values` (or `x`/`y`) and must have room for
 * `n` values; the vector-returning overloads are thin wrappers that allocate
 * the result and call the buffer version.
 */

void rolling_sum_into(const double* values, std::size_t n, std::size_t window,
                      double* out);
std::vector<double> rolling_sum(const double* values, std::size_t n,
                                std::size_t window);

void rolling_mean_into(const double* values, std::size_t n, std::size_t window,
                       double* out);
std::vector<double> rolling_mean(const double* values, std::size_t n,
                                 std::size_t window);

void rolling_std_into(const double* values, std::size_t n, std::size_t window,
                      std::size_t ddof, double* out);
std::vector<double> rolling_std(const double* values, std::size_t n,
                                std::size_t window, std::size_t ddof);

void rolling_min_into(const double* values, std::size_t n, std::size_t window,
                      double* out);
std::vector<double> rolling_min(const double* values, std::size_t n,
                                std::size_t window);

void rolling_max_into(const double* values, std::size_t n, std::size_t window,
                      double* out);
std::vector<double> rolling_max(const double* values, std::size_t n,
                                std::size_t window);

void rolling_zscore_into(const double* values, std::size_t n,
                         std::size_t window, std::size_t ddof, double* out);
std::vector<double> rolling_zscore(const double* values, std::size_t n,
                                   std::size_t window, std::size_t ddof);

void rolling_quantile_into(const double* values, std::size_t n,
                           std::size_t window, double q, double* out);
std::vector<double> rolling_quantile(const double* values, std::size_t n,
                                     std::size_t window, double q);

void rolling_corr_into(const double* x, const double* y, std::size_t n,
                       std::size_t window, double* out);
std::vector<double> rolling_corr(const double* x, const double* y, std::size_t n,
                                 std::size_t window);

void rolling_cov_into(const double* x, const double* y, std::size_t n,
                      std::size_t window, std::size_t ddof, double* out);
std::vector<double> rolling_cov(const double* x, const double* y, std::size_t n,
                                std::size_t window, std::size_t ddof);

void rolling_beta_into(const double* y, const double* x, std::size_t n,
                       std::size_t window, double* out);
std::vector<double> rolling_beta(const double* y, const double* x, std::size_t n,
                                 std::size_t window);

void rolling_slope_into(const double* values, std::size_t n, std::size_t window,
                        double* out);
std::vector<double> rolling_slope(const double* values, std::size_t n,
                                  std::size_t window);

void rolling_r2_into(const double* values, std::size_t n, std::size_t window,
                     double* out);
std::vector<double> rolling_r2(const double* values, std::size_t n,
                               std::size_t window);

}  // namespace rflux::qflux
