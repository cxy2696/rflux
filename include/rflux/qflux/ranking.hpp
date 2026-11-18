#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::qflux {

/*
 * Cross-sectional kernels.  Each function has an `_into` overload that writes
 * exactly `n` values into caller-owned storage; the vector-returning overloads
 * allocate and forward.  `out` must not alias any input.
 */

/* Average ascending (1-based) rank; ties share their mean rank. */
void rank_into(const double* values, std::size_t n, double* out);
std::vector<double> rank(const double* values, std::size_t n);

/* (rank - 0.5) / n_valid. */
void percentile_rank_into(const double* values, std::size_t n, double* out);
std::vector<double> percentile_rank(const double* values, std::size_t n);

/* Cross-sectional z-score over finite values. */
void zscore_into(const double* values, std::size_t n, std::size_t ddof,
                 double* out);
std::vector<double> zscore(const double* values, std::size_t n,
                           std::size_t ddof = 0);

/* Clip to empirical quantiles [lower_q, upper_q]. */
void winsorize_into(const double* values, std::size_t n, double lower_q,
                    double upper_q, double* out);
std::vector<double> winsorize(const double* values, std::size_t n,
                              double lower_q, double upper_q);

/* Demean within integer groups; group -1 is excluded. */
void neutralize_into(const double* values, std::size_t n,
                     const std::int32_t* groups, std::size_t group_count,
                     double* out);
std::vector<double> neutralize(const double* values, std::size_t n,
                               const std::int32_t* groups,
                               std::size_t group_count);

}  // namespace rflux::qflux
