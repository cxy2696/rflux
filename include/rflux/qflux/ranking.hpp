#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::qflux {

/* Average ascending (1-based) rank; ties share their mean rank. */
std::vector<double> rank(const double* values, std::size_t n);

/* (rank - 0.5) / n_valid. */
std::vector<double> percentile_rank(const double* values, std::size_t n);

/* Cross-sectional z-score over finite values. */
std::vector<double> zscore(const double* values, std::size_t n,
                           std::size_t ddof = 0);

/* Clip to empirical quantiles [lower_q, upper_q]. */
std::vector<double> winsorize(const double* values, std::size_t n,
                              double lower_q, double upper_q);

/* Demean within integer groups; group -1 is excluded. */
std::vector<double> neutralize(const double* values, std::size_t n,
                               const std::int32_t* groups,
                               std::size_t group_count);

}  // namespace rflux::qflux
