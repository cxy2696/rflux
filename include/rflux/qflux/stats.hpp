#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::qflux {

struct Moments {
    double mean = 0.0;
    double variance = 0.0;
    double stddev = 0.0;
    double skewness = 0.0;
    double excess_kurtosis = 0.0;
    std::size_t n_valid = 0;
};

struct TailRisk {
    double var = 0.0;
    double cvar = 0.0;
};

/*
 * Finite-value descriptive statistics.  variance/stddev use ddof = 1.
 * skewness is adjusted Fisher-Pearson G1; excess kurtosis is unbiased G2.
 * Undefined moments are NaN and never throw merely because the sample is
 * short (but at least one finite observation is required).
 */
Moments describe(const double* values, std::size_t n);

/* Equal-weight cross-sectional mean of a row-major n_obs x n_assets return
 * panel.  Rows without a finite asset stay NaN, so callers can distinguish a
 * missing cross-section from a zero return. */
[[nodiscard]] std::vector<double> equal_weight_market_returns(
    const double* returns, std::size_t n_obs, std::size_t n_assets);

/* Finite-value minimum, maximum, mean and sum; undefined values are NaN. */
struct FiniteStats {
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double sum = 0.0;
    std::size_t count = 0;
};

[[nodiscard]] FiniteStats finite_stats(const double* values, std::size_t n);

/* Copy of the finite entries in input order. */
[[nodiscard]] std::vector<double> finite_values(const double* values,
                                                std::size_t n);

/* Last finite value; sets *found and returns NaN when none exists. */
[[nodiscard]] double last_finite(const double* values, std::size_t n,
                                 bool* found);

/*
 * Asset indices ordered by value.  `descending` selects largest-first; ties
 * keep the lower index first.  At most `k` indices are returned.
 */
[[nodiscard]] std::vector<std::size_t> top_k_indices(
    const double* values, std::size_t n, std::size_t k,
    bool descending = true);

/* Per-row argmax of a row-major n_obs x n_cols matrix. */
[[nodiscard]] std::vector<std::int32_t> row_argmax(
    const double* values, std::size_t n_obs, std::size_t n_cols);

/* Index of the first maximum; a NaN input wins at its first position. */
[[nodiscard]] std::size_t argmax_index(const double* values, std::size_t n);

/* Maximum int32 value; returns 0 for an empty input. */
[[nodiscard]] std::int32_t max_i32(const std::int32_t* values, std::size_t n);

/* Type-7 quantile over finite values; q in [0,1]. */
double quantile(const double* values, std::size_t n, double q);

/* Pairwise-complete Pearson correlation. */
double pearson(const double* x, const double* y, std::size_t n);

/* Pairwise-complete correlation matrix of a row-major n_obs x n_assets panel. */
void correlation_matrix_into(const double* panel, std::size_t n_obs,
                             std::size_t n_assets, double* out);
std::vector<double> correlation_matrix(const double* panel, std::size_t n_obs,
                                       std::size_t n_assets);

/* Annualized Sharpe ratio; risk_free is an annual rate. */
double sharpe(const double* returns, std::size_t n, double periods_per_year,
              double risk_free = 0.0);

/* Annualized Sortino ratio with a zero downside target. */
double sortino(const double* returns, std::size_t n, double periods_per_year,
               double risk_free = 0.0);

/* Historical VaR/CVaR as positive losses at `confidence`. */
TailRisk historical_var_cvar(const double* returns, std::size_t n,
                             double confidence = 0.95);

}  // namespace rflux::qflux
