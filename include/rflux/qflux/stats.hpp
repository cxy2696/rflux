#pragma once

#include <cstddef>
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

/* Type-7 quantile over finite values; q in [0,1]. */
double quantile(const double* values, std::size_t n, double q);

/* Pairwise-complete Pearson correlation. */
double pearson(const double* x, const double* y, std::size_t n);

/* Pairwise-complete correlation matrix of a row-major n_obs x n_assets panel. */
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
