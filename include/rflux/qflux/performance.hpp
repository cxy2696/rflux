#pragma once

#include <cstddef>
#include <vector>

namespace rflux::qflux {

struct PerformanceSummary {
    std::size_t n_periods = 0;
    std::size_t elapsed_periods = 0;
    double total_return = 0.0;
    double annualized_return = 0.0;
    double annualized_volatility = 0.0;
    double sharpe = 0.0;
    double sortino = 0.0;
    double max_drawdown = 0.0;
    double calmar = 0.0;
    double var = 0.0;
    double cvar = 0.0;
    double hit_rate = 0.0;
};

/*
 * Annualized performance and historical tail risk.  NaN returns are excluded
 * from statistics.  `periods` is the elapsed-period count used for geometric
 * annualisation; pass 0 to use the input length.
 */
[[nodiscard]] PerformanceSummary performance_summary(
    const double* returns, std::size_t n, double periods_per_year,
    double risk_free, double confidence, std::size_t periods = 0);

/* Compounded equity curve; finite returns are compounded, NaN gaps are flat. */
[[nodiscard]] std::vector<double> equity_curve(const double* returns,
                                               std::size_t n,
                                               double initial = 1.0);

}  // namespace rflux::qflux
