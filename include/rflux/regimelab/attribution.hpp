#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::regimelab {

/* Statistics for one supplied regime label; undefined fields are NaN. */
struct RegimeStateStats {
    std::size_t periods = 0;
    double period_share = 0.0;
    std::size_t n_returns = 0;
    double mean_return = 0.0;
    double annualized_mean_return = 0.0;
    double annualized_volatility = 0.0;
    double sharpe = 0.0;
    double var = 0.0;
    double cvar = 0.0;
    double hit_rate = 0.0;
    double sum_return = 0.0;
    double share_of_total_sum = 0.0;
};

struct RegimeAttribution {
    std::size_t state_count = 0;
    std::size_t labelled_observations = 0;
    std::vector<RegimeStateStats> states;
    std::vector<std::size_t> transition_counts;       /* K x K */
    std::vector<double> transition_probabilities;     /* K x K, NaN row */
};

/*
 * Summarize periodic returns by an existing state-label series.  Labels are
 * -1 (unlabelled) or non-negative state ids; consecutive valid labels form the
 * transition count matrix.  State statistics use finite returns only.
 */
[[nodiscard]] RegimeAttribution regime_attribution(
    const double* returns, const std::int32_t* labels, std::size_t n,
    double periods_per_year, double risk_free, double confidence);

}  // namespace rflux::regimelab
