#include "rflux/regimelab/attribution.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "internal/common.hpp"
#include "rflux/qflux/performance.hpp"
#include "rflux/qflux/stats.hpp"

namespace rflux::regimelab {

RegimeAttribution regime_attribution(const double* returns,
                                     const std::int32_t* labels,
                                     std::size_t n, double periods_per_year,
                                     double risk_free, double confidence) {
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require_pointer(labels, "labels must not be NULL");
    rflux::detail::require(n > 0, "returns must not be empty");
    rflux::detail::require(std::isfinite(periods_per_year) &&
                               periods_per_year > 0.0,
                           "periods_per_year must be positive");
    rflux::detail::require(std::isfinite(risk_free),
                           "risk_free must be finite");
    rflux::detail::require(std::isfinite(confidence) &&
                               confidence > 0.0 && confidence < 1.0,
                           "confidence must be in (0, 1)");

    std::int32_t maximum = -1;
    std::size_t labelled = 0;
    for (std::size_t i = 0; i < n; ++i) {
        rflux::detail::require(labels[i] >= -1,
                               "labels must be -1 or non-negative");
        if (labels[i] >= 0) {
            ++labelled;
            maximum = std::max(maximum, labels[i]);
        }
    }
    rflux::detail::require(maximum >= 0,
                           "labels must contain at least one valid state");
    const std::size_t state_count =
        static_cast<std::size_t>(maximum) + 1;
    rflux::detail::checked_mul(state_count, state_count,
                               "transition matrix dimensions overflow");

    long double total_sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(returns[i])) {
            total_sum += static_cast<long double>(returns[i]);
        }
    }

    RegimeAttribution result;
    result.state_count = state_count;
    result.labelled_observations = labelled;
    result.states.resize(state_count);
    for (std::size_t state = 0; state < state_count; ++state) {
        RegimeStateStats& entry = result.states[state];
        std::vector<double> selected;
        selected.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (labels[i] == static_cast<std::int32_t>(state)) {
                ++entry.periods;
                if (std::isfinite(returns[i])) {
                    selected.push_back(returns[i]);
                }
            }
        }
        entry.period_share =
            static_cast<double>(entry.periods) / static_cast<double>(n);
        entry.n_returns = selected.size();
        entry.mean_return = rflux::detail::kNaN;
        entry.annualized_mean_return = rflux::detail::kNaN;
        entry.annualized_volatility = rflux::detail::kNaN;
        entry.sharpe = rflux::detail::kNaN;
        entry.var = rflux::detail::kNaN;
        entry.cvar = rflux::detail::kNaN;
        entry.hit_rate = rflux::detail::kNaN;
        entry.sum_return = rflux::detail::kNaN;
        entry.share_of_total_sum = rflux::detail::kNaN;
        if (selected.empty()) continue;

        long double sum = 0.0L;
        std::size_t hits = 0;
        for (double value : selected) {
            sum += static_cast<long double>(value);
            if (value > 0.0) ++hits;
        }
        const long double mean =
            sum / static_cast<long double>(selected.size());
        entry.mean_return = static_cast<double>(mean);
        entry.annualized_mean_return =
            entry.mean_return * periods_per_year;
        entry.hit_rate = static_cast<double>(hits) /
                         static_cast<double>(selected.size());
        entry.sum_return = static_cast<double>(sum);
        if (total_sum != 0.0L) {
            entry.share_of_total_sum =
                static_cast<double>(sum / total_sum);
        }
        const rflux::qflux::TailRisk tail =
            rflux::qflux::historical_var_cvar(
                selected.data(), selected.size(), confidence);
        entry.var = tail.var;
        entry.cvar = tail.cvar;
        if (selected.size() >= 2) {
            long double variance = 0.0L;
            for (double value : selected) {
                const long double diff =
                    static_cast<long double>(value) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<long double>(selected.size() - 1);
            entry.annualized_volatility =
                std::sqrt(static_cast<double>(variance)) *
                std::sqrt(periods_per_year);
            entry.sharpe = rflux::qflux::sharpe(
                selected.data(), selected.size(), periods_per_year,
                risk_free);
        }
    }

    result.transition_counts.assign(state_count * state_count, 0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (labels[i] < 0 || labels[i + 1] < 0) continue;
        const std::size_t first = static_cast<std::size_t>(labels[i]);
        const std::size_t second = static_cast<std::size_t>(labels[i + 1]);
        ++result.transition_counts[first * state_count + second];
    }
    result.transition_probabilities.assign(state_count * state_count,
                                           rflux::detail::kNaN);
    for (std::size_t row = 0; row < state_count; ++row) {
        std::size_t total = 0;
        for (std::size_t column = 0; column < state_count; ++column) {
            total += result.transition_counts[row * state_count + column];
        }
        if (total == 0) continue;
        for (std::size_t column = 0; column < state_count; ++column) {
            result.transition_probabilities[row * state_count + column] =
                static_cast<double>(
                    result.transition_counts[row * state_count + column]) /
                static_cast<double>(total);
        }
    }
    return result;
}

}  // namespace rflux::regimelab
