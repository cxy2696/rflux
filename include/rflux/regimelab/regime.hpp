#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::regimelab {

struct VolatilityRegimeResult {
    std::vector<double> volatility;  /* raw sample std, length n */
    std::vector<std::int32_t> labels; /* 0 low, 1 normal, 2 high, -1 warm-up */
};

struct TrendRegimeResult {
    std::vector<double> fast_ma;
    std::vector<double> slow_ma;
    std::vector<std::int32_t> labels; /* 0 down, 1 sideways, 2 up, -1 warm-up */
};

/*
 * Strictly causal volatility regime classification.  The current realized
 * volatility is compared with empirical quantiles of the preceding `lookback`
 * valid volatility values.  At least 20 prior values are required.
 */
VolatilityRegimeResult volatility_regime(const double* returns, std::size_t n,
                                         std::size_t vol_window = 20,
                                         std::size_t lookback = 252,
                                         double low_q = 0.25,
                                         double high_q = 0.75);

/* Causal SMA-spread trend classification. */
TrendRegimeResult trend_regime(const double* prices, std::size_t n,
                               std::size_t fast_window = 20,
                               std::size_t slow_window = 60,
                               double threshold = 0.0);

struct StickyRegimeResult {
    std::vector<std::int32_t> labels;
    std::vector<std::int32_t> switches;
};

/*
 * Hysteresis-smoothed causal labels from row-major state probabilities.  A
 * challenger replaces the incumbent only after `min_duration` bars, when the
 * challenger reaches `entry_threshold` and the incumbent falls to
 * `exit_threshold`.
 */
StickyRegimeResult sticky_regime(const double* probabilities,
                                 std::size_t n_obs, std::size_t n_states,
                                 std::size_t min_duration,
                                 double entry_threshold,
                                 double exit_threshold);

/*
 * Risk-off score: max(volatility_label, 2 - trend_label).  Undefined inputs
 * stay undefined (-1).
 */
std::vector<std::int32_t> composite_regime(
    const std::int32_t* volatility_labels, const std::int32_t* trend_labels,
    std::size_t n);

}  // namespace rflux::regimelab
