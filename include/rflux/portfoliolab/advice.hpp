#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rflux::portfoliolab {

/* One per-asset signal row used by the research rebalance proposal.  A NaN
 * marks a signal that is not defined for the available history. */
struct AdviceSignalRow {
    double last_price = 0.0;
    double momentum_20 = 0.0;
    double momentum_60 = 0.0;
    double trend = 0.0;
    double rsi_14 = 0.0;
    double daily_sigma = 0.0;
    double reference_stop = 0.0;
};

struct AdviceSignals {
    std::vector<AdviceSignalRow> rows;
    std::vector<double> scores;
};

/*
 * Compute the 20/60-day momentum, SMA trend, RSI, 20-day daily sigma and
 * 2-sigma reference stop for every asset, then combine the cross-sectional
 * z-scores and RSI adjustment into a score in [-1, 1].  `prices` and
 * `returns` are row-major n_obs x n_assets panels.
 */
[[nodiscard]] AdviceSignals advice_signals(const double* prices,
                                           const double* returns,
                                           std::size_t n_obs,
                                           std::size_t n_assets);

struct AdviceAllocation {
    std::vector<double> base_weights;
    std::vector<double> current_weights;
    std::vector<double> target_weights;
    std::vector<double> deltas;
    std::vector<std::int32_t> actions; /* 1 increase, 0 hold, -1 decrease */
    double market_annualized_volatility = 0.0;
    double portfolio_annualized_volatility = 0.0;
    double volatility_scale = 1.0;
    double exposure = 1.0;
    double cash_weight = 0.0;
    double current_cash_weight = 0.0;
    double turnover = 0.0;
};

/*
 * Build the capped HRP base weights, tilt them by `scores`, scale exposure by
 * the proposed portfolio's ex-ante volatility and the supplied regime scale,
 * and resolve the current (or equal-weight) book.  `covariance` is a
 * row-major n_assets x n_assets matrix; `market_returns` may be empty.
 */
[[nodiscard]] AdviceAllocation advice_allocation(
    const double* covariance, const double* scores, std::size_t n_assets,
    const double* current_weights, bool has_current_weights,
    const double* market_returns, std::size_t n_obs,
    double periods_per_year, double max_weight, double target_volatility,
    double regime_scale);

struct AdviceRegime {
    std::int32_t latest_state = 0;
    double expected_next_return = 0.0;
    double stress_probability = 0.0;
    double regime_scale = 1.0;
};

/* Turn filtered/forecast state probabilities and state means into the
 * exposure scale used by an advice proposal. */
[[nodiscard]] AdviceRegime advice_regime(const double* filtered,
                                         const double* forecast,
                                         const double* means,
                                         std::size_t n_states);

struct AdviceSizingRow {
    double target_notional = 0.0;
    double stop_distance_pct = 0.0;
    std::int64_t target_shares = 0;
    std::int64_t risk_limited_shares = 0;
    std::int64_t recommended_shares = 0;
    bool has_notional = false;
    bool has_stop_distance = false;
    bool has_target_shares = false;
    bool has_risk_limited_shares = false;
    bool has_recommended_shares = false;
};

/*
 * Position sizing for a long-only proposal.  `last_prices` and
 * `reference_stops` are the signal columns; when `portfolio_value` is absent
 * every share count stays undefined.
 */
[[nodiscard]] std::vector<AdviceSizingRow> advice_sizing(
    const double* target_weights, const double* last_prices,
    const double* reference_stops, std::size_t n_assets,
    double portfolio_value, bool has_portfolio_value, double risk_per_trade);

}  // namespace rflux::portfoliolab
