#include "rflux/factorlab/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "internal/common.hpp"

namespace rflux::factorlab {
namespace {

using rflux::detail::kNaN;

void require_panel(const double* values, std::size_t n_obs,
                   std::size_t n_assets, const char* name) {
    rflux::detail::require_pointer(values, name);
    rflux::detail::require(n_obs > 0, "n_obs must be positive");
    rflux::detail::require(n_assets > 0, "n_assets must be positive");
    rflux::detail::checked_mul(n_obs, n_assets, "panel dimensions overflow");
}

/* np.array_split sizes: the first `count % n_quantiles` buckets carry one
 * extra member.  Keeping the same rule makes the C++ and Python reference
 * results interchangeable. */
std::vector<std::size_t> bucket_offsets(std::size_t count,
                                        std::size_t n_quantiles) {
    std::vector<std::size_t> offsets(n_quantiles + 1, 0);
    const std::size_t base = count / n_quantiles;
    const std::size_t remainder = count % n_quantiles;
    std::size_t position = 0;
    for (std::size_t bucket = 0; bucket < n_quantiles; ++bucket) {
        position += base + (bucket < remainder ? 1 : 0);
        offsets[bucket + 1] = position;
    }
    return offsets;
}

struct CrossSection {
    std::vector<std::size_t> order;
    std::vector<std::size_t> offsets;
};

/* Eligible assets (finite factor and forward return) sorted by factor over
 * their fixed positions; a stable sort keeps ties in asset order like
 * np.argsort(kind="mergesort"). */
CrossSection sort_cross_section(const double* factor, const double* forward,
                                std::size_t row, std::size_t n_assets,
                                std::size_t n_quantiles) {
    CrossSection section;
    section.order.reserve(n_assets);
    const std::size_t base = row * n_assets;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        if (std::isfinite(factor[base + asset]) &&
            std::isfinite(forward[base + asset])) {
            section.order.push_back(asset);
        }
    }
    std::stable_sort(section.order.begin(), section.order.end(),
                     [&](std::size_t left, std::size_t right) {
                         return factor[base + left] < factor[base + right];
                     });
    section.offsets = bucket_offsets(section.order.size(), n_quantiles);
    return section;
}

double bucket_mean(const double* forward, std::size_t row,
                   std::size_t n_assets, const std::vector<std::size_t>& order,
                   std::size_t begin, std::size_t end) {
    if (begin == end) return kNaN;
    long double sum = 0.0L;
    for (std::size_t index = begin; index < end; ++index) {
        sum += static_cast<long double>(forward[row * n_assets + order[index]]);
    }
    return static_cast<double>(sum / static_cast<long double>(end - begin));
}

}  // namespace

std::vector<double> forward_returns(const double* prices, std::size_t n_obs,
                                    std::size_t n_assets, std::size_t periods,
                                    bool log) {
    require_panel(prices, n_obs, n_assets, "prices must not be NULL");
    rflux::detail::require(periods > 0, "periods must be positive");
    std::vector<double> result(n_obs * n_assets, kNaN);
    if (n_obs <= periods) return result;
    for (std::size_t row = 0; row + periods < n_obs; ++row) {
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            const double base = prices[row * n_assets + asset];
            const double future = prices[(row + periods) * n_assets + asset];
            if (!std::isfinite(base) || !std::isfinite(future)) continue;
            if (log) {
                if (base <= 0.0 || future <= 0.0) continue;
                result[row * n_assets + asset] = std::log(future / base);
            } else {
                if (base == 0.0) continue;
                result[row * n_assets + asset] = future / base - 1.0;
            }
        }
    }
    return result;
}

std::vector<double> quantile_portfolio_returns(
    const double* factor, const double* forward, std::size_t n_obs,
    std::size_t n_assets, std::size_t n_quantiles) {
    require_panel(factor, n_obs, n_assets, "factor must not be NULL");
    require_panel(forward, n_obs, n_assets,
                  "forward_returns must not be NULL");
    rflux::detail::require(n_quantiles >= 2,
                           "n_quantiles must be at least two");
    rflux::detail::checked_mul(n_quantiles, n_obs,
                               "quantile result dimensions overflow");
    std::vector<double> result(n_quantiles * n_obs, kNaN);
    for (std::size_t row = 0; row < n_obs; ++row) {
        const CrossSection section =
            sort_cross_section(factor, forward, row, n_assets, n_quantiles);
        for (std::size_t bucket = 0; bucket < n_quantiles; ++bucket) {
            result[bucket * n_obs + row] =
                bucket_mean(forward, row, n_assets, section.order,
                            section.offsets[bucket],
                            section.offsets[bucket + 1]);
        }
    }
    return result;
}

LongShortBacktest long_short_backtest(const double* factor,
                                      const double* forward,
                                      std::size_t n_obs, std::size_t n_assets,
                                      std::size_t n_quantiles,
                                      double cost_bps) {
    require_panel(factor, n_obs, n_assets, "factor must not be NULL");
    require_panel(forward, n_obs, n_assets,
                  "forward_returns must not be NULL");
    rflux::detail::require(n_quantiles >= 2,
                           "n_quantiles must be at least two");
    rflux::detail::require(std::isfinite(cost_bps) && cost_bps >= 0.0,
                           "cost_bps must be finite and non-negative");
    rflux::detail::checked_mul(n_quantiles, n_obs,
                               "quantile result dimensions overflow");
    rflux::detail::require(n_obs < std::numeric_limits<std::size_t>::max(),
                           "n_obs is too large");

    LongShortBacktest result;
    result.gross_returns.assign(n_obs, 0.0);
    result.costs.assign(n_obs, 0.0);
    result.turnover.assign(n_obs, 0.0);
    result.net_returns.assign(n_obs, 0.0);
    result.equity.assign(n_obs + 1, 1.0);
    result.quantile_returns.assign(n_quantiles * n_obs, kNaN);

    std::vector<double> previous_weights(n_assets, 0.0);
    std::vector<double> current_weights(n_assets, 0.0);
    for (std::size_t row = 0; row < n_obs; ++row) {
        const CrossSection section =
            sort_cross_section(factor, forward, row, n_assets, n_quantiles);
        for (std::size_t bucket = 0; bucket < n_quantiles; ++bucket) {
            result.quantile_returns[bucket * n_obs + row] =
                bucket_mean(forward, row, n_assets, section.order,
                            section.offsets[bucket],
                            section.offsets[bucket + 1]);
        }

        std::fill(current_weights.begin(), current_weights.end(), 0.0);
        std::size_t low_bucket = n_quantiles;
        std::size_t high_bucket = n_quantiles;
        for (std::size_t bucket = 0; bucket < n_quantiles; ++bucket) {
            if (section.offsets[bucket] == section.offsets[bucket + 1]) {
                continue;
            }
            if (low_bucket == n_quantiles) low_bucket = bucket;
            high_bucket = bucket;
        }

        double gross = 0.0;
        if (low_bucket != n_quantiles && high_bucket != low_bucket) {
            const std::size_t low_begin = section.offsets[low_bucket];
            const std::size_t low_end = section.offsets[low_bucket + 1];
            const std::size_t high_begin = section.offsets[high_bucket];
            const std::size_t high_end = section.offsets[high_bucket + 1];
            const double low_weight =
                -1.0 / static_cast<double>(low_end - low_begin);
            const double high_weight =
                1.0 / static_cast<double>(high_end - high_begin);
            for (std::size_t index = low_begin; index < low_end; ++index) {
                current_weights[section.order[index]] = low_weight;
            }
            for (std::size_t index = high_begin; index < high_end; ++index) {
                current_weights[section.order[index]] = high_weight;
            }
            gross = bucket_mean(forward, row, n_assets, section.order,
                                high_begin, high_end) -
                    bucket_mean(forward, row, n_assets, section.order,
                                low_begin, low_end);
        }

        double absolute_change = 0.0;
        for (std::size_t asset = 0; asset < n_assets; ++asset) {
            absolute_change +=
                std::abs(current_weights[asset] - previous_weights[asset]);
        }
        result.gross_returns[row] = gross;
        result.turnover[row] = 0.5 * absolute_change;
        result.costs[row] = result.turnover[row] * cost_bps / 10000.0;
        result.net_returns[row] = gross - result.costs[row];
        result.equity[row + 1] =
            result.equity[row] * (1.0 + result.net_returns[row]);
        previous_weights.swap(current_weights);
    }
    return result;
}

IcSummary summarize_ic(const double* ic, std::size_t n) {
    rflux::detail::require_pointer(ic, "ic must not be NULL");
    IcSummary summary;
    std::size_t count = 0;
    long double sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const double value = ic[i];
        if (!std::isfinite(value)) continue;
        sum += static_cast<long double>(value);
        ++count;
    }
    if (count == 0) {
        summary.mean = kNaN;
        summary.std = kNaN;
        summary.information_ratio = kNaN;
        summary.positive_ratio = kNaN;
        summary.periods = 0;
        return summary;
    }
    const long double mean = sum / static_cast<long double>(count);
    long double squared_deviation = 0.0L;
    std::size_t positive = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double value = ic[i];
        if (!std::isfinite(value)) continue;
        const long double deviation =
            static_cast<long double>(value) - mean;
        squared_deviation += deviation * deviation;
        if (value > 0.0) ++positive;
    }
    summary.mean = static_cast<double>(mean);
    summary.std =
        count >= 2
            ? static_cast<double>(std::sqrt(
                  squared_deviation / static_cast<long double>(count - 1)))
            : kNaN;
    summary.information_ratio =
        std::isfinite(summary.std) && summary.std > 0.0
            ? summary.mean / summary.std
            : kNaN;
    summary.positive_ratio =
        static_cast<double>(positive) / static_cast<double>(count);
    summary.periods = count;
    return summary;
}

}  // namespace rflux::factorlab
