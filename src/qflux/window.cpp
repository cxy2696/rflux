#include "rflux/qflux/window.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

#include "internal/common.hpp"
#include "internal/moments.hpp"

namespace rflux::qflux {
namespace {

using rflux::detail::kNaN;
using rflux::detail::Moments1;
using rflux::detail::Moments2;
using rflux::detail::require;
using rflux::detail::require_pointer;
using rflux::detail::SlidingMoments1;
using rflux::detail::SlidingMoments2;

void validate_rolling(const double* values, std::size_t n, std::size_t window,
                      std::size_t ddof = 0) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(window > 0 && window <= n, "window must be in [1, n]");
    require(ddof < window, "ddof must be smaller than window");
}

void require_output(const double* out) {
    require_pointer(out, "out must not be NULL");
}

/*
 * Shared window traversal for every rolling kernel.  `valid(index)` decides
 * whether an observation participates, `push`/`pop` maintain the window state,
 * and `emit` runs only for a complete window (warm-up finished and no missing
 * value inside).  Centralising this loop keeps the warm-up, expiry and
 * missing-data rules identical across the moment, extreme and quantile
 * kernels.
 */
template <typename Valid, typename Push, typename Pop, typename Emit>
void for_each_complete_window(std::size_t n, std::size_t window, Valid valid,
                              Push push, Pop pop, Emit emit) {
    std::size_t missing = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (valid(i)) {
            push(i);
        } else {
            ++missing;
        }
        if (i >= window) {
            const std::size_t old = i - window;
            if (valid(old)) {
                pop(old);
            } else {
                --missing;
            }
        }
        if (i + 1 >= window && missing == 0) emit(i);
    }
}

/* Zero-based ranks and interpolation weight of the Type-7 position. */
struct QuantilePosition {
    std::size_t lower = 0;
    std::size_t upper = 0;
    double fraction = 0.0;
};

inline QuantilePosition quantile_position(std::size_t window, double q) {
    const double position = q * static_cast<double>(window - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    return {lower, std::min(lower + 1, window - 1),
            position - static_cast<double>(lower)};
}

template <typename Lookup>
double quantile_from_ranks(const QuantilePosition& position, Lookup lookup) {
    const double low = lookup(position.lower);
    if (position.lower == position.upper) return low;
    return low + position.fraction * (lookup(position.upper) - low);
}

/* Shared output/warm-up scaffold for both order-statistic backends. */
template <typename Push, typename Pop, typename Lookup>
void rolling_quantile_apply(const double* values, std::size_t n,
                            std::size_t window, double q, double* out,
                            Push push, Pop pop, Lookup lookup) {
    std::fill(out, out + n, kNaN);
    const QuantilePosition position = quantile_position(window, q);
    for_each_complete_window(
        n, window, [&](std::size_t i) { return std::isfinite(values[i]); },
        push, pop, [&](std::size_t i) {
            out[i] = quantile_from_ranks(position, lookup);
        });
}

/*
 * One pass over the window boundaries for std and z-score.  Both statistics
 * need the same stable sliding moments, so the caller receives mean and scale
 * through a callback and writes its own output without an intermediate pair of
 * vectors.
 */
template <typename Emit>
void for_each_center_scale(const double* values, std::size_t n,
                           std::size_t window, std::size_t ddof, Emit emit) {
    SlidingMoments1 moments;
    const long double denominator = static_cast<long double>(window - ddof);
    for_each_complete_window(
        n, window, [&](std::size_t i) { return std::isfinite(values[i]); },
        [&](std::size_t i) { moments.push(values[i]); },
        [&](std::size_t) { moments.pop(); },
        [&](std::size_t i) {
            const Moments1 block = moments.moments();
            long double variance = block.m2 / denominator;
            if (variance < 0.0L) variance = 0.0L;
            emit(i, block.mean, std::sqrt(static_cast<double>(variance)));
        });
}

/* Order-statistic tree over globally compressed values, used for windows too
 * large for the sorted-buffer path. */
class Fenwick {
public:
    explicit Fenwick(std::size_t n) : tree_(n + 1, 0) {
        while ((top_bit_ << 1) < tree_.size()) top_bit_ <<= 1;
    }

    void add(std::size_t index, std::ptrdiff_t delta) {
        for (std::size_t i = index + 1; i < tree_.size(); i += i & (~i + 1)) {
            tree_[i] += delta;
        }
    }

    /* 1-based rank -> zero-based compressed index. */
    std::size_t kth(std::size_t rank) const {
        std::size_t index = 0;
        for (std::size_t bit = top_bit_; bit != 0; bit >>= 1) {
            const std::size_t next = index + bit;
            if (next < tree_.size() &&
                tree_[next] < static_cast<std::ptrdiff_t>(rank)) {
                index = next;
                rank -= static_cast<std::size_t>(tree_[next]);
            }
        }
        return index;
    }

private:
    std::vector<std::ptrdiff_t> tree_;
    std::size_t top_bit_ = 1;
};

template <typename Emit>
void for_each_pair_window(const double* x, const double* y, std::size_t n,
                          std::size_t window, Emit emit) {
    SlidingMoments2 moments;
    for_each_complete_window(
        n, window,
        [&](std::size_t i) {
            return std::isfinite(x[i]) && std::isfinite(y[i]);
        },
        [&](std::size_t i) { moments.push(x[i], y[i]); },
        [&](std::size_t) { moments.pop(); },
        [&](std::size_t i) { emit(i, moments.moments()); });
}

/*
 * Windows up to this length use a sorted buffer: insert/erase are a
 * binary search plus a contiguous shift, which beats a Fenwick tree over a
 * global compression of the whole series.  Longer windows keep the Fenwick
 * path because the shift cost grows with the window size.
 */
constexpr std::size_t kSortedWindow = 1024;

void rolling_quantile_fenwick(const double* values, std::size_t n,
                              std::size_t window, double q, double* out) {
    std::vector<double> coordinates;
    coordinates.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) coordinates.push_back(values[i]);
    }
    std::sort(coordinates.begin(), coordinates.end());
    coordinates.erase(std::unique(coordinates.begin(), coordinates.end()),
                      coordinates.end());

    Fenwick fenwick(coordinates.size());
    std::vector<std::size_t> compressed(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            compressed[i] = static_cast<std::size_t>(
                std::lower_bound(coordinates.begin(), coordinates.end(),
                                 values[i]) -
                coordinates.begin());
        }
    }

    rolling_quantile_apply(
        values, n, window, q, out,
        [&](std::size_t i) { fenwick.add(compressed[i], 1); },
        [&](std::size_t old) { fenwick.add(compressed[old], -1); },
        [&](std::size_t rank) { return coordinates[fenwick.kth(rank + 1)]; });
}

void rolling_quantile_sorted(const double* values, std::size_t n,
                             std::size_t window, double q, double* out) {
    std::vector<double> sorted;
    sorted.reserve(window + 1);
    rolling_quantile_apply(
        values, n, window, q, out,
        [&](std::size_t i) {
            sorted.insert(std::lower_bound(sorted.begin(), sorted.end(),
                                           values[i]),
                          values[i]);
        },
        [&](std::size_t old) {
            const auto it =
                std::lower_bound(sorted.begin(), sorted.end(), values[old]);
            sorted.erase(it);
        },
        [&](std::size_t rank) { return sorted[rank]; });
}

}  // namespace

void rolling_sum_into(const double* values, std::size_t n, std::size_t window,
                      double* out) {
    validate_rolling(values, n, window);
    require_output(out);
    long double sum = 0.0L;
    std::size_t missing = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            sum += static_cast<long double>(values[i]);
        } else {
            ++missing;
        }
        if (i >= window) {
            const double old = values[i - window];
            if (std::isfinite(old)) {
                sum -= static_cast<long double>(old);
            } else {
                --missing;
            }
        }
        out[i] = (i + 1 >= window && missing == 0)
                     ? static_cast<double>(sum)
                     : kNaN;
    }
}

std::vector<double> rolling_sum(const double* values, std::size_t n,
                                std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_sum_into(values, n, window, out.data());
    return out;
}

void rolling_mean_into(const double* values, std::size_t n, std::size_t window,
                       double* out) {
    rolling_sum_into(values, n, window, out);
    const double divisor = static_cast<double>(window);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(out[i])) out[i] /= divisor;
    }
}

std::vector<double> rolling_mean(const double* values, std::size_t n,
                                 std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_mean_into(values, n, window, out.data());
    return out;
}

namespace {

template <typename Emit>
void rolling_center_scale_into(const double* values, std::size_t n,
                               std::size_t window, std::size_t ddof,
                               double* out, Emit emit) {
    validate_rolling(values, n, window, ddof);
    require_output(out);
    std::fill(out, out + n, kNaN);
    for_each_center_scale(values, n, window, ddof, emit);
}

enum class PairStatistic { correlation, covariance, beta };

/*
 * One implementation for the three rolling two-series statistics.  They share
 * the same stable two-variable moments and differ only in the final ratio.
 */
void rolling_pair_stat_into(const double* x, const double* y, std::size_t n,
                            std::size_t window, std::size_t ddof,
                            PairStatistic statistic, double* out) {
    require_pointer(x, "x must not be NULL");
    require_pointer(y, "y must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");
    require(ddof < window, "ddof must be smaller than window");
    require_output(out);
    std::fill(out, out + n, kNaN);
    for_each_pair_window(
        x, y, n, window, [&](std::size_t i, const Moments2& block) {
            switch (statistic) {
                case PairStatistic::correlation:
                    if (block.m2_x <= 0.0L || block.m2_y <= 0.0L) return;
                    out[i] = std::max(
                        -1.0, std::min(1.0,
                                       static_cast<double>(
                                           block.co_moment /
                                           std::sqrt(block.m2_x *
                                                     block.m2_y))));
                    break;
                case PairStatistic::covariance:
                    out[i] = static_cast<double>(
                        block.co_moment /
                        static_cast<long double>(window - ddof));
                    break;
                case PairStatistic::beta:
                    if (block.m2_x > 0.0L) {
                        out[i] = static_cast<double>(block.co_moment /
                                                     block.m2_x);
                    }
                    break;
            }
        });
}

}  // namespace

void rolling_std_into(const double* values, std::size_t n, std::size_t window,
                      std::size_t ddof, double* out) {
    rolling_center_scale_into(
        values, n, window, ddof, out,
        [&](std::size_t i, long double, double scale) { out[i] = scale; });
}

std::vector<double> rolling_std(const double* values, std::size_t n,
                                std::size_t window, std::size_t ddof) {
    std::vector<double> out(n, kNaN);
    rolling_std_into(values, n, window, ddof, out.data());
    return out;
}

namespace {

void rolling_extreme_into(const double* values, std::size_t n,
                          std::size_t window, bool minimum, double* out) {
    std::deque<std::size_t> candidates;
    std::size_t missing = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            while (!candidates.empty()) {
                const double candidate = values[candidates.back()];
                if (minimum ? candidate < values[i] : candidate > values[i]) {
                    break;
                }
                candidates.pop_back();
            }
            candidates.push_back(i);
        } else {
            ++missing;
        }
        if (i >= window) {
            const std::size_t expired = i - window;
            if (std::isfinite(values[expired])) {
                if (!candidates.empty() && candidates.front() == expired) {
                    candidates.pop_front();
                }
            } else {
                --missing;
            }
        }
        out[i] = (i + 1 >= window && missing == 0 && !candidates.empty())
                     ? values[candidates.front()]
                     : kNaN;
    }
}

}  // namespace

void rolling_min_into(const double* values, std::size_t n, std::size_t window,
                      double* out) {
    validate_rolling(values, n, window);
    require_output(out);
    rolling_extreme_into(values, n, window, true, out);
}

std::vector<double> rolling_min(const double* values, std::size_t n,
                                std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_min_into(values, n, window, out.data());
    return out;
}

void rolling_max_into(const double* values, std::size_t n, std::size_t window,
                      double* out) {
    validate_rolling(values, n, window);
    require_output(out);
    rolling_extreme_into(values, n, window, false, out);
}

std::vector<double> rolling_max(const double* values, std::size_t n,
                                std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_max_into(values, n, window, out.data());
    return out;
}

void rolling_zscore_into(const double* values, std::size_t n,
                         std::size_t window, std::size_t ddof, double* out) {
    rolling_center_scale_into(
        values, n, window, ddof, out,
        [&](std::size_t i, long double mean, double scale) {
            if (scale > 0.0) {
                out[i] = (values[i] - static_cast<double>(mean)) / scale;
            }
        });
}

std::vector<double> rolling_zscore(const double* values, std::size_t n,
                                   std::size_t window, std::size_t ddof) {
    std::vector<double> out(n, kNaN);
    rolling_zscore_into(values, n, window, ddof, out.data());
    return out;
}

void rolling_quantile_into(const double* values, std::size_t n,
                           std::size_t window, double q, double* out) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(window > 0 && window <= n, "window must be in [1, n]");
    require(std::isfinite(q) && q >= 0.0 && q <= 1.0, "q must be in [0, 1]");
    require_output(out);

    if (q <= 0.0) {
        rolling_extreme_into(values, n, window, true, out);
        return;
    }
    if (q >= 1.0) {
        rolling_extreme_into(values, n, window, false, out);
        return;
    }
    if (window <= kSortedWindow) {
        rolling_quantile_sorted(values, n, window, q, out);
    } else {
        rolling_quantile_fenwick(values, n, window, q, out);
    }
}

std::vector<double> rolling_quantile(const double* values, std::size_t n,
                                     std::size_t window, double q) {
    std::vector<double> out(n, kNaN);
    rolling_quantile_into(values, n, window, q, out.data());
    return out;
}

void rolling_corr_into(const double* x, const double* y, std::size_t n,
                       std::size_t window, double* out) {
    rolling_pair_stat_into(x, y, n, window, 0,
                           PairStatistic::correlation, out);
}

std::vector<double> rolling_corr(const double* x, const double* y, std::size_t n,
                                 std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_corr_into(x, y, n, window, out.data());
    return out;
}

void rolling_cov_into(const double* x, const double* y, std::size_t n,
                      std::size_t window, std::size_t ddof, double* out) {
    rolling_pair_stat_into(x, y, n, window, ddof,
                           PairStatistic::covariance, out);
}

std::vector<double> rolling_cov(const double* x, const double* y, std::size_t n,
                                std::size_t window, std::size_t ddof) {
    std::vector<double> out(n, kNaN);
    rolling_cov_into(x, y, n, window, ddof, out.data());
    return out;
}

void rolling_beta_into(const double* y, const double* x, std::size_t n,
                       std::size_t window, double* out) {
    rolling_pair_stat_into(x, y, n, window, 0, PairStatistic::beta, out);
}

std::vector<double> rolling_beta(const double* y, const double* x, std::size_t n,
                                 std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_beta_into(y, x, n, window, out.data());
    return out;
}

namespace {

void rolling_slope_r2_into(const double* values, std::size_t n,
                           std::size_t window, bool slope_output, double* out) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");
    require_output(out);
    std::fill(out, out + n, kNaN);

    SlidingMoments2 moments;
    for_each_complete_window(
        n, window, [&](std::size_t i) { return std::isfinite(values[i]); },
        [&](std::size_t i) { moments.push(static_cast<double>(i), values[i]); },
        [&](std::size_t) { moments.pop(); },
        [&](std::size_t i) {
            const Moments2 block = moments.moments();
            if (block.m2_x <= 0.0L) return;
            if (slope_output) {
                out[i] = static_cast<double>(block.co_moment / block.m2_x);
            } else if (block.m2_y > 0.0L) {
                const long double r2 =
                    block.co_moment * block.co_moment /
                    (block.m2_x * block.m2_y);
                out[i] =
                    std::max(0.0, std::min(1.0, static_cast<double>(r2)));
            }
        });
}

}  // namespace

void rolling_slope_into(const double* values, std::size_t n, std::size_t window,
                        double* out) {
    rolling_slope_r2_into(values, n, window, true, out);
}

std::vector<double> rolling_slope(const double* values, std::size_t n,
                                  std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_slope_into(values, n, window, out.data());
    return out;
}

void rolling_r2_into(const double* values, std::size_t n, std::size_t window,
                     double* out) {
    rolling_slope_r2_into(values, n, window, false, out);
}

std::vector<double> rolling_r2(const double* values, std::size_t n,
                               std::size_t window) {
    std::vector<double> out(n, kNaN);
    rolling_r2_into(values, n, window, out.data());
    return out;
}

}  // namespace rflux::qflux
