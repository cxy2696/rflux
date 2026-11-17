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
using rflux::detail::require;
using rflux::detail::require_pointer;

void validate_rolling(const double* values, std::size_t n, std::size_t window,
                      std::size_t ddof = 0) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(window > 0 && window <= n, "window must be in [1, n]");
    require(ddof < window, "ddof must be smaller than window");
}

struct CenterScale {
    std::vector<double> mean;
    std::vector<double> scale;
};

/* One pass over the window boundaries for std and z-score. */
CenterScale rolling_center_scale(const double* values, std::size_t n,
                                 std::size_t window, std::size_t ddof) {
    validate_rolling(values, n, window, ddof);
    CenterScale result{std::vector<double>(n, kNaN),
                       std::vector<double>(n, kNaN)};

    rflux::detail::SlidingMoments1 moments;
    std::size_t missing = 0;
    const long double denominator =
        static_cast<long double>(window - ddof);

    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            moments.push(values[i]);
        } else {
            ++missing;
        }
        if (i >= window) {
            const double old = values[i - window];
            if (std::isfinite(old)) {
                moments.pop();
            } else {
                --missing;
            }
        }
        if (i + 1 < window || missing != 0) continue;

        const rflux::detail::Moments1 block = moments.moments();
        long double variance = block.m2 / denominator;
        if (variance < 0.0L) variance = 0.0L;
        result.mean[i] = static_cast<double>(block.mean);
        result.scale[i] = std::sqrt(static_cast<double>(variance));
    }
    return result;
}

class Fenwick {
public:
    explicit Fenwick(std::size_t n) : tree_(n + 1, 0) {}

    void add(std::size_t index, std::ptrdiff_t delta) {
        for (std::size_t i = index + 1; i < tree_.size(); i += i & (~i + 1)) {
            tree_[i] += delta;
        }
    }

    /* 1-based rank -> zero-based compressed index. */
    std::size_t kth(std::size_t rank) const {
        std::size_t index = 0;
        std::size_t bit = 1;
        while ((bit << 1) < tree_.size()) bit <<= 1;
        for (; bit != 0; bit >>= 1) {
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
};

template <typename Emit>
void for_each_pair_window(const double* x, const double* y, std::size_t n,
                          std::size_t window, Emit emit) {
    rflux::detail::SlidingMoments2 moments;
    std::size_t missing = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool valid = std::isfinite(x[i]) && std::isfinite(y[i]);
        if (valid) {
            moments.push(x[i], y[i]);
        } else {
            ++missing;
        }
        if (i >= window) {
            const std::size_t old = i - window;
            if (std::isfinite(x[old]) && std::isfinite(y[old])) {
                moments.pop();
            } else {
                --missing;
            }
        }
        if (i + 1 >= window && missing == 0) {
            emit(i, moments.moments());
        }
    }
}

}  // namespace

std::vector<double> rolling_quantile(const double* values, std::size_t n,
                                     std::size_t window, double q) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(window > 0 && window <= n, "window must be in [1, n]");
    require(std::isfinite(q) && q >= 0.0 && q <= 1.0,
            "q must be in [0, 1]");

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

    std::vector<double> out(n, kNaN);
    std::size_t missing = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            fenwick.add(compressed[i], 1);
        } else {
            ++missing;
        }
        if (i >= window) {
            const std::size_t old = i - window;
            if (std::isfinite(values[old])) {
                fenwick.add(compressed[old], -1);
            } else {
                --missing;
            }
        }
        if (i + 1 < window || missing != 0) continue;

        const double position =
            q * static_cast<double>(window - 1);
        const std::size_t lower =
            static_cast<std::size_t>(std::floor(position));
        const std::size_t upper = std::min(lower + 1, window - 1);
        const double low_value =
            coordinates[fenwick.kth(lower + 1)];
        const double high_value =
            coordinates[fenwick.kth(upper + 1)];
        const double fraction =
            position - static_cast<double>(lower);
        out[i] = low_value + fraction * (high_value - low_value);
    }
    return out;
}

std::vector<double> rolling_sum(const double* values, std::size_t n,
                                std::size_t window) {
    validate_rolling(values, n, window);
    std::vector<double> out(n, kNaN);
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
        if (i + 1 >= window && missing == 0) {
            out[i] = static_cast<double>(sum);
        }
    }
    return out;
}

std::vector<double> rolling_mean(const double* values, std::size_t n,
                                 std::size_t window) {
    std::vector<double> out = rolling_sum(values, n, window);
    for (double& value : out) {
        if (std::isfinite(value)) value /= static_cast<double>(window);
    }
    return out;
}

std::vector<double> rolling_std(const double* values, std::size_t n,
                                std::size_t window, std::size_t ddof) {
    return rolling_center_scale(values, n, window, ddof).scale;
}

std::vector<double> rolling_extreme(const double* values, std::size_t n,
                                   std::size_t window, bool minimum) {
    validate_rolling(values, n, window);
    std::vector<double> out(n, kNaN);
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
        if (i + 1 >= window && missing == 0 && !candidates.empty()) {
            out[i] = values[candidates.front()];
        }
    }
    return out;
}

std::vector<double> rolling_min(const double* values, std::size_t n,
                                std::size_t window) {
    return rolling_extreme(values, n, window, true);
}

std::vector<double> rolling_max(const double* values, std::size_t n,
                                std::size_t window) {
    return rolling_extreme(values, n, window, false);
}

std::vector<double> rolling_zscore(const double* values, std::size_t n,
                                   std::size_t window, std::size_t ddof) {
    const CenterScale stats = rolling_center_scale(values, n, window, ddof);
    std::vector<double> out(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(stats.scale[i]) && stats.scale[i] > 0.0 &&
            std::isfinite(values[i])) {
            out[i] = (values[i] - stats.mean[i]) / stats.scale[i];
        }
    }
    return out;
}

std::vector<double> rolling_corr(const double* x, const double* y, std::size_t n,
                                 std::size_t window) {
    require_pointer(x, "x must not be NULL");
    require_pointer(y, "y must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");

    std::vector<double> out(n, kNaN);
    for_each_pair_window(x, y, n, window,
                         [&](std::size_t i, const rflux::detail::Moments2& block) {
                             if (block.m2_x <= 0.0L || block.m2_y <= 0.0L) {
                                 return;
                             }
                             double corr = static_cast<double>(
                                 block.co_moment /
                                 std::sqrt(block.m2_x * block.m2_y));
                             out[i] = std::max(-1.0, std::min(1.0, corr));
                         });
    return out;
}

std::vector<double> rolling_cov(const double* x, const double* y, std::size_t n,
                                std::size_t window, std::size_t ddof) {
    require_pointer(x, "x must not be NULL");
    require_pointer(y, "y must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");
    require(ddof < window, "ddof must be smaller than window");

    std::vector<double> out(n, kNaN);
    for_each_pair_window(x, y, n, window,
                         [&](std::size_t i, const rflux::detail::Moments2& block) {
                             out[i] = static_cast<double>(
                                 block.co_moment /
                                 static_cast<long double>(window - ddof));
                         });
    return out;
}

std::vector<double> rolling_beta(const double* y, const double* x, std::size_t n,
                                std::size_t window) {
    require_pointer(y, "y must not be NULL");
    require_pointer(x, "x must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");

    std::vector<double> out(n, kNaN);
    for_each_pair_window(x, y, n, window,
                         [&](std::size_t i, const rflux::detail::Moments2& block) {
                             if (block.m2_x > 0.0L) {
                                 out[i] = static_cast<double>(
                                     block.co_moment / block.m2_x);
                             }
                         });
    return out;
}

namespace {

void rolling_slope_r2_impl(const double* values, std::size_t n,
                           std::size_t window, bool slope_output,
                           std::vector<double>& result) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");
    result.assign(n, kNaN);

    rflux::detail::SlidingMoments2 moments;
    std::size_t missing = 0;

    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            moments.push(static_cast<double>(i), values[i]);
        } else {
            ++missing;
        }
        if (i >= window) {
            const std::size_t old = i - window;
            if (std::isfinite(values[old])) {
                moments.pop();
            } else {
                --missing;
            }
        }
        if (i + 1 < window || missing != 0) continue;

        const rflux::detail::Moments2 block = moments.moments();
        if (block.m2_x <= 0.0L) continue;
        const double slope = static_cast<double>(block.co_moment / block.m2_x);
        if (slope_output) {
            result[i] = slope;
        } else if (block.m2_y > 0.0L) {
            const long double r2 =
                block.co_moment * block.co_moment /
                (block.m2_x * block.m2_y);
            result[i] = std::max(0.0, std::min(1.0, static_cast<double>(r2)));
        }
    }
}

}  // namespace

std::vector<double> rolling_slope(const double* values, std::size_t n,
                                  std::size_t window) {
    std::vector<double> out;
    rolling_slope_r2_impl(values, n, window, true, out);
    return out;
}

std::vector<double> rolling_r2(const double* values, std::size_t n,
                               std::size_t window) {
    std::vector<double> out;
    rolling_slope_r2_impl(values, n, window, false, out);
    return out;
}

}  // namespace rflux::qflux
