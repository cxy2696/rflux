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

}  // namespace

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
        if (i + 1 < window || missing != 0) continue;

        const rflux::detail::Moments2 block = moments.moments();
        if (block.m2_x <= 0.0L || block.m2_y <= 0.0L) continue;
        double corr = static_cast<double>(
            block.co_moment / std::sqrt(block.m2_x * block.m2_y));
        corr = std::max(-1.0, std::min(1.0, corr));
        out[i] = corr;
    }
    return out;
}

std::vector<double> rolling_beta(const double* y, const double* x, std::size_t n,
                                std::size_t window) {
    require_pointer(y, "y must not be NULL");
    require_pointer(x, "x must not be NULL");
    require(n > 0, "n must be positive");
    require(window >= 2 && window <= n, "window must be in [2, n]");

    std::vector<double> out(n, kNaN);
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
        if (i + 1 < window || missing != 0) continue;

        const rflux::detail::Moments2 block = moments.moments();
        if (block.m2_x <= 0.0L) continue;
        out[i] = static_cast<double>(block.co_moment / block.m2_x);
    }
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
