#include "rflux/qflux/ranking.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "internal/common.hpp"

namespace rflux::qflux {
namespace {

using rflux::detail::kNaN;
using rflux::detail::quantile_sorted;
using rflux::detail::require;
using rflux::detail::require_pointer;

struct RankedValue {
    double value;
    std::size_t index;
};

std::vector<RankedValue> sorted_finite(const double* values, std::size_t n) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    std::vector<RankedValue> items;
    items.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) items.push_back({values[i], i});
    }
    std::sort(items.begin(), items.end(),
              [](const RankedValue& a, const RankedValue& b) {
                  if (a.value != b.value) return a.value < b.value;
                  return a.index < b.index;
              });
    return items;
}

}  // namespace

std::vector<double> rank(const double* values, std::size_t n) {
    const std::vector<RankedValue> items = sorted_finite(values, n);
    std::vector<double> out(n, kNaN);
    std::size_t begin = 0;
    while (begin < items.size()) {
        std::size_t end = begin + 1;
        while (end < items.size() && items[end].value == items[begin].value) {
            ++end;
        }
        const double average_rank =
            (static_cast<double>(begin + 1) + static_cast<double>(end)) / 2.0;
        for (std::size_t k = begin; k < end; ++k) {
            out[items[k].index] = average_rank;
        }
        begin = end;
    }
    return out;
}

std::vector<double> percentile_rank(const double* values, std::size_t n) {
    std::vector<double> out = rank(values, n);
    const std::size_t valid =
        static_cast<std::size_t>(std::count_if(out.begin(), out.end(),
                                               [](double value) {
                                                   return std::isfinite(value);
                                               }));
    if (valid == 0) return out;
    for (double& value : out) {
        if (std::isfinite(value)) {
            value = (value - 0.5) / static_cast<double>(valid);
        }
    }
    return out;
}

std::vector<double> zscore(const double* values, std::size_t n,
                           std::size_t ddof) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(ddof <= 1, "ddof must be 0 or 1");
    std::vector<double> out(n, kNaN);

    std::size_t valid = 0;
    long double sum = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            sum += static_cast<long double>(values[i]);
            ++valid;
        }
    }
    if (valid == 0 || valid <= ddof) return out;
    const long double mean = sum / static_cast<long double>(valid);
    long double squared = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) continue;
        const long double diff =
            static_cast<long double>(values[i]) - mean;
        squared += diff * diff;
    }
    const long double denominator =
        static_cast<long double>(valid - ddof);
    const double standard_deviation =
        std::sqrt(static_cast<double>(squared / denominator));
    if (!(standard_deviation > 0.0)) return out;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            out[i] = (values[i] - static_cast<double>(mean)) /
                     standard_deviation;
        }
    }
    return out;
}

std::vector<double> winsorize(const double* values, std::size_t n,
                              double lower_q, double upper_q) {
    require_pointer(values, "values must not be NULL");
    require(n > 0, "n must be positive");
    require(std::isfinite(lower_q) && std::isfinite(upper_q) &&
                lower_q >= 0.0 && lower_q <= upper_q && upper_q <= 1.0,
            "quantiles must satisfy 0 <= lower_q <= upper_q <= 1");

    std::vector<double> finite;
    finite.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) finite.push_back(values[i]);
    }
    std::vector<double> out(n, kNaN);
    if (finite.empty()) return out;
    std::sort(finite.begin(), finite.end());
    const double lower = quantile_sorted(finite, lower_q);
    const double upper = quantile_sorted(finite, upper_q);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(values[i])) {
            out[i] = std::max(lower, std::min(upper, values[i]));
        }
    }
    return out;
}

std::vector<double> neutralize(const double* values, std::size_t n,
                               const std::int32_t* groups,
                               std::size_t group_count) {
    require_pointer(values, "values must not be NULL");
    require_pointer(groups, "groups must not be NULL");
    require(n > 0, "n must be positive");
    require(group_count > 0, "group_count must be positive");

    std::vector<long double> sums(group_count, 0.0L);
    std::vector<std::size_t> counts(group_count, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t group = groups[i];
        if (group == -1) continue;
        require(group >= 0 &&
                    static_cast<std::size_t>(group) < group_count,
                "groups must be -1 or in [0, group_count)");
        if (std::isfinite(values[i])) {
            sums[static_cast<std::size_t>(group)] +=
                static_cast<long double>(values[i]);
            ++counts[static_cast<std::size_t>(group)];
        }
    }

    std::vector<double> out(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t group = groups[i];
        if (group == -1 || !std::isfinite(values[i])) continue;
        const std::size_t index = static_cast<std::size_t>(group);
        if (counts[index] == 0) continue;
        const double mean =
            static_cast<double>(sums[index] /
                                static_cast<long double>(counts[index]));
        out[i] = values[i] - mean;
    }
    return out;
}

}  // namespace rflux::qflux
